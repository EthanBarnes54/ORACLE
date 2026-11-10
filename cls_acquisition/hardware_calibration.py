"""The hardware calibrations: the amplifier's settling, the converters' linearity, the cooler.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The bench tool (cls-cal, A. Raggio) in ORACLE. The settle test steps the
platform while the line meter samples on its own timer: the settle model,
and a sweep's wait (:func:`Recommended_Wait`). The linearity sweep reads
each converter on a count-based ladder (up, down, both or random), finer
across 0 V, with 0 V checks, and fits the line, each branch, their seam
(Mode C), curvature, hysteresis, 0 V drift and the INL on 25 V knots; a
proposed correction expires. The cooler monitor gives level, drift, noise
by band, mains harmonics and spectrum. Every move is ramped and confirmed
through the acquisition process, and each record written as it goes
(:class:`Record`).

Depends on: numpy; cls_acquisition.acquisition, .calibration, .scan_plan,
.hardware.mcc_dac.
"""

import os
import time
from datetime import datetime

import numpy as np

from cls_acquisition.acquisition import REPLY_TIMEOUT_S, Ramp
from cls_acquisition.calibration import (PASS, THRESHOLDS, WARN, Calibrating, Calibration_Error,
                                         Line_Fit, Step_At_Zero, _Quietly)
from cls_acquisition.hardware.mcc_dac import AMPLIFIER_GAIN, CH_NEG, CH_POS, LSB_V
from cls_acquisition.scan_plan import Scan_Plan

DIRECTIONS = ('Up', 'Down', 'Up then down (hysteresis)', 'Random (separates drift)')
PRE_STEP_FRACTION = 0.1             # of a settle capture, sampled before the step
SETTLE_BAND_SIGMAS = 5.0            # a narrower band is left by noise alone, long after settling
MIN_STEP_SIGMAS = 10.0              # a smaller step cannot be told from the noise
MAX_GAIN_ERROR = 0.02               # a branch further from 1 is a divider or wiring fault
BRANCHES = ((CH_POS, 'positive'), (CH_NEG, 'negative'))
POWER_LINE_HZ = 50.0                # an NPLC's length at IGISOL
STOP_POLL_S = 0.05                  # how often a long wait checks for a stop
FETCH_MARGIN_S = 10.0               # beyond a capture's own length before it is given up
CODE_V = LSB_V * AMPLIFIER_GAIN     # one converter code at the platform
INL_KNOT_V = 25.0                   # the non-linearity repeats every 1024 codes (312.5 V)
INL_MIN_POINTS = 5                  # readings a knot averages, so code rounding averages out
STREAM_CHUNK = 20000                # readings per R?
STALL_S = 5.0                       # no new readings for this long and a stream has stopped
MIN_SPECTRUM_READINGS = 64
PEAK_SNR = 6.0                      # above the spectrum's local median, for an oscillation
PEAKS = 10
NOISE_BANDS_HZ = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
WAIT_FACTOR, MIN_WAIT_S = 3.0, 0.2  # the bench tool's linearity wait: 3x the slowest step
PRECISE_READINGS, PRECISE_NPLC = 5, 10.0    # the bench tool's 'Read DMM'


def _Today(today=None):
    return today or f"{datetime.now():%Y-%m-%d}"


class Bench_Plan():
    """The setpoints a hardware calibration writes, from 0 V, in order, and
    the label of each it measures at (``Marks``: index to label)."""

    def __init__(self, ramp_step_v):
        self.Setpoints, self.Marks = [], {}
        self._Ramp_Step_V = ramp_step_v

    def Ramp_To(self, volts, label=None):
        """Ramped moves to ``volts``, measured there if it has a ``label``."""
        start = self.Setpoints[-1] if self.Setpoints else 0.0
        self.Setpoints += Ramp(start, volts, self._Ramp_Step_V)
        if label is not None:
            self.Marks[len(self.Setpoints) - 1] = label

    def Step_To(self, volts, label):
        """A measured step, written in one move however large."""
        self.Setpoints.append(float(volts))
        self.Marks[len(self.Setpoints) - 1] = label


def Settle_Plan(steps, reps, ramp_step_v):
    """Each ``(from, to)`` of ``steps`` stepped out and back ``reps`` times,
    each step labelled ``(step, mode, rep, from, to)``."""
    plan = Bench_Plan(ramp_step_v)
    for number, (start, stop) in enumerate(steps, 1):
        plan.Ramp_To(start)
        for rep in range(1, reps + 1):
            plan.Step_To(stop, (number, 'out', rep, start, stop))
            plan.Step_To(start, (number, 'back', rep, stop, start))
    return plan


def Linearity_Rungs(start, stop, step_v, direction, fine_span_v=0.0, fine_step_v=0.0, seed=0):
    """The rungs of :func:`Linearity_Plan` as ``(volts, way)``, in visiting
    order, without the ramps: cheap to redo as parameters are typed. Raises
    Scan_Plan_Error for a ladder that cannot be laid out."""
    rungs = Scan_Plan([start], [stop], step_v).Setpoints
    if fine_span_v > 0 and fine_step_v > 0:
        fine = Scan_Plan([-fine_span_v], [fine_span_v], fine_step_v).Setpoints
        fine = fine[(fine >= rungs[0]) & (fine <= rungs[-1])]
        rungs = np.unique(np.round(np.concatenate([rungs, fine]), 9))
    rungs = rungs.tolist()
    return {
        DIRECTIONS[0]: lambda: [(volts, 'up') for volts in rungs],
        DIRECTIONS[1]: lambda: [(volts, 'down') for volts in rungs[::-1]],
        DIRECTIONS[2]: lambda: ([(volts, 'up') for volts in rungs]
                                + [(volts, 'down') for volts in rungs[-2::-1]]),
        DIRECTIONS[3]: lambda: [(float(volts), 'random')
                                for volts in np.random.default_rng(seed).permutation(rungs)],
    }[direction]()


def Linearity_Marks(rungs, zero_every):
    """How many points a sweep of ``rungs`` measures, its 0 V checks included:
    every ramp ends on a new setpoint, so no two marks share one."""
    return len(rungs) + (-(-len(rungs) // zero_every) + 1 if zero_every else 0)


def Linearity_Plan(start, stop, step_v, direction, ramp_step_v, fine_span_v=0.0,
                   fine_step_v=0.0, zero_every=0, seed=0, skip=0):
    """A ladder laid out by count, finer across ``+-fine_span_v``, visited in
    ``direction`` and measured at each rung after the first ``skip`` (taken by
    the sweep this one resumes), with 0 V every ``zero_every`` rungs and at
    the end: ``('sweep', way, volts)`` and ``('zero', '', 0.0)`` marks."""
    sequence = Linearity_Rungs(start, stop, step_v, direction, fine_span_v, fine_step_v,
                               seed)[skip:]
    plan = Bench_Plan(ramp_step_v)
    for i, (volts, way) in enumerate(sequence):
        if zero_every and i % zero_every == 0:
            plan.Ramp_To(0.0, label=('zero', '', 0.0))
        plan.Ramp_To(volts, label=('sweep', way, volts))
    if zero_every:
        plan.Ramp_To(0.0, label=('zero', '', 0.0))
    return plan


def Run_Bench(link, meter, setup, plan, ramp, measure, stopped):
    """Write ``plan`` by step and confirm, ``setup()`` readying the meter and
    every unmeasured move at ``ramp`` ``(step V, rate V/s)``. At each mark
    ``measure(write, label)`` calls ``write()``, which returns its
    confirmation; ``stopped()`` ends the plan early. Every exit ramps back to
    0 V. Returns the 'Calibrate' reply."""
    step_v, rate = ramp
    last = 0.0

    def write(index, volts):
        nonlocal last
        step = link.Send('Calibration_Step', index=index)
        if step['Index'] != index:
            raise Calibration_Error(f"The acquisition process confirmed setpoint "
                                    f"{step['Index']} when {index} was sent, so the "
                                    f"measurement stopped.")
        last = volts
        return step

    def down():
        link.Send('Stop_Calibration', timeout=abs(last) / rate + REPLY_TIMEOUT_S,
                  ramp_step_v=step_v, ramp_rate_v_s=rate)

    with Calibrating(link, meter, setup, plan.Setpoints, down) as dac:
        for index, volts in enumerate(plan.Setpoints):
            if stopped():
                break
            if index in plan.Marks:
                measure(lambda: write(index, volts), plan.Marks[index])
            else:
                time.sleep(abs(volts - last) / rate)
                write(index, volts)
    return dac


# ── Settle time ─────────────────────────────────────────────────

def _Crossing(times, progress, level):
    above = np.nonzero(progress >= level)[0]
    if not len(above) or above[0] == 0:
        raise Calibration_Error(f"The step never crossed {level:.0%} of its height inside "
                                f"the capture; lengthen it or check the meter sees the "
                                f"platform.")
    i = above[0]
    fraction = (level - progress[i - 1]) / (progress[i] - progress[i - 1])
    return times[i - 1] + fraction * (times[i] - times[i - 1])


def Step_Response(times_s, volts, tolerance_v):
    """One step's response captured from before it: times from its 10 %
    crossing, deviation from the final value, and the figures. Settled: within
    ``tolerance_v`` (or SETTLE_BAND_SIGMAS of noise, if wider) for good."""
    times_s, volts = np.asarray(times_s, dtype=float), np.asarray(volts, dtype=float)
    count = len(volts)
    initial = float(np.median(volts[:max(int(count * PRE_STEP_FRACTION / 2), 2)]))
    tail = volts[-max(count // 5, 3):]
    # from sample to sample, so a tail still drifting cannot pass for noise
    # and widen the band until a slow amplifier counts as settled
    final, noise = float(np.median(tail)), float(np.std(np.diff(tail)) / np.sqrt(2))
    height = final - initial
    if abs(height) < MIN_STEP_SIGMAS * noise or height == 0:
        raise Calibration_Error(f"The capture shows a step of {height:.3f} V against "
                                f"{noise:.3f} V of noise, so the platform did not follow "
                                f"the command.")
    progress = (volts - initial) / height
    t10, t90 = _Crossing(times_s, progress, 0.1), _Crossing(times_s, progress, 0.9)
    band = max(tolerance_v, SETTLE_BAND_SIGMAS * noise)
    outside = np.nonzero(np.abs(volts - final) > band)[0]
    if len(outside) and outside[-1] == count - 1:
        raise Calibration_Error(f"The platform had not settled within {band:.3f} V by the "
                                f"end of the capture; lengthen it.")
    settled_at = times_s[outside[-1] + 1] if len(outside) else times_s[0]
    return {'Times_S': times_s - t10, 'Deviation_V': volts - final, 'Final_V': final,
            'T10_S': t10,
            'Settle_S': max(settled_at - t10, 0.0), 'Rise_S': t90 - t10,
            'Overshoot_Pct': max(float(np.max(progress[times_s >= t10])) - 1.0, 0.0) * 100,
            'Noise_V': noise, 'Band_V': band}


def Settle_Test(link, meter, divider, steps, reps, window_s, interval_s, aperture_s,
                hold_s, tolerance_v, ramp, progress, stopped, tail_s=0.0, tail_nplc=1.0,
                tail_interval_s=0.1, records=None):
    """Each step of :func:`Settle_Plan`, captured from PRE_STEP_FRACTION of
    ``window_s`` before it, then read ``tail_s`` more at ``tail_nplc``;
    ``progress(result)`` after each, added to ``records['settle']`` and
    ``['settle_traces']``. Returns the results in order."""
    results = []
    fast = {'count': int(round(window_s / interval_s)), 'interval_s': interval_s,
            'aperture_s': aperture_s}
    tail = {'count': max(int(round(tail_s / tail_interval_s)), 3),
            'interval_s': tail_interval_s, 'nplc': tail_nplc, 'autozero': True}

    def setup():
        # a meter that cannot sample so often for this aperture samples as often
        # as it can: fewer readings then fill the same window
        spacing = meter.Capture(**fast)
        if spacing > interval_s * 1.001:
            fast['count'] = max(int(round(window_s / spacing)), 3)
            meter.Capture(**fast)

    def measure(write, label):
        number, mode, rep, start, stop = label
        time.sleep(hold_s)
        armed = meter.Arm()
        time.sleep(PRE_STEP_FRACTION * window_s)
        step = write()
        interval = meter.Capture_Interval_S
        volts = meter.Fetch_When_Done(fast['count'], fast['count'] * interval
                                      + FETCH_MARGIN_S) * divider
        response = Step_Response(interval * np.arange(len(volts)), volts, tolerance_v)
        lag = response['T10_S'] - (step['T_Command_Ns'] - armed) / 1e9
        result = dict(response, Step=number, Mode=mode, Rep=rep, From_V=start, To_V=stop,
                      Lag_S=lag, Tail_Shift_V=np.nan, Tail_Drift_V=np.nan)
        if tail_s:
            spacing = meter.Capture(**tail)
            meter.Arm()
            # autozero doubles every reading's time
            creep = meter.Fetch_When_Done(tail['count'], tail['count'] * max(
                spacing, 2 * tail_nplc / POWER_LINE_HZ) + FETCH_MARGIN_S) * divider
            meter.Capture(**fast)
            result['Tail_Shift_V'] = float(np.median(creep[-max(len(creep) // 5, 1):])
                                           - response['Final_V'])
            result['Tail_Drift_V'] = Line_Fit(spacing * np.arange(len(creep)), creep)[0] \
                * spacing * (len(creep) - 1)
        results.append(result)
        if records:
            records['settle'].Add(*([result[key]] for key, _ in SETTLE_COLUMNS))
            records['settle_traces'].Add([number] * len(volts), [mode] * len(volts),
                                         [rep] * len(volts), result['Times_S'],
                                         result['Deviation_V'])
        progress(result)

    Run_Bench(link, meter, setup, Settle_Plan(steps, reps, ramp[0]), ramp, measure, stopped)
    return results


# a settle result's record: (result key, CSV column)
SETTLE_COLUMNS = (('Step', 'step'), ('Mode', 'mode'), ('Rep', 'rep'), ('From_V', 'from_v'),
                  ('To_V', 'to_v'), ('Lag_S', 'lag_s'), ('Settle_S', 'settle_s'),
                  ('Rise_S', 'rise_s'), ('Overshoot_Pct', 'overshoot_pct'),
                  ('Noise_V', 'noise_v'), ('Band_V', 'band_v'),
                  ('Tail_Shift_V', 'tail_shift_v'), ('Tail_Drift_V', 'tail_drift_v'),
                  ('Final_V', 'final_v'))
SETTLE_TRACE_COLUMNS = ('step', 'mode', 'rep', 'time_from_10pct_s', 'deviation_v')


def _From_Command(result):
    """How long after its command a step had settled; a lag below zero (the
    meter's first sample after INIT) counts as none."""
    return max(result['Lag_S'], 0.0) + result['Settle_S']


def Settle_Steps(results):
    """Per step and direction: the slowest and mean lag plus settling, the
    largest overshoot and tail shift, over its repetitions."""
    steps = []
    for key in sorted({(r['Step'], r['Mode']) for r in results}, key=lambda k: (k[0], k[1])):
        rows = [r for r in results if (r['Step'], r['Mode']) == key]
        needed = [_From_Command(r) for r in rows]
        shifts = [abs(r['Tail_Shift_V']) for r in rows if np.isfinite(r['Tail_Shift_V'])]
        steps.append({'Step': key[0], 'Mode': key[1], 'From_V': rows[0]['From_V'],
                      'To_V': rows[0]['To_V'], 'Reps': len(rows), 'Worst_S': max(needed),
                      'Mean_S': float(np.mean(needed)),
                      'Overshoot_Pct': max(r['Overshoot_Pct'] for r in rows),
                      'Tail_Shift_V': max(shifts) if shifts else None})
    return steps


def Settle_Summary(results):
    """``(verdict, sentence)``: the slowest step against ORACLE's SETTLE."""
    if not results:
        return WARN, "No step was measured."
    worst = max(results, key=_From_Command)
    needed, settle = _From_Command(worst), THRESHOLDS['SettleS']
    sentence = (f"Slowest: step {worst['Step']} {worst['Mode']} ({worst['From_V']:g} to "
                f"{worst['To_V']:g} V), {worst['Settle_S'] * 1e3:.1f} ms to settle after "
                f"a {max(worst['Lag_S'], 0.0) * 1e3:.1f} ms lag, {needed * 1e3:.1f} ms from "
                f"the command against ORACLE's SETTLE of {settle * 1e3:.0f} ms.")
    return (PASS if needed <= settle else WARN), sentence


def Recommended_Wait(results):
    """A linearity sweep's wait per step, as the bench tool recommends:
    WAIT_FACTOR times the slowest step from its command, at least MIN_WAIT_S."""
    return max(MIN_WAIT_S, WAIT_FACTOR * max(_From_Command(r) for r in results))


def Settle_Model(results, today=None):
    """The scan's settle model from a settle test: per jump size, the slowest
    step's lag plus settling, non-decreasing in the jump, dated."""
    slowest = {}
    for r in results:
        jump = round(abs(r['To_V'] - r['From_V']), 6)
        slowest[jump] = max(slowest.get(jump, 0.0), _From_Command(r))
    jumps = sorted(slowest)
    settles = np.maximum.accumulate([slowest[jump] for jump in jumps])
    return {'measured': _Today(today),
            'points': [[float(jump), float(s)] for jump, s in zip(jumps, settles)]}


# ── Linearity ───────────────────────────────────────────────────

LINEARITY_COLUMNS = ('kind', 'direction', 'set_v', 'read_v', 'sigma_v', 'channel', 'dac_v',
                     'time_s')


def Linearity_Sweep(link, meter, divider, start, stop, step_v, direction, settle_s, readings,
                    nplc, ramp, progress, stopped, fine_span_v=0.0, fine_step_v=0.0,
                    zero_every=0, seed=0, autocal=False, records=None, skip=0):
    """Read every mark of :func:`Linearity_Plan` ``readings`` times at ``nplc``
    with autozero, ``settle_s`` after its write (autocalibrated first if
    ``autocal``); ``progress(point)`` after each, added to
    ``records['linearity']``. Returns :func:`Linearity_Analysis` of what was
    read, with the 'Calibrate' reply as 'Dac'."""
    points = []
    started = time.monotonic()

    def setup():
        meter.Capture(readings, nplc=nplc, autozero=True)
        if autocal:
            meter.Autocal()

    def measure(write, label):
        kind, way, _ = label
        step = write()
        time.sleep(settle_s)
        meter.Arm()
        # autozero doubles every reading's time
        volts = meter.Fetch_When_Done(readings, readings * 2 * nplc / POWER_LINE_HZ
                                      + FETCH_MARGIN_S) * divider
        point = {'Kind': kind, 'Direction': way, 'Set_V': step['Volts'],
                 'Read_V': float(np.mean(volts)),
                 'Sigma_V': float(np.std(volts, ddof=1)) if len(volts) > 1 else 0.0,
                 'Channel': step['Channel'], 'Dac_V': step['Dac_V'],
                 'Time_S': time.monotonic() - started}
        points.append(point)
        if records:
            records['linearity'].Add(*([point[key]] for key in POINT_KEYS))
        progress(point)

    plan = Linearity_Plan(start, stop, step_v, direction, ramp[0], fine_span_v, fine_step_v,
                          zero_every, seed, skip)
    dac = Run_Bench(link, meter, setup, plan, ramp, measure, stopped)
    return dict(Linearity_Analysis(points), Dac=dac)


POINT_KEYS = ('Kind', 'Direction', 'Set_V', 'Read_V', 'Sigma_V', 'Channel', 'Dac_V', 'Time_S')


def Linearity_Analysis(points):
    """The line through the rungs (uncertainties, residual, curvature, INL);
    each branch's line, their 0 V crossings and gain mismatch; hysteresis;
    the 0 V checks' drift; and the INL table where the sweep was dense enough."""
    sweep = [p for p in points if p['Kind'] == 'sweep']
    if len({p['Set_V'] for p in sweep}) < 3:
        raise Calibration_Error(f"The sweep read {len(sweep)} rungs, too few to fit a line "
                                f"through.")
    commanded = np.array([p['Set_V'] for p in sweep])
    readback = np.array([p['Read_V'] for p in sweep])
    (gain, offset), covariance = np.polyfit(commanded, readback, 1, cov='unscaled')
    residual = readback - (gain * commanded + offset)
    dof = max(len(commanded) - 2, 1)
    errors = np.sqrt(np.diag(covariance) * np.sum(residual ** 2) / dof)
    full_scale = float(np.max(np.abs(commanded)))
    fits = Branch_Fits(commanded, readback)
    result = {'Points': points, 'Set_V': commanded, 'Read_V': readback,
              'Sigma_V': np.array([p['Sigma_V'] for p in sweep]),
              'Channel': np.array([p['Channel'] for p in sweep]),
              'Gain': float(gain), 'Gain_Err': float(errors[0]), 'Offset_V': float(offset),
              'Offset_Err_V': float(errors[1]),
              'Residual_Rms_V': float(np.sqrt(np.mean(residual ** 2))),
              'Residual_Max_V': float(np.max(np.abs(residual))),
              'Inl_Ppm_Fs': float(np.max(np.abs(residual)) / full_scale * 1e6),
              'Quadratic_V_Per_Kv2': None, 'Fits': fits,
              'Seam_V': Step_At_Zero(commanded, readback), 'Gain_Mismatch_Ppm': None,
              'Hysteresis': Hysteresis(commanded, readback), 'Zero_Drift': None,
              'Inl': Inl_Table(commanded, readback, fits)}
    if len(np.unique(commanded)) > 4:
        result['Quadratic_V_Per_Kv2'] = float(np.polyfit(commanded, readback, 2)[0] * 1e6)
    if len(fits) == len(BRANCHES):
        result['Gain_Mismatch_Ppm'] = (fits[CH_POS]['Gain'] / fits[CH_NEG]['Gain'] - 1) * 1e6
    zeros = [p for p in points if p['Kind'] == 'zero']
    if len(zeros) > 1:
        t = np.array([p['Time_S'] for p in zeros])
        v = np.array([p['Read_V'] for p in zeros])
        result['Zero_Drift'] = {'Span_V': float(np.ptp(v)), 'Checks': len(zeros),
                                'Slope_V_Per_H': (float(np.polyfit(t, v, 1)[0] * 3600)
                                                  if np.ptp(t) > 0 else None)}
    return result


def Branch_Fits(commanded, readback):
    """Each branch's line, for every branch with three points or more. 0 V
    is in neither: both converters sit at mid-code there."""
    fits = {}
    for channel, side in ((CH_POS, commanded > 0), (CH_NEG, commanded < 0)):
        if len(np.unique(commanded[side])) >= 3:
            gain, offset, rms = Line_Fit(commanded[side], readback[side])
            fits[channel] = {'Gain': gain, 'Offset_V': offset, 'Rms_V': rms,
                             'Points': int(side.sum())}
    return fits


def Hysteresis(commanded, readback):
    """Mean and largest (second - first) over setpoints read up then down, or
    None: noise averages out of the mean, hysteresis does not. 0 V, on
    neither converter, is left out."""
    values, inverse, counts = np.unique(np.round(commanded, 6), return_inverse=True,
                                        return_counts=True)
    twice = [readback[inverse == i] for i in np.nonzero((counts == 2) & (values != 0))[0]]
    if not twice:
        return None
    shifts = np.array([b - a for a, b in twice])
    return {'Mean_V': float(np.mean(shifts)), 'Max_V': float(np.max(np.abs(shifts))),
            'Points': len(shifts)}


def Inl_Table(commanded, readback, fits):
    """Each branch's INL: its residual averaged over knots INL_KNOT_V apart
    (each with INL_MIN_POINTS readings), the shape's rms and what is left
    about it (code rounding and noise). None for a sweep too sparse."""
    tables = {}
    for channel, side in ((CH_POS, commanded > 0), (CH_NEG, commanded < 0)):
        if channel not in fits:
            continue
        volts = commanded[side]
        residual = readback[side] - (fits[channel]['Gain'] * volts + fits[channel]['Offset_V'])
        span = volts.max() - volts.min()
        count = int(np.floor(span / INL_KNOT_V)) + 1
        if count < 2:
            continue
        # centred on the branch's range, each averaging the readings within half a knot
        knots = volts.min() + (span - INL_KNOT_V * (count - 1)) / 2 + INL_KNOT_V * np.arange(count)
        near = np.abs(volts[None, :] - knots[:, None]) <= INL_KNOT_V / 2
        if np.any(near.sum(axis=1) < INL_MIN_POINTS):
            continue
        errors = (near * residual[None, :]).sum(axis=1) / near.sum(axis=1)
        smooth = np.interp(volts, knots, errors)
        tables[channel] = {'Knots_V': knots, 'Error_V': errors,
                           'Smooth_Rms_V': float(np.sqrt(np.mean(smooth ** 2))),
                           'Rough_Rms_V': float(np.std(residual - smooth))}
    return tables or None


def Corrected(correction, fits, valid_days, method, inl=None, today=None):
    """The DAC correction that cancels each branch's measured gain and offset,
    and its non-linearity if ``inl`` (:func:`Inl_Table`) covers both,
    composed with the ``correction`` the sweep was taken under."""
    for channel, name in BRANCHES:
        if channel not in fits:
            raise Calibration_Error(f"The sweep has fewer than three points on the {name} "
                                    f"branch, so it cannot correct both converters.")
        gain = fits[channel]['Gain']
        if abs(gain - 1.0) > MAX_GAIN_ERROR:
            raise Calibration_Error(f"The {name} branch's gain is {gain:.4f}, more than "
                                    f"{MAX_GAIN_ERROR:.0%} from 1: a divider or wiring fault, "
                                    f"which a correction would only hide.")
    poly = {}
    # readback = gain * set + offset on each branch, the converter driven by
    # |set| / AMPLIFIER_GAIN; the positive branch's offset adds to the
    # platform's magnitude and the negative's subtracts from it
    for channel, sign in ((CH_POS, -1.0), (CH_NEG, 1.0)):
        slope, intercept = correction['poly'][channel]
        gain, offset = fits[channel]['Gain'], fits[channel]['Offset_V']
        poly[channel] = [float(slope / gain),
                         float(intercept + sign * slope * offset / (AMPLIFIER_GAIN * gain))]
    tables = None
    if inl and all(channel in inl for channel, _ in BRANCHES):
        tables = {}
        for channel, _ in BRANCHES:
            knots, errors = inl[channel]['Knots_V'], inl[channel]['Error_V']
            # the sweep saw only what the table in force left behind
            previous = (correction.get('inl') or {}).get(channel)
            if previous is not None:
                errors = errors + np.interp(knots, *np.asarray(previous, dtype=float))
            tables[channel] = [[float(v) for v in knots], [float(v) for v in errors]]
    return {'instrument': correction['instrument'], 'measured': _Today(today),
            'valid_days': int(valid_days), 'method': method, 'poly': poly, 'inl': tables}


def Precise_Reading(meter, divider, readings=PRECISE_READINGS, nplc=PRECISE_NPLC):
    """``(mean, sigma)`` of ``readings`` readings at ``nplc`` with autozero,
    times ``divider``, the first SETTLE after the call (quick tune may just
    have moved the platform); the meter returns to its poll however it ends."""
    try:
        meter.Capture(readings, nplc=nplc, autozero=True)
        time.sleep(THRESHOLDS['SettleS'])
        meter.Arm()
        # autozero doubles every reading's time
        volts = meter.Fetch_When_Done(readings, readings * 2 * nplc / POWER_LINE_HZ
                                      + FETCH_MARGIN_S) * divider
    except BaseException:
        _Quietly(meter.Release)
        raise
    meter.Release()
    return float(np.mean(volts)), float(np.std(volts, ddof=1)) if len(volts) > 1 else 0.0


# ── Cooler ──────────────────────────────────────────────────────

COOLER_COLUMNS = ('time_s', 'cooler_v')
SPECTRUM_COLUMNS = ('frequency_hz', 'asd_v_rt_hz')


def Cooler_Monitor(meter, divider, duration_s, interval_s, aperture_s, nplc, progress, stopped,
                   line_hz=POWER_LINE_HZ, records=None):
    """The cooler streamed every ``interval_s`` for ``duration_s``, each reading
    over ``nplc`` cycles, else ``aperture_s``; ``progress((times, volts))``
    after each chunk, added to ``records['cooler']``. Returns
    :func:`Noise_Analysis` of what was read, with its times and volts."""
    count = max(int(np.ceil(duration_s / interval_s)), 1)
    integration = {'aperture_s': None if nplc else aperture_s, 'nplc': nplc}
    chunks = []
    try:
        interval = meter.Capture(count, interval_s, stream=True, **integration)
        if interval > interval_s * 1.001:
            # sampled as often as the meter can, for as long as was asked
            count = max(int(np.ceil(duration_s / interval)), 1)
            interval = meter.Capture(count, interval_s, stream=True, **integration)
        meter.Arm()
        taken, last_new = 0, time.monotonic()
        while taken < count and not stopped():
            volts = meter.Read_Available(min(STREAM_CHUNK, count - taken)) * divider
            if len(volts):
                times = interval * (taken + np.arange(len(volts)))
                chunks.append(volts)
                taken, last_new = taken + len(volts), time.monotonic()
                if records:
                    records['cooler'].Add(times, volts)
                progress((times, volts))
            elif time.monotonic() - last_new > max(STALL_S, 20 * interval):
                raise Calibration_Error(f"The cooler meter stopped after {taken} of {count} "
                                        f"readings: its memory overflowed, or it was "
                                        f"interrupted.")
            if len(volts) < STREAM_CHUNK:
                time.sleep(STOP_POLL_S)
    except BaseException:
        _Quietly(meter.Release)
        raise
    meter.Release()
    volts = np.concatenate(chunks) if chunks else np.zeros(0)
    if len(volts) < MIN_SPECTRUM_READINGS:
        raise Calibration_Error(f"The monitor took {len(volts)} readings before it stopped; "
                                f"its spectrum needs {MIN_SPECTRUM_READINGS}.")
    integration = nplc / line_hz if nplc else aperture_s
    return dict(Noise_Analysis(volts, interval, integration, line_hz),
                Times_S=interval * np.arange(len(volts)), Volts=volts)


def _Welch(x, interval_s, length):
    """One-sided power spectral density (V^2/Hz) of ``x``: Hann-windowed
    segments of ``length``, half overlapping, each less its mean, averaged."""
    window = np.hanning(length)
    power = np.zeros(length // 2 + 1)
    starts = range(0, len(x) - length + 1, length // 2)
    for start in starts:
        segment = x[start:start + length]
        power += np.abs(np.fft.rfft((segment - segment.mean()) * window)) ** 2
    psd = power / len(starts) * 2 * interval_s / np.sum(window ** 2)
    psd[0] /= 2
    if length % 2 == 0:
        psd[-1] /= 2
    return np.fft.rfftfreq(length, interval_s), psd


def _Spectral_Floor(amplitude):
    """The running median of an amplitude spectrum over +-30 % in frequency
    (at least +-10 bins), so a 1/f background is not taken for peaks."""
    count = len(amplitude)
    centres = np.unique(np.round(np.geomspace(2, count - 1, 200)).astype(int))
    medians = []
    for centre in centres:
        low = max(2, min(int(centre / 1.3), centre - 10))
        high = min(count, max(int(centre * 1.3), centre + 10) + 1)
        medians.append(np.median(amplitude[low:high]))
    return np.interp(np.log(np.maximum(np.arange(count), 1)), np.log(centres), medians)


def Noise_Analysis(volts, interval_s, integration_s=0.0, line_hz=POWER_LINE_HZ, peaks=PEAKS,
                   snr=PEAK_SNR):
    """A steady voltage's evenly spaced record: level, drift, noise before and
    after detrending by decade, oscillations and spectrum. An oscillation
    stands ``snr`` times above the Hann spectrum's local median; its
    amplitude is its main lobe's power, mains harmonics are named, and
    ``integration_s`` gives the meter's attenuation, sinc(f T). The Welch
    spectrum gives the white-noise floor."""
    volts = np.asarray(volts, dtype=float)
    count = len(volts)
    if count < MIN_SPECTRUM_READINGS:
        raise Calibration_Error(f"A record of {count} readings is too short for a spectrum.")
    times = interval_s * np.arange(count)
    slope, intercept = np.polyfit(times, volts, 1)
    x = volts - (intercept + slope * times)
    window = np.hanning(count)
    sum_w, sum_w2 = np.sum(window), np.sum(window ** 2)
    spectrum = np.abs(np.fft.rfft(x * window)) ** 2
    frequency = np.fft.rfftfreq(count, interval_s)
    resolution = frequency[1]
    psd = spectrum * 2 * interval_s / sum_w2
    psd[0] /= 2
    if count % 2 == 0:
        psd[-1] /= 2

    edges = [resolution] + [e for e in NOISE_BANDS_HZ if resolution < e < frequency[-1]]
    edges.append(frequency[-1] + resolution)
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (frequency >= low) & (frequency < high)
        if inside.any():
            bands.append({'Low_Hz': float(low), 'High_Hz': float(min(high, frequency[-1])),
                          'Rms_V': float(np.sqrt(np.sum(psd[inside]) * resolution))})

    amplitude = 2 * np.sqrt(spectrum) / sum_w     # a tone centred on its bin, peak
    floor = _Spectral_Floor(amplitude)
    k = np.arange(2, len(amplitude) - 2)
    maxima = k[(amplitude[k] >= amplitude[k - 1]) & (amplitude[k] > amplitude[k + 1])
               & (amplitude[k] > snr * floor[k])]
    picked = []
    for centre in maxima[np.argsort(-amplitude[maxima])]:
        if all(abs(centre - p) > 4 for p in picked):
            picked.append(int(centre))
        if len(picked) >= peaks:
            break
    found = []
    for centre in picked:
        # where the tone falls between bins, from the three about it
        a, b, g = np.log(np.sqrt(spectrum[centre - 1:centre + 2]) + 1e-300)
        bend = a - 2 * b + g
        hz = (centre + (0.5 * (a - g) / bend if bend else 0.0)) * resolution
        lobe = spectrum[max(centre - 2, 1):centre + 3].sum()
        peak_v = 2 * np.sqrt(lobe / (count * sum_w2))
        harmonic = int(round(hz / line_hz)) if line_hz > 0 else 0
        mains = harmonic >= 1 and abs(hz - harmonic * line_hz) <= max(
            3 * resolution, 0.005 * harmonic * line_hz)
        found.append({'Freq_Hz': float(hz), 'Amplitude_V': float(peak_v),
                      'Rms_V': float(peak_v / np.sqrt(2)),
                      'Snr': float(amplitude[centre] / floor[centre]),
                      'Mains_Harmonic': harmonic if mains else 0,
                      'Integration_Gain': float(abs(np.sinc(hz * integration_s)))})
    oscillation = float(np.sqrt(sum(p['Rms_V'] ** 2 for p in found)))
    detrended = float(np.std(x))

    length = count if count < 2048 else 2 ** int(np.log2(count / 8))
    welch_f, welch_psd = _Welch(x, interval_s, max(length, 256) if count >= 256 else count)
    asd = np.sqrt(welch_psd[1:])
    return {'Count': count, 'Interval_S': float(interval_s), 'Duration_S': count * interval_s,
            'Mean_V': float(np.mean(volts)), 'Sigma_V': float(np.std(volts)),
            'Min_V': float(np.min(volts)), 'Max_V': float(np.max(volts)),
            'Peak_To_Peak_V': float(np.ptp(volts)), 'Drift_V_Per_Min': float(slope * 60),
            'Rms_Detrended_V': detrended, 'Peak_To_Peak_Detrended_V': float(np.ptp(x)),
            'Bands': bands, 'Peaks': found, 'Oscillation_Rms_V': oscillation,
            'Random_Rms_V': float(np.sqrt(max(detrended ** 2 - oscillation ** 2, 0.0))),
            'Noise_Floor_V_Rt_Hz': float(np.median(asd)),
            'Frequency_Hz': welch_f[1:], 'Asd': asd}


# ── Records ─────────────────────────────────────────────────────

class Record():
    """One measurement's CSV under ``<directory>/hardware_calibration``:
    settings as ``#`` lines, rows flushed as taken, status and results as
    ``#`` lines once it ends. Never overwrites."""

    def __init__(self, directory, kind, header, columns):
        folder = os.path.join(directory, 'hardware_calibration')
        os.makedirs(folder, exist_ok=True)
        self.Path = os.path.join(folder, f"{kind}_{datetime.now():%Y-%m-%d_%H%M%S}.csv")
        self._File = open(self.Path, 'x', encoding='utf-8', newline='')
        for key, value in header.items():
            self._File.write(f"# {key}: {value}\n")
        self._File.write(','.join(columns) + '\n')
        self._File.flush()

    def Add(self, *columns):
        """Rows, given as their columns."""
        for row in zip(*columns):
            self._File.write(','.join(v if isinstance(v, str) else f"{v:.9g}" for v in row)
                             + '\n')
        self._File.flush()

    def Close(self, status, results=None):
        """The measurement's end; safe to call more than once."""
        if self._File.closed:
            return
        self._File.write(f"# status: {status}\n")
        for key, value in (results or {}).items():
            self._File.write(f"# {key}: {value}\n")
        self._File.close()


# ── Reading records back ────────────────────────────────────────

def Read_Record(path):
    """``(header, columns, footer)`` of a record: settings and results as
    {key: text}, rows as {column: [text, ...]}."""
    header, footer, names, rows = {}, {}, None, []
    with open(path, encoding='utf-8', newline='') as fh:
        for line in fh:
            line = line.rstrip('\r\n')
            if line.startswith('# '):
                key, _, value = line[2:].partition(': ')
                (header if names is None else footer)[key] = value
            elif names is None:
                names = line.split(',')
            elif line:
                rows.append(line.split(','))
    if names is None:
        raise Calibration_Error(f"{os.path.basename(path)} holds no columns, so it is not a "
                                f"hardware calibration record.")
    return header, {name: [row[i] for row in rows] for i, name in enumerate(names)}, footer


def _Numbers(values):
    return np.array([float(v) if v else np.nan for v in values])


def _Need(columns, wanted, path):
    missing = [name for name in wanted if name not in columns]
    if missing:
        raise Calibration_Error(f"{os.path.basename(path)} lacks the columns "
                                f"{', '.join(missing)}, so it is not that kind of record.")


def _Beside(path, kind):
    """The ``kind`` record started with the one at ``path``: the same time,
    or the nearest within two seconds; None if there is none."""
    folder, name = os.path.split(path)
    stamp = name[name.index('_2') + 1:-len('.csv')]
    when = datetime.strptime(stamp, '%Y-%m-%d_%H%M%S')
    nearest = None
    for other in os.listdir(folder):
        if not (other.startswith(f"{kind}_2") and other.endswith('.csv')):
            continue
        try:
            apart = abs((datetime.strptime(other[len(kind) + 1:-len('.csv')],
                                           '%Y-%m-%d_%H%M%S') - when).total_seconds())
        except ValueError:
            continue
        if apart <= 2 and (nearest is None or apart < nearest[0]):
            nearest = (apart, os.path.join(folder, other))
    return nearest[1] if nearest else None


def Read_Settle(path):
    """A settle record's results as Settle_Test returned them, each step's
    trace from the settle_traces record beside it (empty where none)."""
    _, columns, _ = Read_Record(path)
    _Need(columns, [column for _, column in SETTLE_COLUMNS[:-1]], path)
    results = []
    for i in range(len(columns['step'])):
        result = {key: (columns[column][i] if key == 'Mode' else
                        int(columns[column][i]) if key in ('Step', 'Rep') else
                        float(columns[column][i] or 'nan'))
                  for key, column in SETTLE_COLUMNS if column in columns}
        result.setdefault('Final_V', np.nan)
        result['Times_S'] = result['Deviation_V'] = np.zeros(0)
        results.append(result)
    traces = _Beside(path, 'settle_traces')
    if traces is not None:
        _, rows, _ = Read_Record(traces)
        keys = list(zip(rows['step'], rows['mode'], rows['rep']))
        times, deviations = _Numbers(rows['time_from_10pct_s']), _Numbers(rows['deviation_v'])
        for result in results:
            mine = np.array([key == (str(result['Step']), result['Mode'], str(result['Rep']))
                             for key in keys], dtype=bool)
            result['Times_S'], result['Deviation_V'] = times[mine], deviations[mine]
    return results


def Read_Linearity(path):
    """The points a linearity record holds, as Linearity_Sweep took them."""
    _, columns, _ = Read_Record(path)
    _Need(columns, LINEARITY_COLUMNS, path)
    numbers = {name: _Numbers(columns[name]) for name in LINEARITY_COLUMNS[2:]}
    return [{'Kind': columns['kind'][i], 'Direction': columns['direction'][i],
             'Set_V': float(numbers['set_v'][i]), 'Read_V': float(numbers['read_v'][i]),
             'Sigma_V': float(numbers['sigma_v'][i]), 'Channel': int(numbers['channel'][i]),
             'Dac_V': float(numbers['dac_v'][i]), 'Time_S': float(numbers['time_s'][i])}
            for i in range(len(columns['kind']))]


def Read_Table_Sweep(path):
    """``(points, rungs)`` of the sweep at ``path`` and those it ``Resumes``,
    on the first one's clock, each cut after its last 0 V check; ``rungs`` is
    how many of the plan's rungs they hold."""
    chain, seen = [], set()
    while path and path not in seen:
        seen.add(path)
        header, _, _ = Read_Record(path)
        chain.insert(0, (path, header))
        earlier = header.get('Resumes')
        path = os.path.join(os.path.dirname(path), earlier) if earlier else None
    points, rungs, first = [], 0, None
    for path, header in chain:
        taken = Read_Linearity(path)
        zeros = [i for i, point in enumerate(taken) if point['Kind'] == 'zero']
        taken = taken[:zeros[-1] + 1] if zeros else []
        when = datetime.strptime(header['Date'], '%Y-%m-%d %H:%M:%S')
        first = first or when
        shift = (when - first).total_seconds()
        points += [dict(point, Time_S=point['Time_S'] + shift) for point in taken]
        rungs += sum(point['Kind'] == 'sweep' for point in taken)
    return points, rungs


def Read_Cooler(path):
    """``(times, volts, integration_s)`` of a cooler record: the integration
    from its settings, the NPLC's if it was set, else the aperture's."""
    header, columns, _ = Read_Record(path)
    _Need(columns, COOLER_COLUMNS, path)
    nplc = header.get('nplc', 'None')
    integration = (float(nplc) / POWER_LINE_HZ if nplc not in ('None', '')
                   else float(header.get('aperture_s', 0.0)))
    return _Numbers(columns['time_s']), _Numbers(columns['cooler_v']), integration
