"""The hardware calibrations: the amplifier's settling, the converters' linearity, the cooler.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

These answer the briefing's open questions about the rack itself, which
no software change can. The settle test steps the platform while the
line meter samples it on its own timer, and measures how long the
amplifier takes to settle: the number SETTLE stands in for, and from
which the scan's settle model is made. The
linearity sweep reads each converter branch over its range and fits it,
so the seam at 0 V (Mode C) can be measured and corrected; the
correction it proposes carries its date and an expiry, because the two
branches drift apart by up to about 0.1 V a day. The cooler monitor
records the cooler's voltage and its noise spectrum.

The DAC is driven as in calibration, by step and confirm through the
acquisition process ('Calibrate', 'Calibration_Step', 'Stop_Calibration'),
so only a plan validated whole against the DAC's limit is ever written.
Moves between measured points are ramped, and every exit ramps the
platform back to 0 V and leaves the meter on its live poll.

Times in a step response are from its 10 % crossing, on the meter's own
clock, so they need nothing from the PC's. The lag from the command to
that crossing is the one number that compares the two clocks, and is
approximate: it assumes the meter's first sample is taken as INIT goes
out.

Depends on: numpy; cls_acquisition.acquisition, .calibration, .scan_plan
and .hardware.mcc_dac.
"""

import os
import time
from datetime import datetime

import numpy as np

from cls_acquisition.acquisition import REPLY_TIMEOUT_S, Ramp
from cls_acquisition.calibration import (PASS, THRESHOLDS, WARN, Calibrating, Calibration_Error,
                                         Line_Fit, Step_At_Zero, _Quietly)
from cls_acquisition.hardware.mcc_dac import AMPLIFIER_GAIN, CH_NEG, CH_POS
from cls_acquisition.scan_plan import Scan_Plan

DIRECTIONS = ('Up', 'Down', 'Up then down (hysteresis)')
PRE_STEP_FRACTION = 0.1             # of a settle capture, sampled before the step
SETTLE_BAND_SIGMAS = 5.0            # a narrower band is left by noise alone, long after settling
MIN_STEP_SIGMAS = 10.0              # a smaller step cannot be told from the noise
MAX_GAIN_ERROR = 0.02               # a branch further from 1 is a divider or wiring fault
BRANCHES = ((CH_POS, 'positive'), (CH_NEG, 'negative'))
POWER_LINE_HZ = 50.0                # an NPLC's length at IGISOL
STOP_POLL_S = 0.05                  # how often a long wait checks for a stop


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


def Linearity_Plan(start, stop, step_v, direction, ramp_step_v):
    """A ladder laid out by count (no float accumulation), measured at every rung."""
    up = Scan_Plan([start], [stop], step_v).Setpoints.tolist()
    points = {'Up': up, 'Down': up[::-1], DIRECTIONS[2]: up + up[-2::-1]}[direction]
    plan = Bench_Plan(ramp_step_v)
    for volts in points:
        plan.Ramp_To(volts, label=volts)
    return plan


def Run_Bench(link, meter, capture, plan, ramp, measure, stopped):
    """Write ``plan`` by step and confirm. ``capture`` sets the meter up
    (:meth:`_Meter.Capture`'s arguments); ``ramp`` is ``(step V, rate V/s)``,
    the pace of every unmeasured move; ``measure(write, label)`` is called at
    each mark and must call ``write()`` itself, which writes the point and
    returns its confirmation. ``stopped()`` true ends the plan early. Every
    exit ramps the platform back to 0 V. Returns the 'Calibrate' reply."""
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

    with Calibrating(link, meter, lambda: meter.Capture(**capture), plan.Setpoints,
                     down) as dac:
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
    """The response to one step captured from before it: times from its 10 %
    crossing, the deviation from the final value, and the figures.
    Settled means inside ``tolerance_v`` of the final value for good, or
    inside SETTLE_BAND_SIGMAS of the noise if that is wider."""
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
                hold_s, tolerance_v, ramp, progress, stopped):
    """Each step of :func:`Settle_Plan` captured from ``PRE_STEP_FRACTION`` of
    ``window_s`` before it; ``progress(result)`` after each. Returns the
    results in order."""
    results = []

    def measure(write, label):
        number, mode, rep, start, stop = label
        time.sleep(hold_s)
        armed = meter.Arm()
        time.sleep(PRE_STEP_FRACTION * window_s)
        step = write()
        time.sleep((1 - PRE_STEP_FRACTION) * window_s)
        volts = meter.Fetch() * divider
        response = Step_Response(interval_s * np.arange(len(volts)), volts, tolerance_v)
        lag = response['T10_S'] - (step['T_Command_Ns'] - armed) / 1e9
        result = dict(response, Step=number, Mode=mode, Rep=rep, From_V=start, To_V=stop,
                      Lag_S=lag)
        results.append(result)
        progress(result)

    capture = {'count': int(round(window_s / interval_s)), 'interval_s': interval_s,
               'aperture_s': aperture_s}
    Run_Bench(link, meter, capture, Settle_Plan(steps, reps, ramp[0]), ramp, measure, stopped)
    return results


def Settle_Summary(results):
    """``(verdict, sentence)``: the slowest step against ORACLE's SETTLE."""
    if not results:
        return WARN, "No step was measured."
    worst = max(results, key=lambda r: r['Lag_S'] + r['Settle_S'])
    needed, settle = worst['Lag_S'] + worst['Settle_S'], THRESHOLDS['SettleS']
    sentence = (f"Slowest: step {worst['Step']} {worst['Mode']} ({worst['From_V']:g} to "
                f"{worst['To_V']:g} V), {worst['Settle_S'] * 1e3:.1f} ms to settle after "
                f"a {worst['Lag_S'] * 1e3:.1f} ms lag, {needed * 1e3:.1f} ms from the "
                f"command against ORACLE's SETTLE of {settle * 1e3:.0f} ms.")
    return (PASS if needed <= settle else WARN), sentence


def Settle_Model(results, today=None):
    """The scan's settle model from a settle test: for each jump size, the
    slowest step's lag plus settling, made non-decreasing in the jump (a
    larger jump never settles faster), dated."""
    slowest = {}
    for r in results:
        jump = round(abs(r['To_V'] - r['From_V']), 6)
        slowest[jump] = max(slowest.get(jump, 0.0), r['Lag_S'] + r['Settle_S'])
    jumps = sorted(slowest)
    settles = np.maximum.accumulate([slowest[jump] for jump in jumps])
    return {'measured': _Today(today),
            'points': [[float(jump), float(s)] for jump, s in zip(jumps, settles)]}


# ── Linearity ───────────────────────────────────────────────────

def Linearity_Sweep(link, meter, divider, start, stop, step_v, direction, settle_s, readings,
                    nplc, ramp, progress, stopped):
    """Read every rung of :func:`Linearity_Plan` ``readings`` times, ``settle_s``
    after its write; ``progress(point)`` after each. Returns the arrays,
    the fits and the 'Calibrate' reply."""
    commanded, means, sigmas = [], [], []

    def measure(write, _label):
        step = write()
        time.sleep(settle_s)
        meter.Arm()
        time.sleep(readings * nplc / POWER_LINE_HZ)     # FETC? then waits only for autozero
        volts = meter.Fetch() * divider
        commanded.append(step['Volts'])
        means.append(float(np.mean(volts)))
        sigmas.append(float(np.std(volts)))
        progress({'Set_V': step['Volts'], 'Read_V': means[-1], 'Sigma_V': sigmas[-1],
                  'Channel': step['Channel']})

    plan = Linearity_Plan(start, stop, step_v, direction, ramp[0])
    dac = Run_Bench(link, meter, {'count': readings, 'nplc': nplc}, plan, ramp, measure,
                    stopped)
    commanded, means = np.array(commanded), np.array(means)
    return {'Set_V': commanded, 'Read_V': means, 'Sigma_V': np.array(sigmas), 'Dac': dac,
            'Fits': Branch_Fits(commanded, means), 'Seam_V': Step_At_Zero(commanded, means),
            'Hysteresis_V': Hysteresis(commanded, means)}


def Branch_Fits(commanded, readback):
    """Each branch's line, for every branch with three points or more. 0 V
    is in neither: it leaves the platform on whichever amplifier was driven
    last, so it belongs to the side the sweep came from."""
    fits = {}
    for channel, side in ((CH_POS, commanded > 0), (CH_NEG, commanded < 0)):
        if side.sum() >= 3:
            gain, offset, rms = Line_Fit(commanded[side], readback[side])
            fits[channel] = {'Gain': gain, 'Offset_V': offset, 'Rms_V': rms,
                             'Points': int(side.sum())}
    return fits


def Hysteresis(commanded, readback):
    """The mean of (second reading - first) over the setpoints read on the way
    up and again on the way down, or None if none was read twice: noise
    averages out of it, hysteresis does not. 0 V is left out, since its
    two readings differ by the seam."""
    values, inverse, counts = np.unique(np.round(commanded, 6), return_inverse=True,
                                        return_counts=True)
    twice = [readback[inverse == i] for i in np.nonzero((counts == 2) & (values != 0))[0]]
    return float(np.mean([b - a for a, b in twice])) if twice else None


def Corrected(correction, fits, valid_days, method, today=None):
    """The DAC correction that cancels each branch's measured gain and offset,
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
    return {'instrument': correction['instrument'], 'measured': _Today(today),
            'valid_days': int(valid_days), 'method': method, 'poly': poly}


# ── Cooler ──────────────────────────────────────────────────────

def Cooler_Monitor(meter, divider, duration_s, block_s, interval_s, aperture_s, progress,
                   stopped):
    """Blocks of ``block_s`` sampled every ``interval_s`` for ``duration_s``;
    ``progress((times, volts))`` after each. Returns the record and its
    spectrum."""
    count = int(round(block_s / interval_s))
    blocks, starts = [], []
    try:
        meter.Capture(count, interval_s, aperture_s)
        t0 = time.monotonic()
        while time.monotonic() - t0 < duration_s and not stopped():
            starts.append(time.monotonic() - t0)
            meter.Arm()
            deadline = time.monotonic() + block_s
            while time.monotonic() < deadline and not stopped():
                time.sleep(STOP_POLL_S)
            blocks.append(meter.Fetch() * divider)
            progress((starts[-1] + interval_s * np.arange(len(blocks[-1])), blocks[-1]))
    except BaseException:
        _Quietly(meter.Release)
        raise
    meter.Release()
    if not blocks:
        raise Calibration_Error("The monitor was stopped before its first block.")
    times = np.concatenate([s + interval_s * np.arange(len(b)) for s, b in zip(starts, blocks)])
    volts = np.concatenate(blocks)
    full = [b for b in blocks if len(b) == count]
    frequency, asd = Amplitude_Spectral_Density(full or blocks[:1], interval_s)
    drift = Line_Fit(times, volts)[0] if len(volts) > 1 else 0.0
    return {'Times_S': times, 'Volts': volts, 'Frequency_Hz': frequency, 'Asd': asd,
            'Mean_V': float(np.mean(volts)), 'Sigma_V': float(np.std(volts)),
            'Peak_To_Peak_V': float(np.ptp(volts)), 'Drift_V_Per_Min': float(drift * 60)}


def Amplitude_Spectral_Density(blocks, interval_s):
    """``(frequency Hz, V/sqrt(Hz))``, one-sided, averaged over equal blocks,
    each detrended and Hann-windowed, so drift and the block edges do not
    leak into it."""
    length = min(len(b) for b in blocks)
    window, x = np.hanning(length), np.arange(length)
    detrended = [b[:length] - np.polyval(np.polyfit(x, b[:length], 1), x) for b in blocks]
    power = np.mean([np.abs(np.fft.rfft(d * window)) ** 2 for d in detrended], axis=0)
    asd = np.sqrt(2.0 * power * interval_s / np.sum(window ** 2))
    return np.fft.rfftfreq(length, interval_s)[1:], asd[1:]


# ── Records ─────────────────────────────────────────────────────

def Save(directory, kind, header, columns, rows):
    """One measurement as a CSV under ``<directory>/hardware_calibration``,
    its settings and results as ``#`` lines above the columns. Returns the
    path; never overwrites."""
    folder = os.path.join(directory, 'hardware_calibration')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{kind}_{datetime.now():%Y-%m-%d_%H%M%S}.csv")
    with open(path, 'x', encoding='utf-8', newline='') as fh:
        for key, value in header.items():
            fh.write(f"# {key}: {value}\n")
        fh.write(','.join(columns) + '\n')
        for row in zip(*rows):
            fh.write(','.join(v if isinstance(v, str) else f"{v:.9g}" for v in row) + '\n')
    return path
