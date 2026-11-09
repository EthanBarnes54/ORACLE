"""Scan performance: how the rack's behaviour during a run shaped its spectrum.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A spectrum is counts against the scanning voltage, and every part of the
pipeline can move a count to the wrong voltage or lose it: the DAC and
its amplifier, the loop's timing, the cooler, the laser, the beam, the
detectors. This reads a finished run (its file, and its monitor record
if it has one) and asks of each part whether it stayed inside tolerance.
Each check returns PASS, WARN or FAIL with the number it decided on, or
N/A when the run does not hold what it needs. Nothing is cut: what
deserves a look is flagged, and the operator decides in DENIS. The
verdict is the worst check's; it never stops a run being kept.

The spectrum is judged as DENIS builds it: each hit at its voltage tag
less its own cooler reading's departure (DENIS's point-by-point
correction), on one grid from the lowest setpoint to the highest, so
ranges that overlap (the old DAQ allowed them) are one range. A shift or
spread is set against the width of the run's own strongest line, and
given in MHz where the run holds the mass, cooler and laser to convert
it; where no line stands out of the background, the step is the scale.

Statistics hold at any count rate. Rather than scan by scan (a fast
sawtooth has a few hits a scan), the run is cut into periods that hold
enough counts to say something, and a deviation is flagged only when it
is both improbable under counting statistics and large enough to
matter. Idle time is what the bunch count does not account for, so a
run with a hit a minute holds no false pauses.

The line's position is fitted period by period against the run's own
summed spectrum, which needs no lineshape or hyperfine model and uses
every peak. The monitors (the HV against its calibration, the laser
against the setpoint DENIS takes) say how far they should have moved
it, and what they leave unexplained is reported. The calibration is
judged as DENIS fits it (a straight line, its robust outlier rule) and
against the runs before it.

A run of the old DAQ numbered its bunches by the tagger's count and
stepped without waiting. Its scan type and the count's origin are found
from whichever fits best, the bunches it took before a step settled are
estimated from the settle model, and a bunch count that parted from the
voltage steps (the collinear DAQ's D5) is reported, not cut.

Depends on: numpy, PyYAML; cls_acquisition.calibration, .run_file,
.scan_plan and .hardware.keysight.
"""

import math
import os
from datetime import datetime

import numpy as np
import yaml

from cls_acquisition.calibration import DIVIDER, PASS, THRESHOLDS, WARN, Settle_For
from cls_acquisition.hardware.keysight import COOLER_DIVIDER
from cls_acquisition.run_file import ORACLE_KEY, Recorded_Divider
from cls_acquisition.scan_plan import Scan_Plan, Scan_Plan_Error

FAIL, NA = 'FAIL', 'N/A'
RANK = {NA: -1, PASS: 0, WARN: 1, FAIL: 2}
VOLTAGE, TIMING, BEAM, SOURCES = 'Voltage scan', 'Timing', 'Beam and spectrum', 'Sources'

NUMERICAL_SCATTER = 1e-9                # of a calibration's span: below it, no scatter at all

# Every number the analysis decides on. A pair is (WARN above, FAIL above).
TOLERANCES = {
    'LineShare': (0.10, 0.25),          # a shift or spread, against the line's width
    'StepShare': (0.5, 1.0),            # ... against the step, where no line stands out
    'Significance': 3.0,                # standard errors before a shift is taken as real
    'ExcessZ': 2.326,                   # chi2 beyond counting: one-sided 1 %
    'Improbable': 1e-3,                 # a deficit less likely than this under counting
    'LineSigmas': 5.0,                  # a line stands this far out of its background
    'LineCounts': 50,                   # counts over background a period needs to place it
    'ShapeCounts': 200,                 # hits a period needs, with no line to go by
    'MaxPeriods': 12,                   # periods the spectrum is followed over, at most
    'BeamCounts': 20,                   # hits a beam period should expect
    'BeamPeriodS': 10.0,                # its shortest length
    'MaxBeamPeriods': 300,
    'BeamLossFraction': 0.5,            # a period below this share of the usual rate
    'BeamSpread': 0.3,                  # rms variation, beyond counting, worth a warning
    'ShapeSpread': 0.15,
    'ShapeCellCounts': 5,               # a spectrum cell's expected counts in a period
    'ContaminatedFraction': (0.01, 0.05),   # bunches taken before their step had settled
    'KeepingUp': 0.02,                  # reads slower than the triggers by this share
    'StallS': 0.5,                      # time the bunches and steps do not account for
    'PauseS': 5.0,
    'SuspectSigma': 3.0,                # DENIS's robust rule for calibration points
    'SuspectImpactV': 0.05,             # ... worth a flag only if they move the axis this far
    'ChannelCounts': 20,                # hits a channel should expect in a period
    'ChannelFall': 0.3,                 # a channel down to this share of its usual counts
    'TofCounts': 50,                    # TOF peak hits a period needs
    'TofGateShare': 0.5,                # the gate keeps less of the TOF peak than this
    'TofShareChange': 0.1,              # ... or its share moves this much over the run
    'TofDriftUs': 0.5,
}
WIRED_CHANNELS = (1, 2, 3, 4)               # the tagger's inputs, for a run that does not say
DC_PERIOD_S = 0.001                         # the DC trigger
TIMESTAMP_RESOLUTION_S = 0.01               # the run file's timestamps are rounded to this
VOLTS_DECIMALS = 6                          # setpoints compare to the microvolt
MAX_REJECT_ITER = 10                        # DENIS's, for its robust calibration rule
FIT_ROWS = 64                               # shifts tried at once, so the fit stays small
NULL_BUDGET = 2e7                           # bins times shifts the random deals may cost

C_LIGHT = 299792458.0
AMU_EV = 931.49410242e6                     # one atomic mass unit, in eV


class Performance_Error(ValueError):
    pass


# ── Statistics ──────────────────────────────────────────────────

def _Excess(chi2_per_dof, dof):
    """True if a chi2/ndf this large is beyond counting statistics at the
    one-sided 1 % level (Wilson-Hilferty), so a short run does not cry wolf."""
    if dof < 1:
        return False
    shrink = 2.0 / (9.0 * dof)
    z = (chi2_per_dof ** (1.0 / 3.0) - (1.0 - shrink)) / np.sqrt(shrink)
    return bool(z > TOLERANCES['ExcessZ'])


def _At_Most(count, mean):
    """P(X <= count) for X ~ Poisson(mean): summed exactly for a small count,
    else through the gamma distribution it equals (Wilson-Hilferty)."""
    if mean <= 0:
        return 1.0
    if count < 30:
        return min(1.0, sum(math.exp(i * math.log(mean) - mean - math.lgamma(i + 1))
                            for i in range(int(count) + 1)))
    shape = count + 1.0
    z = ((mean / shape) ** (1 / 3) - (1 - 1 / (9 * shape))) / math.sqrt(1 / (9 * shape))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _Short(observed, expected):
    """True if ``observed`` counts fall short of ``expected`` both by enough to
    matter and beyond what counting explains."""
    return (observed < TOLERANCES['BeamLossFraction'] * expected
            and _At_Most(observed, expected) < TOLERANCES['Improbable'])


def _Grade(value, limits):
    warn, fail = limits
    return FAIL if value > fail else WARN if value > warn else PASS


def _Worst(*verdicts):
    return max(verdicts, key=RANK.get)


def _Rise(t, values, errors):
    """The weighted straight line through ``values`` against ``t``: ``(its
    signed rise over the span of t, the line, the rise's standard error)``."""
    w = 1.0 / np.maximum(errors, 1e-12) ** 2
    t_mean, v_mean = np.sum(w * t) / np.sum(w), np.sum(w * values) / np.sum(w)
    spread = np.sum(w * (t - t_mean) ** 2)
    if not spread:
        return 0.0, np.full_like(values, v_mean), 0.0
    slope = np.sum(w * (t - t_mean) * (values - v_mean)) / spread
    span = np.ptp(t)
    return slope * span, v_mean + slope * (t - t_mean), span / np.sqrt(spread)


def _Weighted(values, weights):
    """``(mean, rms spread)`` of the finite ``values``, each weighted by its
    ``weights`` (unweighted if none carries any), or None if none is finite."""
    ok = np.isfinite(values)
    if not ok.any():
        return None
    v, w = values[ok], weights[ok]
    if not w.sum():
        w = np.ones_like(v)
    mean = float(np.sum(w * v) / np.sum(w))
    return mean, float(np.sqrt(np.sum(w * (v - mean) ** 2) / np.sum(w)))


def _Hits_Per_Sample(sample_t, hit_t):
    """How many hits, from their times in order, lie nearer each sample than
    its neighbours: the weight a reading carries for the spectrum."""
    edges = np.r_[-np.inf, (sample_t[1:] + sample_t[:-1]) / 2.0, np.inf]
    return np.diff(np.searchsorted(hit_t, edges)).astype(float)


def _Distinct(x, y):
    """A plotted series less every point that repeats the one before it. A
    bunch's hits share its time and the readings taken with it, so one point
    per hit draws each bunch over and over: millions of points, and a
    gigabyte in the plot, for the picture its distinct points make."""
    moved = np.ones(len(x), dtype=bool)
    moved[1:] = (x[1:] != x[:-1]) | (y[1:] != y[:-1])
    return x[moved], y[moved]


# ── The Doppler shift, and what a volt means ────────────────────

def Ion_Frequency(beam_v, nu_laser_hz, mass_amu, laser):
    """The frequency the ion sees, as DENIS computes it: the laser times its
    harmonic, Doppler-shifted at the beam's speed."""
    gamma = 1.0 + laser['charge'] * np.asarray(beam_v, dtype=float) / (mass_amu * AMU_EV)
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    shift = (1.0 + beta) / (1.0 - beta)
    doppler = np.sqrt(shift if 'anti' in laser['geometry'] else 1.0 / shift)
    return laser['harmonic'] * nu_laser_hz * doppler


def Volts_Per_Hz(beam_v, nu_laser_hz, mass_amu, laser):
    """How far one hertz more at the laser moves the resonance, in scan volts:
    positive when to a higher scan voltage (a laser the ions meet head on),
    negative for a collinear one."""
    # the ion's frequency is in proportion to the laser's, so one hertz at the
    # laser moves it by its value at 1 Hz; one volt more beam energy moves it
    # by its slope, and the scan voltage takes energy from the beam
    per_volt = (Ion_Frequency(beam_v + 1.0, nu_laser_hz, mass_amu, laser)
                - Ion_Frequency(beam_v - 1.0, nu_laser_hz, mass_amu, laser)) / 2.0
    return float(Ion_Frequency(beam_v, 1.0, mass_amu, laser) / per_volt)


def Dividers(header):
    """``(scan, cooler)``: the run's divider ratios as it recorded them, or
    the constants for a run that did not, never today's settings."""
    return (Recorded_Divider(header, 'Scan') or DIVIDER,
            Recorded_Divider(header, 'Cooler') or COOLER_DIVIDER)


def _Beam_V(header, axis):
    """The beam's energy mid-scan, in volts, or NaN if the run lacks it."""
    try:
        return (float(header.get('CoolerVoltage', np.nan)) * Dividers(header)[1]
                - float(axis.mean()))
    except (TypeError, ValueError):
        return np.nan


def _Mhz_Per_V(header, laser, wavenumber_cm1, axis):
    """MHz at the ions per scan volt mid-scan, or None if the run lacks the
    mass, the cooler or the laser to say."""
    try:
        mass = float(header.get('MassAMU', np.nan))
        beam_v, nu = _Beam_V(header, axis), float(wavenumber_cm1) * C_LIGHT * 100.0
        if not (mass > 0 and beam_v > 0 and nu > 0):
            return None
        per_volt = (Ion_Frequency(beam_v + 1.0, nu, mass, laser)
                    - Ion_Frequency(beam_v - 1.0, nu, mass, laser)) / 2.0
    except (KeyError, TypeError, ValueError):
        return None
    return abs(float(per_volt)) / 1e6


class _Scale():
    """What a voltage means for this run: its share of the strongest line's
    width (of the step, where no line stands out), and its size in MHz at
    the ions where the run holds what converts it."""

    def __init__(self, step, width_v=None, mhz_per_v=None):
        self.Step, self.Width, self.Mhz_Per_V = step, width_v, mhz_per_v

    def Share(self, volts):
        return abs(volts) / (self.Width or self.Step)

    def Limits(self):
        return TOLERANCES['LineShare' if self.Width else 'StepShare']

    def Grade(self, volts):
        return _Grade(self.Share(volts), self.Limits())

    def Matters(self, volts):
        return self.Share(volts) > self.Limits()[0]

    def Short(self, volts):
        if self.Mhz_Per_V:
            return f"{abs(volts) * self.Mhz_Per_V:.1f} MHz"
        return f"{abs(volts):.3f} V"

    def Text(self, volts):
        """``volts`` as an operator weighs it: in volts, in MHz, and against
        the line."""
        parts = [f"{abs(volts):.3f} V"]
        if self.Mhz_Per_V:
            parts.append(f"{abs(volts) * self.Mhz_Per_V:.1f} MHz")
        parts.append(f"{self.Share(volts):.0%} of the line width" if self.Width
                     else f"{self.Share(volts):.2f} steps")
        return ', '.join(parts)


# ── The report ──────────────────────────────────────────────────

class _Report():

    def __init__(self):
        self.Sections = {name: [] for name in (VOLTAGE, TIMING, BEAM, SOURCES)}
        self.Series, self.Flagged, self.Budget = {}, [], {}

    def Add(self, section, check, verdict, value, message):
        self.Sections[section].append(
            {'Check': check, 'Verdict': verdict, 'Value': value, 'Message': message})

    def Flag(self, text):
        self.Flagged.append(text)

    def Result(self, summary, budget):
        sections = {name: checks for name, checks in self.Sections.items() if checks}
        checks = [c for section in sections.values() for c in section]
        verdict = max((c['Verdict'] for c in checks), key=RANK.get, default=NA)
        unchecked = sum(c['Verdict'] == NA for c in checks)
        if unchecked:
            summary += (f" {unchecked} of the {len(checks)} checks need what this run does not "
                        f"hold (N/A).")
        return {'Verdict': verdict, 'Summary': summary, 'Budget': budget, 'Sections': sections,
                'Series': self.Series, 'Flagged': self.Flagged}


def Analyse(header, raw, monitor, laser, settle_model=None, history=()):
    """The report on one run: ``Verdict``, ``Summary``, ``Budget`` (the line's
    width and what moved it, at the ions, in one sentence), ``Sections``
    (name to checks), ``Series`` for the plots and ``Flagged``, sentences
    on what deserves a look (nothing is cut). ``laser`` is the experiment's
    {'harmonic', 'geometry', 'charge'}, used if the run does not record its
    own; ``settle_model`` estimates an old DAQ run's settling; ``history``
    holds earlier runs' calibrations, ``(run, CalSet, CalReadback, scan
    divider or None)``, newest first, of runs calibrated as this one was
    (run_file.Earlier_Calibrations). Readbacks are scaled by each run's
    own divider ratios (:func:`Dividers`)."""
    if not len(raw):
        raise Performance_Error("The run holds no events, so there is nothing to analyse.")
    record = header.get(ORACLE_KEY) or {}
    recorded_laser = bool(record.get('Laser'))
    laser = dict(record.get('Laser') or laser)
    sequence, axis, note = _Plan(header)
    step = float(header['StepSize'])
    per_step = max(int(header.get('BunchesPerChannel', 1) or 1), 1)
    report = _Report()

    times = raw[:, 0]
    by_time = None if np.all(times[1:] >= times[:-1]) else np.argsort(times, kind='stable')
    times_t = times if by_time is None else times[by_time]
    t0 = float(times_t[0])

    tags = _Tags(report, sequence, per_step, raw[:, 1], raw[:, 2], record, monitor)
    bunch = raw[:, 2] - tags['Origin']
    by_bunch = None if np.all(bunch[1:] >= bunch[:-1]) else np.lexsort((times, bunch))
    bunch_b = bunch if by_bunch is None else bunch[by_bunch]
    period, bunched = _Timing(report, bunch_b, times if by_bunch is None else times[by_bunch],
                              header, record, monitor, per_step, sequence, tags, settle_model, t0)
    _Rates(report, bunch_b, tags, len(sequence), per_step)
    del bunch_b

    # the spectrum as DENIS builds it, the cooler taken out hit by hit
    bins = _Bins(raw, axis, step, Dividers(header)[1])
    bins_t = bins if by_time is None else bins[by_time]
    del bins
    scanned = np.isin(np.round(axis, VOLTS_DECIMALS), np.round(sequence, VOLTS_DECIMALS))
    total = np.bincount(bins_t[bins_t >= 0], minlength=len(axis)).astype(float)
    line = _Line(axis, total, scanned)
    wavenumber = header.get('LaserSetpoint', np.nan)
    if monitor is not None and np.isfinite(monitor.get('wavenumber_cm1', [np.nan])).any():
        wavenumber = np.nanmean(monitor['wavenumber_cm1'])
    scale = _Scale(step, line[1] if line else None,
                   _Mhz_Per_V(header, laser, wavenumber, axis))

    _Calibration(report, header, scale, history)
    hv = _Readback(report, monitor, header, scale, times_t, t0)
    light = _Laser(report, monitor, header, laser, recorded_laser, scale, times_t, axis, t0)
    _Cooler(report, raw, monitor, t0, Dividers(header)[1])
    bunch_t = bunch if by_time is None else bunch[by_time]
    per_scan = len(sequence) * per_step
    expectation = None
    if tags['Reliable'] and tags['Complete'] >= 3:
        tagged = np.rint((raw[:, 1] - axis[0]) / step).astype(np.intp)
        seen = np.bincount(tagged[(tagged >= 0) & (tagged < len(axis))], minlength=len(axis))
        del tagged
        expectation = _Expectation(sequence, tags['Kind'], per_step, axis, seen, bunch.max())
    _Beam(report, times_t, bunch_t, monitor, tags, expectation, per_scan * period, per_scan, t0)
    _Spectrum(report, bins_t, bunch_t, times_t, axis, scanned, scale, line,
              _Prediction(hv, light, times_t), tags, per_scan)
    del bunch_t
    _Channels(report, raw[:, 3] if by_time is None else raw[by_time, 3], times_t,
              record.get('Channels') or WIRED_CHANNELS, t0)
    _Tof(report, raw[:, 4] if by_time is None else raw[by_time, 4], times_t, record, bunched)

    summary = (f"{tags['Complete']} complete {tags['Type']} scans of {len(sequence)} "
               f"setpoints, {per_step} bunches per step, {len(raw)} hits over "
               f"{(times_t[-1] - t0) / 60:.1f} min; bunches every {period * 1e3:.1f} ms.{note}")
    return report.Result(summary, _Budget(scale, report.Budget))


def _Budget(scale, budget):
    """The line's width and what moved it, at the ions, in one sentence."""
    if scale.Width:
        text = f"The line is {scale.Width:.3f} V wide" + (
            f" ({scale.Width * scale.Mhz_Per_V:.0f} MHz)." if scale.Mhz_Per_V else ".")
    else:
        text = (f"No line stands out of the background, so sizes are set against the "
                f"{scale.Step:g} V step.")
    parts = [f"{name} {scale.Short(budget[key])}" for name, key in
             (('HV', 'HV'), ('laser', 'Laser')) if key in budget]
    if 'Moved' in budget:
        rise, error = budget['Moved']
        parts.append(f"the line itself moved {scale.Short(rise)} ± {scale.Short(error)}")
    if parts:
        text += " Shifts and spreads it suffered: " + ', '.join(parts) + '.'
    return text


# ── The voltage axis ────────────────────────────────────────────

def _Ladder(low, high, step):
    return low + step * np.arange(int(round((high - low) / step)) + 1)


def _Plan(header):
    """``(sequence, axis, note)``: the setpoints one scan commands, in order,
    and the grid from the lowest to the highest the spectrum is binned on.
    Ranges the old DAQ took and ORACLE would refuse (overlapping ones, or
    one of a single point) are taken as it commanded them, each range's
    ladder in turn, and binned as one range."""
    try:
        ranges = [sorted((float(a), float(b))) for a, b in header['ScanningRanges']]
        step = float(header['StepSize'])
    except (KeyError, TypeError, ValueError) as e:
        raise Performance_Error(f"The run's scanning ranges cannot be read: {e}") from None
    if not ranges or not step > 0:
        raise Performance_Error(f"The run's scanning ranges ({ranges}) and step ({step:g} V) "
                                f"hold no scan.")
    note = ''
    try:
        sequence = Scan_Plan(*zip(*ranges), step).Setpoints
    except Scan_Plan_Error:
        sequence = np.concatenate([_Ladder(low, high, step) for low, high in ranges])
        if len(np.unique(np.round(sequence, VOLTS_DECIMALS))) < len(sequence):
            note = (f" Its ranges overlap, so the overlap was scanned twice; it is binned as "
                    f"one range, {sequence.min():g} to {sequence.max():g} V.")
    return sequence, _Ladder(sequence.min(), sequence.max(), step), note


def _Bins(raw, axis, step, cooler_divider):
    """Each hit's bin on ``axis`` as DENIS places it: its voltage tag less its
    cooler reading's departure from the run's median (DENIS corrects every
    hit by its own reading), or -1 off the grid. A hit with no cooler
    reading stays at its tag here; the cooler check counts them, as DENIS
    cannot place them. One working array, rounded in place."""
    cooler = raw[:, 5]
    known = np.isfinite(cooler) & (cooler != 0)
    work = np.where(known, cooler, 0.0)
    if known.any():
        work[known] -= np.median(cooler[known])
    work *= -cooler_divider
    work += raw[:, 1]
    work -= axis[0]
    work /= step
    np.rint(work, out=work)
    bins = work.astype(np.int32)
    bins[(bins < 0) | (bins >= len(axis))] = -1
    return bins


def _Step_Index(steps, n, kind):
    """Which of a scan's ``n`` setpoints each step counted from the start
    commands: a triangle runs every other scan backwards."""
    position, scan = steps % n, steps // n
    reverse = (kind == 'Triangle') & (scan % 2 == 1)
    return np.where(reverse, n - 1 - position, position).astype(np.intp)


def _Tags(report, sequence, per_step, volts, bunch, record, monitor):
    """Whether each hit's voltage is the setpoint its bunch was due at: for
    ORACLE's runs, whose loop writes both from one count, a test of the
    file; for the old DAQ's, whose bunch number was the tagger's count, of
    whether that count stayed with the steps. Its scan type and the count's
    origin are what the run records, or what fits it best."""
    n = len(sequence)
    planned = np.round(sequence, VOLTS_DECIMALS)
    rounded = np.round(volts, VOLTS_DECIMALS)

    def follows(origin, kind):
        return rounded == planned[_Step_Index((bunch - origin) // per_step, n, kind)]

    recorded = record.get('ScanType')
    if recorded:
        origin, kind = 0, 'Triangle' if recorded == 'Triangle' else 'Sawtooth'
    else:
        trials = {(o, k): follows(o, k).mean() for o in (0, 1) for k in ('Sawtooth', 'Triangle')}
        origin, kind = max(trials, key=trials.get)
    agree = follows(origin, kind)
    wrong = int(np.count_nonzero(~agree))
    mismatch = wrong / len(volts)
    off_plan = int(np.count_nonzero(~np.isin(rounded, planned)))
    del agree, rounded

    counted = monitor is not None and 'bunches' in monitor
    total = int(monitor['bunches'][-1]) if counted else int(bunch.max() - origin + 1)
    single = recorded == 'Single scan' or (bunch.max() - origin) // (n * per_step) == 0
    reliable = mismatch < 1e-3
    if record:
        verdict = FAIL if wrong or off_plan else PASS
        message = ("Every hit carries the setpoint its bunch was due at, and every one is in "
                   "the plan." if verdict == PASS else
                   f"{wrong} of {len(volts)} hits carry a voltage other than the setpoint their "
                   f"bunch was due at, {off_plan} one the plan never held. ORACLE's loop writes "
                   f"both from one count, so the file is damaged, or was not written by ORACLE.")
    else:
        verdict = WARN if wrong or off_plan else PASS
        if not wrong:
            message = (f"Every hit carries the setpoint its bunch number points to (the old DAQ's "
                       f"count, from {origin}).")
        else:
            message = (f"{wrong} of {len(volts)} hits ({mismatch:.2%}) carry a voltage other "
                       f"than the setpoint their bunch number points to: the tagger's count and "
                       f"the voltage steps parted (the old DAQ's D5). Which is wrong cannot be "
                       f"told from the file. The spectrum is built from the voltage tags, as "
                       f"DENIS builds it, but "
                       + ("scan numbers, and DENIS's scan filter, are unreliable for this run."
                          if not reliable else "scans after the parting are misnumbered."))
        if off_plan:
            message += f" {off_plan} hits are at a voltage the plan never held."
    report.Add(VOLTAGE, 'Voltage tags', verdict, mismatch, message)
    if not reliable:
        report.Flag("Scan numbers are unreliable: the bunch count parted from the voltage "
                    "steps, so filter this run by time, not by scan, in DENIS.")
    return {'Origin': origin, 'Kind': kind, 'Reliable': reliable,
            'Type': 'single' if single else kind.lower(),
            'Complete': total // (n * per_step)}


def _Order_One(cal_set, readback, keep=None):
    """``(gain, offset)`` of the straight line through the points ``keep``
    picks, as DENIS fits a calibration unless told otherwise."""
    if keep is not None:
        cal_set, readback = cal_set[keep], readback[keep]
    gain, offset = np.polyfit(cal_set, readback, 1)
    return float(gain), float(offset)


def _Suspects(cal_set, readback):
    """The points DENIS's robust rule nominates: the line refitted after each
    pass, the scatter from the median absolute deviation, SuspectSigma of
    it, and never so many that the fit could not stand."""
    n, excluded = len(cal_set), set()
    for _ in range(MAX_REJECT_ITER):
        if n - len(excluded) < 3:
            break
        keep = np.ones(n, dtype=bool)
        keep[list(excluded)] = False
        gain, offset = _Order_One(cal_set, readback, keep)
        residuals = readback - (gain * cal_set + offset)
        used = residuals[keep]
        scatter = 1.4826 * float(np.median(np.abs(used - np.median(used))))
        # a scatter a billionth of the span is arithmetic, not measurement: on a clean
        # line, which points it puts beyond SuspectSigma depends on the maths library
        if scatter <= NUMERICAL_SCATTER * max(float(np.ptp(readback)), 1.0):
            break
        new = {int(i) for i in np.flatnonzero(np.abs(residuals) > TOLERANCES['SuspectSigma']
                                              * scatter)} - excluded
        room = (n - 3) - len(excluded)
        if not new or room <= 0:
            break
        excluded |= set(sorted(new, key=lambda i: -abs(residuals[i]))[:room])
    return sorted(excluded)


def _Calibration_Points(header):
    cal_set = np.asarray(header.get('CalSet', []), dtype=float)
    readback = np.asarray(header.get('CalReadback', []), dtype=float) * Dividers(header)[0]
    n = min(len(cal_set), len(readback))
    return cal_set[:n], readback[:n]


def _Tracking(cal_set, readback):
    return abs(np.corrcoef(cal_set, readback)[0, 1]) if np.ptp(readback) else 0.0


def _Calibration(report, header, scale, history):
    """The start-of-run calibration as DENIS fits and checks it: what its
    straight line leaves, where that puts the peaks, whether points stand
    out under DENIS's robust rule and how far leaving them out would move
    the axis; then against the runs before it."""
    cal_set, readback = _Calibration_Points(header)
    if len(cal_set) < 3:
        report.Add(VOLTAGE, 'Calibration', NA, None, "The run holds no calibration.")
        return
    tracking = _Tracking(cal_set, readback)
    if tracking < THRESHOLDS['MinTrackingR']:
        report.Add(VOLTAGE, 'Calibration', FAIL, tracking,
                   f"The readback does not follow the command (|r| {tracking:.3f}): the HV, "
                   f"its divider or the line meter was not working as the run was calibrated, "
                   f"so its voltage axis cannot be trusted.")
        return
    gain, offset = _Order_One(cal_set, readback)
    residuals = readback - (gain * cal_set + offset)
    worst = float(np.max(np.abs(residuals)))
    message = (f"Fitted as DENIS fits it, a straight line: gain {gain:.5f}, offset "
               f"{offset:+.3f} V. It leaves {np.sqrt(np.mean(residuals ** 2)):.3f} V rms, so a "
               f"peak can sit up to {scale.Text(worst)} from where the axis puts it.")
    suspects = _Suspects(cal_set, readback)
    verdict = scale.Grade(worst)
    if suspects:
        keep = np.ones(len(cal_set), dtype=bool)
        keep[suspects] = False
        trimmed = _Order_One(cal_set, readback, keep)
        # two lines are furthest apart at an end of the scan
        impact = max(abs((gain - trimmed[0]) * x + offset - trimmed[1])
                     for x in (cal_set.min(), cal_set.max()))
        flagged = impact > TOLERANCES['SuspectImpactV']
        where = ', '.join(f"{cal_set[i]:g}" for i in suspects[:5]) + (
            ', ...' if len(suspects) > 5 else '')
        message += (f" {len(suspects)} points stand out under DENIS's robust "
                    f"{TOLERANCES['SuspectSigma']:g}-sigma rule (at {where} V); leaving them out "
                    f"moves the axis by up to {scale.Text(impact)}"
                    + (": worth a look in DENIS." if flagged else ", too little to matter."))
        if flagged:
            verdict = _Worst(verdict, WARN)
            report.Flag(f"Calibration points at {where} V stand out; leaving them out in DENIS "
                        f"moves the axis by up to {scale.Text(impact)}.")
    report.Add(VOLTAGE, 'Calibration', verdict, worst, message)
    _Calibration_History(report, (gain, offset), cal_set, history, scale)


def _Calibration_History(report, line, cal_set, history, scale):
    """This run's calibration against the runs' before it, across this scan:
    how much an isotope shift measured between them depends on which is
    right. Agreeing with any one of them is enough, so one bad neighbour
    cannot raise it."""
    found = []
    for run, set_b, read_b, divider in history:
        set_b = np.asarray(set_b, dtype=float)
        read_b = np.asarray(read_b, dtype=float) * (divider or DIVIDER)
        n = min(len(set_b), len(read_b))
        set_b, read_b = set_b[:n], read_b[:n]
        if n < 3 or _Tracking(set_b, read_b) < THRESHOLDS['MinTrackingR']:
            continue
        gain, offset = _Order_One(set_b, read_b)
        apart = max(abs((line[0] - gain) * x + line[1] - offset)
                    for x in (cal_set.min(), cal_set.max()))
        found.append((run, apart, gain, offset))
    if not found:
        return
    run, apart, gain, offset = found[0]
    closest = min(found, key=lambda f: f[1])
    verdict = WARN if scale.Grade(closest[1]) == FAIL else PASS
    message = (f"Across this scan, its calibration and run {run}'s disagree by up to "
               f"{scale.Text(apart)} (gain {(line[0] / gain - 1) * 1e6:+.0f} ppm, offset "
               f"{line[1] - offset:+.3f} V)")
    if len(found) > 1 and closest[0] != run:
        message += (f"; run {closest[0]}'s, the closest of the last {len(found)}, by "
                    f"{closest[1]:.3f} V")
    message += (". An isotope shift measured between them depends on which is right by that "
                "much" + (": the HV chain (divider, meter or cabling) may have changed; check "
                          "it before trusting either run." if verdict == WARN else "."))
    report.Add(VOLTAGE, 'Calibration against earlier runs', verdict, apart, message)


# ── Timing ──────────────────────────────────────────────────────

def _Timing(report, bunch, times, header, record, monitor, per_step, sequence, tags,
            settle_model, t0):
    """Whether the loop kept up with the triggers, whether it ever stood idle,
    and whether any bunch was taken before its step had settled, from the
    hits in bunch order: ``(the trigger period, whether bunched; None if the
    run does not say)``."""
    timing = record.get('Timing') or {}
    if timing.get('rep_rate_hz'):
        period, bunched = 1.0 / float(timing['rep_rate_hz']), bool(timing.get('bunched', True))
    else:
        # the old DAQ counted every trigger, so the hits say which period was in
        # force; its header keeps the bunched dwell even for a DC run
        dwell = float(header.get('DwellTime', 0) or 0) / 1e3 or DC_PERIOD_S
        span = bunch[-1] - bunch[0]
        measured = (times[-1] - times[0]) / span if span > 0 else np.nan
        period = (min((dwell, DC_PERIOD_S), key=lambda p: abs(np.log(p / measured)))
                  if measured > 0 else dwell)
        bunched = None
    step_of = bunch // per_step
    gaps, bunches, crossings = np.diff(times), np.diff(bunch), np.diff(step_of)
    same = crossings == 0

    span = bunches[same].sum()
    if span >= 10:
        measured = gaps[same].sum() / span
        lag = measured / period - 1.0
        report.Add(TIMING, 'Keeping up', WARN if lag > TOLERANCES['KeepingUp'] else PASS, lag,
                   f"Within steps, bunches were read every {measured * 1e3:.2f} ms against the "
                   f"triggers' {period * 1e3:.2f} ms ({lag:+.1%}).")
    else:
        report.Add(TIMING, 'Keeping up', NA, None,
                   ("One bunch per step" if per_step == 1 else "Too few steps hold two bunches "
                    "with hits") + ", so there is no run of bunches within a step to time the "
                   "reads by; the pauses below still account for the whole run.")

    # time the bunches and steps do not account for: a step may wait out its
    # settling and the next trigger, the reads may lag a little, and the
    # stamps are rounded; what is left is the loop, or the PC, standing idle
    model = record.get('SettleModel') if record else settle_model
    jumps = np.unique(np.round(np.abs(np.diff(np.r_[sequence, sequence[0]])), VOLTS_DECIMALS))
    longest = max([Settle_For(model, j) for j in jumps if j > 0] or [THRESHOLDS['SettleS']])
    idle = gaps - (bunches * period * (1.0 + TOLERANCES['KeepingUp'])
                   + crossings * (longest + period) + 2 * TIMESTAMP_RESOLUTION_S)
    stalls = int(np.count_nonzero((idle > TOLERANCES['StallS']) & (idle <= TOLERANCES['PauseS'])))
    paused = idle > TOLERANCES['PauseS']
    report.Add(TIMING, 'Read stalls', WARN if stalls else PASS, stalls,
               f"{stalls} times the loop stood idle for {TOLERANCES['StallS']:g} to "
               f"{TOLERANCES['PauseS']:g} s beyond what its bunches and steps account for.")
    halted = (int(np.count_nonzero(np.diff(monitor['time']) > TOLERANCES['PauseS']))
              if monitor is not None and len(monitor['time']) > 1 else 0)
    message = (f"{int(paused.sum())} pauses of more than {TOLERANCES['PauseS']:g} s in which no "
               f"bunch was taken, {idle[paused].sum():.0f} s in all")
    if paused.any():
        message += (": the window's own record stopped too, so the PC slept or hung, and any "
                    "drift across them went unrecorded." if halted else
                    ": the loop was held (the triggers stopped, or a device stopped answering).")
    else:
        message += "; the bunch count accounts for the whole run, however few its hits."
    report.Add(TIMING, 'Pauses', WARN if paused.any() else PASS, int(paused.sum()), message)
    del gaps, bunches, crossings, same, idle, paused

    # how far each read trailed the earliest read of its step, on the bunch clock
    latency = times - bunch * period
    step_index = np.maximum(step_of, 0).astype(np.intp)    # a bunch before the origin's
    first = np.full(int(step_index.max()) + 1, np.inf)
    np.minimum.at(first, step_index, latency)
    latency -= first[step_index]
    report.Series['Latency'] = _Distinct(times - t0, latency)
    del latency, first, step_of, step_index

    if record:
        message = ("Settle-aware: the tagger's window stayed shut while each step settled, so "
                   "no bunch was taken on a moving voltage")
        if monitor is not None and 'dropped' in monitor:
            dropped, taken = int(monitor['dropped'][-1]), int(monitor['bunches'][-1])
            message += (f"; {dropped} bunches were dropped while settling, "
                        f"{dropped / max(dropped + taken, 1):.1%} of the beam time")
        report.Add(TIMING, 'Bunches before settling', PASS, 0, message + '.')
    else:
        _Old_Settling(report, sequence, per_step, period, tags, settle_model)
    if monitor is not None and 'skipped' in monitor:
        parted = int(monitor['skipped'][-1])
        report.Add(TIMING, 'Tagger count', WARN if parted else PASS, parted,
                   f"{parted} triggers the tagger numbered never reached the acquisition loop, "
                   f"so no bunch was counted for them and none of their hits is in the run. "
                   f"Every hit that is still carries the setpoint it was read at.")
    return period, bunched


def _Old_Settling(report, sequence, per_step, period, tags, settle_model):
    """The old DAQ wrote each step as it read the step before's last bunch,
    without waiting: a bunch triggered after that write but before the jump
    had settled was taken on a moving voltage. The file holds no trigger
    times, so read delays are taken as nothing, which counts the fewest."""
    measured = bool(settle_model and settle_model.get('points'))
    into = np.abs(np.diff(sequence))
    # a sawtooth flies back to its start; a triangle turns on its last setpoint
    first = abs(sequence[0] - sequence[-1]) if tags['Kind'] == 'Sawtooth' else 0.0
    jumps = np.r_[first, into]
    settles = np.array([Settle_For(settle_model, j) if j > 0 else 0.0 for j in jumps])
    # the triggers come every period after the last bunch; those before the
    # step has settled count, the first at one period
    early = np.clip(np.ceil(settles / period - 1e-9) - 1, 0, per_step)
    per_scan = len(sequence) * per_step
    fraction = float(early.sum() / per_scan)
    how = ("the settle model" if measured else
           f"SETTLE, {THRESHOLDS['SettleS'] * 1e3:.0f} ms for every jump (no settle model is set)")
    where = (", most after the flyback" if early[0] > 0 and early[0] >= early[1:].max(initial=0)
             else "")
    report.Add(TIMING, 'Bunches before settling', _Grade(fraction, TOLERANCES[
        'ContaminatedFraction']), fraction,
        f"The old DAQ stepped as it read a step's last bunch, without waiting for the voltage. "
        f"With {how}, at least {int(early.sum())} of the {per_scan} bunches in each scan "
        f"({fraction:.1%}) were triggered before their step had settled{where}; read delays, "
        f"which the file does not hold, could only add to them.")
    if early.sum():
        report.Flag(f"At least {int(early.sum())} bunches a scan were taken before their step "
                    f"had settled; DENIS can drop them.")


def _Rates(report, bunch, tags, n, per_step):
    """Hits per bunch scan by scan, for the plot, from the hits in bunch order."""
    if not tags['Reliable'] or tags['Complete'] < 1:
        return
    per_scan = n * per_step
    bounds = np.arange(tags['Complete'] + 1) * per_scan - 0.5
    rate = np.diff(np.searchsorted(bunch, bounds)) / per_scan
    report.Series['Rates'] = (np.arange(len(rate)), rate, float(np.median(rate)))


# ── Beam and spectrum ───────────────────────────────────────────

def _Expectation(sequence, kind, per_step, axis, seen, last_bunch):
    """The hits the run's own spectrum leads one to expect, summed step by
    step from its start: each step's setpoint at the rate the whole run saw
    there. ``(bunches taken, hits expected by then)``, so a period expects
    what its own setpoints give, whatever part of a scan it holds."""
    step = axis[1] - axis[0]
    steps = np.arange(int(last_bunch // per_step) + 1)
    at = np.rint((sequence - axis[0]) / step).astype(np.intp)[
        _Step_Index(steps, len(sequence), kind)]
    visited = np.bincount(at, minlength=len(axis)) * float(per_step)
    rate = np.divide(seen.astype(float), visited, out=np.zeros(len(axis)), where=visited > 0)
    return np.arange(len(steps) + 1) * float(per_step), np.r_[0.0, np.cumsum(rate[at] * per_step)]


def _Beam(report, times, bunch, monitor, tags, expectation, scan_s, per_scan, t0):
    """The beam over the run, from the hits in time order, in periods long
    enough to expect BeamCounts hits each. A period expects what its own
    setpoints give at the run's rates there (``expectation``), so a stretch
    of background is no loss; where scans cannot be told apart, a period
    holds two scans at least, so its setpoints weigh alike. One below half
    its expectation, beyond what counting explains, is a loss; how much more
    the beam varies than counting allows is its stability. Bunches come from
    the loop's count, so a slow trigger is no loss either."""
    counted = monitor is not None and len(monitor.get('bunches', ())) > 1
    if counted and np.ptp(monitor['bunches']) > 0:
        clock_t, clock_b = monitor['time'], monitor['bunches']
    else:
        clock_t, clock_b = times, bunch
    clock_b = np.maximum.accumulate(np.asarray(clock_b, dtype=float))
    start, end = min(times[0], clock_t[0]), max(times[-1], clock_t[-1])
    duration = end - start
    shortest = 0.0 if expectation is not None else 2.0 * scan_s
    if duration <= 0 or (expectation is None and duration < 4.0 * scan_s):
        report.Add(BEAM, 'Beam losses', NA, None,
                   "Too few scans to tell a loss of beam from the spectrum's own shape.")
        return
    length = max(TOLERANCES['BeamPeriodS'], TOLERANCES['BeamCounts'] * duration / len(times),
                 shortest)
    count = min(int(duration // length), TOLERANCES['MaxBeamPeriods'])
    if count < 2:
        report.Add(BEAM, 'Beam losses', NA, None,
                   "The run is too short, for its count rate, to follow the beam over time.")
        return
    edges = np.linspace(start, end, count + 1)
    edges[-1] = np.nextafter(end, np.inf)
    hits = np.diff(np.searchsorted(times, edges)).astype(float)
    taken_at = np.interp(edges, clock_t, clock_b)
    taken = np.diff(taken_at)
    expected = (np.diff(np.interp(taken_at, *expectation)) if expectation is not None
                else taken.copy())
    usable = expected > 0
    if not usable.any() or not hits[usable].sum():
        report.Add(BEAM, 'Beam losses', NA, None,
                   "The run's bunch count does not advance, so its beam cannot be followed.")
        return
    lost = np.zeros(count, dtype=bool)
    for _ in range(2):          # the usual level, then again without the losses
        keep = usable & ~lost
        level = hits[keep].sum() / expected[keep].sum()
        lost = usable & np.array([_Short(h, level * e) for h, e in zip(hits, expected)])
    spans = []
    for k in np.flatnonzero(lost):
        if spans and spans[-1][1] == k:
            spans[-1][1] = k + 1
        else:
            spans.append([k, k + 1])
    per_bunch = hits[usable].sum() / max(taken[usable].sum(), 1.0)
    pitch = edges[1] - edges[0]
    if spans:
        minutes = sum(edges[b] - edges[a] for a, b in spans) / 60.0
        report.Add(BEAM, 'Beam losses', WARN, len(spans),
                   f"The beam fell below half its usual level, beyond what counting explains, "
                   f"{len(spans)} times, {minutes:.1f} min in all (judged over {count} periods "
                   f"of {pitch:.0f} s against what their setpoints give, {per_bunch:.3g} hits "
                   f"per bunch on average).")
        for a, b in spans:
            when = (f"Beam below half its usual level from {(edges[a] - t0) / 60:.1f} to "
                    f"{(edges[b] - t0) / 60:.1f} min into the run")
            if tags['Reliable']:
                first, last = (taken_at[[a, b]] // per_scan).astype(int)
                when += f" (scans {first} to {last})" if last > first else f" (scan {first})"
            report.Flag(when + '.')
    else:
        report.Add(BEAM, 'Beam losses', PASS, 0,
                   f"The beam held: no period of {pitch:.0f} s fell below half of what its "
                   f"setpoints give beyond what counting explains ({per_bunch:.3g} hits per "
                   f"bunch on average).")
    keep = usable & ~lost
    want = level * expected[keep]
    dof = int(keep.sum()) - 1
    if dof >= 1:
        chi2 = float(np.sum((hits[keep] - want) ** 2 / want))
        spread = np.sqrt(max(chi2 - dof, 0.0) / want.sum())
        varied = _Excess(chi2 / dof, dof) and spread > TOLERANCES['BeamSpread']
        report.Add(BEAM, 'Beam stability', WARN if varied else PASS, spread,
                   f"Between periods the beam varied {spread:.0%} rms beyond counting "
                   f"statistics" + (": peak heights taken at different times are not "
                                    "comparable." if varied else "."))
    report.Series['Beam'] = ((edges[:-1] + edges[1:]) / 2.0 - t0,
                             np.where(usable, hits / np.where(usable, level * expected, 1.0),
                                      np.nan), lost)


def _Line(axis, spectrum, scanned):
    """``(centre, width)`` in volts of the strongest line in ``spectrum``,
    between its half-height crossings, or None if no line stands
    LineSigmas standard errors out of the background or one runs off the
    scan. Smoothed over three bins first, so no single noisy bin is taken
    for the line; that widens a line a few bins wide a little, which a
    scale can bear."""
    if len(axis) < 3:
        return None
    smooth = np.convolve(spectrum, np.ones(3) / 3.0, mode='same')
    background = float(np.percentile(smooth[scanned], 25))
    top = int(np.argmax(np.where(scanned, smooth, -np.inf)))
    height = smooth[top] - background
    if height < TOLERANCES['LineSigmas'] * np.sqrt(max(background, 1.0) / 3.0):
        return None
    half = background + height / 2.0
    edges = []
    for direction in (-1, 1):
        i = top
        while 0 <= i + direction < len(axis) and smooth[i + direction] > half:
            i += direction
        j = i + direction
        if not 0 <= j < len(axis):
            return None             # the line runs off the scan
        across = (smooth[i] - half) / (smooth[i] - smooth[j])
        edges.append(axis[i] + direction * across * (axis[1] - axis[0]))
    return float(axis[top]), float(edges[1] - edges[0])


def _Smoothed(spectrum, sigma_bins):
    """``spectrum`` smoothed by a Gaussian ``sigma_bins`` wide, cut at three."""
    reach = max(int(np.ceil(3 * sigma_bins)), 1)
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) / sigma_bins) ** 2)
    return np.convolve(spectrum, kernel / kernel.sum(), mode='same')


def _Shift(counts, template, axis, reach, resolution):
    """``(shift, its standard error from the likelihood's curvature)`` that
    best lays ``template`` over ``counts``, a Poisson likelihood with the
    scale profiled out, searched coarsely over ``reach`` either way and then
    finely to ``resolution``; None if the best lies at the edge of the reach
    or the likelihood is too flat to say. ``template`` must be smooth: a
    jagged one has false minima as sharp as its noise."""
    floor = 1e-6 * template.max() + 1e-12
    hit = counts > 0
    total = counts.sum()

    def nll(grid):
        out = np.empty(len(grid))
        for lo in range(0, len(grid), FIT_ROWS):
            part = grid[lo:lo + FIT_ROWS]
            shifted = np.maximum(np.interp(axis[None, :] - part[:, None], axis, template), floor)
            out[lo:lo + FIT_ROWS] = (total * np.log(shifted.sum(axis=1))
                                     - np.log(shifted[:, hit]) @ counts[hit])
        return out

    coarse = np.linspace(-reach, reach, 41)
    best = int(np.argmin(nll(coarse)))
    if best in (0, len(coarse) - 1):
        return None
    pitch = coarse[1] - coarse[0]
    fine = np.arange(coarse[best] - pitch, coarse[best] + pitch + resolution / 2, resolution)
    values = nll(fine)
    best = int(np.argmin(values))
    if best in (0, len(fine) - 1):
        return None
    below, here, above = values[best - 1:best + 2]
    curve = below - 2.0 * here + above
    if curve <= 0:
        return None
    return fine[best] + resolution * (below - above) / (2.0 * curve), resolution / np.sqrt(curve)


def _Null_Spread(total, sizes, axis, reach, resolution, sigma_bins, draws=16):
    """How far the fitted shift of a period of the mean size scatters when
    time does not matter: the run's own counts dealt at random into periods
    of the same sizes (time's order taken away), each fitted as the real
    ones are. At a low count rate the likelihood's curvature understates
    the error, and this is the error the real periods are judged by. Dealt
    from a fixed seed, so a run always gets the same answer."""
    rng = np.random.default_rng(0)
    mean = float(np.mean(sizes))
    whole = total.astype(np.int64)
    found = []
    for _ in range(draws):
        left = whole.copy()
        for size in sizes:
            part = rng.multivariate_hypergeometric(left, int(size))
            left -= part
            fitted = _Shift(part.astype(float), _Smoothed(total - part, sigma_bins), axis, reach,
                            resolution)
            if fitted is not None:
                found.append(fitted[0] * np.sqrt(size / mean))
    return float(np.std(found)) if len(found) > 2 else None


def _Spectrum(report, bins, bunch, times, axis, scanned, scale, line, prediction, tags,
              per_scan):
    """The spectrum over the run in periods of whole scans (of scan pairs
    for a triangle, whose scans run opposite ways), so every setpoint
    weighs alike in each, each holding LineCounts over the background
    (ShapeCounts hits where no line stands out): where the line sat in
    each, fitted against the rest of the run and set against the monitors'
    prediction; and whether the spectrum's shape held."""
    unit = 2 if tags['Kind'] == 'Triangle' else 1
    units = tags['Complete'] // unit
    why = None
    if not tags['Reliable']:
        why = ("Scan numbers are unreliable for this run, so periods cannot be cut at whole "
               "scans, and cut anywhere else they weigh setpoints unequally.")
    elif units < 2:
        why = (f"Fewer than two complete {'scan pairs' if unit == 2 else 'scans'}, so the "
               f"spectrum cannot be compared over time with every setpoint weighed alike.")
    if why:
        report.Add(BEAM, 'Line position', NA, None, why)
        report.Add(BEAM, 'Spectrum shape', NA, None, why)
        return
    within = (bins >= 0) & (bunch >= 0) & (bunch < units * unit * per_scan)
    total = np.bincount(bins[within], minlength=len(axis)).astype(float)
    background = float(np.percentile(total[scanned], 25))
    signal = float(np.sum(np.clip(total[scanned] - background, 0, None)))
    count = min(TOLERANCES['MaxPeriods'], units,
                int(signal // TOLERANCES['LineCounts']) if line
                else int(within.sum() // TOLERANCES['ShapeCounts']))
    if count < 2:
        why = (f"The line holds {signal:.0f} counts over the background; following it takes "
               f"{2 * TOLERANCES['LineCounts']}." if line else
               "No line stands out of the background, and too few hits to compare periods.")
        report.Add(BEAM, 'Line position', NA, None, why)
        report.Add(BEAM, 'Spectrum shape', NA, None, why)
        return
    starts = [group[0] for group in np.array_split(np.arange(units), count)]
    edges = np.array(starts + [units]) * unit * per_scan
    label = np.searchsorted(edges, bunch, side='right') - 1
    label, kept_bins, kept_times = label[within], bins[within], times[within]
    del within
    counts = np.bincount(label * len(axis) + kept_bins, minlength=count * len(axis)).reshape(
        count, len(axis)).astype(float)
    first, last = np.full(count, np.inf), np.full(count, -np.inf)
    np.minimum.at(first, label, kept_times)
    np.maximum.at(last, label, kept_times)
    del label, kept_bins, kept_times
    if line:
        _Position(report, counts, list(zip(first, last)), axis, total, scale, prediction)
    else:
        report.Add(BEAM, 'Line position', NA, None,
                   "No line stands out of the background, so its position cannot be followed.")
    _Shape(report, counts)


def _Position(report, counts, ranges, axis, total, scale, prediction):
    step = axis[1] - axis[0]
    reach = min(max(2.0 * scale.Width, 3.0 * step), 0.5 * (axis[-1] - axis[0]))
    resolution = min(step, scale.Width) / 20.0
    # each period against the rest of the run, smoothed over a quarter of the
    # line's width; a symmetric smoothing does not move a shift's estimate
    sigma_bins = max(scale.Width / 4.0 / step, 1.0)
    sizes = counts.sum(axis=1)
    # fewer deals on a long scan: each fit costs its bins times the shifts tried
    draws = int(np.clip(NULL_BUDGET / (len(counts) * 82 * len(axis)), 4, 16))
    null = _Null_Spread(total, sizes, axis, reach, resolution, sigma_bins, draws)
    found, lost = [], 0
    for k, row in enumerate(counts):
        fitted = _Shift(row, _Smoothed(total - row, sigma_bins), axis, reach, resolution)
        if fitted is None:
            lost += 1
            continue
        shift, error = fitted
        if null:
            error = max(error, null * np.sqrt(np.mean(sizes) / sizes[k]))
        found.append((np.mean(ranges[k]), shift, error, k))
    if len(found) < 2:
        report.Add(BEAM, 'Line position', NA, None,
                   "The line could not be placed in enough periods to follow it.")
        return
    t, shifts, errors, kept = (np.array(column) for column in zip(*found))
    rise, trend, error = _Rise(t, shifts, errors)
    residuals = shifts - trend
    dof = len(shifts) - 2
    chi2 = float(np.sum((residuals / errors) ** 2))
    wander = float(np.sqrt(max(np.mean(residuals ** 2) - np.mean(errors ** 2), 0.0)))
    moved = abs(rise) > TOLERANCES['Significance'] * error
    wandered = _Excess(chi2 / dof, dof) if dof >= 1 else False
    verdict = _Worst(scale.Grade(rise) if moved else PASS,
                     scale.Grade(wander) if wandered else PASS, WARN if lost else PASS)
    message = (f"Fitted against the rest of the run in {len(shifts)} periods of whole scans, "
               f"the line moved "
               f"{rise:+.3f} ± {error:.3f} V over it ({scale.Text(rise)})"
               + ("" if moved else ", within its error"))
    message += (f"; between periods it wandered {scale.Text(wander)} rms beyond counting."
                if wandered else "; between periods it held still within counting.")
    if lost:
        message += (f" In {lost} periods it could not be placed: it moved more than "
                    f"{reach:.3f} V, or too few counts.")
    report.Add(BEAM, 'Line position', verdict, rise, message)
    if verdict != PASS:
        report.Flag(f"The line moved {scale.Text(rise)} over the run.")
    report.Budget['Moved'] = (rise, error)
    if prediction is not None:
        _Closure(report, t, shifts, errors, [ranges[k] for k in kept.astype(int)], prediction,
                 scale)


def _Prediction(hv, light, times):
    """Where the monitors say the line went, in scan volts, on the monitor's
    samples, with the hits each sample stands for: the HV's error against its
    calibration and the laser's against DENIS's setpoint, summed, each filled
    across its own gaps. None if neither was recorded."""
    sources = [s for s in (hv, light) if s is not None]
    if not sources:
        return None
    sample_t = sources[0][0]
    moved = np.zeros(len(sample_t))
    for t, volts in sources:
        ok = np.isfinite(volts)
        if ok.sum() < 2:
            continue
        moved += np.interp(sample_t, t[ok], volts[ok])
    names = ' and '.join(name for name, s in (('the HV against its calibration', hv),
                                               ("the laser against DENIS's setpoint", light))
                         if s is not None)
    return sample_t, moved, _Hits_Per_Sample(sample_t, times), names


def _Closure(report, t, shifts, errors, ranges, prediction, scale):
    """The line's movement against the monitors' prediction of it: what they
    leave unexplained moved something nobody was watching."""
    sample_t, moved, weights, names = prediction
    predicted = []
    for lo, hi in ranges:
        inside = (sample_t >= lo) & (sample_t <= hi)
        stats = _Weighted(moved[inside], weights[inside]) if inside.any() else None
        predicted.append(stats[0] if stats else np.nan)
    predicted = np.array(predicted)
    ok = np.isfinite(predicted)
    if ok.sum() < 2:
        return
    t, shifts, errors, predicted = t[ok], shifts[ok], errors[ok], predicted[ok]
    left, _, error = _Rise(t, shifts - predicted, errors)
    expected = _Rise(t, predicted, errors)[0]
    unexplained = abs(left) > TOLERANCES['Significance'] * error and scale.Matters(left)
    report.Add(SOURCES, 'What moved the line', WARN if unexplained else PASS, left,
               f"The monitors ({names}) predict the line moved {expected:+.3f} V over the run; "
               f"after them, {left:+.3f} ± {error:.3f} V of its movement is left "
               f"({scale.Text(left)})"
               + (": something not monitored moved it, such as the beam energy, or the "
                  "wavemeter itself if the laser is locked to it." if unexplained else
                  ", within what the monitors and counting explain."))
    if unexplained:
        report.Flag(f"{scale.Text(left)} of the line's movement is unexplained by the monitors.")


def _Shape(report, counts):
    """Whether the periods' spectra share one shape, on bins merged until each
    expects ShapeCellCounts in every period, so the chi2 means something at
    any count rate; its size is the rms departure beyond counting."""
    periods = len(counts)
    total = counts.sum(axis=0)
    grand = total.sum()
    need = TOLERANCES['ShapeCellCounts'] * periods
    group, cell, held = np.empty(len(total), dtype=np.intp), 0, 0.0
    for j, value in enumerate(total):
        group[j] = cell
        held += value
        if held >= need:
            cell, held = cell + 1, 0.0
    if 0 < held < need and cell:
        group[group == cell] = cell - 1         # the remainder joins its neighbour
    cells = int(group.max()) + 1
    if cells < 2:
        report.Add(BEAM, 'Spectrum shape', NA, None,
                   "Too few counts to compare the periods' spectra.")
        return
    merged = np.array([np.bincount(group, weights=row, minlength=cells) for row in counts])
    expected = np.outer(merged.sum(axis=1), merged.sum(axis=0)) / grand
    chi2 = float(np.sum((merged - expected) ** 2 / expected))
    dof = (periods - 1) * (cells - 1)
    spread = float(np.sqrt(max(chi2 - dof, 0.0) / grand))
    changed = _Excess(chi2 / dof, dof) and spread > TOLERANCES['ShapeSpread']
    report.Add(BEAM, 'Spectrum shape', WARN if changed else PASS, spread,
               f"Over {periods} periods the spectrum kept its shape to {spread:.0%} rms beyond "
               f"counting statistics, on {cells} bins of at least {need:.0f} counts"
               + (": peak heights changed against one another (laser power, the beam's make-"
                  "up, or the line moving)." if changed else "."))


def _Channels(report, channels, times, wired, t0):
    """Each wired channel, from the hits in time order: one that recorded
    nothing, or whose counts fell away during the run, is a detector, cable
    or threshold fault until shown otherwise."""
    channel = np.clip(channels, 0, 63).astype(np.intp)
    counts = np.bincount(channel, minlength=64)
    wired = [int(c) for c in wired]
    silent = [c for c in wired if not counts[c]]
    active = [c for c in range(64) if counts[c]]
    shares = counts / counts.sum()
    problems = [f"Channel {c} recorded no hits: its detector, cable or discriminator "
                f"threshold may be faulty, or it is not connected." for c in silent]
    periods = min(TOLERANCES['MaxPeriods'] * 2,
                  int(min(counts[active]) // TOLERANCES['ChannelCounts']))
    if periods >= 2:
        bounds = np.linspace(0, len(channel), periods + 1).astype(int)
        for c in active:
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                seen = np.count_nonzero(channel[lo:hi] == c)
                expected = shares[c] * (hi - lo)
                if (seen < TOLERANCES['ChannelFall'] * expected
                        and _At_Most(seen, expected) < TOLERANCES['Improbable']):
                    problems.append(f"Channel {c}'s counts fell to {seen / expected:.0%} of its "
                                    f"usual share {(times[lo] - t0) / 60:.1f} min into the run: "
                                    f"its detector or cable may have failed during it.")
                    break
    listed = ', '.join(f"ch {c}: {shares[c]:.1%}" for c in active)
    if problems:
        report.Add(BEAM, 'Channels', WARN, len(problems), ' '.join(problems) + f" ({listed}.)")
        for problem in problems:
            report.Flag(problem)
    else:
        report.Add(BEAM, 'Channels', PASS, 0,
                   f"Every wired channel counted throughout: {listed}.")


def _Tof(report, tof, times, record, bunched):
    """The time-of-flight peak against the gate the run recorded, period by
    period: a peak drifting against its gate loses counts unevenly over the
    run. Without a gate, the peak's own drift. In DC mode there is no peak."""
    if bunched is False:
        report.Add(TIMING, 'Time of flight', NA, None,
                   "DC beam: there is no time-of-flight peak to follow.")
        return
    low, high = float(tof.min()), float(tof.max())
    width = max((high - low) / 1e5, 0.1)
    # over five bins, so no single noisy bin is taken for the peak or its edge
    histogram = np.convolve(np.bincount(((tof - low) / width).astype(np.intp)),
                            np.ones(5) / 5.0, mode='same')
    background = float(np.median(histogram))
    top = int(np.argmax(histogram))
    height = histogram[top] - background
    if height < TOLERANCES['LineSigmas'] * np.sqrt(max(background, 1.0) / 5.0):
        report.Add(TIMING, 'Time of flight', NA, None,
                   "No time-of-flight peak stands out of the background.")
        return
    half = background + height / 2.0
    left, right = top, top
    while left > 0 and histogram[left - 1] > half:
        left -= 1
    while right < len(histogram) - 1 and histogram[right + 1] > half:
        right += 1
    spread = (right - left + 1) * width
    centre = low + (top + 0.5) * width
    near = np.abs(tof - centre) <= spread * 1.5
    inside_peak = np.flatnonzero(near)
    periods = min(TOLERANCES['MaxPeriods'], len(inside_peak) // TOLERANCES['TofCounts'])
    if periods < 2:
        report.Add(TIMING, 'Time of flight', NA, None, "Too few hits in the TOF peak.")
        return
    groups = np.array_split(inside_peak, periods)
    gate = (record.get('View') or {}).get('tof_gate_us') if record else None
    if gate:
        start, stop = float(min(gate)), float(max(gate))
        kept = [np.count_nonzero((tof[g] >= start) & (tof[g] <= stop)) for g in groups]
        sizes = np.array([len(g) for g in groups], dtype=float)
        share = np.array(kept) / sizes
        overall = float(np.sum(kept) / sizes.sum())
        moved = False
        if 0 < overall < 1:
            chi2 = float(np.sum((np.array(kept) - sizes * overall) ** 2
                                / (sizes * overall * (1 - overall))))
            moved = (_Excess(chi2 / (periods - 1), periods - 1)
                     and np.ptp(share) > TOLERANCES['TofShareChange'])
        few = overall < TOLERANCES['TofGateShare']
        message = (f"The TOF gate, {start:.1f} to {stop:.1f} µs, keeps {overall:.0%} of the peak "
                   f"near {centre:.1f} µs")
        message += (f"; over the run its share went from {share[0]:.0%} to {share[-1]:.0%}: the "
                    f"peak moved against the gate (the buncher's timing, or the trigger delay), "
                    f"so periods are counted unequally." if moved else ", steady over the run.")
        if few:
            message += " Most of the peak falls outside it."
        report.Add(TIMING, 'Time of flight', WARN if moved or few else PASS, overall, message)
        if moved:
            report.Flag("The TOF peak moved against the gate during the run.")
        return
    middles = np.array([np.median(times[g]) for g in groups])
    medians = np.array([np.median(tof[g]) for g in groups])
    # the standard error of a median
    errors = np.array([np.sqrt(np.pi / 2) * np.std(tof[g]) / np.sqrt(len(g)) for g in groups])
    drift, _, error = _Rise(middles, medians, np.maximum(errors, 1e-6))
    moved = abs(drift) > TOLERANCES['TofDriftUs'] and abs(drift) > 3 * error
    report.Add(TIMING, 'Time of flight', WARN if moved else PASS, drift,
               f"No TOF gate was recorded with the run; the peak near {centre:.1f} µs moved "
               f"{drift:+.2f} ± {error:.2f} µs over it.")


# ── The sources ─────────────────────────────────────────────────

def _Readback(report, monitor, header, scale, times, t0):
    """The HV during the scan against the calibration's own reading at each
    setpoint, so the converter's non-linearity and code rounding, which the
    calibration holds, are not taken for drift. Weighted by the hits taken
    meanwhile, and the meter's own scatter taken out. Returns ``(sample
    times, where it moved the line in scan volts)``, or None."""
    cal_set, cal_read = _Calibration_Points(header)
    if monitor is None or len(cal_set) < 3:
        report.Add(SOURCES, 'HV against its calibration', NA, None,
                   "The run has no monitor record or no calibration, so the HV during the scan "
                   "cannot be judged.")
        return None
    order = np.argsort(cal_set)
    cal_set, cal_read = cal_set[order], cal_read[order]
    t, setpoint, readback = monitor['time'], monitor['setpoint_v'], monitor['readback_v']
    # a sample whose setpoint was already on at the sample before: on for a
    # tick at least, and the meter's poll is fresher than that
    steady = (np.r_[False, setpoint[1:] == setpoint[:-1]] & np.isfinite(readback)
              & (setpoint >= cal_set[0]) & (setpoint <= cal_set[-1]))
    if steady.sum() < 3:
        report.Add(SOURCES, 'HV against its calibration', NA, None,
                   "Too few readings on a settled setpoint: the steps were shorter than the "
                   "monitor's 0.1 s sample.")
        return None
    since = np.full(len(t), np.nan)
    since[steady] = readback[steady] - np.interp(setpoint[steady], cal_set, cal_read)
    mean, spread = _Weighted(since, _Hits_Per_Sample(t, times))
    pairs = steady[1:] & steady[:-1] & (setpoint[1:] == setpoint[:-1])
    noise = float(np.std(np.diff(since)[pairs]) / np.sqrt(2)) if pairs.sum() >= 3 else 0.0
    wander = float(np.sqrt(max(spread ** 2 - noise ** 2, 0.0)))
    worst = max(abs(mean), wander)
    report.Add(SOURCES, 'HV against its calibration', scale.Grade(worst), worst,
               f"Against the calibration's own reading at each setpoint, the HV sat "
               f"{mean:+.3f} V off and wandered {wander:.3f} V rms beyond the meter's own "
               f"scatter ({noise:.3f} V), weighted by the hits taken meanwhile. DENIS takes the "
               f"calibration as it was, so the line sits {scale.Text(mean)} off and is "
               f"broadened by {scale.Text(wander)}.")
    report.Series['Readback'] = (t[steady] - t0, since[steady])
    report.Budget['HV'] = worst
    return t, -since                # the line moves against the HV's error


def _Laser(report, monitor, header, laser, recorded, scale, times, axis, t0):
    """The wavemeter during the run against the setpoint DENIS takes for the
    whole of it, weighted by the hits taken meanwhile and put in scan volts
    through the Doppler shift. Returns ``(sample times, where it moved the
    line in scan volts)``, or None."""
    if monitor is None or not np.isfinite(monitor.get('wavenumber_cm1', [np.nan])).any():
        report.Add(SOURCES, 'Laser', NA, None,
                   "The wavemeter was not recorded during this run, so only its value at the "
                   "start is known.")
        return None
    t, wavenumber = monitor['time'], monitor['wavenumber_cm1']
    beam_v, mass = _Beam_V(header, axis), float(header.get('MassAMU', np.nan))
    if not (beam_v > 0 and mass > 0):
        report.Add(SOURCES, 'Laser', NA, None,
                   "The run's mass or cooler voltage is unknown, so the laser cannot be put in "
                   "scan volts.")
        return None
    setpoint = float(header.get('LaserSetpoint', np.nan))
    reference = setpoint if np.isfinite(setpoint) else float(np.nanmean(wavenumber))
    per_hz = Volts_Per_Hz(beam_v, np.nanmean(wavenumber) * C_LIGHT * 100.0, mass, laser)
    off_hz = (wavenumber - reference) * C_LIGHT * 100.0
    volts = off_hz * per_hz
    mean, spread = _Weighted(volts, _Hits_Per_Sample(t, times))
    worst = max(abs(mean), spread)
    source = f"harmonic {laser['harmonic']}, {laser['geometry']}" + (
        "" if recorded else "; today's settings, as the run does not record its own")
    report.Add(SOURCES, 'Laser', scale.Grade(worst), worst,
               f"DENIS takes the laser at {reference:.5f} cm⁻¹ for the whole run; the "
               f"wavemeter read {mean / per_hz / 1e6:+.1f} MHz from it on average and wandered "
               f"{spread / abs(per_hz) / 1e6:.1f} MHz rms, weighted by the hits taken meanwhile. "
               f"That moves the line {scale.Text(mean)} and broadens it by "
               f"{scale.Text(spread)} ({source}).")
    report.Series['Laser'] = (t - t0, off_hz / 1e6)
    report.Budget['Laser'] = worst
    return t, volts


def _Cooler(report, raw, monitor, t0, cooler_divider):
    """DENIS corrects every hit by its own cooler reading, so the cooler's
    changes move no line; what matters is that every hit has a reading."""
    cooler = raw[:, 5]
    missing = int(np.count_nonzero(~np.isfinite(cooler) | (cooler == 0)))
    t, volts = ((monitor['time'], monitor['cooler_v']) if monitor is not None
                else (raw[:, 0], cooler * cooler_divider))
    ok = np.isfinite(volts) & (volts != 0)
    if ok.sum() < 3:
        report.Add(SOURCES, 'Cooler', WARN if missing else NA, missing,
                   "Too few cooler readings to say how it behaved"
                   + (f"; {missing} hits carry none, so DENIS cannot place them." if missing
                      else "."))
        return
    message = (f"DENIS corrects each hit by its own cooler reading, so the cooler's changes do "
               f"not move the line; it varied {np.std(volts[ok]):.3f} V rms over the run.")
    if missing:
        message += (f" {missing} hits carry no reading, so DENIS cannot place them: the cooler "
                    f"meter stopped answering.")
        report.Flag(f"{missing} hits have no cooler reading; DENIS cannot place them.")
    report.Add(SOURCES, 'Cooler', WARN if missing else PASS, missing, message)
    report.Series['Cooler'] = _Distinct(t[ok] - t0, volts[ok] - np.mean(volts[ok]))


def Export(path, result):
    """The report beside the run as ``run_NNNN_performance.yaml``: the
    verdict, the summary, every check and what was flagged. Nothing in it
    cuts data; the operator decides that in DENIS. Returns the path written."""
    out = path[:-len('.asdf')] + '_performance.yaml'
    record = {
        'run': os.path.basename(path), 'analysed': f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        'verdict': result['Verdict'], 'summary': result['Summary'],
        'budget': result.get('Budget', ''),
        'checks': {section: [{'check': c['Check'], 'verdict': c['Verdict'],
                              'message': c['Message']} for c in checks]
                   for section, checks in result['Sections'].items()},
        'flagged': list(result.get('Flagged', [])),
    }
    with open(out, 'w', encoding='utf-8') as fh:
        yaml.safe_dump(record, fh, sort_keys=False, allow_unicode=True, width=100)
    return out
