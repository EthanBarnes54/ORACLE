"""Scan performance: how far the rack's behaviour during a run shaped its data.

Date:    2026-10-29
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A spectrum is counts against the scanning voltage, and every part of the
pipeline can move a count to the wrong voltage or the wrong amount:
the DAC and its amplifier, the loop's timing, the cooler, the laser, the
beam. This reads a finished run (its file, and its monitor record if it
has one) and asks of each part whether it stayed inside tolerance. Each
check returns PASS, WARN or FAIL with the number it decided on, or N/A
when the run does not hold what it needs.

Tolerances are set in scan volts against the run's own step size, the
one scale on which every effect can be compared: a drift of half a step
over the run smears a peak by half a bin, whatever caused it. The cooler
shifts the beam energy volt for volt; the laser is converted through
the Doppler shift at the run's mass and beam energy.

The analysis recommends cuts: scans to exclude, with the reason, and
the bunches to drop at the start of every step. Scans are numbered as
DENIS numbers them (bunch // bunches per scan), so the exclusions can be
handed to DENIS's scan filter as they are.

Scans are found from the bunch numbers, not from the voltages: the
setpoint each bunch should have been taken at follows from its number,
the plan and the bunches per step, so a hit tagged with any other
voltage was mis-tagged (the collinear DAQ's D5). Which scan type, and
whether the tagger counts bunches from 0 or 1, is taken from whichever
fits the run best.

Depends on: numpy, PyYAML; cls_acquisition.calibration, .scan_plan and
.hardware.keysight.
"""

import os
from datetime import datetime

import numpy as np
import yaml

from cls_acquisition.calibration import DIVIDER, PASS, THRESHOLDS, WARN, Line_Fit, Step_At_Zero
from cls_acquisition.hardware.keysight import COOLER_DIVIDER
from cls_acquisition.scan_plan import Scan_Plan, Scan_Plan_Error

FAIL, NA = 'FAIL', 'N/A'
RANK = {NA: -1, PASS: 0, WARN: 1, FAIL: 2}
VOLTAGE, TIMING, BEAM, SOURCES = 'Voltage scan', 'Timing', 'Beam and spectrum', 'Cooler and laser'

# Every number the analysis decides on. A pair is (WARN above, FAIL above).
TOLERANCES = {
    'MistaggedFraction': (0.0, 0.001),      # hits not at the voltage their bunch was due
    'DriftSteps': (0.5, 1.0),               # a drift over the run, in steps
    'SpreadSteps': (0.25, 0.5),             # a spread (1 sigma) about the drift, in steps
    'CalibrationRmsSteps': (0.25, 0.5),     # the calibration's residual, in steps
    'ContaminatedFraction': (0.01, 0.05),   # bunches taken before their step had settled
    'ExcessZ': 2.326,                       # chi2 beyond counting: one-sided 1 %
    'BeamLossFraction': 0.5,                # a scan below this share of the median rate
    'CentroidSigmas': 3.0,                  # a scan's peak this far off the trend
    'MinPeakCounts': 25,                    # fewer above background give no centroid
    'KeepingUp': 0.02,                      # reads slower than the bunches by this share
    'StallS': 0.5,                          # a read this much later than the one before
    'PauseS': 5.0,                          # no read for this long, settling included
    'TofDriftUs': 0.5,
    'DeadChannelShare': 0.01,
}
DC_PERIOD_S = 0.001                         # the DC trigger, against the bunched dwell
TIMESTAMP_RESOLUTION_S = 0.01               # the run file's timestamps are rounded to this
VOLTS_DECIMALS = 6                          # setpoints compare to the microvolt

C_LIGHT = 299792458.0
AMU_EV = 931.49410242e6                     # one atomic mass unit, in eV


class Performance_Error(ValueError):
    pass


def _Excess(chi2_per_dof, dof):
    """True if a chi2/ndf this large is beyond counting statistics at the
    one-sided 1 % level (Wilson-Hilferty), so a short run does not cry wolf."""
    if dof < 1:
        return False
    shrink = 2.0 / (9.0 * dof)
    z = (chi2_per_dof ** (1.0 / 3.0) - (1.0 - shrink)) / np.sqrt(shrink)
    return bool(z > TOLERANCES['ExcessZ'])


def _Grade(value, limits):
    warn, fail = limits
    return FAIL if value > fail else WARN if value > warn else PASS


def _Trend(t, values, errors=None):
    """The straight line through ``values`` against ``t``, weighted by
    ``errors`` if given: ``(its rise over the span of t, the line, the
    rise's error)``. The error means something only when ``errors`` do."""
    t, values = np.asarray(t, dtype=float), np.asarray(values, dtype=float)
    if len(t) < 2 or not np.ptp(t):
        return 0.0, np.full_like(values, values.mean()), 0.0
    weights = None if errors is None else 1.0 / np.maximum(errors, 1e-9)
    (slope, intercept), covariance = np.polyfit(t, values, 1, w=weights, cov='unscaled')
    span = np.ptp(t)
    return abs(slope) * span, slope * t + intercept, float(np.sqrt(covariance[0, 0])) * span


def Ion_Frequency(beam_v, nu_laser_hz, mass_amu, laser):
    """The frequency the ion sees, as DENIS computes it: the laser times its
    harmonic, Doppler-shifted at the beam's speed."""
    gamma = 1.0 + laser['charge'] * np.asarray(beam_v, dtype=float) / (mass_amu * AMU_EV)
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    shift = (1.0 + beta) / (1.0 - beta)
    doppler = np.sqrt(shift if 'anti' in laser['geometry'] else 1.0 / shift)
    return laser['harmonic'] * nu_laser_hz * doppler


def Volts_Per_Hz(beam_v, nu_laser_hz, mass_amu, laser):
    """How many scan volts move the resonance as far as one hertz at the laser."""
    # the ion's frequency is in proportion to the laser's, so one hertz at the
    # laser moves it by its value at 1 Hz; one volt moves it by its slope
    per_volt = abs(Ion_Frequency(beam_v + 1.0, nu_laser_hz, mass_amu, laser)
                   - Ion_Frequency(beam_v - 1.0, nu_laser_hz, mass_amu, laser)) / 2.0
    return Ion_Frequency(beam_v, 1.0, mass_amu, laser) / per_volt


class _Report():

    def __init__(self):
        self.Sections, self.Series, self.Cuts = {}, {}, {}

    def Add(self, section, check, verdict, value, message):
        self.Sections.setdefault(section, []).append(
            {'Check': check, 'Verdict': verdict, 'Value': value, 'Message': message})

    def Cut(self, scans, reason):
        for scan in np.atleast_1d(scans):
            self.Cuts.setdefault(int(scan), []).append(reason)

    def Result(self, summary, leading):
        checks = [c for section in self.Sections.values() for c in section]
        verdict = max((c['Verdict'] for c in checks), key=RANK.get, default=NA)
        return {'Verdict': verdict, 'Summary': summary, 'Sections': self.Sections,
                'Series': self.Series, 'Cuts': dict(sorted(self.Cuts.items())),
                'Leading_Bunches': leading}


def Analyse(header, raw, monitor, laser, settle_s=THRESHOLDS['SettleS']):
    """The report on one run: ``Verdict``, ``Summary``, ``Sections`` (name to
    checks), ``Series`` for the plots and ``Cuts`` (scan to reasons), plus
    ``Leading_Bunches``, the bunches to drop at the start of every step.
    ``laser`` is the experiment's {'harmonic', 'geometry', 'charge'}."""
    if not len(raw):
        raise Performance_Error("The run holds no events, so there is nothing to analyse.")
    try:
        plan = Scan_Plan(*zip(*header['ScanningRanges']), header['StepSize']).Setpoints
    except (Scan_Plan_Error, KeyError, TypeError, ValueError) as e:
        raise Performance_Error(f"The run's scanning ranges do not make a plan: {e}") from None
    step = float(header['StepSize'])
    per_step = max(int(header['BunchesPerChannel']), 1)
    times, bunch = raw[:, 0], raw[:, 2].astype(np.int64)
    t0 = times.min()
    report = _Report()

    scan = _Voltage_Tags(report, plan, per_step, raw[:, 1], bunch, monitor)
    line = _Calibration(report, header, step)
    period, leading = _Timing(report, monitor, header, times, bunch - scan['Origin'], per_step,
                              settle_s, t0)
    _Readback(report, monitor, line, step, t0)
    _Beam_And_Spectrum(report, plan, per_step, raw, scan, step, t0)
    _Cooler(report, raw, monitor, step, t0)
    _Laser(report, monitor, header, laser, plan, step, t0)
    _Channels(report, raw)

    summary = (f"{scan['Complete']} complete {scan['Type'].lower()} scans of {len(plan)} "
               f"setpoints, {per_step} bunches per step, {len(raw)} hits over "
               f"{(times.max() - t0) / 60:.1f} min; bunches every {period * 1e3:.1f} ms.")
    return report.Result(summary, leading)


# ── The voltage axis ────────────────────────────────────────────

def _Voltage_Tags(report, plan, per_step, volts, bunch, monitor):
    """Which setpoint each hit should carry, from its bunch; how many do not."""
    n = len(plan)
    rounded = np.round(volts, VOLTS_DECIMALS)
    best = None
    for origin in (0, 1):
        visit = (bunch - origin) // per_step
        position, scan = visit % n, visit // n
        for kind in ('Sawtooth', 'Triangle'):
            # a triangle runs every other scan backwards
            reverse = (kind == 'Triangle') & (scan % 2 == 1)
            index = np.where(reverse, n - 1 - position, position)
            agree = rounded == np.round(plan[index], VOLTS_DECIMALS)
            if best is None or agree.mean() > best['Agree'].mean():
                best = {'Origin': origin, 'Type': kind, 'Index': index, 'Scan': scan,
                        'Agree': agree}
    if best['Scan'].max() == 0:
        best['Type'] = 'Single'
    mistagged = 1.0 - best['Agree'].mean()
    report.Add(VOLTAGE, 'Tags against bunches',
               _Grade(mistagged, TOLERANCES['MistaggedFraction']), mistagged,
               f"{np.count_nonzero(~best['Agree'])} of {len(volts)} hits carry a voltage other "
               f"than the setpoint their bunch was due at ({mistagged:.3%})."
               + (" The bunch count and the voltage steps parted: a read skipped or timed "
                  "out (the collinear DAQ's D5)." if mistagged else ""))
    if mistagged:
        report.Cut(np.unique(best['Scan'][~best['Agree']]), 'hits at the wrong voltage')
    off_plan = int(np.count_nonzero(~np.isin(rounded, np.round(plan, VOLTS_DECIMALS))))
    report.Add(VOLTAGE, 'Setpoints in the plan', FAIL if off_plan else PASS, off_plan,
               f"{off_plan} hits at a voltage the plan never held.")
    # the file holds only bunches with hits; the loop's own count says how
    # many it took, so a run ending on empty bunches is not taken as cut short
    counted = monitor is not None and 'bunches' in monitor
    total = int(monitor['bunches'][-1]) if counted else int(bunch.max() - best['Origin'] + 1)
    best['Complete'] = total // (n * per_step)
    if total % (n * per_step):
        report.Cut(best['Complete'], 'incomplete: the scan was stopped part way' if counted
                   else 'incomplete, or it ended on bunches without hits')
    return best


def _Calibration(report, header, step):
    """The start-of-run calibration's line, ``(gain, offset)``, or None."""
    cal_set = np.asarray(header.get('CalSet', []), dtype=float)
    readback = np.asarray(header.get('CalReadback', []), dtype=float) * DIVIDER
    if len(cal_set) < 3:
        report.Add(VOLTAGE, 'Calibration', NA, None, "The run holds no calibration.")
        return None
    tracking = abs(np.corrcoef(cal_set, readback)[0, 1]) if np.ptp(readback) else 0.0
    gain, offset, rms = Line_Fit(cal_set, readback)
    verdict = (FAIL if tracking < THRESHOLDS['MinTrackingR']
               else _Grade(rms / step, TOLERANCES['CalibrationRmsSteps']))
    report.Add(VOLTAGE, 'Calibration', verdict, rms,
               f"Readback tracks the command at |r| {tracking:.4f}; gain {gain:.5f}, offset "
               f"{offset:.3f} V, residual {rms:.3f} V rms ({rms / step:.2f} steps).")
    seam = Step_At_Zero(cal_set, readback)
    if seam is not None:
        report.Add(VOLTAGE, 'Seam at 0 V',
                   WARN if abs(seam) > THRESHOLDS['MaxPolarityStepV'] else PASS, seam,
                   f"The converter branches' lines meet 0 V {seam:+.3f} V apart; DAC correction "
                   f"{'applied' if header.get('DacCorrectionApplied') else 'not applied'}.")
    return gain, offset


def _Readback(report, monitor, line, step, t0):
    """The platform as the line meter saw it during the scan, against what
    the start-of-run calibration's ``line`` says it should have read."""
    if monitor is None or line is None:
        report.Add(VOLTAGE, 'Voltage during the scan', NA, None,
                   "The run has no monitor record or no calibration, so the voltage during "
                   "the scan cannot be judged.")
        return
    setpoint, readback = monitor['setpoint_v'], monitor['readback_v']
    # a sample whose setpoint was already on at the sample before: on for a
    # tick at least, and the meter's poll is fresher than that
    steady = np.r_[False, setpoint[1:] == setpoint[:-1]] & np.isfinite(readback)
    if steady.sum() < 3:
        report.Add(VOLTAGE, 'Voltage during the scan', NA, None,
                   "Too few readings on a settled setpoint: the steps were shorter than the "
                   "monitor's 0.1 s sample.")
        return
    gain, offset = line
    since = readback[steady] - (gain * setpoint[steady] + offset)
    t = monitor['time'][steady] - t0
    drift, shift = _Trend(t, since)[0], abs(float(np.median(since)))
    worst = max(drift, shift)
    report.Add(VOLTAGE, 'Voltage since the calibration',
               _Grade(worst / step, TOLERANCES['DriftSteps']), worst,
               f"The readback sat {np.median(since):+.3f} V from the calibration's line and "
               f"drifted {drift:.3f} V over the run ({worst / step:.2f} steps at worst), over "
               f"{steady.sum()} steady readings.")
    report.Series['Readback'] = (t, since)


# ── Timing ──────────────────────────────────────────────────────

def _Timing(report, monitor, header, times, bunch, per_step, settle_s, t0):
    """Whether the loop kept up with the bunches, and whether any bunch was
    taken before its step had settled: ``(the bunch period, the bunches to
    drop at the start of every step)``. Within a step the bunches come one
    per period; between steps a settle-aware loop pauses on purpose."""
    order = np.lexsort((times, bunch))
    b, t = bunch[order], times[order]
    step_of = b // per_step
    same = np.diff(step_of) == 0
    span = np.diff(b)[same].sum()
    measured = np.diff(t)[same].sum() / span if span else np.nan
    # the header keeps the bunched dwell even for a DC run, so the reads say
    # which period was in force
    periods = (float(header.get('DwellTime', 0)) / 1e3 or DC_PERIOD_S, DC_PERIOD_S)
    nominal = (min(periods, key=lambda p: abs(np.log(p / measured))) if measured > 0
               else periods[0])
    lag_share = measured / nominal - 1.0 if np.isfinite(measured) else 0.0
    report.Add(TIMING, 'Keeping up', WARN if lag_share > TOLERANCES['KeepingUp'] else PASS,
               lag_share, f"Within steps, bunches were read every {measured * 1e3:.2f} ms "
                          f"against {nominal * 1e3:.2f} ms nominal ({lag_share:+.1%}).")
    # how far each read trailed the earliest read of its step, on the bunch clock
    latency = t - b * nominal
    first = np.full(step_of.max() + 1, np.inf)
    np.minimum.at(first, step_of, latency)
    latency -= first[step_of]
    report.Series['Latency'] = (t - t0, latency)
    stalls = int(np.count_nonzero(np.diff(latency)[same] > TOLERANCES['StallS']))
    report.Add(TIMING, 'Read stalls', WARN if stalls else PASS, stalls,
               f"{stalls} reads came more than {TOLERANCES['StallS']:g} s later than the one "
               f"before within a step.")
    gaps = np.diff(t)
    paused = gaps > TOLERANCES['PauseS']
    report.Add(TIMING, 'Pauses', WARN if paused.any() else PASS, int(paused.sum()),
               f"{int(paused.sum())} pauses of more than {TOLERANCES['PauseS']:g} s with no "
               f"read at all, {gaps[paused].sum():.0f} s in total"
               + (": the PC or the acquisition process stopped (sleep, App Nap, a hung "
                  "driver), and any drift across them went unrecorded." if paused.any()
                  else "."))
    late = float(np.percentile(latency, 95))
    if monitor is not None and 'dropped' in monitor:
        dropped, parted = int(monitor['dropped'][-1]), int(monitor['skipped'][-1])
        report.Add(TIMING, 'Bunches before settling', PASS, 0,
                   f"Settle-aware stepping: the tagger's window was held shut after every "
                   f"step for its jump's settling time, and {dropped} bunches in flight or "
                   f"settling were dropped. Reads trailed their bunches by {late * 1e3:.0f} ms "
                   f"(95th percentile).")
        report.Add(TIMING, 'Tagger count', WARN if parted else PASS, parted,
                   f"The tagger's bunch count and the loop's parted by {parted}: a read "
                   f"missed a trigger, or timed out with none. Every hit still carries the "
                   f"setpoint it was read at.")
        return nominal, 0
    # a run without the loop's own record: the next step went out as the
    # step's last bunch was read, so a bunch triggered within that latency
    # plus SETTLE of it was taken before the voltage had settled
    leading = min(int(np.ceil((late + settle_s) / nominal)), per_step)
    contaminated = leading / per_step
    report.Add(TIMING, 'Bunches before settling',
               _Grade(contaminated, TOLERANCES['ContaminatedFraction']), contaminated,
               f"Not settle-aware: with reads {late * 1e3:.0f} ms behind their bunches (95th "
               f"percentile, {TIMESTAMP_RESOLUTION_S * 1e3:.0f} ms resolution) and SETTLE of "
               f"{settle_s * 1e3:.0f} ms, about {leading} of every {per_step} bunches per step "
               f"were triggered before the step had settled; drop them in the analysis.")
    return nominal, leading


# ── Beam and spectrum ───────────────────────────────────────────

def _Beam_And_Spectrum(report, plan, per_step, raw, scan, step, t0):
    complete = scan['Complete']
    if complete < 1:
        report.Add(BEAM, 'Scans', NA, 0, "No scan completed, so scans cannot be compared.")
        return
    keep = scan['Scan'] < complete
    counts = np.zeros((complete, len(plan)))
    np.add.at(counts, (scan['Scan'][keep], scan['Index'][keep]), 1)
    totals = counts.sum(axis=1)
    rate = totals / (len(plan) * per_step)
    floor = TOLERANCES['BeamLossFraction'] * np.median(rate)
    lost = np.nonzero(rate < floor)[0]
    report.Add(BEAM, 'Beam losses', WARN if len(lost) else PASS, len(lost),
               f"{len(lost)} of {complete} scans counted below "
               f"{TOLERANCES['BeamLossFraction']:.0%} of the median rate "
               f"({np.median(rate):.3f} per bunch).")
    report.Cut(lost, f"beam lost: rate below {TOLERANCES['BeamLossFraction']:.0%} of the median")
    report.Series['Rates'] = (np.arange(complete), rate, floor)
    kept = np.setdiff1d(np.arange(complete), lost)
    if len(kept) > 1:
        mean = totals[kept].mean()
        dispersion = float(np.sum((totals[kept] - mean) ** 2) / mean / (len(kept) - 1))
        varied = _Excess(dispersion, len(kept) - 1)
        report.Add(BEAM, 'Rate stability', WARN if varied else PASS, dispersion,
                   f"The scans' totals scatter {dispersion:.2f} times as much as counting "
                   f"alone would over {len(kept)} scans"
                   f"{': the beam intensity varied' if varied else ', within statistics'}.")
        share = totals[kept] / totals[kept].sum()
        expected = counts[kept].sum(axis=0)[None, :] * share[:, None]
        filled = expected > 0
        dof = (len(kept) - 1) * (len(plan) - 1)
        chi2 = float(np.sum((counts[kept] - expected)[filled] ** 2 / expected[filled]) / dof)
        changed = _Excess(chi2, dof)
        report.Add(BEAM, 'Scan-to-scan shape', WARN if changed else PASS, chi2,
                   f"The scans' spectra differ from their mean at chi2/ndf {chi2:.2f}"
                   f"{': more than counting explains' if changed else ''}.")
    _Centroids(report, plan, counts, kept, scan, step, raw[:, 0] - t0)
    _Tof(report, raw, scan, complete)


def _Centroids(report, plan, counts, kept, scan, step, times):
    """The strongest peak's centroid scan by scan: the sum of every drift."""
    total = counts[kept].sum(axis=0)
    background = np.median(total)
    top = int(np.argmax(total))
    half = background + (total[top] - background) / 2
    low, high = top, top
    while low > 0 and total[low - 1] > half:
        low -= 1
    while high < len(plan) - 1 and total[high + 1] > half:
        high += 1
    width = high - low + 1
    window = slice(max(low - width, 0), min(high + width, len(plan) - 1) + 1)
    v = plan[window]
    peaked, centroids, errors, mids = [], [], [], []
    for k in kept:
        above = np.clip(counts[k, window] - np.median(counts[k]), 0, None)
        if above.sum() < TOLERANCES['MinPeakCounts']:
            continue
        c = float(np.sum(v * above) / above.sum())
        peaked.append(int(k))
        centroids.append(c)
        errors.append(float(np.sqrt(np.sum(above * (v - c) ** 2)) / above.sum()))
        mids.append(float(np.median(times[scan['Scan'] == k])))
    if len(centroids) < 3:
        report.Add(BEAM, 'Peak position', NA, None,
                   f"Fewer than three scans have {TOLERANCES['MinPeakCounts']} counts over "
                   f"background in the strongest peak.")
        return
    c, e = np.array(centroids), np.array(errors)
    drift, trend, _ = _Trend(mids, c, e)
    pulls = (c - trend) / np.maximum(e, 1e-9)
    jitter = float(np.sum(pulls ** 2) / (len(c) - 2)) if len(c) > 2 else 0.0
    report.Add(BEAM, 'Peak drift', _Grade(drift / step, TOLERANCES['DriftSteps']), drift,
               f"The peak near {np.mean(c):.2f} V moved {drift:.3f} V over the run "
               f"({drift / step:.2f} steps), across {len(c)} scans.")
    report.Add(BEAM, 'Peak jitter', WARN if _Excess(jitter, len(c) - 2) else PASS, jitter,
               f"Scan to scan the peak scatters about its trend at chi2/ndf {jitter:.2f}.")
    peaked = np.array(peaked)
    report.Cut(peaked[np.abs(pulls) > TOLERANCES['CentroidSigmas']],
               f"peak more than {TOLERANCES['CentroidSigmas']:g} sigma off the trend")
    report.Series['Centroids'] = (peaked, c, e, trend)


def _Tof(report, raw, scan, complete):
    """The TOF peak scan by scan: a drift beyond its own scatter is the
    buncher's timing or the beam energy moving."""
    tof = raw[:, 4]
    histogram, edges = np.histogram(tof, bins=np.arange(tof.min(), tof.max() + 0.2, 0.1))
    peak = edges[np.argmax(histogram)]
    near = np.abs(tof - peak) < 3.0
    scans, medians, errors = [], [], []
    for k in range(complete):
        hits = tof[near & (scan['Scan'] == k)]
        if len(hits) >= 10:
            scans.append(k)
            medians.append(np.median(hits))
            # the standard error of a median
            errors.append(np.sqrt(np.pi / 2) * np.std(hits) / np.sqrt(len(hits)))
    if len(scans) < 3:
        report.Add(TIMING, 'Time of flight', NA, None, "Too few hits in the TOF peak.")
        return
    drift, _, error = _Trend(scans, medians, errors)
    moved = drift > TOLERANCES['TofDriftUs'] and drift > 3 * error
    report.Add(TIMING, 'Time of flight', WARN if moved else PASS, drift,
               f"The TOF peak near {peak:.1f} µs drifted {drift:.2f} ± {error:.2f} µs over "
               f"{len(scans)} scans.")


# ── The cooler, the laser, the channels ─────────────────────────

def _Drift_Check(report, name, t, values_v, step, what):
    """Drift and spread of a quantity already in scan volts."""
    ok = np.isfinite(values_v)
    if ok.sum() < 3 or not np.ptp(t[ok]):
        report.Add(SOURCES, name, NA, None, f"Too few readings of the {what}.")
        return
    drift, line, _ = _Trend(t[ok], values_v[ok])
    spread = float(np.std(values_v[ok] - line))
    # the scatter holds the meter's own noise as well, so it can only warn
    verdict = max(_Grade(drift / step, TOLERANCES['DriftSteps']),
                  min(_Grade(spread / step, TOLERANCES['SpreadSteps']), WARN, key=RANK.get),
                  key=RANK.get)
    report.Add(SOURCES, name, verdict, drift,
               f"The {what} drifted {drift:.3f} V and scattered {spread:.3f} V about it, in "
               f"scan volts ({drift / step:.2f} and {spread / step:.2f} steps; the scatter "
               f"includes the meter's own noise).")


def _Cooler(report, raw, monitor, step, t0):
    # the run file keeps the cooler as its meter reads it, as the collinear DAQ did
    t, cooler = ((monitor['time'], monitor['cooler_v']) if monitor is not None
                 else (raw[:, 0], raw[:, 5] * COOLER_DIVIDER))
    lost = int(np.count_nonzero(~np.isfinite(cooler)))
    if lost:
        report.Add(SOURCES, 'Cooler readings', WARN, lost,
                   f"{lost} cooler readings were missing: the meter stopped answering.")
    _Drift_Check(report, 'Cooler', t - t0, cooler, step, 'cooler voltage')
    if np.isfinite(cooler).any():
        report.Series['Cooler'] = (t - t0, cooler - np.nanmean(cooler))


def _Laser(report, monitor, header, laser, plan, step, t0):
    mass = float(header.get('MassAMU', np.nan))
    beam_v = float(header.get('CoolerVoltage', np.nan)) * COOLER_DIVIDER - float(np.mean(plan))
    if monitor is None or not np.isfinite(monitor['wavenumber_cm1']).any():
        report.Add(SOURCES, 'Laser', NA, None,
                   "The wavemeter was not recorded during this run, so only its value at "
                   "the start is known.")
        return
    if not (np.isfinite(mass) and np.isfinite(beam_v) and beam_v > 0):
        report.Add(SOURCES, 'Laser', NA, None, "The run's mass or cooler voltage is unknown, "
                                              "so the laser cannot be put in scan volts.")
        return
    wavenumber = monitor['wavenumber_cm1']
    deviation_hz = (wavenumber - np.nanmean(wavenumber)) * C_LIGHT * 100.0
    volts_per_hz = Volts_Per_Hz(beam_v, np.nanmean(wavenumber) * C_LIGHT * 100.0, mass, laser)
    _Drift_Check(report, 'Laser', monitor['time'] - t0, deviation_hz * volts_per_hz, step,
                 f"laser ({volts_per_hz * 1e6:.3f} V per MHz at harmonic {laser['harmonic']}, "
                 f"{laser['geometry']})")
    report.Series['Laser'] = (monitor['time'] - t0, deviation_hz / 1e6)


def _Channels(report, raw):
    channels, counts = np.unique(raw[:, 3].astype(int), return_counts=True)
    shares = counts / counts.sum()
    weak = [int(c) for c, s in zip(channels, shares) if s < TOLERANCES['DeadChannelShare']]
    report.Add(BEAM, 'Channels', WARN if weak else PASS, len(weak),
               ', '.join(f"ch {c}: {s:.1%}" for c, s in zip(channels, shares))
               + (f"; channel {', '.join(map(str, weak))} barely counted: check its "
                  f"threshold and cabling." if weak else "."))


def Export(path, result):
    """The report beside the run as ``run_NNNN_performance.yaml``, its cuts
    under DENIS's ``scan_filters`` key so DENIS can take them as they are.
    Returns the path written."""
    out = path[:-len('.asdf')] + '_performance.yaml'
    excluded = sorted(result['Cuts'])
    record = {
        'run': os.path.basename(path), 'analysed': f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        'verdict': result['Verdict'], 'summary': result['Summary'],
        'checks': {section: [{'check': c['Check'], 'verdict': c['Verdict'],
                              'message': c['Message']} for c in checks]
                   for section, checks in result['Sections'].items()},
        'cuts': result['Cuts'],
        'leading_bunches_per_step': int(result['Leading_Bunches']),
        'scan_filters': {os.path.abspath(path): {'excluded': excluded}} if excluded else {},
    }
    with open(out, 'w', encoding='utf-8') as fh:
        yaml.safe_dump(record, fh, sort_keys=False, allow_unicode=True, width=100)
    return out
