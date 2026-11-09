"""The systematics' inputs for a run: what each source of error amounts to, at the line.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A. Raggio's budget for Bi at 306 nm (November 2026) drew every source at
random, recomputed every event's frequency and refitted, run by run. That
needs the fit, so it is DENIS's; what ORACLE gives it is the inputs: each
source's width in scan volts at the line, and in MHz at the ions through
the Doppler shift where the run holds what converts it, each marked as
measured on this run, declared (a number the settings give, from a
campaign's reference runs or a datasheet), or not yet known (what a
later work package measures). The beam's offset from the cooler's
readback dominated the centroid there (8 V, 73 MHz) and is not the DAQ's
to measure: reference runs are, so a run can be tagged as one.

The table is shown on the Scan Performance tab, written beside the run
(``run_NNNN_systematics.yaml``), and posted to the ELOG: a short block as
each run is analysed, the whole table on request.

Depends on: numpy, PyYAML.
"""

import os
from datetime import datetime

import numpy as np
import yaml

MEASURED, DECLARED, UNKNOWN = 'measured', 'declared', 'not yet known'


def _Row(source, kind, volts, mhz_per_v, detail):
    mhz = None if volts is None or not mhz_per_v else abs(volts) * mhz_per_v
    return {'Source': source, 'Kind': kind, 'Volts': None if volts is None else abs(volts),
            'Mhz': mhz, 'Detail': detail}


def _Calibration(cal_set, readback):
    """``(offset error V, gain error, the axis's error at the scan's ends V)``
    of the straight line through the calibration, or None."""
    if len(cal_set) < 4 or np.ptp(cal_set) == 0:
        return None
    # the coefficients are (gain, offset); their covariance scaled by the residuals' scatter
    _, cov = np.polyfit(cal_set, readback, 1, cov=True)
    ends = np.array([[cal_set.min(), 1.0], [cal_set.max(), 1.0]])
    reach = np.sqrt(np.max(np.einsum('ij,jk,ik->i', ends, cov, ends)))
    return float(np.sqrt(cov[1, 1])), float(np.sqrt(cov[0, 0])), float(reach)


def Table(header, record, monitor, dividers, budget, mhz_per_v, volts_per_hz, detectors,
          timing, declared):
    """The run's systematics' inputs, a row per source: ``Source``, ``Kind``
    (MEASURED, DECLARED or UNKNOWN), ``Volts`` (its width at the line in
    scan volts, or None), ``Mhz`` (at the ions, or None) and ``Detail``.
    ``dividers`` are the run's (scan, cooler) ratios, ``budget`` what Scan
    Performance measured (scan volts by 'HV', 'Laser', 'Cooler'),
    ``volts_per_hz`` scan volts per hertz of the laser's fundamental,
    ``detectors`` each PMT's afterpulsing, ``timing`` the Timing checks'
    verdicts, ``declared`` the settings' systematics."""
    rows = []
    region = record.get('InteractionRegion')
    offset = (declared.get('beam_offset_v') or {}).get(region)
    reference = " This run is tagged as a reference run." if record.get('ReferenceRun') else ""
    if offset is not None and len(offset) == 2:
        rows.append(_Row('Beam offset δU', DECLARED, offset[1], mhz_per_v,
                         f"{offset[0]:+g} ± {offset[1]:g} V for the {region}, from the "
                         f"campaign's reference runs (settings: systematics, beam_offset_v); "
                         f"it also scales every splitting by δU/2U.{reference}"))
    else:
        which = f"the {region}" if region else "a run whose interaction region was not stated"
        rows.append(_Row('Beam offset δU', UNKNOWN, None, mhz_per_v,
                         f"None declared for {which}: set systematics: beam_offset_v from the "
                         f"campaign's reference runs. It dominated the centroid for Bi in "
                         f"November 2026 (8 V, 73 MHz).{reference}"))

    ratios = record.get('Dividers') or {}
    cooler_v = float(header.get('CoolerVoltage', np.nan)) * dividers[1]
    scan_v = float(np.max(np.abs(np.asarray(header.get('CalSet', [0.0]), dtype=float))))
    for name, key, level, what in (('Cooler divider', 'Cooler', cooler_v, 'the cooler'),
                                   ('Scan divider', 'Scan', scan_v, 'the largest setpoint')):
        ppm = (ratios.get(key) or {}).get('ppm')
        if ppm is None or not np.isfinite(level):
            rows.append(_Row(name, UNKNOWN, None, mhz_per_v,
                             "The run does not record its divider's stated accuracy."))
        else:
            rows.append(_Row(name, DECLARED, level * ppm * 1e-6, mhz_per_v,
                             f"±{ppm:g} ppm of {what}'s {level:.0f} V (settings: hardware)."))

    cal_set = np.asarray(header.get('CalSet', []), dtype=float)
    readback = np.asarray(header.get('CalReadback', []), dtype=float) * dividers[0]
    fit = _Calibration(cal_set[:len(readback)], readback[:len(cal_set)])
    table = record.get('TableFit')
    if isinstance(table, dict):
        # a + b*T at the table's furthest point from 0 V in the scan, with their covariance
        reach = table['Reach_T_V']
        axis = float(np.sqrt(table['A_Err_V'] ** 2 + (table['B_Err'] * reach) ** 2
                             + 2.0 * table.get('Cov_AB', 0.0) * reach))
        rows.append(_Row('Run calibration', MEASURED, axis, mhz_per_v,
                         f"Fitted to bench table {table['Table'][:8]}: a ± "
                         f"{table['A_Err_V'] * 1e3:.1f} mV, b ± {table['B_Err'] * 1e6:.1f} ppm, "
                         f"the axis ± {axis:.3f} V at the scan's furthest setpoint; the "
                         f"readings sit {table['Rms_V'] * 1e3:.1f} mV rms from the table."))
    elif fit is None:
        rows.append(_Row('Run calibration', UNKNOWN, None, mhz_per_v,
                         "The run holds no calibration to fit."))
    else:
        offset_error, gain_error, reach = fit
        rows.append(_Row('Run calibration', MEASURED, reach, mhz_per_v,
                         f"The straight line through the run's calibration: offset ± "
                         f"{offset_error:.3f} V, gain ± {gain_error * 1e6:.0f} ppm, the axis "
                         f"± {reach:.3f} V at the scan's ends; no bench table was in force."))
    rows.append(_Row('Bench table repeatability', UNKNOWN, None, mhz_per_v,
                     f"Bench table {record['ScanTable'][:8]} was in force; a table check "
                     f"measures how well it still fits, and none has been made yet."
                     if record.get('ScanTable') else "No bench table was in force."))
    drift = declared.get('zero_drift_v')
    rows.append(_Row('0 V drift during the run', DECLARED if drift else UNKNOWN, drift,
                     mhz_per_v, f"{drift * 1e3:.0f} mV, assumed as in the November 2026 budget "
                                f"until ORACLE checks 0 V during runs." if drift else
                     "Neither measured nor declared."))
    for name, key, detail in (
            ('HV against its calibration', 'HV',
             "The line meter's readback during the scan against the calibration's own reading "
             "at each setpoint."),
            ('Wavemeter during the run', 'Laser',
             "The wavemeter's mean and wander against the setpoint DENIS takes."),
            ('Cooler scatter', 'Cooler',
             f"Its readings about their {record.get('CoolerRollingS') or 15:g} s rolling mean: "
             f"the mains ripple, which broadens the line, and the meter's noise.")):
        rows.append(_Row(name, MEASURED, budget[key], mhz_per_v, detail) if key in budget
                    else _Row(name, UNKNOWN, None, mhz_per_v,
                              "The run does not hold what measures it (its monitor record)."))
    accuracy = declared.get('wavemeter_mhz')
    rows.append(_Row('Wavemeter accuracy', DECLARED if accuracy else UNKNOWN,
                     accuracy * 1e6 * volts_per_hz if accuracy and volts_per_hz else None,
                     mhz_per_v, f"{accuracy:g} MHz at the fundamental per run (settings: "
                                f"systematics, wavemeter_mhz)." if accuracy else
                     "Not declared."))

    flagged = {c: d for c, d in detectors.items() if d['standard_errors'] > 3.0
               and d['afterpulse_share'] > 0.005}
    rows.append(_Row('Afterpulsing', MEASURED, None, mhz_per_v, (
        '; '.join(f"channel {c}: {d['afterpulse_share']:.1%} of its counts"
                  + (f", {d['suggested_dead_time_us']:g} µs dead time suggested"
                     if d['suggested_dead_time_us'] is not None else '')
                  for c, d in flagged.items()) or "no PMT afterpulses beyond its control")
        + ". It inflates the errors, not the centroid."))
    dropped, skipped = ((int(monitor['dropped'][-1]), int(monitor['skipped'][-1]))
                        if monitor is not None and len(monitor.get('dropped', ())) else
                        (None, None))
    worst = max(timing, key=['N/A', 'PASS', 'WARN', 'FAIL'].index, default='N/A')
    rows.append(_Row('Health', MEASURED, None, mhz_per_v,
                     (f"{skipped} triggers missed and {dropped} bunches dropped while steps "
                      f"settled; " if skipped is not None else "")
                     + f"the timing checks' worst verdict {worst}."))
    return rows


def Largest(rows):
    """The row of the largest term, in MHz where any has them, else volts."""
    sized = [r for r in rows if r['Mhz'] is not None] or [r for r in rows
                                                            if r['Volts'] is not None]
    return max(sized, key=lambda r: r['Mhz'] if r['Mhz'] is not None else r['Volts'],
               default=None)


def _Size(row):
    if row['Volts'] is None:
        return ''
    return f"{row['Volts']:.3f} V" + (f", {row['Mhz']:.1f} MHz" if row['Mhz'] is not None
                                      else '')


def Elog_Text(rows, run, full=False):
    """The ELOG's text for ``run`` (its file's name, run_NNNN): a short
    block (the calibration, the drift, the cooler, the afterpulsing and
    the largest term), or ``full``, every row."""
    title = (f"SYSTEMATIC INPUTS, {run} (estimated by ORACLE"
             + ("" if full else "; the full table is on Scan Performance") + ")\n")
    by = {r['Source']: r for r in rows}
    if full:
        lines = [f"{r['Source']} ({r['Kind']}){': ' + _Size(r) if _Size(r) else ''}. "
                 f"{r['Detail']}" for r in rows]
    else:
        lines = [by['Run calibration']['Detail'],
                 f"0 V drift: {_Size(by['0 V drift during the run']) or 'unknown'} "
                 f"({by['0 V drift during the run']['Kind']}); HV against its calibration: "
                 f"{_Size(by['HV against its calibration']) or 'not measured'}",
                 f"Cooler scatter: {_Size(by['Cooler scatter']) or 'not measured'}",
                 f"Afterpulsing: {by['Afterpulsing']['Detail']}"]
    largest = Largest(rows)
    if largest is not None:
        lines.append(f"Largest term: {largest['Source']}, {_Size(largest)} ({largest['Kind']})")
    return title + '\n'.join(lines) + '\n'


def Export(path, rows):
    """The table beside the run as ``run_NNNN_systematics.yaml``; returns the
    path written."""
    out = path[:-len('.asdf')] + '_systematics.yaml'
    record = {'run': os.path.basename(path),
              'estimated': f"{datetime.now():%Y-%m-%d %H:%M:%S}",
              'units': 'volts: width at the line in scan volts; mhz: at the ions',
              'sources': [{'source': r['Source'], 'kind': r['Kind'], 'volts': r['Volts'],
                           'mhz': r['Mhz'], 'detail': r['Detail']} for r in rows]}
    with open(out, 'w', encoding='utf-8') as fh:
        yaml.safe_dump(record, fh, sort_keys=False, allow_unicode=True, width=100)
    return out
