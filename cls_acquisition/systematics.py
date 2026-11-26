"""The systematics' inputs for a run: what each source of error amounts to, at the line.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A. Raggio's budget (Bi, 306 nm, November 2026) refits every run, which is
DENIS's work; ORACLE gives its inputs: each source's width in scan volts
and in MHz at the line, marked measured, declared (from the settings) or
not yet known. The beam's offset from the cooler readback comes from
reference runs, which a run can be tagged as. The table is shown on Scan
Performance, written as ``run_NNNN_systematics.yaml`` (each source keyed by
what it moves in cls_tools, with the arguments cls_tools loads the run with)
and posted to the ELOG (a short block per run, the whole on request).

Depends on: numpy, PyYAML; cls_acquisition.scan_table.
"""

import glob
import os
from datetime import datetime

import numpy as np
import yaml

from cls_acquisition import scan_table

MEASURED, DECLARED, UNKNOWN = 'measured', 'declared', 'not yet known'
FRAME = ('VAccDiv', 'VCoolDiv', 'VCoolOffset')      # clstools.CLSDataFrame's arguments


def _Row(source, kind, volts, mhz_per_v, detail, contract=(None,) * 5, value=None):
    """A source's row; ``contract`` is how cls_tools takes it (schema 3 of the yaml):
    its id, the clstools.CLSDataFrame attribute it moves (None: informational, or
    cls_tools' own fit), its 1 sigma in its unit (None: not yet known), the unit and
    how runs share it; ``value`` is that attribute's value for the run, where ORACLE
    knows it."""
    mhz = None if volts is None or not mhz_per_v else abs(volts) * mhz_per_v
    row_id, key, width, unit, correlation = contract
    return {'Source': source, 'Kind': kind, 'Volts': None if volts is None else abs(volts),
            'Mhz': mhz, 'Detail': detail, 'Id': row_id, 'Key': key, 'Value': value,
            'Width': None if width is None else abs(float(width)), 'Unit': unit,
            'Correlation': correlation}


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
    """A row per source: ``Source``, ``Kind`` (MEASURED, DECLARED, UNKNOWN),
    ``Volts`` (at the line), ``Mhz`` (at the ions) and ``Detail``, from the
    run's (scan, cooler) ``dividers``, Scan Performance's ``budget``,
    ``volts_per_hz``, ``detectors``, ``timing`` verdicts and ``declared``."""
    rows = []
    offset = declared.get('beam_offset_v')
    reference = " This run is tagged as a reference run." if record.get('ReferenceRun') else ""
    if isinstance(offset, (list, tuple)) and len(offset) == 2:
        rows.append(_Row('Beam offset δU', DECLARED, offset[1], mhz_per_v,
                         f"{offset[0]:+g} ± {offset[1]:g} V, from the campaign's reference "
                         f"runs (settings: systematics, beam_offset_v); it also scales every "
                         f"splitting by δU/2U.{reference}",
                         ('beam_offset', 'VCoolOffset', offset[1], 'V', 'campaign'),
                         float(offset[0])))
    else:
        rows.append(_Row('Beam offset δU', UNKNOWN, None, mhz_per_v,
                         f"None declared: set systematics: beam_offset_v from the campaign's "
                         f"reference runs. It dominated the centroid for Bi in November 2026 "
                         f"(8 V, 73 MHz).{reference}",
                         ('beam_offset', 'VCoolOffset', None, 'V', 'campaign')))

    ratios = record.get('Dividers') or {}
    cooler_v = float(header.get('CoolerVoltage', np.nan)) * dividers[1]
    scan_v = float(np.max(np.abs(np.asarray(header.get('ScanningRanges') or header.get(
        'CalSet', [0.0]), dtype=float))))
    for name, key, level, what, row_id, frame, ratio in (
            ('Cooler divider', 'Cooler', cooler_v, 'the cooler', 'cooler_divider',
             'VCoolDiv', dividers[1]),
            ('Scan divider', 'Scan', scan_v, 'the largest setpoint', 'scan_divider',
             'VAccDiv', dividers[0])):
        ppm = (ratios.get(key) or {}).get('ppm')
        if ppm is None or not np.isfinite(level):
            rows.append(_Row(name, UNKNOWN, None, mhz_per_v,
                             "The run does not record its divider's stated accuracy.",
                             (row_id, frame, None, 'ppm', 'campaign'), float(ratio)))
        else:
            rows.append(_Row(name, DECLARED, level * ppm * 1e-6, mhz_per_v,
                             f"±{ppm:g} ppm of {what}'s {level:.0f} V (settings: hardware).",
                             (row_id, frame, ppm, 'ppm', 'campaign'), float(ratio)))

    cal_set = np.asarray(header.get('CalSet', []), dtype=float)
    readback = np.asarray(header.get('CalReadback', []), dtype=float) * dividers[0]
    fit = _Calibration(cal_set[:len(readback)], readback[:len(cal_set)])
    table = record.get('TableFit')
    if isinstance(table, dict):
        # a + b*T at the table's furthest point from 0 V in the scan, with their covariance
        reach = table['Reach_T_V']
        axis = float(np.sqrt(table['A_Err_V'] ** 2 + (table['B_Err'] * reach) ** 2
                             + 2.0 * table.get('Cov_AB', 0.0) * reach))
        gain = (f"b ± {table['B_Err'] * 1e6:.1f} ppm"
                if table.get('Scale_Mode', 'free') == 'free' else
                "b = 1, the bench's, as cls_tools takes it under 400 V of calibration")
        rows.append(_Row('Run calibration', MEASURED, axis, mhz_per_v,
                         f"Fitted to bench table {table['Table'][:8]}: a ± "
                         f"{table['A_Err_V'] * 1e3:.1f} mV, {gain}, "
                         f"the axis ± {axis:.3f} V at the scan's furthest setpoint; the "
                         f"readings sit {table['Rms_V'] * 1e3:.1f} mV rms from the table.",
                         ('run_calibration', None, axis, 'V', 'run')))
    elif fit is None:
        rows.append(_Row('Run calibration', UNKNOWN, None, mhz_per_v,
                         "The run holds no calibration to fit.",
                         ('run_calibration', None, None, 'V', 'run')))
    else:
        offset_error, gain_error, reach = fit
        rows.append(_Row('Run calibration', MEASURED, reach, mhz_per_v,
                         f"The straight line through the run's calibration: offset ± "
                         f"{offset_error:.3f} V, gain ± {gain_error * 1e6:.0f} ppm, the axis "
                         f"± {reach:.3f} V at the scan's ends; no bench table was in force.",
                         ('run_calibration', None, reach, 'V', 'run')))
    rows.append(_Row('Bench table repeatability', UNKNOWN, None, mhz_per_v,
                     f"Bench table {record['ScanTable'][:8]} was in force; a table check "
                     f"measures how well it still fits, and none has been made yet."
                     if record.get('ScanTable') else "No bench table was in force.",
                     ('table_repeatability', 'Scan_calibration', None, 'V', 'table'),
                     record.get('ScanTable')))
    drift = declared.get('zero_drift_v')
    rows.append(_Row('0 V drift during the run', DECLARED if drift else UNKNOWN, drift,
                     mhz_per_v, f"{drift * 1e3:.0f} mV, assumed as in the November 2026 budget "
                                f"until ORACLE checks 0 V during runs." if drift else
                     "Neither measured nor declared.",
                     ('zero_drift', 'Scan_offset_v', drift, 'V', 'run')))
    ids = {'HV': 'hv_calibration', 'Laser': 'wavemeter_run', 'Cooler': 'cooler_scatter'}
    for name, key, detail in (
            ('HV against its calibration', 'HV',
             "The line meter's readback during the scan against the calibration's own reading "
             "at each setpoint."),
            ('Wavemeter during the run', 'Laser',
             "The wavemeter's mean and wander against the setpoint DENIS takes."),
            ('Cooler scatter', 'Cooler',
             f"Its readings about their mean over four periods of the mains ripple as they "
             f"alias it, {record.get('CoolerRollingS') or 15:g} s at least, as cls_tools takes "
             f"it: the ripple, which broadens the line, and the meter's noise.")):
        contract = (ids[key], None, budget.get(key), 'V', 'run')
        rows.append(_Row(name, MEASURED, budget[key], mhz_per_v, detail, contract)
                    if key in budget else _Row(
                        name, UNKNOWN, None, mhz_per_v,
                        "The run does not hold what measures it (its monitor record).",
                        contract))
    accuracy = declared.get('wavemeter_mhz')
    rows.append(_Row('Wavemeter accuracy', DECLARED if accuracy else UNKNOWN,
                     accuracy * 1e6 * volts_per_hz if accuracy and volts_per_hz else None,
                     mhz_per_v, f"{accuracy:g} MHz at the fundamental per run (settings: "
                                f"systematics, wavemeter_mhz)." if accuracy else
                     "Not declared.",
                     ('wavemeter', 'Laser_set', accuracy, 'MHz', 'run')))

    flagged = {c: d for c, d in detectors.items() if d['standard_errors'] > 3.0
               and d['afterpulse_share'] > 0.005}
    rows.append(_Row('Afterpulsing', MEASURED, None, mhz_per_v, (
        '; '.join(f"channel {c}: {d['afterpulse_share']:.1%} of its counts"
                  + (f", {d['suggested_dead_time_us']:g} µs dead time suggested"
                     if d['suggested_dead_time_us'] is not None else '')
                  for c, d in flagged.items()) or "no PMT afterpulses beyond its control")
        + ". It inflates the errors, not the centroid.",
        ('afterpulsing', None, None, None, 'run')))
    dropped, skipped = ((int(monitor['dropped'][-1]), int(monitor['skipped'][-1]))
                        if monitor is not None and len(monitor.get('dropped', ())) else
                        (None, None))
    worst = max(timing, key=['N/A', 'PASS', 'WARN', 'FAIL'].index, default='N/A')
    rows.append(_Row('Health', MEASURED, None, mhz_per_v,
                     (f"{skipped} triggers missed and {dropped} bunches dropped while steps "
                      f"settled; " if skipped is not None else "")
                     + f"the timing checks' worst verdict {worst}.",
                     ('health', None, None, None, 'run')))
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


def Clstools(path, rows):
    """What cls_tools needs beyond the run file to load it as ORACLE took it:
    ``CLSDataFrame``'s dividers and the beam offset declared,
    and ``Use_Scan_Calibration``'s table, the one in force as cls_tools reads
    it (written from the library if it is not yet), relative to the run's
    folder, or None if the run's ``<data>/tables`` does not hold it."""
    calls = {'CLSDataFrame': {r['Key']: r['Value'] for r in rows
                              if r['Key'] in FRAME and r['Value'] is not None}}
    table = next((r['Value'] for r in rows if r['Key'] == 'Scan_calibration'), None)
    if table:
        folder = os.path.dirname(path)
        found = glob.glob(os.path.join(folder, '..', '..', '..', scan_table.FOLDER,
                                       f"table_*_{table[:8]}.csv"))
        try:
            out = scan_table.Clstools(scan_table.Load(found[0]), found[0]) if found else None
        except (OSError, KeyError, scan_table.Table_Error):
            out = None
        # with '/', which every system reads, as the runs travel from the beamline's PC
        calls['Use_Scan_Calibration'] = {
            'table': out and os.path.relpath(out, folder).replace(os.sep, '/')}
    return calls


def Export(path, rows):
    """The table beside the run as ``run_NNNN_systematics.yaml``; returns the
    path written."""
    out = path[:-len('.asdf')] + '_systematics.yaml'
    # schema 3 keys each row by the clstools attribute it moves, for cls_tools and the
    # DENIS built on it; schema 2's DENIS keys went with DENIS's own budget import
    record = {'schema': 3, 'run': os.path.basename(path),
              'estimated': f"{datetime.now():%Y-%m-%d %H:%M:%S}",
              'units': 'volts: width at the line in scan volts; mhz: at the ions; key: the '
                       'clstools.CLSDataFrame attribute it moves (null: informational, or '
                       "cls_tools' own fit); value: its value for this run where ORACLE knows "
                       'it; width: 1 sigma in unit (null: not yet known)',
              # data = clstools.CLSDataFrame(**CLSDataFrame), then Load_Run and
              # Use_Scan_Calibration(**Use_Scan_Calibration), its table relative to this file
              'clstools': Clstools(path, rows),
              'sources': [{'source': r['Source'], 'id': r.get('Id'), 'kind': r['Kind'],
                           'volts': r['Volts'], 'mhz': r['Mhz'], 'detail': r['Detail'],
                           'key': r.get('Key'), 'value': r.get('Value'),
                           'width': r.get('Width'), 'unit': r.get('Unit'),
                           'correlation': r.get('Correlation')}
                          for r in rows]}
    with open(out, 'w', encoding='utf-8') as fh:
        yaml.safe_dump(record, fh, sort_keys=False, allow_unicode=True, width=100)
    return out
