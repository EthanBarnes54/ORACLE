"""The run file, in the ASDF format the archive and its readers expect.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The header's keys are frozen (cls_tools and DENIS read them); events
follow as float64 rows of :data:`DATA_COLS` in ASDF's streamed block,
under ``<data>/<year>/<month>/<day>/run_NNNN.asdf``. ORACLE's own record
sits under :data:`ORACLE_KEY`, which readers pass over: the view, the rack,
its dividers and DAC codes, the bench table and fit. Readers scale by the
run's own dividers (:func:`Recorded_Divider`). Numbers count on from the
highest in the folder; no file is overwritten. ``run_NNNN_monitor.csv``
beside it records the rack every 0.1 s.

Depends on: numpy, asdf.
"""

import glob
import os
import re

import asdf
import numpy as np
from asdf import AsdfFile, Stream

DATA_COLS = ['timestamp', 'voltage', 'bunch_number', 'channel', 'time', 'cooler']
MONITOR_COLS = ('time', 'round', 'setpoint_v', 'readback_v', 'cooler_v', 'wavenumber_cm1',
                'bunches', 'dropped', 'skipped')

# the header in the archive's order. SortingUtilities is fixed; the
# caller supplies the rest
HEADER_KEYS = ('SortingUtilities', 'Run', 'Experiment', 'Date', 'BunchesPerChannel',
               'StepSize', 'DwellTime', 'MassAMU', 'MassSetPoint', 'CoolerVoltage',
               'LaserSetpoint', 'ScanningRanges', 'CalSet', 'CalReadback',
               'DacCorrectionApplied', 'DacCorrection')
SORTING_UTILITIES = 'https://github.com/andry3vi/cls_tools.git'
ORACLE_KEY = 'ORACLE'

# the ASDF standard the archive was written to, which its readers know
ASDF_STANDARD = '1.5.0'

_RUN_NAME = re.compile(r'run_(\d+)\.asdf$')


def _Numbered(directory):
    """``(run number, path)`` of every run file under ``directory``, newest first."""
    paths = glob.glob(os.path.join(directory, '*', '*', '*', 'run_*.asdf'))
    return sorted(((int(m.group(1)), p) for p, m in zip(paths, map(_RUN_NAME.search, paths))
                   if m), reverse=True)


def Last_Run_Number(directory):
    """The highest run number under ``directory``, or 0 if there is none."""
    runs = _Numbered(directory)
    return runs[0][0] if runs else 0


def Create(directory, header, record=None):
    """Write a new run's header (every key of :data:`HEADER_KEYS` but
    SortingUtilities, no others, so the format cannot drift), with ``record``
    under :data:`ORACLE_KEY` if given; returns the path."""
    expected = set(HEADER_KEYS[1:])
    if set(header) != expected:
        raise ValueError(f"A run header must hold exactly {sorted(expected)}; this one "
                         f"differs by {sorted(set(header) ^ expected)}.")
    date = header['Date']
    folder = os.path.join(directory, f"{date.year}", f"{date.month:02d}", f"{date.day:02d}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"run_{header['Run']:04d}.asdf")

    tree = {'SortingUtilities': SORTING_UTILITIES}
    tree.update((key, header[key]) for key in HEADER_KEYS[1:])
    if record is not None:
        tree[ORACLE_KEY] = record
    tree.update(raw_header=DATA_COLS, raw=Stream([len(DATA_COLS)], np.float64))
    asdf_file = AsdfFile(tree, version=ASDF_STANDARD)
    for key in ('CalSet', 'CalReadback'):
        asdf_file.set_array_storage(tree[key], 'inline')
    with open(path, 'xb') as fh:
        asdf_file.write_to(fh)
    return path


def Append(path, rows):
    with open(path, 'ab') as fh:
        fh.write(np.ascontiguousarray(rows, dtype=np.float64).tobytes())


def Runs(directory):
    """Every run file under ``directory``, newest run number first."""
    return [path for _, path in _Numbered(directory)]


def Recorded_Divider(tree, which):
    """The ratio of the run's ``which`` divider ('Scan' or 'Cooler') as its
    record holds it, or None for a run that does not record one."""
    try:
        return float(tree[ORACLE_KEY]['Dividers'][which]['ratio'])
    except (KeyError, TypeError, ValueError):
        return None


def Earlier_Calibrations(path, directory, count=3):
    """``(run, CalSet, CalReadback, scan divider or None)`` of the ``count``
    runs before ``path`` taken by the same DAQ with the same DacCorrection
    (else they measure different things), newest first; only those keys are
    read, and an unreadable run is passed over."""
    found = _RUN_NAME.search(path)
    if found is None or not os.path.isdir(directory):
        return []
    try:
        with asdf.open(path, memmap=False) as af:
            correction = af.tree.get('DacCorrection')
    except (OSError, ValueError):
        return []
    number, calibrations = int(found.group(1)), []
    for run, other in _Numbered(directory):
        if run >= number:
            continue
        try:
            with asdf.open(other, memmap=False) as af:
                if af.tree.get('DacCorrection') != correction:
                    continue
                calibrations.append((run, np.array(af.tree['CalSet'], dtype=float),
                                     np.array(af.tree['CalReadback'], dtype=float),
                                     Recorded_Divider(af.tree, 'Scan')))
        except (OSError, KeyError, ValueError, TypeError):
            continue
        if len(calibrations) == count:
            break
    return calibrations


def Earlier_Table_Fits(directory, table_id, before=None, count=5):
    """``(run, a, b)`` of the ``count`` runs (below ``before``, if given) fitted
    to bench table ``table_id``, newest first, from ORACLE's record alone."""
    if not table_id or not os.path.isdir(directory):
        return []
    fits = []
    for run, other in _Numbered(directory):
        if before is not None and run >= before:
            continue
        try:
            with asdf.open(other, memmap=False) as af:
                fit = (af.tree.get(ORACLE_KEY) or {}).get('TableFit')
                # a run not fitted keeps the sentence saying why instead
                if isinstance(fit, dict) and fit.get('Table') == table_id:
                    fits.append((run, float(fit['A_V']), float(fit['B'])))
        except (OSError, KeyError, ValueError, TypeError):
            continue
        if len(fits) == count:
            break
    return fits


def Monitor_Path(path):
    return path[:-len('.asdf')] + '_monitor.csv'


def Append_Monitor(path, rows):
    """Add rows of :data:`MONITOR_COLS` to the run's monitor record."""
    monitor = Monitor_Path(path)
    with open(monitor, 'a', encoding='utf-8', newline='') as fh:
        if fh.tell() == 0:
            fh.write(','.join(MONITOR_COLS) + '\n')
        np.savetxt(fh, np.asarray(rows, dtype=float), delimiter=',', fmt='%.13g')


def Read(path):
    """``(header, raw rows, monitor)``: the header (with :data:`ORACLE_KEY` if
    recorded), an (n, 6) array, and the monitor's columns, or None without."""
    with asdf.open(path, memmap=False) as af:
        # without memmap the arrays are read into memory, so they outlive the
        # file; the header's are copied out, and the events, the run's whole
        # size, are taken as read rather than copied a second time
        header = {key: np.array(af.tree[key]) if hasattr(af.tree[key], 'shape') else af.tree[key]
                  for key in (*HEADER_KEYS, ORACLE_KEY) if key in af.tree}
        raw = np.asarray(af.tree['raw'], dtype=float).reshape(-1, len(DATA_COLS))
    monitor = None
    if os.path.exists(Monitor_Path(path)):
        table = np.genfromtxt(Monitor_Path(path), delimiter=',', names=True)
        monitor = {name: np.atleast_1d(table[name]) for name in table.dtype.names}
    return header, raw, monitor
