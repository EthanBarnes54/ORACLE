"""The run file, in the ASDF format the archive and its readers expect.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The header's keys are frozen, as cls_tools and DENIS read them; the
events follow as raw float64 rows of :data:`DATA_COLS`, appended as
ASDF's streamed block, under ``<data>/<year>/<month>/<day>/run_NNNN.asdf``.
ORACLE's own record of the run sits under one more key, :data:`ORACLE_KEY`,
which readers that look keys up by name pass over: what opening the run
needs to show it as taken, and what the rack was and held (identity,
dividers, DAC codes, the bench table and its fit). It is written with the
header, from settings locked for the scan. A reader takes a run's own
divider ratios (:func:`Recorded_Divider`), never today's settings.

A run's number is one more than the highest in the data folder, so
numbers survive a restart and a refused calibration leaves no gap; no
file is ever overwritten. ``run_NNNN_monitor.csv`` beside it records the
rack every 0.1 s (setpoint, readback, cooler, wavemeter, the loop's
counts), separate so nothing that reads run files sees a change.

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
# every trigger of a bunched scan beside the run (run_NNNN_cooler.npy): the tagger's
# number, the bunch_number it became (-1: dropped while its step settled), when it was
# read (s, as timestamp), the cooler meter's volts then in force (as cooler, Vrfq) and
# how long before the read the meter had taken them (s)
COOLER_LOG = np.dtype([('trigger', '<i8'), ('bunch', '<i8'), ('time', '<f8'),
                       ('cooler', '<f8'), ('age_s', '<f4')])

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
    """Write a new run's header, and ORACLE's ``record`` of it under
    :data:`ORACLE_KEY` if given, and return the file's path.

    ``header`` holds every key of :data:`HEADER_KEYS` but SortingUtilities;
    any other set of keys is refused, so the format cannot drift.
    """
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
    """The start-of-run calibrations of the ``count`` runs before ``path`` in
    ``directory`` calibrated as it was (same DAQ, same DacCorrection),
    newest first, as ``(run, CalSet, CalReadback, scan divider or None)``;
    only those keys are read, and an unreadable run is passed over. Across
    DAQs or corrections two calibrations measure different things (the old
    DAQ's Mode B; a correction moves every setpoint), so would differ with
    nothing wrong in either."""
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
    """``(run, a, b)`` of the ``count`` runs in ``directory`` (numbered below
    ``before``, if given) whose calibrations were fitted to the bench table
    ``table_id``, newest first: how a run's 0 V level and gain have moved
    against the same table. Only ORACLE's record is read, never the events."""
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


def Cooler_Log_Path(path, part=False):
    return path[:-len('.asdf')] + ('_cooler.part' if part else '_cooler.npy')


def Append_Cooler_Log(path, rows):
    """Add rows of (trigger, bunch, time, cooler, age) to the run's cooler log,
    kept as taken in its ``.part`` until :func:`Close_Cooler_Log`."""
    rows = np.asarray(rows, dtype=float)
    log = np.empty(len(rows), COOLER_LOG)
    for i, name in enumerate(COOLER_LOG.names):
        log[name] = rows[:, i]
    with open(Cooler_Log_Path(path, part=True), 'ab') as fh:
        log.tofile(fh)


def Close_Cooler_Log(path):
    """The run's cooler log made a .npy, which keeps its own columns."""
    part = Cooler_Log_Path(path, part=True)
    if os.path.exists(part):
        np.save(Cooler_Log_Path(path), np.fromfile(part, COOLER_LOG))
        os.remove(part)


def Read_Cooler_Log(path):
    """The run's cooler log (:data:`COOLER_LOG`'s columns), as closed or as
    far as it was taken, or None for a run that has none."""
    if os.path.exists(Cooler_Log_Path(path)):
        return np.load(Cooler_Log_Path(path))
    if os.path.exists(Cooler_Log_Path(path, part=True)):
        return np.fromfile(Cooler_Log_Path(path, part=True), COOLER_LOG)
    return None


def Read(path):
    """``(header, raw rows, monitor)``: the header as a dict, with ORACLE's
    record under :data:`ORACLE_KEY` if the run has one, the rows as an
    (n, 6) array, and the monitor record as a dict of columns, or None for
    a run that has none (one written before it existed, or by the old DAQ)."""
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
