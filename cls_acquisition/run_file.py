"""The run file, in the ASDF format the archive and its readers expect.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The header's keys and their meaning are frozen: cls_tools and DENIS read
these files, and the archived campaigns fix the format. The events
follow the header as raw float64 rows of :data:`DATA_COLS`, appended
while the scan runs, as ASDF's streamed block. Files go to
``<data>/<year>/<month>/<day>/run_NNNN.asdf``.

ORACLE's own record of the run sits beside the header under one more
key, :data:`ORACLE_KEY`, which readers that look keys up by name pass
over: what the archive's keys do not hold and opening the run needs to
show it as it was taken (the scan type, the trigger timing sent to the
rack, the plots' view, the calibration gate's verdict), and what the
rack was (its identity, its dividers, the DAC's codes). It is written
with the header, before the first event, from settings that are locked
for the whole scan, so it holds for every event in the file. A reader
takes a run's own divider ratios (:func:`Recorded_Divider`) where it has
them, never today's settings.

A run's number is one more than the highest already in the data folder,
counted when the run starts, so numbers survive a restart and a refused
calibration leaves no gap. An existing file is never overwritten.

Beside each run, ``run_NNNN_monitor.csv`` records what the rack did while
it was taken, sampled by the window: the setpoint, the line meter's live
readback, the cooler, the wavemeter, and the loop's running counts of
bunches taken (the run file holds only those with hits), bunches dropped
while steps settled, and triggers it missed. Its last row is written as
the run closes. The frozen header holds those
once, at the start; the Scan Performance tab needs them over time. It is
a separate file so nothing that reads run files sees a change.

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
    """The start-of-run calibrations of the ``count`` runs numbered before
    ``path`` in ``directory`` that were calibrated as it was, newest first,
    as ``(run, CalSet, CalReadback, scan divider)``, the divider as the run
    recorded it or None: only those keys are read, never the events. A run
    that cannot be read is passed over.

    As it was means by the same DAQ under the same DAC correction (the
    DacCorrection text, which the old DAQ's runs do not have). Across
    either the two calibrations measure different things: the old DAQ
    paired some readings with the wrong setpoint (Mode B), and a correction
    moves every setpoint's platform voltage. Set against each other they
    would differ with nothing wrong in either."""
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
