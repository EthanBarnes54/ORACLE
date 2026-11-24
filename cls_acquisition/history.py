"""The rack's history (WP10): every bench record and every run on one
timeline, marked where the rack changed.

Date:    2026-11-24
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

It shows whether a hardware change worked: a new Trek or DAC in the settle
records, the bench tables and their checks; the dividers and the DAC in each
run's 0 V level and gain against the table; a cooler meter tied to the bunch
or a filtered supply in each run's cooler scatter; a PMT's HV, threshold or
replacement in its afterpulsing. It reads only what is kept and measures
nothing again: the bench records' names and purposes, the health checks and
the tables in the library; each run's ORACLE record (its fit to the bench
table, its rack) and the reports written beside it once its analysis ended.

Depends on: asdf, numpy, PyYAML; cls_acquisition.hardware, health, run_file,
scan_table.
"""

import os
import re
from datetime import datetime

import asdf
import numpy as np
import yaml

from cls_acquisition import health, run_file, scan_table
from cls_acquisition.hardware import Changes

BENCH = {'settle': 'Settle time', 'linearity': 'Linearity sweep', 'cooler': 'Cooler monitor'}
PURPOSES = {'bench table': 'Bench table sweep', 'table check': 'Bench table check'}
_NAME = re.compile(r'([a-z_]+)_(\d{4}-\d\d-\d\d_\d{6})\.(csv|yaml)$')


def _Header(path):
    """A bench record's settings, its ``#`` lines above the columns."""
    header = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if not line.startswith('# '):
                break
            key, _, value = line[2:].rstrip('\r\n').partition(': ')
            header[key] = value
    return header


def _Yaml(path):
    """A report beside a run, or {} where there is none."""
    try:
        with open(path, encoding='utf-8') as fh:
            return yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return {}


def Bench(directory):
    """``[(time, what), ...]``, oldest first: each bench record (not their
    companions), health check with its verdict, and table in the library."""
    found = []
    folder = os.path.join(directory, 'hardware_calibration')
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else ():
        match = _NAME.match(name)
        if not match or match.group(1) not in (*BENCH, 'health'):
            continue
        kind, path = match.group(1), os.path.join(folder, name)
        when = datetime.strptime(match.group(2), '%Y-%m-%d_%H%M%S').timestamp()
        if kind == 'health':
            verdicts = (_Yaml(path).get('Verdicts') or {}).values()
            what = f"Health check, {max(verdicts, key=health.RANK.get, default=health.NA)}"
        elif kind == 'linearity':           # a table's sweep or check, or one of its own
            what = PURPOSES.get(_Header(path).get('Purpose'), BENCH[kind])
        else:
            what = BENCH[kind]
        found.append((when, what))
    for _, meta in scan_table.Library(directory):
        when = datetime.strptime(str(meta.get('measured'))[:19], '%Y-%m-%d %H:%M:%S')
        found.append((when.timestamp(), f"Bench table {meta['id'][:8]} made"))
    return sorted(found)


def Runs(directory):
    """Each run, oldest first: ``Run``, ``Time``, ``A_V`` (its 0 V level
    against the bench table), ``B_Ppm`` (b - 1, where its calibration was
    wide enough to fit b) and ``Cooler_V`` (its readings' scatter about the
    analysis' mean), as arrays with NaN where a run holds none; ``Afterpulse``,
    ``{channel: (times, shares)}``; ``Racks``, ``[(time, run, changes)]`` at
    each run whose rack differs from the run before's."""
    rows, shares, racks, last = [], {}, [], None
    for path in reversed(run_file.Runs(directory)):
        try:
            with asdf.open(path, memmap=False) as af:
                when, record = af.tree['Date'], dict(af.tree.get(run_file.ORACLE_KEY) or {})
        except (OSError, KeyError, ValueError, TypeError):
            continue
        run, t = int(os.path.basename(path)[4:-len('.asdf')]), when.timestamp()
        fit = record.get('TableFit') if isinstance(record.get('TableFit'), dict) else {}
        free = 'B' in fit and fit.get('Scale_Mode', 'free') == 'free'
        base = path[:-len('.asdf')]
        cooler = next((s['volts'] for s in _Yaml(f"{base}_systematics.yaml").get('sources', [])
                       if s.get('id') == 'cooler_scatter' and s.get('volts') is not None), np.nan)
        rows.append((run, t, fit.get('A_V', np.nan),
                     (fit['B'] - 1.0) * 1e6 if free else np.nan, cooler))
        for key, entry in (_Yaml(f"{base}_performance.yaml").get('detectors') or {}).items():
            shares.setdefault(int(key.split()[-1]), []).append((t, entry['afterpulse_share']))
        identity = record.get('RackIdentity')
        changed = Changes(last, identity) if identity and last else []
        if changed:
            racks.append((t, run, changed))
        last = identity or last
    columns = np.array(rows, dtype=float).reshape(-1, 5).T
    return {**dict(zip(('Run', 'Time', 'A_V', 'B_Ppm', 'Cooler_V'), columns)),
            'Afterpulse': {c: tuple(np.array(v).T) for c, v in sorted(shares.items())},
            'Racks': racks}


def Collect(directory):
    """Everything the history page shows: :func:`Runs`, and ``Bench``."""
    return dict(Runs(directory), Bench=Bench(directory))
