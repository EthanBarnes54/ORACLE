"""The bench table: the platform voltage each DAC code really gives, measured slowly and kept.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The DAC is no straight line: codes are 0.305 V apart and its integral
non-linearity a few hundred mV off any line, each converter its own way.
A. Raggio's bench table (November 2026) measures it slowly, with 0 V
checks between, and each run is fitted to it, V = a + b*T
(:func:`Fit_Run`): 262 mV rms for a linear calibration became about 10 mV.

A table is built from a linearity record: each reading less the 0 V level
interpolated between the checks (it warms up by 0.2 V in the first
hour), averaged by converter and code, the codes being what the
converters held, as each run records them (DacCodes). A code not visited
is interpolated and said to be; nothing is extrapolated.

The library under ``<data>/tables`` keeps one CSV per table, its details
as ``#`` lines above the rows (the format proposed for cls_tools), named
by date and the first eight characters of its ID, the SHA-256 of every
number in it: never overwritten, so a run that names its table names
exactly what it was fitted to. The table in force is named in the
settings (``scan_table``); one from another rack is warned of, never
refused, since it is only applied after collection.

Depends on: numpy; cls_acquisition.hardware.mcc_dac.
"""

import hashlib
import json
import os
from datetime import datetime

import numpy as np

from cls_acquisition.hardware.mcc_dac import CH_NEG, CH_POS, CODES, FULL_SCALE_V, LSB_V

FOLDER = 'tables'
COLUMNS = ('channel', 'code', 'setpoint_v', 'volts', 'sigma_v', 'readings')
CHECK_RMS_V = 0.015                 # a table check's residual it passes at (to tune on the bench)
WARMING_V_PER_H = 0.05              # 0 V checks still drifting this fast: warming up
DRIFT_TOLERANCE_V = 0.010           # the 0 V drift worth a check: the table's own repeatability
REPEATABILITY_V = 0.010             # a table's day to day, once 0 V is taken out (Raggio)
STALE_FACTOR = 2.0                  # a run's residual this many times what is expected: stale
BRANCHES = ((CH_POS, 'positive'), (CH_NEG, 'negative'))


class Table_Error(ValueError):
    pass


class Table():
    """One table: ``Meta`` (its measurement's details, its ``id``), and for
    each converter its codes in order with their platform volts relative
    to the 0 V level, standard errors, readings and the setpoint that first
    wrote them."""

    def __init__(self, channels, meta):
        self.Channels = channels        # channel: dict of equal-length arrays, by code
        self.Meta = dict(meta)
        self.Meta['id'] = self.Meta.get('id') or _Hash(channels)
        self.Id = self.Meta['id']

    def Volts(self, channels, codes):
        """``(volts, exact)`` at each converter and code: the table's own
        where it measured that code, else interpolated between the nearest
        it measured either side on that converter (``exact`` False), NaN
        outside what it measured."""
        channels, codes = np.asarray(channels, dtype=int), np.asarray(codes, dtype=int)
        volts = np.full(len(codes), np.nan)
        exact = np.zeros(len(codes), dtype=bool)
        for channel, table in self.Channels.items():
            mine = channels == channel
            if not mine.any():
                continue
            known, inside = table['code'], (codes >= table['code'][0]) & (
                codes <= table['code'][-1])
            here = mine & inside
            volts[here] = np.interp(codes[here], known, table['volts'])
            exact[here] = np.isin(codes[here], known)
        return volts, exact

    def Rows(self):
        """Every row of the table's file, converter by converter."""
        return [(channel, int(t['code'][i]), float(t['setpoint_v'][i]), float(t['volts'][i]),
                 float(t['sigma_v'][i]), int(t['readings'][i]))
                for channel, t in sorted(self.Channels.items()) for i in range(len(t['code']))]


def _Hash(channels):
    """The table's ID: the SHA-256 of every number its file holds, the volts
    to the microvolt, so a file changed after it was written is found."""
    digest = hashlib.sha256()
    for channel in sorted(channels):
        table = channels[channel]
        digest.update(np.int64(channel).tobytes())
        for column in COLUMNS[1:]:
            values = np.asarray(table[column], dtype=float)
            scale = 1.0 if column in ('code', 'readings') else 1e6
            digest.update(np.rint(values * scale).astype(np.int64).tobytes())
    return digest.hexdigest()


def Codes(dac_volts):
    """The code each converter voltage was written as: mcc_dac.Quantised's
    rounding, for many at once."""
    return np.clip(np.rint((np.asarray(dac_volts) + FULL_SCALE_V) / LSB_V), 0,
                   CODES - 1).astype(np.int64)


def Zero_Level(points):
    """``(times, volts)`` of the 0 V checks among a sweep's ``points``."""
    zeros = [p for p in points if p['Kind'] == 'zero']
    return (np.array([p['Time_S'] for p in zeros]), np.array([p['Read_V'] for p in zeros]))


def Build(points, meta=None):
    """The table from a linearity sweep's ``points`` (hardware_calibration's):
    each sweep point less the 0 V level at its time, from the checks among
    them, averaged by converter and code. ``meta`` adds the measurement's
    details (its rack, dividers, settings); refuses a sweep with fewer
    than two 0 V checks, since its readings cannot be set against 0 V."""
    zero_t, zero_v = Zero_Level(points)
    if len(zero_t) < 2:
        raise Table_Error(f"The sweep took {len(zero_t)} checks of 0 V; a table needs two at "
                          f"least, to set its readings against the 0 V level as it drifts.")
    sweep = [p for p in points if p['Kind'] == 'sweep']
    if not sweep:
        raise Table_Error("The sweep holds no points to make a table from.")
    t = np.array([p['Time_S'] for p in sweep])
    volts = np.array([p['Read_V'] for p in sweep]) - np.interp(t, zero_t, zero_v)
    channel = np.array([p['Channel'] for p in sweep], dtype=int)
    code = Codes(np.array([p['Dac_V'] for p in sweep]))
    setpoint = np.array([p['Set_V'] for p in sweep])
    sigma = np.array([p['Sigma_V'] for p in sweep])
    channels = {}
    for c in np.unique(channel):
        mine = channel == c
        codes, first, inverse, counts = np.unique(code[mine], return_index=True,
                                                  return_inverse=True, return_counts=True)
        mean = np.bincount(inverse, volts[mine]) / counts
        spread = np.sqrt(np.maximum(np.bincount(inverse, volts[mine] ** 2) / counts - mean ** 2,
                                    0.0))
        # a code read once has its readings' own scatter; read again, theirs
        one = np.bincount(inverse, sigma[mine]) / counts
        error = np.where(counts > 1, spread / np.sqrt(np.maximum(counts - 1, 1)), one)
        # to the microvolt, as the file keeps them, so the ID is the same read back
        channels[int(c)] = {'code': codes, 'setpoint_v': np.round(setpoint[mine][first], 6),
                            'volts': np.round(mean, 6), 'sigma_v': np.round(error, 6),
                            'readings': counts}
    hours = (zero_t - zero_t[0]) / 3600.0
    last = hours >= hours[-1] - 1.0
    tail = np.polyfit(hours[last], zero_v[last], 1)[0] if last.sum() >= 3 else np.nan
    info = {'measured': f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            'span_v': [float(setpoint.min()), float(setpoint.max())],
            'points': len(sweep),
            'zero_v': {'first': float(zero_v[0]), 'last': float(zero_v[-1]),
                       'span': float(np.ptp(zero_v)), 'hours': float(hours[-1]),
                       'last_hour_v_per_h': float(tail),
                       'warming': bool(abs(tail) > WARMING_V_PER_H)},
            # its file keeps no 0 V checks, so it keeps what they said of the drift
            'check_interval_s': Check_Interval(zero_t, zero_v) if len(zero_t) >= 8 else None}
    info.update(meta or {})
    return Table(channels, info)


def Analysis(table, zero=None):
    """What a table says of the converters: per branch its gain and the
    residual about its straight line (rms, and peak to peak: the
    non-linearity, and the residual itself, against setpoint), and how
    often 0 V should be checked during a run (:func:`Check_Interval`, from
    its 0 V checks ``zero`` = (times, volts) if given, else as it kept)."""
    result = {'Branches': {}, 'Residuals': {}}
    for channel, name in BRANCHES:
        t = table.Channels.get(channel)
        if t is None or len(t['code']) < 3:
            continue
        gain, offset = np.polyfit(t['setpoint_v'], t['volts'], 1)
        residual = t['volts'] - (gain * t['setpoint_v'] + offset)
        result['Branches'][name] = {'Gain': float(gain), 'Codes': len(t['code']),
                                    'Rms_V': float(np.sqrt(np.mean(residual ** 2))),
                                    'Peak_To_Peak_V': float(np.ptp(residual))}
        result['Residuals'][channel] = (t['setpoint_v'], residual)
    result['Check_Interval_S'] = (Check_Interval(*zero) if zero is not None
                                  else table.Meta.get('check_interval_s'))
    return result


def Allan(times, volts):
    """``(taus, deviations)`` of the 0 V checks, overlapping, on a grid at
    their median spacing (they come between sweep points, not on a clock)."""
    step = float(np.median(np.diff(times)))
    grid = np.arange(times[0], times[-1], step)
    level = np.interp(grid, times, volts)
    taus, deviations = [], []
    m = 1
    while 2 * m < len(level):
        means = np.convolve(level, np.ones(m) / m, mode='valid')
        differences = means[m:] - means[:-m]
        taus.append(m * step)
        deviations.append(float(np.sqrt(0.5 * np.mean(differences ** 2))))
        m *= 2
    return np.array(taus), np.array(deviations)


def Check_Interval(times, volts):
    """The longest time between 0 V checks during a run over which the 0 V
    level moves by less than DRIFT_TOLERANCE_V (its Allan deviation), or
    None if the checks never stay within it."""
    taus, deviations = Allan(times, volts)
    within = taus[deviations <= DRIFT_TOLERANCE_V]
    return float(within.max()) if len(within) else None


def Fit(table, channels, codes, readback, sigma=None):
    """V = a + b*T: the run's calibration ``readback`` (platform volts) at
    each converter and code, fitted to the table, weighted by ``sigma`` if
    given. Returns ``A_V``, ``A_Err_V``, ``B``, ``B_Err``, ``Rms_V``,
    ``Chi2`` (per degree of freedom, with ``sigma``), ``Points`` and
    ``Interpolated`` (points at codes the table did not visit); raises
    Table_Error if too few points fall inside the table."""
    t, exact = table.Volts(channels, codes)
    readback = np.asarray(readback, dtype=float)
    inside = np.isfinite(t) & np.isfinite(readback)
    if inside.sum() < 3:
        raise Table_Error(f"Only {int(inside.sum())} of the calibration's {len(readback)} "
                          f"points fall inside the table {table.Id[:8]}; it does not cover "
                          f"this scan.")
    x, y = t[inside], readback[inside]
    weights = None if sigma is None else 1.0 / np.asarray(sigma, dtype=float)[inside]
    (b, a), cov = np.polyfit(x, y, 1, w=weights, cov='unscaled' if sigma is not None else True)
    residual = y - (a + b * x)
    dof = max(int(inside.sum()) - 2, 1)
    chi2 = (float(np.sum((residual * weights) ** 2) / dof) if weights is not None
            else None)
    return {'Table': table.Id, 'A_V': float(a), 'A_Err_V': float(np.sqrt(cov[1, 1])),
            'B': float(b), 'B_Err': float(np.sqrt(cov[0, 0])), 'Cov_AB': float(cov[0, 1]),
            'Rms_V': float(np.sqrt(np.mean(residual ** 2))), 'Chi2': chi2,
            'Points': int(inside.sum()), 'Interpolated': int(np.count_nonzero(~exact[inside])),
            'Reach_T_V': float(np.max(np.abs(x)))}


def Fit_Run(table, channels, codes, readback, noise_v):
    """A run's calibration fitted to the table, V = a + b*T (WP5), each point
    weighted by the meter's noise ``noise_v`` (platform volts) and the
    table's own repeatability: the fit (:func:`Fit`) with ``Expected_V``,
    the residual those two account for, and ``Verdict``, WARN when the
    residual is STALE_FACTOR times that, as the table no longer describes
    the converters (stale, or the hardware changed). Never a refusal: the
    table is applied after collection."""
    expected = float(np.hypot(noise_v, REPEATABILITY_V))
    fit = Fit(table, channels, codes, readback, np.full(len(readback), expected))
    stale = fit['Points'] >= 10 and fit['Rms_V'] > STALE_FACTOR * expected
    fit.update(Expected_V=expected, Verdict='WARN' if stale else 'PASS')
    return fit


def Summary(fit):
    """One line of a run's fit to the table, for the ELOG and the log."""
    return (f"a = {fit['A_V'] * 1e3:+.1f} ± {fit['A_Err_V'] * 1e3:.1f} mV, b - 1 = "
            f"{(fit['B'] - 1.0) * 1e6:+.1f} ± {fit['B_Err'] * 1e6:.1f} ppm, residual "
            f"{fit['Rms_V'] * 1e3:.1f} mV rms over {fit['Points']} points  {fit['Verdict']}")


def Check(table, channels, codes, readback):
    """``(verdict, fit)`` of a table check: PASS if a few hundred readings
    across the span sit within CHECK_RMS_V of the table once fitted, else
    WARN, since the table no longer describes the converters."""
    fit = Fit(table, channels, codes, readback)
    return ('PASS' if fit['Rms_V'] <= CHECK_RMS_V else 'WARN'), fit


def Compare(new, old):
    """Each converter's difference between two tables over the codes both
    measured, their mean taken out (the 0 V level each was set against):
    ``{branch: {'Codes', 'Rms_V', 'Max_V'}}``."""
    found = {}
    for channel, name in BRANCHES:
        a, b = new.Channels.get(channel), old.Channels.get(channel)
        if a is None or b is None:
            continue
        common, ia, ib = np.intersect1d(a['code'], b['code'], return_indices=True)
        if len(common) < 3:
            continue
        difference = a['volts'][ia] - b['volts'][ib]
        difference -= difference.mean()
        found[name] = {'Codes': len(common),
                       'Rms_V': float(np.sqrt(np.mean(difference ** 2))),
                       'Max_V': float(np.max(np.abs(difference)))}
    return found


def Matches(table, identity):
    """``(same, sentence)``: whether the table was measured on the rack
    ``identity`` names (its DAC board's unique ID, or its name and kind),
    and if not, why that matters."""
    theirs = (table.Meta.get('rack') or {})
    if not theirs:
        return False, (f"Table {table.Id[:8]} does not record the rack it was measured on, so "
                       f"whether it describes this one cannot be checked.")
    mine_dac, their_dac = identity.get('Dac') or {}, theirs.get('Dac') or {}
    if (theirs.get('Kind') == identity.get('Kind') and not their_dac.get('UniqueId')
            and their_dac.get('Board') and their_dac.get('Board') in str(mine_dac.get('Board'))):
        return False, (f"Table {table.Id[:8]} records its DAC board ({their_dac['Board']}) but "
                       f"not the board's unique ID, so whether it is this rack's board cannot "
                       f"be checked; the table check will say whether it still fits.")
    same = (theirs.get('Kind') == identity.get('Kind')
            and their_dac.get('Board') == mine_dac.get('Board')
            and their_dac.get('UniqueId') == mine_dac.get('UniqueId'))
    board = lambda dac: ' '.join(str(dac[k]) for k in ('Board', 'UniqueId') if dac.get(k))
    return same, ("" if same else
                  f"Table {table.Id[:8]} was measured on another rack (its DAC "
                  f"{board(their_dac) or 'unrecorded'}, here {board(mine_dac) or 'unknown'}): "
                  f"it is only applied after collection, but re-measure the table before "
                  f"trusting it.")


def Save(table, directory):
    """The table into the library under ``directory``; returns its path. A
    table already there is never written over: the same table is found
    where it is."""
    folder = os.path.join(directory, FOLDER)
    os.makedirs(folder, exist_ok=True)
    stamp = (table.Meta.get('measured') or f"{datetime.now():%Y-%m-%d}")[:10].replace('-', '')
    path = os.path.join(folder, f"table_{stamp}_{table.Id[:8]}.csv")
    if os.path.exists(path):
        try:
            same = Load(path).Id == table.Id
        except Table_Error as e:
            raise Table_Error(f"{e} Nothing was written over it.") from None
        if not same:
            raise Table_Error(f"{path} holds another table; nothing was written over it.")
        return path
    with open(path, 'x', encoding='utf-8', newline='') as fh:
        fh.write("# ORACLE bench table: platform volts relative to the 0 V level, by DAC "
                 "converter and code\n")
        for key, value in table.Meta.items():
            fh.write(f"# {key}: {json.dumps(value)}\n")
        fh.write(','.join(COLUMNS) + '\n')
        for row in table.Rows():
            fh.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]}\n")
    return path


def Load(path):
    """The table a library file holds; raises Table_Error naming what is
    wrong with one that is not a table."""
    meta, rows = {}, []
    with open(path, encoding='utf-8', newline='') as fh:
        lines = fh.read().splitlines()
    for line in lines:
        if line.startswith('# ') and ': ' in line:
            key, _, value = line[2:].partition(': ')
            try:
                meta[key] = json.loads(value)
            except ValueError:
                continue
    body = [line for line in lines if line and not line.startswith('#')]
    if not body or tuple(body[0].split(',')) != COLUMNS:
        raise Table_Error(f"{path} is not a bench table: its columns are not "
                          f"{', '.join(COLUMNS)}.")
    try:
        rows = np.array([[float(v) for v in line.split(',')] for line in body[1:]])
    except ValueError as e:
        raise Table_Error(f"{path} holds a row that is not numbers: {e}.") from None
    channels = {}
    for c in np.unique(rows[:, 0]).astype(int):
        mine = rows[rows[:, 0] == c]
        channels[int(c)] = {'code': mine[:, 1].astype(np.int64), 'setpoint_v': mine[:, 2],
                            'volts': mine[:, 3], 'sigma_v': mine[:, 4],
                            'readings': mine[:, 5].astype(int)}
    table = Table(channels, meta)
    if table.Id != _Hash(channels):
        raise Table_Error(f"{path}'s numbers do not match its ID {table.Id[:8]}: the file was "
                          f"changed after it was written.")
    return table


def Library(directory):
    """``(path, meta)`` of every table under ``directory``, newest first."""
    folder = os.path.join(directory, FOLDER)
    if not os.path.isdir(folder):
        return []
    found = []
    for name in sorted(os.listdir(folder), reverse=True):
        if name.startswith('table_') and name.endswith('.csv'):
            try:
                found.append((os.path.join(folder, name), Load(os.path.join(folder, name)).Meta))
            except (OSError, Table_Error):
                continue
    return found


def Suggest(directory, identity):
    """The newest table in the library measured on the rack ``identity``
    names, or None."""
    for path, meta in Library(directory):
        if Matches(Table({}, meta), identity)[0]:
            return path
    return None
