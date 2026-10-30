"""Scan performance on synthetic runs whose faults are known, and the loop's
settle-aware stepping that the analysis relies on.

Each run is built as the rack would take it: one line, 4 V wide, that the
cooler, the laser (through the Doppler shift) and the HV's error move as
they move the real one, hits drawn bunch by bunch, and a monitor record
sampled every 0.1 s. A clean run must pass, an ultra-low-rate one must
raise no false alarm, and each planted fault must be found, flagged and
never cut."""

import time

import numpy as np
import pytest
import yaml

from cls_acquisition import scan_performance as sp
from cls_acquisition.calibration import Settle_For, THRESHOLDS
from cls_acquisition.hardware_calibration import Settle_Model
from cls_acquisition.hardware.keysight import COOLER_DIVIDER
from cls_acquisition.run_file import ORACLE_KEY

LASER = {'harmonic': 4, 'geometry': 'anti-collinear', 'charge': 1}
PLAN = np.arange(0.0, 41.0)
COOLER_V, WAVENUMBER, MASS = 29950.0, 16956.17, 59.0
PER_HZ = sp.Volts_Per_Hz(COOLER_V - PLAN.mean(), WAVENUMBER * sp.C_LIGHT * 100, MASS, LASER)


def _Flat(value):
    return lambda t: np.full_like(np.asarray(t, dtype=float), value)


def _Run(plan=PLAN, ranges=None, per_step=4, scans=8, period=0.1, kind='Sawtooth', legacy=False,
         origin=0, rate=_Flat(1.0), shift=_Flat(0.0), signal=6.0, background=0.3,
         cooler=_Flat(COOLER_V), laser_cm=_Flat(WAVENUMBER), hv_error=_Flat(0.0),
         calibration=lambda v: v, channels=None, tof=None, gate=(48.0, 56.0), bunched=True,
         pause=None, parted_at=None, seed=3):
    """``(header, raw, monitor)`` of a run with one line at 20 V, 4 V wide.
    The line sits where its ions resonate: a cooler higher by a volt moves it
    a volt up the scan, the laser by PER_HZ a hertz, and an HV that puts out
    more than it is told moves it down by as much."""
    rng = np.random.default_rng(seed)
    n = len(plan)
    bunches = np.arange(scans * n * per_step)
    volts = plan[sp._Step_Index(bunches // per_step, n, kind)]
    t = bunches * period
    if pause:
        t = t + np.where(bunches >= pause[0], pause[1], 0.0)
    centre = (20.0 + shift(t) + (cooler(t) - COOLER_V)
              + (laser_cm(t) - WAVENUMBER) * sp.C_LIGHT * 100 * PER_HZ - hv_error(t))
    mean = rate(t) * (background + signal / (1.0 + ((volts - centre) / 2.0) ** 2))
    hits = rng.poisson(mean)
    times = np.repeat(t, hits)
    count = len(times)
    numbers = np.repeat(bunches, hits) + origin
    if parted_at is not None:          # the old DAQ's D5: the tagger's count runs on
        numbers = numbers + np.where(numbers >= parted_at, 7, 0)
    raw = np.column_stack([
        np.round(times, 2), np.repeat(volts, hits), numbers,
        channels(times, count, rng) if channels else rng.integers(1, 5, count),
        tof(times, count, rng) if tof else rng.normal(52.0, 1.0, count),
        np.repeat(cooler(t), hits) / COOLER_DIVIDER])
    header = {'ScanningRanges': ranges or [[float(plan.min()), float(plan.max())]],
              'StepSize': float(plan[1] - plan[0]) if len(plan) > 1 else 1.0,
              'BunchesPerChannel': per_step, 'DwellTime': period * 1e3, 'MassAMU': MASS,
              'CoolerVoltage': COOLER_V / COOLER_DIVIDER, 'LaserSetpoint': WAVENUMBER,
              'CalSet': np.unique(plan), 'CalReadback': calibration(np.unique(plan)) / 1000.0,
              'DacCorrectionApplied': 0}
    if legacy:
        return header, raw, None
    header[ORACLE_KEY] = {
        'ScanType': kind, 'SettleModel': None, 'Laser': dict(LASER), 'Channels': [1, 2, 3, 4],
        'Timing': {'bunched': bunched, 'rep_rate_hz': 1.0 / period, 'gate_start_us': 0.0,
                   'gate_stop_us': 200.0},
        'View': {'tof_gate_us': list(gate) if gate else None}}
    sampled = np.arange(0.0, t[-1] + 0.1, 0.1)
    taken = np.searchsorted(t, sampled, side='right')
    setpoint = volts[np.clip(taken - 1, 0, len(volts) - 1)]
    monitor = {'time': sampled, 'round': np.zeros_like(sampled), 'setpoint_v': setpoint,
               'readback_v': calibration(setpoint) + hv_error(sampled),
               'cooler_v': cooler(sampled), 'wavenumber_cm1': laser_cm(sampled),
               'bunches': taken.astype(float), 'dropped': np.zeros_like(sampled),
               'skipped': np.zeros_like(sampled)}
    return header, raw, monitor


def _Check(result, name):
    return next(c for checks in result['Sections'].values() for c in checks
                if c['Check'] == name)


def _Verdicts(result):
    return {c['Check']: c['Verdict'] for checks in result['Sections'].values() for c in checks}


def _Messages(result):
    return ' '.join(c['Message'] for checks in result['Sections'].values() for c in checks)


# ── A run with nothing wrong, at any count rate ─────────────────

def test_A_Clean_Run_Passes_Every_Check_And_Flags_Nothing():
    result = sp.Analyse(*_Run(), LASER)
    assert set(_Verdicts(result).values()) == {sp.PASS}, _Verdicts(result)
    assert result['Verdict'] == sp.PASS and result['Flagged'] == []
    assert 'sawtooth' in result['Summary'] and '8 complete' in result['Summary']
    # the line's own width is the scale, and in MHz at the ions
    assert result['Budget'].startswith('The line is') and 'MHz' in result['Budget']
    assert 'Cuts' not in result


def test_An_Ultra_Low_Rate_Run_Raises_No_False_Alarm():
    """A hit every ten seconds: the bunch count, not the hits, says whether
    the loop stood idle, and the beam is judged over periods long enough to
    expect something."""
    header, raw, monitor = _Run(per_step=1, scans=200, period=0.2, background=0.01, signal=0.1,
                                kind='Triangle')
    assert len(raw) < 400 and np.diff(np.sort(raw[:, 0])).max() > 30
    result = sp.Analyse(header, raw, monitor, LASER)
    verdicts = _Verdicts(result)
    for name in ('Pauses', 'Read stalls', 'Beam losses', 'Beam stability', 'Voltage tags'):
        assert verdicts[name] == sp.PASS, (name, _Check(result, name)['Message'])
    # one bunch a step leaves nothing to time within a step: said, never 'nan ms'
    assert verdicts['Keeping up'] == sp.NA
    assert 'one bunch per step' in _Check(result, 'Keeping up')['Message'].lower()
    assert 'nan' not in _Messages(result) and result['Verdict'] != sp.FAIL


def test_A_Real_Pause_Is_Found_At_Any_Rate():
    header, raw, monitor = _Run(per_step=1, scans=200, period=0.2, background=0.01, signal=0.1,
                                pause=(4000, 60.0))
    check = _Check(sp.Analyse(header, raw, monitor, LASER), 'Pauses')
    assert check['Verdict'] == sp.WARN and check['Value'] == 1, check['Message']


def test_A_Still_Line_Is_Not_Reported_As_Moving():
    """The line's position errors come from the run's own counts dealt at
    random into periods, so a line that never moved is seldom said to."""
    alarms = sum(_Check(sp.Analyse(*_Run(scans=12, signal=3.0, seed=seed), LASER),
                        'Line position')['Verdict'] != sp.PASS for seed in range(20))
    assert alarms <= 2


# ── The beam ────────────────────────────────────────────────────

def test_A_Beam_Loss_Is_Flagged_With_Its_Time_And_Nothing_Is_Cut(tmp_path):
    result = sp.Analyse(*_Run(rate=lambda t: np.where((t > 50) & (t < 80), 0.1, 1.0)), LASER)
    assert _Check(result, 'Beam losses')['Verdict'] == sp.WARN
    [flag] = [f for f in result['Flagged'] if f.startswith('Beam below half')]
    assert 'min into the run' in flag and 'scan' in flag
    t, level, lost = result['Series']['Beam']
    assert lost.any() and np.all((t[lost] > 40) & (t[lost] < 90))
    path = str(tmp_path / 'run_0007.asdf')
    with open(sp.Export(path, result), encoding='utf-8') as fh:
        report = yaml.safe_load(fh)
    assert 'scan_filters' not in report and flag in report['flagged']
    assert report['verdict'] == result['Verdict'] and report['run'] == 'run_0007.asdf'


def test_A_Stretch_Of_Background_Is_No_Beam_Loss():
    """A slow scan spends minutes off the line: a period expects what its own
    setpoints give, so its few hits there are no loss."""
    plan = np.arange(0.0, 200.0)
    header, raw, monitor = _Run(plan=plan, per_step=1, scans=4, period=1.0, background=0.05,
                                shift=_Flat(80.0))
    result = sp.Analyse(header, raw, monitor, LASER)
    assert _Check(result, 'Beam losses')['Verdict'] == sp.PASS, _Check(result,
                                                                       'Beam losses')['Message']


# ── What moved the line ─────────────────────────────────────────

def test_A_Laser_Drift_Moves_The_Line_And_The_Monitors_Explain_It():
    span = 8 * len(PLAN) * 4 * 0.1
    drift_cm = 2.0 / (PER_HZ * sp.C_LIGHT * 100)            # 2 V of line, half its width
    result = sp.Analyse(*_Run(laser_cm=lambda t: WAVENUMBER + drift_cm * t / span), LASER)
    position = _Check(result, 'Line position')
    assert position['Verdict'] != sp.PASS
    assert position['Value'] == pytest.approx(2.0, abs=0.6)
    assert _Check(result, 'Laser')['Verdict'] != sp.PASS
    assert _Check(result, 'What moved the line')['Verdict'] == sp.PASS


def test_A_Movement_No_Monitor_Saw_Is_Reported_As_Unexplained():
    span = 8 * len(PLAN) * 4 * 0.1
    result = sp.Analyse(*_Run(shift=lambda t: 2.0 * t / span), LASER)
    assert _Check(result, 'Line position')['Verdict'] != sp.PASS
    assert _Check(result, 'What moved the line')['Verdict'] == sp.WARN
    assert any('unexplained' in f for f in result['Flagged'])


def test_The_Cooler_Is_Corrected_Hit_By_Hit_As_DENIS_Corrects_It():
    """The cooler moves the line on the scan's axis, but every hit carries its
    reading, so on DENIS's axis the line stands still."""
    span = 8 * len(PLAN) * 4 * 0.1
    result = sp.Analyse(*_Run(cooler=lambda t: COOLER_V + 3.0 * t / span), LASER)
    assert _Check(result, 'Line position')['Verdict'] == sp.PASS
    assert _Check(result, 'Cooler')['Verdict'] == sp.PASS


def test_A_Hit_Without_A_Cooler_Reading_Is_Flagged():
    header, raw, monitor = _Run()
    raw[:50, 5] = np.nan
    result = sp.Analyse(header, raw, monitor, LASER)
    assert _Check(result, 'Cooler')['Verdict'] == sp.WARN
    assert any('no cooler reading' in f for f in result['Flagged'])


def test_The_HV_Is_Judged_Against_The_Calibrations_Own_Reading():
    """The converter's non-linearity, which the calibration holds, is not
    drift; a real drift after the calibration is, and it explains the line's
    movement."""
    inl = lambda v: v + 0.3 * np.sin(v)                         # noqa: E731
    still = sp.Analyse(*_Run(calibration=inl), LASER)
    assert _Check(still, 'HV against its calibration')['Value'] < 0.02
    span = 8 * len(PLAN) * 4 * 0.1
    drifting = sp.Analyse(*_Run(calibration=inl, hv_error=lambda t: 2.0 * t / span), LASER)
    assert _Check(drifting, 'HV against its calibration')['Verdict'] != sp.PASS
    assert _Check(drifting, 'What moved the line')['Verdict'] == sp.PASS


def test_The_Run_Records_Its_Own_Laser_Settings():
    header, raw, monitor = _Run()
    header[ORACLE_KEY]['Laser'] = {'harmonic': 2, 'geometry': 'collinear', 'charge': 1}
    message = _Check(sp.Analyse(header, raw, monitor, LASER), 'Laser')['Message']
    assert 'harmonic 2, collinear' in message and "today's settings" not in message
    del header[ORACLE_KEY]['Laser']
    message = _Check(sp.Analyse(header, raw, monitor, LASER), 'Laser')['Message']
    assert 'harmonic 4' in message and "today's settings" in message


# ── The calibration ─────────────────────────────────────────────

def test_A_Calibration_Is_Judged_As_DENIS_Fits_It():
    """A first point taken while the HV settled stands out under DENIS's rule,
    and leaving it out moves the axis enough to flag."""
    settling = lambda v: v + np.where(v == 0.0, 3.0, 0.0)      # noqa: E731
    result = sp.Analyse(*_Run(calibration=settling), LASER)
    check = _Check(result, 'Calibration')
    assert check['Verdict'] != sp.PASS and 'stand out' in check['Message']
    assert any(f.startswith('Calibration points at 0 V') for f in result['Flagged'])


def test_A_Calibration_Is_Set_Against_The_Runs_Before_It():
    header, raw, monitor = _Run()
    same = [(6, PLAN, PLAN / 1000.0)]
    moved = [(6, PLAN, (PLAN + 2.0) / 1000.0), (5, PLAN, (PLAN + 2.1) / 1000.0)]
    agreed = _Check(sp.Analyse(header, raw, monitor, LASER, history=same),
                    'Calibration against earlier runs')
    assert agreed['Verdict'] == sp.PASS and 'run 6' in agreed['Message']
    apart = _Check(sp.Analyse(header, raw, monitor, LASER, history=moved),
                   'Calibration against earlier runs')
    assert apart['Verdict'] == sp.WARN and apart['Value'] == pytest.approx(2.0, abs=0.01)
    assert 'isotope shift' in apart['Message']


# ── The detectors and the time of flight ────────────────────────

def test_A_Silent_Or_Failing_Channel_Is_Reported_As_A_Possible_Fault():
    silent = sp.Analyse(*_Run(channels=lambda t, n, rng: rng.choice([1, 3, 4], n)), LASER)
    check = _Check(silent, 'Channels')
    assert check['Verdict'] == sp.WARN and 'Channel 2 recorded no hits' in check['Message']
    half = 8 * len(PLAN) * 4 * 0.1 / 2
    failing = sp.Analyse(*_Run(channels=lambda t, n, rng: np.where(
        t < half, rng.integers(1, 5, n), rng.choice([1, 2, 4], n))), LASER)
    assert "Channel 3's counts fell" in _Check(failing, 'Channels')['Message']


def test_The_TOF_Peak_Is_Followed_Against_Its_Gate():
    span = 8 * len(PLAN) * 4 * 0.1
    moving = sp.Analyse(*_Run(tof=lambda t, n, rng: rng.normal(52.0 + 6.0 * t / span, 1.0)),
                        LASER)
    check = _Check(moving, 'Time of flight')
    assert check['Verdict'] == sp.WARN and 'moved against the gate' in check['Message']
    dc = sp.Analyse(*_Run(bunched=False), LASER)
    assert _Check(dc, 'Time of flight')['Verdict'] == sp.NA


# ── The old DAQ's runs ──────────────────────────────────────────

def test_An_Old_Run_That_Waited_Long_Enough_Took_No_Bunch_Unsettled():
    """One bunch every 200 ms, and SETTLE is 50 ms: none was triggered before
    its step settled (the analysis once counted one, and failed every run)."""
    header, raw, _ = _Run(per_step=1, scans=40, period=0.2, legacy=True, origin=1)
    result = sp.Analyse(header, raw, None, LASER)
    check = _Check(result, 'Bunches before settling')
    assert check['Verdict'] == sp.PASS and check['Value'] == 0
    assert _Check(result, 'Voltage tags')['Verdict'] == sp.PASS
    assert _Check(result, 'Laser')['Verdict'] == sp.NA


def test_An_Old_Fast_Run_Is_Judged_By_The_Settle_Model():
    header, raw, _ = _Run(per_step=20, scans=4, period=0.001, legacy=True)
    blunt = _Check(sp.Analyse(header, raw, None, LASER), 'Bunches before settling')
    assert blunt['Verdict'] == sp.FAIL and 'no settle model is set' in blunt['Message']
    model = {'points': [[1.0, 0.0005], [40.0, 0.008]]}
    fine = _Check(sp.Analyse(header, raw, None, LASER, settle_model=model),
                  'Bunches before settling')
    assert fine['Value'] < blunt['Value'] and 'the settle model' in fine['Message']


def test_An_Old_Run_Whose_Count_Parted_Is_Warned_Not_Failed():
    """The old DAQ's D5: which of the bunch count and the voltage is wrong
    cannot be told, so nothing is cut and nothing is judged scan by scan."""
    header, raw, _ = _Run(legacy=True, parted_at=600)
    result = sp.Analyse(header, raw, None, LASER)
    assert _Check(result, 'Voltage tags')['Verdict'] == sp.WARN
    assert _Check(result, 'Line position')['Verdict'] == sp.NA
    assert any('Scan numbers are unreliable' in f for f in result['Flagged'])
    assert 'Rates' not in result['Series']


def test_Overlapping_Ranges_Are_Analysed_As_One_Range():
    """The old DAQ allowed them, and commanded each range's ladder in turn."""
    plan = np.r_[np.arange(0.0, 21.0), np.arange(10.0, 41.0)]
    header, raw, _ = _Run(plan=plan, ranges=[[0.0, 20.0], [10.0, 40.0]], legacy=True)
    result = sp.Analyse(header, raw, None, LASER)
    assert 'overlap' in result['Summary'] and '0 to 40 V' in result['Summary']
    assert _Check(result, 'Voltage tags')['Verdict'] == sp.PASS


def test_A_Bunch_Is_Drawn_Once_However_Many_Hits_It_Holds():
    """Its hits share its time and readings, so its points coincide: the plot
    gets each once, the checks still weigh every hit."""
    header, raw, _ = _Run(legacy=True)
    result = sp.Analyse(header, raw, None, LASER)
    bunches = len(np.unique(raw[:, 2]))
    for name in ('Latency', 'Cooler'):
        times, values = result['Series'][name]
        assert len(times) == len(values) == bunches < len(raw), name


# ── The pieces ──────────────────────────────────────────────────

def test_Excess_Is_Judged_Against_Its_Degrees_Of_Freedom():
    assert not sp._Excess(2.15, 4)
    assert sp._Excess(3.0, 50) and not sp._Excess(1.0, 50)


def test_Poisson_Tails_Are_Exact_Where_It_Matters():
    assert sp._At_Most(0, 10.0) == pytest.approx(np.exp(-10.0))
    assert sp._At_Most(3, 2.0) == pytest.approx(0.857123, rel=1e-5)
    assert sp._At_Most(40, 60.0) == pytest.approx(0.0039828, rel=0.01)  # the gamma form


def test_The_Laser_Converts_As_The_Doppler_Shift_Says():
    beam_v, mass, nu = 30000.0, 59.0, 16956.17 * sp.C_LIGHT * 100
    beta = np.sqrt(2 * beam_v / (mass * sp.AMU_EV))
    # first order: the ion sees h nu, moving h nu beta / 2U per volt and h per
    # laser hertz, so the harmonic cancels; a head-on laser moves the line up
    expected = 2 * beam_v / (nu * beta)
    assert sp.Volts_Per_Hz(beam_v, nu, mass, LASER) == pytest.approx(expected, rel=2e-3)
    collinear = dict(LASER, geometry='collinear')
    assert sp.Volts_Per_Hz(beam_v, nu, mass, collinear) == pytest.approx(-expected, rel=2e-3)


# ── The loop's settle-aware stepping ────────────────────────────

def test_The_Settle_Model_Grows_With_The_Jump():
    results = [{'From_V': 0, 'To_V': 10, 'Lag_S': 0.001, 'Settle_S': 0.01},
               {'From_V': 0, 'To_V': 1000, 'Lag_S': 0.001, 'Settle_S': 0.05},
               {'From_V': 1000, 'To_V': 0, 'Lag_S': 0.001, 'Settle_S': 0.07},
               {'From_V': -500, 'To_V': 500, 'Lag_S': 0.001, 'Settle_S': 0.04}]
    model = Settle_Model(results, today='2026-10-25')
    # the slowest at each size, never falling as the jump grows
    assert np.array(model['points']) == pytest.approx(np.array([[10, 0.011], [1000, 0.071]]))
    assert Settle_For(None, 5000) == THRESHOLDS['SettleS']
    assert Settle_For(model, 1) == pytest.approx(0.011 * 1.25)
    assert Settle_For(model, 2000) > Settle_For(model, 1000) > Settle_For(model, 500)


def test_A_Large_Jump_Holds_The_Window_Shut_For_Longer(loop):
    model = {'points': [[1.0, 0.0], [2000.0, 0.4]]}
    timing = {'bunched': True, 'rep_rate_hz': 200.0, 'release_us': 100, 'tagger_delay_us': 0,
              'gate_start_us': 0.0, 'gate_stop_us': 200.0}
    start = time.monotonic()
    assert loop.Send('Start_Scan', setpoints=[0.0, 2000.0], scan_type='Single scan',
                     bunches_per_step=1, timing=timing, settle=model)[0]
    packets = loop.Packets(until_end=True)
    assert time.monotonic() - start > 0.4 * 1.25           # the 2 kV step waited its settling
    rows = np.vstack([p[1] for p in packets[:-1]] or [np.empty((0, 6))])
    # the loop's own count: exactly one bunch a step, from 0
    assert set(rows[:, 2]) <= {0.0, 1.0}
    assert all(rows[rows[:, 2] == 1, 1] == 2000.0)


def test_A_Missed_Trigger_Is_Counted(loop):
    loop.Loop._Check_Count(10, 0)
    loop.Loop._Check_Count(13, 1)                           # the tagger moved on 3, we on 1
    assert loop.State.Skipped.value == 2
