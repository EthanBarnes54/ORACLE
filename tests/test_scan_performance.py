"""Scan performance on synthetic runs whose faults are known, and the loop's
settle-aware stepping that the analysis relies on."""

import time

import numpy as np
import pytest
import yaml

from cls_acquisition import scan_performance as sp
from cls_acquisition.calibration import Settle_For, THRESHOLDS
from cls_acquisition.hardware_calibration import Settle_Model
from cls_acquisition.hardware.keysight import COOLER_DIVIDER

LASER = {'harmonic': 4, 'geometry': 'anti-collinear', 'charge': 1}
PLAN = np.arange(0.0, 11.0)
PER_STEP, SCANS, PERIOD = 4, 6, 0.01


def _Run(peak_shift=lambda scan: 0.0, origin=0, rate=lambda scan: 1.0, scans=SCANS):
    """A sawtooth run with one peak at 5 V, one row per hit, and a monitor
    record from a settle-aware loop."""
    rng = np.random.default_rng(3)
    rows = []
    for bunch in range(scans * len(PLAN) * PER_STEP):
        scan, step = divmod(bunch // PER_STEP, len(PLAN))
        volts = PLAN[step]
        peak = np.exp(-0.5 * (volts - 5.0 - peak_shift(scan)) ** 2)     # 1 V wide
        mean = rate(scan) * (0.3 + 6.0 * peak)
        for _ in range(rng.poisson(mean)):
            rows.append([bunch * PERIOD, volts, bunch + origin, rng.integers(1, 5),
                         rng.normal(52.0, 1.5), 29950.0 / COOLER_DIVIDER])
    raw = np.array(rows)
    header = {'ScanningRanges': [[0.0, 10.0]], 'StepSize': 1.0, 'BunchesPerChannel': PER_STEP,
              'DwellTime': PERIOD * 1e3, 'MassAMU': 59.0, 'CoolerVoltage': 2.995,
              'CalSet': PLAN, 'CalReadback': PLAN / 1000.0, 'DacCorrectionApplied': 0}
    t = np.arange(0.0, raw[:, 0].max(), 0.1)
    monitor = {'time': t, 'setpoint_v': np.zeros_like(t), 'readback_v': np.zeros_like(t),
               'cooler_v': np.full_like(t, 29950.0), 'wavenumber_cm1': np.full_like(t, 16956.17),
               'bunches': np.full_like(t, scans * len(PLAN) * PER_STEP),
               'dropped': np.zeros_like(t), 'skipped': np.zeros_like(t)}
    return header, raw, monitor


def _Check(result, name):
    return next(c for checks in result['Sections'].values() for c in checks
                if c['Check'] == name)


def test_A_Clean_Run_Passes_And_Is_Read_As_It_Was_Taken():
    result = sp.Analyse(*_Run(), LASER)
    assert 'sawtooth' in result['Summary'] and f"{SCANS} complete" in result['Summary']
    assert result['Verdict'] != sp.FAIL and result['Cuts'] == {}
    for name in ('Tags against bunches', 'Peak drift', 'Beam losses', 'Bunches before settling'):
        assert _Check(result, name)['Verdict'] == sp.PASS, name


def test_A_Mistagged_Step_Fails_And_Its_Scan_Is_Cut():
    header, raw, monitor = _Run()
    wrong = (raw[:, 2] // PER_STEP == len(PLAN) * 2 + 3)       # scan 2, step 3
    raw[wrong, 1] = PLAN[4]
    result = sp.Analyse(header, raw, monitor, LASER)
    assert _Check(result, 'Tags against bunches')['Verdict'] == sp.FAIL
    assert 'hits at the wrong voltage' in result['Cuts'][2]


def test_A_Beam_Loss_Is_Cut_And_A_Drifting_Peak_Fails():
    result = sp.Analyse(*_Run(rate=lambda scan: 0.2 if scan == 3 else 1.0), LASER)
    assert result['Cuts'][3] == ['beam lost: rate below 50% of the median']
    drifting = sp.Analyse(*_Run(peak_shift=lambda scan: 0.5 * scan), LASER)
    assert _Check(drifting, 'Peak drift')['Verdict'] == sp.FAIL


def test_An_Old_Run_Gets_The_Latency_Estimate_And_Its_Bunch_Origin_Is_Found():
    header, raw, _ = _Run(origin=1)
    result = sp.Analyse(header, raw, None, LASER)
    assert _Check(result, 'Tags against bunches')['Verdict'] == sp.PASS
    # SETTLE spans five 10 ms bunches: every one of the four in a step
    assert result['Leading_Bunches'] == PER_STEP
    assert _Check(result, 'Laser')['Verdict'] == sp.NA


def test_A_Pause_Is_Flagged():
    header, raw, monitor = _Run()
    raw[raw[:, 2] >= 100, 0] += 60.0
    assert _Check(sp.Analyse(header, raw, monitor, LASER), 'Pauses')['Verdict'] == sp.WARN


def test_A_Bunch_Is_Drawn_Once_However_Many_Hits_It_Holds():
    """Its hits share its time and readings, so its points coincide: the plot
    gets each once, the checks still weigh every hit."""
    header, raw, _ = _Run()
    result = sp.Analyse(header, raw, None, LASER)
    bunches = len(np.unique(raw[:, 2]))
    for name in ('Latency', 'Cooler'):
        times, values = result['Series'][name]
        assert len(times) == len(values) == bunches < len(raw), name
    assert _Check(result, 'Tags against bunches')['Message'].startswith(f"0 of {len(raw)} hits")


def test_Hits_Are_Grouped_By_Scan_As_A_Mask_Picks_Them():
    rng = np.random.default_rng(0)
    scans, values = rng.integers(0, 5, 200), rng.random(200)
    for order in (np.arange(200), np.argsort(scans, kind='stable')):    # as taken, and sorted
        mine, theirs = scans[order], values[order]
        for k, group in zip((0, 2, 4, 7), sp._By_Scan(mine, theirs, (0, 2, 4, 7))):
            assert np.array_equal(group, theirs[mine == k])


def test_The_Export_Hands_DENIS_Its_Scan_Filter(tmp_path):
    result = sp.Analyse(*_Run(rate=lambda scan: 0.1 if scan == 1 else 1.0), LASER)
    path = str(tmp_path / 'run_0007.asdf')
    with open(sp.Export(path, result)) as fh:
        record = yaml.safe_load(fh)
    assert record['scan_filters'] == {path: {'excluded': [1]}}
    assert record['verdict'] == result['Verdict']


def test_Excess_Is_Judged_Against_Its_Degrees_Of_Freedom():
    assert not sp._Excess(2.15, 4)
    assert sp._Excess(3.0, 50) and not sp._Excess(1.0, 50)


def test_The_Laser_Converts_As_The_Doppler_Shift_Says():
    beam_v, mass, nu = 30000.0, 59.0, 16956.17 * sp.C_LIGHT * 100
    beta = np.sqrt(2 * beam_v / (mass * sp.AMU_EV))
    # first order: the ion sees h nu, moving h nu beta / 2U per volt and h per
    # laser hertz, so the harmonic cancels
    expected = 2 * beam_v / (nu * beta)
    assert sp.Volts_Per_Hz(beam_v, nu, mass, LASER) == pytest.approx(expected, rel=2e-3)


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
