"""The hardware calibrations: their plans, the step-response, linearity and noise
analyses, the DAC correction they propose, their records, and a run of each on the
simulated rack."""

import multiprocessing

import numpy as np
import pytest

from cls_acquisition import hardware_calibration as hc
from cls_acquisition.acquisition import Ramp
from cls_acquisition.calibration import DIVIDER, Calibration_Error
from cls_acquisition.hardware import mcc_dac, simulated
from cls_acquisition.hardware.mcc_dac import CH_NEG, CH_POS
from cls_acquisition.scan_plan import Scan_Plan_Error
from tests.test_meters import _Meter

HALF_CODE_V = hc.CODE_V / 2


def test_Ramps_Never_Move_Further_Than_A_Step():
    assert Ramp(0.0, 120.0, 50.0) == pytest.approx([40.0, 80.0, 120.0])
    assert Ramp(100.0, 100.0, 50.0) == [100.0]


def test_A_Settle_Plan_Ramps_To_Each_Start_Then_Steps_Out_And_Back():
    plan = hc.Settle_Plan([(100.0, 300.0)], 2, 50.0)
    assert plan.Setpoints == pytest.approx([50, 100, 300, 100, 300, 100])
    assert plan.Marks == {2: (1, 'out', 1, 100.0, 300.0), 3: (1, 'back', 1, 300.0, 100.0),
                          4: (1, 'out', 2, 100.0, 300.0), 5: (1, 'back', 2, 300.0, 100.0)}


def _Measured(plan):
    return [(label, plan.Setpoints[i]) for i, label in sorted(plan.Marks.items())]


def test_A_Hysteresis_Sweep_Reads_Every_Rung_Twice_But_The_Top():
    plan = hc.Linearity_Plan(-100.0, 100.0, 50.0, hc.DIRECTIONS[2], 50.0)
    measured = [volts for _, volts in _Measured(plan)]
    assert measured == pytest.approx([-100, -50, 0, 50, 100, 50, 0, -50, -100])
    assert [label[1] for label, _ in _Measured(plan)] == ['up'] * 5 + ['down'] * 4
    with pytest.raises(Scan_Plan_Error):
        hc.Linearity_Plan(0.0, 10.0, 50.0, 'Up', 50.0)


def test_A_Sweep_Adds_A_Fine_Grid_Across_Zero_And_Checks_Zero_As_It_Goes():
    plan = hc.Linearity_Plan(-100.0, 100.0, 50.0, 'Up', 50.0, fine_span_v=4.0,
                             fine_step_v=2.0, zero_every=3)
    measured = _Measured(plan)
    rungs = [volts for (kind, _, _), volts in measured if kind == 'sweep']
    assert rungs == pytest.approx([-100, -50, -4, -2, 0, 2, 4, 50, 100])
    kinds = [kind for (kind, _, _), _ in measured]
    # 0 V read before every third rung and once more at the end
    assert kinds == ['zero', 'sweep', 'sweep', 'sweep', 'zero', 'sweep', 'sweep', 'sweep',
                     'zero', 'sweep', 'sweep', 'sweep', 'zero']
    assert all(volts == 0.0 for (kind, _, _), volts in measured if kind == 'zero')


def test_A_Random_Sweep_Visits_Every_Rung_Once_In_An_Order_Its_Seed_Fixes():
    rungs = lambda seed: [volts for _, volts in _Measured(hc.Linearity_Plan(
        -200.0, 200.0, 50.0, hc.DIRECTIONS[3], 500.0, seed=seed))]
    assert sorted(rungs(1)) == pytest.approx(np.arange(-200.0, 201.0, 50.0))
    assert rungs(1) == rungs(1) and rungs(1) != rungs(2)


def _First_Order(tau, height=1000.0, noise=0.1, interval=1e-4, count=4000, step_at=0.04):
    t = interval * np.arange(count)
    rng = np.random.default_rng(1)
    response = np.where(t < step_at, 0.0, height * (1 - np.exp(-(t - step_at) / tau)))
    return t, response + rng.normal(0.0, noise, count)


def test_A_First_Order_Step_Settles_When_Theory_Says():
    tau = 0.003
    result = hc.Step_Response(*_First_Order(tau), tolerance_v=1.0)
    assert result['Rise_S'] == pytest.approx(tau * np.log(9), rel=0.03)
    # inside 1 V of 1000 V: tau ln(1000), less the time to the 10 % crossing
    assert result['Settle_S'] == pytest.approx(tau * (np.log(1000) - np.log(1 / 0.9)), rel=0.05)
    assert result['Overshoot_Pct'] < 0.1 and result['Noise_V'] == pytest.approx(0.1, rel=0.2)


def test_No_Step_Or_No_Settling_Is_Said_So():
    t, flat = _First_Order(0.003, height=0.0)
    with pytest.raises(Calibration_Error, match="did not follow"):
        hc.Step_Response(t, flat, 1.0)
    with pytest.raises(Calibration_Error, match="had not settled"):
        hc.Step_Response(*_First_Order(1.0), tolerance_v=1.0)


@pytest.mark.parametrize('jump, settle_ms', [(1000.0, 16.9), (2000.0, 20.3)])
def test_The_Simulated_Platform_Settles_As_The_Bench_Measured(jump, settle_ms):
    # the bench: inside 0.5 V of a 1 kV jump in 16-17.4 ms, of 2 kV in 19.9-20.5 ms,
    # rising 10-90 % in 0.4 ms and overshooting by 0.18 %
    t = 0.0002 * np.arange(2500)
    volts = jump * (1 - np.array([simulated.Step_Remaining(s - 0.05) for s in t]))
    result = hc.Step_Response(t, volts, tolerance_v=0.5)
    assert result['Settle_S'] * 1e3 == pytest.approx(settle_ms, abs=1.0)
    assert result['Rise_S'] * 1e3 == pytest.approx(0.4, abs=0.15)
    assert result['Overshoot_Pct'] == pytest.approx(0.176, abs=0.02)


def test_The_Branch_Fits_Leave_Zero_To_Neither_Side():
    commanded = np.array([-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0])
    readback = np.where(commanded > 0, 1.001 * commanded + 0.2, 0.999 * commanded - 0.2)
    readback[3] = 5.0                       # whatever 0 V reads must not move either fit
    fits = hc.Branch_Fits(commanded, readback)
    assert fits[CH_POS]['Gain'] == pytest.approx(1.001)
    assert fits[CH_NEG]['Offset_V'] == pytest.approx(-0.2)
    hysteresis = hc.Hysteresis(np.array([0.0, 10.0, 20.0, 20.0, 10.0, 0.0]),
                               np.array([0.3, 10.1, 20.0, 20.3, 10.2, -0.3]))
    assert hysteresis == {'Mean_V': pytest.approx(0.2), 'Max_V': pytest.approx(0.3),
                          'Points': 2}
    assert hc.Hysteresis(np.array([0.0, 10.0]), np.array([0.0, 10.0])) is None


def _Simulated_Sweep(correction, start, stop, step, noise_v=0.01, seed=3):
    """A sweep of the simulated converters, each rung read once it has settled."""
    platform = simulated.Platform(multiprocessing.get_context('spawn'))
    dac = mcc_dac.Dac(simulated.Converters(platform).V_Out, correction)
    rng, points = np.random.default_rng(seed), []
    for volts in np.arange(start, stop + 1e-9, step):
        channel, dac_volts = dac.Write(volts)
        points.append({'Kind': 'sweep', 'Direction': 'up', 'Set_V': volts,
                       'Read_V': platform._State[1] + rng.normal(0.0, noise_v),
                       'Sigma_V': noise_v, 'Channel': channel, 'Dac_V': dac_volts,
                       'Time_S': 0.0})
    return hc.Linearity_Analysis(points)


def _Measured_Rack(start, stop, step):
    """What a noise-free sweep of the measured rack would read: its readback
    less setpoint, before any code rounding."""
    commanded = np.arange(start, stop + 1e-9, step)
    readback = commanded + np.array([simulated.Platform_Error(v) for v in commanded])
    return commanded, readback


def test_The_Simulated_Rack_Reads_Zero_Alike_Whichever_Converter_Was_Last():
    # the bench: both converters at mid-code read -0.90 to -0.99 V, after a
    # positive rung or a negative one alike, and the readback runs on through 0 V
    platform = simulated.Platform(multiprocessing.get_context('spawn'))
    dac = mcc_dac.Dac(simulated.Converters(platform).V_Out)
    zeros = []
    for volts in (500.0, 0.0, -500.0, 0.0, -1.0, 1.0):
        dac.Write(volts)
        zeros.append(platform._State[1])
    assert zeros[1] == zeros[3] == pytest.approx(-1.03, abs=0.01)
    # 1 V either side of 0 V differs in readback less setpoint by the code rounding alone
    assert abs((zeros[4] + 1.0) - (zeros[5] - 1.0)) < 2 * HALF_CODE_V


def test_The_Correction_Cancels_What_The_Sweep_Measured():
    fits = {CH_POS: {'Gain': 1.00055, 'Offset_V': 0.2185},
            CH_NEG: {'Gain': 0.99945, 'Offset_V': -0.2185}}
    correction = hc.Corrected(mcc_dac.CORRECTION, fits, 1, 'test', today='2026-10-25')
    platform = simulated.Platform(multiprocessing.get_context('spawn'))
    converters = simulated.Converters(platform)
    dac = mcc_dac.Dac(converters.V_Out, correction)
    for volts in (-3000.0, -10.0, 10.0, 3000.0):
        channel, _ = dac.Write(volts)
        # the platform as converters with exactly those gains and offsets put it
        dac_volts = converters._Out[channel]
        sign = 1.0 if channel == CH_POS else -1.0
        platform_volts = fits[channel]['Gain'] * sign * 1000 * dac_volts + fits[channel][
            'Offset_V']
        assert platform_volts == pytest.approx(volts, abs=HALF_CODE_V)
    assert correction['measured'] == '2026-10-25' and correction['valid_days'] == 1
    assert correction['inl'] is None


def test_A_Dense_Sweep_Measures_The_Non_Linearity_And_Its_Table_Removes_It():
    measured = _Simulated_Sweep(mcc_dac.CORRECTION, -900.0, 900.0, 2.0)
    # what reads as a seam is the two branches' lines meeting 0 V apart
    assert measured['Seam_V'] == pytest.approx(
        hc.Step_At_Zero(*_Measured_Rack(-900.0, 900.0, 2.0)), abs=0.05)
    tables = measured['Inl']
    assert set(tables) == {CH_POS, CH_NEG}
    assert np.diff(tables[CH_POS]['Knots_V']) == pytest.approx(hc.INL_KNOT_V)
    # over +-900 V the negative branch's shape is the larger, as on the bench
    assert tables[CH_NEG]['Smooth_Rms_V'] > 0.1 and tables[CH_POS]['Smooth_Rms_V'] > 0.03
    linear = hc.Corrected(mcc_dac.CORRECTION, measured['Fits'], 1, 'test')
    full = hc.Corrected(mcc_dac.CORRECTION, measured['Fits'], 1, 'test', inl=tables)
    off = lambda result: np.sqrt(np.mean((result['Read_V'] - result['Set_V']) ** 2))
    after_linear = _Simulated_Sweep(linear, -880.0, 880.0, 2.0)
    after_full = _Simulated_Sweep(full, -880.0, 880.0, 2.0)
    assert abs(after_linear['Seam_V']) < 0.05
    # all that is left is the code rounding: a code wide, uniform, 0.088 V rms
    assert off(after_full) < 0.105 < off(after_linear)
    assert all(t['Smooth_Rms_V'] < 0.03 for t in after_full['Inl'].values())
    # a sweep under the table composes with it rather than replacing it
    again = hc.Corrected(full, after_full['Fits'], 1, 'test', inl=after_full['Inl'])
    assert off(_Simulated_Sweep(again, -860.0, 860.0, 2.0)) < 0.105


def test_A_Sparse_Sweep_Proposes_No_Table():
    assert _Simulated_Sweep(mcc_dac.CORRECTION, -900.0, 900.0, 50.0)['Inl'] is None


def test_A_Correction_Is_Refused_For_A_Missing_Branch_Or_A_Wild_Gain():
    one = {CH_POS: {'Gain': 1.0, 'Offset_V': 0.0}}
    with pytest.raises(Calibration_Error, match="negative branch"):
        hc.Corrected(mcc_dac.CORRECTION, one, 1, 'test')
    wild = {**one, CH_NEG: {'Gain': 0.9, 'Offset_V': 0.0}}
    with pytest.raises(Calibration_Error, match="divider or wiring"):
        hc.Corrected(mcc_dac.CORRECTION, wild, 1, 'test')


def test_The_Noise_Analysis_Finds_The_Mains_Its_Level_And_The_Noise_Floor():
    interval, sigma = 0.001, 0.17
    t = interval * np.arange(60000)
    rng = np.random.default_rng(4)
    volts = (29884.8 + 0.438 * np.sin(2 * np.pi * 50.0 * t)
             + 0.03 * np.sin(2 * np.pi * 7.3 * t) + rng.normal(0.0, sigma, len(t)))
    result = hc.Noise_Analysis(volts, interval, integration_s=0.0002)
    mains, slow = result['Peaks'][0], result['Peaks'][1]
    assert mains['Freq_Hz'] == pytest.approx(50.0, abs=0.01) and mains['Mains_Harmonic'] == 1
    assert mains['Amplitude_V'] == pytest.approx(0.438, rel=0.03)
    assert slow['Freq_Hz'] == pytest.approx(7.3, abs=0.01) and slow['Mains_Harmonic'] == 0
    assert result['Mean_V'] == pytest.approx(29884.8, abs=0.01)
    assert result['Random_Rms_V'] == pytest.approx(sigma, rel=0.05)
    # white noise: sigma * sqrt(2 dt) per root hertz
    assert result['Noise_Floor_V_Rt_Hz'] == pytest.approx(sigma * np.sqrt(2 * interval),
                                                          rel=0.1)
    total = np.sqrt(sum(b['Rms_V'] ** 2 for b in result['Bands']))
    assert total == pytest.approx(result['Rms_Detrended_V'], rel=0.02)
    with pytest.raises(Calibration_Error, match="too short"):
        hc.Noise_Analysis(volts[:10], interval)


def test_A_Record_Keeps_Its_Rows_As_Taken_And_Is_Never_Overwritten(tmp_path):
    record = hc.Record(str(tmp_path), 'cooler', {'Interval': 0.01}, ('t', 'mode'))
    record.Add([0.0, 1.0], ['out', 'back'])
    with open(record.Path) as fh:
        assert fh.read() == "# Interval: 0.01\nt,mode\n0,out\n1,back\n"
    record.Close('stopped', {'Mean_V': 1.5})
    record.Close('completed')
    with open(record.Path) as fh:
        assert fh.read().endswith("1,back\n# status: stopped\n# Mean_V: 1.5\n")
    with pytest.raises(FileExistsError):
        open(record.Path, 'x')


def test_A_Capture_Holds_Off_The_Poll_And_Returns_Its_Block():
    meter, inst = _Meter(simulated.Keysight(lambda: 2e-3))
    meter.Capture(40, interval_s=0.001, aperture_s=0.0002)
    with pytest.raises(RuntimeError, match="cannot go from capturing to stepping"):
        meter.Step_Config()
    meter.Capture(30, interval_s=0.002, aperture_s=0.0002)          # set up again
    meter.Arm()
    block = meter.Fetch_When_Done(30, 5.0)
    assert len(block) == 30 and block == pytest.approx(2e-3)
    meter.Release()
    assert meter.Mode == 'continuous' and inst._Settings['SAMP:SOUR'] == 'IMM'
    with pytest.raises(ValueError, match="does not fit the meter's memory"):
        meter.Capture(10 ** 6)
    meter.Close()


def test_The_Loop_Refuses_A_Plan_Beyond_The_Limit_Before_Writing(loop):
    ok, why = loop.Send('Calibrate', setpoints=[0.0, 4500.0])
    assert not ok and "beyond the 4000 V limit" in why and "Nothing was written" in why
    assert not loop.State.Calibrating.is_set()
    ok, why = loop.Send('Start_Scan', setpoints=[-5000.0, 0.0], scan_type='Single scan',
                        bunches_per_step=1, timing={})
    assert not ok and "beyond the 4000 V limit" in why
    assert all(abs(volts) * mcc_dac.AMPLIFIER_GAIN <= 4000.0 for _, volts in loop.Writes)


def test_Stopping_A_Hardware_Calibration_Ramps_The_Platform_Down(loop):
    assert loop.Send('Calibrate', setpoints=[1000.0])[0]
    assert loop.Send('Calibration_Step', index=0)[0]
    start = len(loop.Writes)
    assert loop.Send('Stop_Calibration', ramp_step_v=250.0, ramp_rate_v_s=1e6)[0]
    live = [volts * mcc_dac.AMPLIFIER_GAIN for _, volts in loop.Writes[start:][1::2]]
    assert live == pytest.approx([750, 500, 250, 0], abs=HALF_CODE_V)


def test_A_Correction_Changes_Only_When_Nothing_Runs(loop):
    correction = dict(mcc_dac.CORRECTION, measured='2026-10-25', valid_days=1)
    assert loop.Send('Calibrate', setpoints=[0.0])[0]
    ok, why = loop.Send('Set_Correction', correction=correction)
    assert not ok and "calibration is running" in why
    assert loop.Send('Stop_Calibration')[0]
    ok, reply = loop.Send('Set_Correction', correction=correction)
    assert ok and reply['Dac_Correction'][0] == 1


def test_Each_Measurement_Runs_On_The_Simulated_Rack(tmp_path):
    from cls_acquisition import settings
    from cls_acquisition.acquisition import Rack_Link
    from cls_acquisition.hardware import Rack
    from cls_acquisition.hardware.keysight import COOLER_DIVIDER
    rack = Rack(settings.DEFAULTS, multiprocessing.get_context('spawn'))
    link = Rack_Link(rack.Loop_Spec())
    link.Send('Ping', timeout=30)
    line, cooler = rack.Open_Meters()
    never = lambda: False
    records = {kind: hc.Record(str(tmp_path), kind, {}, columns) for kind, columns in (
        ('settle', [c for _, c in hc.SETTLE_COLUMNS]),
        ('settle_traces', hc.SETTLE_TRACE_COLUMNS), ('linearity', hc.LINEARITY_COLUMNS),
        ('cooler', hc.COOLER_COLUMNS))}
    try:
        # 0.2 ms apart is shorter than the meter can sample at a 0.2 ms aperture: it
        # samples every 0.22 ms instead, and the capture shrinks to fill the same window
        steps = hc.Settle_Test(link, line, DIVIDER, [(0.0, 500.0)], 1, 0.2, 0.0002, 0.0002,
                               0.05, 0.5, (250.0, 1e5), lambda r: None, never, tail_s=0.3,
                               tail_nplc=0.2, tail_interval_s=0.01, records=records)
        assert [r['Mode'] for r in steps] == ['out', 'back']
        assert len(steps[0]['Times_S']) == round(0.2 / 0.00022)
        # the measured rack overshoots 0.9 V on 500 V and is inside 0.5 V in about 13 ms
        assert all(0.005 < r['Settle_S'] < 0.03 for r in steps)
        assert all(abs(r['Tail_Shift_V']) < 0.2 for r in steps)
        sweep = hc.Linearity_Sweep(link, line, DIVIDER, -500.0, 500.0, 100.0, 'Up', 0.05, 5,
                                   1, (250.0, 1e5), lambda p: None, never, zero_every=4,
                                   records=records)
        expected = hc.Branch_Fits(*_Measured_Rack(-500.0, 500.0, 100.0))
        assert sweep['Fits'][CH_POS]['Gain'] == pytest.approx(expected[CH_POS]['Gain'],
                                                              abs=1e-3)
        assert sweep['Fits'][CH_NEG]['Offset_V'] == pytest.approx(
            expected[CH_NEG]['Offset_V'], abs=0.3)
        assert sweep['Zero_Drift']['Checks'] == 4
        monitor = hc.Cooler_Monitor(cooler, COOLER_DIVIDER, 0.6, 0.002, 0.0002, None,
                                    lambda b: None, never, records=records)
        assert monitor['Count'] == 300
        assert monitor['Mean_V'] == pytest.approx(simulated.COOLER_V, abs=1.0)
        assert line.Mode == cooler.Mode == 'continuous' and link.State.Voltage.value == 0.0
        for record in records.values():
            record.Close('completed')
            with open(record.Path) as fh:
                assert len(fh.read().splitlines()) > 3
        # and each record reads back as it was taken, for its page's Open record
        read = hc.Read_Settle(records['settle'].Path)
        assert [(r['Step'], r['Mode'], r['Rep']) for r in read] == [
            (r['Step'], r['Mode'], r['Rep']) for r in steps]
        for back, taken in zip(read, steps):
            for key in ('Lag_S', 'Settle_S', 'Rise_S', 'Overshoot_Pct', 'Final_V'):
                assert back[key] == pytest.approx(taken[key], rel=1e-8)
            assert back['Deviation_V'] == pytest.approx(taken['Deviation_V'], rel=1e-8,
                                                        abs=1e-12)
        points = hc.Read_Linearity(records['linearity'].Path)
        assert [p['Kind'] for p in points] == [p['Kind'] for p in sweep['Points']]
        assert hc.Linearity_Analysis(points)['Gain'] == pytest.approx(sweep['Gain'], rel=1e-8)
        times, volts, _ = hc.Read_Cooler(records['cooler'].Path)
        assert len(volts) == 300 and np.mean(volts) == pytest.approx(monitor['Mean_V'])
    finally:
        line.Close()
        cooler.Close()
        link.Close()


def test_The_Settle_Model_Grows_With_The_Jump_And_Never_Counts_A_Lag_Below_Zero():
    results = [{'From_V': 0.0, 'To_V': 1000.0, 'Lag_S': 0.003, 'Settle_S': 0.017},
               {'From_V': 0.0, 'To_V': 100.0, 'Lag_S': -0.001, 'Settle_S': 0.002},
               {'From_V': 1000.0, 'To_V': -1000.0, 'Lag_S': 0.002, 'Settle_S': 0.016}]
    model = hc.Settle_Model(results, today='2026-10-27')
    # 2 kV settled sooner than 1 kV here, but a larger jump is never taken as faster
    assert model == {'measured': '2026-10-27',
                     'points': [[100.0, 0.002], [1000.0, 0.02], [2000.0, 0.02]]}


def test_The_Line_Meter_Is_Read_Precisely_And_Left_On_Its_Poll():
    meter, _ = _Meter(simulated.Keysight(lambda: 0.1234567, simulated.LINE_NOISE_V))
    mean, sigma = hc.Precise_Reading(meter, DIVIDER)
    # five readings at NPLC 10: the 0.065 V scatter at 200 us falls by sqrt(1000)
    assert mean == pytest.approx(123.4567, abs=0.02) and sigma < 0.02
    assert meter.Mode == 'continuous'
    meter.Close()


def test_A_Precise_Reading_That_Fails_Still_Leaves_The_Meter_On_Its_Poll(monkeypatch):
    meter, _ = _Meter(simulated.Keysight(lambda: 0.1))

    def lost(count, timeout_s):
        raise RuntimeError("The meter at SIM did not answer: timeout.")

    monkeypatch.setattr(meter, 'Fetch_When_Done', lost)
    with pytest.raises(RuntimeError, match="did not answer"):
        hc.Precise_Reading(meter, DIVIDER)
    assert meter.Mode == 'continuous'
    meter.Close()


def test_The_Recommended_Wait_Is_Three_Slowest_Steps_From_Their_Command_And_At_Least_0_2_S():
    assert hc.Recommended_Wait([{'Lag_S': 0.003, 'Settle_S': 0.0175}]) == pytest.approx(0.2)
    assert hc.Recommended_Wait([{'Lag_S': 0.003, 'Settle_S': 0.0175},
                                {'Lag_S': -0.01, 'Settle_S': 0.1}]) == pytest.approx(0.3)
