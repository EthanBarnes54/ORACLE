"""The hardware calibrations: their plans, the step-response and linearity analysis,
the DAC correction they propose, and a run of each on the simulated rack."""

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


def test_Ramps_Never_Move_Further_Than_A_Step():
    assert Ramp(0.0, 120.0, 50.0) == pytest.approx([40.0, 80.0, 120.0])
    assert Ramp(100.0, 100.0, 50.0) == [100.0]


def test_A_Settle_Plan_Ramps_To_Each_Start_Then_Steps_Out_And_Back():
    plan = hc.Settle_Plan([(100.0, 300.0)], 2, 50.0)
    assert plan.Setpoints == pytest.approx([50, 100, 300, 100, 300, 100])
    assert plan.Marks == {2: (1, 'out', 1, 100.0, 300.0), 3: (1, 'back', 1, 300.0, 100.0),
                          4: (1, 'out', 2, 100.0, 300.0), 5: (1, 'back', 2, 300.0, 100.0)}


def test_A_Hysteresis_Sweep_Reads_Every_Rung_Twice_But_The_Top():
    plan = hc.Linearity_Plan(-100.0, 100.0, 50.0, hc.DIRECTIONS[2], 50.0)
    measured = [plan.Setpoints[i] for i in plan.Marks]
    assert measured == pytest.approx([-100, -50, 0, 50, 100, 50, 0, -50, -100])
    with pytest.raises(Scan_Plan_Error):
        hc.Linearity_Plan(0.0, 10.0, 50.0, 'Up', 50.0)


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


def test_The_Branch_Fits_Leave_Zero_To_Neither_Side():
    commanded = np.array([-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0])
    readback = np.where(commanded > 0, 1.001 * commanded + 0.2, 0.999 * commanded - 0.2)
    readback[3] = 5.0                       # whatever 0 V reads must not move either fit
    fits = hc.Branch_Fits(commanded, readback)
    assert fits[CH_POS]['Gain'] == pytest.approx(1.001)
    assert fits[CH_NEG]['Offset_V'] == pytest.approx(-0.2)
    assert hc.Hysteresis(np.array([0.0, 10.0, 10.0, 0.0]), np.array([0.3, 10.1, 10.2, -0.3])) \
        == pytest.approx(0.1)


def test_The_Correction_Cancels_What_The_Sweep_Measured():
    fits = {CH_POS: {'Gain': 1.00055, 'Offset_V': 0.2185},
            CH_NEG: {'Gain': 0.99945, 'Offset_V': -0.2185}}
    correction = hc.Corrected(mcc_dac.CORRECTION, fits, 1, 'test', today='2026-10-25')
    platform = simulated.Platform(multiprocessing.get_context('spawn'))
    dac = mcc_dac.Dac(simulated.Converters(platform).V_Out, correction)
    for volts in (-3000.0, -10.0, 10.0, 3000.0):
        dac.Write(volts)
        platform_volts = platform._State[1]             # where the platform is going
        assert platform_volts == pytest.approx(volts, abs=1e-9)
    assert correction['measured'] == '2026-10-25' and correction['valid_days'] == 1


def test_A_Correction_Is_Refused_For_A_Missing_Branch_Or_A_Wild_Gain():
    one = {CH_POS: {'Gain': 1.0, 'Offset_V': 0.0}}
    with pytest.raises(Calibration_Error, match="negative branch"):
        hc.Corrected(mcc_dac.CORRECTION, one, 1, 'test')
    wild = {**one, CH_NEG: {'Gain': 0.9, 'Offset_V': 0.0}}
    with pytest.raises(Calibration_Error, match="divider or wiring"):
        hc.Corrected(mcc_dac.CORRECTION, wild, 1, 'test')


def test_White_Noise_Has_A_Flat_Spectrum_At_Its_Level():
    sigma, interval = 0.5, 0.01
    rng = np.random.default_rng(2)
    blocks = [rng.normal(0.0, sigma, 1000) for _ in range(20)]
    frequency, asd = hc.Amplitude_Spectral_Density(blocks, interval)
    assert frequency[-1] == pytest.approx(0.5 / interval)
    assert np.mean(asd ** 2) == pytest.approx(sigma ** 2 * 2 * interval, rel=0.05)


def test_A_Record_Is_Never_Overwritten(tmp_path):
    path = hc.Save(str(tmp_path), 'cooler', {'Mean_V': 1.0}, ('t', 'mode'),
                   [[0.0, 1.0], ['out', 'back']])
    with open(path) as fh:
        assert fh.read() == "# Mean_V: 1.0\nt,mode\n0,out\n1,back\n"


def test_A_Capture_Holds_Off_The_Poll_And_Returns_Its_Block():
    meter, inst = _Meter(simulated.Keysight(lambda: 2e-3))
    meter.Capture(40, interval_s=0.001, aperture_s=0.0002)
    with pytest.raises(RuntimeError, match="cannot go from capturing to stepping"):
        meter.Step_Config()
    meter.Arm()
    block = meter.Fetch()
    assert len(block) == 40 and block == pytest.approx(2e-3)
    meter.Release()
    assert meter.Mode == 'continuous' and inst._Settings['SAMP:SOUR'] == 'IMM'
    with pytest.raises(ValueError, match="does not fit the meter's memory"):
        meter.Capture(10 ** 6)
    meter.Close()


def test_The_Loop_Refuses_A_Plan_Beyond_The_Limit_Before_Writing(loop):
    ok, why = loop.Send('Calibrate', setpoints=[0.0, 6500.0])
    assert not ok and "beyond the 6000 V limit" in why and "Nothing was written" in why
    assert not loop.State.Calibrating.is_set()
    ok, why = loop.Send('Start_Scan', setpoints=[-7000.0, 0.0], scan_type='Single scan',
                        bunches_per_step=1, timing={})
    assert not ok and "beyond the 6000 V limit" in why
    assert all(abs(volts) * mcc_dac.AMPLIFIER_GAIN <= 6000.0 for _, volts in loop.Writes)


def test_Stopping_A_Hardware_Calibration_Ramps_The_Platform_Down(loop):
    assert loop.Send('Calibrate', setpoints=[1000.0])[0]
    assert loop.Send('Calibration_Step', index=0)[0]
    start = len(loop.Writes)
    assert loop.Send('Stop_Calibration', ramp_step_v=250.0, ramp_rate_v_s=1e6)[0]
    live = [volts * mcc_dac.AMPLIFIER_GAIN for _, volts in loop.Writes[start:][1::2]]
    assert live == pytest.approx([750, 500, 250, 0])


def test_A_Correction_Changes_Only_When_Nothing_Runs(loop):
    correction = dict(mcc_dac.CORRECTION, measured='2026-10-25', valid_days=1)
    assert loop.Send('Calibrate', setpoints=[0.0])[0]
    ok, why = loop.Send('Set_Correction', correction=correction)
    assert not ok and "calibration is running" in why
    assert loop.Send('Stop_Calibration')[0]
    ok, reply = loop.Send('Set_Correction', correction=correction)
    assert ok and reply['Dac_Correction'][0] == 1


def test_Each_Measurement_Runs_On_The_Simulated_Rack():
    from cls_acquisition import settings
    from cls_acquisition.acquisition import Rack_Link
    from cls_acquisition.hardware import Rack
    from cls_acquisition.hardware.keysight import COOLER_DIVIDER
    rack = Rack(settings.DEFAULTS, multiprocessing.get_context('spawn'))
    link = Rack_Link(rack.Loop_Spec())
    link.Send('Ping', timeout=30)
    line, cooler = rack.Open_Meters()
    never = lambda: False
    try:
        steps = hc.Settle_Test(link, line, DIVIDER, [(0.0, 500.0)], 1, 0.2, 0.0005, 0.0002,
                               0.05, 1.0, (250.0, 1e5), lambda r: None, never)
        assert [r['Mode'] for r in steps] == ['out', 'back']
        # the simulator settles with tau 3 ms: inside 5 sigma of its noise in about 20 ms
        assert all(0.005 < r['Settle_S'] < 0.05 for r in steps)
        # 20 readings a point put the seam's own uncertainty near 0.06 V
        sweep = hc.Linearity_Sweep(link, line, DIVIDER, -500.0, 500.0, 100.0, 'Up', 0.05, 20, 1,
                                   (250.0, 1e5), lambda p: None, never)
        assert sweep['Fits'][CH_POS]['Gain'] == pytest.approx(simulated.BRANCH_GAIN[0] / 1000,
                                                              abs=1e-3)
        assert sweep['Seam_V'] == pytest.approx(2 * simulated.BRANCH_OFFSET[0], abs=0.25)
        monitor = hc.Cooler_Monitor(cooler, COOLER_DIVIDER, 0.6, 0.3, 0.01, 0.001,
                                    lambda b: None, never)
        assert monitor['Mean_V'] == pytest.approx(simulated.COOLER_V, abs=1.0)
        assert line.Mode == cooler.Mode == 'continuous' and link.State.Voltage.value == 0.0
    finally:
        line.Close()
        cooler.Close()
        link.Close()
