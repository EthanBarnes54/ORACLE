"""Step and confirm, the retry rule, and the gate.

The sequence runs against a fake link and meter that share one fake
clock, so timing faults are injected exactly and nothing sleeps: an
early read is a stamp short of SETTLE by a chosen amount, a clock offset
moves every confirmation's stamp.
"""

import multiprocessing
import random
import time
import types

import numpy as np
import pytest

from cls_acquisition import calibration
from cls_acquisition import hardware_calibration as hc
from cls_acquisition.acquisition import Acquisition_Error
from cls_acquisition.calibration import (DIVIDER, REJECT, THRESHOLDS, WARN, Calibration,
                                         Calibration_Error, Quality_Report, Run_Calibration)
from cls_acquisition.hardware import keysight, mcc_dac, simulated
from cls_acquisition.hardware.visa_session import Visa_Session
from cls_acquisition.scan_plan import Scan_Plan
from tests.bench import Loop_Bench

SETTLE_NS = round(THRESHOLDS['SettleS'] * 1e9)
HELD = THRESHOLDS['HeldReadings']
DAC = {'Dac_Status': ('PASS', 'identity'), 'Dac_Correction': (0, 'none')}


class Fake_Rack():
    """The acquisition process and the line meter on one fake ns clock.
    ``early`` maps a setpoint to the gaps after its write that its first
    reads report, whatever the clock says; later reads stamp the clock."""

    def __init__(self, plan, early=None, offset_ns=0, refuse=None, misconfirm=None):
        self.Plan = plan
        self.Early = {k: list(v) for k, v in (early or {}).items()}
        self.Offset, self.Refuse, self.Misconfirm = offset_ns, refuse, misconfirm
        self.Clock, self.Index, self.T_Command = 0, None, 0
        self.Sent, self.Waits, self.Reads = [], [], 0
        self.Mode = 'continuous'
        self.State = types.SimpleNamespace(Voltage=types.SimpleNamespace(value=0.0))

    def Send(self, command, **params):
        self.Sent.append(command)
        if command == 'Calibrate':
            return DAC
        if command == 'Stop_Calibration':
            return {}
        index = params['index']
        if index == self.Refuse:
            raise Acquisition_Error(f"Setpoint {index} refused.")
        self.Clock += 1_000_000
        self.Index, self.T_Command = index, self.Clock + self.Offset
        return {'Index': index + (index == self.Misconfirm), 'Volts': self.Plan.Setpoints[index],
                'T_Command_Ns': self.T_Command}

    def Step_Config(self):
        self.Mode = 'stepping'

    def Read_Settled(self, wait_s):
        self.Waits.append(wait_s)
        self.Reads += 1
        self.Clock += round(wait_s * 1e9)
        gaps = self.Early.get(self.Index)
        stamp = self.T_Command + gaps.pop(0) if gaps else self.Clock
        return stamp, self.Plan.Setpoints[self.Index] / DIVIDER

    def Release(self):
        self.Mode = 'continuous'


@pytest.fixture
def plan():
    return Scan_Plan([-50], [50], 5)


def _Assert_Cleaned_Up(rack):
    assert rack.Mode == 'continuous'
    assert rack.Sent[-1] == 'Stop_Calibration'


def test_Healthy_Calibration_Pairs_Every_Setpoint_With_Its_Own_Reading(plan):
    rack = Fake_Rack(plan)
    cal = Run_Calibration(rack, rack, plan)
    assert np.array_equal(cal.Commanded, plan.Setpoints)
    assert np.allclose(cal.Readings * DIVIDER, plan.Setpoints)
    assert cal.Gaps_Ns.min() >= SETTLE_NS and cal.Retried == 0
    assert rack.Reads == len(plan) + HELD           # and the last setpoint held
    _Assert_Cleaned_Up(rack)
    report = Quality_Report(cal, plan)
    assert report['Verdict'] == 'PASS', report
    assert report['Thresholds'] == THRESHOLDS       # the numbers it was judged by, kept


def test_Early_Read_Is_Discarded_And_The_Reading_Requested_Again(plan):
    rack = Fake_Rack(plan, early={4: [SETTLE_NS - 3_000_000]})
    cal = Run_Calibration(rack, rack, plan)
    assert rack.Reads == len(plan) + 1 + HELD          # one re-request, no extra write
    assert rack.Sent.count('Calibration_Step') == len(plan)
    assert rack.Waits[5] == pytest.approx(3e-3 + THRESHOLDS['RetryMarginS'])
    assert cal.Retried == 1 and cal.Gaps_Ns.min() >= SETTLE_NS
    settle = next(c for c in Quality_Report(cal, plan)['Checks'] if c['Check'] == 'settle')
    assert settle['Verdict'] == WARN


def test_Persistent_Early_Read_Stops_At_The_Point_Cap(plan):
    rack = Fake_Rack(plan, early={2: [SETTLE_NS - 1_000_000] * 10})
    with pytest.raises(Calibration_Error, match=r"Setpoint 2 .* too early 3 times"):
        Run_Calibration(rack, rack, plan)
    assert rack.Reads == 2 + THRESHOLDS['MaxAttemptsPerPoint']
    _Assert_Cleaned_Up(rack)


def test_Stamp_Before_Its_Write_Aborts_At_Once(plan):
    rack = Fake_Rack(plan, offset_ns=10 * SETTLE_NS)
    with pytest.raises(Calibration_Error, match="clocks disagree"):
        Run_Calibration(rack, rack, plan)
    assert rack.Reads == 1
    _Assert_Cleaned_Up(rack)


def test_Small_Clock_Offset_Converges_Within_The_Budget(plan):
    rack = Fake_Rack(plan, early={i: [SETTLE_NS - 2_000_000] for i in range(5)})
    cal = Run_Calibration(rack, rack, plan)
    assert cal.Retried == 5


def test_Constant_Offset_Trips_The_Calibration_Cap(plan):
    rack = Fake_Rack(plan, early={i: [SETTLE_NS - 2_000_000] for i in range(len(plan))})
    with pytest.raises(Calibration_Error, match="limit is 10"):
        Run_Calibration(rack, rack, plan)
    assert rack.Reads == 2 * THRESHOLDS['MaxRetriesPerCalibration'] + 1
    _Assert_Cleaned_Up(rack)


def test_Refused_Step_Aborts_And_Cleans_Up(plan):
    rack = Fake_Rack(plan, refuse=3)
    with pytest.raises(Acquisition_Error, match="Setpoint 3 refused"):
        Run_Calibration(rack, rack, plan)
    _Assert_Cleaned_Up(rack)


def test_Confirmation_For_Another_Setpoint_Aborts(plan):
    rack = Fake_Rack(plan, misconfirm=6)
    with pytest.raises(Calibration_Error, match="confirmed setpoint 7 when setpoint 6"):
        Run_Calibration(rack, rack, plan)
    _Assert_Cleaned_Up(rack)


def test_A_Range_Join_Waits_As_Long_As_Its_Jump_Takes():
    # the brief named the first write and any range join as the worst cases, not a ramp step
    plan = Scan_Plan([-3000, 2990], [-2990, 3000], 5)
    model = {'points': [[5.0, 0.01], [6000.0, 0.08]]}
    join = int(np.argmax(np.abs(np.diff(plan.Setpoints)))) + 1
    need = calibration.Settle_For(model, abs(plan.Setpoints[join] - plan.Setpoints[join - 1]))
    assert need > THRESHOLDS['SettleS']
    rack = Fake_Rack(plan)
    cal = Run_Calibration(rack, rack, plan, settle_model=model)
    assert rack.Waits[0] == pytest.approx(calibration.Settle_For(model, 3000.0))
    assert rack.Waits[1] == THRESHOLDS['SettleS']   # a 5 V step: SETTLE
    assert rack.Waits[join] == pytest.approx(need)
    assert cal.Settle_Ns[join] == round(need * 1e9)
    # read SETTLE after the join's write, it had not settled: read again, not kept
    rack = Fake_Rack(plan, early={join: [SETTLE_NS + 1]})
    cal = Run_Calibration(rack, rack, plan, settle_model=model)
    assert cal.Retried == 1 and cal.Gaps_Ns[join] >= round(need * 1e9)
    # and the gate holds the arrays to the same number
    cal.Gaps_Ns[join] = SETTLE_NS + 1
    report = Quality_Report(cal, plan)
    assert _Verdict(report, 'settle') == REJECT and '2990.00 V' in calibration.Rejections(report)


class Lagging_Rack(Fake_Rack):
    """Mode B on the bench: the platform trails the command by ``k`` setpoints while
    the calibration steps, from ``rest`` before the first; the meter reads what
    the platform holds. Held after the last write, the platform catches up."""

    def __init__(self, plan, k, rest):
        super().__init__(plan)
        self.K, self.Rest, self.Taken = k, rest, 0

    def Read_Settled(self, wait_s):
        stamp, _ = super().Read_Settled(wait_s)
        self.Taken += 1
        if self.Taken > len(self.Plan):
            return stamp, self.Plan.Setpoints[-1] / DIVIDER
        held = self.Index - self.K
        volts = self.Plan.Setpoints[held] if held >= 0 else self.Rest
        return stamp, volts / DIVIDER


@pytest.mark.parametrize('k', [1, 2, 5, 12])
@pytest.mark.parametrize('rest', ['at rest', 'at the first setpoint'])
def test_A_Lagged_Calibration_Is_Refused_Before_Any_Scan(k, rest):
    # Mode B's k, which the audit found at 1, 3 and 5, on the beamline's ramp. A shifted
    # ramp still tracks and spans: before the held reading the gate let every one through
    plan = Scan_Plan([-450], [475], 2)
    rack = Lagging_Rack(plan, k, 0.0 if rest == 'at rest' else plan.Setpoints[0])
    report = Quality_Report(Run_Calibration(rack, rack, plan), plan)
    for check in ('length', 'tracking', 'span', 'settle', 'plan'):
        assert _Verdict(report, check) == 'PASS', check
    assert _Verdict(report, 'held') == REJECT and report['Verdict'] == REJECT
    assert report['HeldApartV'] == pytest.approx(-2.0 * k)
    assert f"({-k:+.1f} steps)" in calibration.Rejections(report)


class Stale_Meter(Fake_Rack):
    """A meter whose answers trail its requests by ``k``: the platform is where it
    should be, the readings are not."""

    def __init__(self, plan, k):
        super().__init__(plan)
        self.K, self.Answers = k, []

    def Read_Settled(self, wait_s):
        stamp, volts = super().Read_Settled(wait_s)
        self.Answers.append(volts)
        return stamp, self.Answers[max(len(self.Answers) - 1 - self.K, 0)]


@pytest.mark.parametrize('k', [1, 3, 12])
def test_A_Meter_Answering_Late_Is_Refused_Before_Any_Scan(k):
    # Mode B by another route: the readings, not the platform, trail. A lag of one
    # leaves the held readings steady and a step from the last; more reads them as a ramp
    plan = Scan_Plan([-450], [475], 2)
    rack = Stale_Meter(plan, k)
    report = Quality_Report(Run_Calibration(rack, rack, plan), plan)
    assert _Verdict(report, 'tracking') == 'PASS' and _Verdict(report, 'span') == 'PASS'
    assert _Verdict(report, 'held') == REJECT
    assert ('(-1.0 steps)' if k == 1 else 'scatter by') in calibration.Rejections(report)


def test_A_Calibrate_That_Goes_Unanswered_Is_Still_Stopped(plan):
    # one that timed out may yet be carried out, leaving the process calibrating for good

    class Unanswered(Fake_Rack):

        def Send(self, command, **params):
            if command == 'Calibrate':
                self.Sent.append(command)
                raise Acquisition_Error("The acquisition process did not answer 'Calibrate' "
                                        "within 10 s.")
            return super().Send(command, **params)

    rack = Unanswered(plan)
    with pytest.raises(Acquisition_Error, match="did not answer"):
        Run_Calibration(rack, rack, plan)
    _Assert_Cleaned_Up(rack)


class _Slow_Link():
    """The loop as Run_Calibration sees it, each reply held back at random."""

    def __init__(self, bench, delay):
        self.State, self._Bench, self._Delay = bench.State, bench, delay

    def Send(self, command, timeout=None, **params):
        time.sleep(self._Delay())
        ok, payload = self._Bench.Send(command, **params)
        time.sleep(self._Delay())
        if not ok:
            raise Acquisition_Error(payload)
        return payload


def test_Each_Reading_Holds_Its_Own_Setpoint_Whatever_The_Latencies():
    # Mode B's cause was two clocks lined up afterwards, which any latency shifted by k
    # readings. Here the real loop, meter driver and calibration run with the loop's
    # replies, the DAC's writes and the meter's answers each delayed at random by up to
    # 0.2 s (seven of the old 30 ms trigger periods), and each reading is judged by the
    # setpoint the platform really held as the meter sampled it
    plan = Scan_Plan([-30], [30], 2)
    bench = Loop_Bench()
    delays = {site: random.Random(seed) for seed, site in enumerate(('link', 'dac', 'meter'))}

    def delay(site):
        rng = delays[site]
        return rng.uniform(0.0, 0.2) if rng.random() < 0.3 else 0.0

    write = bench.Dac._V_Out
    bench.Dac._V_Out = lambda channel, volts: (time.sleep(delay('dac')), write(channel, volts))
    sampled, held = {}, {}

    def source():
        sampled['now'] = bench.Platform._State[1], bench.Platform.Volts()
        return sampled['now'][1] / DIVIDER

    class Slow_Meter(simulated.Keysight):

        def query(self, command):
            reply = super().query(command)
            if command.strip().upper() == 'READ?':
                # what the platform held as this reading was taken, by the reading itself
                held[float(reply)] = sampled['now']
            time.sleep(delay('meter'))
            return reply

    line = keysight.Line_Meter(Visa_Session.Open('SIM::LATENCY', lambda _: Slow_Meter(source)))
    try:
        cal = Run_Calibration(_Slow_Link(bench, lambda: delay('link')), line, plan)
    finally:
        line.Close()
        bench.Close()
    assert cal.Retried == 0
    # where the platform sits for each setpoint, on the measured rack's own curve
    expected = np.array([v + simulated.Platform_Error(v) for v in plan.Setpoints])
    targets, platform = np.array([held[reading] for reading in cal.Readings]).T
    holding = np.abs(targets[:, None] - expected[None, :]).argmin(axis=1)
    assert np.array_equal(holding, np.arange(len(plan)))        # lag 0, at every point
    assert np.max(np.abs(platform - targets)) < 0.01            # and settled when read
    assert np.max(np.abs(cal.Readings * DIVIDER - targets)) < 0.5


# ── The gate ────────────────────────────────────────────────────────────

def _Cal(plan, readback_volts, gaps_ns=None, retried=0, dac_status=('PASS', 'identity')):
    steps = [{'Index': i, 'Volts': v, 'T_Command_Ns': 0} for i, v in enumerate(plan.Setpoints)]
    stamps = np.full(len(steps), SETTLE_NS + 1) if gaps_ns is None else gaps_ns
    return Calibration(steps, np.asarray(readback_volts) / DIVIDER, stamps,
                       np.full(len(steps), SETTLE_NS), retried,
                       {'Dac_Status': dac_status, 'Dac_Correction': (0, 'none')},
                       np.full(HELD, readback_volts[-1] / DIVIDER))


def _Verdict(report, check):
    return next(c['Verdict'] for c in report['Checks'] if c['Check'] == check)


def test_Gate_Refuses_A_Dead_Readback(plan):
    # Mode A: the platform read at rest, microvolts of noise for the whole ramp
    noise = np.random.default_rng(1).normal(0, 2e-5, len(plan)) * DIVIDER
    report = Quality_Report(_Cal(plan, noise), plan)
    assert report['Verdict'] == REJECT and _Verdict(report, 'tracking') == REJECT


def test_Gate_Refuses_Half_The_Span(plan):
    report = Quality_Report(_Cal(plan, 0.5 * plan.Setpoints), plan)
    assert _Verdict(report, 'span') == REJECT


def test_Gate_Refuses_A_Short_Calibration(plan):
    cal = _Cal(plan, plan.Setpoints)
    cal.Readings = cal.Readings[:-1]
    assert Quality_Report(cal, plan)['Verdict'] == REJECT


def test_Gate_Refuses_An_Early_Reading_In_The_Arrays(plan):
    gaps = np.full(len(plan), SETTLE_NS + 1)
    gaps[3] = SETTLE_NS - 1
    assert _Verdict(Quality_Report(_Cal(plan, plan.Setpoints, gaps), plan), 'settle') == REJECT


def test_Gate_Warns_On_The_Seam_At_Zero(plan):
    seam = plan.Setpoints + np.where(plan.Setpoints >= 0, 0.2185, -0.2185)
    report = Quality_Report(_Cal(plan, seam), plan)
    assert report['Verdict'] == WARN and _Verdict(report, 'polarity') == WARN


def test_Gate_Passes_The_Dac_Verdict_Through(plan):
    report = Quality_Report(_Cal(plan, plan.Setpoints, dac_status=('REJECT', 'expired')), plan)
    assert _Verdict(report, 'dac_calibration') == REJECT
    assert 'dac_calibration: expired' in calibration.Rejections(report)


@pytest.mark.parametrize('measured', [False, True])
def test_The_Beamline_Geometry_On_The_Measured_Rack(measured):
    # the audit's hardening found a gate that refused the ramp the beamline runs, which
    # no test had used: -450 to +475 V in 2 V steps, here through the measured rack's
    # converters (seam, non-linearity, code rounding) and its meter's scatter
    plan = Scan_Plan([-450], [475], 2)
    platform = simulated.Platform(multiprocessing.get_context('spawn'))
    rng = np.random.default_rng(5)
    correction = mcc_dac.CORRECTION
    if measured:
        dac = mcc_dac.Dac(simulated.Converters(platform).V_Out)
        points = []
        for volts in np.arange(-1000.0, 1001.0, 2.0):
            channel, _ = dac.Write(volts)
            points.append({'Kind': 'sweep', 'Direction': 'up', 'Set_V': volts,
                           'Read_V': platform._State[1], 'Channel': channel, 'Sigma_V': 0.0,
                           'Dac_V': 0.0, 'Time_S': 0.0})
        sweep = hc.Linearity_Analysis(points)
        correction = hc.Corrected(correction, sweep['Fits'], 1, 'test', inl=sweep['Inl'])
    dac = mcc_dac.Dac(simulated.Converters(platform).V_Out, correction)

    class Measured_Rack(Fake_Rack):

        def Read_Settled(self, wait_s):
            stamp, _ = super().Read_Settled(wait_s)
            dac.Write(self.Plan.Setpoints[self.Index])
            return stamp, (platform._State[1] + rng.normal(0.0, 0.065)) / DIVIDER

    rack = Measured_Rack(plan)
    report = Quality_Report(Run_Calibration(rack, rack, plan), plan)
    assert len(plan) == 463
    for check in ('length', 'tracking', 'span', 'settle', 'held', 'plan'):
        assert _Verdict(report, check) == 'PASS', check
    # the branches' lines meet 0 V 0.5 V apart on the bench; a measured correction brings
    # them within 0.1 V
    assert _Verdict(report, 'polarity') == ('PASS' if measured else WARN)
    assert report['Verdict'] == ('PASS' if measured else WARN)
