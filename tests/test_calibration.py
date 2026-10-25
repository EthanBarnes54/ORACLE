"""Step and confirm, the retry rule, and the gate.

The sequence runs against a fake link and meter that share one fake
clock, so timing faults are injected exactly and nothing sleeps: an
early read is a stamp short of SETTLE by a chosen amount, a clock offset
moves every confirmation's stamp.
"""

import types

import numpy as np
import pytest

from cls_acquisition import calibration
from cls_acquisition.acquisition import Acquisition_Error
from cls_acquisition.calibration import (DIVIDER, REJECT, THRESHOLDS, WARN, Calibration,
                                         Calibration_Error, Quality_Report, Run_Calibration)
from cls_acquisition.scan_plan import Scan_Plan

SETTLE_NS = round(THRESHOLDS['SettleS'] * 1e9)
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
    assert rack.Reads == len(plan)
    _Assert_Cleaned_Up(rack)
    report = Quality_Report(cal, plan)
    assert report['Verdict'] == 'PASS', report


def test_Early_Read_Is_Discarded_And_The_Reading_Requested_Again(plan):
    rack = Fake_Rack(plan, early={4: [SETTLE_NS - 3_000_000]})
    cal = Run_Calibration(rack, rack, plan)
    assert rack.Reads == len(plan) + 1                 # one re-request, no extra write
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


# ── The gate ────────────────────────────────────────────────────────────

def _Cal(plan, readback_volts, gaps_ns=None, retried=0, dac_status=('PASS', 'identity')):
    steps = [{'Index': i, 'Volts': v, 'T_Command_Ns': 0} for i, v in enumerate(plan.Setpoints)]
    stamps = np.full(len(steps), SETTLE_NS + 1) if gaps_ns is None else gaps_ns
    return Calibration(steps, np.asarray(readback_volts) / DIVIDER, stamps, retried,
                       {'Dac_Status': dac_status, 'Dac_Correction': (0, 'none')})


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
