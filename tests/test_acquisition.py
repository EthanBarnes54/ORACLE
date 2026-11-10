"""The acquisition loop's instruction protocol, scan path and calibration steps,
and the link to it across a real process."""

import threading
import time

import numpy as np
import pytest

from cls_acquisition import acquisition
from cls_acquisition.acquisition import Acquisition_Error, Rack_Link
from cls_acquisition.hardware import mcc_dac
from cls_acquisition.scan_plan import Scan_Plan
from tests.bench import Loop_Bench

HALF_CODE_V = mcc_dac.LSB_V * mcc_dac.AMPLIFIER_GAIN / 2   # a write's rounding, at the platform
TIMING = {'bunched': True, 'rep_rate_hz': 200.0, 'release_us': 100, 'tagger_delay_us': 0,
          'gate_start_us': 0.0, 'gate_stop_us': 200.0}


def _Scan(loop, setpoints, scan_type='Single scan', bunches=2, timing=TIMING):
    return loop.Send('Start_Scan', setpoints=list(setpoints), scan_type=scan_type,
                     bunches_per_step=bunches, timing=timing)


def test_Every_Instruction_Gets_One_Reply(loop):
    assert loop.Send('Ping') == (True, {'Warnings': []})
    ok, why = loop.Send('Make_Coffee')
    assert not ok and "not an instruction" in why
    ok, why = loop.Send('Start_Scan', setpoints=[0, 1])      # missing parameters
    assert not ok and "failed in the acquisition process" in why
    assert loop.Send('Ping')[0]                              # and the loop is still alive


def test_Calibration_Steps_Write_Then_Stamp(loop):
    plan = Scan_Plan([-10], [10], 10).Setpoints
    assert not loop.Send('Calibration_Step', index=0)[0]     # no calibration in progress
    ok, dac = loop.Send('Calibrate', setpoints=list(plan))
    assert ok and dac['Dac_Status'][0] == 'PASS'
    assert not loop.Send('Calibrate', setpoints=list(plan))[0]      # one at a time
    assert not loop.Send('Calibration_Step', index=len(plan))[0]    # outside the plan
    ok, step = loop.Send('Calibration_Step', index=0)
    assert ok and step['Index'] == 0 and step['Volts'] == -10.0
    assert step['Code'] == mcc_dac.Quantised(step['Dac_V'])[0]
    assert step['Code'] * mcc_dac.LSB_V - mcc_dac.FULL_SCALE_V == pytest.approx(step['Dac_V'])
    assert loop.Writes[-2:] == [(mcc_dac.CH_POS, 0.0), (mcc_dac.CH_NEG, pytest.approx(
        0.01, abs=mcc_dac.LSB_V / 2))]
    assert loop.State.Voltage.value == -10.0
    writes = len(loop.Writes)
    loop.Packets()                                  # time passes; the idle branch must not
    assert len(loop.Writes) == writes               # overwrite the setpoint under the meter
    assert loop.Send('Stop_Calibration') == (True, {})


def test_Nothing_Starts_While_Something_Else_Runs(loop):
    assert loop.Send('Start_Triggering', timing=TIMING)[0]
    ok, why = loop.Send('Calibrate', setpoints=[0.0, 1.0])
    assert not ok and why == "Cannot calibrate while the triggers are running."
    assert not _Scan(loop, [0.0, 1.0])[0]
    assert loop.Send('Stop_Triggering')[0]


def test_A_Count_Of_Triggers_And_Hits_Keeps_Nothing_And_Leaves_Them_Off(loop):
    """The health check's count: every trigger the period gives numbered and
    read, each input's hits counted, nothing sent as data, the triggers off."""
    ok, counted = loop.Send('Count', timing=TIMING, seconds=1.0)
    assert ok and counted['Seconds'] == pytest.approx(1.0, abs=0.1)
    assert counted['Expected'] == pytest.approx(200.0 * counted['Seconds'])
    assert abs(counted['Numbered'] - counted['Expected']) <= 3
    assert counted['Received'] == counted['Numbered']
    assert set(counted['Counts']) <= {1, 2, 3, 4} and sum(counted['Counts'].values()) > 0
    assert loop.Data.empty() and not loop.Pulses._Running
    assert loop.Send('Start_Triggering', timing=TIMING)[0]
    assert not loop.Send('Count', timing=TIMING, seconds=1.0)[0]   # not over a stream
    assert loop.Send('Stop_Triggering')[0]


def test_Nothing_Can_Pulse_The_Bunch_Line_During_A_Calibration(loop):
    # the bunch trigger and the calibration's readings stay apart: while the meter pairs
    # readings with setpoints the PulseBlaster, and the buncher with it, stays off
    assert loop.Send('Calibrate', setpoints=[0.0, 10.0])[0]
    ok, why = loop.Send('Start_Triggering', timing=TIMING)
    assert not ok and why == "Cannot start the triggers while a calibration is running."
    assert not _Scan(loop, [0.0, 10.0])[0]
    assert not loop.Pulses._Running
    assert loop.Send('Stop_Calibration')[0]


def test_Single_Scan_Tags_Each_Hit_With_Its_Setpoint_And_Ends_Marked(loop):
    plan = Scan_Plan([25], [40], 1).Setpoints
    assert loop.Send('Ping')[0]
    time.sleep(0.2)                                 # the idle loop's 0 V is written by now
    start = len(loop.Writes)
    loop.State.Cooler.value, loop.State.Cooler_At.value = 2.99, time.perf_counter()
    assert _Scan(loop, plan, bunches=3)[0]
    packets = loop.Packets(until_end=True)
    assert packets[-1] == ('scan_end', None)
    # each setpoint commanded once, in order, and nothing put 0 V back mid-scan
    live = [volts * mcc_dac.AMPLIFIER_GAIN for _, volts in loop.Writes[start:][1::2]]
    assert live[:len(plan)] == pytest.approx(plan, abs=HALF_CODE_V)
    rows = np.vstack([p[1] for p in packets if p[0] == 'scan'] or [np.empty((0, 6))])
    assert set(rows[:, 1]) <= set(plan)
    assert loop.State.Bunches.value == 3 * len(plan)
    # every trigger read logged, in the tagger's order, with the bunch it became (each
    # once; a dropped one -1) and the cooler reading in force, with its age
    logged = np.vstack([p[1] for p in packets if p[0] == 'triggers'])
    assert np.all(np.diff(logged[:, 0]) > 0)
    assert sorted(logged[logged[:, 1] >= 0, 1]) == list(range(3 * len(plan)))
    assert set(rows[:, 2]) <= set(logged[:, 1]) and np.all(logged[:, 3] == 2.99)
    assert np.all((logged[:, 4] >= 0) & (logged[:, 4] < 30.0))
    assert not loop.State.Scanning.is_set()
    loop.Packets()
    assert loop.State.Voltage.value == 0.0          # the platform back at rest


def test_A_Scan_Given_Its_Rounds_Ends_Itself_After_The_Last(loop):
    """As a queued scan is: a triangle of three rounds, up, down and up, then
    its end marked, with nothing asked of it in between."""
    assert loop.Send('Start_Scan', setpoints=[30.0, 31.0, 32.0], scan_type='Triangle',
                     bunches_per_step=1, timing=TIMING, rounds=3)[0]
    packets = loop.Packets(until_end=True)
    assert packets[-1] == ('scan_end', None)
    assert loop.State.Round.value == 3 and loop.State.Bunches.value == 3 * 3
    assert not loop.State.Scanning.is_set()
    ok, why = loop.Send('Start_Scan', setpoints=[30.0, 31.0], scan_type='Sawtooth',
                        bunches_per_step=1, timing=TIMING, rounds=0)
    assert not ok and 'needs one at least' in why


def test_Triangle_Reverses_And_Stop_After_Scan_Ends_The_Round(loop):
    assert _Scan(loop, [30.0, 31.0, 32.0], scan_type='Triangle', bunches=1)[0]
    while loop.State.Round.value < 3:
        time.sleep(0.01)
    assert loop.Send('Stop_After_Scan')[0]
    last_round = loop.State.Round.value
    loop.Packets(until_end=True)
    assert loop.State.Round.value == last_round and not loop.State.Scanning.is_set()
    commanded = [volts * mcc_dac.AMPLIFIER_GAIN for _, volts in loop.Writes if volts]
    assert commanded[:5] == pytest.approx([30, 31, 32, 31, 30], abs=HALF_CODE_V)


def test_Dc_Mode_Records(loop):
    # the old DAQ's DC path read a counter stub that always returned 0
    timing = dict(TIMING, bunched=False)
    assert _Scan(loop, [31.0, 32.0], bunches=200, timing=timing)[0]
    packets = loop.Packets(until_end=True)
    assert sum(len(rows) for kind, rows in packets if kind == 'scan') > 0


def test_Stop_Now_Lands_Within_A_Bunch(loop):
    assert _Scan(loop, [30.0, 31.0], scan_type='Sawtooth', bunches=100000)[0]
    assert loop.Send('Stop_Now')[0]
    assert loop.Packets(until_end=True, timeout=2.0)[-1] == ('scan_end', None)


def _Rows(packets):
    return np.vstack([rows for kind, rows in packets if kind == 'scan'] or [np.empty((0, 6))])


def test_A_Trigger_Outage_Holds_The_Step_And_Says_Why(loop, monkeypatch):
    # the collinear DAQ's D5: a read that timed out moved the scan on with no trigger behind it
    monkeypatch.setattr(acquisition, 'NO_TRIGGER_S', 0.3)
    assert _Scan(loop, [30.0, 31.0], bunches=40)[0]
    time.sleep(0.1)
    loop.Pulses.Stop()                              # the triggers stop reaching the tagger
    time.sleep(0.1)                                 # what was in flight is read
    held = loop.State.Bunches.value
    time.sleep(1.5)
    assert loop.State.Bunches.value == held         # no trigger, no bunch
    assert loop.State.Scanning.is_set() and loop.State.Silent.value > 0.3
    loop.Pulses.Start()
    rows = _Rows(loop.Packets(until_end=True))
    assert loop.State.Silent.value == 0.0 and loop.State.Skipped.value == 0
    # each step holds its own forty bunches, every one a trigger the tagger saw
    assert set(np.unique(rows[:, 2])) <= set(range(80))
    assert np.all(rows[rows[:, 2] < 40, 1] == 30.0) and np.all(rows[rows[:, 2] >= 40, 1] == 31.0)


def test_The_Outage_Sentence_Names_The_Likely_Cause():
    sentence = acquisition.No_Trigger(12.4)
    assert sentence.startswith("No trigger has reached the time tagger for 12 s")
    assert "possible malfunction or misconnection of PulseBlaster channel 18" in sentence


def test_A_Cycle_Longer_Than_A_Read_Still_Takes_Exactly_Its_Bunches(loop):
    # with a dwell of up to 100 s, most reads end with no trigger yet; none is a bunch
    slow = dict(TIMING, rep_rate_hz=1.0 / (3 * acquisition.READ_WAIT_S))
    assert _Scan(loop, [30.0, 31.0], bunches=2, timing=slow)[0]
    rows = _Rows(loop.Packets(until_end=True))
    assert loop.State.Bunches.value == 4 and loop.State.Skipped.value == 0
    assert np.all(rows[rows[:, 2] < 2, 1] == 30.0) and np.all(rows[rows[:, 2] >= 2, 1] == 31.0)


def test_A_Read_Holding_Several_Triggers_Takes_Each_As_Its_Own_Bunch(loop, monkeypatch):
    # the tagger hands over every trigger it has ready, so a loop that falls behind gets
    # several in one read. Counted as one bunch, as the old DAQ counted reads, a step ran
    # on more triggers than its quota and their hits shared one bunch number
    take = loop.Loop._Take_Instructions
    monkeypatch.setattr(loop.Loop, '_Take_Instructions',
                        lambda wait=0.0: (time.sleep(0.02), take(wait)))
    setpoints = [30.0, 31.0, 32.0]
    assert _Scan(loop, setpoints, bunches=20)[0]            # a 5 ms cycle, read every 20 ms
    rows = _Rows(loop.Packets(until_end=True))
    state, leftover = loop.State, loop.Tagger.Drain()
    assert state.Bunches.value == 60 and state.Skipped.value == 0
    # every trigger the tagger saw was taken as one bunch or dropped, none merged
    assert loop.Tagger._Bunch == state.Bunches.value + state.Dropped.value + leftover
    for k, volts in enumerate(setpoints):
        assert np.all(rows[(rows[:, 2] >= 20 * k) & (rows[:, 2] < 20 * (k + 1)), 1] == volts)


def test_A_Stray_Stop_Calibration_Leaves_The_Dac_Alone(loop):
    # the GUI stops any 'Calibrate' that failed, and a running scan is one reason it fails:
    # with no calibration in progress the stop must not ramp the platform anywhere
    assert loop.Send('Output_Volts', on=True)[0]
    loop.State.Requested.value = 300.0
    loop.Packets()
    writes = len(loop.Writes)
    assert loop.Send('Stop_Calibration', ramp_step_v=50.0, ramp_rate_v_s=1000.0) == (True, {})
    loop.Packets()
    assert len(loop.Writes) == writes and loop.State.Voltage.value == 300.0


def test_Shutdown_Leaves_The_Platform_At_Zero(loop):
    assert loop.Send('Output_Volts', on=True)[0]
    loop.State.Requested.value = 1500.0
    loop.Packets()
    assert loop.State.Voltage.value == 1500.0
    loop.Close()
    assert loop.Writes[-2:] == [(mcc_dac.CH_NEG, 0.0), (mcc_dac.CH_POS, 0.0)]


def test_An_Orphaned_Loop_Stops():
    parent = threading.Event()
    parent.set()
    bench = Loop_Bench(parent.is_set)
    assert bench.Send('Ping')[0]
    parent.clear()                                  # the GUI's process is gone
    bench.Thread.join(timeout=2)
    assert not bench.Thread.is_alive()
    bench.Close()


def test_A_Refused_Request_Turns_The_Output_Off(loop):
    assert loop.Send('Output_Volts', on=True)[0]
    loop.State.Requested.value = 20000.0            # beyond the converters' range
    loop.Packets()
    assert not loop.State.Outputting.is_set()


def test_The_Link_Talks_To_A_Spawned_Process():
    from multiprocessing import get_context
    from cls_acquisition import settings
    from cls_acquisition.hardware import Rack
    link = Rack_Link(Rack(settings.DEFAULTS, get_context('spawn')).Loop_Spec())
    assert link.Send('Ping', timeout=30) == {'Warnings': []}
    assert link.Send('Identity') == {'Dac': {'Board': 'simulated USB-231', 'UniqueId': None},
                                     'Tagger': {'Wrapper': 'simulated', 'Serial': None}}
    with pytest.raises(Acquisition_Error, match="not an instruction"):
        link.Send('Make_Coffee')
    link.Close()
    assert not link.Alive()
    with pytest.raises(Acquisition_Error, match="has stopped"):
        link.Send('Ping')


def test_A_Timing_That_Does_Not_Fit_Is_Refused_With_Its_Sentence_And_Nothing_Moves(loop):
    loop.Send('Calibrate', setpoints=[0.0, 10.0])
    loop.Send('Calibration_Step', index=1)
    loop.Send('Stop_Calibration')
    writes = len(loop.Writes)
    late = dict(TIMING, release_us=10000)                   # a 5 ms cycle at 200 Hz
    ok, why = _Scan(loop, [20.0, 30.0], timing=late)
    assert not ok and "bunch release at 10000 µs" in why and "not started" in why
    ok, why = loop.Send('Start_Triggering', timing=dict(TIMING, gate_stop_us=6000.0))
    assert not ok and "closes 6000 µs into the 5000 µs cycle" in why
    state = loop.State
    assert not state.Scanning.is_set() and not state.Triggering.is_set()
    assert state.Round.value == 0
    time.sleep(0.2)
    # the idle loop puts 0 V back after the calibration, and nothing else is written
    assert all(volts == 0.0 for _, volts in loop.Writes[writes:])
