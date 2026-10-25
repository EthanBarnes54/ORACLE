"""The acquisition loop's instruction protocol, scan path and calibration steps,
and the link to it across a real process."""

import threading
import time

import numpy as np
import pytest

from cls_acquisition.acquisition import Acquisition_Error, Rack_Link
from cls_acquisition.hardware import mcc_dac
from cls_acquisition.scan_plan import Scan_Plan
from tests.bench import Loop_Bench

TIMING = {'bunched': True, 'rep_rate_hz': 200.0, 'release_us': 100, 'tagger_delay_us': 0,
          'gate_start_us': 0.0, 'gate_stop_us': 200.0}


def _Scan(loop, setpoints, scan_type='Single scan', bunches=2, timing=TIMING):
    return loop.Send('Start_Scan', setpoints=list(setpoints), scan_type=scan_type,
                     bunches_per_step=bunches, timing=timing)


def test_Every_Instruction_Gets_One_Reply(loop):
    assert loop.Send('Ping') == (True, {})
    ok, why = loop.Send('Make_Coffee')
    assert not ok and "not an instruction" in why
    ok, why = loop.Send('Start_Scan', setpoints=[0, 1])      # missing parameters
    assert not ok and "failed in the acquisition process" in why
    assert loop.Send('Ping') == (True, {})                   # and the loop is still alive


def test_Calibration_Steps_Write_Then_Stamp(loop):
    plan = Scan_Plan([-10], [10], 10).Setpoints
    assert not loop.Send('Calibration_Step', index=0)[0]     # no calibration in progress
    ok, dac = loop.Send('Calibrate', setpoints=list(plan))
    assert ok and dac['Dac_Status'][0] == 'PASS'
    assert not loop.Send('Calibrate', setpoints=list(plan))[0]      # one at a time
    assert not loop.Send('Calibration_Step', index=len(plan))[0]    # outside the plan
    ok, step = loop.Send('Calibration_Step', index=0)
    assert ok and step['Index'] == 0 and step['Volts'] == -10.0
    assert loop.Writes[-2:] == [(mcc_dac.CH_POS, 0.0), (mcc_dac.CH_NEG, 0.01)]
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


def test_Single_Scan_Tags_Each_Hit_With_Its_Setpoint_And_Ends_Marked(loop):
    plan = Scan_Plan([25], [40], 1).Setpoints
    assert loop.Send('Ping')[0]
    time.sleep(0.2)                                 # the idle loop's 0 V is written by now
    start = len(loop.Writes)
    assert _Scan(loop, plan, bunches=3)[0]
    packets = loop.Packets(until_end=True)
    assert packets[-1] == ('scan_end', None)
    # each setpoint commanded once, in order, and nothing put 0 V back mid-scan
    live = [volts * mcc_dac.AMPLIFIER_GAIN for _, volts in loop.Writes[start:][1::2]]
    assert live[:len(plan)] == pytest.approx(plan)
    rows = np.vstack([p[1] for p in packets[:-1]] or [np.empty((0, 6))])
    assert set(rows[:, 1]) <= set(plan)
    assert loop.State.Bunches.value == 3 * len(plan)
    assert not loop.State.Scanning.is_set()
    loop.Packets()
    assert loop.State.Voltage.value == 0.0          # the platform back at rest


def test_Triangle_Reverses_And_Stop_After_Scan_Ends_The_Round(loop):
    assert _Scan(loop, [30.0, 31.0, 32.0], scan_type='Triangle', bunches=1)[0]
    while loop.State.Round.value < 3:
        time.sleep(0.01)
    assert loop.Send('Stop_After_Scan')[0]
    last_round = loop.State.Round.value
    loop.Packets(until_end=True)
    assert loop.State.Round.value == last_round and not loop.State.Scanning.is_set()
    commanded = [volts * mcc_dac.AMPLIFIER_GAIN for _, volts in loop.Writes if volts]
    assert commanded[:5] == pytest.approx([30, 31, 32, 31, 30])


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
    assert bench.Send('Ping') == (True, {})
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
    assert link.Send('Ping', timeout=30) == {}
    with pytest.raises(Acquisition_Error, match="not an instruction"):
        link.Send('Make_Coffee')
    link.Close()
    assert not link.Alive()
    with pytest.raises(Acquisition_Error, match="has stopped"):
        link.Send('Ping')
