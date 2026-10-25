"""The VISA session and the Keysight drivers, on the simulated meter."""

import itertools
import time

import pytest

from cls_acquisition.calibration import Calibration_Error
from cls_acquisition.hardware import keysight, simulated
from cls_acquisition.hardware.visa_session import Instrument_IO_Error, Scpi_Error, Visa_Session

_Names = (f"SIM::{i}::INSTR" for i in itertools.count())


class Counting_Keysight(simulated.Keysight):
    """Counts READ? queries, and can return a burst, fail, or silently
    ignore the commands that start with ``ignore``."""

    def __init__(self, burst=1, fail=False, ignore=None):
        super().__init__(lambda: 1.2345e-3)
        self.Reads, self.Burst, self.Fail, self.Ignore = 0, burst, fail, ignore

    def write(self, command):
        if not (self.Ignore and command.startswith(self.Ignore)):
            super().write(command)

    def query(self, command):
        if self.Fail:
            raise Instrument_IO_Error("The instrument at SIM did not answer: timeout.")
        if command.strip().upper() == 'READ?':
            self.Reads += 1
            if self.Burst > 1:
                return ','.join(['+1.0E-03'] * self.Burst)
        return super().query(command)


def _Meter(inst, kind=keysight.Line_Meter):
    return kind(Visa_Session.Open(next(_Names), lambda _: inst)), inst


def test_One_Session_Per_Instrument_Until_The_Last_Holder_Closes():
    name = next(_Names)
    first = Visa_Session.Open(name, lambda _: simulated.Keysight(lambda: 0.0))
    second = Visa_Session.Open(name, lambda _: pytest.fail("a second session was opened"))
    assert first is second
    first.Close()
    assert second.Alive and second.Transaction(lambda inst: inst.Query('*OPC?')) == '1'
    second.Close()
    assert not second.Alive


def test_Instrument_Errors_Reach_The_Caller_Named():
    session = Visa_Session.Open(next(_Names), lambda _: Counting_Keysight(fail=True))
    with pytest.raises(Instrument_IO_Error, match="did not answer"):
        session.Transaction(lambda inst: inst.Query('READ?'))
    session.Close()


def test_Poll_Reads_The_Meter():
    meter, inst = _Meter(Counting_Keysight())
    time.sleep(0.3)
    assert meter.Voltage == pytest.approx(1.2345e-3) and inst.Reads > 0
    meter.Close()


def test_Stepping_Holds_Off_The_Poll_And_Release_Restores_It():
    meter, inst = _Meter(Counting_Keysight())
    meter.Step_Config()
    reads = inst.Reads
    time.sleep(0.3)
    assert inst.Reads == reads                      # no READ? of the poll's own
    stamp, value = meter.Read_Settled(0.01)
    assert inst.Reads == reads + 1 and value == pytest.approx(1.2345e-3)
    assert stamp <= time.perf_counter_ns()
    meter.Release()
    meter.Release()                                 # a second release is harmless
    assert meter.Mode == 'continuous'
    time.sleep(0.3)
    assert inst.Reads > reads + 1
    meter.Close()


def test_Illegal_Moves_Are_Refused():
    meter, _ = _Meter(Counting_Keysight())
    with pytest.raises(Calibration_Error, match="not stepping"):
        meter.Read_Settled(0.0)
    meter.Step_Config()
    with pytest.raises(RuntimeError, match="cannot go from stepping to stepping"):
        meter.Step_Config()
    meter.Release()
    meter.Close()


def test_A_Burst_Where_One_Reading_Was_Asked_For_Is_Refused():
    # the ungated burst of Mode A, in its new form
    meter, _ = _Meter(Counting_Keysight(burst=5))
    meter.Step_Config()
    with pytest.raises(Calibration_Error, match="returned 5 values"):
        meter.Read_Settled(0.0)
    meter.Release()
    meter.Close()


def test_A_Setting_The_Meter_Did_Not_Take_Is_Refused():
    meter, inst = _Meter(Counting_Keysight(ignore='SAMP:COUN'))
    inst._Settings['SAMP:COUN'] = '3'
    with pytest.raises(Scpi_Error, match="SAMP:COUN 3"):
        meter.Step_Config()
    meter.Release()
    assert meter.Mode == 'continuous'
    meter.Close()


def test_Cooler_Meter_Polls():
    meter, inst = _Meter(Counting_Keysight(), keysight.Cooler_Meter)
    time.sleep(0.3)
    assert inst.Reads > 0
    meter.Close()
