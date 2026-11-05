"""The VISA session and the Keysight drivers, on the simulated meter."""

import itertools
import time

import numpy as np
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


class External_Keysight(Counting_Keysight):
    """A meter on its trigger input that will not leave it, even for *RST."""

    DEFAULTS = dict(simulated.Keysight.DEFAULTS, **{'TRIG:SOUR': 'EXT'})

    def __init__(self):
        super().__init__(ignore='TRIG:SOUR')


def test_A_Meter_On_Its_Trigger_Input_Is_Refused_In_Every_Mode():
    # the line meter's trigger input is the buncher's line: on it, the meter reads on the
    # bunch clock rather than when asked, which is how the old DAQ's readings slipped
    with pytest.raises(Scpi_Error, match="TRIG:SOUR EXT"):
        _Meter(External_Keysight())
    meter, inst = _Meter(Counting_Keysight())
    inst.Ignore, inst._Settings['TRIG:SOUR'] = 'TRIG:SOUR', 'EXT'
    with pytest.raises(Scpi_Error, match="TRIG:SOUR EXT"):
        meter.Capture(5, aperture_s=0.0002)
    meter.Mode = 'continuous'
    with pytest.raises(Scpi_Error, match="TRIG:SOUR reads EXT"):
        meter.Step_Config()
    inst.Ignore = None
    meter.Release()
    assert meter.Mode == 'continuous'
    meter.Close()


def test_Every_Reading_In_Every_Mode_Is_Taken_On_Command():
    sent = []

    class Recording(simulated.Keysight):

        def write(self, command):
            sent.append(command.strip().upper())
            super().write(command)

    meter, _ = _Meter(Recording(lambda: 1.0e-3))
    meter.Step_Config()
    meter.Read_Settled(0.0)
    meter.Release()
    meter.Capture(3, interval_s=0.001, aperture_s=0.0002)
    meter.Arm()
    meter.Fetch_When_Done(3, 5.0)
    meter.Release()
    meter.Capture(3, nplc=1, stream=True)
    meter.Release()
    meter.Close()
    assert {c for c in sent if c.startswith('TRIG:SOUR')} == {'TRIG:SOUR IMM'}


def test_Cooler_Meter_Polls():
    meter, inst = _Meter(Counting_Keysight(), keysight.Cooler_Meter)
    time.sleep(0.3)
    assert inst.Reads > 0
    meter.Close()


def test_An_Overloaded_Reading_Is_Never_Taken_For_A_Voltage():
    # 9.9E37 is the meter's mark for a reading beyond its range
    meter, inst = _Meter(simulated.Keysight(lambda: 50.0))
    time.sleep(0.3)
    assert np.isnan(meter.Voltage)
    meter.Step_Config()
    with pytest.raises(Scpi_Error, match="marked 1 of 1 readings .* over range or invalid"):
        meter.Read_Settled(0.0)
    meter.Release()
    meter.Close()


def test_Only_A_Truevolt_Meter_Is_Taken_And_Its_Memory_Is_Asked():

    class Other(simulated.Keysight):

        def query(self, command):
            if command == '*IDN?':
                return 'Agilent Technologies,34401A,0,1\n'
            return super().query(command)

    with pytest.raises(Scpi_Error, match="34401A.*not a Keysight 3446x or 3447x"):
        _Meter(Other(lambda: 0.0))

    class With_Memory(simulated.Keysight):

        def query(self, command):
            return 'MEM,DIG\n' if command == '*OPT?' else super().query(command)

    meter, _ = _Meter(With_Memory(lambda: 0.0))
    assert meter.Memory == keysight.MEM_MEMORY
    meter.Close()


def test_A_Timed_Capture_Uses_The_Interval_The_Meter_Set():
    meter, inst = _Meter(simulated.Keysight(lambda: 1e-3))
    # at a 1 ms aperture the meter cannot sample every 0.5 ms
    interval = meter.Capture(10, interval_s=0.0005, aperture_s=0.001)
    assert interval == pytest.approx(0.001 + 2e-5) == meter.Capture_Interval_S
    assert inst._Settings['VOLT:ZERO:AUTO'] == 'OFF'
    meter.Release()
    meter.Close()


def test_A_Streamed_Capture_Is_Read_Out_While_It_Runs():
    meter, _ = _Meter(simulated.Keysight(lambda: 3e-3))
    meter.Capture(60, interval_s=0.002, aperture_s=0.0002, stream=True)
    meter.Autocal()
    meter.Arm()
    values, deadline = [], time.monotonic() + 5.0
    while len(values) < 60 and time.monotonic() < deadline:
        values.extend(meter.Read_Available(25))
        time.sleep(0.02)
    assert len(values) == 60 and np.allclose(values, 3e-3)
    assert len(meter.Read_Available(25)) == 0       # erased as it was read
    meter.Release()
    meter.Close()


def test_A_Meter_Is_Left_Reset_As_It_Closes():
    """As the bench tool leaves it, so the next program finds it at power-on."""

    class Recording(simulated.Keysight):
        def __init__(self):
            super().__init__(lambda: 1e-3)
            self.Written = []

        def write(self, command):
            self.Written.append(command)
            super().write(command)

    meter, inst = _Meter(Recording())
    meter.Capture(10, 0.001, aperture_s=2e-4)
    inst.Written.clear()
    meter.Close()
    assert inst.Written == ['ABOR', '*RST', '*CLS']
