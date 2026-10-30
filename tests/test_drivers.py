"""The DAC routing, the trigger patterns, and the vendor drivers on stub vendor modules.

The vendor libraries (mcculw, spinapi, timetagger4, pyvisa) are replaced
in sys.modules by recording stubs, as the old DAQ's digital twin did, so
each driver is checked against the calls the rack's libraries expect.
"""

import sys
import time
import types

import pytest

from cls_acquisition import triggers
from cls_acquisition.hardware import mcc_dac, pulseblaster, timetagger, visa_session
from cls_acquisition.hardware.mcc_dac import CH_NEG, CH_POS, Dac


def _Dac():
    calls = []
    return Dac(lambda channel, volts: calls.append((channel, volts))), calls


def test_Zero_Goes_Down_The_Positive_Branch():
    dac, _ = _Dac()
    assert dac.Write(0.0) == (CH_POS, 0.0)


def test_The_Outgoing_Converter_Is_Released_First():
    dac, calls = _Dac()
    dac.Write(-1500.0)
    assert calls == [(CH_POS, 0.0), (CH_NEG, pytest.approx(1.5, abs=mcc_dac.LSB_V / 2))]
    calls.clear()
    dac.Write(2500.0)
    assert calls == [(CH_NEG, 0.0), (CH_POS, pytest.approx(2.5, abs=mcc_dac.LSB_V / 2))]


def test_Writes_Land_On_The_Codes_The_Board_Rounds_To():
    # the bench's records: from_eng_units rounds (v + 10) / 20 * 65536 to the nearest code
    assert mcc_dac.Quantised(0.0) == (32768, 0.0)
    code, volts = mcc_dac.Quantised(1.0004)
    assert code == round(11.0004 / 20 * 65536) and volts == pytest.approx(code * 20 / 65536 - 10)
    assert mcc_dac.Quantised(12.0)[0] == 65535 and mcc_dac.Quantised(-12.0)[0] == 0
    dac, calls = _Dac()
    channel, written = dac.Write(1000.0)
    assert written == calls[-1][1] == mcc_dac.Quantised(1.0)[1]


def test_Beyond_The_Limit_Or_Full_Scale_Is_Refused_Before_Anything_Is_Written():
    dac, calls = _Dac()
    with pytest.raises(ValueError, match="beyond the 4000 V limit"):
        dac.Write(-4000.5)
    dac.Limit_V = 20000.0
    with pytest.raises(ValueError, match="beyond its 10 V range"):
        dac.Write(10001.0)
    assert calls == []


def test_The_Non_Linearity_Table_Is_Taken_Off_And_Never_Extrapolated():
    knots = [[100.0, 200.0, 300.0], [0.0, 0.5, 0.0]]
    correction = dict(mcc_dac.CORRECTION, measured='2026-10-26', valid_days=1,
                      inl={CH_POS: knots})
    calls = []
    dac = Dac(lambda channel, volts: calls.append((channel, volts)), correction)
    dac.Write(200.0)
    # 0.5 V of the converter's own at 200 V is commanded away
    assert calls[-1][1] * mcc_dac.AMPLIFIER_GAIN == pytest.approx(199.5, abs=0.2)
    dac.Write(-50.0)                               # the negative branch has no table
    with pytest.raises(ValueError, match="outside the 75 to 325 V"):
        dac.Write(400.0)
    with pytest.raises(ValueError, match="positive converter's non-linearity"):
        dac.Check([0.0, 60.0])
    assert "INL pos 3 knots 100 to 300 V" in dac.Correction_Record()[1]
    # 0 V is where the platform rests: the idle loop must always be able to write it
    dac.Write(0.0)


def test_A_Target_The_Table_Carries_Across_Zero_Keeps_Its_Sign():
    # 0.5 V of the converter's own at 0.2 V: the target is -0.3 V, driven as such
    correction = dict(mcc_dac.CORRECTION, measured='2026-10-26', valid_days=1,
                      inl={CH_POS: [[0.0, 50.0], [0.5, 0.5]]})
    calls = []
    dac = Dac(lambda channel, volts: calls.append((channel, volts)), correction)
    dac.Write(0.2)
    assert calls[-1][0] == CH_POS
    assert calls[-1][1] * mcc_dac.AMPLIFIER_GAIN == pytest.approx(-0.3, abs=0.16)


def test_An_Expired_Correction_Is_Refused():
    correction = dict(mcc_dac.CORRECTION, measured='2026-01-01', valid_days=30)
    dac = Dac(lambda *_: None, correction)
    assert Dac(lambda *_: None).Calibration_Status()[0] == 'PASS'
    later = time.mktime((2026, 10, 25, 0, 0, 0, 0, 0, -1))
    assert dac.Calibration_Status(now=later)[0] == 'REJECT'
    assert dac.Correction_Record()[0] == 1


def _Old_Time_Slices(line_pattern):
    """The old DAQ's transformLinesToTimeSlices, as the oracle."""
    boundaries = sorted(set([0] + [t for line in line_pattern for t in line[1:3]]))
    slices = []
    for i in range(len(boundaries) - 1):
        slices.append([boundaries[i + 1] - boundaries[i], 0])
        for k in line_pattern:
            if k[1] <= boundaries[i] and k[2] >= boundaries[i + 1]:
                slices[-1][1] |= 1 << (k[0] - 1)
    return slices


@pytest.mark.parametrize('pattern', [triggers.Bunched_Pattern(10.0, 100, 20),
                                     triggers.Dc_Pattern()])
def test_Time_Slices_Match_The_Old_Transform(pattern):
    assert triggers.Time_Slices(pattern) == _Old_Time_Slices(pattern)


def test_Bunched_Pattern_Follows_The_Dwell():
    pattern = triggers.Bunched_Pattern(10.0, 100, 20)
    assert triggers.Period_Ns(pattern) == pytest.approx(1e8)
    assert [triggers.CH_TAGGER, 20000, 20100] in pattern


# ── Vendor drivers on stubs ─────────────────────────────────────────────

class Spinapi_Stub(types.ModuleType):

    def __init__(self, init_status=0):
        super().__init__('spinapi.spinapi')
        self.Calls, self._Init = [], init_status

    def __getattr__(self, name):
        if not name.startswith('pb_'):
            raise AttributeError(name)

        def call(*args):
            self.Calls.append((name, args))
            if name == 'pb_init':
                return self._Init
            if name == 'pb_inst_pbonly':
                return len([c for c in self.Calls if c[0] == 'pb_inst_pbonly']) - 1
            return 'stub error' if name == 'pb_get_error' else 0
        return call


def _Install_Spinapi(monkeypatch, stub):
    package = types.ModuleType('spinapi')
    package.spinapi = stub
    monkeypatch.setitem(sys.modules, 'spinapi', package)
    monkeypatch.setitem(sys.modules, 'spinapi.spinapi', stub)


def test_PulseBlaster_Programs_A_Loop_Of_Slices(monkeypatch):
    stub = Spinapi_Stub()
    _Install_Spinapi(monkeypatch, stub)
    blaster = pulseblaster.Pulse_Blaster()
    stub.Calls.clear()
    pattern = triggers.Bunched_Pattern(10.0, 100, 20)
    blaster.Program(pattern)
    instructions = [args for name, args in stub.Calls if name == 'pb_inst_pbonly']
    slices = triggers.Time_Slices(pattern)
    assert [(i[0], i[3]) for i in instructions] == [(bits, d) for d, bits in slices]
    assert instructions[-1][1:3] == (pulseblaster.BRANCH, 0)    # back to the first
    blaster.Stop()
    assert stub.Calls[-1][0] == 'pb_start'


def test_PulseBlaster_That_Will_Not_Start_Says_So(monkeypatch):
    _Install_Spinapi(monkeypatch, Spinapi_Stub(init_status=-1))
    with pytest.raises(RuntimeError, match="did not initialise: stub error"):
        pulseblaster.Pulse_Blaster()


class Card_Stub():
    Made = []

    def __init__(self, **options):
        self.Options, self.Replies = options, [(1, []), (0, [[7, 3, 2000], [7, 4, 4000]])]
        Card_Stub.Made.append(self)

    def getPackets(self):
        return self.Replies.pop(0) if self.Replies else (1, [])

    def startReading(self):
        pass

    def stopReading(self):
        pass

    def stop(self):
        pass


def test_Tagger_Configures_Each_Input_And_Reads_Microseconds(monkeypatch):
    monkeypatch.setitem(sys.modules, 'timetagger4', types.SimpleNamespace(TimeTagger=Card_Stub))
    tagger = timetagger.Time_Tagger([-0.2, 0.1], 2.5)
    tagger.Configure(10.0, 210.0)
    tagger.Start()
    options = Card_Stub.Made[-1].Options
    assert options['channel_0_rising'] is False and options['channel_1_rising'] is True
    assert (options['channel_1_start'], options['channel_1_stop']) == (20000, 420000)
    rows = tagger.Read(timeout=1.0)
    assert rows.tolist() == [[7, 3, 1.0], [7, 4, 2.0]]
    assert tagger.Read(timeout=0.01) is None                    # no trigger: a timeout
    tagger.Stop()
    tagger.Stop()


def test_The_Tagger_Window_Stays_Inside_The_Timestamp_Range(monkeypatch):
    monkeypatch.setitem(sys.modules, 'timetagger4', types.SimpleNamespace(TimeTagger=Card_Stub))
    tagger = timetagger.Time_Tagger([-0.2, 0.1], 2.5)
    with pytest.raises(ValueError, match="8388.6 us timestamp range"):
        tagger.Configure(0.0, 9000.0)
    with pytest.raises(ValueError, match="does not fit"):
        tagger.Configure(50.0, 50.0)


def test_The_Old_Wrappers_Edge_Fault_Is_Said_At_Start_Up(monkeypatch):
    rack_levels = [-0.2, -0.6, 0.1, 0.1]
    old = types.SimpleNamespace(TimeTagger=Card_Stub)
    monkeypatch.setitem(sys.modules, 'timetagger4', old)
    [warning] = timetagger.Time_Tagger(rack_levels, 2.5).Warnings
    assert "input 3 and 4 on input 2's falling edge" in warning
    corrected = types.SimpleNamespace(TimeTagger=Card_Stub, ORACLE_WRAPPER=1)
    monkeypatch.setitem(sys.modules, 'timetagger4', corrected)
    assert timetagger.Time_Tagger(rack_levels, 2.5).Warnings == []
    assert timetagger.Edge_Fault([-0.2, -0.6, -0.1, -0.3], False) is None


class Ul_Error(Exception):

    def __init__(self, errorcode, message):
        super().__init__(message)
        self.errorcode, self.message = errorcode, message


def _Install_Mcculw(monkeypatch, name='USB-231', fail_write=False):
    calls = []

    def a_out(*args):
        if fail_write:
            raise Ul_Error(19, "Device not responding")
        calls.append(args)

    ul = types.SimpleNamespace(
        ULError=Ul_Error, get_board_name=lambda board: name, a_out=a_out,
        from_eng_units=lambda board, rng, volts: round((volts + 10) / 20 * 65536))
    enums = types.SimpleNamespace(ULRange=types.SimpleNamespace(BIP10VOLTS='BIP10'))
    monkeypatch.setitem(sys.modules, 'mcculw', types.SimpleNamespace(ul=ul, enums=enums))
    monkeypatch.setitem(sys.modules, 'mcculw.enums', enums)
    return calls


def test_Mcc_Writes_Through_The_Board_And_Starts_At_Zero(monkeypatch):
    calls = _Install_Mcculw(monkeypatch)
    dac = mcc_dac.Open_Mcc(board=1)
    assert calls == [(1, CH_NEG, 'BIP10', 32768), (1, CH_POS, 'BIP10', 32768)]
    dac.Write(500.0)
    assert calls[-1] == (1, CH_POS, 'BIP10', round(10.5 / 20 * 65536))


def test_Mcc_Refuses_Another_Board_And_Names_A_Failed_Write(monkeypatch):
    _Install_Mcculw(monkeypatch, name='USB-1208FS')
    with pytest.raises(mcc_dac.Dac_Error, match="board 1 is a USB-1208FS, not the USB-231"):
        mcc_dac.Open_Mcc(board=1)
    _Install_Mcculw(monkeypatch, fail_write=True)
    with pytest.raises(mcc_dac.Dac_Error, match="failed to write .* Device not responding "
                                                r"\(UL error 19\)"):
        mcc_dac.Open_Mcc(board=1)


def _Install_Pyvisa(monkeypatch, locked=False):

    class Visa_Io_Error(Exception):
        pass

    class Resource():
        timeout = 0
        Calls = []

        def lock_excl(self, timeout):
            if locked:
                raise Visa_Io_Error("VI_ERROR_RSRC_LOCKED")
            self.Calls.append('lock')

        def unlock(self):
            self.Calls.append('unlock')

        def close(self):
            self.Calls.append('close')

        def query(self, command):
            raise Visa_Io_Error("VI_ERROR_TMO")

    manager = types.SimpleNamespace(open_resource=lambda name: Resource())
    monkeypatch.setitem(sys.modules, 'pyvisa', types.SimpleNamespace(
        ResourceManager=lambda: manager, errors=types.SimpleNamespace(VisaIOError=Visa_Io_Error)))
    monkeypatch.setattr(visa_session, '_Manager', None)
    return Resource


def test_Pyvisa_Io_Errors_Become_Instrument_Io_Errors(monkeypatch):
    resource = _Install_Pyvisa(monkeypatch)
    inst = visa_session.Pyvisa_Connect('USB0::STUB::INSTR')
    with pytest.raises(visa_session.Instrument_IO_Error, match="USB0::STUB::INSTR did not answer"):
        inst.query('READ?')
    inst.close()
    assert resource.Calls == ['lock', 'unlock', 'close']


def test_A_Meter_Another_Program_Holds_Is_Refused_By_Name(monkeypatch):
    resource = _Install_Pyvisa(monkeypatch, locked=True)
    with pytest.raises(visa_session.Instrument_IO_Error, match="held by another program"):
        visa_session.Pyvisa_Connect('USB0::STUB::INSTR')
    assert resource.Calls == ['close']


# ── Timings that do not fit their cycle ─────────────────────────────────

TIMING = {'bunched': True, 'rep_rate_hz': 100.0, 'release_us': 100, 'tagger_delay_us': 0,
          'gate_start_us': 0.0, 'gate_stop_us': 200.0}


@pytest.mark.parametrize('change, refusal', [
    ({'release_us': 10000}, "bunch release at 10000 µs is not inside the 10 ms cycle"),
    ({'tagger_delay_us': 9999.95}, "tagger's trigger at 9999.95 µs is not inside"),
    ({'tagger_delay_us': 5000, 'gate_stop_us': 5000.1}, "closes 10000.1 µs into the 10000 µs"),
    ({'bunched': False, 'gate_stop_us': 1000.5}, "closes 1000.5 µs into the 1000 µs cycle"),
])
def test_A_Timing_Whose_Edges_Leave_The_Cycle_Is_Refused(change, refusal):
    with pytest.raises(ValueError, match=refusal):
        triggers.Pattern(dict(TIMING, **change))


def test_A_Timing_Up_To_The_Cycles_End_Is_Kept_As_The_Old_Daq_Had_It():
    edge = dict(TIMING, release_us=9999, tagger_delay_us=5000, gate_stop_us=5000)
    assert triggers.Pattern(edge) == triggers.Bunched_Pattern(100.0, 9999, 5000)
    assert triggers.Period_Ns(triggers.Pattern(edge)) == pytest.approx(1e7)
    assert triggers.Pattern(dict(TIMING, bunched=False, gate_stop_us=1000.0)) == (
        triggers.Dc_Pattern())
