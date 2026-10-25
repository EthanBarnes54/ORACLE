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
    assert calls == [(CH_POS, 0.0), (CH_NEG, 1.5)]
    calls.clear()
    dac.Write(2500.0)
    assert calls == [(CH_NEG, 0.0), (CH_POS, 2.5)]


def test_Beyond_The_Limit_Or_Full_Scale_Is_Refused_Before_Anything_Is_Written():
    dac, calls = _Dac()
    with pytest.raises(ValueError, match="beyond the 6000 V limit"):
        dac.Write(-6000.5)
    dac.Limit_V = 20000.0
    with pytest.raises(ValueError, match="beyond its 10 V range"):
        dac.Write(10001.0)
    assert calls == []


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


def test_Mcc_Writes_Through_The_Board_And_Starts_At_Zero(monkeypatch):
    calls = []
    ul = types.SimpleNamespace(v_out=lambda *args: calls.append(args))
    enums = types.SimpleNamespace(ULRange=types.SimpleNamespace(BIP10VOLTS='BIP10'))
    monkeypatch.setitem(sys.modules, 'mcculw', types.SimpleNamespace(ul=ul, enums=enums))
    monkeypatch.setitem(sys.modules, 'mcculw.enums', enums)
    dac = mcc_dac.Open_Mcc(board=1)
    assert calls == [(1, CH_NEG, 'BIP10', 0.0), (1, CH_POS, 'BIP10', 0.0)]
    dac.Write(500.0)
    assert calls[-1] == (1, CH_POS, 'BIP10', 0.5)


def test_Pyvisa_Io_Errors_Become_Instrument_Io_Errors(monkeypatch):

    class Visa_Io_Error(Exception):
        pass

    class Resource():
        timeout = 0

        def query(self, command):
            raise Visa_Io_Error("VI_ERROR_TMO")

    manager = types.SimpleNamespace(open_resource=lambda name: Resource())
    monkeypatch.setitem(sys.modules, 'pyvisa', types.SimpleNamespace(
        ResourceManager=lambda: manager, errors=types.SimpleNamespace(VisaIOError=Visa_Io_Error)))
    monkeypatch.setattr(visa_session, '_Manager', None)
    inst = visa_session.Pyvisa_Connect('USB0::STUB::INSTR')
    with pytest.raises(visa_session.Instrument_IO_Error, match="USB0::STUB::INSTR did not answer"):
        inst.query('READ?')
