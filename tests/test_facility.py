"""The facility's services against stand-ins for py-elog and pyepics.

The ELOG is posted to as the old DAQ posted to it: an entry started with
its author, 'Measurement' and the experiment, every addition read back
and posted again onto the same entry, a screenshot attached and then
deleted. A post that fails is reported and never stops the next. PVs
are monitored, so a read never waits on the network.
"""

import sys
import time
import types

import pytest

from cls_acquisition import facility


class Logbook():
    """py-elog's logbook: entries by id, each a message, attributes and attachments."""

    def __init__(self, fail=0):
        self.Entries, self.Calls, self.Fail = {}, [], fail

    def post(self, message, msg_id=None, attachments=None, **attributes):
        self.Calls.append(('post', message, msg_id, attachments, attributes))
        if self.Fail:
            self.Fail -= 1
            raise ConnectionError("elog.igisol:8020 refused the connection")
        if attachments:
            for path in attachments:
                with open(path, 'rb') as fh:
                    assert fh.read() == b'PNG'      # the file is still there to send
        msg_id = msg_id or len(self.Entries) + 1
        self.Entries[msg_id] = (message, attributes or self.Entries.get(msg_id, ('', {}))[1],
                                list(attachments or []))
        return msg_id

    def read(self, msg_id):
        self.Calls.append(('read', msg_id))
        return self.Entries[msg_id]


@pytest.fixture
def logbook(monkeypatch):
    book = Logbook()
    opened = []
    module = types.SimpleNamespace(open=lambda url: opened.append(url) or book)
    monkeypatch.setitem(sys.modules, 'elog', module)
    book.Opened = opened
    return book


def _Settled(elog):
    elog.Close()


def test_An_Entry_Is_Started_And_Added_To_As_The_Old_Daq_Did(logbook):
    elog = facility.Elog('http://elog.igisol:8020/Collinear/', 'ebarnes', True)
    elog.New_Entry('IS700')
    elog.Append('START SCAN NUMBER 12', 'IS700')
    elog.Append('\n\nSCAN KILLED', 'IS700')
    _Settled(elog)
    assert logbook.Opened == ['http://elog.igisol:8020/Collinear/']
    first = logbook.Calls[0]
    assert first[0] == 'post' and first[2] is None
    assert first[4] == {'Author': 'ebarnes', 'What': 'Measurement', 'Experiment': 'IS700'}
    [entry] = logbook.Entries.values()
    assert entry[0].endswith('\nSTART SCAN NUMBER 12\n\n\nSCAN KILLED')
    # every addition read the entry back and posted it again onto the same id
    assert [c[0] for c in logbook.Calls[1:]] == ['read', 'post', 'read', 'post']
    assert all(c[2] == 1 for c in logbook.Calls[1:] if c[0] == 'post')


def test_An_Addition_With_No_Entry_Open_Starts_One(logbook):
    elog = facility.Elog('url', 'ebarnes', True)
    elog.Append('beam tuned', 'IS700')
    _Settled(elog)
    assert len(logbook.Entries) == 1
    assert logbook.Entries[1][0].endswith('\nbeam tuned')


def test_An_Attachment_Is_Sent_Then_Deleted(logbook, tmp_path):
    screen = tmp_path / 'Scan12_Round3.png'
    screen.write_bytes(b'PNG')
    elog = facility.Elog('url', 'ebarnes', True)
    elog.Append('(see attached spectra Scan12_Round3)', 'IS700', str(screen))
    _Settled(elog)
    assert logbook.Entries[1][2] == [str(screen)] and not screen.exists()


def test_A_Failed_Post_Is_Reported_And_The_Next_Still_Goes(monkeypatch, tmp_path):
    book = Logbook(fail=1)
    monkeypatch.setitem(sys.modules, 'elog', types.SimpleNamespace(open=lambda url: book))
    screen = tmp_path / 'screen.png'
    screen.write_bytes(b'PNG')
    elog = facility.Elog('url', 'ebarnes', True)
    elog.Append('lost', 'IS700', str(screen))
    elog.Append('kept', 'IS700')
    _Settled(elog)
    assert 'refused the connection' in elog.Last_Error
    assert not screen.exists()                  # even a failed post lets its file go
    assert book.Entries[1][0].endswith('\nkept')


def test_The_Author_Defaults_To_The_Login_Name(logbook, monkeypatch):
    monkeypatch.setattr(facility.getpass, 'getuser', lambda: 'operator')
    elog = facility.Elog('url', '', True)
    elog.New_Entry('IS700')
    _Settled(elog)
    assert logbook.Calls[0][4]['Author'] == 'operator'


def test_A_Disabled_Elog_Never_Opens_The_Logbook(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, 'elog', types.SimpleNamespace(
        open=lambda url: pytest.fail("a disabled ELOG opened the logbook")))
    elog = facility.Elog('url', 'ebarnes', False)
    with caplog.at_level('INFO', logger='oracle.facility'):
        elog.Append('not sent', 'IS700')
        _Settled(elog)
    assert 'not sent' in caplog.text


def test_Pvs_Are_Monitored_And_Read_Without_Waiting(monkeypatch):
    made = []

    class Pv():
        def __init__(self, name, auto_monitor, callback):
            assert auto_monitor
            self.Name, self.Callback, self.Connected = name, callback, True
            made.append(self)

        def disconnect(self):
            self.Connected = False

    monkeypatch.setitem(sys.modules, 'epics', types.SimpleNamespace(PV=Pv))
    names = {'wavenumber': 'FURIOS:Wavemeter2:Wavenumber1', 'mass': 'IGISOL:MagnetControl:Mass'}
    pvs = facility.Pv_Monitors(names, simulated=False)
    assert sorted(pv.Name for pv in made) == sorted(names.values())
    assert pvs.Value('wavenumber') is None         # nothing pushed yet: no wait, no guess
    started = time.monotonic()
    made[0].Callback(value=16956.17, pvname=made[0].Name)
    assert pvs.Value('wavenumber') == 16956.17 and time.monotonic() - started < 0.1
    pvs.Close()
    assert not any(pv.Connected for pv in made)
