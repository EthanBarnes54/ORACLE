"""Settings and the facility services' offline behaviour."""

import logging
import multiprocessing
import os
import sys
import time
import types

import pytest
import yaml

from cls_acquisition import facility, settings
from cls_acquisition.hardware import Rack


def test_Update_Rewrites_Only_Its_Keys(monkeypatch, tmp_path):
    path = tmp_path / 'settings.yaml'
    monkeypatch.setenv('ORACLE_SETTINGS_PATH', str(path))
    path.write_text(yaml.safe_dump({'rack': 'hardware', 'elog': {'author': 'someone'}}))
    settings.Update(experiment='I311')
    stored = yaml.safe_load(path.read_text())
    assert stored == {'rack': 'hardware', 'elog': {'author': 'someone'}, 'experiment': 'I311'}
    loaded = settings.Load()
    assert loaded['elog'] == dict(settings.DEFAULTS['elog'], author='someone')
    assert loaded['instruments'] == settings.DEFAULTS['instruments']


def test_A_Simulated_Rack_Never_Writes_Among_The_Real_Runs():
    simulated = dict(settings.DEFAULTS, data_directory='/archive')
    hardware = dict(simulated, rack='hardware')
    assert settings.Data_Directory(hardware) == '/archive'
    assert settings.Data_Directory(simulated) == os.path.join(settings.APP_ROOT, 'data',
                                                              'simulated')


def test_A_Disabled_Elog_Logs_What_It_Would_Have_Sent(caplog, tmp_path):
    attachment = tmp_path / 'screen.png'
    attachment.write_bytes(b'png')
    elog = facility.Elog('http://nowhere', '', enabled=False)
    with caplog.at_level(logging.INFO, logger='oracle.facility'):
        elog.Append('START SCAN NUMBER 1', 'test', str(attachment))
        elog.Close()
    assert 'START SCAN NUMBER 1' in caplog.text
    assert not attachment.exists() and elog.Last_Error is None


def test_Simulated_Pvs_Answer_Without_Epics():
    pvs = facility.Pv_Monitors({'wavenumber': 'NOT:A:PV'}, simulated=True)
    assert pvs.Value('wavenumber') == facility.SIMULATED_PVS['wavenumber']
    assert pvs.Value('unknown') is None
    pvs.Close()


def test_The_Coolers_Pvs_Come_From_The_Poll_And_Mark_A_Lost_Reading(monkeypatch):

    class Driver():
        Params, Status, Updates = {}, {}, 0

        def setParam(self, reason, value):
            Driver.Params[reason] = value

        def setParamStatus(self, reason, alarm, severity):
            Driver.Status[reason] = severity

        def updatePVs(self):
            Driver.Updates += 1

    class Server():
        Made = []

        def __init__(self):
            Server.Made.append(self)

        def createPV(self, prefix, pvdb):
            self.Names = [prefix + reason for reason in pvdb]

        def process(self, timeout):
            time.sleep(timeout)

    monkeypatch.setitem(sys.modules, 'pcaspy', types.SimpleNamespace(
        SimpleServer=Server, Driver=Driver,
        Alarm=types.SimpleNamespace(NO_ALARM=0, READ_ALARM=1),
        Severity=types.SimpleNamespace(NO_ALARM=0, INVALID_ALARM=3)))
    monkeypatch.setattr(facility, 'COOLER_PUBLISH_S', 0.0)
    readings = iter([29884.0, 29886.0] + [float('nan')] * 1000)
    server = facility.Cooler_Pv_Server('FURIOS:Cooler:', lambda: next(readings))
    deadline = time.monotonic() + 5.0
    while Driver.Status.get('Readback') != 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    server.Close()
    assert Server.Made[0].Names == ['FURIOS:Cooler:Readback', 'FURIOS:Cooler:InstantValue']
    assert Driver.Params == {'Readback': 29885.0, 'InstantValue': 29886.0}
    assert Driver.Status == {'Readback': 3, 'InstantValue': 3}     # invalid, not a stale value


def test_The_Commit_Ran_From_Is_Named_And_Nothing_Is_Asked_Of_A_Pc_Without_Git(monkeypatch):
    if not os.path.isdir(os.path.join(settings.CODE_ROOT, '.git')):
        pytest.skip("ORACLE is not running from a git checkout here")
    commit = settings.Commit()
    assert len(commit['Hash']) == 40 and isinstance(commit['Dirty'], bool)

    def no_git(*_, **__):
        raise FileNotFoundError("git")

    monkeypatch.setattr(settings.subprocess, 'run', no_git)
    assert settings.Commit() is None


def test_On_Windows_Git_Runs_Without_A_Console_Window(monkeypatch):
    """The Windows branch, run here on a stand-in: a window's git call would
    otherwise flash a console up on the rack PC."""
    flags = []

    def run(args, **kwargs):
        flags.append(kwargs.get('creationflags'))
        return types.SimpleNamespace(stdout='abc\n')

    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(settings.subprocess, 'CREATE_NO_WINDOW', 0x08000000, raising=False)
    monkeypatch.setattr(settings.subprocess, 'run', run)
    assert settings.Commit() == {'Hash': 'abc', 'Dirty': True}
    assert flags == [0x08000000, 0x08000000]


def test_The_Dividers_Default_To_The_Bench_And_A_Ratio_That_Is_No_Number_Is_Refused():
    from cls_acquisition.calibration import DIVIDER
    from cls_acquisition.hardware.keysight import COOLER_DIVIDER
    rack = Rack(settings.DEFAULTS, multiprocessing.get_context('spawn'))
    assert (rack.Scan_Divider, rack.Cooler_Divider) == (DIVIDER, COOLER_DIVIDER)
    assert rack.Dividers()['Scan'] == settings.DEFAULTS['hardware']['scan_divider']
    for ratio in (0.0, -1000.0, '1000', None):
        config = settings._Merged(settings.DEFAULTS,
                                  {'hardware': {'cooler_divider': {'ratio': ratio}}})
        with pytest.raises(ValueError, match="cooler divider a ratio"):
            Rack(config, multiprocessing.get_context('spawn'))


def test_A_Rolling_Window_That_Is_Not_A_Positive_Time_Is_Refused():
    assert settings.DEFAULTS['cooler']['rolling_s'] == 15.0
    for window in (0.0, -15.0, '15 s'):
        config = settings._Merged(settings.DEFAULTS, {'cooler': {'rolling_s': window}})
        with pytest.raises(ValueError, match="cooler: rolling_s"):
            Rack(config, multiprocessing.get_context('spawn'))


def test_Calibration_Readings_Of_A_Length_The_Meter_Does_Not_Take_Are_Refused():
    assert settings.DEFAULTS['calibration']['nplc'] == 1
    for nplc in (0.5, 0, '1'):
        config = settings._Merged(settings.DEFAULTS, {'calibration': {'nplc': nplc}})
        with pytest.raises(ValueError, match="calibration: nplc"):
            Rack(config, multiprocessing.get_context('spawn'))


def test_Pvs_With_No_Name_Are_Not_Asked_Of_Epics(monkeypatch):
    made = []
    epics = types.ModuleType('epics')
    epics.PV = lambda name, **_: made.append(name) or types.SimpleNamespace(
        disconnect=lambda: None)
    monkeypatch.setitem(sys.modules, 'epics', epics)
    pvs = facility.Pv_Monitors(settings.DEFAULTS['pvs'], simulated=False)
    assert made == ['FURIOS:Wavemeter2:Wavenumber1', 'IGISOL:MagnetControl:Mass',
                    'IGISOL:MagnetControl:Target']
    assert pvs.Value('pmt_hv_1') is None


def test_A_Limit_Above_The_Amplifiers_Rating_Is_Refused():
    config = settings._Merged(settings.DEFAULTS, {'instruments': {'hv_limit_v': 6000.0}})
    with pytest.raises(ValueError, match="rating of 4000 V"):
        Rack(config, multiprocessing.get_context('spawn'))
    assert settings.DEFAULTS['instruments']['hv_limit_v'] == 4000.0
