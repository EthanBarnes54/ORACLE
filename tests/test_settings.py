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


def test_A_Limit_Above_The_Amplifiers_Rating_Is_Refused():
    config = settings._Merged(settings.DEFAULTS, {'instruments': {'hv_limit_v': 6000.0}})
    with pytest.raises(ValueError, match="rating of 4000 V"):
        Rack(config, multiprocessing.get_context('spawn'))
    assert settings.DEFAULTS['instruments']['hv_limit_v'] == 4000.0
