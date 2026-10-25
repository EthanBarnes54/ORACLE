"""Settings and the facility services' offline behaviour."""

import logging
import os

import yaml

from cls_acquisition import facility, settings


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
