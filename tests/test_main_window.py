"""The window end to end on the simulated rack, through a real acquisition process:
a scan is calibrated, gated, written and closed; a refused calibration writes nothing."""

import glob
import multiprocessing
import os
import time

import asdf
import pytest

from cls_acquisition import settings
from cls_acquisition.acquisition import Rack_Link
from cls_acquisition.facility import Elog, Pv_Monitors
from cls_acquisition.hardware import Rack
from gui import main_window


def _Pump(app, until, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, 'APP_ROOT', str(tmp_path))
    # its own settings: the correction this module puts in force stays here
    monkeypatch.setenv('ORACLE_SETTINGS_PATH', str(tmp_path / 'settings.yaml'))
    dialogs = []
    for kind in ('critical', 'warning'):
        monkeypatch.setattr(main_window.QMessageBox, kind,
                            lambda parent, title, text, kind=kind: dialogs.append((title, text)))
    config = settings.Load()
    rack = Rack(config, multiprocessing.get_context('spawn'))
    link = Rack_Link(rack.Loop_Spec())
    link.Send('Ping', timeout=30)
    line, cooler = rack.Open_Meters()
    w = main_window.Main_Window(config, rack, link, line, cooler, Pv_Monitors({}, True),
                                Elog('', '', False))
    w.Dialogs = dialogs
    scan = w.Scan
    scan.From_Boxes[0].setValue(20.0)
    scan.To_Boxes[0].setValue(40.0)
    scan.Step_Box.setValue(2.0)
    scan.Dwell_Box.setValue(10)
    scan.Per_Step_Box.setValue(2)
    scan.Type_Combo.setCurrentText('Single scan')
    yield w
    w.close()
    assert not link.Alive()


def _Run_Files(w):
    return glob.glob(os.path.join(w._Data_Dir, '*', '*', '*', 'run_*.asdf'))


def test_A_Scan_Is_Calibrated_Gated_Written_And_Closed(qapp, window):
    state = window._Link.State
    window.Scan.Per_Step_Box.setValue(30)       # hits enough that the run is never empty
    window.Scan.Start_Button.click()
    assert not window.Scan.Control_Group.isEnabled()          # disabled at once
    assert _Pump(qapp, lambda: state.Scanning.is_set()), window.Dialogs
    assert _Pump(qapp, lambda: not state.Scanning.is_set() and window._Run is None)
    [path] = _Run_Files(window)
    with asdf.open(path) as f:
        assert f['Run'] == 1 and len(f['CalSet']) == 11
        assert f['raw'].shape[1] == 6
    assert 'PASS' in window.Calibration.Report_Label.text()
    assert window.Scan.Control_Group.isEnabled() and window.Dialogs == []
    # the run's monitor record, and its performance analysed as it closed
    assert os.path.exists(path[:-len('.asdf')] + '_monitor.csv')
    assert _Pump(qapp, lambda: window.Performance.Result is not None)
    analysed, result = window.Performance.Result
    assert analysed == path and result['Verdict'] in ('PASS', 'WARN')
    assert window.Performance.Run_Combo.currentData() == path


def test_A_Dead_Readback_Is_Refused_And_Nothing_Is_Written(qapp, window, monkeypatch, caplog):

    def dead_meter(wait_s):
        # Mode A: the meter reads the platform at rest whatever is commanded
        time.sleep(wait_s)
        return time.perf_counter_ns(), 2e-5

    monkeypatch.setattr(window._Line, 'Read_Settled', dead_meter)
    window.Scan.Start_Button.click()
    assert _Pump(qapp, lambda: window.Dialogs)
    title, text = window.Dialogs[0]
    assert title == 'Run rejected' and 'tracking' in text
    assert _Run_Files(window) == [] and not window._Link.State.Scanning.is_set()
    # no run file keeps it, so the session log does, ELOG or not
    [refused] = [r.getMessage() for r in caplog.records if 'refused by the gate' in r.getMessage()]
    assert refused.startswith("Run 1's calibration") and 'tracking:' in refused


def test_A_Calibration_Lagged_By_Twelve_Starts_No_Scan(qapp, window, monkeypatch):
    # Mode B with k = 12, here a meter whose answers trail its requests by twelve: the ramp
    # still tracks and spans, and the gate refuses it on the setpoint read held
    read, answers = window._Line.Read_Settled, []

    def lagging(wait_s):
        stamp, volts = read(wait_s)
        answers.append(volts)
        return stamp, answers[max(len(answers) - 13, 0)]

    monkeypatch.setattr(window._Line, 'Read_Settled', lagging)
    window.Scan.From_Boxes[0].setValue(-150.0)
    window.Scan.To_Boxes[0].setValue(150.0)
    window.Scan.Start_Button.click()
    assert _Pump(qapp, lambda: window.Dialogs)
    title, text = window.Dialogs[0]
    # its held readings are as stale as the rest, and read the held setpoint as a ramp
    assert title == 'Run rejected' and text.startswith('held:') and 'a lag' in text
    assert 'tracking' not in text and 'span' not in text
    assert _Run_Files(window) == [] and not window._Link.State.Scanning.is_set()


def test_The_Hardware_Calibrations_Run_Save_And_Correct_The_Dac(qapp, window):
    hardware = window.Hardware
    hardware.Ramp_Rate_Box.setValue(100000.0)
    hardware.Ramp_Step_Box.setValue(500.0)
    settle = hardware.Settle
    settle.To_Boxes[0].setValue(500.0)
    settle.Window_Box.setValue(200.0)
    settle.Hold_Box.setValue(0.05)
    settle.Start_Button.click()
    assert not window.Scan.Control_Group.isEnabled()          # the DAC is spoken for
    assert _Pump(qapp, lambda: window._Worker is None)
    assert settle.Table.rowCount() == 6 and 'PASS' in settle.Report_Label.text()

    linearity = hardware.Linearity
    for box, value in ((linearity.From_Box, -400.0), (linearity.To_Box, 400.0),
                       (linearity.Step_Box, 100.0), (linearity.Settle_Box, 0.02),
                       (linearity.Readings_Box, 2)):
        box.setValue(value)
    linearity.Nplc_Combo.setCurrentText('1')
    linearity.Start_Button.click()
    assert _Pump(qapp, lambda: window._Worker is None)
    assert linearity.Use_Button.isEnabled()
    linearity.Use_Button.click()
    assert settings.Load()['dac_correction']['measured'] is not None
    report = linearity.Report_Label.text()
    assert '>gain and offset<' in report and 'in force now' in report
    # the sweep's own results stay above the correction, not replaced by it
    assert 'Straight-line fit' in report and 'do not also apply the bench table' in report

    cooler = hardware.Cooler
    cooler.Duration_Box.setValue(1.0)
    cooler.Interval_Box.setValue(2.0)
    cooler.Start_Button.click()
    assert _Pump(qapp, lambda: window._Worker is None)
    assert cooler.Peak_Table.rowCount() > 0
    records = glob.glob(os.path.join(window._Data_Dir, 'hardware_calibration', '*.csv'))
    assert {os.path.basename(r).split('_2')[0] for r in records} == {
        'settle', 'settle_traces', 'linearity', 'cooler', 'cooler_spectrum'}
    for record in records:
        with open(record) as fh:
            text = fh.read()
        assert '# status: completed' in text
        if 'cooler_spectrum' in record:
            # the averaged spectrum, written once the monitor ends, as the bench's spectrum.csv
            rows = [line for line in text.splitlines() if line and not line.startswith('#')]
            assert rows[0] == 'frequency_hz,asd_v_rt_hz' and len(rows) > 10
    assert window.Dialogs == []
