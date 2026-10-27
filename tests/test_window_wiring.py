"""Every control of the window, wired to what it drives, on the simulated rack.

The collinear DAQ's controls (scan, calibration, ELOG, quick tune,
triggers, DC mode), the bench tool's (arming, stopping, the recommended
wait, the precise reading, the settle model and the DAC correction), and
what the window owes the operator when things go wrong: a timing that
cannot run, a run file that cannot be written, an acquisition process
that dies, a close at the wrong moment. Each test drives the real window
and a real acquisition process, and reads back what the rack, the run
file, the records, the settings and the ELOG were given.
"""

import os
import re

import asdf
import numpy as np
import pytest
import yaml
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QLabel, QMessageBox

from cls_acquisition import hardware_calibration as hc
from cls_acquisition import run_file, settings
from cls_acquisition.hardware import simulated
from cls_acquisition.hardware.visa_session import Scpi_Error
from gui import main_window
from tests.window import Close_Window, Open_Window, Pump, Run_Files

YES, NO = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No


@pytest.fixture
def window(qapp, monkeypatch, tmp_path):
    w = Open_Window(monkeypatch, tmp_path)
    yield w
    Close_Window(qapp, w)


def _Scan(qapp, window, scan_type='Single scan'):
    """Start a scan and wait until it runs."""
    window.Scan.Type_Combo.setCurrentText(scan_type)
    window.Scan.Start_Button.click()
    assert Pump(qapp, lambda: window._Link.State.Scanning.is_set()), window.Dialogs


def _Finished(qapp, window):
    state = window._Link.State
    assert Pump(qapp, lambda: not state.Scanning.is_set() and window._Run is None)


def _Answer(monkeypatch, answer):
    """Answer every question the window asks with ``answer``; returns them."""
    asked = []
    monkeypatch.setattr(main_window.QMessageBox, 'question',
                        lambda parent, title, text, *_: asked.append((title, text)) or answer)
    return asked


def _Fast_Bench(window):
    hardware = window.Hardware
    hardware.Ramp_Rate_Box.setValue(100000.0)
    hardware.Ramp_Step_Box.setValue(500.0)
    settle = hardware.Settle
    settle.To_Boxes[0].setValue(500.0)
    settle.Reps_Box.setValue(1)
    settle.Window_Box.setValue(200.0)
    settle.Hold_Box.setValue(0.05)
    linearity = hardware.Linearity
    for box, value in ((linearity.From_Box, -400.0), (linearity.To_Box, 400.0),
                       (linearity.Step_Box, 50.0), (linearity.Fine_Span_Box, 0.0),
                       (linearity.Zero_Box, 0), (linearity.Settle_Box, 0.02),
                       (linearity.Readings_Box, 2)):
        box.setValue(value)
    linearity.Nplc_Combo.setCurrentText('1')


def _Record(window, kind):
    """The text of the newest ``kind`` record."""
    import glob
    paths = sorted(p for p in glob.glob(os.path.join(window._Data_Dir, 'hardware_calibration',
                                                     f'{kind}_2*.csv')))
    with open(paths[-1], encoding='utf-8') as fh:
        return fh.read()


# ── The collinear DAQ's controls ────────────────────────────────

def test_A_Calibration_On_Its_Own_Fills_Its_Plots_And_Writes_No_Run(qapp, window):
    calibration = window.Calibration
    calibration.From_Box.setValue(-100.0)
    calibration.To_Box.setValue(100.0)
    calibration.Step_Box.setValue(20.0)
    calibration.Start_Button.click()
    assert not calibration.Control_Group.isEnabled()
    assert not window.Quick_Group.isEnabled()            # nothing else may drive the DAC
    assert Pump(qapp, lambda: window._Worker is None)
    report = calibration.Report_Label.text()
    assert 'REJECT' not in report and ('PASS' in report or 'WARN' in report)
    for curve, count in zip(calibration._Curves, (11, 11, 10)):
        assert len(curve.getData()[0]) == count
    assert Run_Files(window) == [] and window.Dialogs == []
    assert calibration.Control_Group.isEnabled() and window.Quick_Group.isEnabled()


def test_A_Scan_Tells_The_Elog_And_The_Header_What_The_Old_Daq_Did(qapp, window):
    window.Scan.Experiment_Edit.setText('Ge-So-Offline')
    _Scan(qapp, window)
    _Finished(qapp, window)
    [start] = [text for text in window._Elog.Texts() if 'START SCAN NUMBER' in text]
    for piece in ('START SCAN NUMBER 1', 'from 20 to 40 V', 'Bunches per channel: 2',
                  'Stepsize [V]:        2', 'Dwell time [ms]:     10',
                  'Laser Set Point [cm-1]: 16956.17000', 'Mass [amu]:          59.0000',
                  'Calibration:  |r| =', 'DAC correction: none'):
        assert piece in start
    assert {experiment for _, _, experiment, _ in window._Elog.Posts} == {'Ge-So-Offline'}
    [path] = Run_Files(window)
    with asdf.open(path) as f:
        assert f['Experiment'] == 'Ge-So-Offline' and f['Run'] == 1
        assert (f['BunchesPerChannel'], f['StepSize'], f['DwellTime']) == (2, 2.0, 10)
        assert f['MassAMU'] == 59.0 and f['LaserSetpoint'] == pytest.approx(16956.17)
        assert f['ScanningRanges'] == [[20.0, 40.0]] and f['DacCorrectionApplied'] == 0
        # the cooler as its meter reads it, as the old DAQ kept it
        assert f['CoolerVoltage'] * 1e4 == pytest.approx(simulated.COOLER_V, abs=2.0)
        raw = np.asarray(f['raw'])
    assert set(np.unique(raw[:, 1])) <= set(np.arange(20.0, 41.0, 2.0))


def test_The_Elog_Panel_Sends_Text_Starts_Entries_And_Attaches_The_Spectrum(qapp, window):
    scan, elog = window.Scan, window._Elog
    scan.Elog_Edit.setPlainText('beam tuned to 59Co')
    scan.Send_Text_Button.click()
    assert elog.Texts()[-1] == 'beam tuned to 59Co' and scan.Elog_Edit.toPlainText() == ''
    scan.New_Entry_Button.click()
    assert elog.Posts[-1][0] == 'new'
    scan.Screen_Button.click()
    _, text, _, attachment = elog.Posts[-1]
    assert text.startswith('(see attached spectra Scan') and attachment.endswith('.png')
    assert elog.Attachments[-1].startswith(b'\x89PNG')


def test_Stop_Now_Ends_The_Run_At_Once_And_Says_So(qapp, window):
    _Scan(qapp, window, 'Sawtooth')
    assert Pump(qapp, lambda: window._Live.Count > 0)
    window.Scan.Stop_Button.click()
    _Finished(qapp, window)
    assert window._Elog.Texts()[-1] == "\n\nSCAN KILLED"
    [path] = Run_Files(window)
    assert len(run_file.Read(path)[1]) > 0
    assert window.Scan.Start_Button.text() == 'Start scan'


def test_Stop_After_Scan_Finishes_The_Round_It_Is_In(qapp, window):
    _Scan(qapp, window, 'Sawtooth')
    window.Scan.Start_Button.click()                       # now 'Stop after scan'
    _Finished(qapp, window)
    assert window._Elog.Texts()[-1].startswith("\n\nSCAN STOPPED AFTER ")
    _, _, monitor = run_file.Read(Run_Files(window)[0])
    # every bunch of every round: 11 setpoints of 2 bunches, nothing cut short
    assert monitor['bunches'][-1] > 0 and monitor['bunches'][-1] % 22 == 0


def test_Triggers_Stream_To_The_Plots_But_Never_To_A_File(qapp, window):
    state = window._Link.State
    window.Trigger_Button.click()
    assert Pump(qapp, lambda: state.Triggering.is_set())
    assert Pump(qapp, lambda: window.Trigger_Button.text() == 'Disable triggers')
    assert not window.Scan.Control_Group.isEnabled() and not window.Scan.Params_Group.isEnabled()
    assert Pump(qapp, lambda: window._Live.Count > 0, timeout=20)
    window.Trigger_Button.click()
    assert Pump(qapp, lambda: not state.Triggering.is_set())
    assert Pump(qapp, lambda: window.Scan.Control_Group.isEnabled())
    assert Run_Files(window) == []


def test_Quick_Tune_Puts_Its_Voltage_On_The_Platform_And_Takes_It_Off(qapp, window):
    state = window._Link.State
    assert not window.Voltage_Box.isEnabled() and window.Voltage_Box.maximum() == 4000.0
    window.Volt_Button.click()
    assert Pump(qapp, lambda: state.Outputting.is_set() and window.Voltage_Box.isEnabled())
    assert window.Volt_Button.text() == 'Disable voltage'
    window.Voltage_Box.setValue(150.0)
    assert Pump(qapp, lambda: state.Voltage.value == 150.0)
    expected = 150.0 + simulated.Platform_Error(150.0)
    assert Pump(qapp, lambda: abs(window._Line.Voltage * 1e3 - expected) < 0.5)
    # the slider spans the sweep range, the box follows it
    window.Sweep_From.setValue(-100.0)
    window.Sweep_To.setValue(100.0)
    window.Slider.setValue(7500)
    assert window.Voltage_Box.value() == pytest.approx(50.0)
    assert Pump(qapp, lambda: state.Voltage.value == pytest.approx(50.0))
    window.Volt_Button.click()
    assert Pump(qapp, lambda: not state.Outputting.is_set() and state.Voltage.value == 0.0)
    assert Pump(qapp, lambda: not window.Voltage_Box.isEnabled()
                and window.Voltage_Box.value() == 0.0)


def test_A_Dc_Scan_Counts_Seconds_At_The_Dc_Trigger_With_Its_Gate_Settable(qapp, window):
    scan = window.Scan
    scan.Dc_Radio.setChecked(True)
    assert scan.Per_Step_Label.text() == 'Seconds per channel'
    assert scan.Per_Step_Box.value() == pytest.approx(0.1) and scan.Bunches_Per_Step() == 100
    assert not scan.Dwell_Box.isEnabled() and not scan.Release_Box.isEnabled()
    # DC hits come through the tagger, so its window is in force and settable
    assert scan.Gate_Start_Box.isEnabled() and scan.Gate_Length_Box.isEnabled()
    assert scan.Trigger_Hz() == 1000.0
    scan.Per_Step_Box.setValue(0.02)
    _Scan(qapp, window)
    assert Pump(qapp, lambda: window.Mode_Label.text() == 'DC')
    _Finished(qapp, window)
    with asdf.open(Run_Files(window)[0]) as f:
        assert f['BunchesPerChannel'] == 20


@pytest.mark.parametrize('set_up, refusal', [
    (lambda s: s.Release_Box.setValue(20000), "bunch release at 20000 µs"),
    (lambda s: s.Tagger_Delay_Box.setValue(10000), "tagger's trigger at 10000 µs"),
    (lambda s: (s.Dwell_Box.setValue(5), s.Tagger_Delay_Box.setValue(200),
                s.Gate_Length_Box.setValue(4900)),
     "window closes 5100 µs into the 5000 µs cycle"),
    (lambda s: (s.Dc_Radio.setChecked(True), s.Gate_Length_Box.setValue(1500)),
     "window closes 1500 µs into the 1000 µs cycle"),
])
def test_A_Timing_That_Does_Not_Fit_Its_Cycle_Is_Refused_Before_Anything_Runs(
        qapp, window, set_up, refusal):
    set_up(window.Scan)
    window.Scan.Start_Button.click()
    window.Trigger_Button.click()
    assert [title for title, _ in window.Dialogs] == ['Scan not started', 'Triggers not started']
    assert all(refusal in text for _, text in window.Dialogs)
    state = window._Link.State
    assert not state.Scanning.is_set() and not state.Triggering.is_set()
    assert window._Worker is None and Run_Files(window) == []


def test_Runs_Are_Numbered_On_From_The_Data_Folder(qapp, window):
    for number in (1, 2):
        _Scan(qapp, window)
        _Finished(qapp, window)
        assert window.Run_Label.text() == str(number)
    assert [os.path.basename(p) for p in Run_Files(window)] == ['run_0001.asdf',
                                                                'run_0002.asdf']
    assert window._Next_Run() == 3


def test_A_Run_File_That_Cannot_Be_Written_Stops_The_Scan(qapp, window, monkeypatch):

    def full(path, rows):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(run_file, 'Append', full)
    _Scan(qapp, window, 'Sawtooth')
    assert Pump(qapp, lambda: any(t == 'Scan stopped' for t, _ in window.Dialogs))
    assert Pump(qapp, lambda: not window._Link.State.Scanning.is_set())
    assert window._Run is None
    assert 'No space left on device' in dict(window.Dialogs)['Scan stopped']


def test_A_Dead_Acquisition_Process_Closes_The_Run_And_Frees_The_Window(qapp, window):
    _Scan(qapp, window, 'Sawtooth')
    assert Pump(qapp, lambda: window._Live.Count > 0)
    window._Link._Process.terminate()
    assert Pump(qapp, lambda: any(t == 'Acquisition stopped' for t, _ in window.Dialogs))
    state = window._Link.State
    assert window._Run is None and not state.Scanning.is_set()
    assert window.Scan.Start_Button.text() == 'Start scan'
    assert not window._Awake.Held
    [path] = Run_Files(window)
    header, raw, monitor = run_file.Read(path)          # closed on what it recorded
    assert header['Run'] == 1 and len(raw) > 0 and monitor is not None


def test_The_Window_Will_Not_Close_Under_A_Calibration_And_Asks_During_A_Scan(
        qapp, window, monkeypatch):
    calibration = window.Calibration
    calibration.From_Box.setValue(-300.0)
    calibration.To_Box.setValue(300.0)
    calibration.Step_Box.setValue(20.0)
    calibration.Start_Button.click()
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted() and window._Link.Alive()
    assert Pump(qapp, lambda: window._Worker is None)

    asked = _Answer(monkeypatch, NO)
    _Scan(qapp, window, 'Sawtooth')
    event = QCloseEvent()
    window.closeEvent(event)
    assert asked and not event.isAccepted() and window._Link.State.Scanning.is_set()
    window.Scan.Stop_Button.click()
    _Finished(qapp, window)


def test_The_Experiment_Name_Is_Kept_In_The_Settings(qapp, window):
    window.Scan.Experiment_Edit.setText('IS700')
    window.Scan.Experiment_Edit.editingFinished.emit()
    assert settings.Load()['experiment'] == 'IS700'


def test_The_Folders_Open_And_A_Missing_Simulated_One_Is_Made(qapp, window, monkeypatch):
    opened = []
    monkeypatch.setattr(main_window.QDesktopServices, 'openUrl',
                        lambda url: opened.append(url.toLocalFile()) or True)
    menu = window.menuBar().actions()[0].menu()
    for action in menu.actions():
        if action.text() in ('Open data folder', 'Open log folder'):
            action.trigger()
    assert [os.path.normpath(p) for p in opened] == [
        os.path.normpath(window._Data_Dir),
        os.path.normpath(os.path.join(settings.APP_ROOT, 'logs'))]
    assert all(os.path.isdir(p) for p in opened) and window.Dialogs == []


def test_Autoscale_All_Puts_Every_Plot_Back_To_Following_Its_Data(qapp, window):
    from gui import plots
    every = window.findChildren(plots.Plot_Widget)
    assert len(every) >= 15
    for plot in every:
        plot.setXRange(0, 1)
    assert not any(all(p.getViewBox().autoRangeEnabled()) for p in every)
    window._Autoscale_All()
    assert all(all(p.getViewBox().autoRangeEnabled()) for p in every)


def test_A_Rack_Warning_Stays_In_The_Status_Bar(qapp, monkeypatch, tmp_path):
    warning = "The time tagger's wrapper times input 3 and 4 on input 2's falling edge."
    w = Open_Window(monkeypatch, tmp_path, warnings=(warning,))
    try:
        [label] = [l for l in w.statusBar().findChildren(QLabel)
                   if l.text().strip() == 'Rack warning']
        assert label.toolTip() == warning
    finally:
        Close_Window(qapp, w)


def test_The_Pc_Is_Kept_Awake_While_Anything_Runs(qapp, window):
    assert Pump(qapp, lambda: not window._Awake.Held)
    _Scan(qapp, window)
    assert Pump(qapp, lambda: window._Awake.Held)
    _Finished(qapp, window)
    assert Pump(qapp, lambda: not window._Awake.Held)


# ── The bench tool's controls ───────────────────────────────────

def test_On_The_Rack_Nothing_Runs_Until_The_Operator_Arms_The_High_Voltage(
        qapp, window, monkeypatch):
    _Fast_Bench(window)
    monkeypatch.setattr(window._Rack, 'Simulated', False)
    asked = _Answer(monkeypatch, NO)
    window.Hardware.Settle.Start_Button.click()
    assert asked and '±500 V' in asked[0][1] and window._Worker is None
    assert not os.path.isdir(os.path.join(window._Data_Dir, 'hardware_calibration'))
    asked = _Answer(monkeypatch, YES)
    window.Hardware.Settle.Start_Button.click()
    assert asked and window._Worker is not None
    assert Pump(qapp, lambda: window._Worker is None)
    assert '# status: completed' in _Record(window, 'settle')


def test_The_Settle_Page_Hands_Its_Recommended_Wait_To_The_Linearity_Sweep(qapp, window):
    _Fast_Bench(window)
    hardware = window.Hardware
    assert not hardware.Settle.Wait_Button.isEnabled()
    hardware.Settle.Start_Button.click()
    assert Pump(qapp, lambda: window._Worker is None)
    assert hardware.Settle.Wait_Button.isEnabled()
    hardware.Settle.Wait_Button.click()
    results = hardware.Settle.Result[0]
    assert hardware.Linearity.Settle_Box.value() == pytest.approx(
        round(hc.Recommended_Wait(results), 2))
    assert hardware.Pages.currentWidget() is hardware.Linearity


def test_A_Settle_Model_In_Use_Is_Saved_And_Sent_With_Every_Scan(qapp, window, monkeypatch):
    _Fast_Bench(window)
    settle = window.Hardware.Settle
    settle.Start_Button.click()
    assert Pump(qapp, lambda: window._Worker is None)
    settle.Use_Button.click()
    model = settings.Load()['settle_model']
    assert model['points'] and model['points'][0][0] == 500.0
    assert settle.Model_Label.text().startswith('Measured')
    sent = []
    send = window._Link.Send
    monkeypatch.setattr(window._Link, 'Send', lambda command, **p: (
        sent.append((command, p)), send(command, **p))[1])
    _Scan(qapp, window)
    _Finished(qapp, window)
    [params] = [p for command, p in sent if command == 'Start_Scan']
    assert params['settle'] == model
    settle.Clear_Button.click()
    assert settings.Load()['settle_model'] is None


def test_A_Correction_Measured_Under_Another_Is_Refused_And_One_Can_Be_Cleared(qapp, window):
    _Fast_Bench(window)
    linearity = window.Hardware.Linearity
    linearity.Start_Button.click()
    assert Pump(qapp, lambda: window._Worker is None)
    linearity.Use_Button.click()
    assert settings.Load()['dac_correction']['measured'] is not None
    # the same sweep again, now that the correction it measured under has changed
    linearity.Use_Button.setEnabled(True)
    linearity.Use_Button.click()
    assert window.Dialogs[-1][0] == 'Correction not changed'
    assert 'changed since this sweep' in window.Dialogs[-1][1]
    linearity.Clear_Button.click()
    assert settings.Load()['dac_correction'] is None
    assert 'corrects nothing' in linearity.Report_Label.text()


def test_A_Stopped_Sweep_Keeps_What_It_Took_Analysed(qapp, window):
    _Fast_Bench(window)
    linearity = window.Hardware.Linearity
    linearity.Start_Button.click()
    assert Pump(qapp, lambda: len([p for p in linearity._Points if p['Kind'] == 'sweep']) >= 5)
    linearity.Stop_Button.click()
    assert Pump(qapp, lambda: window._Worker is None)
    record = _Record(window, 'linearity')
    assert '# status: stopped' in record and '# Gain:' in record
    assert linearity.Result is not None and window.Dialogs == []


def test_A_Failed_Sweep_Keeps_Its_Partial_Analysis(qapp, window, monkeypatch):
    _Fast_Bench(window)
    fetch, calls = window._Line.Fetch_When_Done, []

    def failing(count, timeout_s):
        calls.append(count)
        if len(calls) > 6:
            raise Scpi_Error("The meter at SIM reported errors on a capture: -230 Data stale.")
        return fetch(count, timeout_s)

    monkeypatch.setattr(window._Line, 'Fetch_When_Done', failing)
    window.Hardware.Linearity.Start_Button.click()
    assert Pump(qapp, lambda: window._Worker is None)
    record = _Record(window, 'linearity')
    assert '# status: failed: The meter at SIM reported errors' in record
    assert '# Gain:' in record                          # the six rungs it read, analysed
    title, text = window.Dialogs[-1]
    assert title == 'Linearity sweep failed' and 'What it took is in' in text
    assert window._Line.Mode == 'continuous'            # and the meter is back on its poll


def test_A_Failed_Cooler_Monitor_Keeps_Its_Partial_Analysis_And_Spectrum(
        qapp, window, monkeypatch):
    cooler = window.Hardware.Cooler
    cooler.Duration_Box.setValue(5)
    cooler.Interval_Box.setValue(2.0)
    read, calls = window._Cooler.Read_Available, []

    def failing(most):
        values = read(most)
        calls.append(len(values))
        if sum(calls) > 300:
            raise Scpi_Error("The meter at SIM stopped answering.")
        return values

    monkeypatch.setattr(window._Cooler, 'Read_Available', failing)
    cooler.Start_Button.click()
    assert Pump(qapp, lambda: window._Worker is None)
    record = _Record(window, 'cooler')
    assert '# status: failed' in record and '# Mean_V:' in record
    spectrum = [line for line in _Record(window, 'cooler_spectrum').splitlines()
                if line and not line.startswith('#')]
    assert spectrum[0] == ','.join(hc.SPECTRUM_COLUMNS) and len(spectrum) > 10


def test_The_Line_Meter_Is_Read_Precisely_Against_The_Platforms_Setpoint(qapp, window):
    state = window._Link.State
    window.Volt_Button.click()
    assert Pump(qapp, lambda: window.Voltage_Box.isEnabled())
    window.Voltage_Box.setValue(500.0)
    # straight after the write, as an operator may click: the reading waits it out
    assert Pump(qapp, lambda: state.Voltage.value == 500.0)
    window.Hardware.Read_Button.click()
    assert not window.Quick_Group.isEnabled()           # the platform holds still meanwhile
    assert Pump(qapp, lambda: window._Worker is None)
    text = window.Hardware.Read_Label.text()
    found = re.search(r"readback − setpoint ([+-][\d.]+) V \(setpoint 500 V\)", text)
    assert found, text
    # the measured rack's departure at 500 V, to the converter's code rounding
    assert float(found.group(1)) == pytest.approx(simulated.Platform_Error(500.0), abs=0.2)
    assert window._Line.Mode == 'continuous'


# ── After the run ───────────────────────────────────────────────

def test_A_Finished_Run_Is_Analysed_And_Exported_For_Denis(qapp, window):
    _Scan(qapp, window)
    _Finished(qapp, window)
    assert Pump(qapp, lambda: window.Performance.Result is not None)
    window.Performance.Export_Button.click()
    [path] = Run_Files(window)
    exported = path[:-len('.asdf')] + '_performance.yaml'
    with open(exported, encoding='utf-8') as fh:
        report = yaml.safe_load(fh)
    assert report['run'] == 'run_0001.asdf' and 'scan_filters' in report
    assert exported in window.statusBar().currentMessage()



def _Open_Run_Action(window):
    return next(action for action in window.menuBar().actions()[0].menu().actions()
                if action.text() == 'Open run...')


def test_A_Run_File_Is_Shown_As_It_Was_Taken(qapp, window, monkeypatch):
    """The old DAQ's Open: a saved run back on the Scan tab and its
    calibration on the Scan Calibration tab."""
    _Scan(qapp, window)
    _Finished(qapp, window)
    [path] = Run_Files(window)
    header, raw, _ = run_file.Read(path)
    window._Live.Clear()
    window.Calibration.Begin(window.Calibration.Plan())
    monkeypatch.setattr(main_window.QFileDialog, 'getOpenFileName',
                        lambda *_, **__: (path, 'Run files (run_*.asdf)'))
    _Open_Run_Action(window).trigger()
    assert window.Tabs.currentWidget() is window.Scan
    assert window._Live.Count == len(raw)
    assert Pump(qapp, lambda: window._Live.Spectrum.sum() == len(raw))
    assert window._Live.Volt_Edges[[0, -1]].tolist() == [19.0, 41.0]
    assert len(window.Calibration._Curves[0].getData()[0]) == len(header['CalSet']) == 11
    assert "run_0001.asdf's calibration, 11 points" in window.Calibration.Report_Label.text()
    assert window.statusBar().currentMessage().startswith(f"Showing run_0001.asdf as it was "
                                                          f"taken: {len(raw)} hits")
    assert window.Dialogs == []


def test_A_Run_Is_Not_Opened_Over_A_Scan_Or_From_A_File_That_Is_Not_One(
        qapp, window, monkeypatch, tmp_path):
    junk = tmp_path / 'run_0099.asdf'
    junk.write_bytes(b'not an ASDF file')
    monkeypatch.setattr(main_window.QFileDialog, 'getOpenFileName',
                        lambda *_, **__: (str(junk), ''))
    _Open_Run_Action(window).trigger()
    title, text = window.Dialogs[-1]
    assert title == 'Run not opened' and text.startswith('run_0099.asdf could not be read')
    _Scan(qapp, window, 'Sawtooth')
    _Open_Run_Action(window).trigger()
    assert window.Dialogs[-1] == ('Run not opened', "A scan, the triggers or a calibration is "
                                  "running, and the plots are theirs; open the run once it "
                                  "has finished.")
    window.Scan.Stop_Button.click()
    _Finished(qapp, window)


def test_Each_Hardware_Page_Shows_A_Saved_Record_As_It_Was_Taken(qapp, window, monkeypatch):
    _Fast_Bench(window)
    hardware = window.Hardware
    hardware.Cooler.Duration_Box.setValue(1)
    hardware.Cooler.Interval_Box.setValue(2.0)
    taken = {}
    for panel in hardware.Panels:
        panel.Start_Button.click()
        assert Pump(qapp, lambda: window._Worker is None)
        taken[panel.Title] = (panel.Result[1][0], panel.Report_Label.text(), panel.Result[0])
        panel.Begin()                                   # cleared, as a new session's page
        panel.Running(False)
    for panel in hardware.Panels:
        path, report, result = taken[panel.Title]
        monkeypatch.setattr(main_window.QFileDialog, 'getOpenFileName',
                            lambda *_, path=path, **__: (path, ''))
        panel.Open_Button.click()
        assert panel.Result[1] == [path]
        # the same analysis as when it was taken, from the record alone; a record
        # keeps nine figures, 0.1 mV of the cooler's 30 kV, and its last digits move
        if panel is hardware.Cooler:
            for key in ('Mean_V', 'Rms_Detrended_V', 'Oscillation_Rms_V'):
                assert panel.Result[0][key] == pytest.approx(result[key], rel=1e-4)
        else:
            assert panel.Report_Label.text().split('<br>')[0] == report.split('<br>')[0]
    assert hardware.Settle.Table.rowCount() == 2 and hardware.Settle.Wait_Button.isEnabled()
    assert len(hardware.Settle.Response_Plot.listDataItems()) == 2
    assert not hardware.Linearity.Use_Button.isEnabled()
    assert 'Opened from its record' in hardware.Linearity.Report_Label.text()
    assert hardware.Cooler.Peak_Table.rowCount() > 0 and window.Dialogs == []


def test_A_File_That_Is_No_Record_Is_Refused_On_Its_Page(qapp, window, monkeypatch, tmp_path):
    other = tmp_path / 'cooler_2026-10-27_120000.csv'
    other.write_text("# Rack: simulated\nstep,mode\n1,out\n", encoding='utf-8')
    monkeypatch.setattr(main_window.QFileDialog, 'getOpenFileName',
                        lambda *_, **__: (str(other), ''))
    window.Hardware.Cooler.Open_Button.click()
    title, text = window.Dialogs[-1]
    assert title == 'Record not opened' and 'lacks the columns time_s, cooler_v' in text
    assert 'FAIL' in window.Hardware.Cooler.Report_Label.text()
