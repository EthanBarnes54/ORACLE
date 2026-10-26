"""ORACLE's main window and entry point.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The old DAQ's window: the Scan, Scan Calibration and Time graphs tabs,
with the bench tool's Hardware Calibration and the Scan Performance
analysis between them and the data viewer's Run viewer after them,
above the quick-tune controls and the experiment information. The window
composes the pieces and owns the run: Start scan calibrates the scan's
own plan, puts the calibration through the gate, and only then writes
the run file's header and starts the scan; a REJECT writes nothing and
says why. Data are appended to the run file as they arrive, and the file
is closed when the acquisition process marks the scan's end. A hardware
calibration runs on the same worker slot as a calibration, so it locks
the same controls; its record is saved beside the run files, and a
linearity sweep's DAC correction goes into force only when the operator
says so, and into the settings with it. Each run is analysed for its
performance as it ends, off the GUI thread.

``main()`` sets up the session log, one instance per user (two would
fight over the instruments), the theme and splash, and the rack: the
acquisition process, the meters and the facility services, all closed
again if any one of them fails to open.

Depends on: PySide6, pyqtgraph; cls_acquisition and the gui tabs.
"""

__version__ = '0.2.0'

import contextlib
import getpass
import logging
import math
import multiprocessing
import os
import platform
import sys
import tempfile
import time
from datetime import datetime
from functools import partial

import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QFontDatabase,
                           QKeySequence, QPainter, QPixmap)
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (QApplication, QFileDialog, QGridLayout, QGroupBox, QLabel,
                               QMainWindow, QMessageBox, QPushButton, QSlider, QSplashScreen,
                               QTabWidget, QWidget)

from cls_acquisition import (calibration, hardware_calibration, run_file, scan_performance,
                             settings)
from cls_acquisition.acquisition import Acquisition_Error, Rack_Link
from cls_acquisition.facility import Cooler_Pv_Server, Elog, Pv_Monitors
from cls_acquisition.hardware import Rack
from cls_acquisition.hardware import mcc_dac
from cls_acquisition.hardware.keysight import COOLER_DIVIDER
from cls_acquisition.scan_plan import Scan_Plan_Error
from gui import plots, session_log, theme
from gui.calibration_tab import Calibration_Tab
from gui.hardware_tab import Hardware_Tab
from gui.live_data import Live_Data
from gui.performance_tab import Performance_Tab
from gui.scan_tab import Scan_Tab
from gui.shared_widgets import Job_Worker, Spin, Verdict_Html
from gui.time_tab import Time_Tab
from gui.viewer_tab import Viewer_Tab

TICK_MS = 100
MONITOR_FLUSH_TICKS = 10            # the monitor record is written once a second
SLIDER_STEPS = 10000
STARTUP_TIMEOUT_S = 30.0            # the acquisition process opens its devices first

log = logging.getLogger('oracle.gui')


def _Number(value, fmt):
    return 'unavailable' if value is None else format(value, fmt)


class Main_Window(QMainWindow):

    def __init__(self, config, rack, link, line_meter, cooler_meter, pvs, elog, warnings=()):
        super().__init__()
        self._Rack, self._Link = rack, link
        self._Line, self._Cooler = line_meter, cooler_meter
        self._Pvs, self._Elog = pvs, elog
        self._State = link.State
        self._Data_Dir = settings.Data_Directory(config)
        self._Live = Live_Data()
        self._Volt_Edges = np.array([0.0, 1.0])
        self._Run = None                # path of the run file being written
        self._Pending = None            # the scan waiting on its calibration
        self._Worker = None             # the calibration in progress, scan or hardware
        self._Correction = config['dac_correction'] or mcc_dac.CORRECTION
        self._Settle_Model = config['settle_model']
        self._Analysis = None           # the performance analysis in progress
        self._Laser = config['laser']
        self._Monitor = []              # monitor rows of the run, not yet written
        self._Death_Shown = False

        rack_name = 'simulated rack' if rack.Simulated else 'rack'
        self.setWindowTitle(f"ORACLE v{__version__} ({rack_name})")
        self.resize(1250, 1040)

        self.Tabs = QTabWidget(objectName='MainTabs', documentMode=True)
        self.Tabs.tabBar().setDrawBase(False)
        self.Tabs.setStyleSheet(theme.Main_Tabs_Qss())
        self._Limit_V = config['instruments']['hv_limit_v']
        self.Scan, self.Time = Scan_Tab(self._Limit_V), Time_Tab()
        self.Calibration = Calibration_Tab(self._Limit_V)
        self.Hardware = Hardware_Tab(self._Limit_V)
        self.Performance = Performance_Tab()
        self.Viewer = Viewer_Tab(self._Data_Dir, config['laser'])
        for tab, icon, name in ((self.Scan, 'atom', 'Scan controls'),
                                (self.Calibration, 'crosshair', 'Scan Calibration'),
                                (self.Hardware, 'zap', 'Hardware Calibration'),
                                (self.Performance, 'stethoscope', 'Scan Performance'),
                                (self.Time, 'hourglass', 'Time graphs'),
                                (self.Viewer, 'bar-chart-2', 'Run viewer')):
            self.Tabs.addTab(tab, theme.Icon(icon), name)
        self.Scan.Experiment_Edit.setText(config['experiment'])

        central = QWidget()
        layout = QGridLayout(central)
        layout.addWidget(self.Tabs, 0, 0)
        layout.addWidget(self._Build_Quick_Tune(), 1, 0)
        layout.addWidget(self._Build_Information(), 2, 0)
        layout.setRowStretch(0, 1)
        self.setCentralWidget(central)
        self._Build_Menus()
        rack_label = QLabel(f"  {rack_name.capitalize()}  ")
        if rack.Simulated:
            rack_label.setStyleSheet(f"color: {theme.TRACE_ALT}; font-weight: bold;")
        self.statusBar().addPermanentWidget(rack_label)
        self.statusBar().showMessage(f"Run files go to {self._Data_Dir}")
        if warnings:
            # a fault of the rack's own that ORACLE cannot mend: shown for the
            # whole session, not once, and its sentences in the tooltip
            warning_label = QLabel("  Rack warning  ")
            warning_label.setStyleSheet(f"color: {theme.VERDICT_COLOURS['WARN']}; "
                                        f"font-weight: bold;")
            warning_label.setToolTip('\n\n'.join(warnings))
            self.statusBar().addPermanentWidget(warning_label)
            for warning in warnings:
                log.warning("Rack: %s", warning)

        self.Scan.Start_Button.clicked.connect(self._Toggle_Scan)
        self.Scan.Stop_Button.clicked.connect(self._Stop_Now)
        self.Scan.Send_Text_Button.clicked.connect(self._Send_Elog_Text)
        self.Scan.New_Entry_Button.clicked.connect(
            lambda: self._Elog.New_Entry(self._Experiment()))
        self.Scan.Screen_Button.clicked.connect(self._Send_Screen)
        self.Scan.Experiment_Edit.editingFinished.connect(
            lambda: settings.Update(experiment=self._Experiment()))
        self.Calibration.Start_Button.clicked.connect(self._Calibrate_Only)
        hardware = self.Hardware
        on_line = (link, line_meter, calibration.DIVIDER)
        for panel, measure in (
                (hardware.Settle, partial(hardware_calibration.Settle_Test, *on_line)),
                (hardware.Linearity, partial(hardware_calibration.Linearity_Sweep, *on_line)),
                (hardware.Cooler, partial(hardware_calibration.Cooler_Monitor, cooler_meter,
                                          COOLER_DIVIDER))):
            panel.Start_Button.clicked.connect(
                lambda _=False, panel=panel, measure=measure: self._Run_Bench(panel, measure))
            panel.Stop_Button.clicked.connect(self._Stop_Bench)
        hardware.Linearity.Use_Button.clicked.connect(self._Use_Correction)
        hardware.Linearity.Clear_Button.clicked.connect(lambda: self._Set_Correction(None))
        hardware.Settle.Use_Button.clicked.connect(lambda: self._Set_Settle_Model(
            hardware_calibration.Settle_Model(hardware.Settle.Result[0])))
        hardware.Settle.Clear_Button.clicked.connect(lambda: self._Set_Settle_Model(None))
        hardware.Settle.Show_Model(self._Settle_Model)
        performance = self.Performance
        performance.Refresh_Button.clicked.connect(self._Refresh_Runs)
        performance.Browse_Button.clicked.connect(self._Browse_Run)
        performance.Analyse_Button.clicked.connect(
            lambda: performance.Selected() and self._Analyse(performance.Selected()))
        performance.Export_Button.clicked.connect(self._Export_Performance)
        self._Refresh_Runs()
        if performance.Follow_Check.isChecked() and performance.Selected():
            self._Analyse(performance.Selected())

        self._Timer = QTimer(self, interval=TICK_MS, timeout=self._Tick)
        self._Timer.start()

    # ── Layout ───────────────────────────────────────────────────

    def _Build_Quick_Tune(self):
        self.Quick_Group = QGroupBox('Quick tune controls')
        grid = QGridLayout(self.Quick_Group)
        self.Trigger_Button = QPushButton('Enable triggers', clicked=self._Toggle_Triggers)
        self.Volt_Button = QPushButton('Enable voltage', clicked=self._Toggle_Voltage)
        limit = self._Limit_V
        self.Voltage_Box = Spin(0.0, -limit, limit, 1, ' V')
        self.Sweep_From = Spin(-min(3000.0, limit), -limit, limit, 1)
        self.Sweep_To = Spin(min(3000.0, limit), -limit, limit, 1)
        self.Slider = QSlider(Qt.Orientation.Horizontal, maximum=SLIDER_STEPS)
        for column, widget in enumerate((self.Trigger_Button, self.Volt_Button,
                                         QLabel('Set voltage:'), self.Voltage_Box,
                                         self.Sweep_From, self.Slider, self.Sweep_To)):
            grid.addWidget(widget, 0, column)
        grid.setColumnStretch(5, 1)
        self.Voltage_Box.valueChanged.connect(self._Box_Moved)
        self.Slider.valueChanged.connect(self._Slider_Moved)
        return self.Quick_Group

    def _Build_Information(self):
        group = QGroupBox('Experiment information')
        grid = QGridLayout(group)
        (self.Cooler_Label, self.Readback_Label, self.Wavenumber_Label,
         self.Run_Label, self.Round_Label, self.Mode_Label) = (QLabel() for _ in range(6))
        if os.path.isdir(self._Data_Dir):
            self.Run_Label.setText(str(run_file.Last_Run_Number(self._Data_Dir)))
        cells = ((0, 'Cooler voltage:', self.Cooler_Label),
                 (0, 'Readback voltage:', self.Readback_Label),
                 (0, 'Wavenumber:', self.Wavenumber_Label),
                 (1, 'Run number:', self.Run_Label),
                 (1, 'Scan number:', self.Round_Label),
                 (1, 'Buncher:', self.Mode_Label))
        for i, (row, text, label) in enumerate(cells):
            grid.addWidget(QLabel(text), row, 2 * (i % 3))
            grid.addWidget(label, row, 2 * (i % 3) + 1)
            grid.setColumnStretch(2 * (i % 3) + 1, 1)
        return group

    def _Build_Menus(self):
        file_menu = self.menuBar().addMenu('&File')
        for text, folder in (('Open data folder', self._Data_Dir),
                             ('Open log folder', os.path.join(settings.APP_ROOT, 'logs'))):
            action = QAction(theme.Icon('folder-open'), text, self)
            action.triggered.connect(lambda _=False, f=folder: QDesktopServices.openUrl(
                QUrl.fromLocalFile(f)))
            file_menu.addAction(action)
        view_runs = QAction(theme.Icon('bar-chart-2'), 'Open runs in the viewer...', self)
        view_runs.triggered.connect(lambda: (self.Tabs.setCurrentWidget(self.Viewer),
                                             self.Viewer.Browse()))
        file_menu.addAction(view_runs)
        file_menu.addSeparator()
        file_menu.addAction('Quit', self.close)
        view_menu = self.menuBar().addMenu('&View')
        autoscale = QAction(theme.Icon('rotate-ccw'), 'Autoscale all plots', self,
                            shortcut=QKeySequence('Ctrl+0'))
        autoscale.triggered.connect(self._Autoscale_All)
        view_menu.addAction(autoscale)
        self.menuBar().addMenu('&Help').addAction('About ORACLE', self._About)

    def _Autoscale_All(self):
        for plot in self.findChildren(plots.Plot_Widget):
            plot.Autoscale()

    def _About(self):
        box = QMessageBox(
            QMessageBox.Icon.NoIcon, 'About ORACLE',
            f"<b>ORACLE v{__version__}</b><br>Online Readout &amp; Acquisition for Collinear "
            f"Laser spectroscopy Experiments<br><br>"
            f"Rack: {'simulated' if self._Rack.Simulated else 'hardware'}<br>"
            f"Run files: {self._Data_Dir}<br>Settings: {settings.Settings_Path()}",
            QMessageBox.StandardButton.Ok, self)
        box.setIconPixmap(theme.Logo().scaled(180, 180, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))
        box.exec()

    # ── The run ──────────────────────────────────────────────────

    def _Experiment(self):
        return self.Scan.Experiment_Edit.text()

    def _Send(self, command, **params):
        """An instruction from a button; a refusal is shown, never raised."""
        try:
            return self._Link.Send(command, **params)
        except Acquisition_Error as e:
            QMessageBox.warning(self, 'Instruction refused', str(e))
            return None

    def _Next_Run(self):
        """The next run number, from the data folder; refuses an unreachable
        folder on the rack rather than restart the numbering from 1."""
        if self._Rack.Simulated:
            os.makedirs(self._Data_Dir, exist_ok=True)
        elif not os.path.isdir(self._Data_Dir):
            raise OSError(f"The data folder {self._Data_Dir} is not reachable, so no run "
                          f"number can be given.")
        return run_file.Last_Run_Number(self._Data_Dir) + 1

    def _Start_Live(self, plan):
        self._Live.Clear()
        self._Volt_Edges = plan.Bin_Edges() if plan is not None else np.array([0.0, 1.0])

    def _Toggle_Scan(self):
        if self._State.Scanning.is_set():
            if self._Send('Stop_After_Scan') is not None:
                self._Elog.Append(f"\n\nSCAN STOPPED AFTER {self._State.Round.value} ROUNDS",
                                  self._Experiment())
            return
        if self._Worker is not None:
            return                      # a second click before the controls caught up
        try:
            plan = self.Scan.Plan()
            timing = self.Scan.Timing()
            run = self._Next_Run()
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, 'Scan not started', str(e))
            return

        self._Start_Live(plan)
        self.Sweep_From.setValue(plan.Setpoints.min())
        self.Sweep_To.setValue(plan.Setpoints.max())
        when = datetime.now()
        pvs = {key: self._Pvs.Value(key) for key in ('wavenumber', 'mass', 'mass_setpoint')}
        bunches = self.Scan.Bunches_Per_Step()
        text = (f"\n--------------------------------------\n{when:%d/%m/%Y %H:%M:%S}\n"
                f"START SCAN NUMBER {run}\nscanning ranges:\n"
                + ''.join(f"from {low:g} to {high:g} V\n" for low, high in plan.Ranges)
                + f"\nBunches per channel: {bunches}\n"
                f"Stepsize [V]:        {plan.Step:g}\n"
                f"Dwell time [ms]:     {self.Scan.Dwell_Box.value()}\n"
                f"Laser Set Point [cm-1]: {_Number(pvs['wavenumber'], '.5f')}\n"
                f"Mass [amu]:          {_Number(pvs['mass'], '.4f')}\n"
                f"Mass setpoint [A or amu]:          {_Number(pvs['mass_setpoint'], '.4f')}\n")
        self._Pending = {'plan': plan, 'run': run, 'when': when, 'pvs': pvs, 'text': text,
                         'bunches': bunches, 'timing': timing,
                         'scan_type': self.Scan.Type_Combo.currentText()}
        self._Calibrate(plan)

    def _Begin_Scan(self, pending, cal, report):
        """After the calibration: the gate's verdict, then the file and the scan."""
        applied, description = cal.Dac_Correction
        text = (pending['text'] + f"Calibration:  {calibration.Summary(report)}\n"
                f"DAC correction: {description}\n"
                + ''.join(f"  {c['Verdict']}  {c['Check']}: {c['Message']}\n"
                          for c in report['Checks'] if c['Verdict'] != calibration.PASS))
        if report['Verdict'] == calibration.REJECT:
            self._Elog.Append(text + "RUN REJECTED: no file written, no scan started.\n",
                              self._Experiment())
            QMessageBox.critical(self, 'Run rejected', f"{calibration.Rejections(report)}\n\n"
                                                       f"No file written, no scan started.")
            return
        self._Elog.Append(text, self._Experiment())

        def pv(key):
            value = pending['pvs'][key]
            return math.nan if value is None else float(value)

        plan = pending['plan']
        header = {'Run': pending['run'], 'Experiment': self._Experiment(),
                  'Date': pending['when'], 'BunchesPerChannel': pending['bunches'],
                  'StepSize': plan.Step, 'DwellTime': self.Scan.Dwell_Box.value(),
                  'MassAMU': pv('mass'), 'MassSetPoint': pv('mass_setpoint'),
                  'CoolerVoltage': self._State.Cooler.value, 'LaserSetpoint': pv('wavenumber'),
                  'ScanningRanges': plan.Ranges, 'CalSet': cal.Commanded,
                  'CalReadback': cal.Readings, 'DacCorrectionApplied': applied,
                  'DacCorrection': description}
        try:
            self._Run = run_file.Create(self._Data_Dir, header)
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, 'Run file not written', str(e))
            return
        started = self._Send('Start_Scan', setpoints=plan.Setpoints.tolist(),
                             scan_type=pending['scan_type'],
                             bunches_per_step=pending['bunches'], timing=pending['timing'],
                             settle=self._Settle_Model)
        if started is None:
            with contextlib.suppress(OSError):
                os.remove(self._Run)    # a header with no scan behind it is not a run
            self._Run = None
            return
        self.Run_Label.setText(str(pending['run']))
        log.info("Run %d started: %s", pending['run'], self._Run)

    def _End_Run(self):
        finished = self._Run
        if finished is not None:
            self._Sample_Monitor()      # the loop's final counts, now it has stopped
            self._Flush_Monitor()
            log.info("Run file closed: %s", finished)
        self._Run = None
        self._Monitor = []
        if finished is not None:
            self._Refresh_Runs(keep=finished)
            if self.Performance.Follow_Check.isChecked():
                self._Analyse(finished)

    def _Stop_Now(self):
        if self._Send('Stop_Now') is not None:
            self._Elog.Append("\n\nSCAN KILLED", self._Experiment())

    def _Toggle_Triggers(self):
        if self._State.Triggering.is_set():
            self._Send('Stop_Triggering')
            return
        try:
            timing = self.Scan.Timing()
        except ValueError as e:
            QMessageBox.warning(self, 'Triggers not started', str(e))
            return
        try:
            plan = self.Scan.Plan()
        except Scan_Plan_Error:
            plan = None             # the TOF still shows; the spectrum has no axis
        self._Start_Live(plan)
        self._Send('Start_Triggering', timing=timing)

    def _Toggle_Voltage(self):
        on = not self._State.Outputting.is_set()
        if on:
            # from the voltage the box shows, which is what is on the platform
            self._State.Requested.value = self.Voltage_Box.value()
        self._Send('Output_Volts', on=on)

    def _Box_Moved(self, volts):
        self._State.Requested.value = volts
        self._Sync_Slider(volts)

    def _Sync_Slider(self, volts):
        low, high = self.Sweep_From.value(), self.Sweep_To.value()
        if high > low:
            self.Slider.blockSignals(True)
            self.Slider.setValue(round((volts - low) / (high - low) * SLIDER_STEPS))
            self.Slider.blockSignals(False)

    def _Slider_Moved(self, position):
        low, high = self.Sweep_From.value(), self.Sweep_To.value()
        self.Voltage_Box.setValue(low + (high - low) * position / SLIDER_STEPS)

    # ── Calibration ──────────────────────────────────────────────

    def _Calibrate_Only(self):
        if self._Worker is not None:
            return
        try:
            plan = self.Calibration.Plan()
        except Scan_Plan_Error as e:
            QMessageBox.warning(self, 'Calibration not started', str(e))
            return
        self._Calibrate(plan)

    def _Calibrate(self, plan):

        def job(progress, stopped):
            cal = calibration.Run_Calibration(self._Link, self._Line, plan,
                                              lambda *point: progress(point), self._Settle_Model)
            return cal, calibration.Quality_Report(cal, plan)

        self._Worker = Job_Worker(job, self)
        self._Worker.Progress.connect(self.Calibration.Show_Progress)
        self._Worker.Done.connect(lambda result: self._Calibrated(*result))
        self._Worker.Failed.connect(self._Calibration_Failed)
        self._Worker.finished.connect(self._Worker_Finished)
        self.Calibration.Begin(plan)
        self._Worker.start()
        self._Show_State()              # disable the controls now, not at the next tick

    def _Calibrated(self, cal, report):
        self.Calibration.Show_Report(report)
        log.info("Calibration: %s", calibration.Summary(report))
        pending, self._Pending = self._Pending, None
        if pending is not None:
            self._Begin_Scan(pending, cal, report)

    def _Calibration_Failed(self, message):
        self.Calibration.Show_Failure(message)
        pending, self._Pending = self._Pending, None
        if pending is not None:
            self._Elog.Append(pending['text'] + f"Calibration failed: {message}\n"
                              f"No file written, no scan started.\n", self._Experiment())
        QMessageBox.critical(self, 'Calibration failed', message)

    def _Worker_Finished(self):
        self._Worker.Retire()
        self._Worker = None
        self._Show_State()              # the controls back now, as they went

    # ── Hardware calibration ─────────────────────────────────────

    def _Run_Bench(self, panel, measure):
        """Run ``panel``'s measurement, ``measure(progress=, stopped=, records=,
        **its parameters)``, on the worker, its records written as it goes;
        on the rack, only once the operator has armed the high voltage."""
        if self._Worker is not None:
            return
        try:
            params = panel.Params()
        except ValueError as e:
            QMessageBox.warning(self, f"{panel.Title} not started", str(e))
            return
        peak = panel.Max_Volts(params)
        if peak is not None and not self._Rack.Simulated and not self._Arm_Hv(panel.Title, peak):
            return
        header = {'ORACLE': __version__, 'Rack': 'simulated' if self._Rack.Simulated else
                  'hardware', 'Date': f"{datetime.now():%Y-%m-%d %H:%M:%S}", **params}
        records = {}
        try:
            for kind, columns in panel.RECORDS.items():
                records[kind] = hardware_calibration.Record(self._Data_Dir, kind, header,
                                                            columns)
        except OSError as e:
            self._Close_Records(records, "not started: a record could not be written")
            QMessageBox.warning(self, f"{panel.Title} not started",
                                f"Its record could not be written, so it was not started: {e}")
            return
        worker = Job_Worker(lambda progress, stopped: measure(
            progress=progress, stopped=stopped, records=records, **params), self)

        def done(result):
            self._Close_Records(records, 'stopped' if worker.Stopped() else 'completed',
                                panel.Results(result))
            panel.Show_Result(result, [record.Path for record in records.values()])

        def failed(message):
            self._Close_Records(records, f"failed: {message}")
            panel.Show_Failure(message)
            QMessageBox.critical(self, f"{panel.Title} failed",
                                 f"{message}\n\nWhat it took is in "
                                 f"{', '.join(r.Path for r in records.values())}.")

        self._Worker = worker
        worker.Progress.connect(panel.Show_Progress)
        worker.Done.connect(done)
        worker.Failed.connect(failed)
        worker.finished.connect(lambda: panel.Running(False))
        worker.finished.connect(self._Worker_Finished)
        panel.Begin()
        worker.start()
        self._Show_State()

    def _Arm_Hv(self, title, peak):
        """True if the operator confirms the rack is ready for the platform
        to go to ``+-peak`` volts."""
        return QMessageBox.question(
            self, 'Arm high voltage',
            f"The {title.lower()} will drive the platform up to ±{peak:g} V.\n\n"
            f"• The HV area is safe and interlocked.\n"
            f"• The divider is connected to the line and to the line meter.\n\n"
            f"Arm the high voltage and start?") == QMessageBox.StandardButton.Yes

    def _Close_Records(self, records, status, results=None):
        for kind, record in records.items():
            try:
                record.Close(status, (results or {}).get(kind))
            except OSError:
                log.exception("Closing the %s record failed", kind)
            log.info("Hardware calibration record (%s): %s", status, record.Path)

    def _Stop_Bench(self):
        if self._Worker is not None:
            self._Worker.Stop()

    def _Set_Settle_Model(self, model):
        """Scans settle by ``model`` (None: SETTLE for every step) from the next one."""
        self._Settle_Model = model
        settings.Update(settle_model=model)
        self.Hardware.Settle.Use_Button.setEnabled(False)
        self.Hardware.Settle.Show_Model(model)
        log.info("Scan settle model: %s", model)

    def _Use_Correction(self):
        result, paths = self.Hardware.Linearity.Result
        if result['Dac']['Correction'] != self._Correction:
            QMessageBox.warning(self, 'Correction not changed',
                                "The DAC correction has changed since this sweep, so its "
                                "measurement no longer describes the converters; sweep again.")
            return
        try:
            correction = hardware_calibration.Corrected(
                self._Correction, result['Fits'], self.Hardware.Linearity.Valid_Box.value(),
                f"linearity sweep, {os.path.basename(paths[0])}", result['Inl'])
        except calibration.Calibration_Error as e:
            QMessageBox.warning(self, 'Correction not changed', str(e))
            return
        self._Set_Correction(correction)

    def _Set_Correction(self, correction):
        """Put ``correction`` (None: identity) into force and into the settings."""
        reply = self._Send('Set_Correction', correction=correction or mcc_dac.CORRECTION)
        if reply is None:
            return
        self._Correction = correction or mcc_dac.CORRECTION
        settings.Update(dac_correction=correction)
        self.Hardware.Linearity.Use_Button.setEnabled(False)
        verdict, message = reply['Dac_Status']
        log.info("DAC correction: %s", reply['Dac_Correction'][1])
        which = ("gain, offset and non-linearity" if (correction or {}).get('inl')
                 else "gain and offset" if correction else "nothing")
        self.Hardware.Linearity.Report_Label.setText(
            Verdict_Html(verdict, f"{message}; corrects {which}; in force now and saved in the "
                                  f"settings."))

    # ── Scan performance ─────────────────────────────────────────

    def _Refresh_Runs(self, keep=None):
        runs = run_file.Runs(self._Data_Dir) if os.path.isdir(self._Data_Dir) else []
        self.Performance.Set_Runs(runs, keep)

    def _Browse_Run(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Analyse a run', self._Data_Dir,
                                              'Run files (run_*.asdf)')
        if path:
            self._Analyse(path)

    def _Analyse(self, path):
        if self._Analysis is not None:
            return
        laser = self._Laser

        def job(progress, stopped):
            return scan_performance.Analyse(*run_file.Read(path), laser)

        self._Analysis = Job_Worker(job, self)
        self._Analysis.Done.connect(lambda result: self.Performance.Show(path, result))
        self._Analysis.Failed.connect(self.Performance.Show_Failure)
        self._Analysis.finished.connect(self._Analysis_Finished)
        self.Performance.Begin(path)
        self._Analysis.start()

    def _Analysis_Finished(self):
        self._Analysis.Retire()
        self._Analysis = None

    def _Export_Performance(self):
        path, result = self.Performance.Result
        try:
            written = scan_performance.Export(path, result)
        except OSError as e:
            QMessageBox.warning(self, 'Not exported', f"The performance report could not be "
                                                     f"written: {e}")
            return
        self.statusBar().showMessage(f"Performance report and DENIS scan filter: {written}",
                                     10000)

    # ── ELOG ─────────────────────────────────────────────────────

    def _Send_Elog_Text(self):
        self._Elog.Append(self.Scan.Elog_Edit.toPlainText(), self._Experiment())
        self.Scan.Elog_Edit.clear()

    def _Send_Screen(self):
        name = f"Scan{self.Run_Label.text()}_Round{self._State.Round.value}_" \
               f"{datetime.now():%d-%m-%Y_%H-%M-%S}"
        handle, path = tempfile.mkstemp(prefix=name, suffix='.png')
        os.close(handle)
        if not self.Scan.Spectrum_Plot.grab().save(path):
            os.remove(path)
            QMessageBox.warning(self, 'Screen not sent', f"The spectrum could not be saved "
                                                         f"to {path}.")
            return
        self._Elog.Append(f"(see attached spectra {name})", self._Experiment(), path)

    # ── Every tick ───────────────────────────────────────────────

    def _Tick(self):
        if self._Run is not None and self._State.Scanning.is_set():
            self._Sample_Monitor()
        self._Take_Data()
        scan = self.Scan
        self._Live.Configure(self._Volt_Edges, scan.Tof_Edges(), scan.Tof_Plot.Gate(),
                             scan.Channels(), scan.Map_Check.isChecked())
        self._Live.Update()
        if self.Tabs.currentWidget() is scan:
            scan.Draw(self._Live)
        elif self.Tabs.currentWidget() is self.Time:
            self.Time.Draw(self._Live, scan.Trigger_Hz(), scan.Tof_Plot.Gate())
        self._Show_State()
        if self._Elog.Last_Error:
            self.statusBar().showMessage(self._Elog.Last_Error, 10000)
            self._Elog.Last_Error = None
        if not self._Death_Shown and not self._Link.Alive():
            self._Death_Shown = True
            QMessageBox.critical(self, 'Acquisition stopped',
                                 "The acquisition process has stopped; the session log says "
                                 "why. Restart ORACLE.")

    def _Take_Data(self):
        scan_rows = []
        for kind, rows in self._Link.Drain():
            if kind == 'scan_end':
                self._Write(scan_rows)
                scan_rows = []
                self._End_Run()
                continue
            self._Live.Add(rows)
            if kind == 'scan':
                scan_rows.append(rows)
        self._Write(scan_rows)

    def _Sample_Monitor(self):
        """One row of what the rack is doing, for the run's monitor record."""
        wavenumber = self._Pvs.Value('wavenumber')
        state = self._State
        self._Monitor.append((time.time(), state.Round.value, state.Voltage.value,
                              self._Line.Voltage * calibration.DIVIDER,
                              self._Cooler.Voltage * COOLER_DIVIDER,
                              math.nan if wavenumber is None else wavenumber,
                              state.Bunches.value, state.Dropped.value, state.Skipped.value))
        if len(self._Monitor) >= MONITOR_FLUSH_TICKS:
            self._Flush_Monitor()

    def _Flush_Monitor(self):
        rows, self._Monitor = self._Monitor, []
        if not rows:
            return
        try:
            run_file.Append_Monitor(self._Run, rows)
        except OSError:
            # the run itself is not at stake; its performance analysis loses these
            log.exception("Writing the monitor record failed")

    def _Write(self, scan_rows):
        if not scan_rows or self._Run is None:
            return
        try:
            run_file.Append(self._Run, np.vstack(scan_rows))
        except OSError as e:
            log.exception("Writing the run file failed")
            self._Run = None
            self._Send('Stop_Now')
            QMessageBox.critical(self, 'Scan stopped', f"The run file could not be written, "
                                                       f"so the scan was stopped: {e}")

    def _Show_State(self):
        state = self._State
        scanning, triggering = state.Scanning.is_set(), state.Triggering.is_set()
        outputting, calibrating = state.Outputting.is_set(), self._Worker is not None
        self._State.Cooler.value = self._Cooler.Voltage

        self.Cooler_Label.setText(f"{self._Cooler.Voltage * COOLER_DIVIDER:.2f} V")
        self.Readback_Label.setText(f"{self._Line.Voltage * calibration.DIVIDER:.1f} V")
        self.Wavenumber_Label.setText(_Number(self._Pvs.Value('wavenumber'), '.5f') + ' cm⁻¹')
        self.Round_Label.setText(str(state.Round.value))
        self.Mode_Label.setText('Bunched' if self.Scan.Bunched() else 'DC')

        scan = self.Scan
        scan.Params_Group.setEnabled(not (scanning or triggering or calibrating))
        scan.Buncher_Group.setEnabled(not (scanning or triggering or calibrating))
        scan.Control_Group.setEnabled(not (calibrating or triggering))
        scan.Start_Button.setText('Stop after scan' if scanning else 'Start scan')
        scan.Stop_Button.setEnabled(scanning)
        busy = calibrating or scanning
        hardware = self.Hardware
        for widget in (self.Calibration.Params_Group, self.Calibration.Control_Group,
                       self.Quick_Group, scan.New_Entry_Button, hardware.Ramp_Group,
                       hardware.Linearity.Correction_Group, hardware.Settle.Model_Group,
                       *(panel.Params_Group for panel in hardware.Panels),
                       *(panel.Start_Button for panel in hardware.Panels)):
            widget.setEnabled(not busy)
        self.Trigger_Button.setText('Disable triggers' if triggering else 'Enable triggers')
        self.Volt_Button.setText('Disable voltage' if outputting else 'Enable voltage')
        self.Voltage_Box.setEnabled(outputting)
        self.Slider.setEnabled(outputting)
        if not outputting:
            # with the output off the box shows what is on the platform
            self.Voltage_Box.blockSignals(True)
            self.Voltage_Box.setValue(state.Voltage.value)
            self.Voltage_Box.blockSignals(False)
            self._Sync_Slider(state.Voltage.value)

    def closeEvent(self, event):
        if self._Worker is not None:
            self.statusBar().showMessage("A calibration is running; close ORACLE once it "
                                         "has finished.", 10000)
            event.ignore()
            return
        if self._State.Scanning.is_set() and QMessageBox.question(
                self, 'Quit ORACLE', "A scan is running. Stop it and quit?"
                ) != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        self._Timer.stop()
        if self._Analysis is not None:
            self._Analysis.wait()       # a thread cannot outlive its window
        self.Viewer.Close()
        self._Take_Data()
        self._Link.Close()              # stops any scan and leaves the platform at 0 V
        self._Take_Data()
        for close in (self._Line.Close, self._Cooler.Close, self._Pvs.Close, self._Elog.Close,
                      lambda: settings.Update(experiment=self._Experiment())):
            try:
                close()
            except Exception:
                log.exception("Closing ORACLE: %s failed", close.__qualname__)
        event.accept()


def _Single_Instance(app):
    """True if this is the only ORACLE of this user; otherwise the running
    one is raised and this one should exit."""
    name = f"ORACLE-{getpass.getuser()}"
    probe = QLocalSocket()
    probe.connectToServer(name)
    if probe.waitForConnected(300):
        probe.disconnectFromServer()
        return False
    QLocalServer.removeServer(name)            # a stale socket from a crash
    app._Oracle_Server = QLocalServer()
    app._Oracle_Server.listen(name)
    return True


def _Splash():
    """DENIS's splash: the logo on a dark canvas, the name and version below.
    It stays up while the rack opens, which can take STARTUP_TIMEOUT_S."""
    logo = theme.Logo()
    canvas = QPixmap(540, 600)
    canvas.fill(QColor(24, 24, 30))
    painter = QPainter(canvas)
    painter.drawPixmap((540 - logo.width()) // 2, 20, logo)
    # the platform's UI font, sized for 96 dpi because the canvas is laid out in pixels
    family = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    pt_scale = 96 / canvas.logicalDpiY()
    painter.setPen(QColor(220, 220, 220))
    painter.setFont(QFont(family, round(18 * pt_scale), QFont.Weight.Bold))
    painter.drawText(0, 545, 540, 30, Qt.AlignmentFlag.AlignCenter, 'ORACLE')
    painter.setPen(QColor(160, 160, 170))
    painter.setFont(QFont(family, round(10 * pt_scale)))
    painter.drawText(0, 572, 540, 20, Qt.AlignmentFlag.AlignCenter,
                     f"v{__version__} — Online Readout & Acquisition for Collinear Laser "
                     f"spectroscopy Experiments")
    painter.end()
    splash = QSplashScreen(canvas)
    splash.show()
    return splash


def main():
    log_path = session_log.Install(settings.APP_ROOT)
    session_log.Install_Qt_Handler()
    config = settings.Load()

    print('=' * 60)
    print(f"ORACLE v{__version__}")
    print('=' * 60)
    print(f"Python:     {sys.version.split()[0]}")
    print(f"Platform:   {platform.platform()}")
    print(f"App dir:    {settings.APP_ROOT}")
    print(f"Rack:       {config['rack']}")
    print(f"Settings:   {settings.Settings_Path()}")
    if log_path:
        print(f"Log file:   {log_path}")
    print('-' * 60)

    app = QApplication(sys.argv)
    app.setApplicationName('ORACLE')
    if not _Single_Instance(app):
        return 0
    theme.Apply_Theme(app)
    app.setWindowIcon(theme.App_Icon())
    splash = _Splash()
    app.processEvents()

    try:
        with contextlib.ExitStack() as opened:
            rack = Rack(config, multiprocessing.get_context('spawn'))
            link = Rack_Link(rack.Loop_Spec())
            opened.callback(link.Close)
            warnings = link.Send('Ping', timeout=STARTUP_TIMEOUT_S)['Warnings']
            line_meter, cooler_meter = rack.Open_Meters()
            opened.callback(line_meter.Close)
            opened.callback(cooler_meter.Close)
            pvs = Pv_Monitors(config['pvs'], rack.Simulated)
            opened.callback(pvs.Close)
            elog_cfg = config['elog']
            elog = Elog(elog_cfg['url'], elog_cfg['author'],
                        elog_cfg['enabled'] and not rack.Simulated)
            opened.callback(elog.Close)
            cooler_pv, server = config['cooler_pv'], None
            if cooler_pv['enabled'] and not rack.Simulated:
                try:
                    server = Cooler_Pv_Server(cooler_pv['prefix'], lambda: (
                        cooler_meter.Voltage * COOLER_DIVIDER))
                    opened.callback(server.Close)
                except Exception as e:
                    # the facility's copy of the cooler is not worth refusing a run for
                    log.exception("The cooler's PVs are not published")
                    warnings = [*warnings, f"The cooler's PVs are not published: {e}"]
            opened.pop_all()                    # the window owns them from here
    except Exception as e:
        log.exception("ORACLE could not start")
        splash.close()                          # it stays on top and would hide the error
        QMessageBox.critical(None, 'ORACLE could not start',
                             f"{e}\n\nThe session log has the details:\n{log_path}")
        return 1

    window = Main_Window(config, rack, link, line_meter, cooler_meter, pvs, elog, warnings)
    if server is not None:
        app.aboutToQuit.connect(server.Close)
    window.showMaximized()
    splash.finish(window)

    def raise_window():
        # a second launch knocked: bring this one forward instead
        while (knock := app._Oracle_Server.nextPendingConnection()) is not None:
            knock.deleteLater()
        window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized)
        window.raise_()
        window.activateWindow()

    app._Oracle_Server.newConnection.connect(raise_window)
    return app.exec()
