"""ORACLE's main window and entry point.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The old DAQ's window: the Scan, Scan Calibration and Time graphs tabs,
with the bench tool's Hardware Calibration and the Scan Performance
analysis between them, above the quick-tune controls and the experiment
information. Finished runs are viewed and fitted in DENIS. The window
composes the pieces and owns the run: Start scan calibrates the scan's
own plan, puts the calibration through the gate, and only then writes
the run file's header and starts the scan; a REJECT writes nothing and
says why. Data are appended to the run file as they arrive, and the file
is closed when the acquisition process marks the scan's end. A hardware
calibration runs on the same worker slot as a calibration, so it locks
the same controls; its record is saved beside the run files, and a
linearity sweep's DAC correction goes into force only when the operator
says so, and into the settings with it. Every tab shows one run at a
time, named on the Shown line: the latest as ORACLE opens, any opened
(File ▸ Open run, or Scan Performance), or the one being taken, which a
scan or the triggers start by clearing them all for, and which stays on
them, saved and analysed, once it ends. A run is drawn with its own
trigger timing, and opened, it comes back as it stood when it had just
been taken, controls and all, from the record each run file keeps of
what the archive's header does not (run_file.ORACLE_KEY); the plots'
view is part of that record, and held from Start to the scan's end.
The scan queue (gui.scan_queue) is driven from here: each queued scan is
set up on the Scan tab and started as a scan is, with its number of
scans, the next once it has ended cleanly (Automatic) or once the
operator says so (Manual), and anything but a clean end stops the queue.
Analysis runs off the GUI thread. Every input in the window is kept
through a crash (gui.input_memory) and set again as ORACLE opens; those,
not the latest run's, are the controls ORACLE opens with. If the
acquisition process dies, the window clears the flags it left set,
closes a run it was taking on what it recorded, and says so.

``main()`` sets up the session log, one instance per user (two would
fight over the instruments), the theme and splash, and the rack: the
acquisition process, the meters and the facility services, all closed
again if any one of them fails to open.

Depends on: PySide6, pyqtgraph; cls_acquisition and the gui tabs.
"""

__version__ = '0.5.0'

import contextlib
import ctypes
import getpass
import json
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
from PySide6.QtCore import QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QFontDatabase,
                           QKeySequence, QPainter, QPixmap)
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame, QGridLayout, QGroupBox,
                               QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea,
                               QSlider, QSplashScreen, QStyle, QTabWidget, QWidget)

from cls_acquisition import (calibration, hardware_calibration, run_file, scan_performance,
                             settings, systematics)
from cls_acquisition.acquisition import Acquisition_Error, No_Trigger, Rack_Link
from cls_acquisition.facility import Cooler_Pv_Server, Elog, Pv_Monitors
from cls_acquisition.keep_awake import Keep_Awake
from cls_acquisition.hardware import Rack
from cls_acquisition.hardware import mcc_dac
from cls_acquisition.scan_plan import Scan_Plan, Scan_Plan_Error
from gui import plots, session_log, theme
from gui.calibration_tab import Calibration_Tab
from gui.hardware_tab import Hardware_Tab
from gui.input_memory import Input_Memory
from gui.scan_queue import CALIBRATING, Describe, Duration_S
from gui.live_data import Live_Data
from gui.performance_tab import Performance_Tab
from gui.scan_tab import Scan_Tab
from gui.shared_widgets import Duration_Text, Job_Worker, Spin, Verdict_Html
from gui.time_tab import Time_Tab

TICK_MS = 100
MONITOR_FLUSH_TICKS = 10            # the monitor record is written once a second
SLIDER_STEPS = 10000
STARTUP_TIMEOUT_S = 30.0            # the acquisition process opens its devices first
APP_ID = 'io.github.ethanbarnes54.oracle'   # as the macOS bundle's CFBundleIdentifier
NORMAL_SIZE = QSize(1250, 1040)     # px, restored, on a screen with room for it
SCREEN_SHARE = 0.9                  # of the work area at most, leaving room for the frame

log = logging.getLogger('oracle.gui')


def _Number(value, fmt):
    return 'unavailable' if value is None else format(value, fmt)


def _Volts(value):
    """A PV's value as a float, or None if it has none or is not a number: a
    PV that misbehaves is not worth refusing a run for."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class _Window_Grid(QGridLayout):
    """The window's grid, said to have no height-for-width (its wrapped
    Shown line has one): a scroll area holds a widget so laid out at its
    preferred height for the width, not its minimum, and the window would
    scroll on a screen with room for it all."""

    def hasHeightForWidth(self):
        return False


class Main_Window(QMainWindow):

    def __init__(self, config, rack, link, line_meter, cooler_meter, pvs, elog, warnings=()):
        super().__init__()
        self._Rack, self._Link = rack, link
        self._Line, self._Cooler = line_meter, cooler_meter
        self._Pvs, self._Elog = pvs, elog
        self._State = link.State
        self._Data_Dir = settings.Data_Directory(config)
        self._Live = Live_Data(float(config['detectors']['dead_time_us']))
        self._Volt_Edges = np.array([0.0, 1.0])
        self._Run = None                # path of the run file being written
        self._Pending = None            # the scan waiting on its calibration
        self._Worker = None             # the calibration in progress, scan or hardware
        self._Correction = config['dac_correction'] or mcc_dac.CORRECTION
        self._Scan_Divider, self._Cooler_Divider = rack.Scan_Divider, rack.Cooler_Divider
        self._Tagger_Levels = {'Inputs': list(config['instruments']['tagger_levels']),
                               'Trigger': config['instruments']['tagger_trigger_level']}
        # the rack and the code as they are for the whole session, for every record
        try:
            loop = link.Send('Identity')
        except Acquisition_Error as e:
            log.warning("The acquisition process did not say what its devices are: %s", e)
            loop = None
        self._Identity = rack.Identity(loop, line_meter, cooler_meter)
        self._Commit = settings.Commit()
        self._Settle_Model = config['settle_model']
        self._Analysis = None           # the performance analysis in progress
        self._Analysing = None          # the run it is of
        self._Wanted = None             # the run Scan Performance is to show
        self._Laser = config['laser']
        self._Rolling_S = float(config['cooler']['rolling_s'])
        self._Nplc = config['calibration']['nplc']
        self._Dead_Time_Us = float(config['detectors']['dead_time_us'])
        self._Declared = config['systematics']
        self._Systematics_Run = None    # the run whose short systematics block is due
        self._Channels = config['instruments']['tagger_channels']
        self._Monitor = []              # monitor rows of the run, not yet written
        self._Shown_Timing = None       # the shown run's trigger timing; None: the boxes'
        self._Shown_Round = None        # an opened run's last scan; None: the loop's
        self._Queueing = False          # the queue goes on to its next scan as this ends
        self._Queue_Row = None          # the queue's row of the scan being taken
        self._Queue_Held = ''           # why that scan ended short, if it did
        self._Queue_Asks = False        # Manual asks before every scan but the first
        self._Death_Shown = False
        self._Awake = Keep_Awake()      # no sleep while anything runs

        rack_name = 'simulated rack' if rack.Simulated else 'rack'
        self.setWindowTitle(f"ORACLE v{__version__} ({rack_name})")
        # it opens maximized; restored, it is centred in the screen's work area
        # and inside it, title bar and all, however small the screen
        area = self.screen().availableGeometry()
        self.setGeometry(QStyle.alignedRect(
            Qt.LayoutDirection.LeftToRight, Qt.AlignmentFlag.AlignCenter,
            NORMAL_SIZE.boundedTo(area.size() * SCREEN_SHARE), area))

        self.Tabs = QTabWidget(objectName='MainTabs', documentMode=True)
        self.Tabs.tabBar().setDrawBase(False)
        self.Tabs.setStyleSheet(theme.Main_Tabs_Qss())
        self._Limit_V = config['instruments']['hv_limit_v']
        self.Scan, self.Time = Scan_Tab(self._Limit_V), Time_Tab()
        self.Calibration = Calibration_Tab(self._Limit_V, self._Scan_Divider)
        self.Hardware = Hardware_Tab(self._Limit_V)
        self.Performance = Performance_Tab()
        for tab, icon, name in ((self.Scan, 'atom', 'Scan controls'),
                                (self.Calibration, 'crosshair', 'Scan Calibration'),
                                (self.Hardware, 'zap', 'Hardware Calibration'),
                                (self.Performance, 'stethoscope', 'Scan Performance'),
                                (self.Time, 'hourglass', 'Time graphs')):
            self.Tabs.addTab(tab, theme.Icon(icon), name)
        self.Scan.Experiment_Edit.setText(config['experiment'])

        central = QWidget()
        layout = _Window_Grid(central)
        layout.addWidget(self.Tabs, 0, 0)
        layout.addWidget(self._Build_Quick_Tune(), 1, 0)
        layout.addWidget(self._Build_Information(), 2, 0)
        layout.setRowStretch(0, 1)
        # Each tab's controls scroll, so the window fits a short screen; on one
        # too short even for the tabs' plots, the window scrolls too, rather
        # than run off the screen's bottom edge.
        page = QScrollArea(widgetResizable=True, frameShape=QFrame.Shape.NoFrame)
        page.setWidget(central)
        self.setCentralWidget(page)
        self._Build_Menus()
        # what was entered when ORACLE last stopped, however it stopped
        self._Inputs = Input_Memory(settings.Inputs_Path(), self)
        for group, inputs in (('scan', self.Scan.Inputs()), ('time', self.Time.Inputs()),
                              ('calibration', self.Calibration.Inputs()),
                              ('hardware', self.Hardware.Inputs()),
                              ('performance', self.Performance.Inputs()),
                              ('quick_tune', (('sweep_from', self.Sweep_From),
                                              ('sweep_to', self.Sweep_To)))):
            self._Inputs.Add(group, inputs)
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
        self.Scan.Queue.Start_Button.clicked.connect(self._Toggle_Queue)
        self.Scan.Queue.Set_Settle_Model(self._Settle_Model)
        self.Calibration.Start_Button.clicked.connect(self._Calibrate_Only)
        hardware = self.Hardware
        on_line = (link, line_meter, self._Scan_Divider)
        for panel, measure in (
                (hardware.Settle, partial(hardware_calibration.Settle_Test, *on_line)),
                (hardware.Linearity, partial(hardware_calibration.Linearity_Sweep, *on_line)),
                (hardware.Cooler, partial(hardware_calibration.Cooler_Monitor, cooler_meter,
                                          self._Cooler_Divider))):
            panel.Start_Button.clicked.connect(
                lambda _=False, panel=panel, measure=measure: self._Run_Bench(panel, measure))
            panel.Stop_Button.clicked.connect(self._Stop_Bench)
            panel.Open_Button.clicked.connect(
                lambda _=False, panel=panel: self._Open_Record(panel))
        hardware.Read_Button.clicked.connect(self._Read_Line)
        hardware.Linearity.Use_Button.clicked.connect(self._Use_Correction)
        hardware.Linearity.Clear_Button.clicked.connect(lambda: self._Set_Correction(None))
        hardware.Settle.Use_Button.clicked.connect(lambda: self._Set_Settle_Model(
            hardware_calibration.Settle_Model(hardware.Settle.Result[0])))
        hardware.Settle.Clear_Button.clicked.connect(lambda: self._Set_Settle_Model(None))
        hardware.Settle.Show_Model(self._Settle_Model)
        performance = self.Performance
        performance.Refresh_Button.clicked.connect(self._Refresh_Runs)
        performance.Browse_Button.clicked.connect(self._Open_Run)
        performance.Analyse_Button.clicked.connect(
            lambda: performance.Selected() and self._Open_Run(performance.Selected()))
        performance.Export_Button.clicked.connect(self._Export_Performance)
        performance.Elog_Button.clicked.connect(self._Post_Systematics)
        self._Refresh_Runs()
        if performance.Selected():
            # the latest run, on every tab but the controls, which keep what was
            # entered when ORACLE last stopped
            self.Show_Run(performance.Selected(), quiet=True, settings=False)

        self._Timer = QTimer(self, interval=TICK_MS, timeout=self._Tick)
        self._Show_State()              # the controls as the rack stands, from the start
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
        # the one run every tab shows, and whether it is live, saved or from a file
        self.Shown_Label = QLabel('nothing yet', wordWrap=True)
        grid.addWidget(QLabel('Shown:'), 2, 0)
        grid.addWidget(self.Shown_Label, 2, 1, 1, 5)
        return group

    def _Build_Menus(self):
        file_menu = self.menuBar().addMenu('&File')
        open_run = QAction(theme.Icon('file-down'), 'Open run...', self,
                           shortcut=QKeySequence.StandardKey.Open)
        open_run.triggered.connect(self._Open_Run)
        file_menu.addAction(open_run)
        for text, folder in (('Open data folder', self._Data_Dir),
                             ('Open log folder', os.path.join(settings.APP_ROOT, 'logs'))):
            action = QAction(theme.Icon('folder-open'), text, self)
            action.triggered.connect(lambda _=False, f=folder: self._Open_Folder(f))
            file_menu.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction('Quit', self.close)
        view_menu = self.menuBar().addMenu('&View')
        autoscale = QAction(theme.Icon('rotate-ccw'), 'Autoscale all plots', self,
                            shortcut=QKeySequence('Ctrl+0'))
        autoscale.triggered.connect(self._Autoscale_All)
        view_menu.addAction(autoscale)
        self.menuBar().addMenu('&Help').addAction('About ORACLE', self._About)

    def _Open_Run(self, path=None):
        """The run at ``path``, or one chosen from the data folder or anywhere,
        shown on every tab; only while nothing runs, since the live plots are
        the same plots."""
        state = self._State
        if state.Scanning.is_set() or state.Triggering.is_set() or self._Worker is not None:
            QMessageBox.warning(self, 'Run not opened', "A scan, the triggers or a calibration "
                                                        "is running, and the plots are theirs; "
                                                        "open the run once it has finished.")
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, 'Open a run', self._Data_Dir,
                                                  'Run files (run_*.asdf)')
        if path:
            self.Show_Run(path)

    def Show_Run(self, path, quiet=False, settings=True):
        """The run at ``path`` on every tab at once, as it stood when it had
        just been taken: its hits on the Scan tab and the Time graphs, drawn
        with its own tagger window and trigger rate, its calibration and the
        gate's verdict on the Scan Calibration tab, its run and scan numbers,
        and its analysis on the Scan Performance tab; with ``settings``, the
        controls too, as it was taken (scan, buncher and view). What ORACLE's
        record of the run gives is shown; a run older than the record keeps
        today's controls for what it lacks, and the Shown line says so. It
        stays until a scan, the triggers or another run take the tabs.
        Fitting it is DENIS's. ``quiet``: a run that cannot be read is said
        in the log and the Shown line alone, as at start-up."""
        name = os.path.basename(path)
        try:
            run = run_file.Read(path)
            header, raw, monitor = run
            plan = Scan_Plan(*zip(*header['ScanningRanges']), header['StepSize'])
        except Exception as e:
            log.exception("Opening %s failed", path)
            self.Shown_Label.setText(f"nothing: {name} could not be read")
            if not quiet:
                QMessageBox.warning(self, 'Run not opened', f"{name} could not be read as a "
                                                            f"run file: {e}")
            return
        record = header.get(run_file.ORACLE_KEY) or {}
        unshown = self._Show_Settings(header, plan) if settings else []
        self._Start_Live(plan, record.get('Timing'))
        self._Live.Add(raw)
        if monitor is not None and len(monitor.get('round', ())):
            self._Shown_Round = int(monitor['round'][-1])
        elif len(raw):
            # a run without the loop's record: the scan its last bunch was in
            per_scan = len(plan) * max(int(header['BunchesPerChannel']), 1)
            self._Shown_Round = int(raw[:, 2].max() // per_scan) + 1
        self.Run_Label.setText(str(header.get('Run')))
        self.Calibration.Show_Run(name, header.get('CalSet', []), header.get('CalReadback', []),
                                  record.get('Calibration'),
                                  scan_performance.Dividers(header)[0])
        self._Refresh_Runs(keep=path)
        self._Analyse(path, run)
        self.Tabs.setCurrentWidget(self.Scan)
        self.Shown_Label.setText(
            f"{name}, from its file: {len(raw)} hits, {len(plan)} setpoints of {plan.Step:g} V, "
            f"{header.get('BunchesPerChannel')} bunches each "
            f"({header.get('Experiment') or 'no experiment named'})"
            + (f". Not shown, so as set now: {'; '.join(unshown)}." if unshown else ""))
        for sentence in unshown:
            log.info("%s: not shown, %s", name, sentence)
        log.info("Showing %s", path)

    def _Show_Settings(self, header, plan):
        """The controls as the run in ``header`` was taken; returns what of it
        they could not show, as sentences."""
        record = header.get(run_file.ORACLE_KEY) or {}
        try:
            unshown = self.Scan.Show_Settings(header)
            if record.get('View'):
                self.Time.Show_View(record['View'])
        except (KeyError, TypeError, ValueError) as e:
            log.exception("The settings of run %s were not all shown", header.get('Run'))
            return [f"some of its settings, which its file holds in a form this ORACLE does "
                    f"not read ({e})"]
        self.Sweep_From.setValue(plan.Setpoints.min())
        self.Sweep_To.setValue(plan.Setpoints.max())
        return unshown

    def _Clear_Shown(self, shown, waiting):
        """Every tab given over to what starts now (``shown``): the scan plots
        are cleared by _Start_Live, and Scan Performance waits, saying why."""
        self._Wanted = None             # an analysis still running is not shown
        self.Performance.Clear(waiting)
        self.Shown_Label.setText(shown)

    def _Open_Folder(self, folder):
        """Show ``folder`` in Explorer or the Finder, made first if this
        ORACLE owns it; the rack's data folder is only ever looked for."""
        if self._Rack.Simulated or folder != self._Data_Dir:
            os.makedirs(folder, exist_ok=True)
        if not os.path.isdir(folder) or not QDesktopServices.openUrl(QUrl.fromLocalFile(folder)):
            QMessageBox.warning(self, 'Folder not opened', f"The folder {folder} could not be "
                                                           f"opened; is it reachable?")

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

    def _Start_Live(self, plan, timing=None):
        """The tabs cleared for a run of ``plan``, drawn with its ``timing``."""
        self._Live.Clear()
        self._Volt_Edges = plan.Bin_Edges() if plan is not None else np.array([0.0, 1.0])
        self._Shown_Timing, self._Shown_Round = timing, None

    def _Toggle_Scan(self):
        if self._State.Scanning.is_set():
            if self._Send('Stop_After_Scan') is not None:
                self._Hold_Queue(f"stopped by hand after scan {self._State.Round.value}")
                self._Elog.Append(f"\n\nSCAN STOPPED AFTER {self._State.Round.value} ROUNDS",
                                  self._Experiment())
            return
        if self._Worker is not None:
            return                      # a second click before the controls caught up
        refused = self._Start_Scan()
        if refused is not None:
            QMessageBox.warning(self, 'Scan not started', refused)

    def _Start_Scan(self, rounds=None, queued=None):
        """Calibrate the Scan tab's plan, then take it: ``rounds`` scans of
        it (until stopped if None, as its scan type says), ``queued`` its
        ``(place, length)`` in the queue. Returns why it could not start,
        or None once its calibration has."""
        try:
            plan = self.Scan.Plan()
            timing = self.Scan.Timing()
            run = self._Next_Run()
        except (ValueError, OSError) as e:
            return str(e)

        in_queue = f" (queued scan {queued[0]} of {queued[1]})" if queued else ''
        self._Start_Live(plan, timing)
        self._Clear_Shown(f"run_{run:04d}, being calibrated{in_queue}",
                          f"Run {run} is being taken: it is analysed as it ends.")
        self.Sweep_From.setValue(plan.Setpoints.min())
        self.Sweep_To.setValue(plan.Setpoints.max())
        when = datetime.now()
        pvs = {key: self._Pvs.Value(key) for key in ('wavenumber', 'mass', 'mass_setpoint')}
        pmt_hv = {channel: _Volts(self._Pvs.Value(f"pmt_hv_{channel}"))
                  for channel in (1, 2, 3, 4)}
        bunches = self.Scan.Bunches_Per_Step()
        text = (f"\n--------------------------------------\n{when:%d/%m/%Y %H:%M:%S}\n"
                f"START SCAN NUMBER {run}\nscanning ranges:\n"
                + ''.join(f"from {low:g} to {high:g} V\n" for low, high in plan.Ranges)
                + f"\nBunches per channel: {bunches}\n"
                f"Stepsize [V]:        {plan.Step:g}\n"
                f"Dwell time [ms]:     {self.Scan.Dwell_Box.value()}\n"
                f"Laser Set Point [cm-1]: {_Number(pvs['wavenumber'], '.5f')}\n"
                f"Mass [amu]:          {_Number(pvs['mass'], '.4f')}\n"
                f"Mass setpoint [A or amu]:          {_Number(pvs['mass_setpoint'], '.4f')}\n"
                + (f"Queued scan {queued[0]} of {queued[1]}, {rounds} scans\n" if queued
                   else "")
                + ("REFERENCE RUN, for the beam's offset from the cooler's readback\n"
                   if self.Scan.Reference_Check.isChecked() else ""))
        self._Pending = {'plan': plan, 'run': run, 'when': when, 'pvs': pvs, 'text': text,
                         'pmt_hv': {c: v for c, v in pmt_hv.items() if v is not None},
                         'setup': self.Scan.Setup(),
                         'bunches': bunches, 'timing': timing, 'rounds': rounds,
                         'queued': queued, 'scan_type': self.Scan.Type_Combo.currentText()}
        self._Calibrate(plan)
        return None

    def _Begin_Scan(self, pending, cal, report):
        """After the calibration: the gate's verdict, then the file and the scan."""
        applied, description = cal.Dac_Correction
        text = (pending['text'] + f"Calibration:  {calibration.Summary(report)}\n"
                f"DAC correction: {description}\n"
                + ''.join(f"  {c['Verdict']}  {c['Check']}: {c['Message']}\n"
                          for c in report['Checks'] if c['Verdict'] != calibration.PASS))
        if report['Verdict'] == calibration.REJECT:
            # the session log keeps why, with the ELOG off too, as no run file will
            log.warning("Run %d's calibration was refused by the gate; no file written, no "
                        "scan started:\n%s", pending['run'], calibration.Rejections(report))
            self._Elog.Append(text + "RUN REJECTED: no file written, no scan started.\n",
                              self._Experiment())
            self.Shown_Label.setText(f"run_{pending['run']:04d}'s calibration, refused by "
                                     f"the gate: nothing written")
            self._Queue_Scan_Ended('rejected', f"its calibration was refused by the gate: "
                                               f"{calibration.Rejections(report)}")
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
        # what the archive's keys do not hold, all locked while the scan runs, so
        # opening the run shows it as it was taken
        record = {'Version': __version__,
                  'Rack': 'simulated' if self._Rack.Simulated else 'hardware',
                  'ScanType': pending['scan_type'], 'Timing': pending['timing'],
                  'View': {**self.Scan.View(), **self.Time.View()}, 'Calibration': report,
                  'SettleModel': self._Settle_Model, 'Laser': dict(self._Laser),
                  'Channels': list(self._Channels), 'Rounds': pending['rounds'],
                  'Queue': list(pending['queued']) if pending['queued'] else None,
                  # which hardware and code took it, and what the DAC held exactly: the
                  # plan is the calibration's, so its codes are the scan's
                  'RackIdentity': self._Identity, 'Dividers': self._Rack.Dividers(),
                  'DacCodes': {'Channel': cal.Channels.tolist(), 'Code': cal.Codes.tolist()},
                  'DacCorrectionFull': cal.Correction, 'Commit': self._Commit,
                  'InteractionRegion': pending['setup']['InteractionRegion'],
                  'PmtHv': {'Pvs': pending['pmt_hv'] or None,
                            'Entered': pending['setup']['PmtHv']},
                  'TaggerLevels': self._Tagger_Levels,
                  'ReferenceRun': pending['setup']['ReferenceRun'],
                  # the window the analysis takes the cooler's rolling mean over
                  'CoolerRollingS': self._Rolling_S,
                  'CalibrationReadings': {'Nplc': cal.Nplc, 'DurationS': cal.Duration_S},
                  # the dead time the PMTs are judged at; never applied to the data
                  'DeadTimeUs': self._Dead_Time_Us}
        try:
            self._Run = run_file.Create(self._Data_Dir, header, record)
        except (OSError, ValueError) as e:
            self.Shown_Label.setText(f"nothing: run {pending['run']}'s file was not written")
            self._Queue_Scan_Ended('failed', f"its run file was not written: {e}")
            QMessageBox.critical(self, 'Run file not written', str(e))
            return
        started = self._Send('Start_Scan', setpoints=plan.Setpoints.tolist(),
                             scan_type=pending['scan_type'],
                             bunches_per_step=pending['bunches'], timing=pending['timing'],
                             settle=self._Settle_Model, rounds=pending['rounds'])
        if started is None:
            with contextlib.suppress(OSError):
                os.remove(self._Run)    # a header with no scan behind it is not a run
            self._Run = None
            self.Shown_Label.setText(f"nothing: run {pending['run']} did not start")
            self._Queue_Scan_Ended('failed', "the acquisition process refused to start it")
            return
        if self._Queue_Row is not None:
            self.Scan.Queue.Set_State(self._Queue_Row, 'running', f"run {pending['run']}")
        self.Run_Label.setText(str(pending['run']))
        self.Shown_Label.setText(f"{os.path.basename(self._Run)}, live: written to its file "
                                 f"as it comes in")
        log.info("Run %d started: %s", pending['run'], self._Run)
        # Stop is live the moment the scan is, not at the next tick: a click in between
        # would otherwise fall on a disabled button and the scan run on
        self._Show_State()

    def _End_Run(self):
        finished = self._Run
        if self._Queue_Row is not None:
            held, taken = self._Queue_Held, self.Scan.Queue.Why(self._Queue_Row)
            self._Queue_Scan_Ended('stopped' if held else 'done',
                                   f"{held}; {taken}" if held else taken)
        if finished is not None:
            self._Sample_Monitor()      # the loop's final counts, now it has stopped
            self._Flush_Monitor()
            log.info("Run file closed: %s", finished)
        self._Run = None
        self._Monitor = []
        if finished is not None:
            # kept on every tab, as taken, until a scan or another run takes them
            name = os.path.basename(finished)
            self.Shown_Label.setText(f"{name}, as taken: saved to its file")
            self._Refresh_Runs(keep=finished)
            if self.Performance.Follow_Check.isChecked():
                self._Systematics_Run = finished    # its short block goes to the ELOG
                self._Analyse(finished)
            else:
                self.Performance.Clear(f"{name} is saved; Show and analyse puts it here.")

    def _Stop_Now(self):
        if self._Send('Stop_Now') is not None:
            self._Hold_Queue("stopped by hand")
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
        self._Start_Live(plan, timing)
        self._Clear_Shown("the triggers' stream, live: not saved",
                          "The triggers are streaming; nothing of theirs is saved or analysed.")
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

    # ── The scan queue ───────────────────────────────────────────

    def _Toggle_Queue(self):
        """Start the queue at its first scan not done, or let the scan being
        taken be its last."""
        if self._Queueing:
            self._Queueing = False
            self.statusBar().showMessage("The queue stops once this scan ends.", 10000)
            self._Show_State()
            return
        queue = self.Scan.Queue
        try:
            pending = queue.Pending()
            if not pending and queue.Value() and QMessageBox.question(
                    self, 'Queue done', "Every scan in the queue is done. Take them all "
                                        "again?") == QMessageBox.StandardButton.Yes:
                queue.Reset_States()
                pending = queue.Pending()
            self.Scan.Timing()          # the timing every scan of it is taken with
        except ValueError as e:
            QMessageBox.warning(self, 'Queue not started', str(e))
            return
        if not pending:
            if not queue.Value():
                QMessageBox.warning(self, 'Queue not started', "The queue holds no scan: "
                                                               "Add current adds one.")
            return
        mode = queue.Mode_Combo.currentText()
        self._Queueing, self._Queue_Asks = True, False
        log.info("Scan queue started: %d scans, %s", len(pending), mode)
        self._Elog.Append(f"\n\nQUEUE STARTED: {len(pending)} scans, {mode}",
                          self._Experiment())
        self._Queue_Next()

    def _Queue_Next(self):
        """The queue's next scan; in Manual, asked about first, but for the
        first."""
        if not self._Queueing:
            return
        queue = self.Scan.Queue
        try:
            pending = queue.Pending()
        except ValueError as e:
            self._Queue_Over(str(e))
            return
        if not pending:
            self._Queue_Over("every scan in it is done")
            return
        row, entry = pending[0]
        place, bunched = queue.Position(row), self.Scan.Bunched()
        about = Duration_Text(Duration_S(entry, bunched, self.Scan.Trigger_Hz(),
                                         self._Settle_Model))
        if queue.Manual() and self._Queue_Asks and QMessageBox.question(
                self, 'Next queued scan',
                f"Scan {place[0]} of {place[1]} is next: {Describe(entry, bunched)}, about "
                f"{about}.\n\nReady to start it?") != QMessageBox.StandardButton.Yes:
            self._Queue_Over(f"held before scan {place[0]}")
            return
        self.Scan.Show_Entry(entry)
        self._Queue_Row, self._Queue_Held, self._Queue_Asks = row, '', True
        queue.Set_State(row, CALIBRATING)
        refused = self._Start_Scan(rounds=entry['scans'], queued=place)
        if refused is not None:
            self._Queue_Scan_Ended('failed', refused)
            QMessageBox.warning(self, 'Queued scan not started', refused)
        self._Show_State()

    def _Hold_Queue(self, why):
        """The queued scan being taken ends short, for ``why``: the queue stops
        with it."""
        if self._Queue_Row is not None:
            self._Queue_Held = why
            self._Queueing = False

    def _Queue_Scan_Ended(self, state, why):
        """What became of the queued scan, if one was being taken; the next
        once this tick has handed its rows to the run, unless it ended short."""
        row, self._Queue_Row = self._Queue_Row, None
        if row is None:
            return
        self.Scan.Queue.Set_State(row, state, why)
        log.info("Queued scan %d of %d: %s, %s", *self.Scan.Queue.Position(row), state, why)
        if state == 'done' and self._Queueing:
            QTimer.singleShot(0, self._Queue_Next)
        elif state == 'done':
            self._Queue_Over("stopped as asked, after its scan")
        else:
            self._Queue_Over(f"scan {self.Scan.Queue.Position(row)[0]} {state}: {why}")

    def _Queue_Over(self, why):
        self._Queueing = False
        log.info("Scan queue over: %s", why)
        self._Elog.Append(f"\n\nQUEUE OVER: {why}", self._Experiment())
        self.statusBar().showMessage(f"Scan queue over: {why}", 30000)
        self._Show_State()

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
                                              lambda *point: progress(point), self._Settle_Model,
                                              self._Scan_Divider, self._Nplc)
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
        self._Queue_Scan_Ended('failed', f"its calibration failed: {message}")
        pending, self._Pending = self._Pending, None
        if pending is not None:
            self._Elog.Append(pending['text'] + f"Calibration failed: {message}\n"
                              f"No file written, no scan started.\n", self._Experiment())
            self.Shown_Label.setText(f"run_{pending['run']:04d}'s calibration, which failed: "
                                     f"nothing written")
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
                  'hardware', 'Date': f"{datetime.now():%Y-%m-%d %H:%M:%S}", **params,
                  'RackIdentity': json.dumps(self._Identity),
                  'Dividers': json.dumps(self._Rack.Dividers()),
                  'Commit': json.dumps(self._Commit)}
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
            self._Finish_Records(records, 'stopped' if worker.Stopped() else 'completed',
                                 panel, result)
            panel.Show_Result(result, [record.Path for record in records.values()])

        def failed(message):
            # analysed as far as it went, as the bench tool's results_partial was
            status = 'stopped' if worker.Stopped() else 'failed'
            self._Finish_Records(records, f"{status}: {message}", panel, panel.Partial())
            if worker.Stopped():
                panel.Show_Stopped(message)
                return
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

    def _Finish_Records(self, records, status, panel, result):
        """Add ``result``'s closing rows to ``records`` and close them with its
        results; with no result, close them with the status alone."""
        if result is None:
            self._Close_Records(records, status)
            return
        for kind, columns in panel.Rows(result).items():
            try:
                records[kind].Add(*columns)
            except OSError:
                log.exception("Writing the %s record failed", kind)
        self._Close_Records(records, status, panel.Results(result))

    def _Close_Records(self, records, status, results=None):
        for kind, record in records.items():
            try:
                record.Close(status, (results or {}).get(kind))
            except OSError:
                log.exception("Closing the %s record failed", kind)
            log.info("Hardware calibration record (%s): %s", status, record.Path)

    def _Read_Line(self):
        """The line meter read precisely (the bench tool's 'Read DMM'), on the
        worker, against the setpoint on the platform as it starts."""
        if self._Worker is not None:
            return
        setpoint = self._State.Voltage.value

        def job(progress, stopped):
            return hardware_calibration.Precise_Reading(self._Line, self._Scan_Divider)

        self._Worker = Job_Worker(job, self)
        self._Worker.Done.connect(lambda reading: self.Hardware.Show_Reading(*reading, setpoint))
        self._Worker.Failed.connect(lambda message: (
            self.Hardware.Read_Label.setText(Verdict_Html('FAIL', message)),
            QMessageBox.warning(self, 'Line meter not read', message)))
        self._Worker.finished.connect(self._Worker_Finished)
        self.Hardware.Read_Label.setText('Reading...')
        self._Worker.start()
        self._Show_State()

    def _Stop_Bench(self):
        if self._Worker is not None:
            self._Worker.Stop()

    def _Open_Record(self, panel):
        """A record of ``panel``'s measurement, chosen from the data folder's
        hardware_calibration or anywhere, drawn and analysed on its page."""
        if self._Worker is not None:
            return
        kind = next(iter(panel.RECORDS))
        path, _ = QFileDialog.getOpenFileName(
            self, f"Open a {panel.Title.lower()} record",
            os.path.join(self._Data_Dir, 'hardware_calibration'),
            f"{panel.Title} records ({kind}_2*.csv)")
        if not path:
            return
        try:
            panel.Show_Record(path)
        except Exception as e:
            log.exception("Opening %s failed", path)
            panel.Show_Failure(f"{os.path.basename(path)} was not opened: {e}")
            QMessageBox.warning(self, 'Record not opened', f"{os.path.basename(path)} could "
                                                           f"not be shown: {e}")
            return
        log.info("Showing the %s record %s", kind, path)

    def _Set_Settle_Model(self, model):
        """Scans settle by ``model`` (None: SETTLE for every step) from the next one."""
        self._Settle_Model = model
        settings.Update(settle_model=model)
        self.Scan.Queue.Set_Settle_Model(model)
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
        self.Hardware.Linearity.Show_Correction(verdict, message, which)

    # ── Scan performance ─────────────────────────────────────────

    def _Refresh_Runs(self, keep=None):
        runs = run_file.Runs(self._Data_Dir) if os.path.isdir(self._Data_Dir) else []
        self.Performance.Set_Runs(runs, keep)

    def _Analyse(self, path, run=None):
        """Analyse ``path`` for Scan Performance, off the GUI thread, from
        ``run`` (run_file.Read's) if it has just been read, so a large run is
        not read twice. One asked for while another runs waits for it, and is
        read then; only the last asked for is shown, so the tab never shows
        another run than the rest."""
        self._Wanted = path
        self.Performance.Begin(path)
        if self._Analysis is not None:
            return                      # started as the one running finishes
        laser, settle_model, data_dir = self._Laser, self._Settle_Model, self._Data_Dir
        rolling_s, dead_time_us, declared = self._Rolling_S, self._Dead_Time_Us, self._Declared

        def job(progress, stopped):
            header, raw, monitor = run or run_file.Read(path)
            return scan_performance.Analyse(header, raw, monitor, laser, settle_model,
                                            run_file.Earlier_Calibrations(path, data_dir),
                                            rolling_s, dead_time_us, declared)

        self._Analysing = path
        self._Analysis = Job_Worker(job, self)
        self._Analysis.Done.connect(lambda result: self._Analysed(path, result))
        self._Analysis.Failed.connect(lambda message: self._Analysed(path, None, message))
        self._Analysis.finished.connect(self._Analysis_Finished)
        self._Analysis.start()

    def _Analysed(self, path, result, failure=None):
        """``path``'s analysis on the tab if it is still the one wanted, and,
        for a run that has just ended, its short systematics block to the
        ELOG."""
        if path == self._Wanted:
            if result is None:
                self.Performance.Show_Failure(failure)
            else:
                self.Performance.Show(path, result)
        if path == self._Systematics_Run:
            self._Systematics_Run = None
            name = os.path.basename(path)[:-len('.asdf')]
            self._Elog.Append(f"\n{systematics.Elog_Text(result['Systematics'], name)}"
                              if result is not None else
                              f"\nSYSTEMATIC INPUTS, {name}: not estimated, as its analysis "
                              f"failed: {failure}\n", self._Experiment())

    def _Post_Systematics(self):
        """The shown run's whole systematics table to the ELOG."""
        path, result = self.Performance.Result
        text = systematics.Elog_Text(result['Systematics'],
                                     os.path.basename(path)[:-len('.asdf')], full=True)
        self._Elog.Append(f"\n{text}", self._Experiment())
        self.statusBar().showMessage("Systematic inputs appended to the ELOG entry" + (
            "" if self._Elog.Enabled else " (the ELOG is off: kept in the session log)"), 10000)

    def _Analysis_Finished(self):
        self._Analysis.Retire()
        self._Analysis = None
        if self._Wanted is not None and self._Wanted != self._Analysing:
            self._Analyse(self._Wanted)

    def _Export_Performance(self):
        path, result = self.Performance.Result
        try:
            written = [scan_performance.Export(path, result),
                       systematics.Export(path, result['Systematics'])]
        except OSError as e:
            QMessageBox.warning(self, 'Not exported', f"The performance report could not be "
                                                     f"written: {e}")
            return
        self.statusBar().showMessage(f"Performance report and systematic inputs: "
                                     f"{', '.join(map(os.path.basename, written))}", 10000)

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
        scan, timing = self.Scan, self._Shown_Timing
        self._Live.Configure(self._Volt_Edges, scan.Tof_Edges(timing), scan.Tof_Plot.Gate(),
                             scan.Channels(), scan.Map_Check.isChecked())
        self._Live.Update()
        if self.Tabs.currentWidget() is scan:
            scan.Draw(self._Live)
        elif self.Tabs.currentWidget() is self.Time:
            self.Time.Draw(self._Live, scan.Trigger_Hz(timing), scan.Tof_Plot.Gate())
        self._Show_State()
        if self._Elog.Last_Error:
            self.statusBar().showMessage(self._Elog.Last_Error, 10000)
            self._Elog.Last_Error = None
        if not self._Death_Shown and not self._Link.Alive():
            self._Death_Shown = True
            self._Acquisition_Died()

    def _Acquisition_Died(self):
        """The acquisition process is gone. Its flags live on in shared memory,
        so they are cleared here, and a run it was taking is closed on the rows
        it sent, rather than left open with the window in its scanning state."""
        state = self._State
        for flag in (state.Scanning, state.Triggering, state.Outputting, state.Calibrating):
            flag.clear()
        self._Hold_Queue("the acquisition process stopped")
        if self._Run is not None:
            log.error("The acquisition process stopped during run %s", self._Run)
            self._End_Run()
        self._Show_State()
        QMessageBox.critical(self, 'Acquisition stopped',
                             "The acquisition process has stopped; the session log says why, "
                             "and a run it was taking has been closed on what it recorded. "
                             "Restart ORACLE.")

    def _Take_Data(self):
        scan_rows, taken = [], []
        for kind, rows in self._Link.Drain():
            if kind == 'scan_end':
                self._Write(scan_rows)
                scan_rows = []
                self._End_Run()
                continue
            taken.append(rows)
            if kind == 'scan':
                scan_rows.append(rows)
        self._Write(scan_rows)
        if taken:
            # kept in one go: a packet is one bunch, and each is coded as it is kept
            self._Live.Add(np.vstack(taken))

    def _Sample_Monitor(self):
        """One row of what the rack is doing, for the run's monitor record."""
        wavenumber = self._Pvs.Value('wavenumber')
        state = self._State
        self._Monitor.append((time.time(), state.Round.value, state.Voltage.value,
                              self._Line.Voltage * self._Scan_Divider,
                              self._Cooler.Voltage * self._Cooler_Divider,
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
            self._Hold_Queue(f"its run file could not be written: {e}")
            self._Send('Stop_Now')
            QMessageBox.critical(self, 'Scan stopped', f"The run file could not be written, "
                                                       f"so the scan was stopped: {e}")

    def _Show_State(self):
        state = self._State
        scanning, triggering = state.Scanning.is_set(), state.Triggering.is_set()
        outputting, calibrating = state.Outputting.is_set(), self._Worker is not None
        self._State.Cooler.value = self._Cooler.Voltage

        self.Cooler_Label.setText(f"{self._Cooler.Voltage * self._Cooler_Divider:.2f} V")
        self.Readback_Label.setText(f"{self._Line.Voltage * self._Scan_Divider:.1f} V")
        self.Wavenumber_Label.setText(_Number(self._Pvs.Value('wavenumber'), '.5f') + ' cm⁻¹')
        self.Round_Label.setText(str(state.Round.value if self._Shown_Round is None
                                     else self._Shown_Round))
        self.Mode_Label.setText('Bunched' if self.Scan.Bunched() else 'DC')
        if state.Silent.value and (scanning or triggering):
            # held on the bar for as long as the tagger stays quiet
            self.statusBar().showMessage(No_Trigger(state.Silent.value), 2 * TICK_MS)

        scan = self.Scan
        scan.Params_Group.setEnabled(not (scanning or triggering or calibrating))
        scan.Buncher_Group.setEnabled(not (scanning or triggering or calibrating))
        scan.Control_Group.setEnabled(not (calibrating or triggering))
        # the view a run's record holds, from Start until its last event
        locked = scanning or self._Pending is not None
        scan.Lock_View(locked)
        self.Time.Lock_View(locked)
        queue = scan.Queue
        queue.Lock(self._Queueing or self._Queue_Row is not None or calibrating or scanning
                   or triggering)
        queue.Start_Button.setText('Stop queue after this scan' if self._Queueing
                                   else 'Start queue')
        queue.Start_Button.setEnabled(self._Queueing or not (calibrating or scanning
                                                             or triggering))
        scan.Start_Button.setText('Stop after scan' if scanning else 'Start scan')
        scan.Stop_Button.setEnabled(scanning)
        busy = calibrating or scanning
        self._Awake.Hold(busy or triggering or outputting)
        hardware = self.Hardware
        for widget in (self.Calibration.Params_Group, self.Calibration.Control_Group,
                       self.Quick_Group, scan.New_Entry_Button, hardware.Ramp_Group,
                       hardware.Read_Group, hardware.Linearity.Correction_Group,
                       hardware.Settle.Model_Group,
                       *(panel.Params_Group for panel in hardware.Panels),
                       *(panel.Start_Button for panel in hardware.Panels),
                       *(panel.Open_Button for panel in hardware.Panels)):
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
        self._Inputs.Save()
        self._Awake.Hold(False)
        if self._Analysis is not None:
            self._Analysis.wait()       # a thread cannot outlive its window
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


def _Windows_App_Id():
    """Give the process ORACLE's own identity on Windows, so its taskbar
    button carries ORACLE's icon rather than pythonw.exe's and is pinned as
    ORACLE; elsewhere, nothing."""
    if sys.platform != 'win32':
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except (AttributeError, OSError):
        log.warning("Windows did not take ORACLE's taskbar identity")


def _Refuse_Start(message, log_path, splash=None):
    """Say why ORACLE cannot start, with the session log's place; returns 1."""
    if splash is not None:
        splash.close()                          # it stays on top and would hide the error
    QMessageBox.critical(None, 'ORACLE could not start',
                         f"{message}\n\nThe session log has the details:\n{log_path}")
    return 1


def main():
    log_path = session_log.Install(settings.APP_ROOT)
    session_log.Install_Qt_Handler()
    try:
        config, unreadable = settings.Load(), None
    except settings.Settings_Error as e:
        # said in a window below: a launch from the shortcut has no console
        config, unreadable = None, e
        log.error("%s", e)

    print('=' * 60)
    print(f"ORACLE v{__version__}")
    print('=' * 60)
    print(f"Python:     {sys.version.split()[0]}")
    print(f"Platform:   {platform.platform()}")
    print(f"App dir:    {settings.APP_ROOT}")
    print(f"Rack:       {config['rack'] if config else 'unknown: the settings are unreadable'}")
    print(f"Settings:   {settings.Settings_Path()}")
    if log_path:
        print(f"Log file:   {log_path}")
    print('-' * 60)

    _Windows_App_Id()
    app = QApplication(sys.argv)
    app.setApplicationName('ORACLE')
    if not _Single_Instance(app):
        return 0
    theme.Apply_Theme(app)
    app.setWindowIcon(theme.App_Icon())
    if unreadable is not None:
        return _Refuse_Start(unreadable, log_path)
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
                        cooler_meter.Voltage * rack.Cooler_Divider))
                    opened.callback(server.Close)
                except Exception as e:
                    # the facility's copy of the cooler is not worth refusing a run for
                    log.exception("The cooler's PVs are not published")
                    warnings = [*warnings, f"The cooler's PVs are not published: {e}"]
            opened.pop_all()                    # the window owns them from here
    except Exception as e:
        log.exception("ORACLE could not start")
        return _Refuse_Start(e, log_path, splash)

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
