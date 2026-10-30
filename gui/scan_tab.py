"""The Scan tab: scan and buncher parameters, run control, the ELOG panel and the live plots.

Date:    2026-10-30
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The old DAQ's layout, control for control: parameters, buncher controls,
scan controls and the ELOG panel down the left, and the HFS spectrum
over the TOF spectrum and the TOF-HFS map on the right. The tab reads
its own widgets into what the acquisition process needs (the plan, the
trigger timing, bunches per step) and draws from a Live_Data; starting
and stopping belong to the main window. Its voltages are bounded by the
platform's limit, and its gate by the tagger's timestamp range and the
trigger's cycle. DC hits come through the tagger too, so its gate stays
settable in DC mode; only the bunched timing is greyed out there. A TOF
bin finer than MAX_TOF_BINS across the gate is widened to it, as the old
DAQ's was, so a small bin cannot ask the window for gigabytes.

The parameters box has a second page, the scan queue (gui.scan_queue):
scans laid out ahead, each set up here as it starts.

A run is drawn with its own tagger window and trigger rate, which a run
file's record keeps, not with the boxes as they stand. Its view (the TOF
bin, the gate, the channels, the map) is recorded as it starts and held
for the whole scan. ``Show_Settings`` sets every control as a run file
says it was taken; ``Inputs`` names every input, for the window's memory
of them.

Depends on: pyqtgraph, PySide6; cls_acquisition.acquisition,
cls_acquisition.hardware.timetagger, cls_acquisition.run_file,
cls_acquisition.scan_plan, cls_acquisition.triggers, gui.plots,
gui.scan_queue, gui.shared_widgets, gui.theme.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QRadioButton, QSpinBox, QTextEdit,
                               QVBoxLayout, QWidget)

from cls_acquisition import triggers
from cls_acquisition.acquisition import SCAN_TYPES
from cls_acquisition.hardware.timetagger import MAX_WINDOW_US, Check_Window
from cls_acquisition.run_file import ORACLE_KEY
from cls_acquisition.scan_plan import Scan_Plan
from gui import plots, theme
from gui.scan_queue import ORDERS, Scan_Queue
from gui.shared_widgets import Scroll_Column, Spin

RANGES = 7
CHANNELS = (1, 2, 3, 4)
DC_TRIGGER_HZ = 1e9 / triggers.DC_PERIOD_NS
MAX_TOF_BINS = 50000                # across the gate; the old DAQ's cap


def _Row(*widgets):
    row = QHBoxLayout()
    for widget in widgets:
        row.addWidget(widget)
    return row


def _Put(box, value, what, unshown):
    """``value`` into ``box``, and a sentence into ``unshown`` if the box
    cannot hold it (the limits of this window, not the run's)."""
    if not box.minimum() <= value <= box.maximum():
        unshown.append(f"{what} of {value:g}, beyond {box.minimum():g} to {box.maximum():g} "
                       f"here")
    box.setValue(int(round(value)) if isinstance(box, QSpinBox) else float(value))


class Scan_Tab(QWidget):

    def __init__(self, limit_v, parent=None):
        super().__init__(parent)
        self._Limit_V = limit_v
        column = QVBoxLayout()
        column.addWidget(self._Build_Parameters())
        column.addWidget(self._Build_Buncher())
        column.addWidget(self._Build_Control())
        column.addLayout(self._Build_Elog())
        layout = QHBoxLayout(self)
        layout.addWidget(Scroll_Column(column))
        layout.addWidget(self._Build_Plots(), 1)
        self._Drawn = None
        self.Bunched_Radio.toggled.connect(self._Mode_Changed)
        self.Dwell_Box.valueChanged.connect(self._Timing_Changed)
        self.Queue.Add_Button.clicked.connect(lambda: self.Queue.Add_Entry(self.Entry_Now()))
        self.Queue.Show_Requested.connect(lambda entry: self.Show_Entry(entry, show=True))
        self._Mode_Changed()

    # ── Layout ───────────────────────────────────────────────────

    def _Build_Parameters(self):
        """The scan parameters and the scan queue, as pages of one box titled
        with both, as DENIS's optimiser tools are."""
        self.Params_Group = QWidget()
        self.Queue = Scan_Queue(self._Limit_V)
        self.Parameter_Pages = theme.Title_Tabs()
        self.Parameter_Pages.addTab(self.Params_Group, 'Scan parameters')
        self.Parameter_Pages.addTab(self.Queue, 'Scan queue')
        grid = QGridLayout(self.Params_Group)
        grid.setContentsMargins(0, 0, 0, 0)
        self.Type_Combo = QComboBox()
        self.Type_Combo.addItems(SCAN_TYPES)
        self.Per_Step_Label = QLabel()
        self.Per_Step_Box = Spin(1.0, 0.01, 9999999.0)
        self.Step_Box = Spin(1.0, 0.01, 10000.0)
        self.Experiment_Edit = QLineEdit()
        for row, (text, widget) in enumerate((('Scan type', self.Type_Combo),
                                              (None, self.Per_Step_Box),
                                              ('Stepsize (V)', self.Step_Box),
                                              ('Experiment name', self.Experiment_Edit))):
            grid.addWidget(QLabel(text) if text else self.Per_Step_Label, row, 1)
            grid.addWidget(widget, row, 2)
        grid.addWidget(QLabel('From (V)'), 5, 1)
        grid.addWidget(QLabel('To (V)'), 5, 2)
        self.Range_Checks, self.From_Boxes, self.To_Boxes = [], [], []
        for i in range(RANGES):
            check = QCheckBox(checked=i == 0)
            low = Spin(0.0, -self._Limit_V, self._Limit_V)
            high = Spin(100.0, -self._Limit_V, self._Limit_V)
            for column, widget in enumerate((check, low, high)):
                grid.addWidget(widget, 6 + i, column)
            self.Range_Checks.append(check)
            self.From_Boxes.append(low)
            self.To_Boxes.append(high)
        return self.Parameter_Pages

    def _Build_Buncher(self):
        self.Buncher_Group = QGroupBox('Buncher controls')
        column = QVBoxLayout(self.Buncher_Group)
        radios = QHBoxLayout()
        self.Dc_Radio = QRadioButton('DC mode')
        self.Bunched_Radio = QRadioButton('Bunched mode', checked=True)
        radios.addWidget(self.Dc_Radio)
        radios.addWidget(self.Bunched_Radio)
        radios.addStretch(1)
        column.addLayout(radios)

        self.Dwell_Box = Spin(100, 1, 100000, 0)
        self.Release_Box = Spin(100, 0, 1000000, 0)
        self.Beamgate_Box = Spin(0, 0, 1000000, 0)
        self.Beamgate_Box.setEnabled(False)
        self.Beamgate_Box.setToolTip("Not wired: the beamgate output is not in the trigger "
                                     "pattern, as in the old DAQ.")
        self.Tagger_Delay_Box = Spin(0, 0, 1000000, 0)
        self.Gate_Start_Box = Spin(0.0, 0.0, MAX_WINDOW_US)
        self.Gate_Length_Box = Spin(200.0, 0.0, MAX_WINDOW_US)
        self._Bunched_Boxes = (self.Dwell_Box, self.Release_Box, self.Tagger_Delay_Box)
        grid = QGridLayout()
        for row, (text, box, unit) in enumerate((
                ('Dwell time', self.Dwell_Box, 'ms'),
                ('Bunch release length', self.Release_Box, 'µs'),
                ('Beamgate closes', self.Beamgate_Box, 'µs before release'),
                ('DAQ TOF trigger delay', self.Tagger_Delay_Box, 'µs after release'),
                ('DAQ gate starts', self.Gate_Start_Box, 'µs after trigger'),
                ('DAQ gate stops', self.Gate_Length_Box, 'µs after start'))):
            grid.addWidget(QLabel(text), row, 0)
            grid.addWidget(box, row, 1)
            grid.addWidget(QLabel(unit), row, 2)
        column.addLayout(grid)
        return self.Buncher_Group

    def _Build_Control(self):
        self.Control_Group = QGroupBox('Scan controls')
        row = QHBoxLayout(self.Control_Group)
        self.Start_Button = QPushButton('Start scan')
        self.Stop_Button = QPushButton('Stop now!', enabled=False)
        row.addWidget(self.Start_Button)
        row.addWidget(self.Stop_Button)
        return self.Control_Group

    def _Build_Elog(self):
        column = QVBoxLayout()
        buttons = QHBoxLayout()
        self.Send_Text_Button = QPushButton('Send additional text')
        self.New_Entry_Button = QPushButton('Start new entry')
        buttons.addWidget(self.Send_Text_Button)
        buttons.addWidget(self.New_Entry_Button)
        column.addLayout(buttons)
        self.Screen_Button = QPushButton('Send HFS screen')
        column.addWidget(self.Screen_Button)
        self.Elog_Edit = QTextEdit()
        self.Elog_Edit.setPlaceholderText("Text for the ELOG entry")
        column.addWidget(self.Elog_Edit)
        return column

    def _Build_Plots(self):
        group = QGroupBox('Plots')
        grid = QGridLayout(group)
        self.Spectrum_Plot = plots.Plot('Hyperfine-structure spectrum', ('Scanning voltage', 'V'),
                                        ('Counts per setpoint', None), (
            "Photon counts against the scanning voltage, accumulated over every scan of the "
            "run being taken, or of the run opened for display.",
            "Hits on the selected detector channels whose time of flight lies inside the gate "
            "(every hit, if no gate is set) are histogrammed in one bin per voltage setpoint. "
            "The voltage is the setpoint commanded on the DAC, before the calibration and the "
            "per-hit cooler correction that DENIS applies; counts are not normalised to the "
            "number of scans."))
        self.Spectrum_Curve = plots.Dense_Curve(pen=plots.Pen())
        self.Spectrum_Plot.addItem(self.Spectrum_Curve)
        self.Tof_Plot = plots.Gate_Plot()
        self.Map_Plot = plots.Plot('Time of flight against scanning voltage',
                                   ('Scanning voltage', 'V'), ('Time of flight (µs)', None), (
            "A two-dimensional histogram of hits by scanning voltage and time of flight; the "
            "colour gives the counts in each cell, on an automatically scaled inferno scale. "
            "Resonant fluorescence appears as a localised spot at the line's voltage and the "
            "ion bunch's arrival time.",
            "Hits on the selected channels inside the time-of-flight gate, binned by setpoint "
            "and by time of flight; to bound the memory it takes, cells are merged to at most "
            "2000 along each axis. It is computed only while the map is switched on below the "
            "plot."))
        self.Map_Image = pg.ImageItem(colorMap=pg.colormap.get('inferno'))
        self.Map_Plot.addItem(self.Map_Image)
        self.Bin_Box = Spin(1.0, 0.0005, 10000.0, 4, ' µs')
        self.Bin_Box.setKeyboardTracking(False)
        self.Map_Check = QCheckBox('Show time-of-flight map')
        self.Channel_Checks = [QCheckBox(f'Channel {c}', checked=True) for c in CHANNELS]
        # each plot's own control sits in a strip under it, on one row, so the two
        # plots stay the same size; the channels filter every plot and run under both
        grid.addWidget(self.Spectrum_Plot, 0, 0, 1, 2)
        grid.addWidget(self.Tof_Plot, 1, 0)
        grid.addWidget(self.Map_Plot, 1, 1)
        right = Qt.AlignmentFlag.AlignRight
        grid.addLayout(_Row(QLabel('Bin size:'), self.Bin_Box), 2, 0, right)
        grid.addWidget(self.Map_Check, 2, 1, right)
        grid.addLayout(_Row(*self.Channel_Checks), 3, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        return group

    # ── What the widgets ask for ────────────────────────────────

    def _Mode_Changed(self):
        bunched = self.Bunched_Radio.isChecked()
        self.Per_Step_Label.setText('Bunches per channel' if bunched else 'Seconds per channel')
        self.Per_Step_Box.setValue(1.0 if bunched else 0.1)
        for box in self._Bunched_Boxes:
            box.setEnabled(bunched)
        self._Timing_Changed()

    def _Timing_Changed(self):
        """The queue counts its bunches and its time in the mode and at the
        trigger rate set here."""
        self.Queue.Set_Timing(self.Bunched(), self.Trigger_Hz())

    def Entry_Now(self):
        """The scan the parameters page holds, as a queued scan's entry: one
        scan of it, or of its order for a sawtooth or triangle."""
        ticked = [i for i, check in enumerate(self.Range_Checks) if check.isChecked()]
        kind = self.Type_Combo.currentText()
        return {'ranges': [[self.From_Boxes[i].value(), self.To_Boxes[i].value()]
                           for i in ticked],
                'step': self.Step_Box.value(), 'per_step': self.Per_Step_Box.value(),
                'order': kind if kind in ORDERS else ORDERS[0], 'scans': 1}

    def Show_Entry(self, entry, show=False):
        """A queued scan's ``entry`` on the parameters page, as a scan is set
        up to start; with ``show``, the page brought forward."""
        for i, (check, low, high) in enumerate(zip(self.Range_Checks, self.From_Boxes,
                                                   self.To_Boxes)):
            check.setChecked(i < len(entry['ranges']))
            if i < len(entry['ranges']):
                low.setValue(entry['ranges'][i][0])
                high.setValue(entry['ranges'][i][1])
        self.Step_Box.setValue(entry['step'])
        self.Per_Step_Box.setValue(entry['per_step'])
        self.Type_Combo.setCurrentText('Single scan' if entry['scans'] == 1 else entry['order'])
        if show:
            self.Parameter_Pages.setCurrentWidget(self.Params_Group)

    def Plan(self):
        """The Scan_Plan of the ticked ranges; raises Scan_Plan_Error."""
        ticked = [i for i, check in enumerate(self.Range_Checks) if check.isChecked()]
        return Scan_Plan([self.From_Boxes[i].value() for i in ticked],
                         [self.To_Boxes[i].value() for i in ticked], self.Step_Box.value())

    def Bunched(self):
        return self.Bunched_Radio.isChecked()

    def Bunches_Per_Step(self):
        """Bunches per setpoint: as entered when bunched, or seconds at the DC
        trigger rate."""
        if self.Bunched():
            return max(int(self.Per_Step_Box.value()), 1)
        return max(int(round(self.Per_Step_Box.value() * DC_TRIGGER_HZ)), 1)

    def Trigger_Hz(self, timing=None):
        """Triggers a second: ``timing``'s (a run's) if given, else as set."""
        if timing is not None:
            return timing['rep_rate_hz']
        return 1000.0 / self.Dwell_Box.value() if self.Bunched() else DC_TRIGGER_HZ

    def Timing(self):
        """The trigger timing; raises ValueError for a gate past the tagger's
        timestamp range, or any edge or window that does not fit the cycle."""
        start = self.Gate_Start_Box.value()
        Check_Window(start, start + self.Gate_Length_Box.value())
        timing = {'bunched': self.Bunched(), 'rep_rate_hz': self.Trigger_Hz(),
                  'release_us': self.Release_Box.value(),
                  'tagger_delay_us': self.Tagger_Delay_Box.value(),
                  'gate_start_us': start, 'gate_stop_us': start + self.Gate_Length_Box.value()}
        triggers.Pattern(timing)
        return timing

    def Tof_Edges(self, timing=None):
        """The TOF histogram's edges across the tagger's window, ``timing``'s
        (a run's) if given, else as set: a bin apart, or wider if that would
        make more than MAX_TOF_BINS."""
        if timing is None:
            start, length = self.Gate_Start_Box.value(), self.Gate_Length_Box.value()
        else:
            start = timing['gate_start_us']
            length = timing['gate_stop_us'] - start
        size = max(self.Bin_Box.value(), length / MAX_TOF_BINS)
        count = int(round(length / size)) + 1
        return start + size * (np.arange(count + 1) - 0.5)

    def Channels(self):
        return [c for c, check in zip(CHANNELS, self.Channel_Checks) if check.isChecked()]

    def View(self):
        """How the plots show a run, for its record."""
        gate = self.Tof_Plot.Gate()
        return {'tof_bin_us': self.Bin_Box.value(), 'tof_gate_us': gate and list(gate),
                'channels': self.Channels(), 'map': self.Map_Check.isChecked()}

    def Lock_View(self, locked):
        """The view held as it is while a scan runs, so its record holds for
        every event; the plots still pan and zoom."""
        for widget in (self.Bin_Box, self.Map_Check, *self.Channel_Checks):
            widget.setEnabled(not locked)
        self.Tof_Plot.Lock_Gate(locked)

    def Show_Settings(self, header):
        """Every control as the run in ``header`` (run_file.Read's) was taken:
        its ranges, step, bunches, dwell and experiment from the header, and
        its scan type, mode, timing and view from ORACLE's record. Returns
        what of the run they could not show, as sentences."""
        record = header.get(ORACLE_KEY) or {}
        timing, view, unshown = record.get('Timing'), record.get('View'), []
        if record.get('ScanType') in SCAN_TYPES:
            self.Type_Combo.setCurrentText(record['ScanType'])
        if timing is not None:
            # the mode first: changing it resets the bunches per step
            (self.Bunched_Radio if timing['bunched'] else self.Dc_Radio).setChecked(True)
        ranges = [list(r) for r in header.get('ScanningRanges', [])]
        for i, (check, low, high) in enumerate(zip(self.Range_Checks, self.From_Boxes,
                                                   self.To_Boxes)):
            check.setChecked(i < len(ranges))
            if i < len(ranges):
                _Put(low, ranges[i][0], 'a range limit', unshown)
                _Put(high, ranges[i][1], 'a range limit', unshown)
        if len(ranges) > RANGES:
            unshown.append(f"{len(ranges) - RANGES} of its {len(ranges)} ranges (the tab "
                           f"holds {RANGES})")
        _Put(self.Step_Box, float(header['StepSize']), 'its step', unshown)
        bunches = int(header['BunchesPerChannel'])
        _Put(self.Per_Step_Box, bunches if self.Bunched() else bunches / DC_TRIGGER_HZ,
             'its bunches per step', unshown)
        _Put(self.Dwell_Box, int(header['DwellTime']), 'its dwell', unshown)
        self.Experiment_Edit.setText(str(header.get('Experiment') or ''))
        if timing is None:
            unshown.append("its scan type, mode, trigger timing and view, which ORACLE did not "
                           "yet record")
            return unshown
        start = timing['gate_start_us']
        for box, value, what in ((self.Release_Box, timing['release_us'], 'its release'),
                                 (self.Tagger_Delay_Box, timing['tagger_delay_us'],
                                  'its trigger delay'),
                                 (self.Gate_Start_Box, start, 'its gate start'),
                                 (self.Gate_Length_Box, timing['gate_stop_us'] - start,
                                  'its gate length')):
            _Put(box, value, what, unshown)
        if view is not None:
            _Put(self.Bin_Box, view['tof_bin_us'], 'its TOF bin', unshown)
            self.Tof_Plot.Set_Gate(view['tof_gate_us'])
            for channel, check in zip(CHANNELS, self.Channel_Checks):
                check.setChecked(channel in view['channels'])
            self.Map_Check.setChecked(view['map'])
        return unshown

    def Inputs(self):
        """Every input, by name, in the order to set them again: the mode
        before the bunches per step it resets."""
        ranges = [(f"range_{i}_{part}", widget) for i, row in enumerate(zip(
            self.Range_Checks, self.From_Boxes, self.To_Boxes))
            for part, widget in zip(('on', 'from', 'to'), row)]
        return [('scan_type', self.Type_Combo), ('mode', (self.Dc_Radio, self.Bunched_Radio)),
                ('per_step', self.Per_Step_Box), ('step', self.Step_Box), *ranges,
                ('dwell', self.Dwell_Box), ('release', self.Release_Box),
                ('tagger_delay', self.Tagger_Delay_Box), ('gate_start', self.Gate_Start_Box),
                ('gate_length', self.Gate_Length_Box), ('tof_bin', self.Bin_Box),
                ('tof_gate', self.Tof_Plot), ('map', self.Map_Check),
                *((f"channel_{c}", check) for c, check in zip(CHANNELS, self.Channel_Checks)),
                ('elog_text', self.Elog_Edit), ('queue_mode', self.Queue.Mode_Combo),
                ('queue', self.Queue)]

    def Draw(self, live):
        """The plots from ``live``, redrawn only if it or the map's box has
        changed since they last were: a 50,000-bin TOF and a full map cost
        tens of milliseconds a tick to draw again unchanged."""
        drawn = (live, live.Version, self.Map_Check.isChecked())
        if drawn == self._Drawn:
            return
        self._Drawn = drawn
        self.Spectrum_Curve.setData(live.Volt_Edges, live.Spectrum, stepMode='center')
        self.Tof_Plot.Curve.setData(live.Tof_Edges, live.Tof, stepMode='center')
        if self.Map_Check.isChecked():
            volts, tof = live.Volt_Edges, live.Tof_Edges
            self.Map_Image.setImage(live.Tof_Map, autoLevels=True)
            self.Map_Image.setRect(QRectF(volts[0], tof[0], volts[-1] - volts[0],
                                          tof[-1] - tof[0]))
