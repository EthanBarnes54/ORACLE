"""The Scan tab: scan and buncher parameters, run control, the ELOG panel and the live plots.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The old DAQ's layout, control for control: parameters, buncher controls,
scan controls and the ELOG panel down the left, and the HFS spectrum
over the TOF spectrum and the TOF-HFS map on the right. The tab reads
its own widgets into what the acquisition process needs (the plan, the
trigger timing, bunches per step) and draws from a Live_Data; starting
and stopping belong to the main window.

Depends on: pyqtgraph, PySide6; cls_acquisition.acquisition,
cls_acquisition.scan_plan, cls_acquisition.triggers, gui.plots,
gui.shared_widgets.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QRadioButton, QTextEdit, QVBoxLayout,
                               QWidget)

from cls_acquisition import triggers
from cls_acquisition.acquisition import SCAN_TYPES
from cls_acquisition.scan_plan import Scan_Plan
from gui import plots
from gui.shared_widgets import Spin

RANGES = 7
CHANNELS = (1, 2, 3, 4)
DC_TRIGGER_HZ = 1e9 / triggers.DC_PERIOD_NS


def _Row(*widgets):
    row = QHBoxLayout()
    for widget in widgets:
        row.addWidget(widget)
    return row


class Scan_Tab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.addWidget(self._Build_Parameters(), 0, 0)
        layout.addWidget(self._Build_Buncher(), 1, 0)
        layout.addWidget(self._Build_Control(), 2, 0)
        layout.addLayout(self._Build_Elog(), 3, 0)
        layout.addWidget(self._Build_Plots(), 0, 1, 4, 1)
        layout.setColumnStretch(1, 1)
        self.Bunched_Radio.toggled.connect(self._Mode_Changed)
        self._Mode_Changed()

    # ── Layout ───────────────────────────────────────────────────

    def _Build_Parameters(self):
        self.Params_Group = QGroupBox('Scan parameters')
        grid = QGridLayout(self.Params_Group)
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
            low, high = Spin(0.0, -4000.0, 4000.0), Spin(100.0, -4000.0, 4000.0)
            for column, widget in enumerate((check, low, high)):
                grid.addWidget(widget, 6 + i, column)
            self.Range_Checks.append(check)
            self.From_Boxes.append(low)
            self.To_Boxes.append(high)
        return self.Params_Group

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
        self.Gate_Start_Box = Spin(0.0, 0.0, 999999.0)
        self.Gate_Length_Box = Spin(200.0, 0.0, 999999.0)
        self._Timing_Boxes = (self.Dwell_Box, self.Release_Box, self.Tagger_Delay_Box,
                              self.Gate_Start_Box, self.Gate_Length_Box)
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
        self.Spectrum_Plot = plots.Plot('HFS spectrum', ('Scanning voltage', 'V'),
                                        ('Counts', None))
        self.Spectrum_Curve = pg.PlotCurveItem(pen=plots.Pen())
        self.Spectrum_Plot.addItem(self.Spectrum_Curve)
        self.Tof_Plot = plots.Gate_Plot()
        self.Map_Plot = plots.Plot('TOF-HFS', ('Scanning voltage', 'V'),
                                   ('Time of flight (µs)', None))
        self.Map_Image = pg.ImageItem(colorMap=pg.colormap.get('inferno'))
        self.Map_Plot.addItem(self.Map_Image)
        self.Bin_Box = Spin(1.0, 0.0005, 10000.0, 4, ' µs')
        self.Bin_Box.setKeyboardTracking(False)
        self.Map_Check = QCheckBox('TOF vs V plot')
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
        for box in self._Timing_Boxes:
            box.setEnabled(bunched)

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

    def Trigger_Hz(self):
        return 1000.0 / self.Dwell_Box.value() if self.Bunched() else DC_TRIGGER_HZ

    def Timing(self):
        start = self.Gate_Start_Box.value()
        return {'bunched': self.Bunched(), 'rep_rate_hz': self.Trigger_Hz(),
                'release_us': self.Release_Box.value(),
                'tagger_delay_us': self.Tagger_Delay_Box.value(),
                'gate_start_us': start, 'gate_stop_us': start + self.Gate_Length_Box.value()}

    def Tof_Edges(self):
        start = self.Gate_Start_Box.value()
        size = self.Bin_Box.value()
        count = int(round(self.Gate_Length_Box.value() / size)) + 1
        return start + size * (np.arange(count + 1) - 0.5)

    def Channels(self):
        return [c for c, check in zip(CHANNELS, self.Channel_Checks) if check.isChecked()]

    def Draw(self, live):
        self.Spectrum_Curve.setData(live.Volt_Edges, live.Spectrum, stepMode='center')
        self.Tof_Plot.Curve.setData(live.Tof_Edges, live.Tof, stepMode='center')
        if self.Map_Check.isChecked():
            volts, tof = live.Volt_Edges, live.Tof_Edges
            self.Map_Image.setImage(live.Tof_Map, autoLevels=True)
            self.Map_Image.setRect(QRectF(volts[0], tof[0], volts[-1] - volts[0],
                                          tof[-1] - tof[0]))
