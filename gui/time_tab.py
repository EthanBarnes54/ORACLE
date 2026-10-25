"""The Time graphs tab: counts against time, to watch the beam.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The gated hits of the chosen channels, binned by bunch number and drawn
in seconds at the trigger rate in force (the old DAQ binned DC data in
bunches of 1 ms while labelling the bins in seconds).

Depends on: pyqtgraph, PySide6; gui.plots.
"""

import pyqtgraph as pg
from PySide6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QWidget

from gui import plots

CHANNELS = (1, 2, 3, 4)
BIN_SECONDS = ('100', '10', '1', '0.1')


class Time_Tab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        self.Counts_Plot = plots.Plot('Time series', ('Time (s)', None), ('Counts per bin', None))
        self.Curve = pg.PlotCurveItem(pen=plots.Pen())
        self.Counts_Plot.addItem(self.Curve)
        grid.addWidget(self.Counts_Plot, 0, 0, 1, 8)
        self.Channel_Checks = [QCheckBox(f'Channel {c}', checked=c != 4) for c in CHANNELS]
        for column, check in enumerate(self.Channel_Checks):
            grid.addWidget(check, 1, column)
        grid.setColumnStretch(5, 1)
        self.Bin_Combo = QComboBox()
        self.Bin_Combo.addItems(BIN_SECONDS)
        grid.addWidget(self.Bin_Combo, 1, 6)
        grid.addWidget(QLabel('s'), 1, 7)
        grid.setRowStretch(0, 1)

    def Draw(self, live, trigger_hz, gate):
        channels = [c for c, check in zip(CHANNELS, self.Channel_Checks) if check.isChecked()]
        result = live.Counts_In_Time(float(self.Bin_Combo.currentText()), trigger_hz,
                                     channels, gate)
        if result is None:
            self.Curve.setData([], [])
        else:
            self.Curve.setData(*result, stepMode='center')
