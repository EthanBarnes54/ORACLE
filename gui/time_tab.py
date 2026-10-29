"""The Time graphs tab: counts against time, to watch the beam.

Date:    2026-10-29
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The gated hits of the chosen channels, binned by bunch number and drawn
in seconds at the trigger rate in force (the old DAQ binned DC data in
bunches of 1 ms while labelling the bins in seconds). Its channels and
bin are part of a run's recorded view, held while the run is taken.

Depends on: PySide6; gui.plots.
"""

from PySide6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QWidget

from gui import plots

CHANNELS = (1, 2, 3, 4)
BIN_SECONDS = ('100', '10', '1', '0.1')


class Time_Tab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        self.Counts_Plot = plots.Plot('Time series', ('Time (s)', None), ('Counts per bin', None))
        self.Curve = plots.Dense_Curve(pen=plots.Pen())
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
        self._Drawn = None

    def Channels(self):
        return [c for c, check in zip(CHANNELS, self.Channel_Checks) if check.isChecked()]

    def View(self):
        return {'time_bin_s': self.Bin_Combo.currentText(), 'time_channels': self.Channels()}

    def Show_View(self, view):
        """The channels and bin of a run's recorded ``view``."""
        if view.get('time_bin_s') in BIN_SECONDS:
            self.Bin_Combo.setCurrentText(view['time_bin_s'])
        for channel, check in zip(CHANNELS, self.Channel_Checks):
            check.setChecked(channel in view.get('time_channels', CHANNELS))

    def Lock_View(self, locked):
        for widget in (self.Bin_Combo, *self.Channel_Checks):
            widget.setEnabled(not locked)

    def Inputs(self):
        return [('time_bin', self.Bin_Combo),
                *((f"time_channel_{c}", check) for c, check in zip(CHANNELS,
                                                                   self.Channel_Checks))]

    def Draw(self, live, trigger_hz, gate):
        """Counts against time from ``live``, counted again only if the data
        or what they are counted by have changed: each count reads every hit."""
        channels = self.Channels()
        bin_s = float(self.Bin_Combo.currentText())
        drawn = (live, live.Version, bin_s, tuple(channels), trigger_hz, gate)
        if drawn == self._Drawn:
            return
        self._Drawn = drawn
        result = live.Counts_In_Time(bin_s, trigger_hz, channels, gate)
        if result is None:
            self.Curve.setData([], [])
        else:
            self.Curve.setData(*result, stepMode='center')
