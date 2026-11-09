"""The Time graphs tab: counts against time, to watch the beam.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The gated hits of the chosen channels, binned by bunch number and drawn
in seconds at the trigger rate in force (the old DAQ binned DC data in
bunches of 1 ms while labelling the bins in seconds). Its channels and
bin are part of a run's recorded view, held while the run is taken.

Under the plot, each PMT's afterpulsing as the data come in: its pairs
within the dead time on itself over those on the others, against
chance (1), an early warning of a tube going bad. Scan Performance
judges it properly once the run has ended.

Depends on: PySide6; gui.plots, gui.theme.
"""

from PySide6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QWidget

from gui import plots, theme

CHANNELS = (1, 2, 3, 4)
BIN_SECONDS = ('100', '10', '1', '0.1')


class Time_Tab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        self.Counts_Plot = plots.Plot('Counts per time bin', ('Acquisition time (s)', None),
                                      ('Counts per bin', None), (
            "Photon counts in successive time bins, to follow the beam intensity during a scan "
            "or while the triggers run; a fall to the background level indicates a loss of "
            "beam.",
            "Hits on the channels selected below whose time of flight lies inside the gate are "
            "counted in bins of the chosen width. Time is the bunch count divided by the trigger "
            "rate, so every bin holds the same number of bunches and the pauses while voltage "
            "steps settle are left out."))
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
        self.Afterpulse_Label = QLabel()
        self.Afterpulse_Label.setToolTip(
            "For each PMT, how many of its counts follow one of its own within the dead time "
            "(settings: detectors, dead_time_us), over how many follow one on the other PMTs, "
            "each scaled by the counts it could follow: 1 is chance, and afterpulsing raises "
            "it. Counted within each bunch as the data arrive; Scan Performance judges it "
            "against chance once the run has ended.")
        grid.addWidget(self.Afterpulse_Label, 2, 0, 1, 8)
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

    def Show_Afterpulsing(self, ratios, dead_time_us):
        """Each PMT's ratio, in the warning colour above 2."""
        cells = [f"ch {channel}: " + ('–' if ratio is None else
                                      f"<span style='color:{theme.VERDICT_COLOURS['WARN']}'>"
                                      f"{ratio:.1f}</span>" if ratio > 2.0 else f"{ratio:.1f}")
                 for channel, ratio in ratios.items()]
        self.Afterpulse_Label.setText(
            f"Same ÷ other PMT within {dead_time_us:g} µs, against chance:&nbsp;&nbsp;"
            + ('&nbsp;&nbsp;&nbsp;'.join(cells) if cells else 'no counts yet'))

    def Draw(self, live, trigger_hz, gate):
        """Counts against time from ``live``, counted again only if the data
        or what they are counted by have changed: each count reads every hit."""
        channels = self.Channels()
        bin_s = float(self.Bin_Combo.currentText())
        drawn = (live, live.Version, bin_s, tuple(channels), trigger_hz, gate)
        if drawn == self._Drawn:
            return
        self._Drawn = drawn
        self.Show_Afterpulsing(live.Afterpulse_Ratios(), live.Dead_Time_Us)
        result = live.Counts_In_Time(bin_s, trigger_hz, channels, gate)
        if result is None:
            self.Curve.setData([], [])
        else:
            self.Curve.setData(*result, stepMode='center')
