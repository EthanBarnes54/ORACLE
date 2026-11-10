"""The Scan Performance tab: how the rack behaved during a run, and how that shaped its spectrum.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

On the left, the run, its verdict and line width, every check by part
(its sentence in the tooltip), what was flagged and the systematics'
inputs, postable to the ELOG; nothing is cut, and the report exports
beside the run. On the right, a plot per systematic: HV against
calibration, beam, read delays, cooler, laser and each PMT's afterpulsing.

Depends on: numpy, pyqtgraph, PySide6; cls_acquisition.scan_performance,
cls_acquisition.systematics, gui.plots, gui.shared_widgets, gui.theme.
"""

import html
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QLabel,
                               QPushButton, QToolButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from cls_acquisition import scan_performance as sp
from cls_acquisition import systematics
from gui import plots, theme
from gui.shared_widgets import (Muted, Report_Html, Report_Label, Report_Table, Side_By_Side,
                                Verdict_Html)

PANEL_WIDTH = 400                   # px: the checks' names and values side by side
CHANNEL_COLOURS = {1: theme.TRACE, 2: theme.TRACE_ALT, 3: '#FF5CF0', 4: '#FF9F40'}
LEGEND_BRUSH = (13, 13, 13, 220)    # the plots' background, so a legend hides what is behind


def _Item(parent, name, verdict, value=''):
    """A row of the checks tree, its verdict in its colour."""
    item = QTreeWidgetItem(parent, (name, verdict, value))
    item.setForeground(1, QBrush(QColor(theme.VERDICT_COLOURS[verdict])))
    return item


def _Value(value):
    if value is None:
        return ''
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{value:.4g}" if abs(value) > 1e-12 else '0'


class Performance_Tab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.Dataset_Group = QGroupBox('Dataset')
        grid = QGridLayout(self.Dataset_Group)
        self.Run_Combo = QComboBox()
        self.Refresh_Button = QToolButton(icon=theme.Icon('refresh-cw'))
        self.Refresh_Button.setToolTip("Look for new runs in the data folder.")
        self.Browse_Button = QToolButton(icon=theme.Icon('folder-open'))
        self.Browse_Button.setToolTip("Analyse a run file from anywhere.")
        self.Follow_Check = QCheckBox('Analyse each run as it ends', checked=True)
        self.Analyse_Button = QPushButton('Show and analyse')
        self.Analyse_Button.setToolTip("Show the chosen run on every tab, as it was taken, and "
                                       "analyse it here.")
        self.Info_Label = QLabel('No run analysed yet in this session.')
        self.Info_Label.setWordWrap(True)
        grid.addWidget(self.Run_Combo, 0, 0)
        grid.addWidget(self.Refresh_Button, 0, 1)
        grid.addWidget(self.Browse_Button, 0, 2)
        grid.addWidget(self.Follow_Check, 1, 0, 1, 3)
        grid.addWidget(self.Analyse_Button, 2, 0, 1, 3)
        grid.addWidget(self.Info_Label, 3, 0, 1, 3)
        grid.setColumnStretch(0, 1)

        self.Verdict_Group = QGroupBox('Verdict')
        column = QVBoxLayout(self.Verdict_Group)
        self.Verdict_Label = QLabel('—')
        self.Verdict_Label.setStyleSheet('font-size: 20px; font-weight: bold;')
        self.Summary_Label = QLabel()
        self.Summary_Label.setWordWrap(True)
        column.addWidget(self.Verdict_Label)
        column.addWidget(self.Summary_Label)

        self.Checks_Group = QGroupBox('Checks')
        column = QVBoxLayout(self.Checks_Group)
        self.Checks_Tree = QTreeWidget(columnCount=3)
        self.Checks_Tree.setHeaderLabels(('Check', 'Verdict', 'Value'))
        self.Checks_Tree.setRootIsDecorated(True)
        self.Detail_Label = QLabel('Select a check to read what it found.')
        self.Detail_Label.setWordWrap(True)
        column.addWidget(self.Checks_Tree, 1)
        column.addWidget(self.Detail_Label)
        self.Checks_Tree.currentItemChanged.connect(self._Show_Detail)

        self.Flagged_Group = QGroupBox('Flagged for review')
        column = QVBoxLayout(self.Flagged_Group)
        self.Flagged_Label = QLabel()
        self.Flagged_Label.setWordWrap(True)
        self.Export_Button = QPushButton(theme.Icon('file-down'), 'Export report', enabled=False)
        self.Export_Button.setToolTip("Write run_NNNN_performance.yaml beside the run: the "
                                      "verdict, every check and what was flagged. Nothing is "
                                      "cut; that is decided in DENIS.")
        column.addWidget(self.Flagged_Label)
        column.addWidget(self.Export_Button)

        self.Systematics_Group = QGroupBox('Systematic inputs')
        column = QVBoxLayout(self.Systematics_Group)
        self.Systematics_Label = Report_Label('')
        self.Elog_Button = QPushButton(theme.Icon('send'), 'Post systematics to ELOG',
                                       enabled=False)
        self.Elog_Button.setToolTip("Append the whole table, every source with its detail, to "
                                    "the open ELOG entry. A short block is posted as each "
                                    "run is analysed; a simulated rack never posts.")
        column.addWidget(self.Systematics_Label)
        column.addWidget(self.Elog_Button)


        plot_group = QGroupBox('Systematics')
        plot_grid = QGridLayout(plot_group)
        elapsed = ('Elapsed time <i>t</i>', 's')
        from_first = " Elapsed time runs from the run's first recorded hit."
        self.Readback_Plot = plots.Plot('Scan-voltage readback relative to calibration',
                                        elapsed, (f"{plots.U_RB} − {plots.Sym('U', 'cal')}",
                                                  'V'), (
            "How far the platform voltage was, through the run, from where the start-of-run "
            "calibration places it. Zero means no change since the calibration; a trend is "
            "drift.",
            "For each monitor reading taken once its setpoint had settled (sampled every "
            "0.1 s), the line meter's readback less the calibration's own reading at that "
            "setpoint. DENIS applies the calibration as it was taken, so a non-zero value "
            "shifts the line by the same voltage." + from_first))
        self.Beam_Plot = plots.Plot('Beam intensity over time', elapsed,
                                    ('Relative intensity <i>I</i>/⟨<i>I</i>⟩', None), (
            "The beam intensity relative to the run's usual level, period by period: 1 is the "
            "usual level and the lower dashed line half of it. Periods marked in red fell below "
            "half beyond what counting statistics explain, and are flagged for review.",
            "The hits in each period divided by the hits expected there, the run's average "
            "count rate at each setpoint the period visited, summed, so the spectrum's own "
            "structure cancels. A period lasts long enough to expect about 20 hits, and at "
            "least 10 s." + from_first))
        self.Rate_Plot = plots.Plot('Count rate per scan', ('Scan number <i>k</i>', None),
                                    ('Counts per bunch <i>N</i>/<i>n</i><sub>b</sub>', None), (
            "The count rate of each complete scan. Every scan visits every setpoint equally, so "
            "the scans compare directly; the dashed line is their median.",
            "The hits in the scan divided by the bunches it took (setpoints × bunches per "
            "step). Not shown when the run's scan numbers are unreliable."))
        self.Latency_Plot = plots.Plot('Read delay within each step', elapsed,
                                       ('Read delay δ<i>t</i>', 's'), (
            "How far each bunch's read trailed the earliest read in its voltage step. A flat "
            "band near zero means the acquisition kept pace with the triggers.",
            "The read time less the bunch number × the trigger period, less the smallest such "
            "value in the step. The run file's timestamps are rounded to 10 ms, which sets the "
            "resolution." + from_first))
        self.Cooler_Plot = plots.Plot('Cooler platform voltage', elapsed,
                                      (f"{plots.U_COOL} − med({plots.U_COOL})", 'V'), (
            "The cooler platform voltage during the run: each reading (faint) and the rolling "
            "mean the analysis takes as the voltage the ions were accelerated by (bold). The "
            "mean follows the supply's slow drift, which the analysis takes out; the readings' "
            "scatter about it is the mains ripple and the meter's noise, which it cannot.",
            "Readings from the run's monitor record, every 0.1 s, or for a run without one, "
            "the reading stamped on each bunch's hits; less their median. The mean is a boxcar "
            "over the readings within the window centred on each (15 s unless the run "
            "recorded another), never across a gap in them longer than the window; dashed "
            "where the window reaches past the readings at the run's start, end or a gap, and "
            "so holds one side only." + from_first))
        self.Laser_Plot = plots.Plot('Laser frequency deviation', elapsed,
                                     ('<i>ν</i><sub>WM</sub> − <i>ν</i><sub>set</sub> (MHz)',
                                      None), (
            "The wavemeter reading during the run relative to the laser setpoint stored in the "
            "run header, which DENIS uses for the whole run. Through the Doppler shift, a "
            "deviation moves the line.",
            "The wavemeter's wavenumber less the header's setpoint, converted to MHz of the "
            "measured light and sampled every 0.1 s. The Laser check converts it to scan volts "
            "and to MHz at the ions, with the harmonic and geometry recorded with the run."
            + from_first))
        self.Afterpulse_Plot = plots.Plot('Time to the previous count, relative to chance',
                                          ('Time since the previous count Δ<i>t</i>', 'µs'),
                                          ('<i>N</i>(Δ<i>t</i>)/<i>N</i><sub>chance</sub>'
                                           '(Δ<i>t</i>)', None), (
            "For each PMT (by colour), how often a count follows the previous count on the same "
            "PMT (solid) and on any other PMT (dashed) after each time, relative to chance: 1 is "
            "chance. Afterpulsing raises the solid curve at short times alone; light and the "
            "spread of the number of ions per bunch raise both.",
            "Within each bunch (or DC window), the time from each count back to the previous "
            "count, histogrammed in 0.5 µs bins up to 50 µs, divided by the same histogram "
            "with the previous count taken from the next four bunches at the same time in "
            "their bunch, which keeps the bunch's time-of-flight profile."))
        self.Afterpulse_Plot.setLogMode(y=True)
        self.Afterpulse_Plot.addLegend(offset=(-10, 10), labelTextColor=theme.PLOT_FOREGROUND,
                                          brush=LEGEND_BRUSH)
        self.Dispersion_Plot = plots.Plot('Index of dispersion versus dead time',
                                          ('Dead time <i>τ</i>', 'µs'),
                                          ('Index of dispersion <i>σ</i>²/<i>μ</i>', None), (
            "For each PMT (by colour), how far its counts scatter from scan to scan beyond "
            "Poisson's, with no dead time and with each candidate one: 1 is Poisson. "
            "Afterpulsing raises it, and a dead time that removes the afterpulses brings it "
            "down; in bunched mode a dead time also removes real photons at the line's peak.",
            "Counts per setpoint per complete scan, a count left out if the same PMT counted "
            "within the dead time before it in its bunch; for each setpoint the variance "
            "across scans over the mean, pooled over setpoints. Shown only when the run's "
            "scan numbers are reliable and it holds three complete scans."))
        self.Dispersion_Plot.addLegend(offset=(-10, 10), labelTextColor=theme.PLOT_FOREGROUND,
                                          brush=LEGEND_BRUSH)
        self.Plots = (self.Readback_Plot, self.Beam_Plot, self.Rate_Plot, self.Latency_Plot,
                      self.Cooler_Plot, self.Laser_Plot, self.Afterpulse_Plot,
                      self.Dispersion_Plot)
        for i, plot in enumerate(self.Plots):
            plot_grid.addWidget(plot, i // 2, i % 2)

        Side_By_Side(self, ((self.Dataset_Group, 0), (self.Verdict_Group, 0),
                            (self.Checks_Group, 1), (self.Flagged_Group, 0),
                            (self.Systematics_Group, 0)), plot_group).setFixedWidth(PANEL_WIDTH)
        self.Result = None

    def Set_Runs(self, paths, keep=None):
        """The runs to choose from, newest first; ``keep`` stays selected, at
        the top if it was opened from outside the data folder."""
        current = keep or self.Run_Combo.currentData()
        if current and current not in paths:
            paths = [current, *paths]
        self.Run_Combo.blockSignals(True)
        self.Run_Combo.clear()
        for path in paths:
            self.Run_Combo.addItem(os.path.basename(path)[:-len('.asdf')], path)
        index = self.Run_Combo.findData(current)
        self.Run_Combo.setCurrentIndex(max(index, 0))
        self.Run_Combo.blockSignals(False)

    def Selected(self):
        return self.Run_Combo.currentData()

    def Inputs(self):
        return [('follow', self.Follow_Check)]

    def Clear(self, message):
        """Nothing shown, and ``message`` saying why: a scan or the triggers
        have the tabs."""
        self.Result = None
        self.Export_Button.setEnabled(False)
        self.Elog_Button.setEnabled(False)
        self.Systematics_Label.setText('')
        self.Systematics_Label.setToolTip('')
        self.Verdict_Label.setText('—')
        self.Summary_Label.setText('')
        self.Checks_Tree.clear()
        self.Detail_Label.setText('Select a check to read what it found.')
        self.Flagged_Label.setText('')
        for plot in self.Plots:
            plot.clear()
        self.Info_Label.setText(html.escape(message))

    def Begin(self, path):
        self.Info_Label.setText(f"Analysing {html.escape(os.path.basename(path))}...")
        self.Analyse_Button.setEnabled(False)

    def Show_Failure(self, message):
        self.Analyse_Button.setEnabled(True)
        self.Info_Label.setText(Verdict_Html(sp.FAIL, f"Not analysed: {message}"))

    def Show(self, path, result):
        self.Result = (path, result)
        self.Analyse_Button.setEnabled(True)
        self.Export_Button.setEnabled(True)
        self.Info_Label.setText(html.escape(path))
        colour = theme.VERDICT_COLOURS[result['Verdict']]
        self.Verdict_Label.setText(f"<span style='color:{colour}'>{result['Verdict']}</span>")
        self.Summary_Label.setText(f"{html.escape(result['Summary'])}<br><br>"
                                   f"{html.escape(result['Budget'])}")
        self._Fill_Checks(result['Sections'])
        self.Flagged_Label.setText('<br><br>'.join(map(html.escape, result['Flagged']))
                                   or 'Nothing flagged for review.')
        self._Show_Systematics(result.get('Systematics', []))
        self._Draw(result['Series'])

    def _Show_Systematics(self, rows):
        """The inputs as a table, each source's detail in the tooltip."""
        size = lambda value, fmt: '–' if value is None else format(value, fmt)
        self.Systematics_Label.setText(Report_Html([('Each source at the line', [
            Report_Table(('Source', 'Kind', 'V', 'MHz'),
                         [(r['Source'], r['Kind'], size(r['Volts'], '.3f'),
                           size(r['Mhz'], '.1f')) for r in rows]),
            Muted("Widths in scan volts and in MHz at the ions; measured on this run, "
                  "declared in the settings, or not yet known. Hover for each source's "
                  "detail.")])]) if rows else '')
        self.Systematics_Label.setToolTip(
            html.escape(systematics.Elog_Text(rows, '')).replace('\n', '<br>') if rows else '')
        self.Elog_Button.setEnabled(bool(rows))

    def _Fill_Checks(self, sections):
        self.Checks_Tree.clear()
        for name, checks in sections.items():
            worst = max((c['Verdict'] for c in checks), key=sp.RANK.get)
            parent = _Item(self.Checks_Tree, name, worst)
            for check in checks:
                item = _Item(parent, check['Check'], check['Verdict'], _Value(check['Value']))
                item.setToolTip(0, check['Message'])
                item.setData(0, Qt.ItemDataRole.UserRole, check['Message'])
            parent.setExpanded(worst != sp.PASS)
        self.Checks_Tree.resizeColumnToContents(0)

    def _Show_Detail(self, item, _previous=None):
        message = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        self.Detail_Label.setText(message or 'Select a check to read what it found.')

    def _Draw(self, series):
        for plot in self.Plots:
            plot.clear()
        pen, dots = plots.Pen(), plots.Dots()
        if 'Readback' in series:
            self.Readback_Plot.plot(*series['Readback'], **dots)
        if 'Beam' in series:
            # the beam against what each period's setpoints give: 1 is usual, and a
            # period below the dashed half, beyond counting, is marked a loss
            t, level, lost = series['Beam']
            dashed = pg.mkPen(theme.PLOT_FOREGROUND, width=1, style=Qt.PenStyle.DashLine)
            for at in (1.0, sp.TOLERANCES['BeamLossFraction']):
                self.Beam_Plot.addItem(pg.InfiniteLine(at, angle=0, pen=dashed))
            self.Beam_Plot.plot(t, level, pen=pen, connect='finite')
            if lost.any():
                self.Beam_Plot.plot(t[lost], level[lost],
                                    **plots.Dots(theme.VERDICT_COLOURS[sp.FAIL]))
        if 'Rates' in series:
            scans, rate, median = series['Rates']
            self.Rate_Plot.addItem(pg.BarGraphItem(x=scans, height=rate, width=0.8,
                                                   brush=theme.TRACE))
            self.Rate_Plot.addItem(pg.InfiniteLine(median, angle=0, pen=pg.mkPen(
                theme.PLOT_FOREGROUND, width=1, style=Qt.PenStyle.DashLine)))
        if 'Latency' in series:
            self.Latency_Plot.plot(*series['Latency'], **dots)
        if 'Cooler' in series:
            t, readings, mean, partial = series['Cooler']
            self.Cooler_Plot.addItem(plots.Dense_Curve(t, readings, pen=plots.Faint_Pen(),
                                                       connect='finite'))
            # the edges dashed, each overlapping the full stretch by a reading so they join
            edge = partial | np.r_[partial[1:], False] | np.r_[False, partial[:-1]]
            for shown, style in ((~partial, Qt.PenStyle.SolidLine), (edge, Qt.PenStyle.DashLine)):
                self.Cooler_Plot.addItem(plots.Dense_Curve(
                    t, np.where(shown, mean, np.nan), connect='finite',
                    pen=pg.mkPen(theme.TRACE, width=2, style=style)))
        if 'Laser' in series:
            self.Laser_Plot.addItem(plots.Dense_Curve(*series['Laser'], pen=pen,
                                                      connect='finite'))
        dashed = pg.mkPen(theme.PLOT_FOREGROUND, width=1, style=Qt.PenStyle.DashLine)
        for channel, (edges, same, other) in series.get('Afterpulse', {}).items():
            colour = CHANNEL_COLOURS.get(channel, theme.PLOT_FOREGROUND)
            for ratio, style, name in ((same, Qt.PenStyle.SolidLine, f"channel {channel}"),
                                       (other, Qt.PenStyle.DashLine, None)):
                shown = np.where(ratio > 0, ratio, np.nan)
                self.Afterpulse_Plot.plot((edges[:-1] + edges[1:]) / 2.0, shown, connect='finite',
                                          pen=pg.mkPen(colour, width=2, style=style), name=name)
        if 'Afterpulse' in series:
            self.Afterpulse_Plot.addItem(pg.InfiniteLine(0.0, angle=0, pen=dashed))
        if 'Dispersion' in series:
            dead_times, ratios = series['Dispersion']
            self.Dispersion_Plot.addItem(pg.InfiniteLine(1.0, angle=0, pen=dashed))
            for channel, ratio in ratios.items():
                colour = CHANNEL_COLOURS.get(channel, theme.PLOT_FOREGROUND)
                self.Dispersion_Plot.plot(dead_times, ratio, pen=pg.mkPen(colour, width=2),
                                          name=f"channel {channel}",
                                          **{k: v for k, v in plots.Dots(colour).items()
                                             if k != 'pen'})
