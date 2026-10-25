"""The Scan Performance tab: how the rack behaved during a run, and what to cut from it.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Down the left, the run (the latest by default, or any in the data folder
or chosen from disk), the verdict, every check grouped by the part of
the pipeline it judges, and the cuts the analysis recommends, which
export as DENIS's scan filter. On the right, one plot for each
systematic: the voltage against its calibration, the peak's position
scan by scan, the beam rate, how far reads trailed their bunches, the
cooler and the laser. A check's sentence shows under the list when it
is selected, and in its tooltip.

Depends on: numpy, pyqtgraph, PySide6; cls_acquisition.scan_performance,
gui.plots, gui.shared_widgets, gui.theme.
"""

import html
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QPushButton, QToolButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from cls_acquisition import scan_performance as sp
from gui import plots, theme
from gui.shared_widgets import Verdict_Html

PANEL_WIDTH = 400                   # px: the checks' names and values side by side


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
        self.Analyse_Button = QPushButton('Analyse')
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

        self.Cuts_Group = QGroupBox('Recommended cuts')
        column = QVBoxLayout(self.Cuts_Group)
        self.Cuts_Label = QLabel()
        self.Cuts_Label.setWordWrap(True)
        self.Export_Button = QPushButton(theme.Icon('file-down'), 'Export for DENIS',
                                         enabled=False)
        self.Export_Button.setToolTip("Write run_NNNN_performance.yaml beside the run: the "
                                      "report, and the cuts as DENIS's scan_filters.")
        column.addWidget(self.Cuts_Label)
        column.addWidget(self.Export_Button)

        left = QVBoxLayout()
        for group, stretch in ((self.Dataset_Group, 0), (self.Verdict_Group, 0),
                               (self.Checks_Group, 1), (self.Cuts_Group, 0)):
            left.addWidget(group, stretch)
        side = QWidget()
        side.setLayout(left)
        side.setFixedWidth(PANEL_WIDTH)
        left.setContentsMargins(0, 0, 0, 0)

        plot_group = QGroupBox('Systematics')
        plot_grid = QGridLayout(plot_group)
        time_axis = ('Time in run', 's')
        self.Readback_Plot = plots.Plot('Voltage against its calibration', time_axis,
                                        ('Readback - calibration', 'V'))
        self.Centroid_Plot = plots.Plot('Peak position by scan', ('Scan', None),
                                        ('Centroid', 'V'))
        self.Rate_Plot = plots.Plot('Beam rate by scan', ('Scan', None),
                                    ('Hits per bunch', None))
        self.Latency_Plot = plots.Plot('Reads behind their bunches', time_axis,
                                       ('Latency', 's'))
        self.Cooler_Plot = plots.Plot('Cooler', time_axis, ('Cooler - mean', 'V'))
        self.Laser_Plot = plots.Plot('Laser', time_axis, ('Wavemeter - mean (MHz)', None))
        self.Plots = (self.Readback_Plot, self.Centroid_Plot, self.Rate_Plot, self.Latency_Plot,
                      self.Cooler_Plot, self.Laser_Plot)
        for i, plot in enumerate(self.Plots):
            plot_grid.addWidget(plot, i // 2, i % 2)

        layout = QHBoxLayout(self)
        layout.addWidget(side)
        layout.addWidget(plot_group, 1)
        self.Result = None

    def Set_Runs(self, paths, keep=None):
        """The runs to choose from, newest first; ``keep`` stays selected."""
        current = keep or self.Run_Combo.currentData()
        self.Run_Combo.blockSignals(True)
        self.Run_Combo.clear()
        for path in paths:
            self.Run_Combo.addItem(os.path.basename(path)[:-len('.asdf')], path)
        index = self.Run_Combo.findData(current)
        self.Run_Combo.setCurrentIndex(max(index, 0))
        self.Run_Combo.blockSignals(False)

    def Selected(self):
        return self.Run_Combo.currentData()

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
        self.Summary_Label.setText(html.escape(result['Summary']))
        self._Fill_Checks(result['Sections'])
        cuts = [f"scan {scan}: {'; '.join(reasons)}" for scan, reasons in result['Cuts'].items()]
        if result['Leading_Bunches']:
            cuts.append(f"the first {result['Leading_Bunches']} bunches of every step")
        self.Cuts_Label.setText('<br>'.join(map(html.escape, cuts)) or 'None: keep every scan.')
        self._Draw(result['Series'])

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
        if 'Centroids' in series:
            scans, centroids, errors, trend = series['Centroids']
            self.Centroid_Plot.addItem(pg.ErrorBarItem(x=scans, y=centroids, height=2 * errors,
                                                       pen=theme.PLOT_FOREGROUND))
            self.Centroid_Plot.plot(scans, centroids, **dots)
            self.Centroid_Plot.plot(scans, trend, pen=plots.Pen(theme.TRACE_ALT))
        if 'Rates' in series:
            scans, rate, floor = series['Rates']
            self.Rate_Plot.addItem(pg.BarGraphItem(x=scans, height=rate, width=0.8,
                                                   brush=theme.TRACE))
            self.Rate_Plot.addItem(pg.InfiniteLine(floor, angle=0, pen=plots.Pen(
                theme.VERDICT_COLOURS[sp.FAIL])))
        if 'Latency' in series:
            self.Latency_Plot.plot(*series['Latency'], **dots)
        if 'Cooler' in series:
            self.Cooler_Plot.plot(*series['Cooler'], pen=pen)
        if 'Laser' in series:
            self.Laser_Plot.plot(*series['Laser'], pen=pen)
