"""The Run viewer tab: finished runs side by side, on any axis, with a hyperfine model over them.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The collinear data viewer, brought into ORACLE. Down the left, the runs
(opened from the data folder or anywhere, each ticked to show it), the
plot's axis, the channels, and the hyperfine estimator; on the right
the runs' spectra, over their time-of-flight and time-in-run histograms
(each carries a gate, set as on the Scan tab, that every spectrum then
counts within), and on a second page their calibrations. Counts stay on
each run's own scan bins whatever the axis (cls_acquisition.spectra), so
runs taken with different calibrations line up on the calibrated axes.

The estimator draws a transition's hyperfine structure from its nuclear
spin and its levels' A and B factors (cls_acquisition.hfs), the A or B
ratio held fixed if asked, carried onto the axis shown through the
first ticked run's calibration and Doppler shift. It is a guide for the
eye; fitting is DENIS's.

Runs are read on a worker, and the plots redrawn a moment after the last
change, so neither a large file nor a burst of edits stalls the window.

Depends on: numpy, pyqtgraph, PySide6; cls_acquisition.hfs,
cls_acquisition.run_file, cls_acquisition.spectra, gui.plots,
gui.shared_widgets, gui.theme.
"""

import html
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QTabWidget, QToolButton, QVBoxLayout, QWidget)

from cls_acquisition import hfs, run_file, spectra
from gui import plots, theme
from gui.shared_widgets import Form, Job_Worker, Spin

CHANNELS = (1, 2, 3, 4, 5)          # 5: the old DAQ's DC counter
RUN_COLOURS = (theme.TRACE, theme.TRACE_ALT, '#FF5CF0', '#7CFF3F', '#FF8C00', '#9D7CFF')
PANEL_WIDTH = 380                   # px: the estimator's boxes side by side
REDRAW_MS = 150                     # after the last change, before the plots are redrawn
MODEL_POINTS = 3000
MODEL_REACH_FWHM = 8.0              # past the outermost lines the model is drawn


def _Edges(values, width):
    """Bin edges ``width`` apart across ``values``, or None for none."""
    if not len(values):
        return None
    count = max(int(np.ceil(np.ptp(values) / width)), 1)
    return values.min() + width * np.arange(count + 1)


class Viewer_Tab(QWidget):

    def __init__(self, data_dir, laser, parent=None):
        super().__init__(parent)
        self._Data_Dir, self._Laser = data_dir, laser
        self._Runs = {}                 # path: spectra.Run, in the order opened
        self._Loader = None
        self._Redraw = QTimer(self, singleShot=True, interval=REDRAW_MS, timeout=self.Draw)

        side = QWidget()
        left = QVBoxLayout(side)
        left.setContentsMargins(0, 0, 0, 0)
        side.setFixedWidth(PANEL_WIDTH)
        left.addWidget(self._Build_Runs())
        left.addWidget(self._Build_Axis())
        left.addWidget(self._Build_Estimator(), 1)

        self.Pages = QTabWidget()
        spectra_page = QWidget()
        grid = QGridLayout(spectra_page)
        self.Spectrum_Plot = plots.Plot('Spectra', spectra.AXIS_LABELS[spectra.AXES[0]],
                                        ('Counts', None))
        self.Spectrum_Plot.addLegend(offset=(-40, 10))
        self.Tof_Plot = plots.Gate_Plot()
        self.Time_Plot = plots.Gate_Plot(title='Hits in time', x_axis=('Time in run', 's'),
                                         what='time')
        self.Tof_Bin_Box = Spin(0.5, 0.0005, 10000.0, 4, ' µs')
        self.Time_Bin_Box = Spin(1.0, 0.01, 10000.0, 2, ' s')
        grid.addWidget(self.Spectrum_Plot, 0, 0, 1, 2)
        grid.addWidget(self.Tof_Plot, 1, 0)
        grid.addWidget(self.Time_Plot, 1, 1)
        right = Qt.AlignmentFlag.AlignRight
        for column, box in enumerate((self.Tof_Bin_Box, self.Time_Bin_Box)):
            row = QHBoxLayout()
            row.addWidget(QLabel('Bin size:'))
            row.addWidget(box)
            grid.addLayout(row, 2, column, right)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        self.Pages.addTab(spectra_page, 'Spectra')

        calibration_page = QWidget()
        grid = QGridLayout(calibration_page)
        set_axis = ('Set voltage', 'V')
        self.Readback_Plot = plots.Plot('Calibration', set_axis, ('Readback voltage', 'V'))
        self.Difference_Plot = plots.Plot('Readback - set', set_axis, ('Difference', 'V'))
        self.Step_Plot = plots.Plot('Measured step size', set_axis, ('Step', 'V'))
        grid.addWidget(self.Readback_Plot, 0, 0, 1, 2)
        grid.addWidget(self.Difference_Plot, 1, 0)
        grid.addWidget(self.Step_Plot, 1, 1)
        self.Pages.addTab(calibration_page, 'Calibrations')

        layout = QHBoxLayout(self)
        layout.addWidget(side)
        layout.addWidget(self.Pages, 1)
        for signal in (self.Axis_Combo.currentIndexChanged, self.Normalise_Check.toggled,
                       self.Reference_Box.valueChanged, self.Tof_Bin_Box.valueChanged,
                       self.Time_Bin_Box.valueChanged, self.Tof_Plot.Gate_Changed,
                       self.Time_Plot.Gate_Changed, self.Run_List.itemChanged,
                       self.Hfs_Group.toggled,
                       *(check.toggled for check in self.Channel_Checks)):
            signal.connect(self._Changed)
        self.Axis_Combo.currentIndexChanged.connect(self._Axis_Changed)
        self._Axis_Changed()

    # ── Layout ───────────────────────────────────────────────────

    def _Build_Runs(self):
        group = QGroupBox('Runs')
        column = QVBoxLayout(group)
        self.Run_List = QListWidget()
        self.Run_List.setMinimumHeight(110)
        column.addWidget(self.Run_List)
        buttons = QHBoxLayout()
        self.Open_Button = QPushButton(theme.Icon('folder-open'), 'Open...')
        self.Open_Button.setToolTip("Read runs from the data folder or anywhere.")
        self.Latest_Button = QPushButton('Latest run')
        self.Remove_Button = QToolButton(icon=theme.Icon('minus'))
        self.Remove_Button.setToolTip("Put away the selected run.")
        for button in (self.Open_Button, self.Latest_Button, self.Remove_Button):
            buttons.addWidget(button)
        column.addLayout(buttons)
        self.Status_Label = QLabel('No run open.')
        self.Status_Label.setWordWrap(True)
        column.addWidget(self.Status_Label)
        self.Open_Button.clicked.connect(self.Browse)
        self.Latest_Button.clicked.connect(self._Open_Latest)
        self.Remove_Button.clicked.connect(self._Remove)
        return group

    def _Build_Axis(self):
        group = QGroupBox('Plot')
        self.Axis_Combo = QComboBox()
        self.Axis_Combo.addItems(spectra.AXES)
        self.Reference_Box = Spin(0.0, 0.0, 1e6, 5, ' cm⁻¹')
        self.Reference_Box.setToolTip("The ions' wavenumber the frequency axes are drawn "
                                      "from.")
        self.Centre_Button = QPushButton('Centre')
        self.Centre_Button.setToolTip("Take the reference from the first ticked run's middle.")
        self.Centre_Button.clicked.connect(self._Centre)
        reference = QHBoxLayout()
        reference.addWidget(self.Reference_Box, 1)
        reference.addWidget(self.Centre_Button)
        self.Normalise_Check = QCheckBox('Normalise each to its highest bin')
        self.Channel_Checks = [QCheckBox(f'{c}', checked=c != 5) for c in CHANNELS]
        self.Channel_Checks[-1].setToolTip("The old DAQ's DC counter.")
        channels = QHBoxLayout()
        for check in self.Channel_Checks:
            channels.addWidget(check)
        laser = self._Laser
        grid = Form(group, (('x-axis', self.Axis_Combo),))
        grid.addWidget(QLabel('Reference'), 1, 0)
        grid.addLayout(reference, 1, 1)
        grid.addWidget(QLabel('Channels'), 2, 0)
        grid.addLayout(channels, 2, 1)
        grid.addWidget(self.Normalise_Check, 3, 0, 1, 2)
        note = QLabel(f"Laser: harmonic {laser['harmonic']}, {laser['geometry']}, charge "
                      f"{laser['charge']} (settings: laser)")
        note.setWordWrap(True)
        grid.addWidget(note, 4, 0, 1, 2)
        return group

    def _Build_Estimator(self):
        self.Hfs_Group = QGroupBox('Hyperfine estimator', checkable=True, checked=False)
        self.Spin_Box = Spin(0.5, 0.0, 20.0, 1)
        self.J_Lower_Box, self.J_Upper_Box = Spin(0.5, 0.0, 20.0, 1), Spin(1.5, 0.0, 20.0, 1)
        for box in (self.Spin_Box, self.J_Lower_Box, self.J_Upper_Box):
            box.setSingleStep(0.5)          # angular momenta come in halves
        self.A_Lower_Box, self.A_Upper_Box = (Spin(0.0, -1e5, 1e5, 2, ' MHz') for _ in range(2))
        self.B_Lower_Box, self.B_Upper_Box = (Spin(0.0, -1e5, 1e5, 2, ' MHz') for _ in range(2))
        self.A_Ratio_Check = QCheckBox('Fix A upper / lower')
        self.B_Ratio_Check = QCheckBox('Fix B upper / lower')
        self.A_Ratio_Box, self.B_Ratio_Box = Spin(1.0, -1e3, 1e3, 4), Spin(1.0, -1e3, 1e3, 4)
        self.Centroid_Box = Spin(0.0, -1e6, 1e6, 2, ' MHz')
        self.Fwhm_Box = Spin(30.0, 0.1, 1e4, 1, ' MHz')
        self.Scale_Box = Spin(10.0, 0.0, 1e9, 2)
        self.Background_Box = Spin(0.0, 0.0, 1e9, 2)
        grid = Form(self.Hfs_Group, (
            ('I', self.Spin_Box), ('J lower', self.J_Lower_Box), ('J upper', self.J_Upper_Box),
            ('A lower', self.A_Lower_Box), ('A upper', self.A_Upper_Box),
            ('B lower', self.B_Lower_Box), ('B upper', self.B_Upper_Box),
            ('Centroid', self.Centroid_Box), ('FWHM', self.Fwhm_Box),
            ('Scale', self.Scale_Box), ('Background', self.Background_Box)))
        for row, (check, box) in enumerate(((self.A_Ratio_Check, self.A_Ratio_Box),
                                            (self.B_Ratio_Check, self.B_Ratio_Box)),
                                           grid.rowCount()):
            grid.addWidget(check, row, 0)
            grid.addWidget(box, row, 1)
        self.Hfs_Label = QLabel()
        self.Hfs_Label.setWordWrap(True)
        grid.addWidget(self.Hfs_Label, grid.rowCount(), 0, 1, 2)
        boxes = (self.Spin_Box, self.J_Lower_Box, self.J_Upper_Box, self.A_Lower_Box,
                 self.A_Upper_Box, self.B_Lower_Box, self.B_Upper_Box, self.A_Ratio_Box,
                 self.B_Ratio_Box, self.Centroid_Box, self.Fwhm_Box, self.Scale_Box,
                 self.Background_Box)
        for box in boxes:
            box.valueChanged.connect(self._Hfs_Changed)
        for check in (self.A_Ratio_Check, self.B_Ratio_Check):
            check.toggled.connect(self._Hfs_Changed)
        self._Hfs_Changed()
        return self.Hfs_Group

    # ── Runs ─────────────────────────────────────────────────────

    def Browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Open runs', self._Data_Dir,
                                                'Run files (run_*.asdf)')
        if paths:
            self.Open(paths)

    def _Open_Latest(self):
        runs = run_file.Runs(self._Data_Dir) if os.path.isdir(self._Data_Dir) else []
        if not runs:
            self.Status_Label.setText(f"No run in {html.escape(self._Data_Dir)} yet.")
            return
        self.Open(runs[:1])

    def Open(self, paths):
        """Read ``paths`` on a worker and show each as it is read; a run
        already open is read again, so a run still being written catches up."""
        if self._Loader is not None:
            return

        def job(progress, stopped):
            failures = []
            for path in paths:
                if stopped():
                    break
                try:
                    progress(spectra.Run(path))
                except Exception as e:
                    failures.append(f"{os.path.basename(path)}: {e}")
            return failures

        self._Loader = Job_Worker(job, self)
        self._Loader.Progress.connect(self._Add)
        self._Loader.Done.connect(self._Loaded)
        self._Loader.Failed.connect(lambda message: self.Status_Label.setText(
            html.escape(f"Not read: {message}")))
        self._Loader.finished.connect(self._Loader_Finished)
        self.Status_Label.setText(f"Reading {len(paths)} run{'s' * (len(paths) > 1)}...")
        self._Loader.start()

    def _Add(self, run):
        self._Runs[run.Path] = run
        for i in range(self.Run_List.count()):
            if self.Run_List.item(i).data(Qt.ItemDataRole.UserRole) == run.Path:
                break
        else:
            item = QListWidgetItem(f"{run.Name}   {run.Cooler_V:.1f} V")
            item.setData(Qt.ItemDataRole.UserRole, run.Path)
            item.setToolTip(run.Path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.Run_List.addItem(item)
        if not self.Reference_Box.value():
            self._Centre()
        self._Changed()

    def _Loaded(self, failures):
        text = f"{len(self._Runs)} run{'s' * (len(self._Runs) != 1)} open."
        if failures:
            text += " Not read: " + "; ".join(failures)
        self.Status_Label.setText(html.escape(text))

    def _Loader_Finished(self):
        self._Loader.Retire()
        self._Loader = None

    def Close(self):
        """Stop reading runs; a thread cannot outlive its window."""
        if self._Loader is not None:
            self._Loader.Stop()
            self._Loader.wait()

    def _Remove(self):
        for item in self.Run_List.selectedItems():
            self._Runs.pop(item.data(Qt.ItemDataRole.UserRole), None)
            self.Run_List.takeItem(self.Run_List.row(item))
        self._Changed()

    def Shown_Runs(self):
        """The ticked runs, in the list's order."""
        items = (self.Run_List.item(i) for i in range(self.Run_List.count()))
        return [self._Runs[item.data(Qt.ItemDataRole.UserRole)] for item in items
                if item.checkState() == Qt.CheckState.Checked]

    def _Centre(self):
        shown = self.Shown_Runs()
        if not shown:
            return
        try:
            self.Reference_Box.setValue(shown[0].Reference_Cm(self._Laser))
        except spectra.Spectrum_Error as e:
            self.Status_Label.setText(html.escape(str(e)))

    # ── Drawing ──────────────────────────────────────────────────

    def _Axis_Changed(self):
        axis = self.Axis_Combo.currentText()
        label, units = spectra.AXIS_LABELS[axis]
        bottom = self.Spectrum_Plot.getAxis('bottom')
        bottom.setLabel(label, units=units, color=theme.PLOT_FOREGROUND)
        bottom.enableAutoSIPrefix(units is not None)
        frequency = axis in spectra.AXES[3:]
        self.Reference_Box.setEnabled(frequency)
        self.Centre_Button.setEnabled(frequency)

    def _Changed(self, *_):
        self._Redraw.start()

    def _Hfs_Changed(self, *_):
        """Hold a fixed ratio, then describe the lines."""
        for check, ratio, lower, upper in (
                (self.A_Ratio_Check, self.A_Ratio_Box, self.A_Lower_Box, self.A_Upper_Box),
                (self.B_Ratio_Check, self.B_Ratio_Box, self.B_Lower_Box, self.B_Upper_Box)):
            upper.setEnabled(not check.isChecked())
            ratio.setEnabled(check.isChecked())
            if check.isChecked():
                upper.blockSignals(True)
                upper.setValue(lower.value() * ratio.value())
                upper.blockSignals(False)
        try:
            lines = self.Lines()
            self.Hfs_Label.setText(f"{len(lines)} lines")
        except hfs.Hfs_Error as e:
            self.Hfs_Label.setText(html.escape(str(e)))
        self._Changed()

    def Lines(self):
        return hfs.Lines(self.Spin_Box.value(), self.J_Lower_Box.value(),
                         self.J_Upper_Box.value(), self.A_Lower_Box.value(),
                         self.B_Lower_Box.value(), self.A_Upper_Box.value(),
                         self.B_Upper_Box.value())

    def _Gates(self):
        return self.Tof_Plot.Gate(), self.Time_Plot.Gate()

    def Draw(self):
        axis = self.Axis_Combo.currentText()
        reference = self.Reference_Box.value()
        channels = [c for c, check in zip(CHANNELS, self.Channel_Checks) if check.isChecked()]
        tof_gate, time_gate = self._Gates()
        for plot in (self.Spectrum_Plot, self.Tof_Plot, self.Time_Plot, self.Readback_Plot,
                     self.Difference_Plot, self.Step_Plot):
            for item in plot.listDataItems():
                plot.removeItem(item)
        self.Spectrum_Plot.plotItem.legend.clear()
        shown, problems = self.Shown_Runs(), []
        for i, run in enumerate(shown):
            pen = plots.Pen(RUN_COLOURS[i % len(RUN_COLOURS)])
            counts = run.Counts(channels, tof_gate, time_gate)
            try:
                edges = run.Axis(run.Edges, axis, self._Laser, reference)
            except spectra.Spectrum_Error as e:
                problems.append(str(e))
                continue
            if self.Normalise_Check.isChecked() and counts.max() > 0:
                counts = counts / counts.max()
            if edges[0] > edges[-1]:
                edges, counts = edges[::-1], counts[::-1]
            self.Spectrum_Plot.addItem(pg.PlotCurveItem(edges, counts, stepMode='center',
                                                        pen=pen, name=run.Name))
            for plot, column, gate, width in (
                    (self.Tof_Plot, spectra.TOF, (None, time_gate), self.Tof_Bin_Box.value()),
                    (self.Time_Plot, spectra.TIMESTAMP, (tof_gate, None),
                     self.Time_Bin_Box.value())):
                hits = run.Hits(channels, *gate)[:, column]
                if column == spectra.TIMESTAMP:
                    hits = hits - run.Start
                bins = _Edges(hits, width)
                if bins is not None:
                    plot.addItem(pg.PlotCurveItem(bins, np.histogram(hits, bins)[0],
                                                  stepMode='center', pen=pen))
            if len(run.Cal_Set) > 1:
                dots = plots.Dots(RUN_COLOURS[i % len(RUN_COLOURS)])
                self.Readback_Plot.plot(run.Cal_Set, run.Cal_Readback, **dots)
                self.Difference_Plot.plot(run.Cal_Set, run.Cal_Readback - run.Cal_Set, **dots)
                self.Step_Plot.plot(run.Cal_Set[1:], np.diff(run.Cal_Readback), **dots)
        if shown and self.Hfs_Group.isChecked():
            problems += self._Draw_Model(shown[0], axis, reference)
        self.Status_Label.setText(html.escape(' '.join(problems)) if problems else
                                  f"{len(shown)} of {len(self._Runs)} runs shown.")

    def _Draw_Model(self, run, axis, reference):
        try:
            lines = self.Lines()
        except hfs.Hfs_Error as e:
            return [str(e)]
        fwhm, centroid = self.Fwhm_Box.value(), self.Centroid_Box.value()
        positions = [line[2] for line in lines]
        reach = MODEL_REACH_FWHM * fwhm
        offsets = np.linspace(centroid + min(positions) - reach,
                              centroid + max(positions) + reach, MODEL_POINTS)
        model = hfs.Spectrum(offsets, lines, centroid, fwhm, self.Scale_Box.value(),
                             self.Background_Box.value())
        try:
            x = run.Model_Axis(offsets, axis, self._Laser, reference)
        except spectra.Spectrum_Error as e:
            return [str(e)]
        inside = np.isfinite(x)
        if not inside.any():
            return [f"The model lies outside {run.Name}'s scan."]
        self.Spectrum_Plot.addItem(pg.PlotCurveItem(x[inside], model[inside], name='HFS model',
                                                    pen=plots.Pen(theme.PLOT_FOREGROUND)))
        return []
