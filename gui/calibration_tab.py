"""The Scan Calibration tab: a stand-alone calibration and the gate's verdict on it.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The old DAQ's layout: the parameters and the start button on the left,
and on the right the calibration (readback against set) over the
readback's departure from the set voltage and the measured step size.
The plots fill point by point as the calibration runs, and the gate's
verdict and checks are listed under the button once it has finished.

A calibration runs on a worker thread, not the GUI thread, so the window
stays live without pumping events from inside it (the old DAQ's
processEvents loop let a click land mid-calibration); the main window
keeps every control that could interfere disabled until it has finished.

Depends on: numpy, PySide6; cls_acquisition.calibration,
cls_acquisition.scan_plan, gui.plots, gui.shared_widgets.
"""

import numpy as np
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from cls_acquisition import calibration
from cls_acquisition.scan_plan import Scan_Plan
from gui import plots
from gui.shared_widgets import REPORT_WIDTH, Form, Spin, Verdict_Html


class Calibration_Tab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)

        self.Params_Group = QGroupBox('Calibration parameters')
        self.From_Box = Spin(0.0, -10000.0, 10000.0)
        self.To_Box = Spin(3000.0, -10000.0, 10000.0)
        self.Step_Box = Spin(10.0, 0.0, 10000.0)
        Form(self.Params_Group, (('From (V)', self.From_Box), ('To (V)', self.To_Box),
                                 ('Stepsize (V)', self.Step_Box)))

        self.Control_Group = QGroupBox('Calibration control')
        self.Control_Group.setMinimumWidth(REPORT_WIDTH)
        column = QVBoxLayout(self.Control_Group)
        self.Start_Button = QPushButton('Start calibration')
        self.Report_Label = QLabel('No calibration yet in this session.')
        self.Report_Label.setWordWrap(True)
        column.addWidget(self.Start_Button)
        column.addWidget(self.Report_Label)
        column.addStretch(1)

        plot_group = QGroupBox('Plots')
        plot_grid = QGridLayout(plot_group)
        set_axis = ('Set voltage', 'V')
        self.Readback_Plot = plots.Plot('Calibration scan', set_axis, ('Readback voltage', 'V'))
        self.Difference_Plot = plots.Plot('Readback - set', set_axis, ('Difference', 'V'))
        self.Step_Plot = plots.Plot('Measured step size', set_axis, ('Step', 'V'))
        self._Curves = [p.plot(**dict(plots.Dots(), pen=plots.Pen()))
                        for p in (self.Readback_Plot, self.Difference_Plot, self.Step_Plot)]
        plot_grid.addWidget(self.Readback_Plot, 0, 0, 1, 2)
        plot_grid.addWidget(self.Difference_Plot, 1, 0)
        plot_grid.addWidget(self.Step_Plot, 1, 1)

        layout.addWidget(self.Params_Group, 0, 0)
        layout.addWidget(self.Control_Group, 1, 0)
        layout.addWidget(plot_group, 0, 1, 2, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)
        self._Set, self._Readback = [], []

    def Plan(self):
        """The Scan_Plan of the calibration parameters; raises Scan_Plan_Error."""
        return Scan_Plan([self.From_Box.value()], [self.To_Box.value()], self.Step_Box.value())

    def Begin(self, plan):
        self._Set, self._Readback = [], []
        self._Draw()
        self.Report_Label.setText(f"Calibrating {len(plan)} setpoints...")

    def Show_Progress(self, point):
        """``(index, commanded V, reading at the meter)`` of one kept point."""
        _, volts, reading = point
        self._Set.append(volts)
        self._Readback.append(reading * calibration.DIVIDER)
        self._Draw()

    def _Draw(self):
        set_v, readback = np.array(self._Set), np.array(self._Readback)
        self._Curves[0].setData(set_v, readback)
        self._Curves[1].setData(set_v, readback - set_v)
        self._Curves[2].setData(set_v[1:], np.diff(readback))

    def Show_Report(self, report):
        lines = [Verdict_Html(report['Verdict'], calibration.Summary(report))]
        lines += [Verdict_Html(c['Verdict'], f"{c['Check']}: {c['Message']}")
                  for c in report['Checks']]
        self.Report_Label.setText('<br>'.join(lines))

    def Show_Failure(self, message):
        self.Report_Label.setText(Verdict_Html('FAIL', message))
