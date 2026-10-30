"""The Scan Calibration tab: a stand-alone calibration and the gate's verdict on it.

Date:    2026-10-30
Version: 0.2.0
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

import html

import numpy as np
from PySide6.QtWidgets import (QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from cls_acquisition import calibration
from cls_acquisition.scan_plan import Scan_Plan
from gui import plots
from gui.shared_widgets import REPORT_WIDTH, Form, Scroll_Column, Spin, Verdict_Html


class Calibration_Tab(QWidget):

    def __init__(self, limit_v, parent=None):
        super().__init__(parent)
        self.Params_Group = QGroupBox('Calibration parameters')
        self.From_Box = Spin(0.0, -limit_v, limit_v)
        self.To_Box = Spin(min(3000.0, limit_v), -limit_v, limit_v)
        self.Step_Box = Spin(10.0, 0.0, 2 * limit_v)
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
        set_axis = ('Setpoint', 'V')
        self.Readback_Plot = plots.Plot('HV readback against setpoint', set_axis,
                                        ('Readback voltage', 'V'), (
            "The platform voltage measured at each step of the calibration, against the voltage "
            "commanded. A working high-voltage chain lies on a straight line of unit slope.",
            "Each setpoint is written to the DAC and, once the amplifier has settled, the line "
            "meter reads the platform through its 1000:1 divider; a reading taken before the "
            "step had settled is repeated. The run's calibration is the straight line fitted "
            "to these points."))
        self.Difference_Plot = plots.Plot('Readback deviation from setpoint', set_axis,
                                          ('Readback − setpoint', 'V'), (
            "Measured minus commanded voltage at each calibration step: the offset, gain error "
            "and non-linearity of the high-voltage chain.",
            "The readback (line meter reading × 1000) less the setpoint. An offset or a "
            "straight-line trend is absorbed by the linear calibration; curvature and periodic "
            "structure (the DAC's integral non-linearity, and its rounding to codes 0.305 V "
            "apart) remain in the voltage axis unless a DAC correction is in force."))
        self.Step_Plot = plots.Plot('Measured voltage step', set_axis, ('Readback step', 'V'), (
            "The change in readback from one calibration step to the next, which should equal "
            "the commanded step size.",
            "The difference of successive readbacks, plotted at the later setpoint. Scatter "
            "reflects the DAC's code rounding and the meter's noise; an outlier marks a step "
            "that had not settled when it was read."))
        self._Curves = [p.plot(**dict(plots.Dots(), pen=plots.Pen()))
                        for p in (self.Readback_Plot, self.Difference_Plot, self.Step_Plot)]
        plot_grid.addWidget(self.Readback_Plot, 0, 0, 1, 2)
        plot_grid.addWidget(self.Difference_Plot, 1, 0)
        plot_grid.addWidget(self.Step_Plot, 1, 1)

        controls = QVBoxLayout()
        controls.addWidget(self.Params_Group)
        controls.addWidget(self.Control_Group, 1)
        layout = QHBoxLayout(self)
        layout.addWidget(Scroll_Column(controls))
        layout.addWidget(plot_group, 1)
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

    def Inputs(self):
        return [('from', self.From_Box), ('to', self.To_Box), ('step', self.Step_Box)]

    def Show_Run(self, name, commanded, readings, report=None):
        """A run file's own calibration: the voltages commanded and the meter's
        readings, as written, with the gate's ``report`` on them as it was
        given when the run was taken, from ORACLE's record of it. A run with
        no record gets its line and where its branches meet 0 V; the gate is
        not run again, since a file keeps no stamps."""
        commanded = np.asarray(commanded, dtype=float)
        readback = np.asarray(readings, dtype=float) * calibration.DIVIDER
        if len(commanded) != len(readback) or len(commanded) < 2:
            self._Set, self._Readback = [], []
            self._Draw()
            self.Report_Label.setText(html.escape(
                f"{name} holds no usable calibration: {len(readback)} readings for "
                f"{len(commanded)} setpoints.", quote=False))
            return
        self._Set, self._Readback = list(commanded), list(readback)
        self._Draw()
        if report is not None:
            self.Show_Report(report)
            return
        gain, offset, rms = calibration.Line_Fit(commanded, readback)
        seam = calibration.Step_At_Zero(commanded, readback)
        self.Report_Label.setText(html.escape(
            f"{name}'s calibration, {len(commanded)} points: readback = {gain:.6f} × set "
            f"{offset:+.3f} V, residual {rms:.3f} V rms"
            + (f"; its branches' lines meet 0 V {seam:+.3f} V apart" if seam is not None
               else "") + ".", quote=False))
