"""The Scan Calibration tab: a stand-alone calibration and the gate's verdict on it.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The old DAQ's layout: the parameters and the start button on the left,
and on the right the calibration (readback against set) over the
readback's departure from the set voltage and the measured step size.
The plots fill point by point as the calibration runs, and the gate's
verdict, the figures it decided on and each check's verdict are reported
under the button once it has finished.

A calibration runs on a worker thread, not the GUI thread, so the window
stays live without pumping events from inside it (the old DAQ's
processEvents loop let a click land mid-calibration); the main window
keeps every control that could interfere disabled until it has finished.
Readings are scaled to the platform by the scan divider's ratio, the
settings' for a calibration taken here and the run's own for a run shown.

With a bench table in force, the calibration is fitted to it, V = a +
b*T (scan_table.Fit_Run): reported beside the gate, with the runs before
it on the same table, and drawn over the readings' departure from the
setpoint, the converters' shape the table holds.

Depends on: numpy, PySide6; cls_acquisition.calibration,
cls_acquisition.scan_plan, gui.plots, gui.shared_widgets.
"""

import numpy as np
from PySide6.QtWidgets import (QGridLayout, QGroupBox, QHBoxLayout, QPushButton,
                               QVBoxLayout, QWidget)

from cls_acquisition import calibration
from cls_acquisition.scan_plan import Scan_Plan
from gui import plots, theme
from gui.shared_widgets import (REPORT_WIDTH, Figure, Form, Muted, Report_Html, Report_Label,
                                Report_Table, Scroll_Column, Spin, Verdict_Badge)

# the gate's checks as a report names them
CHECK_NAMES = {'dac_calibration': 'DAC calibration'}


class Calibration_Tab(QWidget):

    def __init__(self, limit_v, divider=calibration.DIVIDER, parent=None):
        super().__init__(parent)
        self._Divider = divider
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
        self.Report_Label = Report_Label('No calibration yet in this session.')
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
            "meter reads the platform through the scan divider (1000:1 unless the settings "
            "say otherwise), integrating whole power line cycles so the mains on the divider "
            "averages out; a reading taken before the step had settled is repeated. The "
            "run's calibration is the straight line fitted to these points."))
        self.Difference_Plot = plots.Plot('Readback deviation from setpoint', set_axis,
                                          ('Readback − setpoint', 'V'), (
            "Measured minus commanded voltage at each calibration step: the offset, gain error "
            "and non-linearity of the high-voltage chain.",
            "The readback (line meter reading × the divider's ratio) less the setpoint. An "
            "offset or a straight-line trend is absorbed by the linear calibration; curvature "
            "and periodic structure (the DAC's integral non-linearity, and its rounding to "
            "codes 0.305 V apart) remain in the voltage axis unless a DAC correction is in "
            "force. With a bench table in force, the line is the table fitted to the "
            "readings, a + b·T, less the setpoint."))
        self.Step_Plot = plots.Plot('Measured voltage step', set_axis, ('Readback step', 'V'), (
            "The change in readback from one calibration step to the next, which should equal "
            "the commanded step size.",
            "The difference of successive readbacks, plotted at the later setpoint. Scatter "
            "reflects the DAC's code rounding and the meter's noise; an outlier marks a step "
            "that had not settled when it was read."))
        self._Curves = [p.plot(**dict(plots.Dots(), pen=plots.Pen()))
                        for p in (self.Readback_Plot, self.Difference_Plot, self.Step_Plot)]
        # the bench table fitted to the readings, over their departure from the setpoint
        self._Table_Curve = self.Difference_Plot.plot(pen=plots.Pen(theme.TRACE_ALT))
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
        self._Table_Curve.setData([], [])
        self.Report_Label.setText(f"Calibrating {len(plan)} setpoints...")

    def Show_Progress(self, point):
        """``(index, commanded V, reading at the meter)`` of one kept point."""
        _, volts, reading = point
        self._Set.append(volts)
        self._Readback.append(reading * self._Divider)
        self._Draw()

    def _Draw(self):
        set_v, readback = np.array(self._Set), np.array(self._Readback)
        self._Curves[0].setData(set_v, readback)
        self._Curves[1].setData(set_v, readback - set_v)
        self._Curves[2].setData(set_v[1:], np.diff(readback))

    def Show_Report(self, report, table=None, history=(), model=None):
        """The gate's verdict, the figures it decided on, and each check's
        verdict over what it found; then the fit to the bench table,
        ``table`` (scan_table.Fit_Run's, or a sentence why there is none),
        with the ``history`` of (run, a, b) on the same table, and the
        fitted table ``model`` = (setpoints, volts) drawn."""
        figures = []
        if 'TrackingR' in report:
            figures = [('Tracking |r|', f"{report['TrackingR']:.5f}"),
                       ('Gradient, set per readback', f"{report['FitGradient']:.5f}"),
                       ('Residual rms', Figure(report['FitResidualRmsV'], '.3f', 'V')),
                       ('Shortest settling', f"≥ {Figure(report['ShortestGapMs'], '.1f', 'ms')}")]
        checks = []
        for check in report['Checks']:
            name = CHECK_NAMES.get(check['Check'], check['Check'].capitalize())
            checks += [(name, Verdict_Badge(check['Verdict'])), Muted(check['Message'])]
        self.Report_Label.setText(Report_Html([((report['Verdict'], 'Calibration gate'), figures),
                                               ('Checks', checks),
                                               *self._Table_Sections(table, history)]))
        self._Show_Model(model)

    def _Show_Model(self, model):
        if model is None:
            self._Table_Curve.setData([], [])
            return
        setpoint, volts = (np.asarray(v, dtype=float) for v in model)
        order = np.argsort(setpoint)
        self._Table_Curve.setData(setpoint[order], (volts - setpoint)[order])

    @staticmethod
    def _Table_Sections(table, history):
        """The fit to the bench table as report sections."""
        if table is None:
            return []
        if isinstance(table, str):
            return [('Against the bench table', [Muted(table)])]
        rows = [('Table', table['Table'][:8]),
                ('a (0 V level)', f"{Figure(table['A_V'] * 1e3, '+.1f')} ± "
                                  f"{Figure(table['A_Err_V'] * 1e3, '.1f', 'mV')}"),
                ('b − 1', f"{Figure((table['B'] - 1.0) * 1e6, '+.1f')} ± "
                          f"{Figure(table['B_Err'] * 1e6, '.1f', 'ppm')}"),
                ('Residual rms', f"{Figure(table['Rms_V'] * 1e3, '.1f')} mV against "
                                 f"{Figure(table['Expected_V'] * 1e3, '.1f', 'mV')} expected"),
                ('Points', f"{table['Points']}" + (f", {table['Interpolated']} between the "
                                                    f"table's codes"
                                                    if table['Interpolated'] else ''))]
        if table['Verdict'] != 'PASS':
            rows.append(Muted("The readings sit further from the table than the meter and "
                              "the table's repeatability account for: the table is stale or "
                              "the hardware changed. Measure the table again."))
        sections = [((table['Verdict'], 'Against the bench table'), rows)]
        if history:
            sections.append(('Earlier runs on this table', [Report_Table(
                ('Run', 'a (mV)', 'b − 1 (ppm)'),
                [(run, Figure(a * 1e3, '+.1f'), Figure((b - 1.0) * 1e6, '+.1f'))
                 for run, a, b in history])]))
        return sections

    def Show_Failure(self, message):
        self.Report_Label.setText(Report_Html([(('FAIL', 'Calibration failed'), [message])]))

    def Inputs(self):
        return [('from', self.From_Box), ('to', self.To_Box), ('step', self.Step_Box)]

    def Show_Run(self, name, commanded, readings, report=None, divider=calibration.DIVIDER,
                 table=None, history=(), model=None):
        """A run file's own calibration: the voltages commanded and the meter's
        readings, as written, scaled by the run's own ``divider``, with the
        gate's ``report`` on them as it was given when the run was taken,
        from ORACLE's record of it. A run with no record gets its line and
        where its branches meet 0 V; the gate is not run again, since a file
        keeps no stamps. ``table``, ``history`` and ``model`` are its fit to
        the bench table, as :meth:`Show_Report` takes them."""
        commanded = np.asarray(commanded, dtype=float)
        readback = np.asarray(readings, dtype=float) * divider
        if len(commanded) != len(readback) or len(commanded) < 2:
            self._Set, self._Readback = [], []
            self._Draw()
            self.Report_Label.setText(Report_Html([(name, [
                f"No usable calibration: {len(readback)} readings for {len(commanded)} "
                f"setpoints."])]))
            self._Show_Model(None)
            return
        self._Set, self._Readback = list(commanded), list(readback)
        self._Draw()
        if report is not None:
            self.Show_Report(report, table, history, model)
            return
        self._Show_Model(None)
        gain, offset, rms = calibration.Line_Fit(commanded, readback)
        seam = calibration.Step_At_Zero(commanded, readback)
        self.Report_Label.setText(Report_Html([(f"{name}, its straight line", [
            ('Points', len(commanded)), ('Gain', f"{gain:.6f}"),
            ('Offset', Figure(offset, '+.3f', 'V')), ('Residual rms', Figure(rms, '.3f', 'V')),
            *((('Branch lines meet 0 V', f"{Figure(seam, '+.3f', 'V')} apart"),)
              if seam is not None else ()),
            Muted("Taken before the gate's verdict was recorded with each run, so only its "
                  "line is shown.")])]))
