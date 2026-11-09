"""The Hardware Calibration tab: settle time, linearity and the cooler monitor.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The bench calibration tool's measurements, each on a page of its own in
the Scan Calibration tab's layout: the parameters and the start and stop
buttons on the left, the result under them, and the plots filling as the
measurement runs. The ramp between measured points, and the limit every
setpoint is held to, sit above the pages and apply to all of them.

Each page reads its own widgets into its measurement's parameters
(``Params``, which refuses what cannot run with a sentence), says the
most it will put on the platform (``Max_Volts``, which the window asks
the operator to arm on the rack), names the records it writes as it
runs (``RECORDS``: kind to columns), the rows its result adds once it
ends (``Rows``: the cooler's spectrum) and what of its result closes
them (``Results``). A measurement that fails part way is still analysed
as far as it went (``Partial``), as the bench tool's results_partial
was. The main window runs every page's measurement the same way, off
the GUI thread, keeping everything that could drive the DAC or the
meters disabled until it has finished.

Each page reports its result in titled sections of labelled figures
(shared_widgets.Report_Html), ending with the files it saved, each a
link; what happens to the result afterwards (a correction put in force,
a record opened rather than taken) is added as a section of its own.

Above the pages, beside the ramp, the line meter can be read precisely
(the bench tool's 'Read DMM') to check the wiring and polarity against
whatever quick tune holds; and the settle page hands its recommended
wait to the linearity page, as the bench tool's did.

Depends on: numpy, pyqtgraph, PySide6; cls_acquisition.calibration,
cls_acquisition.hardware_calibration, cls_acquisition.hardware.mcc_dac,
gui.plots, gui.shared_widgets, gui.theme.
"""

import html
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from cls_acquisition import hardware_calibration as hc
from cls_acquisition.calibration import SETTLE_MARGIN, THRESHOLDS, Line_Fit
from cls_acquisition.hardware.mcc_dac import CH_NEG, CH_POS
from gui import plots, theme
from gui.shared_widgets import (REPORT_WIDTH, Duration_Text, Figure, Form, Muted, Report_Html,
                                Report_Label, Report_Table, Saved_Section, Scroll_Column, Spin)

STEP_ROWS = 4
NPLC_CHOICES = ('0.02', '0.2', '1', '10', '100')
TAIL_NPLC_CHOICES = ('0.2', '1', '10')
TAIL_INTERVAL_S = 0.1               # the meter raises it to what the integration needs
BRANCH_COLOURS = {CH_POS: theme.TRACE, CH_NEG: theme.TRACE_ALT}
MODE_COLOURS = {'out': theme.TRACE, 'back': theme.TRACE_ALT}
ZERO_COLOUR = theme.PLOT_FOREGROUND
CSV_BYTES_PER_READING = 30
DEVIATION_SPAN = 10                 # tolerances either side, the deviation plot's view


def _Table(columns):
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.verticalHeader().setVisible(False)
    return table


def _Add_Row(table, cells):
    row = table.rowCount()
    table.insertRow(row)
    for column, value in enumerate(cells):
        table.setItem(row, column, QTableWidgetItem(str(value)))


class _Panel(QWidget):
    """One measurement's page: parameters and control down the left, plots
    on the right. ``Result`` is the last ``(result, record paths)``."""

    RECORDS = {}

    def __init__(self, title, plot_widgets, parent=None):
        super().__init__(parent)
        self.Title = title
        self.Result = None
        self.Params_Group = QGroupBox(f'{title} parameters')
        self.Control_Group = QGroupBox(f'{title} control')
        self.Control_Group.setMinimumWidth(REPORT_WIDTH)
        column = QVBoxLayout(self.Control_Group)
        buttons = QHBoxLayout()
        self.Start_Button = QPushButton(f'Run {title.lower()}')
        self.Stop_Button = QPushButton('Stop', enabled=False)
        self.Open_Button = QPushButton(theme.Icon('folder-open'), 'Open record...')
        self.Open_Button.setToolTip(f"Show a {title.lower()} record saved earlier, here or "
                                    f"by the rack: its plots, its table and its analysis.")
        for button in (self.Start_Button, self.Stop_Button, self.Open_Button):
            buttons.addWidget(button)
        column.addLayout(buttons)
        self.Report_Label = Report_Label('Not run yet in this session.')
        self._Sections, self._Notes = [], {}
        column.addWidget(self.Report_Label)
        column.addStretch(1)
        self._Control_Column = column

        plot_group = QGroupBox('Plots')
        plot_column = QVBoxLayout(plot_group)
        for widget, stretch in plot_widgets:
            plot_column.addWidget(widget, stretch)
        controls = QVBoxLayout()
        controls.addWidget(self.Params_Group)
        controls.addWidget(self.Control_Group, 1)
        layout = QHBoxLayout(self)
        layout.addWidget(Scroll_Column(controls))
        layout.addWidget(plot_group, 1)

    def Max_Volts(self, params):
        """The most the measurement puts on the platform, or None if it
        never drives the DAC."""
        return None

    def Results(self, result):
        """``{record kind: {name: value}}``, written as each record closes."""
        return {}

    def Rows(self, result):
        """``{record kind: columns}`` the result adds once it ends."""
        return {}

    def Partial(self):
        """What has been shown so far, as a result ``Results`` and ``Rows``
        take, or None if it is too little to analyse."""
        return None

    def Show_Record(self, path):
        """The record at ``path`` drawn and analysed as when it was taken;
        raises Calibration_Error naming what it lacks."""
        raise NotImplementedError

    def _Replay(self, items, path):
        """Clear the page as a measurement would, then show ``items``."""
        self.Begin()
        self.Running(False)
        for item in items:
            self.Show_Progress(item)

    def Begin(self):
        self._Sections, self._Notes = [], {}
        self.Report_Label.setText("Measuring...")
        self.Report_Label.setToolTip('')
        self.Stop_Button.setEnabled(True)

    def Running(self, running):
        self.Stop_Button.setEnabled(running)

    def Show_Result(self, result, paths, sections):
        """``result`` reported in ``sections`` (Report_Html's), then the files
        ``paths`` it was saved to."""
        self.Result = (result, paths)
        self._Sections, self._Notes = [*sections, Saved_Section(paths)], {}
        self.Report_Label.setToolTip(f"Saved in {os.path.dirname(paths[0])}" if paths else '')
        self._Render()

    def Add_Note(self, heading, rows):
        """A section added under the report, in place of one with the same title."""
        self._Notes[heading[1] if isinstance(heading, tuple) else heading] = (heading, rows)
        self._Render()

    def _Render(self):
        self.Report_Label.setText(Report_Html([*self._Sections, *self._Notes.values()]))

    def Show_Failure(self, message):
        self._Sections, self._Notes = [(('FAIL', 'Not completed'), [message])], {}
        self._Render()

    def Show_Stopped(self, message):
        self._Sections, self._Notes = [(('WARN', 'Stopped'), [message])], {}
        self._Render()


class Settle_Panel(_Panel):

    COLUMNS = ('step', 'mode', 'rep', 'from V', 'to V', 'lag ms', 'settle ms',
               'rise 10-90 ms', 'overshoot %', 'noise V', 'tail shift V')
    RECORDS = {'settle': [column for _, column in hc.SETTLE_COLUMNS],
               'settle_traces': hc.SETTLE_TRACE_COLUMNS}

    def __init__(self, ramp, limit_v, parent=None):
        since_crossing = ('Time from 10 % crossing', 's')
        self.Response_Plot = plots.Plot('Step response', since_crossing,
                                        ('Platform voltage', 'V'), (
            "The platform voltage through each commanded step of the settle test, outward steps "
            "and return steps in different colours, showing the amplifier's delay, rise and "
            "overshoot.",
            "The line meter samples the platform at the chosen interval and aperture from before "
            "the step is commanded; each trace is shifted in time so that zero is where it "
            "crosses 10 % of the step's height."))
        self.Deviation_Plot = plots.Plot('Settling to the final value', since_crossing,
                                         ('Deviation from final value', 'V'), (
            "The same steps relative to their final value, magnified around zero to show how "
            "each approaches it. The dashed lines mark the settle tolerance.",
            "The final value is the median of the last fifth of each capture. A step has "
            "settled once its trace stays inside the tolerance (or inside a band set by its "
            "noise, if that is wider); the settle times in the table are when each trace last "
            "leaves it."))
        self.Table = _Table(self.COLUMNS)
        super().__init__('Settle test', ((self.Response_Plot, 3), (self.Deviation_Plot, 3),
                                         (self.Table, 2)), parent)
        self._Ramp = ramp
        self.Step_Checks, self.From_Boxes, self.To_Boxes = [], [], []
        grid = QGridLayout(self.Params_Group)
        grid.addWidget(QLabel('From (V)'), 0, 1)
        grid.addWidget(QLabel('To (V)'), 0, 2)
        for i in range(STEP_ROWS):
            check = QCheckBox(checked=i == 0)
            start = Spin(0.0, -limit_v, limit_v, 1)
            stop = Spin(min(1000.0 * (i + 1), limit_v), -limit_v, limit_v, 1)
            for column, widget in enumerate((check, start, stop)):
                grid.addWidget(widget, 1 + i, column)
            self.Step_Checks.append(check)
            self.From_Boxes.append(start)
            self.To_Boxes.append(stop)
        self.Reps_Box = Spin(3, 1, 100, 0)
        self.Window_Box = Spin(500, 10, 20000, 0, ' ms')
        self.Interval_Box = Spin(0.5, 0.02, 100.0, 2, ' ms')
        self.Aperture_Box = Spin(200, 1, 100000, 0, ' µs')
        self.Hold_Box = Spin(2.0, 0.0, 600.0, 2, ' s')
        self.Tolerance_Box = Spin(0.5, 0.001, 100.0, 3, ' V')
        self.Tail_Check = QCheckBox('Drift tail after each step')
        self.Tail_Check.setToolTip("Read the platform for longer at a long integration once "
                                   "the fast capture ends, to see whether it still creeps "
                                   "(dielectric absorption in the cable and divider).")
        self.Tail_Box = Spin(30.0, 1.0, 3600.0, 1, ' s')
        self.Tail_Nplc_Combo = QComboBox()
        self.Tail_Nplc_Combo.addItems(TAIL_NPLC_CHOICES)
        self.Tail_Nplc_Combo.setCurrentText('1')
        for row, (text, widget) in enumerate(
                (('Repetitions', self.Reps_Box), ('Capture window', self.Window_Box),
                 ('Sample interval', self.Interval_Box), ('Aperture', self.Aperture_Box),
                 ('Hold before each step', self.Hold_Box),
                 ('Settle tolerance', self.Tolerance_Box), (None, self.Tail_Check),
                 ('Tail length', self.Tail_Box), ('Tail NPLC', self.Tail_Nplc_Combo)),
                1 + STEP_ROWS):
            if text is None:
                grid.addWidget(widget, row, 0, 1, 3)
                continue
            grid.addWidget(QLabel(text), row, 0, 1, 2)
            grid.addWidget(widget, row, 2)
        self.Tail_Check.toggled.connect(self._Tail_Changed)
        self._Tail_Changed()

        # the band a step must stay inside, either side of its final value
        self._Band = [pg.InfiniteLine(angle=0, pen=pg.mkPen(theme.VERDICT_COLOURS['WARN'],
                                                            style=Qt.PenStyle.DashLine))
                      for _ in range(2)]
        for line in self._Band:
            self.Deviation_Plot.addItem(line)

        self.Model_Group = QGroupBox('Scan settle model')
        model = QVBoxLayout(self.Model_Group)
        self.Model_Label = QLabel()
        self.Model_Label.setWordWrap(True)
        self.Use_Button = QPushButton('Use as scan settle model', enabled=False)
        self.Clear_Button = QPushButton('Clear scan settle model')
        self.Wait_Button = QPushButton('Use recommended wait for linearity', enabled=False)
        self.Wait_Button.setToolTip(f"Set the linearity sweep's settle after each step to "
                                    f"{hc.WAIT_FACTOR:g} times the slowest step here, from "
                                    f"its command, and at least {hc.MIN_WAIT_S:g} s.")
        for widget in (self.Model_Label, self.Use_Button, self.Clear_Button, self.Wait_Button):
            model.addWidget(widget)
        self._Control_Column.insertWidget(1, self.Model_Group)
        self._Results = []

    def _Tail_Changed(self):
        for widget in (self.Tail_Box, self.Tail_Nplc_Combo):
            widget.setEnabled(self.Tail_Check.isChecked())

    def Inputs(self):
        steps = [(f"step_{i}_{part}", widget) for i, row in enumerate(zip(
            self.Step_Checks, self.From_Boxes, self.To_Boxes))
            for part, widget in zip(('on', 'from', 'to'), row)]
        return [*steps, ('reps', self.Reps_Box), ('window', self.Window_Box),
                ('interval', self.Interval_Box), ('aperture', self.Aperture_Box),
                ('hold', self.Hold_Box), ('tolerance', self.Tolerance_Box),
                ('tail', self.Tail_Check), ('tail_length', self.Tail_Box),
                ('tail_nplc', self.Tail_Nplc_Combo)]

    def Show_Model(self, model):
        if not model:
            self.Model_Label.setText(f"None measured: every scan step waits SETTLE "
                                     f"({THRESHOLDS['SettleS'] * 1e3:.0f} ms).")
            return
        points = ', '.join(f"{jump:g} V: {s * 1e3:.1f} ms" for jump, s in model['points'])
        self.Model_Label.setText(f"Measured {model['measured']}: {points}; each step waits "
                                 f"{SETTLE_MARGIN:g} times this for its jump.")

    def Params(self):
        steps = [(start.value(), stop.value()) for check, start, stop
                 in zip(self.Step_Checks, self.From_Boxes, self.To_Boxes) if check.isChecked()]
        if not steps:
            raise ValueError("No step is ticked.")
        if any(start == stop for start, stop in steps):
            raise ValueError("A ticked step goes from a voltage to itself.")
        interval_ms, aperture_us = self.Interval_Box.value(), self.Aperture_Box.value()
        if aperture_us / 1e3 >= interval_ms:
            raise ValueError(f"An aperture of {aperture_us:g} µs does not fit a sample "
                             f"interval of {interval_ms:g} ms.")
        return {'steps': steps, 'reps': self.Reps_Box.value(),
                'window_s': self.Window_Box.value() / 1e3, 'interval_s': interval_ms / 1e3,
                'aperture_s': aperture_us / 1e6, 'hold_s': self.Hold_Box.value(),
                'tolerance_v': self.Tolerance_Box.value(), 'ramp': self._Ramp(),
                'tail_s': self.Tail_Box.value() if self.Tail_Check.isChecked() else 0.0,
                'tail_nplc': float(self.Tail_Nplc_Combo.currentText()),
                'tail_interval_s': TAIL_INTERVAL_S}

    def Max_Volts(self, params):
        return max(abs(v) for step in params['steps'] for v in step)

    def Results(self, results):
        verdict, sentence = hc.Settle_Summary(results)
        return {'settle': {'Verdict': f"{verdict}: {sentence}"}}

    def Partial(self):
        return list(self._Results) or None

    def Begin(self):
        super().Begin()
        self._Results = []
        self.Wait_Button.setEnabled(False)
        for plot in (self.Response_Plot, self.Deviation_Plot):
            for item in plot.listDataItems():
                plot.removeItem(item)
        self._Show_Band(self.Tolerance_Box.value())
        self.Table.setRowCount(0)
        self.Use_Button.setEnabled(False)

    def _Show_Band(self, tolerance):
        """The band a step must stay inside, on a view of the settling, not
        the step: the jump itself runs far off it."""
        for line, sign in zip(self._Band, (1, -1)):
            line.setValue(sign * tolerance)
        self.Deviation_Plot.setYRange(-DEVIATION_SPAN * tolerance, DEVIATION_SPAN * tolerance)

    def Show_Record(self, path):
        results = hc.Read_Settle(path)
        if not results:
            raise hc.Calibration_Error(f"{path} holds no step.")
        self._Replay(results, path)
        self._Show_Band(min(r['Band_V'] for r in results))
        self.Show_Result(results, [path])

    def Show_Progress(self, result):
        self._Results.append(result)
        pen = plots.Pen(MODE_COLOURS[result['Mode']])
        times = result['Times_S']
        self.Response_Plot.plot(times, result['Deviation_V'] + result['Final_V'], pen=pen)
        self.Deviation_Plot.plot(times, result['Deviation_V'], pen=pen)
        shift = result['Tail_Shift_V']
        _Add_Row(self.Table, (result['Step'], result['Mode'], result['Rep'],
                              f"{result['From_V']:g}", f"{result['To_V']:g}",
                              f"{result['Lag_S'] * 1e3:.2f}", f"{result['Settle_S'] * 1e3:.2f}",
                              f"{result['Rise_S'] * 1e3:.2f}", f"{result['Overshoot_Pct']:.2f}",
                              f"{result['Noise_V']:.3f}",
                              f"{shift:+.3f}" if np.isfinite(shift) else '–'))

    def Show_Result(self, results, paths):
        verdict, _ = hc.Settle_Summary(results)
        steps = hc.Settle_Steps(results)
        sections = [((verdict, 'Settling'), ["No step was measured."])]
        if steps:
            slowest = max(steps, key=lambda step: step['Worst_S'])
            tails = any(step['Tail_Shift_V'] is not None for step in steps)
            sections = [((verdict, 'Settling'), [
                ('Slowest step', f"{slowest['Step']} {slowest['Mode']}, "
                                 f"{Figure(slowest['From_V'], 'g')} → "
                                 f"{Figure(slowest['To_V'], 'g', 'V')}"),
                ('Settled after its command', Figure(slowest['Worst_S'] * 1e3, '.1f', 'ms')),
                ("ORACLE's SETTLE", Figure(THRESHOLDS['SettleS'] * 1e3, '.0f', 'ms'))]),
                ('Steps', [Report_Table(
                    ('Step', 'Jump (V)', 'Worst (ms)', 'Mean (ms)', 'Overshoot (%)',
                     *(('Tail (V)',) if tails else ())),
                    [(f"{step['Step']} {step['Mode']}",
                      f"{Figure(step['From_V'], 'g')} → {Figure(step['To_V'], 'g')}",
                      f"{step['Worst_S'] * 1e3:.1f}", f"{step['Mean_S'] * 1e3:.1f}",
                      f"{step['Overshoot_Pct']:.2f}",
                      *((('–' if step['Tail_Shift_V'] is None
                          else f"{step['Tail_Shift_V']:.3f}"),) if tails else ()))
                     for step in steps]),
                    Muted(f"From each step's command, over {max(s['Reps'] for s in steps)} "
                          f"repetitions at most.")])]
        super().Show_Result(results, paths, sections)
        self.Use_Button.setEnabled(bool(results))
        self.Wait_Button.setEnabled(bool(results))


class Linearity_Panel(_Panel):

    RECORDS = {'linearity': hc.LINEARITY_COLUMNS}

    def __init__(self, ramp, limit_v, parent=None):
        set_axis = ('Setpoint', 'V')
        self.Difference_Plot = plots.Plot('Readback deviation from setpoint', set_axis,
                                          ('Readback − setpoint', 'V'), (
            "Readback minus setpoint at every point of the linearity sweep: points on the "
            "positive and negative converter branches in different colours, and the 0 V "
            "reference checks as squares.",
            "Each point is the mean of the readings taken at its setpoint after the chosen "
            "settle time, at the chosen integration time (NPLC). A straight trend is the "
            "chain's gain and offset; curvature and periodic structure are the DAC's integral "
            "non-linearity."))
        self.Residual_Plot = plots.Plot('Residual from a linear fit', set_axis,
                                        ('Residual', 'V'), (
            "What a single straight-line calibration leaves in the voltage axis: the readback "
            "less the line fitted to every sweep point so far.",
            "A least-squares line through the sweep points, refitted as points arrive; the "
            "residual is readback − (gain × setpoint + offset). Its rms and maximum are "
            "reported below the plots, and a DAC correction table built from the sweep removes "
            "it."))
        self.Fit_Label = QLabel()
        self.Fit_Label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.Fit_Label.setWordWrap(True)
        super().__init__('Linearity sweep', ((self.Difference_Plot, 1),
                                             (self.Residual_Plot, 1), (self.Fit_Label, 0)),
                         parent)
        self._Ramp = ramp
        span = min(3000.0, limit_v)
        self.From_Box = Spin(-span, -limit_v, limit_v, 1, ' V')
        self.To_Box = Spin(span, -limit_v, limit_v, 1, ' V')
        self.Step_Box = Spin(50.0, 0.1, 2 * limit_v, 1, ' V')
        self.Fine_Span_Box = Spin(20.0, 0.0, 1000.0, 1, ' V')
        self.Fine_Span_Box.setToolTip("Extra rungs either side of 0 V, where the two "
                                      "converter branches hand over (0: none).")
        self.Fine_Step_Box = Spin(2.0, 0.1, 100.0, 1, ' V')
        self.Direction_Combo = QComboBox()
        self.Direction_Combo.addItems(hc.DIRECTIONS)
        self.Direction_Combo.setCurrentIndex(2)
        self.Seed_Box = Spin(0, 0, 1000000, 0)
        self.Zero_Box = Spin(10, 0, 1000, 0)
        self.Zero_Box.setToolTip("Read 0 V again before every this many rungs, to follow the "
                                 "drift (0: never).")
        self.Settle_Box = Spin(0.5, 0.0, 600.0, 2, ' s')
        self.Readings_Box = Spin(5, 1, 1000, 0)
        self.Nplc_Combo = QComboBox()
        self.Nplc_Combo.addItems(NPLC_CHOICES)
        self.Nplc_Combo.setCurrentText('10')
        self.Autocal_Check = QCheckBox('Autocalibrate the meter first (CAL:ADC?)')
        self.Estimate_Label = QLabel()
        self.Estimate_Label.setWordWrap(True)
        grid = Form(self.Params_Group, (
            ('From', self.From_Box), ('To', self.To_Box), ('Step', self.Step_Box),
            ('Fine grid across 0 V', self.Fine_Span_Box), ('Fine step', self.Fine_Step_Box),
            ('Order', self.Direction_Combo), ('Random seed', self.Seed_Box),
            ('0 V check every', self.Zero_Box), ('Settle after each step', self.Settle_Box),
            ('Readings per point', self.Readings_Box), ('NPLC', self.Nplc_Combo)))
        grid.addWidget(self.Autocal_Check, grid.rowCount(), 0, 1, 2)
        grid.addWidget(self.Estimate_Label, grid.rowCount(), 0, 1, 2)
        for box in (self.From_Box, self.To_Box, self.Step_Box, self.Fine_Span_Box,
                    self.Fine_Step_Box, self.Zero_Box, self.Settle_Box, self.Readings_Box):
            box.valueChanged.connect(self._Estimate)
        for combo in (self.Direction_Combo, self.Nplc_Combo):
            combo.currentIndexChanged.connect(self._Estimate)
        self.Direction_Combo.currentIndexChanged.connect(
            lambda: self.Seed_Box.setEnabled(self.Direction_Combo.currentIndex() == 3))
        self.Seed_Box.setEnabled(False)

        self.Correction_Group = QGroupBox('DAC correction')
        self.Valid_Box = Spin(1, 1, 365, 0, ' days')
        self.Use_Button = QPushButton('Use as DAC correction', enabled=False)
        self.Clear_Button = QPushButton('Clear DAC correction')
        form = Form(self.Correction_Group, (('Valid for', self.Valid_Box),))
        form.addWidget(self.Use_Button, 1, 0, 1, 2)
        form.addWidget(self.Clear_Button, 2, 0, 1, 2)
        self._Control_Column.insertWidget(1, self.Correction_Group)
        # per branch, its points on the difference plot and on the residual plot
        self._Scatters = {channel: [pg.ScatterPlotItem(size=5, pen=None,
                                                       brush=pg.mkBrush(colour))
                                    for _ in range(2)]
                          for channel, colour in BRANCH_COLOURS.items()}
        for difference, residual in self._Scatters.values():
            self.Difference_Plot.addItem(difference)
            self.Residual_Plot.addItem(residual)
        self._Zeros = pg.ScatterPlotItem(size=7, pen=None, symbol='s',
                                         brush=pg.mkBrush(ZERO_COLOUR))
        self.Difference_Plot.addItem(self._Zeros)
        self._Points = []
        self._Estimate()

    def Inputs(self):
        return [('from', self.From_Box), ('to', self.To_Box), ('step', self.Step_Box),
                ('fine_span', self.Fine_Span_Box), ('fine_step', self.Fine_Step_Box),
                ('order', self.Direction_Combo), ('seed', self.Seed_Box),
                ('zero_every', self.Zero_Box), ('settle', self.Settle_Box),
                ('readings', self.Readings_Box), ('nplc', self.Nplc_Combo),
                ('autocal', self.Autocal_Check), ('valid_days', self.Valid_Box)]

    def _Rungs(self, params):
        """The ladder alone: the ramps between its rungs are laid out by the
        measurement, since a 1 V sweep's take a third of a second."""
        return hc.Linearity_Rungs(params['start'], params['stop'], params['step_v'],
                                  params['direction'], params['fine_span_v'],
                                  params['fine_step_v'], params['seed'])

    def _Estimate(self):
        try:
            params = self.Params()
        except ValueError as e:
            self.Estimate_Label.setText(html.escape(str(e)))
            return
        marks = hc.Linearity_Marks(self._Rungs(params), params['zero_every'])
        # each reading integrates its NPLC, and autozero doubles it
        point_s = (self.Settle_Box.value() + 2 * self.Readings_Box.value()
                   * float(self.Nplc_Combo.currentText()) / hc.POWER_LINE_HZ)
        self.Estimate_Label.setText(f"{marks} points with the 0 V checks, about "
                                    f"{Duration_Text(marks * point_s)}")

    def Params(self):
        params = {'start': self.From_Box.value(), 'stop': self.To_Box.value(),
                  'step_v': self.Step_Box.value(),
                  'direction': self.Direction_Combo.currentText(),
                  'settle_s': self.Settle_Box.value(), 'readings': self.Readings_Box.value(),
                  'nplc': float(self.Nplc_Combo.currentText()), 'ramp': self._Ramp(),
                  'fine_span_v': self.Fine_Span_Box.value(),
                  'fine_step_v': self.Fine_Step_Box.value(),
                  'zero_every': self.Zero_Box.value(), 'seed': self.Seed_Box.value(),
                  'autocal': self.Autocal_Check.isChecked()}
        # a ladder that cannot be laid out is refused here, not on the worker
        self._Rungs(params)
        return params

    def Max_Volts(self, params):
        return max(abs(params['start']), abs(params['stop']))

    def Results(self, result):
        lines = {'Gain': f"{result['Gain']:.7f} +- {result['Gain_Err']:.1e}",
                 'OffsetV': f"{result['Offset_V']:+.3f} +- {result['Offset_Err_V']:.3f}",
                 'ResidualRmsV': f"{result['Residual_Rms_V']:.3f}",
                 'ResidualMaxV': f"{result['Residual_Max_V']:.3f}",
                 'InlPpmFs': f"{result['Inl_Ppm_Fs']:.0f}",
                 'QuadraticVPerKv2': result['Quadratic_V_Per_Kv2'],
                 'Fits': result['Fits'], 'SeamV': result['Seam_V'],
                 'GainMismatchPpm': result['Gain_Mismatch_Ppm'],
                 'Hysteresis': result['Hysteresis'], 'ZeroDrift': result['Zero_Drift']}
        if 'Dac' in result:             # a partial result never heard the 'Calibrate' reply
            lines['DacCorrection'] = result['Dac']['Dac_Correction'][1]
        for channel, name in hc.BRANCHES:
            table = (result['Inl'] or {}).get(channel)
            if table is not None:
                lines[f"Inl_{name}"] = {'knots_v': [round(v, 3) for v in table['Knots_V']],
                                        'error_v': [round(v, 4) for v in table['Error_V']]}
        return {'linearity': lines}

    def Partial(self):
        try:
            return hc.Linearity_Analysis(self._Points)
        except hc.Calibration_Error:
            return None

    def Show_Record(self, path):
        points = hc.Read_Linearity(path)
        result = hc.Linearity_Analysis(points)
        # all the points at once, drawn once: a 1 V sweep holds ten thousand
        self._Replay([], path)
        self._Points = points[:-1]
        self.Show_Progress(points[-1])
        self.Show_Result(result, [path])
        # the correction in force when it was swept is not in the record, so its
        # own cannot be composed onto it
        self.Use_Button.setEnabled(False)
        self.Add_Note('Opened from its record', [Muted(
            "The correction in force when it was swept is not recorded, so sweep again to "
            "put a correction in force.")])

    def Begin(self):
        super().Begin()
        self._Points = []
        for scatter in (s for pair in self._Scatters.values() for s in pair):
            scatter.clear()
        self._Zeros.clear()
        self.Fit_Label.setText('')
        self.Use_Button.setEnabled(False)

    def Show_Progress(self, point):
        self._Points.append(point)
        sweep = [p for p in self._Points if p['Kind'] == 'sweep']
        zeros = [p for p in self._Points if p['Kind'] == 'zero']
        self._Zeros.setData([p['Set_V'] for p in zeros], [p['Read_V'] for p in zeros])
        text = (f"last: set {point['Set_V']:.1f} V  read {point['Read_V']:.3f} V  "
                f"σ {point['Sigma_V']:.3f} V")
        if not sweep:
            self.Fit_Label.setText(text)
            return
        set_v, read_v, channel = (np.array([p[key] for p in sweep])
                                  for key in ('Set_V', 'Read_V', 'Channel'))
        fitted = read_v
        if len(np.unique(set_v)) >= 3:
            gain, offset, rms = Line_Fit(set_v, read_v)
            fitted = gain * set_v + offset
            text = (f"running fit ({len(set_v)} points): gain {gain:.6f}, offset "
                    f"{offset:.3f} V, residual rms {rms:.3f} V\n" + text)
        for branch, (difference, residual) in self._Scatters.items():
            side = channel == branch
            difference.setData(set_v[side], (read_v - set_v)[side])
            residual.setData(set_v[side], (read_v - fitted)[side])
        self.Fit_Label.setText(text)

    def Show_Result(self, result, paths):
        fits = result['Fits']
        line = [('Gain', f"{result['Gain']:.7f} ± {result['Gain_Err'] * 1e6:.3g} ppm"),
                ('Offset', f"{Figure(result['Offset_V'], '+.3f')} ± "
                           f"{Figure(result['Offset_Err_V'], '.3f', 'V')}"),
                ('Residual rms', Figure(result['Residual_Rms_V'], '.3f', 'V')),
                ('Largest residual', f"{Figure(result['Residual_Max_V'], '.3f', 'V')} "
                                     f"({result['Inl_Ppm_Fs']:.0f} ppm of full scale)")]
        if result['Quadratic_V_Per_Kv2'] is not None:
            line.append(('Curvature', Figure(result['Quadratic_V_Per_Kv2'], '+.3f', 'V/kV²')))
        branches = [Report_Table(('Branch', 'Gain', 'Offset (V)', 'RMS (V)', 'Points'), [
            (name.capitalize(), f"{fits[channel]['Gain']:.6f}",
             Figure(fits[channel]['Offset_V'], '+.3f'), f"{fits[channel]['Rms_V']:.3f}",
             fits[channel]['Points']) for channel, name in hc.BRANCHES if channel in fits])]
        if result['Seam_V'] is not None:
            branches += [('Lines meet 0 V', f"{Figure(result['Seam_V'], '+.3f', 'V')} apart"),
                         ('Gain mismatch', Figure(result['Gain_Mismatch_Ppm'], '+.0f', 'ppm'))]
        sections = [('Straight-line fit', line), ('Branches', branches)]
        if result['Hysteresis'] is not None:
            sections.append(('Hysteresis, down − up', [
                ('Mean', Figure(result['Hysteresis']['Mean_V'], '+.3f', 'V')),
                ('Largest', Figure(result['Hysteresis']['Max_V'], '.3f', 'V'))]))
        if result['Zero_Drift'] is not None:
            drift = result['Zero_Drift']
            sections.append(('0 V checks', [
                ('Checks', drift['Checks']), ('Span', Figure(drift['Span_V'], '.3f', 'V')),
                *((('Drift', Figure(drift['Slope_V_Per_H'], '+.3f', 'V/h')),)
                  if drift['Slope_V_Per_H'] is not None else ())]))
        if result['Inl']:
            sections.append(('Non-linearity table', [
                Report_Table(('Branch', 'Shape (V rms)', 'Knots', 'Left (V rms)'), [
                    (name.capitalize(), f"{table['Smooth_Rms_V']:.3f}", len(table['Knots_V']),
                     f"{table['Rough_Rms_V']:.3f}")
                    for channel, name in hc.BRANCHES
                    if (table := result['Inl'].get(channel)) is not None]),
                Muted(f"What is left is the code rounding ({hc.CODE_V:.3f} V a code) and "
                      f"noise.")]))
        else:
            sections.append(('Non-linearity table', [
                f"None: it needs {hc.INL_MIN_POINTS} rungs in every {hc.INL_KNOT_V:g} V of both "
                f"branches."]))
        super().Show_Result(result, paths, sections)
        self.Use_Button.setEnabled(len(fits) == len(hc.BRANCHES))

    def Show_Correction(self, verdict, message, corrects):
        """The DAC correction just put in force (or cleared), and the gate's
        verdict on it, under the report."""
        self.Add_Note((verdict, 'DAC correction'), [
            ('Corrects', corrects), ('State', 'in force now, saved in the settings'),
            message[0].upper() + message[1:] + ('' if message.endswith('.') else '.')])


class Cooler_Panel(_Panel):

    PEAK_COLUMNS = ('Hz', 'amplitude V', 'SNR', 'mains', 'meter gain')
    RECORDS = {'cooler': hc.COOLER_COLUMNS, 'cooler_spectrum': hc.SPECTRUM_COLUMNS}

    def __init__(self, parent=None):
        self.Voltage_Plot = plots.Plot('Cooler platform voltage',
                                       ('Time since the start of the record', 's'),
                                       ('Deviation from mean', 'V'), (
            "The cooler (RFQ cooler-buncher) platform voltage over the record, relative to its "
            "mean, showing drift, ripple and noise.",
            "The cooler meter reads the platform through its 10 000:1 divider at the chosen "
            "interval and aperture; the record's mean is subtracted for display. The mean, "
            "drift and noise figures are listed below the plots."))
        self.Asd_Plot = plots.Plot('Amplitude spectral density (ASD) of the cooler voltage',
                                   ('Frequency', 'Hz'), ('ASD (V/√Hz)', None), (
            "The cooler voltage's noise by frequency, on logarithmic axes. Tones such as mains "
            "harmonics stand above the noise floor, and those found are listed in the table.",
            "The record less its straight-line drift is cut into Hann-windowed segments, half "
            "overlapping; their power spectra are averaged (Welch's method) and the square root "
            "taken. Tones are found in the whole record's spectrum, against its local median."))
        self.Asd_Plot.setLogMode(x=True, y=True)
        self.Peak_Table = _Table(self.PEAK_COLUMNS)
        super().__init__('Cooler monitor', ((self.Voltage_Plot, 3), (self.Asd_Plot, 3),
                                            (self.Peak_Table, 2)), parent)
        self.Duration_Box = Spin(60, 1, 86400, 0, ' s')
        self.Interval_Box = Spin(1.0, 0.1, 10000.0, 1, ' ms')
        self.Integration_Combo = QComboBox()
        self.Integration_Combo.addItems(('Aperture: sees ripple and oscillations',
                                         'NPLC: rejects mains, for slow drift'))
        self.Aperture_Box = Spin(0.2, 0.001, 1000.0, 3, ' ms')
        self.Nplc_Combo = QComboBox()
        self.Nplc_Combo.addItems(TAIL_NPLC_CHOICES)
        self.Nplc_Combo.setCurrentText('1')
        self.Estimate_Label = QLabel()
        grid = Form(self.Params_Group, (
            ('Duration', self.Duration_Box), ('Sample interval', self.Interval_Box),
            ('Integration', self.Integration_Combo), ('Aperture', self.Aperture_Box),
            ('NPLC', self.Nplc_Combo)))
        grid.addWidget(self.Estimate_Label, grid.rowCount(), 0, 1, 2)
        for box in (self.Duration_Box, self.Interval_Box):
            box.valueChanged.connect(self._Estimate)
        self.Integration_Combo.currentIndexChanged.connect(self._Integration_Changed)
        self._Curve = plots.Dense_Curve(pen=plots.Pen())
        self.Voltage_Plot.addItem(self._Curve)
        self._Asd_Curve = self.Asd_Plot.plot(pen=plots.Pen())
        self._Times, self._Volts = [], []
        # the record drawn once all the chunks waiting have been added, not once
        # per chunk: each draw takes the whole record, so a long monitor drawn per
        # chunk falls behind its meter and its chunks queue up unread
        self._Redraw = QTimer(self, singleShot=True, interval=0, timeout=self._Draw)
        self._Integration_Changed()
        self._Estimate()

    def _Integration_Changed(self):
        aperture = self.Integration_Combo.currentIndex() == 0
        self.Aperture_Box.setEnabled(aperture)
        self.Nplc_Combo.setEnabled(not aperture)

    def Inputs(self):
        return [('duration', self.Duration_Box), ('interval', self.Interval_Box),
                ('integration', self.Integration_Combo), ('aperture', self.Aperture_Box),
                ('nplc', self.Nplc_Combo)]

    def _Estimate(self):
        count = int(self.Duration_Box.value() / (self.Interval_Box.value() / 1e3))
        size = count * CSV_BYTES_PER_READING
        size = f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} kB"
        self.Estimate_Label.setText(f"{count} readings, the CSV about {size}")

    def Params(self):
        interval_ms, aperture_ms = self.Interval_Box.value(), self.Aperture_Box.value()
        nplc = (None if self.Integration_Combo.currentIndex() == 0
                else float(self.Nplc_Combo.currentText()))
        if nplc is None and aperture_ms >= interval_ms:
            raise ValueError(f"An aperture of {aperture_ms:g} ms does not fit a sample "
                             f"interval of {interval_ms:g} ms.")
        return {'duration_s': self.Duration_Box.value(), 'interval_s': interval_ms / 1e3,
                'aperture_s': aperture_ms / 1e3, 'nplc': nplc}

    def Results(self, result):
        keys = ('Count', 'Interval_S', 'Mean_V', 'Sigma_V', 'Peak_To_Peak_V', 'Drift_V_Per_Min',
                'Rms_Detrended_V', 'Oscillation_Rms_V', 'Random_Rms_V', 'Noise_Floor_V_Rt_Hz')
        lines = {key: result[key] for key in keys}
        lines['Peaks'] = [{k: round(v, 5) for k, v in peak.items()} for peak in result['Peaks']]
        lines['Bands'] = [{k: round(v, 5) for k, v in band.items()} for band in result['Bands']]
        return {'cooler': lines}

    def Rows(self, result):
        return {'cooler_spectrum': (result['Frequency_Hz'], result['Asd'])}

    def Show_Record(self, path):
        times, volts, integration = hc.Read_Cooler(path)
        if len(times) < hc.MIN_SPECTRUM_READINGS:
            raise hc.Calibration_Error(f"{path} holds {len(times)} readings; a spectrum needs "
                                       f"{hc.MIN_SPECTRUM_READINGS}.")
        result = hc.Noise_Analysis(volts, times[1] - times[0], integration)
        self._Replay([(times, volts)], path)
        self.Show_Result(dict(result, Times_S=times, Volts=volts), [path])

    def Partial(self):
        times = np.concatenate(self._Times) if self._Times else np.zeros(0)
        if len(times) < hc.MIN_SPECTRUM_READINGS:
            return None
        params = self.Params()
        integration = (params['nplc'] / hc.POWER_LINE_HZ if params['nplc']
                       else params['aperture_s'])
        return hc.Noise_Analysis(np.concatenate(self._Volts), times[1] - times[0], integration)

    def Begin(self):
        super().Begin()
        self._Redraw.stop()
        self._Times, self._Volts = [], []
        self._Curve.setData([], [])
        self._Asd_Curve.setData([], [])
        self.Peak_Table.setRowCount(0)

    def Show_Progress(self, chunk):
        times, volts = chunk
        self._Times.append(times)
        self._Volts.append(volts)
        if not self._Redraw.isActive():
            self._Redraw.start()

    def _Draw(self):
        volts = np.concatenate(self._Volts)
        self._Curve.setData(np.concatenate(self._Times), volts - volts.mean())

    def Show_Result(self, result, paths):
        if self._Redraw.isActive():
            self._Redraw.stop()
            self._Draw()                # the whole record as it ends, not a pass later
        self._Asd_Curve.setData(result['Frequency_Hz'], result['Asd'])
        for peak in result['Peaks']:
            _Add_Row(self.Peak_Table, (f"{peak['Freq_Hz']:.3f}", f"{peak['Amplitude_V']:.4f}",
                                       f"{peak['Snr']:.1f}",
                                       peak['Mains_Harmonic'] or '–',
                                       f"{peak['Integration_Gain']:.3f}"))
        interval = result['Interval_S']
        sections = [
            ('Record', [('Readings', f"{result['Count']}, every {interval * 1e3:.3g} ms"),
                        ('Duration', Duration_Text(result['Count'] * interval))]),
            ('Level', [('Mean', Figure(result['Mean_V'], '.2f', 'V')),
                       ('Drift', Figure(result['Drift_V_Per_Min'], '+.4f', 'V/min'))]),
            ('About the drift', [
                ('Total', Figure(result['Rms_Detrended_V'], '.3f', 'V rms')),
                ('Oscillations', Figure(result['Oscillation_Rms_V'], '.3f', 'V rms')),
                ('Random', Figure(result['Random_Rms_V'], '.3f', 'V rms')),
                ('Peak to peak', Figure(result['Peak_To_Peak_Detrended_V'], '.3f', 'V')),
                ('Noise floor', Figure(result['Noise_Floor_V_Rt_Hz'] * 1e3, '.2f', 'mV/√Hz'))]),
            ('By frequency band', [(f"{band['Low_Hz']:.3g}–{band['High_Hz']:.3g} Hz",
                                    Figure(band['Rms_V'], '.3f', 'V rms'))
                                   for band in result['Bands']])]
        super().Show_Result(result, paths, sections)


class Hardware_Tab(QWidget):

    def __init__(self, limit_v, parent=None):
        super().__init__(parent)
        self.Ramp_Group = QGroupBox('Ramp between points')
        self.Ramp_Rate_Box = Spin(2000, 1, 100000, 0, ' V/s')
        self.Ramp_Step_Box = Spin(50.0, 0.1, limit_v, 1, ' V')
        row = QHBoxLayout(self.Ramp_Group)
        for text, widget in (('Rate', self.Ramp_Rate_Box), ('Step', self.Ramp_Step_Box)):
            row.addWidget(QLabel(text))
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(QLabel(f"Every setpoint is held to ±{limit_v:g} V (settings: "
                             f"instruments, hv_limit_v)."))
        self.Read_Group = QGroupBox('Line meter')
        self.Read_Button = QPushButton(f"Read ({hc.PRECISE_READINGS} × NPLC "
                                       f"{hc.PRECISE_NPLC:g})")
        self.Read_Button.setToolTip("Read the platform precisely, to check the wiring and "
                                    "polarity against what quick tune holds on it.")
        self.Read_Label = QLabel('–')
        read = QHBoxLayout(self.Read_Group)
        read.addWidget(self.Read_Button)
        read.addWidget(self.Read_Label, 1)
        header = QHBoxLayout()
        header.addWidget(self.Ramp_Group, 1)
        header.addWidget(self.Read_Group, 1)
        self.Settle = Settle_Panel(self.Ramp, limit_v)
        self.Linearity = Linearity_Panel(self.Ramp, limit_v)
        self.Cooler = Cooler_Panel()
        self.Panels = (self.Settle, self.Linearity, self.Cooler)
        self.Pages = QTabWidget()
        for page, name in zip(self.Panels, ('Settle time', 'Linearity', 'Cooler monitor')):
            self.Pages.addTab(page, name)
        column = QVBoxLayout(self)
        column.addLayout(header)
        column.addWidget(self.Pages, 1)
        self.Settle.Wait_Button.clicked.connect(self._Use_Wait)

    def Ramp(self):
        """``(step V, rate V/s)`` of every unmeasured move."""
        return self.Ramp_Step_Box.value(), self.Ramp_Rate_Box.value()

    def Inputs(self):
        """Every input of the ramp and the pages, each page's under its own name."""
        return [('ramp_rate', self.Ramp_Rate_Box), ('ramp_step', self.Ramp_Step_Box),
                *((f"{page}.{name}", widget) for page, panel in (
                    ('settle', self.Settle), ('linearity', self.Linearity),
                    ('cooler', self.Cooler)) for name, widget in panel.Inputs())]

    def _Use_Wait(self):
        """The settle test's recommended wait into the linearity sweep."""
        self.Linearity.Settle_Box.setValue(hc.Recommended_Wait(self.Settle.Result[0]))
        self.Pages.setCurrentWidget(self.Linearity)

    def Show_Reading(self, mean_v, sigma_v, setpoint_v):
        self.Read_Label.setText(f"{mean_v:+.3f} V, σ {sigma_v:.3f} V; readback − setpoint "
                                f"{mean_v - setpoint_v:+.3f} V (setpoint {setpoint_v:g} V)")
