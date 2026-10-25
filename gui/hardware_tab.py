"""The Hardware Calibration tab: settle time, linearity and the cooler monitor.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The bench calibration tool's measurements, each on a page of its own in
the Scan Calibration tab's layout: the parameters and the start and stop
buttons on the left, the result under them, and the plots filling as the
measurement runs. The ramp between measured points, and the limit every
setpoint is held to, sit above the pages and apply to all of them.

Each page reads its own widgets into its measurement's parameters
(``Params``, which refuses what cannot run with a sentence) and turns a
result into the records saved beside the run files (``Records``); the
main window runs every page's measurement the same way, off the GUI
thread, keeping everything that could drive the DAC or the meters
disabled until it has finished.

Depends on: numpy, pyqtgraph, PySide6; cls_acquisition.calibration,
cls_acquisition.hardware_calibration, gui.plots, gui.shared_widgets,
gui.theme.
"""

import html

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from cls_acquisition import hardware_calibration as hc
from cls_acquisition.calibration import SETTLE_MARGIN, THRESHOLDS, Line_Fit
from cls_acquisition.hardware.mcc_dac import CH_NEG, CH_POS
from gui import plots, theme
from gui.shared_widgets import REPORT_WIDTH, Form, Spin, Verdict_Html

STEP_ROWS = 4
NPLC_CHOICES = ('0.02', '0.2', '1', '10', '100')
BRANCH_COLOURS = {CH_POS: theme.TRACE, CH_NEG: theme.TRACE_ALT}
MODE_COLOURS = {'out': theme.TRACE, 'back': theme.TRACE_ALT}
CSV_BYTES_PER_READING = 30
# a settle result's record: (CSV column, result key)
SETTLE_RECORD = (('step', 'Step'), ('mode', 'Mode'), ('rep', 'Rep'), ('from_v', 'From_V'),
                 ('to_v', 'To_V'), ('lag_s', 'Lag_S'), ('settle_s', 'Settle_S'),
                 ('rise_s', 'Rise_S'), ('overshoot_pct', 'Overshoot_Pct'),
                 ('noise_v', 'Noise_V'), ('band_v', 'Band_V'))


class _Panel(QWidget):
    """One measurement's page: parameters and control down the left, plots
    on the right. ``Result`` is the last ``(result, record path)``."""

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
        buttons.addWidget(self.Start_Button)
        buttons.addWidget(self.Stop_Button)
        column.addLayout(buttons)
        self.Report_Label = QLabel('Not run yet in this session.')
        self.Report_Label.setWordWrap(True)
        column.addWidget(self.Report_Label)
        column.addStretch(1)
        self._Control_Column = column

        plot_group = QGroupBox('Plots')
        plot_column = QVBoxLayout(plot_group)
        for widget, stretch in plot_widgets:
            plot_column.addWidget(widget, stretch)
        layout = QGridLayout(self)
        layout.addWidget(self.Params_Group, 0, 0)
        layout.addWidget(self.Control_Group, 1, 0)
        layout.addWidget(plot_group, 0, 1, 2, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)

    def Begin(self):
        self.Report_Label.setText("Measuring...")
        self.Stop_Button.setEnabled(True)

    def Running(self, running):
        self.Stop_Button.setEnabled(running)

    def Show_Result(self, result, path, text):
        self.Result = (result, path)
        self.Report_Label.setText(f"{text}<br>Saved to {html.escape(path)}")

    def Show_Failure(self, message):
        self.Report_Label.setText(Verdict_Html('FAIL', message))


class Settle_Panel(_Panel):

    COLUMNS = ('step', 'mode', 'rep', 'from V', 'to V', 'lag ms', 'settle ms',
               'rise 10-90 ms', 'overshoot %', 'noise V')

    def __init__(self, ramp, parent=None):
        self.Response_Plot = plots.Plot('Step response (time from 10 % crossing)',
                                        ('Time', 's'), ('HV', 'V'))
        self.Deviation_Plot = plots.Plot('Deviation from final value', ('Time', 's'),
                                         ('HV - final', 'V'))
        self.Table = QTableWidget(0, len(self.COLUMNS))
        self.Table.setHorizontalHeaderLabels(self.COLUMNS)
        self.Table.verticalHeader().setVisible(False)
        super().__init__('Settle test', ((self.Response_Plot, 3), (self.Deviation_Plot, 3),
                                         (self.Table, 2)), parent)
        self._Ramp = ramp
        self.Step_Checks, self.From_Boxes, self.To_Boxes = [], [], []
        grid = QGridLayout(self.Params_Group)
        grid.addWidget(QLabel('From (V)'), 0, 1)
        grid.addWidget(QLabel('To (V)'), 0, 2)
        for i in range(STEP_ROWS):
            check = QCheckBox(checked=i == 0)
            start, stop = Spin(0.0, -6000, 6000, 1), Spin(1000.0 * (i + 1), -6000, 6000, 1)
            for column, widget in enumerate((check, start, stop)):
                grid.addWidget(widget, 1 + i, column)
            self.Step_Checks.append(check)
            self.From_Boxes.append(start)
            self.To_Boxes.append(stop)
        self.Reps_Box = Spin(3, 1, 100, 0)
        self.Window_Box = Spin(500, 10, 20000, 0, ' ms')
        self.Interval_Box = Spin(0.5, 0.02, 100.0, 2, ' ms')
        self.Aperture_Box = Spin(200, 1, 100000, 0, ' µs')
        self.Hold_Box = Spin(1.0, 0.0, 600.0, 2, ' s')
        self.Tolerance_Box = Spin(0.5, 0.001, 100.0, 3, ' V')
        for row, (text, widget) in enumerate(
                (('Repetitions', self.Reps_Box), ('Capture window', self.Window_Box),
                 ('Sample interval', self.Interval_Box), ('Aperture', self.Aperture_Box),
                 ('Hold before each step', self.Hold_Box),
                 ('Settle tolerance', self.Tolerance_Box)), 1 + STEP_ROWS):
            grid.addWidget(QLabel(text), row, 0, 1, 2)
            grid.addWidget(widget, row, 2)

        self.Model_Group = QGroupBox('Scan settle model')
        model = QVBoxLayout(self.Model_Group)
        self.Model_Label = QLabel()
        self.Model_Label.setWordWrap(True)
        self.Use_Button = QPushButton('Use as scan settle model', enabled=False)
        self.Clear_Button = QPushButton('Clear scan settle model')
        for widget in (self.Model_Label, self.Use_Button, self.Clear_Button):
            model.addWidget(widget)
        self._Control_Column.insertWidget(1, self.Model_Group)

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
        interval_ms, aperture_us = self.Interval_Box.value(), self.Aperture_Box.value()
        if aperture_us / 1e3 >= interval_ms:
            raise ValueError(f"An aperture of {aperture_us:g} µs does not fit a sample "
                             f"interval of {interval_ms:g} ms.")
        return {'steps': steps, 'reps': self.Reps_Box.value(),
                'window_s': self.Window_Box.value() / 1e3, 'interval_s': interval_ms / 1e3,
                'aperture_s': aperture_us / 1e6, 'hold_s': self.Hold_Box.value(),
                'tolerance_v': self.Tolerance_Box.value(), 'ramp': self._Ramp()}

    def Records(self, results, params):
        """The summary, and every step's trace in long form."""
        verdict, sentence = hc.Settle_Summary(results)
        columns, keys = zip(*SETTLE_RECORD)
        records = [('settle', {**params, 'Verdict': f"{verdict}: {sentence}"}, columns,
                    [[r[key] for r in results] for key in keys])]
        if results:
            lengths = [len(r['Times_S']) for r in results]
            records.append(
                ('settle_traces', params,
                 ('step', 'mode', 'rep', 'time_from_10pct_s', 'deviation_v'),
                 [np.repeat([r[key] for r in results], lengths) for key in ('Step', 'Mode', 'Rep')]
                 + [np.concatenate([r[key] for r in results]) for key in ('Times_S',
                                                                          'Deviation_V')]))
        return records

    def Begin(self):
        super().Begin()
        for plot in (self.Response_Plot, self.Deviation_Plot):
            plot.clear()
        self.Table.setRowCount(0)
        self.Use_Button.setEnabled(False)

    def Show_Progress(self, result):
        pen = plots.Pen(MODE_COLOURS[result['Mode']])
        times = result['Times_S']
        self.Response_Plot.plot(times, result['Deviation_V'] + result['Final_V'], pen=pen)
        self.Deviation_Plot.plot(times, result['Deviation_V'], pen=pen)
        row = self.Table.rowCount()
        self.Table.insertRow(row)
        cells = (result['Step'], result['Mode'], result['Rep'], f"{result['From_V']:g}",
                 f"{result['To_V']:g}", f"{result['Lag_S'] * 1e3:.2f}",
                 f"{result['Settle_S'] * 1e3:.2f}", f"{result['Rise_S'] * 1e3:.2f}",
                 f"{result['Overshoot_Pct']:.2f}", f"{result['Noise_V']:.3f}")
        for column, value in enumerate(cells):
            self.Table.setItem(row, column, QTableWidgetItem(str(value)))

    def Show_Result(self, results, path):
        super().Show_Result(results, path, Verdict_Html(*hc.Settle_Summary(results)))
        self.Use_Button.setEnabled(bool(results))


class Linearity_Panel(_Panel):

    def __init__(self, ramp, parent=None):
        set_axis = ('Setpoint', 'V')
        self.Difference_Plot = plots.Plot('Readback - setpoint', set_axis, ('HV - set', 'V'))
        self.Residual_Plot = plots.Plot('Residual from linear fit', set_axis,
                                        ('Residual', 'V'))
        self.Fit_Label = QLabel()
        self.Fit_Label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        super().__init__('Linearity sweep', ((self.Difference_Plot, 1),
                                             (self.Residual_Plot, 1), (self.Fit_Label, 0)),
                         parent)
        self._Ramp = ramp
        self.From_Box = Spin(-3000.0, -6000, 6000, 1, ' V')
        self.To_Box = Spin(3000.0, -6000, 6000, 1, ' V')
        self.Step_Box = Spin(50.0, 0.1, 6000, 1, ' V')
        self.Direction_Combo = QComboBox()
        self.Direction_Combo.addItems(hc.DIRECTIONS)
        self.Direction_Combo.setCurrentIndex(2)
        self.Settle_Box = Spin(0.5, 0.0, 600.0, 2, ' s')
        self.Readings_Box = Spin(5, 1, 1000, 0)
        self.Nplc_Combo = QComboBox()
        self.Nplc_Combo.addItems(NPLC_CHOICES)
        self.Nplc_Combo.setCurrentText('10')
        self.Estimate_Label = QLabel()
        grid = Form(self.Params_Group, (
            ('From', self.From_Box), ('To', self.To_Box), ('Step', self.Step_Box),
            ('Direction', self.Direction_Combo), ('Settle after each step', self.Settle_Box),
            ('Readings per point', self.Readings_Box), ('NPLC', self.Nplc_Combo)))
        grid.addWidget(self.Estimate_Label, grid.rowCount(), 0, 1, 2)
        for box in (self.From_Box, self.To_Box, self.Step_Box, self.Settle_Box,
                    self.Readings_Box):
            box.valueChanged.connect(self._Estimate)
        for combo in (self.Direction_Combo, self.Nplc_Combo):
            combo.currentIndexChanged.connect(self._Estimate)

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
        self._Points = []
        self._Estimate()

    def _Estimate(self):
        count = max(int(round(abs(self.To_Box.value() - self.From_Box.value())
                              / self.Step_Box.value())) + 1, 2)
        if self.Direction_Combo.currentText() == hc.DIRECTIONS[2]:
            count = 2 * count - 1
        # each reading integrates its NPLC, and autozero doubles it
        point_s = (self.Settle_Box.value() + 2 * self.Readings_Box.value()
                   * float(self.Nplc_Combo.currentText()) / hc.POWER_LINE_HZ)
        self.Estimate_Label.setText(f"{count} points, about {count * point_s / 60:.1f} min")

    def Params(self):
        params = {'start': self.From_Box.value(), 'stop': self.To_Box.value(),
                  'step_v': self.Step_Box.value(),
                  'direction': self.Direction_Combo.currentText(),
                  'settle_s': self.Settle_Box.value(), 'readings': self.Readings_Box.value(),
                  'nplc': float(self.Nplc_Combo.currentText()), 'ramp': self._Ramp()}
        # a ladder that cannot be laid out is refused here, not on the worker
        hc.Linearity_Plan(params['start'], params['stop'], params['step_v'],
                          params['direction'], params['ramp'][0])
        return params

    def Records(self, result, params):
        header = {**params, 'DacCorrection': result['Dac']['Dac_Correction'][1],
                  'Fits': result['Fits'], 'SeamV': result['Seam_V'],
                  'HysteresisV': result['Hysteresis_V']}
        return [('linearity', header, ('set_v', 'read_v', 'sigma_v'),
                 [result['Set_V'], result['Read_V'], result['Sigma_V']])]

    def Begin(self):
        super().Begin()
        self._Points = []
        for scatter in (s for pair in self._Scatters.values() for s in pair):
            scatter.clear()
        self.Fit_Label.setText('')
        self.Use_Button.setEnabled(False)

    def Show_Progress(self, point):
        self._Points.append(point)
        set_v, read_v, channel = (np.array([p[key] for p in self._Points])
                                  for key in ('Set_V', 'Read_V', 'Channel'))
        text = (f"last: set {point['Set_V']:.1f} V  read {point['Read_V']:.3f} V  "
                f"σ {point['Sigma_V']:.3f} V")
        fitted = read_v
        if len(set_v) >= 3:
            gain, offset, rms = Line_Fit(set_v, read_v)
            fitted = gain * set_v + offset
            text = (f"running fit ({len(set_v)} points): gain {gain:.6f}, offset "
                    f"{offset:.3f} V, residual rms {rms:.3f} V\n" + text)
        for branch, (difference, residual) in self._Scatters.items():
            side = channel == branch
            difference.setData(set_v[side], (read_v - set_v)[side])
            residual.setData(set_v[side], (read_v - fitted)[side])
        self.Fit_Label.setText(text)

    def Show_Result(self, result, path):
        fits, lines = result['Fits'], []
        for channel, name in hc.BRANCHES:
            if channel in fits:
                fit = fits[channel]
                lines.append(f"{name}: gain {fit['Gain']:.6f}, offset {fit['Offset_V']:.3f} V, "
                             f"rms {fit['Rms_V']:.3f} V over {fit['Points']} points")
        if result['Seam_V'] is not None:
            lines.append(f"seam at 0 V: {result['Seam_V']:+.3f} V")
        if result['Hysteresis_V'] is not None:
            lines.append(f"hysteresis (down - up): {result['Hysteresis_V']:+.3f} V")
        super().Show_Result(result, path, '<br>'.join(map(html.escape, lines)))
        self.Use_Button.setEnabled(len(fits) == len(hc.BRANCHES))


class Cooler_Panel(_Panel):

    def __init__(self, parent=None):
        self.Voltage_Plot = plots.Plot('Cooler voltage', ('Time', 's'), ('Cooler - mean', 'V'))
        self.Asd_Plot = plots.Plot('Amplitude spectral density (detrended)',
                                   ('Frequency', 'Hz'), ('ASD (V/√Hz)', None))
        self.Asd_Plot.setLogMode(x=True, y=True)
        super().__init__('Cooler monitor', ((self.Voltage_Plot, 1), (self.Asd_Plot, 1)),
                         parent)
        self.Duration_Box = Spin(60, 1, 86400, 0, ' s')
        self.Block_Box = Spin(2.0, 0.1, 20.0, 1, ' s')
        self.Interval_Box = Spin(10.0, 0.1, 10000.0, 1, ' ms')
        self.Aperture_Box = Spin(1.0, 0.001, 1000.0, 3, ' ms')
        self.Estimate_Label = QLabel()
        grid = Form(self.Params_Group, (
            ('Duration', self.Duration_Box), ('Block', self.Block_Box),
            ('Sample interval', self.Interval_Box), ('Aperture', self.Aperture_Box)))
        grid.addWidget(self.Estimate_Label, grid.rowCount(), 0, 1, 2)
        for box in (self.Duration_Box, self.Interval_Box):
            box.valueChanged.connect(self._Estimate)
        self._Curve = self.Voltage_Plot.plot(pen=plots.Pen())
        self._Asd_Curve = self.Asd_Plot.plot(pen=plots.Pen())
        self._Times, self._Volts = [], []
        self._Estimate()

    def _Estimate(self):
        count = int(self.Duration_Box.value() / (self.Interval_Box.value() / 1e3))
        size = count * CSV_BYTES_PER_READING
        size = f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} kB"
        self.Estimate_Label.setText(f"{count} readings, the CSV about {size}")

    def Params(self):
        interval_ms, aperture_ms = self.Interval_Box.value(), self.Aperture_Box.value()
        if aperture_ms >= interval_ms:
            raise ValueError(f"An aperture of {aperture_ms:g} ms does not fit a sample "
                             f"interval of {interval_ms:g} ms.")
        return {'duration_s': self.Duration_Box.value(), 'block_s': self.Block_Box.value(),
                'interval_s': interval_ms / 1e3, 'aperture_s': aperture_ms / 1e3}

    def Records(self, result, params):
        header = {**params, **{key: result[key] for key in
                               ('Mean_V', 'Sigma_V', 'Peak_To_Peak_V', 'Drift_V_Per_Min')}}
        return [('cooler', header, ('time_s', 'cooler_v'), [result['Times_S'], result['Volts']])]

    def Begin(self):
        super().Begin()
        self._Times, self._Volts = [], []
        self._Curve.setData([], [])
        self._Asd_Curve.setData([], [])

    def Show_Progress(self, block):
        times, volts = block
        self._Times.append(times)
        self._Volts.append(volts)
        volts = np.concatenate(self._Volts)
        self._Curve.setData(np.concatenate(self._Times), volts - volts.mean())

    def Show_Result(self, result, path):
        self._Asd_Curve.setData(result['Frequency_Hz'], result['Asd'])
        peak = result['Frequency_Hz'][np.argmax(result['Asd'])]
        super().Show_Result(result, path, html.escape(
            f"mean {result['Mean_V']:.2f} V, σ {result['Sigma_V']:.3f} V, peak to peak "
            f"{result['Peak_To_Peak_V']:.3f} V, drift {result['Drift_V_Per_Min']:.3f} V/min; "
            f"strongest line at {peak:.2f} Hz"))


class Hardware_Tab(QWidget):

    def __init__(self, limit_v, parent=None):
        super().__init__(parent)
        self.Ramp_Group = QGroupBox('Ramp between points')
        self.Ramp_Rate_Box = Spin(2000, 1, 100000, 0, ' V/s')
        self.Ramp_Step_Box = Spin(50.0, 0.1, 6000.0, 1, ' V')
        row = QHBoxLayout(self.Ramp_Group)
        for text, widget in (('Rate', self.Ramp_Rate_Box), ('Step', self.Ramp_Step_Box)):
            row.addWidget(QLabel(text))
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(QLabel(f"Every setpoint is held to ±{limit_v:g} V (settings: "
                             f"instruments, hv_limit_v)."))
        self.Settle, self.Linearity = Settle_Panel(self.Ramp), Linearity_Panel(self.Ramp)
        self.Cooler = Cooler_Panel()
        self.Panels = (self.Settle, self.Linearity, self.Cooler)
        self.Pages = QTabWidget()
        for page, name in zip(self.Panels, ('Settle time', 'Linearity', 'Cooler monitor')):
            self.Pages.addTab(page, name)
        column = QVBoxLayout(self)
        column.addWidget(self.Ramp_Group)
        column.addWidget(self.Pages, 1)

    def Ramp(self):
        """``(step V, rate V/s)`` of every unmeasured move."""
        return self.Ramp_Step_Box.value(), self.Ramp_Rate_Box.value()
