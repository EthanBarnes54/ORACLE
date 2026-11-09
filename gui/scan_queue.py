"""The Scan queue: scans laid out ahead and taken one after another, unattended if need be.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A page of the Scan tab's parameters box, laid out as DENIS's optimiser
pages: options, an editable table, a readout. Each row is one scan and
one run file: its ranges (``20:55, 60:80``), step, bunches per step
(seconds in DC mode), order (Sawtooth, or Triangle up and back) and
number of scans; the buncher timing, experiment and view are the
parameters page's, held for the whole queue.

Every row is checked as it is typed, as a scan is when it starts: a bad
cell turns red, and the readout gives each row's fault, or its setpoints
and duration, and when the queue ends. As in DENIS, the table grows a
row as the last fills, takes rows pasted from a spreadsheet, and Delete
clears cells. The main window takes the scans (Automatic, or Manual,
asking before each after the first); the State column records each row,
and rows and states survive a crash, so a restarted queue takes up at the
first scan not done.

Depends on: numpy, PySide6; cls_acquisition.calibration,
cls_acquisition.scan_plan, gui.shared_widgets, gui.theme.
"""

import re
import sys
from datetime import datetime, timedelta

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFormLayout,
                               QHBoxLayout, QHeaderView, QPlainTextEdit, QPushButton,
                               QStyledItemDelegate, QTableWidget, QTableWidgetItem, QToolButton,
                               QVBoxLayout, QWidget)

from cls_acquisition.calibration import THRESHOLDS, Settle_For
from cls_acquisition.scan_plan import Scan_Plan
from gui import theme
from gui.shared_widgets import Duration_Text

RANGES, STEP, PER_STEP, ORDER, SCANS, STATE = range(6)
COLUMNS = ('ranges', 'step', 'per_step', 'order', 'scans', 'state')
HEADERS = ('Ranges (V)', 'Step (V)', 'Bunches', 'Order', 'Scans', 'State')
ORDERS = ('Sawtooth', 'Triangle')
MODES = ('Automatic', 'Manual: ask before each scan')
MOST_RANGES = 7                     # the parameters page's, where each scan is set up
DECIMALS = 2                        # the parameters page's voltage, step and seconds boxes
DONE, CALIBRATING, RUNNING, INTERRUPTED = 'done', 'calibrating', 'running', 'interrupted'
CALIBRATION_READ_S = 0.01           # a calibration point beyond its settling: its reading
MONO_QSS = ("QPlainTextEdit { font-family: " + ("'Menlo', " if sys.platform == 'darwin' else "")
            + "'Consolas', 'Courier New', 'Liberation Mono', monospace; }")


def Parse_Ranges(text):
    """``[[from, to], ...]`` of ranges written from:to, apart by commas or
    semicolons; raises ValueError with a sentence."""
    ranges = []
    for part in re.split(r'[,;]', text):
        if not part.strip():
            continue
        try:
            low, high = (float(bound) for bound in part.split(':'))
        except ValueError:
            raise ValueError(f"'{part.strip()}' is not a range; write each as from:to in volts, "
                             f"apart by commas: 20:55, 60:80.") from None
        ranges.append([low, high])
    if not ranges:
        raise ValueError("No range is given.")
    return ranges


def Ranges_Text(ranges):
    return ', '.join(f"{low:g}:{high:g}" for low, high in ranges)


def _Number(text, what, whole=False, least=None):
    """``text`` as a number the parameters page can hold exactly; raises
    ValueError naming ``what``."""
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{what.capitalize()} '{text.strip()}' is not a number.") from None
    if not np.isfinite(value) or (whole and not value.is_integer()):
        raise ValueError(f"{what.capitalize()} of {value:g} is not a whole number.")
    if round(value, DECIMALS) != value:
        raise ValueError(f"{what.capitalize()} of {value:g} has more than {DECIMALS} "
                         f"decimals, which the scan parameters do not hold.")
    if least is not None and value < least:
        raise ValueError(f"{what.capitalize()} of {value:g} is below {least:g}.")
    return value


def Entry(cells, bunched, limit_v):
    """The scan a row's ``cells`` (its texts, by column) ask for:
    ``{'ranges', 'step', 'per_step', 'order', 'scans', 'plan'}``, checked as
    a scan is as it starts. Raises ValueError with a sentence."""
    ranges = Parse_Ranges(cells[RANGES])
    if len(ranges) > MOST_RANGES:
        raise ValueError(f"It has {len(ranges)} ranges; a scan holds {MOST_RANGES} at most.")
    for value in (v for r in ranges for v in r):
        _Number(str(value), 'a range limit')
    worst = max(abs(v) for r in ranges for v in r)
    if worst > limit_v:
        raise ValueError(f"{worst:g} V is beyond the ±{limit_v:g} V limit on the platform.")
    step = _Number(cells[STEP], 'its step', least=0.01)
    plan = Scan_Plan(*zip(*ranges), step)
    per_step = (_Number(cells[PER_STEP], 'its bunches per step', whole=True, least=1)
                if bunched else _Number(cells[PER_STEP], 'its seconds per step', least=0.01))
    order = cells[ORDER].strip().capitalize()
    if order not in ORDERS:
        raise ValueError(f"Its order '{cells[ORDER].strip()}' is neither {' nor '.join(ORDERS)}.")
    scans = int(_Number(cells[SCANS], 'its scans', whole=True, least=1))
    return {'ranges': ranges, 'step': step, 'per_step': per_step, 'order': order,
            'scans': scans, 'plan': plan}


def Duration_S(entry, bunched, trigger_hz, settle_model):
    """About how long ``entry`` takes, its calibration with it: each step's
    bunches and settling, and each scan's return to its start."""
    plan = entry['plan']
    bunches = entry['per_step'] if bunched else round(entry['per_step'] * trigger_hz)
    step_s = bunches / trigger_hz + Settle_For(settle_model, entry['step'])
    back_s = Settle_For(settle_model, float(np.ptp(plan.Setpoints)))
    calibration_s = len(plan) * (THRESHOLDS['SettleS'] + CALIBRATION_READ_S)
    return entry['scans'] * (len(plan) * step_s + back_s) + calibration_s


def Describe(entry, bunched):
    """One line of what ``entry`` takes, for the operator."""
    per = (f"{entry['per_step']:g} bunches" if bunched else f"{entry['per_step']:g} s")
    scans = f"{entry['scans']} scan{'s' if entry['scans'] > 1 else ''}"
    return (f"{Ranges_Text(entry['ranges'])} V in {entry['step']:g} V steps: "
            f"{len(entry['plan'])} setpoints, {per} each, {scans}, {entry['order']}")


class _Order_Delegate(QStyledItemDelegate):
    """The Order column chosen from a list, not typed."""

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(ORDERS)
        return combo

    def setEditorData(self, editor, index):
        editor.setCurrentText(index.data() or ORDERS[0])

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText())


class Scan_Queue(QWidget):
    """The queue's page. ``Show_Requested(entry)`` asks for a row on the
    parameters page; ``Value_Changed`` fires as the rows or states change."""

    Show_Requested = Signal(dict)
    Value_Changed = Signal()

    def __init__(self, limit_v, parent=None):
        super().__init__(parent)
        self._Limit_V = limit_v
        self._Bunched, self._Trigger_Hz, self._Settle_Model = True, 10.0, None
        self._Locked = False
        self.Mode_Combo = QComboBox()
        self.Mode_Combo.addItems(MODES)
        self.Mode_Combo.setToolTip("Automatic: each scan starts as the one before ends, so the "
                                   "queue can run unattended. Manual: ORACLE asks before each "
                                   "scan after the first, saying what it will take.")
        self.Table = QTableWidget(1, len(HEADERS), alternatingRowColors=True)
        self.Table.setHorizontalHeaderLabels(HEADERS)
        for column, tip in enumerate((
                "The scan's ranges, from:to in volts, apart by commas: 20:55, 60:80.",
                "The step between setpoints (V).",
                "Bunches at each setpoint; seconds in DC mode, as on the parameters page.",
                "Sawtooth takes every scan upwards; Triangle up, then back down.",
                "How many scans the run takes, a whole number.",
                "What became of the scan: empty while it is queued.")):
            self.Table.horizontalHeaderItem(column).setToolTip(tip)
        header = self.Table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(RANGES, QHeaderView.ResizeMode.Stretch)
        self.Table.verticalHeader().setVisible(False)
        self.Table.verticalHeader().setDefaultSectionSize(
            self.Table.fontMetrics().height() + 8)      # compact rows
        self.Table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.Table.setItemDelegateForColumn(ORDER, _Order_Delegate(self.Table))
        self.Table.setToolTip("One scan per row, taken from the top; a new row appears as the "
                              "last one fills. Paste rows copied from a spreadsheet; Delete "
                              "clears the selected cells.")
        self.Table.installEventFilter(self)
        self.Table.itemChanged.connect(self._On_Item_Changed)
        self.Add_Button = QPushButton('Add current', toolTip="Add the scan parameters as they "
                                                             "stand, as the last scan.")
        self.Show_Button = QPushButton('Show', toolTip="Put the selected scan on the Scan "
                                                       "parameters page.",
                                       clicked=self._Show_Selected)
        self.Up_Button = QToolButton(icon=theme.Icon('chevron-up'), toolTip="Take it sooner.",
                                     clicked=lambda: self._Move(-1))
        self.Down_Button = QToolButton(icon=theme.Icon('chevron-down'), toolTip="Take it later.",
                                       clicked=lambda: self._Move(1))
        self.Remove_Button = QPushButton('Remove', toolTip="Remove the selected scans.",
                                         clicked=self._Remove)
        self.Clear_Button = QPushButton('Clear', toolTip="Remove every scan.",
                                        clicked=self._Clear)
        self.Start_Button = QPushButton('Start queue')
        self.Readout = QPlainTextEdit(readOnly=True, styleSheet=MONO_QSS)
        self.Readout.setMaximumHeight(8 * self.Readout.fontMetrics().height() + 12)

        form = QFormLayout()
        form.addRow('Mode:', self.Mode_Combo)
        buttons = QHBoxLayout()
        for widget in (self.Add_Button, self.Show_Button, self.Up_Button, self.Down_Button,
                       self.Remove_Button):
            buttons.addWidget(widget)
        buttons.addStretch()
        buttons.addWidget(self.Clear_Button)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.addLayout(form)
        column.addWidget(self.Table, 1)
        column.addLayout(buttons)
        column.addWidget(self.Start_Button)
        column.addWidget(self.Readout)
        self.Mode_Combo.currentIndexChanged.connect(self._Refresh)
        self._Refresh()

    # ── What the page is told ───────────────────────────────────

    def Set_Timing(self, bunched, trigger_hz):
        """The parameters page's mode and trigger rate, which every row's
        bunches and time are counted in."""
        self._Bunched, self._Trigger_Hz = bunched, trigger_hz
        self.Table.horizontalHeaderItem(PER_STEP).setText('Bunches' if bunched else 'Seconds')
        self._On_Item_Changed(None)

    def Set_Settle_Model(self, model):
        self._Settle_Model = model
        self._Refresh()

    def Manual(self):
        return self.Mode_Combo.currentIndex() == 1

    def Lock(self, locked):
        """The rows held as they are while the queue or anything else runs."""
        self.Table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers if locked else
                                   QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed
                                   | QAbstractItemView.EditTrigger.AnyKeyPressed)
        for widget in (self.Add_Button, self.Up_Button, self.Down_Button, self.Remove_Button,
                       self.Clear_Button):
            widget.setEnabled(not locked)
        self._Locked = locked

    # ── The rows ────────────────────────────────────────────────

    def _Text(self, row, column):
        item = self.Table.item(row, column)
        return item.text().strip() if item is not None else ''

    def _Cells(self, row):
        return [self._Text(row, column) for column in range(len(COLUMNS))]

    def _Filled(self):
        """The rows that hold a scan, by their index."""
        return [row for row in range(self.Table.rowCount())
                if any(self._Cells(row)[:STATE])]

    def Entry(self, row):
        return Entry(self._Cells(row), self._Bunched, self._Limit_V)

    def State(self, row):
        return self._Text(row, STATE)

    def Pending(self):
        """``[(row, entry), ...]`` of the scans not done, in order; raises
        ValueError naming the first that cannot be taken."""
        pending = []
        for number, row in enumerate(self._Filled(), 1):
            if self.State(row).startswith(DONE):
                continue
            try:
                pending.append((row, self.Entry(row)))
            except ValueError as e:
                raise ValueError(f"Scan {number} of the queue cannot be taken: {e}") from None
        return pending

    def Position(self, row):
        """``(its place in the queue, the queue's length)``."""
        filled = self._Filled()
        return filled.index(row) + 1, len(filled)

    def Set_State(self, row, state, why=''):
        """What became of the scan in ``row``, the reason in its tooltip."""
        item = QTableWidgetItem(state)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setToolTip(why)
        self.Table.setItem(row, STATE, item)

    def Reset_States(self):
        for row in self._Filled():
            self.Set_State(row, '')

    def Add_Entry(self, entry):
        """``entry`` (Scan_Tab.Entry_Now's) as the last scan."""
        row = self._Filled()[-1] + 1 if self._Filled() else 0
        self.Table.setRowCount(max(self.Table.rowCount(), row + 1))
        texts = (Ranges_Text(entry['ranges']), f"{entry['step']:g}", f"{entry['per_step']:g}",
                 entry['order'], f"{entry['scans']}")
        self.Table.blockSignals(True)
        for column, text in enumerate(texts):
            self.Table.setItem(row, column, QTableWidgetItem(text))
        self.Table.blockSignals(False)
        self.Set_State(row, '')

    # ── The window's memory of it ───────────────────────────────

    def Value(self):
        return [dict(zip(COLUMNS, self._Cells(row)), why=self.Why(row))
                for row in self._Filled()]

    def Set_Value(self, rows):
        """The rows as kept; a scan that was running when ORACLE stopped is
        marked interrupted, to be taken again."""
        self.Table.blockSignals(True)
        self.Table.clearContents()
        self.Table.setRowCount(len(rows) + 1)
        for row, cells in enumerate(rows):
            for column, key in enumerate(COLUMNS[:STATE]):
                self.Table.setItem(row, column, QTableWidgetItem(str(cells.get(key, ''))))
        self.Table.blockSignals(False)
        for row, cells in enumerate(rows):
            state = str(cells.get('state', ''))
            self.Set_State(row, INTERRUPTED if state in (CALIBRATING, RUNNING) else state,
                           str(cells.get('why', '')))

    def Why(self, row):
        item = self.Table.item(row, STATE)
        return item.toolTip() if item is not None else ''

    # ── Editing ─────────────────────────────────────────────────

    def _On_Item_Changed(self, _item):
        """Red for a cell that is not a value, one empty row kept at the
        bottom, and the readout again."""
        table, bad = self.Table, QColor(theme.VERDICT_COLOURS['FAIL'])
        table.blockSignals(True)
        for row in range(table.rowCount()):
            cells = self._Cells(row)
            for column in range(STATE):
                item = table.item(row, column)
                if item is not None:
                    item.setData(Qt.ItemDataRole.ForegroundRole,
                                 bad if cells[column] and not self._Fits(column, cells[column])
                                 else None)
        last = table.rowCount() - 1
        if any(self._Cells(last)[:STATE]):
            table.insertRow(last + 1)
        table.blockSignals(False)
        self._Refresh()
        self.Value_Changed.emit()

    def _Fits(self, column, text):
        """Whether ``text`` is a value ``column`` takes, alone."""
        try:
            if column == RANGES:
                Parse_Ranges(text)
            elif column == ORDER:
                return text.capitalize() in ORDERS
            else:
                _Number(text, 'it', whole=column == SCANS or (column == PER_STEP
                                                               and self._Bunched))
        except ValueError:
            return False
        return True

    def eventFilter(self, obj, event):
        """Spreadsheet keys on the table: paste rows (tab-separated, in the
        columns' order) from the current cell, growing the table; Delete or
        Backspace clears the selected cells."""
        if (obj is self.Table and event.type() == QEvent.Type.KeyPress
                and not self._Locked):
            table = self.Table
            if event.matches(QKeySequence.StandardKey.Paste):
                row0, column0 = max(0, table.currentRow()), max(0, table.currentColumn())
                lines = [line.split('\t') for line in QApplication.clipboard().text().splitlines()
                         if line.strip()]
                table.blockSignals(True)
                table.setRowCount(max(table.rowCount(), row0 + len(lines)))
                for i, cells in enumerate(lines):
                    for j, text in enumerate(cells[:STATE - column0]):
                        table.setItem(row0 + i, column0 + j, QTableWidgetItem(text.strip()))
                table.blockSignals(False)
                self._On_Item_Changed(None)
                return True
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                for item in table.selectedItems():
                    if item.column() != STATE:
                        item.setText('')
                return True
        return super().eventFilter(obj, event)

    def _Selected_Rows(self):
        return sorted({index.row() for index in self.Table.selectedIndexes()})

    def _Show_Selected(self):
        rows = self._Selected_Rows() or [self.Table.currentRow()]
        try:
            self.Show_Requested.emit(self.Entry(rows[0]))
        except (ValueError, IndexError):
            pass                        # nothing to show: the readout says why

    def _Move(self, delta):
        rows, filled = self._Selected_Rows(), self._Filled()
        if len(rows) != 1 or rows[0] not in filled:
            return
        row = rows[0]
        other = row + delta
        if other not in filled:
            return
        moved = [(self._Cells(r), self.Why(r)) for r in (other, row)]
        self.Table.blockSignals(True)
        for r, (cells, why) in zip((row, other), moved):
            for column in range(STATE):
                self.Table.setItem(r, column, QTableWidgetItem(cells[column]))
            self.Set_State(r, cells[STATE], why)
        self.Table.blockSignals(False)
        self.Table.selectRow(other)
        self._On_Item_Changed(None)

    def _Remove(self):
        for row in reversed(self._Selected_Rows()):
            self.Table.removeRow(row)
        if not self.Table.rowCount():
            self.Table.setRowCount(1)
        self._On_Item_Changed(None)

    def _Clear(self):
        self.Table.clearContents()
        self.Table.setRowCount(1)
        self._On_Item_Changed(None)

    # ── The readout ─────────────────────────────────────────────

    def _Refresh(self):
        """Each scan's setpoints and time, or its fault; then the queue's."""
        lines, total, left = [], 0.0, 0
        for number, row in enumerate(self._Filled(), 1):
            state = self.State(row)
            try:
                entry = self.Entry(row)
            except ValueError as e:
                lines.append(f"{number:>2}  fault: {e}")
                continue
            seconds = Duration_S(entry, self._Bunched, self._Trigger_Hz, self._Settle_Model)
            done = state.startswith(DONE)
            if not done:
                total, left = total + seconds, left + 1
            lines.append(f"{number:>2}  {len(entry['plan'])} setpoints × {entry['scans']}, "
                         f"about {Duration_Text(seconds)}"
                         + (f"  [{state}: {self.Why(row)}]" if state else ''))
        if not lines:
            lines.append("No scan queued: Add current adds the scan parameters as they stand.")
        elif left:
            end = datetime.now() + timedelta(seconds=total)
            when = f"{end:%H:%M}" if end.date() == datetime.now().date() else f"{end:%a %H:%M}"
            lines.append(f"Queue: {left} scan{'s' if left > 1 else ''} to take, about "
                         f"{Duration_Text(total)}, ending about {when} if started now.")
        else:
            lines.append("Queue: every scan in it is done.")
        self.Readout.setPlainText('\n'.join(lines))
