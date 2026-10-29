"""The scan queue's page: rows read and checked as a scan is as it starts,
edited as DENIS's optimiser tables are, and kept, states and all."""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import QApplication

from gui import scan_queue as sq


def test_Ranges_Are_Read_As_Written_Negatives_And_All():
    assert sq.Parse_Ranges("20:55, 60:80") == [[20.0, 55.0], [60.0, 80.0]]
    assert sq.Parse_Ranges("-100:-50; 50:100,") == [[-100.0, -50.0], [50.0, 100.0]]
    assert sq.Ranges_Text([[-100.0, -50.0], [60.5, 80.0]]) == "-100:-50, 60.5:80"
    for text, says in (("20-55", "'20-55' is not a range"), ("", "No range"),
                       ("1:2:3", "'1:2:3' is not a range")):
        with pytest.raises(ValueError, match=says):
            sq.Parse_Ranges(text)


def _Cells(ranges="20:30", step="0.5", per_step="20", order="Sawtooth", scans="2"):
    return [ranges, step, per_step, order, scans, '']


def test_A_Row_Is_Checked_As_A_Scan_Is_When_It_Starts():
    entry = sq.Entry(_Cells(), True, 4000.0)
    assert len(entry['plan']) == 21 and entry['scans'] == 2 and entry['order'] == 'Sawtooth'
    assert sq.Entry(_Cells(per_step="0.25", order="triangle"), False, 4000.0)['per_step'] == 0.25
    for cells, bunched, says in (
            (_Cells(ranges="0:5000"), True, "5000 V is beyond the ±4000 V limit"),
            (_Cells(ranges="10:10"), True, "fewer than two points"),
            (_Cells(ranges="0:10, 5:15"), True, "overlap"),
            (_Cells(ranges=", ".join(f"{10 * i}:{10 * i + 5}" for i in range(8))), True,
             "8 ranges; a scan holds 7"),
            (_Cells(step="0.005"), True, "more than 2 decimals"),
            (_Cells(per_step="2.5"), True, "not a whole number"),
            (_Cells(per_step="0"), False, "below 0.01"),
            (_Cells(order="Spiral"), True, "neither Sawtooth nor Triangle"),
            (_Cells(scans="0"), True, "below 1")):
        with pytest.raises(ValueError, match=says):
            sq.Entry(cells, bunched, 4000.0)


def test_A_Scans_Time_Grows_With_Its_Bunches_Scans_And_Settling():
    entry = sq.Entry(_Cells(per_step="20", scans="1"), True, 4000.0)
    one = sq.Duration_S(entry, True, 10.0, None)
    # 21 steps of 20 bunches at 10 Hz, settling, the return and the calibration
    assert 21 * 2.0 < one < 21 * 2.0 + 5.0
    entry['scans'] = 3
    assert sq.Duration_S(entry, True, 10.0, None) == pytest.approx(3 * one, rel=0.05)
    slow = {'measured': '2026-10-29', 'points': [[0.5, 0.2], [20.0, 0.4]]}
    assert sq.Duration_S(entry, True, 10.0, slow) > sq.Duration_S(entry, True, 10.0, None)


@pytest.fixture
def page(qapp):
    widget = sq.Scan_Queue(4000.0)
    widget.Set_Timing(True, 10.0)
    yield widget
    widget.deleteLater()


def _Add(page, ranges, scans=1):
    page.Add_Entry({'ranges': ranges, 'step': 2.0, 'per_step': 2.0, 'order': 'Sawtooth',
                    'scans': scans})


def test_The_Table_Keeps_A_Fresh_Row_And_The_Readout_Each_Scans_Fault(page):
    _Add(page, [[20.0, 30.0]], 2)
    _Add(page, [[40.0, 40.0]])
    assert page.Table.rowCount() == 3                       # one empty row kept below
    assert [page.Position(row) for row in (0, 1)] == [(1, 2), (2, 2)]
    lines = page.Readout.toPlainText().splitlines()
    assert lines[0].startswith(" 1  6 setpoints × 2, about")
    assert lines[1].startswith(" 2  fault:") and "fewer than two points" in lines[1]
    with pytest.raises(ValueError, match="Scan 2 of the queue cannot be taken"):
        page.Pending()
    page.Table.item(1, sq.STEP).setText("two")
    colour = page.Table.item(1, sq.STEP).data(Qt.ItemDataRole.ForegroundRole)
    assert colour is not None and colour.name().upper() == "#FF5C5C"


def test_Rows_Paste_From_A_Spreadsheet_Move_And_Go(page, qapp):
    QApplication.clipboard().setText("20:30\t2\t2\tTriangle\t1\n40:50, 60:64\t2\t2\tSawtooth\t3\n")
    page.Table.setCurrentCell(0, 0)
    paste = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    if not paste.matches(QKeySequence.StandardKey.Paste):          # Cmd on macOS
        paste = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_V, Qt.KeyboardModifier.MetaModifier)
    # to the table's own key handling: sent through the application, the
    # modifier would stay held for the selections below
    assert page.eventFilter(page.Table, paste)
    assert [len(entry['plan']) for _, entry in page.Pending()] == [6, 9]
    page.Table.selectRow(1)
    page.Up_Button.click()
    assert page.Entry(0)['ranges'] == [[40.0, 50.0], [60.0, 64.0]]
    page.Table.selectRow(0)
    page.Remove_Button.click()
    assert [entry['order'] for _, entry in page.Pending()] == ['Triangle']
    page.Clear_Button.click()
    assert page.Pending() == [] and page.Table.rowCount() == 1


def test_The_Rows_And_Their_States_Are_Kept_And_A_Running_Scan_Is_Interrupted(page, qapp):
    _Add(page, [[20.0, 30.0]], 2)
    _Add(page, [[40.0, 50.0]])
    page.Set_State(0, 'done', 'run 7')
    page.Set_State(1, 'running', 'run 8')
    kept = page.Value()
    again = sq.Scan_Queue(4000.0)
    again.Set_Value(kept)
    assert [again.State(row) for row in (0, 1)] == ['done', 'interrupted']
    assert again.Why(0) == 'run 7'
    assert [row for row, _ in again.Pending()] == [1]       # takes up at the scan not done
    again.deleteLater()


def test_Its_Rows_Are_Held_While_It_Runs(page):
    page.Lock(True)
    assert not page.Add_Button.isEnabled() and not page.Remove_Button.isEnabled()
    assert page.Table.editTriggers() == page.Table.EditTrigger.NoEditTriggers
    page.Lock(False)
    assert page.Add_Button.isEnabled()
