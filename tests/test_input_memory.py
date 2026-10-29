"""The window's memory of its inputs: every one back as it was left, however
ORACLE stopped, and a file it cannot use costing nothing but its values."""

import os

import pytest

from gui import input_memory
from gui.input_memory import Input_Memory, _Accessors
from tests.window import Pump


def _Values(inputs):
    return {name: _Accessors(widget)[0]() for name, widget in inputs}


@pytest.fixture
def scan_tabs(qapp):
    from gui.scan_tab import Scan_Tab
    tabs = []
    yield lambda: tabs.append(Scan_Tab(4000.0)) or tabs[-1]
    for tab in tabs:
        tab.deleteLater()


def test_Every_Input_Comes_Back_As_It_Was_Left(qapp, tmp_path, scan_tabs):
    """DC mode and its seconds a step among them: the mode is set again
    before the box it resets."""
    path = str(tmp_path / 'inputs.yaml')
    tab, memory = scan_tabs(), Input_Memory(path)
    memory.Add('scan', tab.Inputs())
    tab.Dc_Radio.setChecked(True)
    tab.Per_Step_Box.setValue(0.25)
    tab.Type_Combo.setCurrentText('Triangle')
    tab.Range_Checks[2].setChecked(True)
    tab.From_Boxes[2].setValue(-55.5)
    tab.Release_Box.setValue(250)
    tab.Channel_Checks[1].setChecked(False)
    tab.Tof_Plot.Set_Gate((30.0, 80.0))
    tab.Elog_Edit.setPlainText("beam back at 14:02")
    left = _Values(tab.Inputs())
    memory.Save()
    again = scan_tabs()
    Input_Memory(path).Add('scan', again.Inputs())
    assert _Values(again.Inputs()) == left
    assert again.Per_Step_Label.text() == 'Seconds per channel'


def test_A_Change_Is_Kept_Within_Half_A_Second_And_Never_Half_Written(qapp, tmp_path,
                                                                      scan_tabs):
    path = tmp_path / 'inputs.yaml'
    tab, memory = scan_tabs(), Input_Memory(str(path))
    memory.Add('scan', tab.Inputs())
    tab.Step_Box.setValue(3.5)
    for value in (4.0, 4.5):                    # a burst of typing: one write after it
        tab.Step_Box.setValue(value)
    assert Pump(qapp, path.exists, timeout=2.0)
    assert 'scan.step: 4.5' in path.read_text(encoding='utf-8')
    assert [name for name in os.listdir(tmp_path)] == ['inputs.yaml']   # no stray part


def test_A_File_It_Cannot_Use_Costs_Only_Its_Values(qapp, tmp_path, scan_tabs, caplog):
    path = tmp_path / 'inputs.yaml'
    path.write_text("scan.step: [unclosed\n", encoding='utf-8')
    tab = scan_tabs()
    Input_Memory(str(path)).Add('scan', tab.Inputs())         # unreadable: defaults
    assert tab.Step_Box.value() == 1.0 and 'could not be read' in caplog.text
    path.write_text("scan.step: not a number\nscan.scan_type: Spiral\nscan.dwell: 250\n"
                    "scan.release: 99999999\n", encoding='utf-8')
    tab = scan_tabs()
    Input_Memory(str(path)).Add('scan', tab.Inputs())
    assert tab.Step_Box.value() == 1.0 and tab.Type_Combo.currentText() == 'Sawtooth'
    assert tab.Dwell_Box.value() == 250                         # the rest still come back
    assert tab.Release_Box.value() == tab.Release_Box.maximum()  # held to its limit


def test_Only_Inputs_The_Window_Knows_Are_Taken():
    with pytest.raises(TypeError, match='not an input'):
        _Accessors(object())
    assert input_memory.SAVE_AFTER_MS <= 1000
