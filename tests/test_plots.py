"""How every plot is handled: a hand-zoomed plot stops following its data, and
autoscale, by its button or a double-click, brings it back."""

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF, Qt

from gui import plots


def _Double_Click(view, x):
    """A left double-click at data x, as pyqtgraph's scene delivers it."""
    accepted = []
    return SimpleNamespace(double=lambda: True, button=lambda: Qt.MouseButton.LeftButton,
                           scenePos=lambda: view.mapViewToScene(QPointF(x, 0.0)),
                           accept=lambda: accepted.append(True), Accepted=accepted)


@pytest.fixture
def plot(qapp):
    widget = plots.Plot('Test', ('x', None), ('y', None))
    widget.resize(400, 300)
    widget.show()
    widget.plot(np.arange(100.0), np.arange(100.0) ** 2)
    qapp.processEvents()
    yield widget
    widget.close()


def _Following(widget):
    return all(widget.getViewBox().autoRangeEnabled())


def test_Zooming_By_Hand_Stops_Following_And_Autoscale_Restores_It(qapp, plot):
    plot.setXRange(10, 20)
    assert not _Following(plot)
    plot.Autoscale_Button.click()
    qapp.processEvents()
    assert _Following(plot)
    (low, high), _ = plot.getViewBox().viewRange()
    assert low <= 0 and high >= 99


def test_A_Double_Click_Autoscales(qapp, plot):
    plot.setYRange(0, 5)
    view = plot.getViewBox()
    event = _Double_Click(view, 50.0)
    view.mouseClickEvent(event)
    assert event.Accepted and _Following(plot)


def test_A_Double_Click_On_The_Gate_Removes_It_And_Leaves_The_View(qapp):
    tof = plots.Gate_Plot()
    tof.resize(400, 300)
    tof.show()
    view = tof.getViewBox()
    tof.setXRange(0, 100)
    view.Region = pg.LinearRegionItem(values=(40, 60))
    view.addItem(view.Region)
    view.mouseClickEvent(_Double_Click(view, 50.0))
    assert tof.Gate() is None and not _Following(tof)
    view.mouseClickEvent(_Double_Click(view, 10.0))
    assert _Following(tof)
    tof.close()


def test_The_Corner_Button_Sits_In_The_Top_Right(qapp, plot):
    button = plot.Autoscale_Button
    assert button.isVisible()
    assert button.x() + button.width() <= plot.width() and button.y() < 20
    assert button.x() > plot.width() / 2
