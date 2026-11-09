"""How every plot is handled: a hand-zoomed plot stops following its data, and
autoscale, by its button or a double-click, brings it back."""

from types import SimpleNamespace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF, Qt, QUrl

from gui import plots, theme
from gui.shared_widgets import (Figure, Muted, Report_Html, Report_Label, Report_Table,
                                Saved_Section)


ABOUT = ('A plot for the test.', 'Nothing is calculated.')

def _Double_Click(view, x):
    """A left double-click at data x, as pyqtgraph's scene delivers it."""
    accepted = []
    return SimpleNamespace(double=lambda: True, button=lambda: Qt.MouseButton.LeftButton,
                           scenePos=lambda: view.mapViewToScene(QPointF(x, 0.0)),
                           accept=lambda: accepted.append(True), Accepted=accepted)


@pytest.fixture
def plot(qapp):
    widget = plots.Plot('Test', ('x', None), ('y', None), ABOUT)
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


def _Pixels(widget, qapp):
    qapp.processEvents()
    image = widget.grab().toImage()
    return np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), -1, 4).copy()


@pytest.mark.parametrize('view', (None, (95.0, 105.0), (99.9, 100.1)))
def test_A_Dense_Histogram_Paints_The_Pixels_Of_Every_Bin(qapp, view):
    """A 50,000-bin TOF, whole, zoomed to its peak and to fifty bins: the
    collapsed curve fills exactly the pixels every bin drawn would."""
    rng = np.random.default_rng(1)
    edges = 0.004 * (np.arange(50002) - 0.5)
    counts = rng.poisson(2.0 + 400 * np.exp(-0.5 * ((edges[:-1] - 100) / 3) ** 2)).astype(float)
    widget = plots.Plot('Test', ('x', None), ('y', None), ABOUT)
    widget.resize(500, 320)
    widget.show()
    whole = pg.PlotCurveItem(edges, counts, pen=plots.Pen(), stepMode='center')
    widget.addItem(whole)
    _Pixels(widget, qapp)
    if view is not None:
        widget.setXRange(*view, padding=0)
    widget.getViewBox().disableAutoRange()
    expected = _Pixels(widget, qapp)
    widget.removeItem(whole)
    dense = plots.Dense_Curve(edges, counts, pen=plots.Pen(), stepMode='center')
    widget.addItem(dense)
    assert np.array_equal(_Pixels(widget, qapp), expected)
    assert dense._Painted[1] is not None                    # and it was the collapsed curve
    assert np.array_equal(dense.getData()[1], counts)       # whose data stay every bin
    widget.close()


def test_A_Curve_Collapse_Keeps_Each_Half_Pixel_Extremes():
    x = np.linspace(0.0, 1.0, 4001)
    y = np.sin(40 * x)
    shown_x, shown_y = plots.Collapse(x, y, 100.0 * x)       # 100 px: 200 half pixels
    assert len(shown_x) == 4 * 200 + 4                       # the last vertex has its own
    assert shown_y.max() == y.max() and shown_y.min() == y.min()
    assert shown_x[0] == x[0] and shown_y[-1] == y[-1]
    sparse = np.arange(50.0)
    assert plots.Collapse(sparse, sparse, 10.0 * sparse) is None   # a vertex a pixel: whole


def test_A_Curve_With_Gaps_Is_Painted_As_Ever(qapp):
    """A reading lost leaves a gap, which only the whole curve keeps."""
    widget = plots.Plot('Test', ('x', None), ('y', None), ABOUT)
    widget.resize(300, 200)
    widget.show()
    x = np.arange(100000.0)
    gappy = np.where(x % 997 == 0, np.nan, np.sin(x / 50))
    pictures = []
    for kind in (pg.PlotCurveItem, plots.Dense_Curve):
        curve = kind(x, gappy, pen=plots.Pen(), connect='finite')
        widget.addItem(curve)
        pictures.append(_Pixels(widget, qapp))
        widget.getViewBox().disableAutoRange()
        widget.removeItem(curve)
    assert np.array_equal(*pictures)
    widget.close()


@pytest.mark.parametrize('view', (None, (100.0, 110.0)))
def test_A_Dense_Line_Paints_The_Pixels_Of_Every_Point(qapp, view):
    """A line's segments slant, so it is only cut to the view, never
    collapsed: a standing-in upright segment would fill other pixels."""
    rng = np.random.default_rng(0)
    t = np.repeat(np.arange(3000.0) * 0.1, 8)               # a bunch's hits share a time
    volts = np.repeat(rng.normal(0.0, 0.2, 3000), 8)
    widget = plots.Plot('Test', ('x', None), ('y', None), ABOUT)
    widget.resize(600, 400)
    widget.show()
    pictures = []
    for kind in (pg.PlotCurveItem, plots.Dense_Curve):
        curve = kind(t, volts, pen=plots.Pen(), connect='finite')
        widget.addItem(curve)
        if view is not None:
            widget.setXRange(*view, padding=0)
        pictures.append(_Pixels(widget, qapp))
        widget.getViewBox().disableAutoRange()
        widget.removeItem(curve)
    assert np.array_equal(*pictures)
    widget.close()


def test_A_Report_Lines_Its_Figures_Up_And_Links_The_Files_It_Saved(qapp):
    path = '/data/hardware_calibration/linearity_2026-11-09_120000.csv'
    text = Report_Html([
        (('WARN', 'Straight-line fit'), [('Offset', Figure(-0.27, '+.3f', 'V')),
                                         'Gain <1 & falling', Muted('a footnote')]),
        ('Branches', [Report_Table(('Branch', 'Gain'), [('Positive', '0.999159')])]),
        Saved_Section([path])])
    # each figure right-aligned beside its label, with a true minus sign
    assert "<td style='color:" in text and "<td align='right'>−0.270 V</td>" in text
    assert theme.VERDICT_COLOURS['WARN'] in text and 'Straight-line fit' in text
    assert 'Gain &lt;1 &amp; falling' in text and '>a footnote</span>' in text
    assert "<td align='right'>0.999159</td>" in text
    # the file by its name, a link to it
    assert f"href='{QUrl.fromLocalFile(path).toString()}'" in text
    assert '>linearity_2026-11-09_120000.csv</a>' in text
    label = Report_Label(text)
    assert label.openExternalLinks() and label.wordWrap()
