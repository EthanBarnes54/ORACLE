"""The plots: their dark style, how they are handled, and the TOF plot that carries the gate.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

pyqtgraph, as in the old DAQ, because the plots redraw several times a
second while data come in. Every plot is handled the same way: drag to
pan, right-drag to stretch or squash along the direction dragged, the
wheel to zoom (over an axis, that axis alone), Shift-drag to zoom to a
box. A plot handled by hand stops following its data, so a live update
never pulls the view away; the autoscale button in its corner, or a
double-click on it, puts it back to the full view and following again.

Ctrl-drag on a gate plot (Cmd-drag on macOS) sets its gate, dragging
its edges moves it, and a double-click removes it; on the Scan tab's TOF
plot the spectrum, the TOF-HFS map and the time graph count only hits
inside it, and the run viewer gates its runs the same way.

Depends on: pyqtgraph, PySide6; gui.theme.
"""

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QToolButton

from gui import theme

AXIS_WIDTH = 72
BUTTON_MARGIN = 6
GESTURES = ("Drag to pan, right-drag to stretch or squash, wheel to zoom (over an axis: that "
            "axis alone), Shift-drag to zoom to a box.")


def Plot(title, x_axis, y_axis):
    """A plot in DENIS's dark-plot style. Each axis is ``(label, units)``; an
    axis with units gets pyqtgraph's SI prefix (mV, kV), one without is
    left unscaled."""
    plot = Plot_Widget()
    _Style(plot, title, x_axis, y_axis)
    return plot


class _View(pg.ViewBox):
    """The handling every plot shares; see the module's docstring."""

    def mouseDragEvent(self, event, axis=None):
        shift = QApplication.keyboardModifiers() == Qt.KeyboardModifier.ShiftModifier
        if shift and event.button() == Qt.MouseButton.LeftButton and axis is None:
            self.setMouseMode(pg.ViewBox.RectMode)
            super().mouseDragEvent(event, axis)
            self.setMouseMode(pg.ViewBox.PanMode)
            return
        super().mouseDragEvent(event, axis)

    def mouseClickEvent(self, event):
        if event.double() and event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self.Autoscale()
            return
        super().mouseClickEvent(event)

    def Autoscale(self):
        """Back to the full view, following the data again."""
        self.enableAutoRange()


class Plot_Widget(pg.PlotWidget):
    """A PlotWidget with the shared handling and an autoscale button."""

    def __init__(self, parent=None, view=None):
        super().__init__(parent, background=theme.PLOT_BACKGROUND, viewBox=view or _View())
        self.getPlotItem().hideButtons()            # the corner button below replaces them
        self.Autoscale_Button = QToolButton(self, icon=theme.Icon('rotate-ccw'), autoRaise=True)
        self.Autoscale_Button.setToolTip(f"Autoscale: back to the full view, following the "
                                         f"data (or double-click the plot). {GESTURES}")
        self.Autoscale_Button.clicked.connect(self.Autoscale)

    def Autoscale(self):
        self.getViewBox().Autoscale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # PlotWidget's own __init__ resizes before the button exists, and it
        # hands unknown attributes to its PlotItem, so look in the instance
        button = self.__dict__.get('Autoscale_Button')
        if button is None:
            return
        button.move(self.width() - button.sizeHint().width() - BUTTON_MARGIN, BUTTON_MARGIN)


def _Style(plot, title, x_axis, y_axis):
    plot.setTitle(title, color=theme.PLOT_FOREGROUND)
    for side, (label, units) in (('bottom', x_axis), ('left', y_axis)):
        axis = plot.getAxis(side)
        axis.setPen(theme.PLOT_FOREGROUND)
        axis.setTextPen(theme.PLOT_FOREGROUND)
        axis.setLabel(label, units=units, color=theme.PLOT_FOREGROUND)
        axis.enableAutoSIPrefix(units is not None)
    # one width for every left axis, so plots stacked or side by side keep
    # their data areas lined up whatever their tick labels
    plot.getAxis('left').setWidth(AXIS_WIDTH)
    plot.showGrid(x=True, y=True, alpha=theme.PLOT_GRID_ALPHA)


def Pen(colour=theme.TRACE):
    return pg.mkPen(colour, width=2)


def Dots(colour=theme.TRACE):
    """The keywords for a ``plot()`` of points, not a line."""
    return {'pen': None, 'symbol': 'o', 'symbolSize': 4, 'symbolPen': None,
            'symbolBrush': colour}


class _Gate_View(_View):
    """``On_Change()`` is called once a gate is set, moved or removed."""

    def __init__(self):
        super().__init__()
        self.Region = None
        self.On_Change = lambda: None

    def mouseClickEvent(self, event):
        # a double-click on the gate removes it; only one elsewhere autoscales
        if self.Region is not None and event.double():
            low, high = self.Region.getRegion()
            if low <= self.mapSceneToView(event.scenePos()).x() <= high:
                event.accept()
                self._Remove()
                return
        super().mouseClickEvent(event)

    def mouseDragEvent(self, event, axis=None):
        if QApplication.keyboardModifiers() != Qt.KeyboardModifier.ControlModifier:
            super().mouseDragEvent(event, axis)
            return
        event.accept()
        span = sorted((self.mapSceneToView(event.buttonDownScenePos()).x(),
                       self.mapSceneToView(event.scenePos()).x()))
        if self.Region is None:
            self.Region = pg.LinearRegionItem(values=span, brush=pg.mkBrush(*theme.GATE_FILL))
            self.Region.sigRegionChangeFinished.connect(lambda: self.On_Change())
            self.addItem(self.Region)
        else:
            self.Region.setRegion(span)
        if event.isFinish():
            self.On_Change()

    def _Remove(self):
        self.removeItem(self.Region)
        self.Region = None
        self.On_Change()


class Gate_Plot(Plot_Widget):
    """A histogram and its gate: the TOF spectrum, unless told otherwise.
    ``Gate_Changed`` fires once a gate is set, moved or removed."""

    Gate_Changed = Signal()

    def __init__(self, parent=None, title='TOF spectrum', x_axis=('Time of flight (µs)', None),
                 what='TOF'):
        view = _Gate_View()
        super().__init__(parent, view)
        self._View = view
        view.On_Change = self.Gate_Changed.emit
        _Style(self, title, x_axis, ('Counts', None))
        self.setToolTip(f"Ctrl-drag (Cmd-drag on macOS) to set the {what} gate, drag its "
                        f"edges to move it, double-click it to remove it.")
        self.Curve = pg.PlotCurveItem(pen=Pen())
        self.addItem(self.Curve)

    def Gate(self):
        """``(low, high)`` in µs, or None with no gate set."""
        if self._View.Region is None:
            return None
        return tuple(float(t) for t in self._View.Region.getRegion())
