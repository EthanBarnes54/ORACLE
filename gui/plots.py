"""The plots: their dark style, how they are handled, and the TOF plot that carries the gate.

Date:    2026-10-29
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
inside it.

A curve of far more points than the plot has pixels (a 50,000-bin TOF, a
long cooler record) is a Dense_Curve: the same pixels, painted from only
what is in view, and a histogram from the few vertices that fill them, so
it redraws in milliseconds instead of tens of them.

Depends on: numpy, pyqtgraph, PySide6; gui.theme.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.PlotCurveItem import arrayToLineSegments
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPaintEngine
from PySide6.QtWidgets import QApplication, QToolButton

from gui import theme

AXIS_WIDTH = 72
BUTTON_MARGIN = 6
# Qt's raster engine puts coordinates on a 1/64 px grid, which moves where each
# half device pixel begins by this much, in half pixels
HALF_PIXEL_OFFSET = 1 / 64
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


def Collapse(x, y, device_x):
    """``x, y`` with each run of vertices that fall in one half device pixel
    (each at ``device_x``) replaced by four at the run's first: where it
    comes in, its lowest, its highest and where it goes out. None if that
    would not make the curve at least four times shorter."""
    slot = np.floor(2 * device_x + HALF_PIXEL_OFFSET).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, slot[1:] != slot[:-1]])
    if 4 * len(starts) > len(x) / 4:
        return None
    ends = np.r_[starts[1:], len(x)] - 1
    return np.repeat(x[starts], 4), np.column_stack((
        y[starts], np.minimum.reduceat(y, starts), np.maximum.reduceat(y, starts),
        y[ends])).ravel()


class Dense_Curve(pg.PlotCurveItem):
    """A curve with far more vertices than its plot shows pixels across: a
    50,000-bin TOF, a long cooler record, painted pixel for pixel as every
    vertex would be, from far fewer. Nothing past the plot is painted, so
    only the vertices in view (and the one beyond each edge) are drawn.
    And Qt draws the plots' pen (solid, opaque, 2 px, not antialiased) as
    separate segments, the pixels of each depending only on the half
    device pixel its ends fall in; so a step curve, a histogram, all of
    whose segments run straight across or up, fills in each half pixel
    exactly the pixels of one upright segment from its lowest to its
    highest, and is painted from those (:func:`Collapse`). A line's
    segments slant, and are only cut to the view. Its data and bounds stay
    the whole curve's, and anything else (an export, a fill, another pen,
    a value that is not finite, a transform that is not a plain scale) is
    painted as PlotCurveItem paints it."""

    CLIP_MARGIN_PX = 4                  # beyond the plot, wider than the pen reaches

    def updateData(self, *args, **kargs):
        super().updateData(*args, **kargs)
        x, y = self.xData, self.yData
        # a segment at least, and every vertex a number
        self._Finite = (x is not None and len(x) > 1
                        and bool(np.isfinite(x).all() and np.isfinite(y).all()))
        self._Sorted = self._Finite and bool(np.all(x[1:] >= x[:-1]))
        if self._Finite and self.opts['stepMode'] == 'center':
            # the outline PlotCurveItem draws for a step curve with no fill
            x, y = np.repeat(x, 2)[1:-1], np.repeat(y, 2)
        self._Outline = x, y
        self._Painted = None            # (what it was painted for, segments) of the last paint

    def paint(self, p, opt, widget):
        segments = self._Segments(p)
        if segments is None:
            super().paint(p, opt, widget)
            return
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        p.setPen(self.opts['pen'])
        p.drawLines(*segments.drawargs())

    def _Segments(self, p):
        """The segments to paint with ``p``: the curve's part in view,
        collapsed if it is dense; None if it is to be painted whole."""
        opts, pen, to_device = self.opts, self.opts['pen'], p.deviceTransform()
        if not (self._Finite and p.paintEngine().type() == QPaintEngine.Type.Raster
                and self._exportOpts is False and not opts['antialias']
                and opts['fillLevel'] is None and opts['shadowPen'] is None
                and opts['compositionMode'] is None and isinstance(opts['connect'], str)
                and opts['connect'] in ('all', 'finite') and opts['stepMode'] in (None, 'center')
                and pen.widthF() > 1.0 and pen.style() == Qt.PenStyle.SolidLine
                and pen.isSolid() and pen.color().alphaF() == 1.0 and to_device.isAffine()
                and to_device.m12() == 0.0 and to_device.m21() == 0.0):
            return None
        x, y = self._Outline
        scale, offset = to_device.m11(), to_device.dx()
        first, last = 0, len(x)
        if self._Sorted and p.hasClipping():
            # nothing past the plot is painted, so only the vertices in view and
            # the one beyond each edge, whose segment crosses into it, are drawn
            clip, margin = p.clipBoundingRect(), self.CLIP_MARGIN_PX / abs(scale)
            first = max(int(np.searchsorted(x, clip.left() - margin)) - 1, 0)
            last = min(int(np.searchsorted(x, clip.right() + margin, 'right')) + 1, len(x))
        painted_for = (scale, offset, first, last)
        if self._Painted is None or self._Painted[0] != painted_for:
            clipped = last - first < len(x)
            x, y = x[first:last], y[first:last]
            shown = (Collapse(x, y, scale * x + offset) if opts['stepMode'] == 'center'
                     else None)
            if shown is None and clipped:
                shown = x, y
            self._Painted = (painted_for, None if shown is None else
                             arrayToLineSegments(*shown, connect='all', finiteCheck=False))
        return self._Painted[1]


def Dots(colour=theme.TRACE):
    """The keywords for a ``plot()`` of points, not a line."""
    return {'pen': None, 'symbol': 'o', 'symbolSize': 4, 'symbolPen': None,
            'symbolBrush': colour}


class _Gate_View(_View):
    """``On_Change()`` is called once a gate is set, moved or removed; while
    ``Locked`` it can be neither, and the view pans and zooms as any other."""

    def __init__(self):
        super().__init__()
        self.Region = None
        self.Locked = False
        self.On_Change = lambda: None

    def mouseClickEvent(self, event):
        # a double-click on the gate removes it; only one elsewhere autoscales
        if self.Region is not None and event.double() and not self.Locked:
            low, high = self.Region.getRegion()
            if low <= self.mapSceneToView(event.scenePos()).x() <= high:
                event.accept()
                self._Remove()
                return
        super().mouseClickEvent(event)

    def mouseDragEvent(self, event, axis=None):
        if (self.Locked or QApplication.keyboardModifiers()
                != Qt.KeyboardModifier.ControlModifier):
            super().mouseDragEvent(event, axis)
            return
        event.accept()
        self.Set(sorted((self.mapSceneToView(event.buttonDownScenePos()).x(),
                         self.mapSceneToView(event.scenePos()).x())), event.isFinish())

    def Set(self, span, finished=True):
        """The gate over ``span``, made if there is none; None removes it."""
        if span is None:
            if self.Region is not None:
                self._Remove()
            return
        if self.Region is None:
            self.Region = pg.LinearRegionItem(values=span, brush=pg.mkBrush(*theme.GATE_FILL),
                                              movable=not self.Locked)
            self.Region.sigRegionChangeFinished.connect(lambda: self.On_Change())
            self.addItem(self.Region)
        else:
            self.Region.setRegion(span)
        if finished:
            self.On_Change()

    def Lock(self, locked):
        if locked == self.Locked:
            return
        self.Locked = locked
        if self.Region is not None:
            self.Region.setMovable(not locked)

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
        self.Curve = Dense_Curve(pen=Pen())
        self.addItem(self.Curve)

    def Gate(self):
        """``(low, high)`` in µs, or None with no gate set."""
        if self._View.Region is None:
            return None
        return tuple(float(t) for t in self._View.Region.getRegion())

    def Set_Gate(self, span):
        """The gate over ``(low, high)``; None removes it."""
        self._View.Set(None if span is None else sorted(span))

    def Lock_Gate(self, locked):
        """While locked the gate is neither set, moved nor removed by hand."""
        self._View.Lock(locked)
