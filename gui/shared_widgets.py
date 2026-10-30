"""Shared GUI building blocks: spin boxes, form layouts, verdicts, the
scrolling control column and the worker thread.

Date:    2026-10-30
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

What more than one tab builds the same way. :class:`Scroll_Column` holds
a tab's controls, so the window fits a short screen. :class:`Job_Worker`
keeps a long job off the GUI thread: the job is ``job(progress,
stopped)``, it reports through ``progress`` and checks ``stopped()``
between steps, and what it returns arrives in ``Done``; an exception
arrives in ``Failed`` as its sentence, and is logged in full.

Depends on: PySide6; gui.theme.
"""

import html
import logging
import threading

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtWidgets import (QDoubleSpinBox, QFrame, QGridLayout, QLabel, QScrollArea,
                               QSizePolicy, QSpinBox, QWidget)

from gui import theme

REPORT_WIDTH = 360                  # px: a report's sentences read as lines, not a column

log = logging.getLogger('oracle.gui')


def Spin(value, low, high, decimals=2, suffix=''):
    """A spin box holding ``value`` in ``[low, high]``: whole numbers if
    ``decimals`` is 0."""
    if decimals:
        box = QDoubleSpinBox(decimals=decimals)
    else:
        box, low, high, value = QSpinBox(), int(low), int(high), int(value)
    box.setRange(low, high)
    box.setSuffix(suffix)
    box.setValue(value)
    return box


def Form(group, rows):
    """``group`` laid out as rows of ``(label, widget)``; returns its grid."""
    grid = QGridLayout(group)
    for row, (text, widget) in enumerate(rows):
        grid.addWidget(QLabel(text), row, 0)
        grid.addWidget(widget, row, 1)
    return grid


def Duration_Text(seconds):
    """A length of time as an operator reads it: seconds, minutes or hours."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def Verdict_Html(verdict, text=''):
    """``verdict`` bold in its colour, then ``text``, for a QLabel."""
    return (f"<b style='color:{theme.VERDICT_COLOURS[verdict]}'>{verdict}</b>  "
            f"{html.escape(text)}")


class Scroll_Column(QScrollArea):
    """A tab's control column, laid out by ``layout``, that scrolls when the
    window is too short for it (2026-10-30): the columns set every tab's
    height, and the window was cut off at the bottom of a screen shorter
    than they are. It never scrolls across: it asks for its controls' width
    and the scroll bar's beside them, so the bar never covers a control."""

    def __init__(self, layout, parent=None):
        super().__init__(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        content.setLayout(layout)
        self.setWidget(content)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # the plots beside it take the width; the column takes the height
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def _Width(self, hint):
        return hint.width() + self.verticalScrollBar().sizeHint().width()

    def sizeHint(self):
        hint = self.widget().sizeHint()
        return QSize(self._Width(hint), hint.height())

    def minimumSizeHint(self):
        return QSize(self._Width(self.widget().minimumSizeHint()),
                     super().minimumSizeHint().height())


class Job_Worker(QThread):

    Progress = Signal(object)
    Done = Signal(object)
    Failed = Signal(str)

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self._Job = job
        self._Stop = threading.Event()

    def Stop(self):
        self._Stop.set()

    def Stopped(self):
        return self._Stop.is_set()

    def Retire(self):
        """Let a finished job's thread go now: wait out the moment between its
        finished signal and its end, then leave it to its last reference,
        rather than post a deferred delete that its parent's own deletion
        could race."""
        self.wait()
        self.setParent(None)

    def run(self):
        try:
            self.Done.emit(self._Job(self.Progress.emit, self._Stop.is_set))
        except Exception as e:
            log.exception("A background job failed")
            self.Failed.emit(str(e))
