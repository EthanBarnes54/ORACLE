"""Shared GUI building blocks: inputs, layouts, reports and the worker thread.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

What more than one tab builds alike. :func:`Report_Html` lays results
out as titled sections of aligned figures, small tables, verdict badges
and the saved files as links. :class:`Scroll_Column` lets the window fit
a short screen. :class:`Job_Worker` runs ``job(progress, stopped)`` off
the GUI thread: its result arrives in ``Done``, an exception in ``Failed``
(and the log).

Depends on: PySide6; gui.theme.
"""

import html
import logging
import os
import threading
from collections import namedtuple

from PySide6.QtCore import QSize, Qt, QThread, QUrl, Signal
from PySide6.QtWidgets import (QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLayout, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout,
                               QWidget)

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


def Figure(value, spec, unit=''):
    """``value`` formatted by ``spec``, with its ``unit``, and a true minus sign."""
    return f"{value:{spec}}".replace('-', '−') + (f" {unit}" if unit else '')


class Html(str):
    """Text already marked up, which a report puts in as it is."""


# figures in columns, in a report: a header and rows of cells, the first column
# names and the rest figures
Report_Table = namedtuple('Report_Table', 'columns rows')


def _Text(text):
    return text if isinstance(text, Html) else html.escape(str(text))


def _Grid(table):
    muted = theme.TOKENS['text_muted']
    aligns = ['left'] + ['right'] * (len(table.columns) - 1)
    head = ''.join(f"<td align='{a}' style='color:{muted}'>{_Text(c)}</td>"
                   for a, c in zip(aligns, table.columns))
    body = ''.join('<tr>' + ''.join(f"<td align='{a}'>{_Text(c)}</td>"
                                    for a, c in zip(aligns, row)) + '</tr>'
                   for row in table.rows)
    return (f"<table width='100%' cellspacing='0' cellpadding='1'><tr>{head}</tr>{body}"
            f"</table>")


def Report_Html(sections):
    """A report for a QLabel: ``sections`` of ``(heading, rows)``, a heading a
    title or ``(verdict, title)``; a row a ``(label, value)`` pair (values
    right-aligned), a sentence, or a :class:`Report_Table`."""
    muted, title = theme.TOKENS['text_muted'], theme.TOKENS['accent_title']
    out = []
    for heading, rows in sections:
        verdict, heading = heading if isinstance(heading, tuple) else (None, heading)
        badge = Verdict_Badge(verdict) if verdict else ''
        gap = 'padding-top:8px' if out else ''
        out.append(f"<tr><td style='color:{title}; font-weight:600; {gap}'>{_Text(heading)}"
                   f"</td><td align='right' style='{gap}'>{badge}</td></tr>")
        for row in rows:
            if isinstance(row, Report_Table):
                out.append(f"<tr><td colspan='2'>{_Grid(row)}</td></tr>")
            elif isinstance(row, str):
                out.append(f"<tr><td colspan='2'>{_Text(row)}</td></tr>")
            else:
                label, value = row
                out.append(f"<tr><td style='color:{muted}'>{_Text(label)}</td>"
                           f"<td align='right'>{_Text(value)}</td></tr>")
    return f"<table width='100%' cellspacing='0' cellpadding='1'>{''.join(out)}</table>"


def Muted(text):
    """``text`` in the labels' colour, as a report's footnote or detail."""
    return Html(f"<span style='color:{theme.TOKENS['text_muted']}'>{html.escape(text)}</span>")


def Verdict_Badge(verdict):
    """``verdict`` bold in its colour, as a report's value."""
    return Html(f"<b style='color:{theme.VERDICT_COLOURS[verdict]}'>{verdict}</b>")


def Saved_Section(paths):
    """The files a measurement saved, as a report's section: each by its
    name, a link that opens it; the folder they are in is its tooltip's."""
    accent = theme.TOKENS['accent']
    return ('Saved', [Html(f"<a href='{QUrl.fromLocalFile(path).toString()}' "
                           f"style='color:{accent}; text-decoration:none'>"
                           f"{html.escape(os.path.basename(path))}</a>") for path in paths])


def Report_Label(text):
    """A label for a report: wrapped, its links opened by the system."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setOpenExternalLinks(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    return label


class Scroll_Column(QScrollArea):
    """A tab's control column (``layout``) that scrolls when the window is too
    short for it, never across: it asks for its controls' width plus the
    scroll bar's, so the bar covers nothing."""

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


def Row(*widgets, parent=None):
    """``widgets`` side by side, in ``parent`` if given."""
    row = QHBoxLayout(parent) if parent is not None else QHBoxLayout()
    for widget in widgets:
        row.addWidget(widget)
    return row


def Side_By_Side(page, controls, plots):
    """``page`` laid out as every tab is: ``controls``, (widget or layout,
    stretch) pairs, in a scrolling column, and ``plots`` beside them taking
    the rest; returns the column."""
    column = QVBoxLayout()
    for item, stretch in controls:
        (column.addLayout if isinstance(item, QLayout) else column.addWidget)(item, stretch)
    side = Scroll_Column(column)
    layout = QHBoxLayout(page)
    layout.addWidget(side)
    layout.addWidget(plots, 1)
    return side


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
        """Let a finished job's thread go: wait out the gap between its finished
        signal and its end, then leave it to its last reference (a deferred
        delete could race its parent's deletion)."""
        self.wait()
        self.setParent(None)

    def run(self):
        try:
            self.Done.emit(self._Job(self.Progress.emit, self._Stop.is_set))
        except Exception as e:
            log.exception("A background job failed")
            self.Failed.emit(str(e))
