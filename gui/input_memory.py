"""What the operator has entered in the window, kept however ORACLE ends.

Date:    2026-10-29
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Each tab names its inputs (``Inputs()``: names and widgets, in the order
they are to be set), and the window hands them here under the tab's
name. As ORACLE opens, each is set to what it held when ORACLE last
stopped, whether it was closed or it crashed. From then on, SAVE_AFTER_MS
after the last change, every input's value is written to ``inputs.yaml``
beside the settings: whole, to a file of its own, then put in its place
by a rename, so whatever stops ORACLE leaves the last complete set and
never half of one.

A missing or unreadable file costs only what it held: every input keeps
its default, and the reason is logged. A value an input no longer takes
(a choice since renamed, a number past its limits) is skipped or held to
the limit, never refused. The settings file stays the operator's to
edit; this one is ORACLE's own.

Depends on: PyYAML, PySide6; gui.plots.
"""

import logging
import os
import tempfile

import yaml
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import (QAbstractButton, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox,
                               QTextEdit)

from gui.plots import Gate_Plot

SAVE_AFTER_MS = 500                 # after the last change: one write for a burst of typing

log = logging.getLogger('oracle.gui')


def _Accessors(widget):
    """``(read, set, the signals it changes by)`` for an input: a widget, a
    Gate_Plot's gate, a tuple of radio buttons (one of them checked), or
    anything with ``Value()``, ``Set_Value()`` and ``Value_Changed`` (the
    scan queue)."""
    if isinstance(widget, tuple):
        return (lambda: next(i for i, button in enumerate(widget) if button.isChecked()),
                lambda i: widget[int(i)].setChecked(True), [b.toggled for b in widget])
    if hasattr(widget, 'Set_Value'):
        return widget.Value, widget.Set_Value, [widget.Value_Changed]
    if isinstance(widget, QSpinBox):
        return widget.value, lambda v: widget.setValue(int(v)), [widget.valueChanged]
    if isinstance(widget, QDoubleSpinBox):
        return widget.value, lambda v: widget.setValue(float(v)), [widget.valueChanged]
    if isinstance(widget, QAbstractButton):
        return widget.isChecked, lambda v: widget.setChecked(bool(v)), [widget.toggled]
    if isinstance(widget, QComboBox):
        return (widget.currentText, lambda v: widget.setCurrentText(str(v)),
                [widget.currentTextChanged])
    if isinstance(widget, QLineEdit):
        return widget.text, lambda v: widget.setText(str(v)), [widget.textChanged]
    if isinstance(widget, QTextEdit):
        return widget.toPlainText, lambda v: widget.setPlainText(str(v)), [widget.textChanged]
    if isinstance(widget, Gate_Plot):
        return (lambda: (gate := widget.Gate()) and list(gate), widget.Set_Gate,
                [widget.Gate_Changed])
    raise TypeError(f"{type(widget).__name__} is not an input the window remembers.")


class Input_Memory(QObject):
    """The inputs given to :meth:`Add`, set from ``path`` and kept there."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._Path = path
        self._Read_Back = {}            # key: read
        self._Last = self._Load()
        self._Timer = QTimer(self, singleShot=True, interval=SAVE_AFTER_MS, timeout=self.Save)

    def _Load(self):
        try:
            with open(self._Path, encoding='utf-8') as fh:
                last = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
            log.warning("The last session's inputs in %s could not be read, so every input "
                        "starts at its default: %s", self._Path, e)
            return {}
        if not isinstance(last, dict):
            log.warning("%s holds a %s, not inputs and their values; every input starts at "
                        "its default", self._Path, type(last).__name__)
            return {}
        return last

    def Add(self, group, inputs):
        """``inputs`` (a tab's ``Inputs()``) under ``group``: each set to its
        last value, in their order, then kept as it changes."""
        for name, widget in inputs:
            key = f"{group}.{name}"
            read, put, signals = _Accessors(widget)
            if key in self._Last:
                try:
                    put(self._Last[key])
                except (TypeError, ValueError, IndexError, AttributeError) as e:
                    log.warning("The last value of %s, %r, was not taken back: %s", key,
                                self._Last[key], e)
            self._Read_Back[key] = read
            for signal in signals:
                signal.connect(self._Changed)

    def _Changed(self, *_):
        self._Timer.start()

    def Save(self):
        """Every input's value now: written whole, then put in place."""
        self._Timer.stop()
        values = {}
        for key, read in self._Read_Back.items():
            try:
                values[key] = read()
            except RuntimeError:
                pass                    # its widget is gone with its window
        folder = os.path.dirname(self._Path)
        temporary = None
        try:
            os.makedirs(folder, exist_ok=True)
            handle, temporary = tempfile.mkstemp(prefix='.inputs_', suffix='.yaml', dir=folder)
            with os.fdopen(handle, 'w', encoding='utf-8') as fh:
                yaml.safe_dump(values, fh, sort_keys=False, allow_unicode=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, self._Path)
        except OSError as e:
            log.warning("The window's inputs were not saved to %s: %s", self._Path, e)
            if temporary is not None and os.path.exists(temporary):
                os.remove(temporary)
