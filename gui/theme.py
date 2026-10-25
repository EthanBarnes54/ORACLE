"""One place for the app's look: palette, stylesheet, plot colours and input guards.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

DENIS's dark theme (gui/theme.py there), carried over so the two apps
look and behave as one family: the Fusion dark palette, the group-box
card language with blue titles, the rolodex main tabs, and the wheel
guard. Plots follow DENIS's dark plots: black canvas, white chrome,
neon traces.

The wheel guard matters more here than in DENIS. Every spin box, combo
and slider must be clicked into focus before the wheel changes it, so
scrolling past the quick-tune slider can never move the platform voltage.

Depends on: PySide6.
"""

import os
import sys

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import (QAbstractSpinBox, QComboBox, QDial, QSlider, QTabBar,
                               QWidget)

ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'icons')

TOKENS = {
    'accent': '#42a5f5',            # links, selection, focus, checked marks
    'accent_title': '#6cb6ff',      # group titles
    'select_bg': '#2f5f8f',
    'window': '#2d2d30',
    'field': '#26262a',
    'field_disabled': '#2b2b2d',
    'raised': '#3a3a3f',
    'raised_hover': '#46464c',
    'raised_pressed': '#2a2a2e',
    'border': '#4d4d54',
    'border_soft': '#4a4a4f',
    'border_hover': '#62626a',
    'text': '#e0e0e0',
    'text_disabled': '#777777',
    'group_bg': '#333338',
    'tooltip_bg': '#2a2a2e',
    'tooltip_border': '#5a5a60',
}

# DENIS's dark plots
PLOT_BACKGROUND = '#0d0d0d'
PLOT_FOREGROUND = '#ffffff'
PLOT_GRID_ALPHA = 0.25
TRACE = '#00E5FF'                   # neon cyan, DENIS's first run colour
TRACE_ALT = '#FFE500'               # neon yellow, for a second trace
GATE_FILL = (255, 165, 0, 77)       # DENIS's TOF gate span: orange at 0.3
VERDICT_COLOURS = {'PASS': '#7CFF3F', 'WARN': '#FFE500', 'REJECT': '#FF5C5C', 'FAIL': '#FF5C5C',
                   'N/A': TOKENS['text_disabled']}


def Icon(name):
    """A Lucide icon from icons/lucide, or an empty icon if it is missing."""
    path = os.path.join(ICON_DIR, 'lucide', f'{name}.svg')
    return QIcon(path) if os.path.exists(path) else QIcon()


def App_Icon():
    return QIcon(os.path.join(ICON_DIR, 'oracle_512.png'))


def _Icon_Url(name):
    return '"' + os.path.join(ICON_DIR, 'lucide', f'{name}.svg').replace('\\', '/') + '"'


def Build_Palette():
    p = QPalette()
    for role, rgb in ((QPalette.ColorRole.Window, (45, 45, 48)),
                      (QPalette.ColorRole.WindowText, (220, 220, 220)),
                      (QPalette.ColorRole.Base, (30, 30, 32)),
                      (QPalette.ColorRole.AlternateBase, (45, 45, 48)),
                      (QPalette.ColorRole.ToolTipBase, (42, 42, 46)),
                      (QPalette.ColorRole.ToolTipText, (224, 224, 224)),
                      (QPalette.ColorRole.Text, (220, 220, 220)),
                      (QPalette.ColorRole.Button, (53, 53, 53)),
                      (QPalette.ColorRole.ButtonText, (220, 220, 220)),
                      (QPalette.ColorRole.BrightText, (255, 64, 64)),
                      (QPalette.ColorRole.Link, (66, 165, 245)),
                      (QPalette.ColorRole.Highlight, (66, 165, 245)),
                      (QPalette.ColorRole.HighlightedText, (0, 0, 0))):
        p.setColor(role, QColor(*rgb))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(127, 127, 127))
    return p


def Build_Qss():
    t = dict(TOKENS, check_icon=_Icon_Url('check'), chevron_up=_Icon_Url('chevron-up'),
             chevron_down=_Icon_Url('chevron-down'))
    return """
QMenu { background-color: %(window)s; border: 1px solid #555; padding: 4px; }
QMenu::item { padding: 5px 24px 5px 20px; color: #dddddd; border-radius: 2px; }
QMenu::item:selected { background-color: #4a4a4a; color: #ffffff; font-weight: bold; }
QMenu::item:disabled { color: %(text_disabled)s; }
QMenu::separator { height: 1px; background-color: #444; margin: 4px 8px; }
QMenuBar { background: transparent; color: %(text)s; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background: %(raised_hover)s; border-radius: 3px; }
QDialog { background-color: %(window)s; }
QToolTip {
    background-color: %(tooltip_bg)s; color: %(text)s;
    border: 1px solid %(tooltip_border)s; padding: 5px 8px;
}

QPushButton {
    background-color: %(raised)s; border: 1px solid #5a5a60;
    padding: 4px 12px; border-radius: 3px; color: %(text)s;
}
QPushButton:hover { background-color: %(raised_hover)s; border-color: #6e6e76; }
QPushButton:pressed { background-color: %(raised_pressed)s; }
QPushButton:focus, QPushButton:default { border: 1px solid %(accent)s; }
QPushButton:disabled {
    color: %(text_disabled)s; background-color: #333336; border-color: #44444a;
}

QLineEdit, QPlainTextEdit, QTextEdit {
    background-color: %(field)s; border: 1px solid %(border)s; border-radius: 3px;
    padding: 2px 5px; color: %(text)s;
    selection-background-color: %(accent)s; selection-color: #000000;
}
QLineEdit:hover, QPlainTextEdit:hover, QTextEdit:hover { border-color: %(border_hover)s; }
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus { border: 1px solid %(accent)s; }
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {
    color: %(text_disabled)s; background-color: %(field_disabled)s; border-color: #44444a;
}

QAbstractSpinBox {
    background-color: %(field)s; border: 1px solid %(border)s; border-radius: 3px;
    padding: 1px 4px; color: %(text)s;
    selection-background-color: %(accent)s; selection-color: #000000;
}
QAbstractSpinBox:hover { border-color: %(border_hover)s; }
QAbstractSpinBox:focus { border: 1px solid %(accent)s; }
QAbstractSpinBox:disabled {
    color: %(text_disabled)s; background-color: %(field_disabled)s; border-color: #44444a;
}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    subcontrol-origin: border; width: 16px; background: #33333a;
    border-left: 1px solid %(border)s;
}
QAbstractSpinBox::up-button { subcontrol-position: top right; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {
    background: %(raised_hover)s;
}
QAbstractSpinBox::up-arrow { image: url(%(chevron_up)s); width: 10px; height: 10px; }
QAbstractSpinBox::down-arrow { image: url(%(chevron_down)s); width: 10px; height: 10px; }
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off,
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off { image: none; }

QComboBox {
    background-color: %(field)s; border: 1px solid %(border)s; border-radius: 3px;
    padding: 2px 6px; color: %(text)s; min-height: 18px;
}
QComboBox:hover { border-color: %(border_hover)s; }
QComboBox:focus { border: 1px solid %(accent)s; }
QComboBox:disabled {
    color: %(text_disabled)s; background-color: %(field_disabled)s; border-color: #44444a;
}
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right; width: 20px;
    border-left: 1px solid %(border)s; background: #33333a;
    border-top-right-radius: 3px; border-bottom-right-radius: 3px;
}
QComboBox::down-arrow { image: url(%(chevron_down)s); width: 11px; height: 11px; }
QComboBox QAbstractItemView {
    background-color: %(tooltip_bg)s; color: %(text)s;
    border: 1px solid %(tooltip_border)s;
    selection-background-color: %(select_bg)s; selection-color: #ffffff; outline: 0;
}

QCheckBox, QRadioButton { color: %(text)s; spacing: 6px; }
QCheckBox:disabled, QRadioButton:disabled { color: %(text_disabled)s; }
QCheckBox::indicator {
    width: 15px; height: 15px; border: 1px solid #62626a; border-radius: 3px;
    background: %(field)s;
}
QCheckBox::indicator:hover { border-color: #8a8a92; }
QCheckBox::indicator:checked {
    background-color: %(accent)s; border-color: %(accent)s; image: url(%(check_icon)s);
}
QCheckBox::indicator:disabled { background: %(field_disabled)s; border-color: #44444a; }
QRadioButton::indicator {
    width: 14px; height: 14px; border: 1px solid #62626a; border-radius: 8px;
    background: %(field)s;
}
QRadioButton::indicator:hover { border-color: #8a8a92; }
QRadioButton::indicator:checked {
    border: 1px solid %(accent)s;
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.55, fx:0.5, fy:0.5,
                                stop:0 %(accent)s, stop:0.55 %(accent)s,
                                stop:0.7 %(field)s, stop:1 %(field)s);
}
QRadioButton::indicator:disabled { background: %(field_disabled)s; border-color: #44444a; }

QGroupBox {
    border: 1px solid %(border_soft)s; border-radius: 4px; margin-top: 10px;
    padding: 7px 6px 6px 6px; background-color: %(group_bg)s;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left; left: 9px;
    padding: 0 5px; color: %(accent_title)s; font-weight: bold;
}

QScrollBar:vertical { background: #242428; width: 12px; margin: 0; border: none; }
QScrollBar::handle:vertical {
    background: #4a4a52; min-height: 28px; border-radius: 5px; margin: 2px;
}
QScrollBar::handle:vertical:hover { background: #5f5f68; }
QScrollBar:horizontal { background: #242428; height: 12px; margin: 0; border: none; }
QScrollBar::handle:horizontal {
    background: #4a4a52; min-width: 28px; border-radius: 5px; margin: 2px;
}
QScrollBar::handle:horizontal:hover { background: #5f5f68; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QSlider::groove:horizontal { height: 4px; background: #3a3a40; border-radius: 2px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; margin: -5px 0; border-radius: 7px; background: #d0d0d5;
    border: 1px solid #666;
}
QSlider::handle:horizontal:hover { background: #ffffff; }
QSlider::sub-page:horizontal:disabled { background: #44444a; }
QSlider::handle:horizontal:disabled { background: #6a6a70; }

QSplitter::handle { background: #2c2c30; }
QStatusBar { border-top: 1px solid #3f3f45; }
QStatusBar::item { border: none; }
""" % t


def Main_Tabs_Qss():
    """DENIS's rolodex sheet for the top-level ``#MainTabs``, set on that
    widget so nested tab widgets keep the plain look."""
    return (
        "QTabWidget#MainTabs::pane {"
        "  border-top: 1px solid #404040; background: transparent; top: -1px; }"
        "QTabWidget#MainTabs > QTabBar { background: transparent; }"
        "QTabWidget#MainTabs > QTabBar::tab {"
        "  padding: 4px 18px; margin: 3px 0 0 0; color: #6e6e6e; background: #232323;"
        "  border: 1px solid #303030; border-bottom: none;"
        "  border-top-left-radius: 4px; border-top-right-radius: 4px; }"
        "QTabWidget#MainTabs > QTabBar::tab:!selected:hover {"
        "  color: #b0b0b0; background: #2c2c2c; margin-top: 2px; }"
        "QTabWidget#MainTabs > QTabBar::tab:selected {"
        "  color: #ffffff; background: #353535; border: 1px solid #404040;"
        "  border-bottom: 2px solid #4287c2; margin-top: 0px; margin-bottom: -1px; }")


# ── Wheel guard ──────────────────────────────────────────────────────────
# Widgets whose value must not change from a hover-scroll. Scroll bars are
# left out on purpose: they exist to be wheeled.
_GUARDED_TYPES = (QAbstractSpinBox, QComboBox, QSlider, QDial, QTabBar)
_GUARD_MARK = '_oracleWheelGuarded'


class _Wheel_Blocker(QObject):
    """Swallows the wheel unless the widget has focus, leaving the event to
    its parents, so an enclosing scroll area still scrolls."""

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget)
                and not obj.hasFocus()):
            event.ignore()
            return True
        return False


class Wheel_Focus_Guard(QObject):
    """Arms each guarded widget as it is first polished: wheel focus becomes
    strong focus, and the blocker is installed on it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._Blocker = _Wheel_Blocker(self)

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.Polish and isinstance(obj, _GUARDED_TYPES)
                and not obj.property(_GUARD_MARK)):
            obj.setProperty(_GUARD_MARK, True)
            obj.installEventFilter(self._Blocker)
            if (not isinstance(obj, QTabBar)
                    and obj.focusPolicy() == Qt.FocusPolicy.WheelFocus):
                obj.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        return False


def Apply_Theme(app):
    """Fusion, the dark palette and stylesheet, and the wheel guard."""
    app.setStyle('Fusion')
    if sys.platform == 'darwin':
        # the title bar and native dialogs follow the dark theme
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    app.setPalette(Build_Palette())
    app.setStyleSheet(Build_Qss())
    app._Oracle_Wheel_Guard = Wheel_Focus_Guard(app)
    app.installEventFilter(app._Oracle_Wheel_Guard)
