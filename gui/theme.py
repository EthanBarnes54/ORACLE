"""One place for the app's look: palette, stylesheet, plot colours and input guards.

Date:    2026-10-29
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

DENIS's dark theme (gui/theme.py there), carried over so the two apps
look and behave as one family: the Fusion dark palette, the group-box
card language with blue titles, the rolodex main tabs, the wheel guard,
and the box of pages titled with each (Title_Tabs), as DENIS's optimiser
tools are. Plots follow DENIS's dark plots: black canvas, white chrome,
neon traces.

The wheel guard matters more here than in DENIS. Every spin box, combo
and slider must be clicked into focus before the wheel changes it, so
scrolling past the quick-tune slider can never move the platform voltage.

Depends on: PySide6.
"""

import os
import sys

from PySide6.QtCore import QEvent, QObject, QRect, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (QAbstractSpinBox, QComboBox, QDial, QGroupBox, QSlider,
                               QStackedWidget, QStyle, QStyleOptionGroupBox, QTabBar,
                               QVBoxLayout, QWidget)

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
    'text_muted': '#a3a3ab',        # a report's labels, beside its figures
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
    return QIcon(os.path.join(ICON_DIR, 'oracle.ico'))


def Logo():
    """The logo at 512 px, for the splash and the About box, as DENIS has them."""
    return QPixmap(os.path.join(ICON_DIR, 'oracle_512.png'))


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


class Title_Tabs(QGroupBox):
    """Pages in one group box whose title row names each page
    (2026-10-25). The box and every title are drawn by the style, as for
    any group box: the first title where a group title goes, the others
    after it, each cutting its gap in the border as a title does. The
    current title is as a group title is; the others are darkened and
    lighten under the cursor; a click on one shows its page. Answers the
    QTabWidget calls its pages are driven by. Carried over from DENIS's
    theme, where it holds the optimiser tools."""

    # The sheets' QGroupBox::title left, and the 4 px the style cuts in
    # the border beyond a title's padding (measured, both themes).
    _LEFT, _CUT = 9, 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._Titles, self._Hover = [], -1
        self._Stack = QStackedWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._Stack)
        self.setMouseTracking(True)

    def addTab(self, widget, text):
        self._Titles.append(text)
        self._Stack.addWidget(widget)
        self.update()

    def count(self):
        return self._Stack.count()

    def widget(self, index):
        return self._Stack.widget(index)

    def tabText(self, index):
        return self._Titles[index]

    def currentIndex(self):
        return self._Stack.currentIndex()

    def currentWidget(self):
        return self._Stack.currentWidget()

    def setCurrentIndex(self, index):
        self._Stack.setCurrentIndex(index)
        self.update()

    def setCurrentWidget(self, widget):
        self.setCurrentIndex(self._Stack.indexOf(widget))

    def _Layout(self):
        """(shift, gap, label) of each title: the style's title rect for
        it, the shift that sets it after the titles before, and the gap
        it cuts in the border once shifted."""
        opt = QStyleOptionGroupBox()
        self.initStyleOption(opt)
        opt.subControls |= QStyle.SubControl.SC_GroupBoxLabel
        out, shift = [], 0
        for text in self._Titles:
            opt.text = text
            label = self.style().subControlRect(
                QStyle.ComplexControl.CC_GroupBox, opt,
                QStyle.SubControl.SC_GroupBoxLabel, self)
            pad = max(0, label.left() - self._LEFT)
            gap = QRect(shift + self._LEFT, label.top(),
                        label.width() + 2 * pad + self._CUT, label.height())
            out.append((shift, gap, label))
            shift += label.right() + 1 + label.left()   # its margin again
        return out

    def _Titled(self, text, height):
        """The top ``height`` px of this box as the style paints it titled
        ``text``, as for any group box."""
        dpr = self.devicePixelRatioF()
        img = QImage(round(self.width() * dpr), round(height * dpr),
                     QImage.Format.Format_ARGB32_Premultiplied)
        img.setDevicePixelRatio(dpr)
        # Over the parent's colour, as on screen: the background's
        # antialiased top edge blends with it.
        img.fill((self.parentWidget() or self).palette().color(
            QPalette.ColorRole.Window))
        p = QPainter(img)
        opt = QStyleOptionGroupBox()
        self.initStyleOption(opt)
        opt.text = text         # initStyleOption leaves an untitled box's
        opt.subControls |= QStyle.SubControl.SC_GroupBoxLabel   # label out
        self.style().drawComplexControl(
            QStyle.ComplexControl.CC_GroupBox, opt, p, self)
        p.end()
        return img

    def minimumSizeHint(self):
        """Wide enough for every title: they are never cropped."""
        hint, layout = super().minimumSizeHint(), self._Layout()
        if layout:
            hint.setWidth(max(hint.width(),
                              layout[-1][1].right() + 1 + self._LEFT))
        return hint

    def paintEvent(self, event):
        """The box titled with the first title, as the style draws it, and
        each other title's gap copied in from the box the style draws
        titled with that one (the style sets its own clip as it draws, so
        a shifted draw cannot be clipped). A title not shown is darkened:
        each row of its glyphs blended toward the colour behind it there,
        sampled from that drawing at the title's padding."""
        layout, cur = self._Layout(), self.currentIndex()
        opt = QStyleOptionGroupBox()
        self.initStyleOption(opt)
        if self._Titles:
            opt.text = self._Titles[0]
            opt.subControls |= QStyle.SubControl.SC_GroupBoxLabel
        p = QPainter(self)
        self.style().drawComplexControl(
            QStyle.ComplexControl.CC_GroupBox, opt, p, self)
        dpr = self.devicePixelRatioF()
        for i, (shift, gap, label) in enumerate(layout):
            img = self._Titled(self._Titles[i], gap.bottom() + 1)
            src = QRect(*(round(v * dpr)
                          for v in gap.translated(-shift, 0).getRect()))
            if i:
                p.drawImage(gap, img, src)
            if i == cur:
                continue
            box = label.translated(shift, 0)
            for y in range(box.top(), box.bottom() + 1):
                shade = img.pixelColor(src.left(), round(y * dpr))
                shade.setAlphaF(0.3 if i == self._Hover else 0.55)
                p.fillRect(QRect(box.left(), y, box.width(), 1), shade)
        p.end()

    def _Title_At(self, pos):
        return next((i for i, (_, gap, _) in enumerate(self._Layout())
                     if gap.contains(pos)), -1)

    def mouseMoveEvent(self, event):
        i = self._Title_At(event.position().toPoint())
        if i != self._Hover:
            self._Hover = i
            self.setCursor(Qt.CursorShape.PointingHandCursor if i >= 0
                           else Qt.CursorShape.ArrowCursor)
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._Hover = -1
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        i = self._Title_At(event.position().toPoint())
        if i >= 0 and event.button() == Qt.MouseButton.LeftButton:
            self.setCurrentIndex(i)
        else:
            super().mousePressEvent(event)
