"""ORACLE's PySide6 interface.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The main window and its tabs, the theme, the session log and the plots.
Binds pyqtgraph to PySide6 before any module imports it.

Depends on: PySide6, pyqtgraph; cls_acquisition.
"""

import os

os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PySide6')
