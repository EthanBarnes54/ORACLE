"""ORACLE's test suite: headless, on the simulated rack and stubbed vendor libraries.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Importing this package points every test at a private settings file and
Qt at the offscreen platform, before anything reads either, so no test
can touch the real preferences or open a window.

Depends on: the standard library.
"""

import os as _os
import tempfile as _tempfile

_os.environ.setdefault('ORACLE_SETTINGS_PATH', _os.path.join(
    _tempfile.mkdtemp(prefix='oracle_test_settings_'), 'settings.yaml'))
_os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PySide6')
