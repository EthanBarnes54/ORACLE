"""ORACLE's test suite: headless, on the simulated rack and stubbed vendor libraries.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Importing this package points every test at a private settings file and
Qt at the offscreen platform, before anything reads either, so no test
can touch the real preferences or open a window. The settings path is
forced, not defaulted: one already set in the shell would otherwise be
the file the tests write. Whatever it was is kept in
REAL_SETTINGS_PATH, for the session guard in conftest to watch.

Depends on: the standard library.
"""

import os as _os
import tempfile as _tempfile

REAL_SETTINGS_PATH = _os.environ.get('ORACLE_SETTINGS_PATH')
_os.environ['ORACLE_SETTINGS_PATH'] = _os.path.join(
    _tempfile.mkdtemp(prefix='oracle_test_settings_'), 'settings.yaml')
_os.environ['QT_QPA_PLATFORM'] = 'offscreen'
_os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PySide6')
