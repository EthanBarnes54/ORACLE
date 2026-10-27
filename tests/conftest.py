"""Shared fixtures, and the guards every run of the suite stands behind.

pytest imports the tests package before this file, so its settings path
and offscreen Qt are in place before anything reads them.

As in DENIS, the session fails if a test changed the real settings file
(tests/__init__.py points every test at a private copy; this catches
anything that bypasses it), and if it left run files or session logs in
the repository's own data/ or logs/: a test must write under its
tmp_path, with settings.APP_ROOT pointed there. After each module, any
window a test left open is deleted, so its timers cannot starve the
next module's.
"""

import os

import pytest

import tests
from tests.bench import Loop_Bench

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_SETTINGS = [path for path in (os.path.join(REPO, 'settings', 'settings.yaml'),
                                   tests.REAL_SETTINGS_PATH) if path]


def _Read(path):
    try:
        with open(path, 'rb') as fh:
            return fh.read()
    except OSError:
        return None


def _Listing(folder):
    """Every file under ``folder``, or None if there is no such folder."""
    if not os.path.isdir(folder):
        return None
    return sorted(os.path.relpath(os.path.join(root, name), folder)
                  for root, _, names in os.walk(folder) for name in names)


_START = {path: _Read(path) for path in REAL_SETTINGS}
_FOLDERS = {name: _Listing(os.path.join(REPO, name)) for name in ('data', 'logs')}


def pytest_sessionfinish(session, exitstatus):
    changed = [path for path, before in _START.items() if _Read(path) != before]
    changed += [os.path.join(REPO, name) for name, before in _FOLDERS.items()
                if _Listing(os.path.join(REPO, name)) != before]
    if changed:
        session.exitstatus = 1
        print(f"\nERROR: the test run changed {', '.join(changed)}; a test bypassed "
              f"ORACLE_SETTINGS_PATH or settings.APP_ROOT.")


@pytest.fixture(scope='session')
def qapp():
    from PySide6.QtWidgets import QApplication
    from gui import theme
    app = QApplication.instance() or QApplication([])
    theme.Apply_Theme(app)
    return app


@pytest.fixture
def loop():
    bench = Loop_Bench()
    yield bench
    bench.Close()


@pytest.fixture(autouse=True, scope='module')
def _Quiet_Leftover_Widgets():
    """After each module, stop every timer of the top-level widgets it left
    and hide them. Not close(), which could ask a question and wait for an
    answer; and not deleteLater, as DENIS does: deleting a window from under
    the Python that still holds its pyqtgraph plots crashed the process."""
    yield
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    if QApplication.instance() is None:
        return
    for widget in QApplication.topLevelWidgets():
        try:
            for timer in widget.findChildren(QTimer):
                timer.stop()
            widget.hide()
        except RuntimeError:            # already gone on the C++ side
            pass
