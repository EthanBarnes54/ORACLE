"""Shared fixtures: the Qt application, and an acquisition loop on a simulated rack.

pytest imports the tests package before this file, so its settings path
and offscreen Qt are in place before anything reads them.
"""

import pytest

from tests.bench import Loop_Bench


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
