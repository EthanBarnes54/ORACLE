"""Starting ORACLE, refusing to start, and the rules the whole repository keeps.

``main()`` is run as the shortcut runs it, on the simulated rack: the
acquisition process, the meters and the window come up and close down
again, and a second launch raises the first. Each way it can be refused
(unreadable settings, a limit beyond the amplifier) is said in a window,
since a launch from the shortcut has no console. The session log keeps
what a launch leaves behind however it ends. And the repository keeps
one version everywhere and the house style throughout.
"""

import ast
import glob
import os
import re
import subprocess
import sys
import tomllib
import types

import pytest
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from cls_acquisition import settings
from gui import main_window
from tests.window import Pump

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def launch(qapp, monkeypatch, tmp_path):
    """``main()`` ready to run against ``tmp_path``: its own single-instance
    name, no session log over pytest's streams, every dialog kept, and
    ``exec`` replaced by ``check(window)``, which the test sets."""
    monkeypatch.setattr(settings, 'APP_ROOT', str(tmp_path))
    monkeypatch.setenv('ORACLE_SETTINGS_PATH', str(tmp_path / 'settings.yaml'))
    monkeypatch.setattr(main_window.getpass, 'getuser', lambda: f"oracle-test-{os.getpid()}")
    monkeypatch.setattr(main_window.session_log, 'Install',
                        lambda root: os.path.join(root, 'logs', 'oracle_test.log'))
    monkeypatch.setattr(main_window.session_log, 'Install_Qt_Handler', lambda: None)
    monkeypatch.setattr(main_window, 'QApplication', lambda argv: qapp)
    dialogs = []
    monkeypatch.setattr(main_window.QMessageBox, 'critical',
                        lambda parent, title, text, *_: dialogs.append((title, text)))
    state = {'check': None, 'windows': []}

    def run():
        windows = [w for w in QApplication.topLevelWidgets()
                   if isinstance(w, main_window.Main_Window) and w.isVisible()]
        state['windows'] = windows
        state['check'](windows[0])
        return 0

    monkeypatch.setattr(qapp, 'exec', run, raising=False)
    yield types.SimpleNamespace(dialogs=dialogs, state=state)
    server = getattr(qapp, '_Oracle_Server', None)
    if server is not None:
        server.close()
        qapp._Oracle_Server = None
    QLocalServer.removeServer(f"ORACLE-oracle-test-{os.getpid()}")


def test_Oracle_Starts_On_The_Simulated_Rack_And_Closes_Down_Again(qapp, launch):

    def check(window):
        assert window.windowTitle() == f"ORACLE v{main_window.__version__} (simulated rack)"
        assert window.Tabs.count() == 5 and window._Link.Alive()
        # a second launch knocks, and this one is raised instead
        raised = []
        window.raise_ = lambda: raised.append(True)
        knock = QLocalSocket()
        knock.connectToServer(f"ORACLE-oracle-test-{os.getpid()}")
        assert knock.waitForConnected(2000)
        assert Pump(qapp, lambda: raised, timeout=10)
        knock.disconnectFromServer()
        link = window._Link
        window.close()
        assert not link.Alive()

    launch.state['check'] = check
    assert main_window.main() == 0
    assert launch.state['windows'] and launch.dialogs == []


def test_A_Second_Launch_Hands_Over_To_The_First_And_Ends(qapp, launch):
    first = QLocalServer()
    assert first.listen(f"ORACLE-oracle-test-{os.getpid()}")
    try:
        assert main_window.main() == 0
        assert launch.state['windows'] == [] and launch.dialogs == []
    finally:
        first.close()


def test_Unreadable_Settings_Are_Said_In_A_Window_And_Nothing_Starts(qapp, launch,
                                                                     monkeypatch, tmp_path):
    path = tmp_path / 'settings.yaml'
    path.write_text('instruments:\n  tagger_driver: "C:\\Users\\ethan\\wrapper"\n',
                    encoding='utf-8')
    monkeypatch.setenv('ORACLE_SETTINGS_PATH', str(path))
    assert main_window.main() == 1
    [(title, text)] = launch.dialogs
    assert title == 'ORACLE could not start' and 'not valid YAML' in text
    assert 'oracle_test.log' in text and launch.state['windows'] == []


def test_A_Limit_Beyond_The_Amplifier_Is_Refused_Before_Anything_Opens(qapp, launch):
    settings.Update(instruments={'hv_limit_v': 6000.0})
    assert main_window.main() == 1
    [(title, text)] = launch.dialogs
    assert title == 'ORACLE could not start' and "Trek 609E-6's rating" in text
    assert launch.state['windows'] == []


def test_The_Session_Log_Keeps_What_A_Launch_Leaves_However_It_Ends(tmp_path):
    code = f"""
import logging, os, sys, threading, warnings
sys.path.insert(0, {REPO!r})
from gui import session_log
root = {str(tmp_path)!r}
os.makedirs(os.path.join(root, 'logs'))
for i in range(25):
    old = os.path.join(root, 'logs', f'oracle_old_{{i:02d}}.log')
    open(old, 'w').close()
    os.utime(old, (1e9 + i, 1e9 + i))
path = session_log.Install(root)
print('the banner, µ and all')
logging.getLogger('oracle.test').warning('a logged warning')
warnings.warn('a Python warning')
doomed = threading.Thread(target=lambda: 1 / 0, name='doomed')
doomed.start()
doomed.join()
print('PATH=' + path, flush=True)
raise RuntimeError('the launch ends badly')
"""
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=60,
                          env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    assert done.returncode != 0
    [path] = re.findall(r'PATH=(.+)', done.stdout.decode('utf-8'))
    with open(path.strip(), encoding='utf-8') as fh:
        logged = fh.read()
    for piece in ('the banner, µ and all', 'a logged warning', 'a Python warning',
                  "Uncaught exception in thread 'doomed'", 'ZeroDivisionError',
                  'Uncaught exception', 'RuntimeError: the launch ends badly'):
        assert piece in logged
    kept = sorted(os.listdir(os.path.join(str(tmp_path), 'logs')))
    # the newest 20 before it, and its own
    assert len(kept) == 21 and 'oracle_old_04.log' not in kept and 'oracle_old_05.log' in kept


# ── The repository's own rules ──────────────────────────────────

def _Version_In(path, pattern):
    with open(os.path.join(REPO, path), encoding='utf-8') as fh:
        return re.search(pattern, fh.read(), re.MULTILINE).group(1)


def test_One_Version_Everywhere():
    with open(os.path.join(REPO, 'pyproject.toml'), 'rb') as fh:
        version = tomllib.load(fh)['project']['version']
    assert main_window.__version__ == version
    assert _Version_In('README.md', r'^\*\*Version:\*\* (\S+)') == version
    assert _Version_In('CITATION.cff', r'^version: (\S+)') == version
    for package in ('cls_acquisition/__init__.py', 'gui/__init__.py'):
        assert _Version_In(package, r'^Version: +(\S+)') == version


def _Sources(*folders):
    paths = [os.path.join(REPO, 'gui.py')]
    for folder in folders:
        paths += glob.glob(os.path.join(REPO, folder, '**', '*.py'), recursive=True)
    return sorted(paths)


def test_Every_Line_Fits_In_99_Characters():
    """Characters, not bytes: the section rules are drawn with '─'."""
    long = []
    for path in _Sources('cls_acquisition', 'gui', 'tests'):
        with open(path, encoding='utf-8') as fh:
            long += [f"{os.path.relpath(path, REPO)}:{n}" for n, line in enumerate(fh, 1)
                     if len(line.rstrip('\n')) > 99]
    assert long == []


def test_Every_Module_Carries_Its_Header():
    missing = []
    for path in _Sources('cls_acquisition', 'gui'):
        with open(path, encoding='utf-8') as fh:
            doc = ast.get_docstring(ast.parse(fh.read())) or ''
        missing += [f"{os.path.relpath(path, REPO)}: {key}" for key in
                    ('Date:', 'Version:', 'Author:  Ethan Barnes', 'Depends on:')
                    if key not in doc]
    assert missing == []


def test_No_Failure_Is_Swallowed_Unseen_And_The_Core_Never_Prints():
    """A bare except would hide KeyboardInterrupt and SystemExit as well; the
    core logs, so the acquisition process's word reaches the session log."""
    bare, prints = [], []
    for path in _Sources('cls_acquisition', 'gui'):
        with open(path, encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        name = os.path.relpath(path, REPO)
        bare += [f"{name}:{node.lineno}" for node in ast.walk(tree)
                 if isinstance(node, ast.ExceptHandler) and node.type is None]
        # the wrapper's build script is a console tool, run by hand on the rack PC
        if name.startswith('cls_acquisition') and not name.endswith('build_wrapper.py'):
            prints += [f"{name}:{node.lineno}" for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'print']
    assert bare == [] and prints == []
