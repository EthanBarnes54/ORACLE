"""What has to hold on Windows and on macOS alike.

ORACLE runs on the rack's Windows PC and on a Mac for simulated work,
from the same code, and the suite runs on both. Here: the clocks the two
processes share, the request that keeps the PC awake, the sentences for
a vendor library a system cannot load, settings written with Windows
folders, line endings, folders with spaces and accents (OneDrive's), the
installers' own files, and the rule that the acquisition process starts
without Qt or any vendor library. Each OS's own branch runs where that
OS does; the other's is driven through stand-ins.
"""

import ctypes
import multiprocessing
import os
import subprocess
import sys
import time
import types

import numpy as np
import pytest

from cls_acquisition import hardware_calibration as hc
from cls_acquisition import keep_awake, run_file, scan_performance, settings
from cls_acquisition.hardware import mcc_dac, simulated, visa_session

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = ('pyvisa', 'mcculw', 'spinapi', 'timetagger4', 'epics', 'elog', 'pcaspy')


def _Child_Clock(queue):
    queue.put(time.perf_counter_ns())


# ── One clock for both processes ────────────────────────────────

def test_Both_Processes_Read_One_Clock():
    """Calibration stamps a write in the acquisition process and its reading
    in the window's, and the simulated platform's step is timed across the
    two: perf_counter must be one system-wide clock, as it is on Windows
    (QueryPerformanceCounter) and macOS (mach_absolute_time)."""
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    child = context.Process(target=_Child_Clock, args=(queue,))
    before = time.perf_counter_ns()
    child.start()
    stamp = queue.get(timeout=30)
    after = time.perf_counter_ns()
    child.join(timeout=10)
    assert before < stamp < after


def test_The_Simulated_Rack_Times_Its_Step_Finer_Than_A_Millisecond():
    """A 0.4 ms rise needs a clock finer than Windows' 15.6 ms tick, which
    time.monotonic had there before Python 3.13."""
    assert simulated.CLOCK is time.perf_counter
    assert time.get_clock_info('perf_counter').resolution < 1e-5
    platform = simulated.Platform(multiprocessing.get_context('spawn'))
    platform.Set(1000.0)
    started = time.perf_counter()
    while time.perf_counter() - started < 0.0005:
        pass
    assert 100.0 < platform.Volts() < 1000.0           # part way up, not a 15.6 ms staircase


def test_The_Simulated_0_V_Level_Warms_Up_And_Then_Wanders():
    """0.2 V low at power-on, settled within the first hour, then wandering
    by 10-20 mV over tens of minutes, as the bench's 0 V checks did; a step
    leaves it as it was."""
    assert simulated.Warm_Up(0.0) == pytest.approx(-0.2, abs=1e-9)
    hour = np.arange(3600.0, 3 * 3600.0, 60.0)
    later = np.array([simulated.Warm_Up(t) for t in hour])
    assert np.all(np.abs(later) < 0.025) and 0.01 < np.ptp(later) < 0.04
    platform = simulated.Platform(multiprocessing.get_context('spawn'))
    platform.Set(1000.0)
    time.sleep(0.05)
    t = simulated.CLOCK()
    assert platform.Volts(t) - 1000.0 == pytest.approx(
        simulated.Warm_Up(t - platform._On), abs=0.01)


# ── Keeping the PC awake ────────────────────────────────────────

@pytest.mark.skipif(sys.platform != 'darwin', reason="macOS's own power assertions")
def test_On_macOS_The_Request_Is_A_Power_Assertion_Of_This_Process():
    def mine():
        listing = subprocess.run(['pmset', '-g', 'assertions'], capture_output=True,
                                 text=True).stdout
        return [line for line in listing.splitlines()
                if f"pid {os.getpid()}(" in line and keep_awake.REASON in line]

    awake = keep_awake.Keep_Awake()
    awake.Hold(True)
    try:
        assert any('PreventUserIdleSystemSleep' in line for line in mine())
    finally:
        awake.Hold(False)
    assert mine() == [] and not awake._Warned


@pytest.mark.skipif(sys.platform != 'win32', reason="Windows' own execution state")
def test_On_Windows_The_Request_Is_Taken():
    awake = keep_awake.Keep_Awake()
    awake.Hold(True)
    awake.Hold(False)
    assert not awake._Warned


def test_On_Windows_The_Thread_Asks_To_Stay_Awake_And_Lets_Go(monkeypatch):
    states = []
    kernel32 = types.SimpleNamespace(SetThreadExecutionState=lambda state: states.append(
        state.value) or 1)
    monkeypatch.setattr(ctypes, 'windll', types.SimpleNamespace(kernel32=kernel32),
                        raising=False)
    awake = keep_awake.Keep_Awake('win32')
    for on in (True, True, False, False, True):
        awake.Hold(on)
    continuous, system = keep_awake.ES_CONTINUOUS, keep_awake.ES_SYSTEM_REQUIRED
    assert states == [continuous | system, continuous, continuous | system]


def test_A_Request_That_Fails_Is_Said_Once_And_Never_Stops_A_Run(monkeypatch, caplog):
    monkeypatch.setattr(ctypes, 'windll', types.SimpleNamespace(), raising=False)
    awake = keep_awake.Keep_Awake('win32')
    with caplog.at_level('WARNING', logger='oracle.keep_awake'):
        for on in (True, False, True, False):
            awake.Hold(on)
    assert caplog.text.count('could not be kept awake') == 1


def test_Elsewhere_Nothing_Is_Asked():
    awake = keep_awake.Keep_Awake('linux')
    awake.Hold(True)
    assert awake.Held and not awake._Warned
    awake.Hold(False)


# ── Vendor libraries a system cannot load ───────────────────────

@pytest.mark.parametrize('module, open_device, sentence', [
    ('mcculw', lambda: mcc_dac.Open_Mcc(1), "run only on Windows"),
    ('spinapi', lambda: __import__('cls_acquisition.hardware.pulseblaster',
                                   fromlist=['x']).Pulse_Blaster(), "SpinCore's SpinAPI"),
    ('timetagger4', lambda: __import__('cls_acquisition.hardware.timetagger',
                                       fromlist=['x']).Time_Tagger([0.1] * 4, 2.5),
     "it is a Windows module"),
])
def test_A_Library_This_System_Cannot_Load_Is_Named_With_The_Fix(monkeypatch, module,
                                                                 open_device, sentence):
    monkeypatch.setitem(sys.modules, module, None)     # import fails as on a Mac
    with pytest.raises(Exception, match=sentence) as raised:
        open_device()
    assert 'rack: simulated' in str(raised.value) or 'tagger_driver' in str(raised.value)


def test_No_Visa_Library_Is_Named_With_The_Fix(monkeypatch):

    def no_library():
        raise ValueError("Could not locate a VISA implementation. Install either the IVI "
                         "binary or pyvisa-py.")

    monkeypatch.setitem(sys.modules, 'pyvisa', types.SimpleNamespace(ResourceManager=no_library))
    monkeypatch.setattr(visa_session, '_Manager', None)
    with pytest.raises(visa_session.Instrument_IO_Error, match="NI-VISA or Keysight IO"):
        visa_session.Pyvisa_Connect('USB0::0x2A8D::0x0101::MY57506790::INSTR')


# ── Settings written by hand ────────────────────────────────────

@pytest.fixture
def settings_file(monkeypatch, tmp_path):
    path = tmp_path / 'settings.yaml'
    monkeypatch.setenv('ORACLE_SETTINGS_PATH', str(path))
    return path


WRAPPER = 'C:\\ORACLE\\cls_acquisition\\hardware\\timetagger4_wrapper'


@pytest.mark.parametrize('written', [WRAPPER, f"'{WRAPPER}'", WRAPPER.replace('\\', '/')])
def test_A_Windows_Folder_Written_Bare_Quoted_Once_Or_Forward_Is_Read(settings_file, written):
    settings_file.write_text(f"instruments:\n  tagger_driver: {written}\n", encoding='utf-8')
    assert settings.Load()['instruments']['tagger_driver'].replace('/', '\\') == WRAPPER


@pytest.mark.parametrize('content, says', [
    ('instruments:\n  tagger_driver: "C:\\Users\\ethan\\wrapper"\n'.encode(), 'double quotes'),
    (b'- rack\n- hardware\n', 'holds a list'),
    ('experiment: Ge-So\xb5\n'.encode('latin-1'), 'not UTF-8'),
])
def test_A_Settings_File_ORACLE_Cannot_Read_Is_Refused_With_A_Sentence(settings_file, content,
                                                                      says):
    settings_file.write_bytes(content)
    with pytest.raises(settings.Settings_Error, match=says) as raised:
        settings.Load()
    assert str(settings_file) in str(raised.value)


def test_A_Folder_Saved_From_Windows_Reads_Back_As_It_Was(settings_file):
    folder = 'C:\\Users\\Ethan Barnes\\OneDrive - The University of Manchester\\ORACLE data'
    settings.Update(data_directory=folder, experiment='µ-scan Ge')
    assert settings.Load()['data_directory'] == folder
    assert settings.Load()['experiment'] == 'µ-scan Ge'


# ── Files ───────────────────────────────────────────────────────

def _Header(run):
    from datetime import datetime
    return {'Run': run, 'Experiment': 'IS700 µ', 'Date': datetime(2026, 10, 27, 9, 30),
            'BunchesPerChannel': 2, 'StepSize': 1.0, 'DwellTime': 10, 'MassAMU': 59.0,
            'MassSetPoint': 59.0, 'CoolerVoltage': 2.98848, 'LaserSetpoint': 16956.17,
            'ScanningRanges': [[0.0, 40.0]], 'CalSet': np.arange(0.0, 41.0, 10.0),
            'CalReadback': np.arange(0.0, 41.0, 10.0) / 1000, 'DacCorrectionApplied': 0,
            'DacCorrection': 'none'}


def test_A_Data_Folder_With_Spaces_And_Accents_Holds_Runs_And_Records(tmp_path):
    folder = tmp_path / 'OneDrive - The University of Manchester' / 'Données CLS'
    path = run_file.Create(str(folder), _Header(7))
    rows = np.column_stack([np.full(41, 1.7e9), np.arange(41.0) % 41, np.arange(41.0) * 2,
                            np.ones(41), np.full(41, 52.0), np.full(41, 2.98848)])
    run_file.Append(path, rows)
    run_file.Append_Monitor(path, [[1.7e9, 1, 0.0, 0.1, 29884.8, 16956.17, 82, 0, 0]])
    assert run_file.Runs(str(folder)) == [path] and run_file.Last_Run_Number(str(folder)) == 7
    header, raw, monitor = run_file.Read(path)
    assert header['Experiment'] == 'IS700 µ' and np.array_equal(raw, rows)
    assert monitor['bunches'][0] == 82
    record = hc.Record(str(folder), 'linearity', {'Note': 'ü'}, hc.LINEARITY_COLUMNS)
    record.Add(['sweep'], ['up'], [10.0], [10.2], [0.01], [0], [0.01], [1.0])
    record.Close('completed', {'Gain': 1.0})
    assert os.path.dirname(record.Path) == os.path.join(str(folder), 'hardware_calibration')


def test_Records_And_Monitors_End_Their_Lines_The_Same_On_Every_System(tmp_path):
    """newline='' and '\\n' throughout: a CSV written on the rack's PC reads
    the same on a Mac, with no '\\r' for a reader to trip on."""
    path = run_file.Create(str(tmp_path), _Header(1))
    run_file.Append_Monitor(path, [[1.0] * len(run_file.MONITOR_COLS)] * 3)
    record = hc.Record(str(tmp_path), 'cooler', {'Rack': 'simulated'}, hc.COOLER_COLUMNS)
    record.Add([0.0, 0.001], [29884.8, 29884.9])
    record.Close('completed', {'Mean_V': 29884.85})
    for written in (run_file.Monitor_Path(path), record.Path):
        with open(written, 'rb') as fh:
            content = fh.read()
        assert b'\r' not in content and content.endswith(b'\n')


def test_The_Installers_Keep_Their_Own_Line_Endings():
    with open(os.path.join(REPO, '.gitattributes'), encoding='utf-8') as fh:
        rules = fh.read()
    assert '*.sh        text eol=lf' in rules and '*.bat       text eol=crlf' in rules
    with open(os.path.join(REPO, 'install.bat'), 'rb') as fh:
        batch = fh.read()
    with open(os.path.join(REPO, 'install.sh'), 'rb') as fh:
        shell = fh.read()
    # cmd.exe misreads labels and goto in an LF-only batch file; bash cannot run CRLF
    assert batch.count(b'\r\n') == batch.count(b'\n') and b'\r' not in shell
    assert shell.startswith(b'#!/usr/bin/env bash\n')
    if os.name == 'posix':
        assert os.access(os.path.join(REPO, 'install.sh'), os.X_OK)
    # and the shortcut lands on the Desktop Windows shows, OneDrive's if it is backed up
    assert b"GetFolderPath('Desktop')" in batch and b'attrib +P -U' in batch


def test_The_Performance_Report_Is_Written_Beside_Its_Run_As_This_System_Names_It(tmp_path):
    path = run_file.Create(str(tmp_path / 'With Space'), _Header(3))
    result = {'Verdict': 'WARN', 'Summary': '', 'Budget': '', 'Sections': {},
              'Flagged': ['Beam below half its usual level from 1.0 to 2.0 min into the run.']}
    written = scan_performance.Export(path, result)
    import yaml
    with open(written, encoding='utf-8') as fh:
        report = yaml.safe_load(fh)
    assert os.path.dirname(written) == os.path.dirname(path)
    assert report['run'] == os.path.basename(path) and report['flagged'] == result['Flagged']


# ── Starting the acquisition process ────────────────────────────

def _Python(code):
    return subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True,
                          text=True, timeout=120)


def test_The_Core_Loads_Neither_Qt_Nor_A_Vendor_Library():
    """The acquisition process is spawned and imports the core: Qt would slow
    every start and a vendor library would tie it to one system."""
    done = _Python(
        "import importlib, pkgutil, sys\n"
        "import cls_acquisition, cls_acquisition.hardware\n"
        "for package in (cls_acquisition, cls_acquisition.hardware):\n"
        "    for info in pkgutil.iter_modules(package.__path__):\n"
        "        importlib.import_module(f'{package.__name__}.{info.name}')\n"
        f"print(sorted(m for m in sys.modules if m.split('.')[0] in "
        f"{('PySide6', 'pyqtgraph') + VENDOR!r}))\n")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == '[]'


def test_The_Launcher_Loads_No_Qt_When_A_Spawned_Process_Reads_It():
    """A spawned process runs the launcher again as __mp_main__ (on Windows
    always, and on macOS since spawn is ORACLE's start method)."""
    done = _Python("import runpy, sys\n"
                   "runpy.run_path('gui.py', run_name='__mp_main__')\n"
                   "print('PySide6' in sys.modules)\n")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'False'


# ── The window on each system ───────────────────────────────────

def test_On_Windows_The_Taskbar_Shows_Oracle_And_Elsewhere_Nothing_Is_Asked(monkeypatch):
    from gui import main_window
    asked = []
    shell32 = types.SimpleNamespace(SetCurrentProcessExplicitAppUserModelID=asked.append)
    monkeypatch.setattr(ctypes, 'windll', types.SimpleNamespace(shell32=shell32), raising=False)
    monkeypatch.setattr(main_window.sys, 'platform', 'win32')
    main_window._Windows_App_Id()
    assert asked == [main_window.APP_ID]
    monkeypatch.setattr(main_window.sys, 'platform', 'darwin')
    main_window._Windows_App_Id()
    assert asked == [main_window.APP_ID]
    # the same identity the macOS bundle carries
    with open(os.path.join(REPO, 'install.sh'), encoding='utf-8') as fh:
        assert f"<string>{main_window.APP_ID}</string>" in fh.read()


@pytest.mark.parametrize('system, asked', [('darwin', True), ('win32', False)])
def test_On_macOS_The_Title_Bar_And_Dialogs_Are_Dark_And_Windows_Is_Left_Be(
        qapp, monkeypatch, system, asked):
    """As DENIS has it: dark on macOS, whose title bar would otherwise follow
    the system's light mode; untouched on Windows, where the same call also
    darkens Windows 11's title bar."""
    from PySide6.QtCore import Qt
    from gui import theme
    schemes = []
    hints = types.SimpleNamespace(setColorScheme=schemes.append)
    monkeypatch.setattr(type(qapp), 'styleHints', staticmethod(lambda: hints))
    monkeypatch.setattr(theme.sys, 'platform', system)
    theme.Apply_Theme(qapp)
    assert schemes == ([Qt.ColorScheme.Dark] if asked else [])


def test_No_Wait_Cursor_Is_Ever_Set():
    """Qt draws its wait cursor from an image on macOS, and making it just
    after the Mac woke from sleep crashed DENIS inside CoreGraphics."""
    for folder in ('gui', 'cls_acquisition'):
        for root, _, names in os.walk(os.path.join(REPO, folder)):
            for name in names:
                if name.endswith('.py'):
                    with open(os.path.join(root, name), encoding='utf-8') as fh:
                        assert 'setOverrideCursor' not in fh.read(), name
