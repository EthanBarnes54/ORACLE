"""A run ORACLE writes, opened as DENIS opens it: by cls_tools and by DENIS's own readers.

Everything ORACLE adds to the archive's format (its record of the run under
run_file.ORACLE_KEY, with the scan type, timing, view, gate verdict, settle
model, laser settings and wired channels) must leave the file readable by
what reads run files. A real scan is taken on the simulated rack, and the
file it writes is opened in DENIS's own environment: cls_tools loads it,
computes its voltages, wavelengths and bins, and DENIS reads its
calibration, its beam header and every key in its tree.

Skipped where DENIS is not installed beside ORACLE (CI); ``ORACLE_DENIS``
names DENIS's folder if it lives elsewhere.
"""

import os
import subprocess

import pytest

from cls_acquisition import run_file
from tests.window import Close_Window, Open_Window, Pump, Run_Files

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DENIS = os.environ.get('ORACLE_DENIS') or next(
    (path for path in (os.path.join(os.path.dirname(REPO), name)
                       for name in ('DENIS_mac', 'DENIS')) if os.path.isdir(path)), '')
PYTHON = os.path.join(DENIS, '.venv', 'Scripts' if os.name == 'nt' else 'bin',
                      'python.exe' if os.name == 'nt' else 'python')

# run in DENIS's environment, from its folder, with the run's path as its argument
OPEN_AS_DENIS = r'''
import os, sys
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import asdf, numpy as np
import clstools
from gui.calibration import (fit_calibration, load_run_calibrated, read_beam_header,
                             read_cal_points)

path = sys.argv[1]
data = clstools.CLSDataFrame()
load_run_calibrated(data, path, {}, cal_order=1)       # as DENIS's merge loads every run
data.Compute_Voltages()
data.Compute_WL(Mass=59.0, harmonic=4)
data.Compute_Bins()
assert len(data.Sorted) == data.Size > 0, 'no events'
assert np.isfinite(data.Sorted['V']).all(), 'a voltage is not finite'
set_v, read_raw = read_cal_points(path)
fit = fit_calibration(set_v, read_raw)
cooler_v, laser_cm = read_beam_header(path)
assert cooler_v > 0 and laser_cm > 0, 'the beam header is unread'
with asdf.open(path) as af:
    for key in af.tree:                 # every key, as any reader that walks the tree
        repr(af.tree[key])
    assert np.asarray(af.tree['raw']).shape[1] == 6
print('opened', data.Size, 'events; gain', round(fit.coeffs_v[1], 5))
'''


# with DENIS not found, PYTHON is a relative path that ORACLE's own .venv can satisfy
@pytest.mark.skipif(not DENIS or not os.path.exists(PYTHON),
                    reason="DENIS is not installed beside ORACLE")
def test_A_Run_ORACLE_Writes_Opens_In_cls_tools_And_DENIS(qapp, monkeypatch, tmp_path):
    window = Open_Window(monkeypatch, tmp_path)
    try:
        window.Scan.Start_Button.click()
        state = window._Link.State
        assert Pump(qapp, lambda: state.Scanning.is_set()), window.Dialogs
        assert Pump(qapp, lambda: not state.Scanning.is_set() and window._Run is None)
    finally:
        Close_Window(qapp, window)
    [path] = Run_Files(window)
    header, _, _ = run_file.Read(path)
    record = header[run_file.ORACLE_KEY]
    # what this version adds, there to be read
    assert record['Laser'] == {'harmonic': 4, 'geometry': 'anti-collinear', 'charge': 1}
    assert list(record['Channels']) == [1, 2, 3, 4]
    opened = subprocess.run([PYTHON, '-c', OPEN_AS_DENIS, path], cwd=DENIS, capture_output=True,
                            text=True, timeout=300)
    assert opened.returncode == 0, opened.stderr[-3000:]
    assert opened.stdout.strip().startswith('opened'), opened.stdout
