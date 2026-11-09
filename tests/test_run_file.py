"""The run file keeps the archive's format, and its numbering survives anything."""

import os
from datetime import datetime

import asdf
import numpy as np
import pytest
from asdf import AsdfFile

from cls_acquisition import run_file


def _Header(run=7, when=datetime(2026, 10, 25, 12, 0)):
    return {'Run': run, 'Experiment': 'test', 'Date': when, 'BunchesPerChannel': 5,
            'StepSize': 1.0, 'DwellTime': 50, 'MassAMU': 59.0, 'MassSetPoint': np.nan,
            'CoolerVoltage': 2.995, 'LaserSetpoint': 10920.5, 'ScanningRanges': [[0.0, 60.0]],
            'CalSet': np.arange(3.0), 'CalReadback': np.arange(3.0) / 1000,
            'DacCorrectionApplied': 0, 'DacCorrection': 'none'}


def test_Header_And_Rows_Read_Back_As_The_Archive_Has_Them(tmp_path):
    path = run_file.Create(str(tmp_path), _Header())
    assert path.endswith(os.path.join('2026', '10', '25', 'run_0007.asdf'))
    rows = np.arange(12.0).reshape(2, 6)
    run_file.Append(path, rows)
    run_file.Append(path, rows + 12)
    with open(path, 'rb') as fh:
        assert b'#ASDF_STANDARD 1.5.0' in fh.read(200)
    with asdf.open(path) as f:
        stored = {k for k in f.tree if not k.startswith(('asdf_library', 'history'))}
        assert stored == set(run_file.HEADER_KEYS) | {'raw', 'raw_header'}
        assert f['raw_header'] == run_file.DATA_COLS
        assert np.array_equal(f['raw'][:], np.vstack([rows, rows + 12]))
        assert np.array_equal(f['CalSet'], np.arange(3.0))


def test_The_Format_Cannot_Drift(tmp_path):
    header = _Header()
    header['NewKey'] = 1
    with pytest.raises(ValueError, match="NewKey"):
        run_file.Create(str(tmp_path), header)
    del header['NewKey'], header['DwellTime']
    with pytest.raises(ValueError, match="DwellTime"):
        run_file.Create(str(tmp_path), header)


def test_A_Run_File_Is_Never_Overwritten(tmp_path):
    run_file.Create(str(tmp_path), _Header())
    with pytest.raises(FileExistsError):
        run_file.Create(str(tmp_path), _Header())


def test_Numbering_Takes_The_Highest_Run_In_Any_Folder(tmp_path):
    assert run_file.Last_Run_Number(str(tmp_path)) == 0
    run_file.Create(str(tmp_path), _Header(7120, datetime(2026, 1, 15)))
    run_file.Create(str(tmp_path), _Header(7080, datetime(2026, 1, 16)))
    run_file.Create(str(tmp_path), _Header(12000, datetime(2025, 12, 1)))
    assert run_file.Last_Run_Number(str(tmp_path)) == 12000


def test_Oracles_Record_Sits_Beside_The_Archives_Keys_And_Reads_Back_As_Written(tmp_path):
    record = {'Version': '0.2.0', 'ScanType': 'Triangle',
              'Timing': {'bunched': False, 'rep_rate_hz': 1000.0, 'release_us': 100,
                         'tagger_delay_us': 0, 'gate_start_us': 0.0, 'gate_stop_us': 200.0},
              'View': {'tof_bin_us': 0.5, 'tof_gate_us': None, 'channels': [1, 3], 'map': True},
              'Calibration': {'Verdict': 'WARN', 'Checks': [{'Check': 'settle',
                                                              'Verdict': 'WARN', 'Message': 'm'}]},
              'SettleModel': None}
    path = run_file.Create(str(tmp_path), _Header(), record)
    run_file.Append(path, np.arange(12.0).reshape(2, 6))
    with asdf.open(path) as f:
        stored = {k for k in f.tree if not k.startswith(('asdf_library', 'history'))}
        assert stored == set(run_file.HEADER_KEYS) | {run_file.ORACLE_KEY, 'raw', 'raw_header'}
    header, raw, _ = run_file.Read(path)
    assert header[run_file.ORACLE_KEY] == record and raw.shape == (2, 6)
    older = run_file.Create(str(tmp_path), _Header(8))      # one written before the record
    assert run_file.ORACLE_KEY not in run_file.Read(older)[0]


def _Old_Daq_Run(directory, run, when=datetime(2026, 10, 25, 12, 0)):
    """A run as the old DAQ wrote it: no DAC correction keys."""
    folder = os.path.join(directory, f"{when.year}", f"{when.month:02d}", f"{when.day:02d}")
    os.makedirs(folder, exist_ok=True)
    tree = {'Run': run, 'CalSet': np.arange(3.0), 'CalReadback': np.arange(3.0) / 1000}
    AsdfFile(tree).write_to(os.path.join(folder, f"run_{run:04d}.asdf"))


def test_A_Calibration_Is_Set_Only_Against_Runs_Calibrated_The_Same_Way(tmp_path):
    # the old DAQ paired some readings with the wrong setpoint, and a DAC correction moves
    # every setpoint: across either, two calibrations differ with nothing wrong in either
    directory = str(tmp_path)
    _Old_Daq_Run(directory, 1)
    _Old_Daq_Run(directory, 2)
    paths = {}
    for run, correction in ((3, 'none'), (4, 'measured 2026-11-01'), (5, 'none'), (6, 'none')):
        header = _Header(run)
        header['DacCorrection'] = correction
        paths[run] = run_file.Create(directory, header)
    assert [c[0] for c in run_file.Earlier_Calibrations(paths[6], directory)] == [5, 3]
    assert [c[3] for c in run_file.Earlier_Calibrations(paths[6], directory)] == [None, None]
    assert run_file.Earlier_Calibrations(paths[4], directory) == []
    old = paths[3].replace('run_0003', 'run_0002')
    assert [c[0] for c in run_file.Earlier_Calibrations(old, directory)] == [1]


def test_An_Earlier_Run_Is_Read_With_The_Divider_It_Recorded(tmp_path):
    directory = str(tmp_path)
    record = {'Dividers': {'Scan': {'model': 'KV-10', 'serial': '', 'ratio': 1000.2,
                                    'ppm': 100.0}}}
    run_file.Create(directory, _Header(1), record)
    later = run_file.Create(directory, _Header(2))
    [(run, _, _, divider)] = run_file.Earlier_Calibrations(later, directory)
    assert (run, divider) == (1, 1000.2)
