"""The run file keeps the archive's format, and its numbering survives anything."""

import os
from datetime import datetime

import asdf
import numpy as np
import pytest

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
