"""The rack's history (WP10), read back from a data folder as ORACLE keeps it."""

import os
from datetime import datetime

import numpy as np
import pytest
import yaml

from cls_acquisition import history, run_file
from tests.test_run_file import _Header

RACK = {'Kind': 'hardware', 'Dac': {'Board': 'USB-231', 'UniqueId': 'A1'}}


def _Run(directory, run, when, fit, rack, reports=None):
    path = run_file.Create(str(directory), _Header(run, when),
                           {'TableFit': fit, 'RackIdentity': rack})
    if reports:
        for kind, record in reports.items():
            with open(path[:-len('.asdf')] + f"_{kind}.yaml", 'w', encoding='utf-8') as fh:
                yaml.safe_dump(record, fh)
    return path


@pytest.fixture
def data(tmp_path):
    _Run(tmp_path, 1, datetime(2026, 11, 1, 10), {'A_V': -0.95, 'B': 1.00002,
                                                  'Scale_Mode': 'free'}, RACK)
    _Run(tmp_path, 2, datetime(2026, 11, 2, 10), {'A_V': -0.91, 'B': 1.0, 'Scale_Mode': 'bench'},
         dict(RACK, Dac={'Board': 'USB-231', 'UniqueId': 'B2'}), {
             'systematics': {'sources': [{'id': 'cooler_scatter', 'volts': 0.31}]},
             'performance': {'detectors': {'channel 3': {'afterpulse_share': 0.021}}}})
    _Run(tmp_path, 3, datetime(2026, 11, 3, 10), 'No bench table in force.', None)
    records = tmp_path / 'hardware_calibration'
    records.mkdir()
    for name, header in (('settle_2026-10-30_090000.csv', {}),
                         ('settle_traces_2026-10-30_090000.csv', {}),   # a companion, not shown
                         ('linearity_2026-10-31_090000.csv', {'Purpose': 'table check'})):
        (records / name).write_text(''.join(f"# {k}: {v}\n" for k, v in header.items())
                                    + 'a,b\n1,2\n', encoding='utf-8')
    (records / 'health_2026-11-02_080000.yaml').write_text(
        yaml.safe_dump({'Verdicts': {'Cooler': 'PASS', 'Counts': 'WARN'}}), encoding='utf-8')
    return tmp_path


def test_Each_Run_Is_Set_On_The_Timeline_With_What_It_Kept(data):
    found = history.Collect(str(data))
    assert list(found['Run']) == [1, 2, 3]
    assert found['Time'][0] == datetime(2026, 11, 1, 10).timestamp()
    assert found['A_V'][:2] == pytest.approx([-0.95, -0.91]) and np.isnan(found['A_V'][2])
    # b only where the calibration fitted it; the bench's b of a narrow one is not a measurement
    assert found['B_Ppm'][0] == pytest.approx(20.0) and np.isnan(found['B_Ppm'][1:]).all()
    assert np.isnan(found['Cooler_V'][0]) and found['Cooler_V'][1] == 0.31
    assert list(found['Afterpulse']) == [3] and list(found['Afterpulse'][3][1]) == [0.021]


def test_A_Change_Of_Rack_Is_Marked_At_The_First_Run_On_The_New_One(data):
    [(when, run, changes)] = history.Collect(str(data))['Racks']
    assert run == 2 and when == datetime(2026, 11, 2, 10).timestamp()
    assert any('A1 → B2' in change for change in changes)


def test_The_Bench_Records_Are_Named_By_What_They_Were_For(data):
    bench = [what for _, what in history.Collect(str(data))['Bench']]
    assert bench == ['Settle time', 'Bench table check', 'Health check, WARN']


def test_An_Empty_Data_Folder_Has_No_History(tmp_path):
    found = history.Collect(str(tmp_path))
    assert len(found['Run']) == 0 and found['Bench'] == [] and found['Racks'] == []
    assert os.listdir(tmp_path) == []
