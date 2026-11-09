"""The systematics' inputs a run's analysis gives: each measured, declared or not yet known."""

import numpy as np
import pytest
import yaml

from cls_acquisition import scan_performance as sp
from cls_acquisition import systematics
from tests.test_scan_performance import LASER, ORACLE_KEY, _Run

DECLARED = {'beam_offset_v': {'CEC': [13.0, 8.0], 'LCR': [-30.0, 5.0]}, 'zero_drift_v': 0.030,
            'wavemeter_mhz': 10.0}
DIVIDERS = {'Scan': {'ratio': 1000.0, 'ppm': 100.0}, 'Cooler': {'ratio': 10000.0, 'ppm': 100.0}}


def _Rows(result):
    return {row['Source']: row for row in result['Systematics']}


def _Analysed(region='CEC', legacy=False, declared=DECLARED, **run):
    header, raw, monitor = _Run(legacy=legacy, **run)
    if not legacy:
        header[ORACLE_KEY].update(InteractionRegion=region, Dividers=DIVIDERS,
                                  ReferenceRun=True)
    return header, sp.Analyse(header, raw, monitor, LASER, declared=declared)


def test_Every_Source_Is_There_And_Says_Whether_It_Was_Measured_Or_Declared():
    header, result = _Analysed(calibration=lambda v: 1.0002 * v + 0.3
                               + np.random.default_rng(1).normal(0.0, 0.05, len(v)))
    rows = _Rows(result)
    assert [r['Source'] for r in result['Systematics']] == [
        'Beam offset δU', 'Cooler divider', 'Scan divider', 'Run calibration',
        'Bench table repeatability', '0 V drift during the run', 'HV against its calibration',
        'Wavemeter during the run', 'Cooler scatter', 'Wavemeter accuracy', 'Afterpulsing',
        'Health']
    kinds = {name: row['Kind'] for name, row in rows.items()}
    assert kinds['Beam offset δU'] == kinds['Cooler divider'] == systematics.DECLARED
    assert kinds['0 V drift during the run'] == kinds['Wavemeter accuracy'] == systematics.DECLARED
    assert kinds['Run calibration'] == kinds['Cooler scatter'] == systematics.MEASURED
    assert kinds['Bench table repeatability'] == systematics.UNKNOWN
    # the CEC's declared offset, and a run tagged as a reference run says so
    assert rows['Beam offset δU']['Volts'] == 8.0 and '+13 ± 8 V' in rows['Beam offset δU'][
        'Detail'] and 'reference run' in rows['Beam offset δU']['Detail']
    assert rows['Cooler divider']['Volts'] == pytest.approx(29950.0 * 100e-6)
    assert rows['0 V drift during the run']['Volts'] == 0.030
    # every width in MHz at the ions through the run's own Doppler shift
    per_volt = rows['Beam offset δU']['Mhz'] / 8.0
    for row in result['Systematics']:
        if row['Volts'] is not None:
            assert row['Mhz'] == pytest.approx(row['Volts'] * per_volt)
    assert systematics.Largest(result['Systematics'])['Source'] == 'Beam offset δU'


def test_The_Calibrations_Errors_Are_Its_Straight_Lines_At_The_Scans_Ends():
    _, result = _Analysed(calibration=lambda v: 1.0002 * v + 0.3
                          + np.random.default_rng(1).normal(0.0, 0.05, len(v)))
    detail = _Rows(result)['Run calibration']['Detail']
    plan = np.unique(np.arange(0.0, 41.0))
    readback = 1.0002 * plan + 0.3 + np.random.default_rng(1).normal(0.0, 0.05, len(plan))
    (gain, offset), cov = np.polyfit(plan, readback, 1, cov=True)
    assert f"offset ± {np.sqrt(cov[1, 1]):.3f} V" in detail
    assert f"gain ± {np.sqrt(cov[0, 0]) * 1e6:.0f} ppm" in detail


def test_What_The_Run_Cannot_Say_Is_Not_Yet_Known_Rather_Than_Guessed():
    # an old DAQ run: no monitor, no record of its region or dividers
    _, result = _Analysed(legacy=True)
    rows = _Rows(result)
    for name in ('Beam offset δU', 'Cooler divider', 'Scan divider', 'HV against its calibration',
                 'Wavemeter during the run', 'Bench table repeatability'):
        assert rows[name]['Kind'] == systematics.UNKNOWN and rows[name]['Volts'] is None, name
    assert 'interaction region was not stated' in rows['Beam offset δU']['Detail']
    # nothing declared: no offset for a region the settings do not give
    _, result = _Analysed(region='LCR', declared={})
    rows = _Rows(result)
    assert rows['Beam offset δU']['Kind'] == systematics.UNKNOWN
    assert rows['0 V drift during the run']['Kind'] == systematics.UNKNOWN


def test_An_Afterpulsing_Pmt_Is_Listed_With_Its_Suggested_Dead_Time():
    _, result = _Analysed(scans=150, afterpulse={3: 0.05})
    detail = _Rows(result)['Afterpulsing']['Detail']
    assert detail.startswith('channel 3:') and '5 µs dead time suggested' in detail
    _, clean = _Analysed()
    assert _Rows(clean)['Afterpulsing']['Detail'].startswith('no PMT afterpulses')


def test_The_Elog_Gets_A_Short_Block_And_The_Whole_Table_On_Request(tmp_path):
    _, result = _Analysed(scans=150, afterpulse={3: 0.05})
    short = systematics.Elog_Text(result['Systematics'], 'run_0007')
    lines = short.splitlines()
    assert lines[0] == ("SYSTEMATIC INPUTS, run_0007 (estimated by ORACLE; the full table is "
                        "on Scan Performance)")
    assert lines[1].startswith("The straight line through the run's calibration: offset ±")
    assert lines[2].startswith("0 V drift: 0.030 V") and '(declared)' in lines[2]
    assert lines[3].startswith("Cooler scatter: ") and lines[4].startswith("Afterpulsing: ")
    assert lines[5].startswith("Largest term: Beam offset δU, 8.000 V,")
    full = systematics.Elog_Text(result['Systematics'], 'run_0007', full=True)
    assert all(f"{row['Source']} ({row['Kind']})" in full for row in result['Systematics'])
    written = systematics.Export(str(tmp_path / 'run_0007.asdf'), result['Systematics'])
    with open(written, encoding='utf-8') as fh:
        exported = yaml.safe_load(fh)
    assert written.endswith('run_0007_systematics.yaml')
    assert [s['source'] for s in exported['sources']] == [
        r['Source'] for r in result['Systematics']]


def test_A_Run_Fitted_To_The_Bench_Table_Gives_That_Fits_Errors():
    header, raw, monitor = _Run()
    header[ORACLE_KEY].update(ScanTable='ab' * 32, TableFit={
        'Table': 'ab' * 32, 'A_V': -0.99, 'A_Err_V': 0.004, 'B': 1.00002, 'B_Err': 3e-6,
        'Cov_AB': 0.0, 'Rms_V': 0.0115, 'Reach_T_V': 2000.0})
    rows = _Rows(sp.Analyse(header, raw, monitor, LASER, declared=DECLARED))
    calibration = rows['Run calibration']
    assert calibration['Kind'] == systematics.MEASURED
    assert calibration['Detail'].startswith('Fitted to bench table abababab: a ± 4.0 mV')
    # at the scan's furthest point: a's error and b's over 2 kV, 7.2 mV together
    assert calibration['Volts'] == pytest.approx(np.hypot(0.004, 3e-6 * 2000.0))
    assert 'bench table abababab was in force' in rows['Bench table repeatability'][
        'Detail'].lower()
