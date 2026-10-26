"""The run viewer's analysis: spectra on the plan's bins, every axis carried through the
calibration and the Doppler shift, gates, and the hyperfine model."""

from datetime import datetime

import numpy as np
import pytest

from cls_acquisition import hfs, run_file, spectra
from cls_acquisition.scan_performance import Ion_Frequency

LASER = {'harmonic': 4, 'geometry': 'anti-collinear', 'charge': 1}


def _Run(tmp_path, cal_set=None, cal_readback=None, laser_cm=16956.17):
    """A run of 20 to 40 V in 1 V steps: one hit per setpoint per channel 1
    to 3, a TOF of 50 us, and four DC-counter hits on channel 5."""
    setpoints = np.arange(20.0, 40.5, 1.0)
    cal_set = setpoints if cal_set is None else cal_set
    cal_readback = (1.001 * cal_set - 0.5) if cal_readback is None else cal_readback
    header = {'Run': 7, 'Experiment': 'test', 'Date': datetime(2026, 10, 27, 12),
              'BunchesPerChannel': 1, 'StepSize': 1.0, 'DwellTime': 100,
              'MassAMU': 59.0, 'MassSetPoint': 59.0, 'CoolerVoltage': 2.98848,
              'LaserSetpoint': laser_cm, 'ScanningRanges': [[20.0, 40.0]],
              'CalSet': np.asarray(cal_set, dtype=float),
              'CalReadback': np.asarray(cal_readback, dtype=float) / 1000.0,
              'DacCorrectionApplied': 0, 'DacCorrection': 'none'}
    path = run_file.Create(str(tmp_path), header)
    rows = [[1000.0 + i, volts, i, channel, 50.0, 2.98848]
            for i, volts in enumerate(setpoints) for channel in (1, 2, 3)]
    rows += [[1000.0, 30.0, 10, 5, 0.0, 2.98848]] * 4
    run_file.Append(path, np.array(rows))
    return spectra.Run(path)


def test_Counts_Sit_On_The_Plans_Own_Bins(tmp_path):
    run = _Run(tmp_path)
    assert len(run.Edges) == 22 and run.Edges[0] == pytest.approx(19.5)
    counts = run.Counts([1, 2, 3])
    assert counts.sum() == 63 and np.all(counts == 3)
    assert run.Counts([5])[10] == 4 and run.Channels == [1, 2, 3, 5]


def test_Gates_Pick_Hits_In_Time_Of_Flight_And_In_Time(tmp_path):
    run = _Run(tmp_path)
    assert run.Counts([1], tof_gate=(60.0, 70.0)).sum() == 0
    # the first five setpoints were stepped through in the run's first 4 s
    assert run.Counts([1], time_gate=(0.0, 4.0)).sum() == 5


def test_The_Calibrated_Axis_Moves_The_Edges_And_Keeps_The_Counts(tmp_path):
    run = _Run(tmp_path)
    edges = run.Axis(run.Edges, 'Calibrated voltage', LASER)
    # through each setpoint's own readback, and along the line beyond them
    assert edges == pytest.approx(1.001 * run.Edges - 0.5)


def test_A_Setpoint_Commanded_Twice_Is_Read_As_The_Mean_Of_Its_Readings(tmp_path):
    # the old DAQ's plans commanded a shared endpoint twice (D6)
    cal_set = np.array([20.0, 30.0, 30.0, 40.0])
    run = _Run(tmp_path, cal_set, np.array([20.0, 30.0, 31.0, 40.0]))
    assert run.Cal_Set == pytest.approx([20.0, 30.0, 40.0])
    assert run.Calibrated(30.0) == pytest.approx(30.5)


def test_Beam_Energy_And_Frequency_Follow_The_Cooler_And_The_Doppler_Shift(tmp_path):
    run = _Run(tmp_path)
    energy = run.Axis(run.Edges, 'Beam energy', LASER)
    assert energy == pytest.approx(29884.8 - (1.001 * run.Edges - 0.5))
    reference = run.Reference_Cm(LASER)
    mhz = run.Axis(run.Edges, 'Frequency', LASER, reference)
    expected = (Ion_Frequency(energy, 16956.17 * spectra.CM_HZ, 59.0, LASER)
                - reference * spectra.CM_HZ) / 1e6
    assert mhz == pytest.approx(expected)
    assert abs(mhz[0] + mhz[-1]) < 0.05 * np.ptp(mhz)      # about zero at the middle
    wavenumber = run.Axis(run.Edges, 'Wavenumber', LASER, reference)
    assert wavenumber == pytest.approx(mhz * 1e6 / spectra.CM_HZ)


def test_A_Model_Frequency_Lands_Where_The_Run_Saw_It(tmp_path):
    run = _Run(tmp_path)
    reference = run.Reference_Cm(LASER)
    at_30v = run.Axis([30.0], 'Frequency', LASER, reference)[0]
    for axis in spectra.AXES:
        assert run.Model_Axis([at_30v], axis, LASER, reference)[0] == pytest.approx(
            run.Axis([30.0], axis, LASER, reference)[0], rel=1e-6, abs=1e-6)
    far = run.Model_Axis([at_30v + 1e6], 'Scan voltage', LASER, reference)
    assert np.isnan(far[0])


def test_A_Run_Without_A_Laser_Setpoint_Has_No_Frequency_Axis(tmp_path):
    run = _Run(tmp_path, laser_cm=float('nan'))
    with pytest.raises(spectra.Spectrum_Error, match="no laser setpoint"):
        run.Axis(run.Edges, 'Frequency', LASER)


def test_Wigner_6j_Matches_Its_Closed_Forms():
    assert hfs.Wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1 / 6)
    # {a b c; 0 c b} = (-1)^(a+b+c) / sqrt((2b + 1)(2c + 1))
    for a, b, c in ((1, 0.5, 0.5), (2, 1, 1), (1.5, 1, 0.5), (3, 2, 1)):
        assert hfs.Wigner_6j(a, b, c, 0, c, b) == pytest.approx(
            (-1) ** round(a + b + c) / np.sqrt((2 * b + 1) * (2 * c + 1)))
    assert hfs.Wigner_6j(1, 1, 3, 1, 1, 1) == 0.0          # 1 and 1 cannot make 3


@pytest.mark.parametrize('spin, j_lower, j_upper', [(0.5, 0.5, 0.5), (1.5, 0.5, 1.5),
                                                    (2.5, 1, 2), (3.5, 0.5, 1.5)])
def test_Each_Level_Shares_Strength_By_Its_States(spin, j_lower, j_upper):
    lines = hfs.Lines(spin, j_lower, j_upper, 100.0, 20.0, 50.0, 10.0)
    for end in (0, 1):
        strength = {}
        for line in lines:
            strength[line[end]] = strength.get(line[end], 0.0) + line[3]
        per_state = [s / (2 * f + 1) for f, s in strength.items()]
        assert np.ptp(per_state) == pytest.approx(0.0, abs=1e-12)


def test_A_Spin_Half_Doublet_Splits_As_Its_A_Factors_Say():
    lines = {(fl, fu): (position, intensity)
             for fl, fu, position, intensity in hfs.Lines(0.5, 0.5, 0.5, 1000.0, 0, 300.0, 0)}
    assert set(lines) == {(0, 1), (1, 0), (1, 1)}                 # never 0 to 0
    assert lines[(0, 1)] == pytest.approx((825.0, 0.5))
    assert lines[(1, 1)] == pytest.approx((-175.0, 1.0))
    model = hfs.Spectrum([-175.0, 825.0], list(hfs.Lines(0.5, 0.5, 0.5, 1000.0, 0, 300.0, 0)),
                         0.0, 10.0, 100.0, 5.0)
    assert model == pytest.approx([105.0, 55.0], rel=1e-3)
    with pytest.raises(hfs.Hfs_Error, match="not an electric dipole"):
        hfs.Lines(0.5, 0.5, 2.5, 1, 0, 1, 0)
    with pytest.raises(hfs.Hfs_Error, match="half-whole"):
        hfs.Lines(0.3, 0.5, 0.5, 1, 0, 1, 0)
