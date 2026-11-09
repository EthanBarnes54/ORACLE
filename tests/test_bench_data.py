"""ORACLE's analyses on the rack's own bench records, against what the bench tool made of them.

The bench calibration tool (cls-cal) measured the CLS rack on 24 to 26
October 2026; tests/data/bench holds its readings and its own results.
ORACLE's port of each analysis must give cls-cal's answer on cls-cal's
data, the converter codes ORACLE writes must be the board's, the
correction design must hold on two days' sweeps, and the simulated rack
must be the rack those records describe.
"""

import json
import os

import numpy as np
import pytest

from cls_acquisition import hardware_calibration as hc
from cls_acquisition.calibration import THRESHOLDS
from cls_acquisition.hardware import mcc_dac, simulated
from cls_acquisition.hardware.keysight import COOLER_DIVIDER
from cls_acquisition.hardware.mcc_dac import CH_NEG, CH_POS

BENCH = os.path.join(os.path.dirname(__file__), 'data', 'bench')
KINDS = ('sweep', 'zero')
DIRECTIONS = ('', 'up', 'down', 'random')

with open(os.path.join(BENCH, 'reference.json')) as _fh:
    REFERENCE = json.load(_fh)


def _Load(name):
    return np.load(os.path.join(BENCH, name))


def _Points(name):
    """The bench sweep as the points Linearity_Sweep hands its analysis."""
    d = _Load(name)
    return [{'Kind': KINDS[k], 'Direction': DIRECTIONS[w], 'Set_V': float(s),
             'Read_V': float(r), 'Sigma_V': float(e), 'Channel': CH_POS if s >= 0 else CH_NEG,
             'Dac_V': 0.0, 'Time_S': float(t)}
            for k, w, s, r, e, t in zip(d['kind'], d['direction'], d['set_v'], d['read_v'],
                                        d['sigma_v'], d['time_s'])]


def _Settle_Results():
    """Each bench step through ORACLE's Step_Response, as Settle_Test reports it."""
    d, tolerance = _Load('settle.npz'), REFERENCE['settle']['config']['tolerance_v']
    interval = float(d['interval_s'])
    results = []
    for trace, start, stop, command in zip(d['hv_v'], d['from_v'], d['to_v'], d['command_s']):
        response = hc.Step_Response(interval * np.arange(len(trace)), trace, tolerance)
        results.append(dict(response, Step=0, Mode='out', Rep=1, From_V=float(start),
                            To_V=float(stop), Lag_S=response['T10_S'] - command,
                            Tail_Shift_V=np.nan))
    return results, interval


# ── The ports reproduce the bench tool ──────────────────────────

def test_The_Linearity_Analysis_Gives_The_Bench_Tools_Answer_On_Its_Own_Sweep():
    ours, theirs = hc.Linearity_Analysis(_Points('linearity_1kv.npz')), REFERENCE['linearity_1kv']
    exact = dict(rel=1e-9, abs=1e-9)
    assert ours['Gain'] == pytest.approx(theirs['gain'], **exact)
    assert ours['Offset_V'] == pytest.approx(theirs['offset_v'], **exact)
    assert ours['Residual_Rms_V'] == pytest.approx(theirs['residual_rms_v'], **exact)
    assert ours['Residual_Max_V'] == pytest.approx(theirs['residual_max_abs_v'], **exact)
    assert ours['Inl_Ppm_Fs'] == pytest.approx(theirs['inl_ppm_fs'], **exact)
    assert ours['Quadratic_V_Per_Kv2'] == pytest.approx(theirs['quadratic_v_per_kv2'], **exact)
    # the same uncertainties: the residual's scatter over n - 2 degrees of freedom
    assert ours['Gain_Err'] == pytest.approx(theirs['gain_err'], rel=1e-9)
    assert ours['Offset_Err_V'] == pytest.approx(theirs['offset_err_v'], rel=1e-9)
    for channel, name in hc.BRANCHES:
        fit, branch = ours['Fits'][channel], theirs['branches'][name]
        assert fit['Gain'] == pytest.approx(branch['gain'], **exact)
        assert fit['Offset_V'] == pytest.approx(branch['offset_v'], **exact)
        assert fit['Rms_V'] == pytest.approx(branch['residual_rms_v'], **exact)
    assert ours['Seam_V'] == pytest.approx(theirs['branches']['zero_jump_v'], **exact)
    assert ours['Gain_Mismatch_Ppm'] == pytest.approx(theirs['branches']['gain_mismatch_ppm'],
                                                      rel=1e-6)
    # the largest hysteresis is the same pair; the mean leaves out 0 V, which is on
    # neither converter, where cls-cal kept it
    assert ours['Hysteresis']['Max_V'] == pytest.approx(theirs['hysteresis_max_abs_v'], **exact)
    assert ours['Zero_Drift']['Span_V'] == pytest.approx(theirs['zero_drift_span_v'], **exact)
    assert ours['Zero_Drift']['Slope_V_Per_H'] == pytest.approx(
        theirs['zero_drift_slope_v_per_h'], rel=1e-9)


def test_The_Step_Analysis_Gives_The_Bench_Tools_Answer_Step_By_Step():
    results, interval = _Settle_Results()
    for ours, theirs in zip(results, REFERENCE['settle']['steps']):
        # the two take the level before the step from slightly different windows,
        # which moves the interpolated crossings by microseconds of a 500 us sample
        assert ours['T10_S'] == pytest.approx(theirs['t10_s'], abs=0.05 * interval)
        assert ours['Rise_S'] == pytest.approx(theirs['rise_10_90_s'], abs=0.05 * interval)
        assert ours['Overshoot_Pct'] == pytest.approx(theirs['overshoot_pct'], abs=0.01)
        # cls-cal counts a step already in its band at the 10 % crossing as settled
        # at once; ORACLE at the first sample after the crossing, at most one later
        slack = 0.05 * interval
        assert -slack <= ours['Settle_S'] - theirs['settle_s'] <= interval + slack


def test_The_Bench_Supports_Settle_And_The_Settle_Model_Says_How_Far():
    results, _ = _Settle_Results()
    verdict, sentence = hc.Settle_Summary(results)
    assert verdict == 'PASS', sentence
    model = dict(hc.Settle_Model(results)['points'])
    # the bench's 2 kV jumps settled inside 0.5 V 20.5 ms after the 10 % crossing,
    # a few ms after the command: far inside SETTLE
    assert 0.020 < model[2000.0] < THRESHOLDS['SettleS']
    assert model[10.0] <= model[100.0] <= model[1000.0] <= model[2000.0]
    # the bench tool's own recommendation for the linearity sweep's wait
    assert hc.Recommended_Wait(results) == pytest.approx(
        REFERENCE['settle']['summary']['recommended_wait_s'])


def test_The_Noise_Analysis_Gives_The_Bench_Tools_Answer_On_The_Cooler():
    d, theirs = _Load('cooler.npz'), REFERENCE['cooler']
    config = theirs['config']
    ours = hc.Noise_Analysis(d['cooler_v'], float(d['interval_s']), config['aperture_s'],
                             config['line_freq_hz'], config['n_peaks'])
    for key, name in (('Count', 'n'), ('Mean_V', 'mean_v'), ('Peak_To_Peak_V', 'ptp_v'),
                      ('Rms_Detrended_V', 'rms_detrended_v'),
                      ('Peak_To_Peak_Detrended_V', 'ptp_detrended_v'),
                      ('Oscillation_Rms_V', 'oscillation_rms_v'),
                      ('Random_Rms_V', 'random_noise_rms_v'),
                      ('Noise_Floor_V_Rt_Hz', 'noise_floor_v_per_rthz')):
        assert ours[key] == pytest.approx(theirs[name], rel=1e-9), key
    # the drift is a fit's slope through 60000 readings: a few ulp of the level
    assert ours['Drift_V_Per_Min'] == pytest.approx(theirs['drift_v_per_min'], abs=1e-7)
    assert len(ours['Peaks']) == len(theirs['peaks'])
    for peak, reference in zip(ours['Peaks'], theirs['peaks']):
        assert peak['Freq_Hz'] == pytest.approx(reference['freq_hz'], rel=1e-9)
        assert peak['Amplitude_V'] == pytest.approx(reference['amplitude_v'], rel=1e-9)
        assert peak['Snr'] == pytest.approx(reference['snr'], rel=1e-9)
        assert peak['Mains_Harmonic'] == reference['mains_harmonic']
        assert peak['Integration_Gain'] == pytest.approx(reference['integration_gain'])
    assert [b['Rms_V'] for b in ours['Bands']] == pytest.approx(
        [b['rms_v'] for b in theirs['bands']], rel=1e-9)
    # what the cooler monitor reported on the bench: the mains at 50 Hz, 0.44 V
    mains = ours['Peaks'][0]
    assert mains['Mains_Harmonic'] == 1 and mains['Amplitude_V'] == pytest.approx(0.44, abs=0.01)


# ── The converters and the correction ───────────────────────────

@pytest.mark.parametrize('name', ['linearity_1kv.npz', 'linearity_3kv_25oct.npz',
                                  'linearity_3kv_26oct.npz'])
def test_Every_Write_Lands_On_The_Code_The_Board_Took(name):
    """Every converter voltage cls-cal asked of the USB-231, and the code
    the board's own conversion gave it: ORACLE's rounding must be the board's."""
    d = _Load(name)
    codes = np.array([[mcc_dac.Quantised(v)[0] for v in pair] for pair in d['dac_v']])
    assert np.array_equal(codes, d['dac_code'])
    # and the Dac writes those codes for the setpoints, uncorrected, on the branch cls-cal did
    for set_v, pair, code in zip(d['set_v'][::25], d['dac_v'][::25], d['dac_code'][::25]):
        writes = []
        mcc_dac.Dac(lambda channel, volts: writes.append((channel, volts))).Write(set_v)
        channel, volts = writes[-1]
        assert mcc_dac.Quantised(volts)[0] == code[channel]
        assert volts == pytest.approx(pair[channel], abs=mcc_dac.LSB_V / 2 + 1e-12)


def _Negative_Branch():
    """Both days' sweeps over the part of the negative branch both covered."""
    days = []
    for name in ('linearity_3kv_25oct.npz', 'linearity_3kv_26oct.npz'):
        d = _Load(name)
        days.append((d['set_v'], d['read_v']))
    low = max(day[0].min() for day in days)
    high = min(day[0][day[0] < 0].max() for day in days)
    tables = []
    for set_v, read_v in days:
        keep = (set_v >= low) & (set_v <= high)
        points = [{'Kind': 'sweep', 'Direction': 'up', 'Set_V': s, 'Read_V': r, 'Sigma_V': 0.0,
                   'Channel': CH_NEG, 'Dac_V': 0.0, 'Time_S': 0.0}
                  for s, r in zip(set_v[keep], read_v[keep])]
        tables.append(hc.Linearity_Analysis(points))
    return tables


def test_The_Non_Linearity_Repeats_From_Day_To_Day():
    """The shape the correction's table takes out: on the bench it repeated
    between 25 and 26 October to 0.018 V rms against 0.17 V for the shape."""
    first, second = (day['Inl'][CH_NEG] for day in _Negative_Branch())
    assert first['Smooth_Rms_V'] > 0.1                          # there is a shape to remove
    on_first_knots = np.interp(second['Knots_V'], first['Knots_V'], first['Error_V'])
    assert np.corrcoef(on_first_knots, second['Error_V'])[0, 1] > 0.99
    assert np.sqrt(np.mean((on_first_knots - second['Error_V']) ** 2)) < 0.03


def test_Yesterdays_Table_Takes_Out_Todays_Non_Linearity():
    """25 October's table on 26 October's sweep leaves what 26 October's own
    table would: the code rounding (0.305 V a code) and the noise."""
    first, second = _Negative_Branch()
    set_v, read_v = second['Set_V'], second['Read_V']
    fit = second['Fits'][CH_NEG]
    residual = read_v - (fit['Gain'] * set_v + fit['Offset_V'])
    table = first['Inl'][CH_NEG]
    left = residual - np.interp(set_v, table['Knots_V'], table['Error_V'])
    raw, corrected = np.sqrt(np.mean(residual ** 2)), np.sqrt(np.mean(left ** 2))
    assert corrected < 0.65 * raw
    assert corrected < second['Inl'][CH_NEG]['Rough_Rms_V'] + 0.01


# ── The simulated rack is the bench's ───────────────────────────

def test_The_Simulated_Platform_Is_The_Bench_Sweep():
    """The twin's readback less setpoint is the 25 October up-sweep averaged
    over 50 V about each knot, as its docstring says."""
    d = _Load('linearity_3kv_25oct.npz')
    up = d['direction'] == DIRECTIONS.index('up')
    error = (d['read_v'] - d['set_v'])[up]
    set_v = d['set_v'][up]
    bench = [error[np.abs(set_v - knot) <= 25.0].mean() for knot in simulated.ERROR_KNOTS_V]
    assert np.max(np.abs(np.array(bench) - simulated.ERROR_V)) < 0.002
    # through 0 V at the amplifier's offset of about -1 V, whichever converter was last
    assert simulated.Platform_Error(0.0) == pytest.approx(-1.0, abs=0.1)


@pytest.mark.parametrize('jump', [1000.0, 2000.0])
def test_The_Simulated_Step_Settles_As_The_Bench_Steps_Did(jump):
    d = _Load('settle.npz')
    interval, tolerance = float(d['interval_s']), REFERENCE['settle']['config']['tolerance_v']
    bench = [s for s, start, stop in zip(REFERENCE['settle']['steps'], d['from_v'], d['to_v'])
             if abs(stop - start) == jump]
    times = interval * np.arange(-100, 1000)
    trace = np.array([0.0 if t < 0 else jump * (1.0 - simulated.Step_Remaining(t))
                      for t in times])
    twin = hc.Step_Response(times, trace, tolerance)
    settles = [s['settle_s'] for s in bench]
    assert min(settles) - 0.001 <= twin['Settle_S'] <= max(settles) + 0.001
    assert twin['Overshoot_Pct'] == pytest.approx(
        np.median([s['overshoot_pct'] for s in bench]), abs=0.03)
    # the twin's response is linear: it rises as a 1 kV step did whatever the jump,
    # where the amplifier slews a 2 kV jump in about 1 ms; never slower, then
    rise = np.median([s['rise_10_90_s'] for s in bench])
    if jump == 1000.0:
        assert twin['Rise_S'] == pytest.approx(rise, abs=0.1 * interval)
    else:
        assert twin['Rise_S'] < rise


def test_The_Simulated_Cooler_And_Line_Meter_Are_The_Benchs():
    cooler = REFERENCE['cooler']
    assert simulated.COOLER_V == pytest.approx(cooler['mean_v'], abs=0.1)
    assert simulated.COOLER_DRIFT_V_S * 60 == pytest.approx(cooler['drift_v_per_min'], rel=0.05)
    assert simulated.COOLER_NOISE_V * COOLER_DIVIDER == pytest.approx(
        cooler['random_noise_rms_v'], abs=0.01)
    tones = dict(simulated.COOLER_TONES)
    for harmonic in (1, 3):
        peak = next(p for p in cooler['peaks'] if p['mains_harmonic'] == harmonic)
        assert tones[50.0 * harmonic] == pytest.approx(peak['amplitude_v'], abs=0.005)
    # the line meter at a 200 us aperture, from the last 0.3 s of each bench step, long
    # settled: the 50 Hz it picks up, and the noise a fit of it leaves
    d = _Load('settle.npz')
    t = float(d['interval_s']) * np.arange(600)
    fit = np.column_stack([np.ones_like(t), t, np.sin(2 * np.pi * 50 * t),
                           np.cos(2 * np.pi * 50 * t)])
    mains, noise = [], []
    for trace in d['hv_v']:
        coefficients, *_ = np.linalg.lstsq(fit, trace[-600:], rcond=None)
        mains.append(np.hypot(*coefficients[2:]))
        noise.append(np.std(trace[-600:] - fit @ coefficients))
    assert simulated.LINE_NOISE_V * 1000 == pytest.approx(np.median(noise), rel=0.05)
    assert simulated.LINE_MAINS_V * 1000 == pytest.approx(np.median(mains), rel=0.05)
