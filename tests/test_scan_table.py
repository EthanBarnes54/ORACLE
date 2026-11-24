"""The bench table, built from the bench tool's own ±3 kV sweeps of 25 and 26 October 2026.

The 25 October sweep makes a table; the 26 October one, a day later,
must sit on it within the repeatability A. Raggio found (about 10 mV once
the 0 V level is taken out), as a run's calibration is fitted to it.
"""

import os

import numpy as np
import pytest

from cls_acquisition import scan_table as st
from cls_acquisition.hardware.mcc_dac import CH_NEG, CH_POS

BENCH = os.path.join(os.path.dirname(__file__), 'data', 'bench')


def _Points(sweep, zero):
    """A bench sweep and its 0 V checks as the points a linearity sweep takes."""
    # each array read out of its archive once: indexing the archive reads it again every time
    d, z = (dict(np.load(os.path.join(BENCH, name))) for name in (sweep, zero))
    points = [{'Kind': 'zero', 'Direction': '', 'Set_V': 0.0, 'Read_V': float(v),
               'Sigma_V': float(e), 'Channel': CH_POS, 'Dac_V': 0.0, 'Time_S': float(t)}
              for t, v, e in zip(z['time_s'], z['read_v'], z['sigma_v'])]
    for i, volts in enumerate(d['set_v']):
        channel = CH_POS if volts >= 0 else CH_NEG
        points.append({'Kind': 'sweep', 'Direction': 'up', 'Set_V': float(volts),
                       'Read_V': float(d['read_v'][i]), 'Sigma_V': float(d['sigma_v'][i]),
                       'Channel': channel, 'Dac_V': float(d['dac_v'][i, channel]),
                       'Time_S': float(d['time_s'][i])})
    return points


@pytest.fixture(scope='module')
def oct25():
    points = _Points('linearity_3kv_25oct.npz', 'zero_3kv_25oct.npz')
    return st.Build(points, {'measured': '2026-10-25 02:14:14',
                             'rack': {'Kind': 'hardware', 'Dac': {'Board': 'USB-231',
                                                                  'UniqueId': 'A1'}}}), points


@pytest.fixture(scope='module')
def oct26():
    points = _Points('linearity_3kv_26oct.npz', 'zero_3kv_26oct.npz')
    sweep = [p for p in points if p['Kind'] == 'sweep']
    return points, (np.array([p['Channel'] for p in sweep]),
                    st.Codes([p['Dac_V'] for p in sweep]),
                    np.array([p['Read_V'] for p in sweep]), np.array([p['Time_S'] for p in sweep]))


def test_A_Day_Later_The_Converters_Sit_On_The_Table_Within_Its_Repeatability(oct25, oct26):
    table, _ = oct25
    points, (channels, codes, readback, times) = oct26
    zero_t, zero_v = st.Zero_Level(points)
    verdict, fit = st.Check(table, channels, codes, readback - np.interp(times, zero_t, zero_v))
    # A. Raggio: about 10 mV rms with the 0 V level taken out; one table here, not twelve runs
    assert verdict == 'PASS' and fit['Rms_V'] == pytest.approx(0.0116, abs=0.002)
    assert abs(fit['A_V']) < 0.005 and abs(fit['B'] - 1.0) < 20e-6
    assert fit['Points'] == len(readback) and fit['Interpolated'] == 0
    # fitted to the run's raw readback, as a run is: a is the run's 0 V level
    raw = st.Fit(table, channels, codes, readback)
    assert raw['A_V'] == pytest.approx(-1.0, abs=0.05) and raw['Rms_V'] < 0.03
    # where a straight line leaves the converters' shape, hundreds of mV
    branches = st.Analysis(table)['Branches']
    assert all(b['Rms_V'] > 0.15 and b['Peak_To_Peak_V'] > 1.0 for b in branches.values())
    assert {b['Codes'] for b in branches.values()} == {3000, 3001}


def test_The_0_V_Level_Is_Taken_Out_As_It_Warms_And_Gives_The_Check_Interval(oct25):
    table, points = oct25
    zero = table.Meta['zero_v']
    # the bench's: from -0.80 V, 0.2 V in the first hours, settled by the end
    assert zero['first'] == pytest.approx(-0.80, abs=0.01) and zero['span'] == pytest.approx(
        0.22, abs=0.02)
    assert not zero['warming'] and abs(zero['last_hour_v_per_h']) < st.WARMING_V_PER_H
    interval = st.Analysis(table, st.Zero_Level(points))['Check_Interval_S']
    assert 5 * 60 < interval < 30 * 60
    assert st.Analysis(table)['Check_Interval_S'] == table.Meta['check_interval_s'] == interval


def test_Warming_Up_Is_Said_While_The_0_V_Checks_Still_Move_Clear_Of_Their_Noise(oct25):
    """The bench's own checks: warming in their first hour, settled in their
    last; a few early checks too close together to tell say nothing."""
    times, volts = st.Zero_Level(oct25[1])
    early = times <= times[0] + 3600.0
    # falling 0.13 V/h, as the 0 V level did through its first hour
    assert st.Warming(times[early], volts[early]) == pytest.approx(-0.135, abs=0.01)
    assert st.Warming(times, volts) is None
    assert st.Warming(times[:3], volts[:3] + np.array([0.0, 0.003, -0.002])) is None
    assert st.Warming(times[:2], volts[:2]) is None


def test_A_Second_Days_Table_Agrees_With_The_First(oct25, oct26):
    table, _ = oct25
    later = st.Build(oct26[0])
    compared = st.Compare(later, table)
    assert list(compared) == ['negative'] and compared['negative']['Rms_V'] < 0.02


def test_A_Code_The_Table_Did_Not_Visit_Is_Interpolated_And_Said_To_Be(oct25):
    table, _ = oct25
    known = table.Channels[CH_POS]['code']
    between = known[100] + 1 if known[101] - known[100] > 1 else known[100] + 2
    volts, exact = table.Volts([CH_POS, CH_POS, CH_NEG], [known[100], between, 10])
    assert volts[0] == table.Channels[CH_POS]['volts'][100] and exact[0]
    assert np.isfinite(volts[1]) and not exact[1]
    assert np.isnan(volts[2])                          # outside what it measured: nothing
    with pytest.raises(st.Table_Error, match="does not cover this scan"):
        st.Fit(table, [CH_NEG] * 5, [10] * 5, np.zeros(5))


def test_The_Library_Keeps_Every_Table_As_It_Was_And_Suggests_The_Racks_Own(oct25, tmp_path):
    table, _ = oct25
    path = st.Save(table, str(tmp_path))
    assert os.path.basename(path) == f"table_20261025_{table.Id[:8]}.csv"
    assert st.Save(table, str(tmp_path)) == path        # found, not written again
    again = st.Load(path)
    assert again.Id == table.Id and again.Meta['rack'] == table.Meta['rack']
    assert np.array_equal(again.Channels[CH_NEG]['code'], table.Channels[CH_NEG]['code'])
    # a file changed after it was written is refused, as is writing another over it
    text = open(path, encoding='utf-8').read()
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text.replace(',0.', ',1.', 1))
    with pytest.raises(st.Table_Error, match="changed after it was written"):
        st.Load(path)
    with pytest.raises(st.Table_Error, match="Nothing was written over it"):
        st.Save(table, str(tmp_path))
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    # the newest table measured on this rack is the one suggested
    rack = {'Kind': 'hardware', 'Dac': {'Board': 'USB-231', 'UniqueId': 'A1'}}
    other = dict(rack, Dac={'Board': 'USB-231', 'UniqueId': 'B2'})
    assert st.Suggest(str(tmp_path), rack) == path and st.Suggest(str(tmp_path), other) is None
    same, why = st.Matches(table, other)
    assert not same and 'measured on another rack' in why and 'B2' in why
    assert [p for p, _ in st.Library(str(tmp_path))] == [path]
    # the bench tool recorded the board but not its unique ID: that cannot be checked
    unknown = st.Table(table.Channels, dict(table.Meta, id=None, rack={
        'Kind': 'hardware', 'Dac': {'Board': 'USB-231', 'UniqueId': None}}))
    same, why = st.Matches(unknown, rack)
    assert not same and "not the board's unique ID" in why


def _Run_Calibration(points, rng, low=-2000.0, high=-1000.0, every=5, noise=0.007):
    """A run's calibration out of a sweep: every ``every`` volts across the
    scan, one reading each at the meter's ``noise`` (none: as read), with its
    setpoints to the microvolt, as a plan writes them."""
    pick = [p for p in points if p['Kind'] == 'sweep' and low <= p['Set_V'] <= high
            and int(p['Set_V']) % every == 0]
    return (np.round([p['Set_V'] for p in pick], 6), np.array([p['Channel'] for p in pick]),
            st.Codes([p['Dac_V'] for p in pick]),
            np.array([p['Read_V'] for p in pick]) + (rng.normal(0.0, noise, len(pick))
                                                     if noise else 0.0))


def test_A_Runs_Calibration_Fits_The_Table_With_Its_0_V_Level_And_Gain(oct25, oct26):
    table, _ = oct25
    setpoints, channels, codes, readback = _Run_Calibration(oct26[0], np.random.default_rng(1))
    fit = st.Fit_Run(table, setpoints, channels, codes, readback, 0.007)
    # a is the run's 0 V level, known to a few mV over two hundred points (Raggio: 6 mV
    # over 293); b within tens of ppm of one; the residual the meter and the table's own
    assert fit['Verdict'] == 'PASS' and fit['Points'] == len(readback) > 190
    assert fit['A_V'] == pytest.approx(-0.99, abs=0.02) and fit['A_Err_V'] < 0.006
    assert abs(fit['B'] - 1.0) < 50e-6 and fit['Chi2'] == pytest.approx(1.0, abs=0.3)
    assert fit['Rms_V'] < st.STALE_FACTOR * fit['Expected_V']
    assert st.Summary(fit).startswith('a = -99') and st.Summary(fit).endswith('PASS')


def test_A_Table_That_No_Longer_Describes_The_Converters_Is_Warned_Of(oct25, oct26):
    table, _ = oct25
    setpoints, channels, codes, readback = _Run_Calibration(oct26[0], np.random.default_rng(2))
    # the converters' shape moved by 40 codes since the table was measured
    moved = st.Table({c: dict(t, volts=np.roll(t['volts'], 40))
                      for c, t in table.Channels.items()}, {})
    fit = st.Fit_Run(moved, setpoints, channels, codes, readback, 0.007)
    assert fit['Verdict'] == 'WARN' and fit['Rms_V'] > 0.1


# what cls_tools main (23d1e76) gives on the same points, its table this one's .json:
# ScanCalibration.fit_run(set, readback, gain='auto') -> offset, scale, rejected
@pytest.mark.parametrize('low, high, every, bad, mode, offset, scale, rejected', [
    (-2000, -1000, 5, 3, 'free', -0.991712145, 0.999975700331, [-1985.0]),
    (-1100, -1060, 2, 1, 'bench', -0.965156550, 1.0, [-1098.0]),     # run 7268's reading
    (-1400, -1000, 4, None, 'free', -0.965701642, 0.999997716072, []),   # 400 V: b fitted
    (-1396, -1000, 4, None, 'bench', -0.962966632, 1.0, [])])            # 396 V: the bench's
def test_A_Runs_Fit_Is_The_One_cls_tools_Applies(oct25, oct26, low, high, every, bad, mode,
                                                offset, scale, rejected):
    table, _ = oct25
    setpoints, channels, codes, readback = _Run_Calibration(oct26[0], None, low, high, every,
                                                            noise=0.0)
    if bad is not None:
        readback[bad] += 0.802        # the first reading after a jump, not yet settled
    fit = st.Fit_Run(table, setpoints, channels, codes, readback, 0.007)
    assert (fit['Scale_Mode'], fit['Rejected']) == (mode, rejected)
    assert fit['A_V'] == pytest.approx(offset, abs=1e-9)
    assert fit['B'] == pytest.approx(scale, abs=1e-12)
    assert fit['Points'] == len(setpoints) - len(rejected)
    assert ('b = 1 (bench)' in st.Summary(fit)) == (mode == 'bench')


def test_A_Sweep_Without_Its_0_V_Checks_Makes_No_Table():
    points = [p for p in _Points('linearity_3kv_26oct.npz', 'zero_3kv_26oct.npz')
              if p['Kind'] == 'sweep']
    with pytest.raises(st.Table_Error, match="took 0 checks of 0 V"):
        st.Build(points)


def _Bench_Tool_Sweep(folder, setpoints=(-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)):
    """A sweep as the bench tool (cls-cal) writes it: meta.json and
    points.csv, its set voltage and readings in kV, 0 V checks between."""
    import json
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'meta.json'), 'w') as fh:
        json.dump({'kind': 'linearity', 'started': '2026-10-25T02:14:14',
                   'software': 'cls-cal 0.1.0'}, fh)
    head = ("index,kind,direction,set_kv,t_s,ao0_v,ao1_v,ao0_code,ao1_code,n,mean_dmm_v,"
            "std_dmm_v,mean_kv,std_kv,sem_kv,error_v\n")
    rows, t = [], 0.0
    for kind, kv in [('zero', 0.0)] + [('sweep', v) for v in setpoints] + [('zero', 0.0)]:
        t += 2.5
        read = -0.000976 if kind == 'zero' else kv * 0.9992 - 0.000976
        dac = abs(kv)
        rows.append(f"{len(rows)},{kind},{'up' if kind == 'sweep' else ''},{kv},{t},"
                    f"{dac if kv >= 0 else 0.0},{dac if kv < 0 else 0.0},0,0,5,{read},1e-6,"
                    f"{read},1e-6,4e-7,0\n")
    with open(os.path.join(folder, 'points.csv'), 'w') as fh:
        fh.write(head + ''.join(rows))
    return os.path.join(folder, 'points.csv')


def test_The_Bench_Tools_Own_Sweep_Makes_A_Table(tmp_path):
    """The 24-26 October 2026 sweeps are the bench tool's records: made
    into a table as they are, both converters, the 0 V level taken out."""
    from cls_acquisition import hardware_calibration as hc
    meta, points = hc.Read_Bench_Tool(_Bench_Tool_Sweep(str(tmp_path / 'sweep')))
    assert meta['measured'] == '2026-10-25 02:14:14'
    table = st.Build(points, meta)
    assert sorted(table.Channels) == [CH_POS, CH_NEG]
    volts, exact = table.Volts([CH_POS], st.Codes([2.0]))
    assert exact[0] and volts[0] == pytest.approx(2000.0 * 0.9992, abs=1e-6)
    with pytest.raises(hc.Calibration_Error, match="not a bench tool sweep"):
        with open(tmp_path / 'other.csv', 'w') as fh:
            fh.write("kind,set_kv\nsweep,1.0\n")
        (tmp_path / 'meta.json').write_text('{}')
        hc.Read_Bench_Tool(str(tmp_path / 'other.csv'))


@pytest.mark.skipif(not os.path.isdir(os.path.join(
    os.path.dirname(BENCH), '..', '..', 'ORIGINAL_DAQ_STUFF', 'cls_calibration_tests', 'data',
    '20261025_021414_linearity_3kvspan')), reason="the bench tool's records are not here")
def test_The_Real_Sweep_Makes_The_Table_The_Next_Day_Sits_On(oct26):
    from cls_acquisition import hardware_calibration as hc
    folder = os.path.join(os.path.dirname(BENCH), '..', '..', 'ORIGINAL_DAQ_STUFF',
                          'cls_calibration_tests', 'data', '20261025_021414_linearity_3kvspan')
    meta, points = hc.Read_Bench_Tool(os.path.join(folder, 'points.csv'))
    table = st.Build(points, meta)
    zeros, (channels, codes, readback, times) = oct26
    zero_t, zero_v = st.Zero_Level(zeros)
    verdict, fit = st.Check(table, channels, codes, readback - np.interp(times, zero_t, zero_v))
    assert verdict == 'PASS' and fit['Rms_V'] < 0.013


def test_Every_Saved_Table_Has_Its_cls_tools_Copy_Beside_It(oct25, tmp_path):
    """The rules clstools.calibration.ScanCalibration.from_dict reads a table by
    (format v1), and each setpoint as the DAC writes it, through its own converter."""
    import json
    table, _ = oct25
    path = st.Save(table, str(tmp_path))
    out = path[:-len('.csv')] + '.json'
    with open(out, encoding='utf-8') as fh:
        d = json.load(fh)
    assert (d['format'], d['format_version']) == ('clstools-scan-calibration', 1)
    assert d['id'] == f"table_20261025_{table.Id[:8]}" and d['measured'] == '2026-10-25'
    set_v, shape_v = np.array(d['set_v']), np.array(d['shape_v'])
    assert set_v.shape == shape_v.shape and np.all(np.diff(set_v) > 0)
    n = sum(len(t['code']) for t in table.Channels.values())
    assert len(set_v) == n and set_v[0] == -3000.0 and set_v[-1] == 3000.0
    by = {c: dict(zip(t['setpoint_v'], t['volts'])) for c, t in table.Channels.items()}
    assert [by[CH_POS if v >= 0 else CH_NEG][v] for v in set_v] == list(shape_v)
    zero = table.Meta['zero_v']
    assert d['bench_offset_v'] == pytest.approx(0.5 * (zero['first'] + zero['last']))
    assert d['repeat_rms_v'] == st.REPEATABILITY_V
    assert d['provenance']['id'] == table.Id and d['provenance']['rack'] == table.Meta['rack']
    assert st.Save(table, str(tmp_path)) == path and st.Clstools(table, path) == out
    assert [p for p, _ in st.Library(str(tmp_path))] == [path]   # not a table of its own
