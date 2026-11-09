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
    d, z = np.load(os.path.join(BENCH, sweep)), np.load(os.path.join(BENCH, zero))
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


def test_A_Sweep_Without_Its_0_V_Checks_Makes_No_Table():
    points = [p for p in _Points('linearity_3kv_26oct.npz', 'zero_3kv_26oct.npz')
              if p['Kind'] == 'sweep']
    with pytest.raises(st.Table_Error, match="took 0 checks of 0 V"):
        st.Build(points)
