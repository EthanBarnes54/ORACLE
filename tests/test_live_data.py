"""The live histograms: incremental updates agree with a full recount."""

import numpy as np
import pytest

from cls_acquisition.hardware.timetagger import BIN_US
from gui import live_data
from gui.live_data import Live_Data

VOLT_EDGES = np.arange(-0.5, 10.5)
TOF_EDGES = np.arange(-0.5, 100.5)


def _Rows(n, seed):
    rng = np.random.default_rng(seed)
    rows = np.zeros((n, 6))
    rows[:, 1] = rng.integers(0, 10, n)
    rows[:, 2] = np.sort(rng.integers(1, 1000, n))
    rows[:, 3] = rng.integers(1, 5, n)
    rows[:, 4] = rng.uniform(0, 100, n)
    return rows


def test_Updates_In_Packets_Agree_With_One_Recount():
    live = Live_Data()
    live.Configure(VOLT_EDGES, TOF_EDGES, (20.0, 60.0), (1, 3), True)
    packets = [_Rows(3000, seed) for seed in range(5)]
    for packet in packets:
        live.Add(packet)
        live.Update()
    rows = np.vstack(packets)
    hits = rows[np.isin(rows[:, 3], (1, 3))]
    gated = hits[(hits[:, 4] > 20.0) & (hits[:, 4] < 60.0)]
    assert np.array_equal(live.Spectrum, np.histogram(gated[:, 1], VOLT_EDGES)[0])
    assert np.array_equal(live.Tof, np.histogram(hits[:, 4], TOF_EDGES)[0])
    assert live.Tof_Map.sum() == len(gated)


def test_A_New_Gate_Recounts_Everything_Once():
    live = Live_Data()
    live.Configure(VOLT_EDGES, TOF_EDGES, None, (1, 2, 3, 4), False)
    live.Add(_Rows(1000, 1))
    live.Update()
    assert live.Spectrum.sum() == 1000
    live.Configure(VOLT_EDGES, TOF_EDGES, (0.0, 50.0), (1, 2, 3, 4), False)
    live.Update()
    assert 0 < live.Spectrum.sum() < 1000
    assert not live.Update()                        # nothing new: nothing to do


def test_A_Run_Kept_In_Blocks_Counts_As_One_Array_Would(monkeypatch):
    """Packets that straddle blocks, a regate and the time graph all count
    as a recount of every row at once does."""
    monkeypatch.setattr(live_data, 'FIRST_ROWS', 64)
    monkeypatch.setattr(live_data, 'BLOCK_ROWS', 1000)
    live = Live_Data()
    live.Configure(VOLT_EDGES, TOF_EDGES, None, (1, 2, 3, 4), False)
    packets = [_Rows(n, seed) for seed, n in enumerate((10, 700, 2500, 0, 1300))]
    for packet in packets:
        live.Add(packet)
        live.Update()
    assert all(len(kept._Blocks) == 4 for kept in live._Kept.values()) and live.Count == 4510
    rows = np.vstack(packets)
    live.Configure(VOLT_EDGES, TOF_EDGES, (20.0, 60.0), (2, 4), False)
    live.Update()
    hits = rows[np.isin(rows[:, 3], (2, 4))]
    gated = hits[(hits[:, 4] > 20.0) & (hits[:, 4] < 60.0)]
    assert np.array_equal(live.Spectrum, np.histogram(gated[:, 1], VOLT_EDGES)[0])
    assert np.array_equal(live.Tof, np.histogram(hits[:, 4], TOF_EDGES)[0])
    edges, counts = live.Counts_In_Time(0.1, 333.3, (2, 4), (20.0, 60.0))
    per_bin = 0.1 * 333.3
    bunches = gated[:, 2]
    expected = np.bincount(((bunches - bunches.min()) // per_bin).astype(int))
    assert np.array_equal(counts, expected) and len(edges) == len(counts) + 1


def _Tagged(n, seed):
    """Rows as the rack takes them: setpoints of a plan, times in tagger bins."""
    rows = _Rows(n, seed)
    rows[:, 1] = 20.0 + 0.3 * rows[:, 1]
    rows[:, 4] = np.rint(rows[:, 4] / BIN_US) * BIN_US
    return rows


def _Recount(rows, live):
    hits = rows[np.isin(rows[:, 3], (1, 3))]
    gated = hits[(hits[:, 4] > 20.0) & (hits[:, 4] < 60.0)]
    assert np.array_equal(live.Spectrum, np.histogram(gated[:, 1], live.Volt_Edges)[0])
    assert np.array_equal(live.Tof, np.histogram(hits[:, 4], live.Tof_Edges)[0])
    edges, counts = live.Counts_In_Time(0.1, 333.3, (1, 3), (20.0, 60.0))
    bunches = gated[:, 2]
    per_bin = max(0.1 * 333.3, np.ptp(bunches) / live_data.MAX_TIME_BINS, 1.0)
    assert np.array_equal(counts, np.bincount(((bunches - bunches.min()) // per_bin).astype(int)))


def _Held(live):
    return {name: (column._Open.dtype.name if column._Coded else 'float64')
            for name, column in zip(('volts', 'bunch', 'channel', 'tof'), live._Kept.values())}


def test_A_Run_Is_Held_In_The_Narrowest_Codes_That_Give_It_Back(monkeypatch):
    monkeypatch.setattr(live_data, 'FIRST_ROWS', 64)
    monkeypatch.setattr(live_data, 'BLOCK_ROWS', 1000)
    live = Live_Data()
    live.Configure(np.arange(19.85, 23.0, 0.3), TOF_EDGES, (20.0, 60.0), (1, 3), False)
    packets = [_Tagged(n, seed) for seed, n in enumerate((900, 1700, 400))]
    for packet in packets:
        live.Add(packet)
        live.Update()
    assert _Held(live) == {'volts': 'uint16', 'bunch': 'int32', 'channel': 'int8',
                           'tof': 'int32'}
    rows = np.vstack(packets)
    for column, kept in live._Kept.items():                 # every value back, bit for bit
        assert np.array_equal(np.concatenate(list(kept.Values(0, live.Count))), rows[:, column])
    _Recount(rows, live)


@pytest.mark.parametrize('column, value, held', (
    (4, 12.3456789, 'tof'),             # a time off the tagger's bins, from an old run
    (2, 2.0 ** 31, 'bunch'),            # past int32
    (3, 2.5, 'channel')))
def test_A_Value_No_Code_Holds_Turns_Its_Column_To_Float64(column, value, held):
    live = Live_Data()
    live.Configure(np.arange(19.85, 23.0, 0.3), TOF_EDGES, (20.0, 60.0), (1, 3), False)
    first, odd = _Tagged(800, 1), _Tagged(300, 2)
    odd[5, column] = value
    for packet in (first, odd):
        live.Add(packet)
        live.Update()
    assert _Held(live)[held] == 'float64'
    assert sum(kind == 'float64' for kind in _Held(live).values()) == 1
    _Recount(np.vstack((first, odd)), live)


def test_The_Version_Moves_Only_When_Something_Drawn_Changes():
    live = Live_Data()
    live.Configure(VOLT_EDGES, TOF_EDGES, None, (1, 2, 3, 4), False)
    seen = live.Version
    live.Configure(VOLT_EDGES.copy(), TOF_EDGES.copy(), None, [1, 2, 3, 4], False)
    assert not live.Update() and live.Version == seen      # the same settings, no rows
    live.Add(_Rows(10, 1))
    live.Update()
    assert live.Version > seen
    seen = live.Version
    live.Configure(VOLT_EDGES, TOF_EDGES, (0.0, 50.0), (1, 2, 3, 4), False)
    assert live.Version > seen


def test_A_Tab_Redraws_Only_When_Its_Data_Change(qapp, monkeypatch):
    from gui.scan_tab import Scan_Tab
    from gui.time_tab import Time_Tab
    scan, time_tab, live = Scan_Tab(4000.0), Time_Tab(), Live_Data()
    live.Add(_Rows(100, 1))
    live.Update()
    drawn = []
    monkeypatch.setattr(scan.Spectrum_Curve, 'setData', lambda *_, **__: drawn.append('scan'))
    monkeypatch.setattr(time_tab.Curve, 'setData', lambda *_, **__: drawn.append('time'))
    for _ in range(3):
        scan.Draw(live)
        time_tab.Draw(live, 10.0, None)
    assert drawn == ['scan', 'time']
    time_tab.Bin_Combo.setCurrentText('1')                  # what it is counted by
    time_tab.Draw(live, 10.0, None)
    live.Add(_Rows(5, 2))                                   # and new data, for both
    live.Update()
    scan.Draw(live)
    time_tab.Draw(live, 10.0, None)
    assert drawn == ['scan', 'time', 'time', 'scan', 'time']
    scan.deleteLater()
    time_tab.deleteLater()


def test_Counts_In_Time_Bin_By_Bunch_At_The_Trigger_Rate():
    live = Live_Data()
    rows = _Rows(500, 2)
    live.Add(rows)
    edges, counts = live.Counts_In_Time(1.0, 10.0, (1, 2, 3, 4), None)
    assert counts.sum() == 500
    assert np.allclose(np.diff(edges), 1.0)         # 10 bunches a bin at 10 Hz


def test_A_Fine_Tof_Bin_Is_Widened_Rather_Than_Ask_For_Gigabytes(qapp):
    from gui.scan_tab import MAX_TOF_BINS, Scan_Tab
    tab = Scan_Tab(4000.0)
    tab.Gate_Length_Box.setValue(8000.0)
    tab.Bin_Box.setValue(0.0005)
    edges = tab.Tof_Edges()
    assert len(edges) - 1 <= MAX_TOF_BINS + 1
    assert edges[0] < 0.0 < 8000.0 < edges[-1]               # the whole gate is still covered
    tab.Gate_Length_Box.setValue(200.0)
    tab.Bin_Box.setValue(1.0)
    assert np.allclose(np.diff(tab.Tof_Edges()), 1.0)        # a bin that fits is kept
    tab.deleteLater()


def test_The_Map_Is_Held_Only_While_Shown_And_At_A_Screens_Resolution():
    from gui.live_data import MAX_MAP_BINS
    live = Live_Data()
    fine_volts = np.arange(-3000.5, 3001.0, 0.1)             # 60001 setpoints
    fine_tof = np.arange(-0.25, 8000.0, 0.5)
    live.Configure(fine_volts, fine_tof, None, (1, 2, 3, 4), False)
    assert live.Tof_Map.size == 0
    live.Configure(fine_volts, fine_tof, None, (1, 2, 3, 4), True)
    assert max(live.Tof_Map.shape) <= MAX_MAP_BINS
    assert live.Map_Volt_Edges[[0, -1]].tolist() == fine_volts[[0, -1]].tolist()
    assert live.Map_Tof_Edges[[0, -1]].tolist() == fine_tof[[0, -1]].tolist()
    rows = _Rows(2000, 4)
    rows[:, 1] = np.random.default_rng(5).uniform(-3000, 3000, len(rows))
    live.Add(rows)
    live.Update()
    assert live.Tof_Map.sum() == len(rows) == live.Spectrum.sum()
