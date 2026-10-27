"""The live histograms: incremental updates agree with a full recount."""

import numpy as np

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
    for seed in range(5):
        live.Add(_Rows(3000, seed))
        live.Update()
    rows = live.Rows
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
