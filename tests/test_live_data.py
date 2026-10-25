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
