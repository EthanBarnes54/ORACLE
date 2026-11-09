"""The PMTs' pair counting, against brute force, and the live counters built on it."""

import time

import numpy as np
import pytest

from cls_acquisition import detectors
from cls_acquisition.hardware import simulated
from cls_acquisition.hardware.timetagger import BIN_US
from gui.live_data import Live_Data


def _Counts(seed=3, bunches=300, afterpulse=0.0, channel=3):
    """Rows of the run file's columns: a few counts a bunch on four PMTs, timed
    across a 200 us window, ``afterpulse`` of ``channel``'s followed 2 to 5 us
    later by another on it."""
    rng = np.random.default_rng(seed)
    bunch = np.repeat(np.arange(bunches), rng.poisson(6, bunches))
    times = np.rint(rng.uniform(0.0, 200.0, len(bunch)) / BIN_US) * BIN_US
    channels = rng.integers(1, 5, len(bunch))
    parents = (channels == channel) & (rng.random(len(bunch)) < afterpulse)
    bunch = np.concatenate([bunch, bunch[parents]])
    times = np.concatenate([times, times[parents] + np.rint(
        rng.uniform(2.0, 5.0, np.count_nonzero(parents)) / BIN_US) * BIN_US])
    channels = np.concatenate([channels, channels[parents]])
    rows = np.zeros((len(bunch), 6))
    rows[:, 2], rows[:, 3], rows[:, 4] = bunch, channels, times
    return rows


def test_Every_Pair_In_A_Window_Before_A_Count_In_Its_Bunch_Is_Counted_And_No_Other():
    rows = _Counts()
    keys = detectors.Keys(rows[:, 2], rows[:, 4])
    order = np.argsort(keys)
    keys, bunch, times, channel = keys[order], rows[order, 2], rows[order, 4], rows[order, 3]
    own, others = channel == 3, channel != 3
    for low, high, shift in ((0.0, 10.0, 0), (5.0, 50.0, 0), (0.0, 10.0, 2)):
        for sources, after in ((own, True), (others, False)):
            counted = detectors.Counted(keys[own], keys[sources], low, high, shift, after)
            for i, n in zip(np.flatnonzero(own)[:200], counted[:200]):
                gap = times[i] - times[sources]
                inside = (bunch[sources] == bunch[i] + shift) & (gap >= low) & (gap <= high)
                if after and not low and not shift:
                    inside &= gap > 0           # strictly before, so not itself
                assert n == np.count_nonzero(inside)


def test_The_Previous_Count_Is_The_Nearest_Before_In_Its_Bunch():
    rows = _Counts(seed=4)
    keys = np.sort(detectors.Keys(rows[:, 2], rows[:, 4]))
    found = detectors.Previous(keys, keys)
    bunch, times = keys >> 24, (keys & (2 ** 24 - 1)) * BIN_US
    for i in range(0, len(keys), 7):
        earlier = (bunch == bunch[i]) & (times < times[i])
        expected = times[i] - times[earlier].max() if earlier.any() else np.nan
        assert found[i] == pytest.approx(expected, nan_ok=True)


@pytest.mark.parametrize('afterpulse', [0.0, 0.05])
def test_The_Excess_Is_The_Afterpulses_And_Stands_Out_Only_When_They_Are_There(afterpulse):
    rows = _Counts(bunches=3000, afterpulse=afterpulse)
    keys = detectors.Keys(rows[:, 2], rows[:, 4])
    order = np.argsort(keys)
    pairs = detectors.By_Channel(rows[order, 3], keys[order], [1, 2, 3, 4])
    excess, errors, own, others = pairs[3].Excess(0.0, 10.0)
    assert excess / pairs[3].Count == pytest.approx(afterpulse / (1 + afterpulse), abs=0.01)
    assert (errors > 3.0) == bool(afterpulse)
    assert others == pytest.approx(1.0, abs=0.1)
    for channel in (1, 2, 4):
        assert abs(pairs[channel].Excess(0.0, 10.0)[1]) < 3.0


def test_The_Live_Ratio_Is_One_By_Chance_And_Warns_Of_A_Tube_That_Afterpulses():
    live = Live_Data(10.0)
    for packet in np.array_split(_Counts(bunches=2000), 50):     # whole bunches each
        live.Add(packet)
    assert all(ratio == pytest.approx(1.0, abs=0.25) for ratio in
               live.Afterpulse_Ratios().values())
    live.Clear()
    live.Add(_Counts(bunches=2000, afterpulse=0.3))
    ratios = live.Afterpulse_Ratios()
    assert ratios[3] > 2.0 and all(ratios[c] < 1.3 for c in (1, 2, 4))
    live.Clear()
    assert live.Afterpulse_Ratios() == {}


def test_The_Simulated_Pmts_Afterpulse_And_Spread_Only_When_Told(monkeypatch):
    import multiprocessing
    platform = simulated.Platform(multiprocessing.get_context('spawn'))
    platform.Set(32.0)                                  # on the strongest resonance
    # settled first: the bunches take a few ms here, and a rising platform counts fewer ions
    time.sleep(0.05)

    def counts(tagger, edges=1500):
        tagger.Start()
        for _ in range(edges):
            tagger.Edge(0.5)
        rows = tagger.Read(0.1)
        return rows[rows[:, 1] != 0]

    plain = counts(simulated.Tagger(platform))
    monkeypatch.setattr(simulated, 'AFTERPULSE', {3: 0.5})
    pulsed = counts(simulated.Tagger(platform))
    share = np.count_nonzero(pulsed[:, 1] == 3) / np.count_nonzero(plain[:, 1] == 3)
    assert share == pytest.approx(1.5, rel=0.05)
    # each afterpulse 2 to 5 us after a count on its own input, in its bunch: beyond chance,
    # one pair on it in that window for every afterpulse, and none on the others
    keys = detectors.Keys(pulsed[:, 0], pulsed[:, 2])
    order = np.argsort(keys)
    pairs = detectors.By_Channel(pulsed[order, 1], keys[order], [1, 2, 3, 4])
    # unbiased, scattering 0.016 at this many bunches
    assert pairs[3].Excess(2.0, 5.0)[0] / pairs[3].Count == pytest.approx(0.5 / 1.5, abs=0.06)
    assert abs(pairs[1].Excess(2.0, 5.0)[1]) < 3.0
    monkeypatch.setattr(simulated, 'AFTERPULSE', {})
    monkeypatch.setattr(simulated, 'ION_SPREAD', 0.5)
    spread = counts(simulated.Tagger(platform))
    per_bunch = np.bincount(spread[:, 0].astype(int))[1:]
    assert np.var(per_bunch) > 2.0 * np.mean(per_bunch)    # beyond Poisson's
