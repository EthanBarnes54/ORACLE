"""The cooler's rolling mean, against a brute-force reference on uneven readings with gaps."""

import numpy as np
import pytest

from cls_acquisition import cooler


def _Readings(seed=4):
    """Readings at uneven times (bunches of a few hundred ms, jittered), with
    a 40 s gap in them, longer than the window, and a 6 s one, shorter."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.02, 0.4, 600))
    t[300:] += 40.0
    t[450:] += 6.0
    return t, 29884.8 + 0.3 * rng.normal(size=len(t)) + 0.002 * t


def _Brute(t, v, at, window_s, centred):
    """Each mean as its definition says: every reading in the window, one by one."""
    means, partial = [], []
    stretches = np.split(np.arange(len(t)), np.flatnonzero(np.diff(t) > window_s) + 1)
    for q in at:
        low, high = (q - window_s / 2, q + window_s / 2) if centred else (q - window_s, q)
        inside = [i for i in range(len(t)) if low <= t[i] <= high]
        means.append(np.mean(v[inside]) if inside else np.nan)
        if not inside:
            partial.append(True)
            continue
        stretch = next(s for s in stretches if inside[0] in s)
        partial.append(low < t[stretch[0]] or high > t[stretch[-1]])
    return np.array(means), np.array(partial)


@pytest.mark.parametrize('centred', [True, False])
def test_The_Mean_Is_Every_Reading_In_Its_Window_And_No_Other(centred):
    t, v = _Readings()
    at = np.concatenate([t, np.linspace(t[0] - 10.0, t[-1] + 10.0, 400)])
    mean, partial = cooler.Rolling_Mean(t, v, 15.0, centred, at=at)
    reference, edges = _Brute(t, v, at, 15.0, centred)
    assert np.array_equal(np.isnan(mean), np.isnan(reference))
    assert np.allclose(mean[~np.isnan(mean)], reference[~np.isnan(reference)], rtol=0,
                       atol=1e-9)
    assert np.array_equal(partial, edges)


def test_A_Gap_Longer_Than_The_Window_Is_Never_Bridged():
    t, v = _Readings()
    t, v = t[:300], v[:300]
    later = np.concatenate([t, t[-1] + 40.0 + t[:5] - t[0]])
    shifted = np.concatenate([v, v[:5] + 100.0])            # far off, after the gap
    mean, partial = cooler.Rolling_Mean(later, shifted, 15.0)
    # no mean before the gap holds a reading after it, nor one after it a reading before
    assert np.all(mean[:300] < 29900.0) and np.all(mean[300:] > 29950.0)
    # where the readings stop and start again, the mean holds one side only
    assert partial[299] and partial[300]
    gap, edge = cooler.Rolling_Mean(later, shifted, 15.0, at=[t[-1] + 20.0])
    assert np.isnan(gap[0]) and edge[0]


def test_A_Window_Is_Marked_Partial_Only_Where_It_Reaches_Past_The_Readings():
    t = np.arange(0.0, 60.0, 0.1)
    mean, partial = cooler.Rolling_Mean(t, np.ones_like(t), 15.0)
    assert np.all(mean == 1.0)
    # half a window at each end centred; a whole window at the start trailing
    assert np.array_equal(partial, (t < 7.5 - 1e-9) | (t > t[-1] - 7.5 + 1e-9))
    _, trailing = cooler.Rolling_Mean(t, np.ones_like(t), 15.0, centred=False)
    assert np.array_equal(trailing, t < 15.0 - 1e-9)


def test_A_Trailing_Mean_Lags_A_Ramp_By_Half_Its_Window():
    t = np.arange(0.0, 120.0, 0.1)
    centred, _ = cooler.Rolling_Mean(t, 0.01 * t, 15.0)
    trailing, partial = cooler.Rolling_Mean(t, 0.01 * t, 15.0, centred=False)
    middle = (t > 20.0) & (t < 100.0)
    # on an exact 0.1 s grid a window's edge lands on a reading, which rounding may take
    # in or leave out: half a reading's worth of the ramp, 0.5 mV
    half_reading = 0.01 * 0.1 / 2 + 1e-12
    assert np.allclose(centred[middle], 0.01 * t[middle], rtol=0, atol=half_reading)
    assert np.allclose(trailing[middle & ~partial], 0.01 * (t[middle & ~partial] - 7.5), rtol=0,
                       atol=half_reading)


def test_A_Window_That_Is_Not_Positive_Is_Refused():
    with pytest.raises(ValueError, match="longer than 0 s"):
        cooler.Rolling(np.arange(3.0), np.ones(3), 0.0)
