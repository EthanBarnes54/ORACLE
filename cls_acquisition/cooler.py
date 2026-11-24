"""The cooler's voltage as the analysis takes it: a rolling mean over time.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The meter reads for 200-500 us every 50 ms, unsynchronised with the
bunch, so each reading catches the 50 Hz ripple (0.33 V) at a random phase
(A. Raggio, November 2026). The analysis takes a boxcar mean of the
readings instead (seconds, not readings): trailing over 15 s for a live
display; centred for a finished run, over the window cls_tools takes
(:func:`Window`), its faulty readings left out (:func:`Faulty`). It never
spans a gap longer than the window, and is marked partial where it
reaches past its stretch.

Depends on: numpy.
"""

import numpy as np

ROLLING_S = 15.0                    # the analysis' window (A. Raggio, November 2026)
# cls_tools' rule for a finished run (calibration.cooler_ripple, CoolerRipple.window_s): four
# periods of the mains ripple as the readings alias it, at least ROLLING_S. The period is the
# strongest line between 0.05 and 4.9 Hz of the readings on a 0.1 s grid, their 30 s running
# mean taken out; a run under 120 s or 300 readings has none. A reading 5 V from the run's
# median is a readback fault.
ALIAS_PERIODS, GRID_S, SLOW_S, MAX_DEV_V = 4.0, 0.1, 30.0, 5.0
MIN_SPAN_S, MIN_READINGS = 120.0, 300


def Faulty(volts):
    """Whether each reading (platform volts) is further than MAX_DEV_V from the
    run's median: a readback fault, which cls_tools leaves out."""
    volts = np.asarray(volts, dtype=float)
    return np.abs(volts - np.median(volts)) > MAX_DEV_V if len(volts) else volts.astype(bool)


def Window(times, volts, least_s=ROLLING_S):
    """``(window_s, alias period s)`` cls_tools averages a finished run's
    fault-free readings over (seconds, in order; platform volts):
    ALIAS_PERIODS alias periods, at least ``least_s``; ``least_s`` and NaN
    for a run too short to show the alias."""
    t, v = np.asarray(times, dtype=float), np.asarray(volts, dtype=float)
    if len(t) < MIN_READINGS or t[-1] - t[0] < MIN_SPAN_S:
        return float(least_s), np.nan
    n = int((t[-1] - t[0]) / GRID_S) + 1
    grid = np.full(n, np.nan)
    grid[np.minimum(((t - t[0]) / GRID_S).astype(int), n - 1)] = v
    grid = grid[np.maximum.accumulate(np.where(np.isfinite(grid), np.arange(n), 0))]
    # the running mean as pandas centres it: half the width before each point
    width, index = int(SLOW_S / GRID_S), np.arange(n)
    sums = np.concatenate(([0.0], np.cumsum(grid)))
    low, high = np.clip(index - width // 2, 0, n), np.clip(index + width - width // 2, 0, n)
    fast = (grid - (sums[high] - sums[low]) / (high - low))[high - low >= int(10 / GRID_S)]
    frequency = np.fft.rfftfreq(len(fast), GRID_S)
    power = np.abs(np.fft.rfft(fast - fast.mean())) ** 2
    line = (frequency > 0.05) & (frequency < 4.9)
    if len(fast) < 16 or not line.any():
        return float(least_s), np.nan
    period = float(1.0 / frequency[line][np.argmax(power[line])])
    return float(max(least_s, ALIAS_PERIODS * period)), period


class Rolling():
    """A rolling mean of the readings ``volts`` taken at ``times`` (seconds,
    in order, all finite), ``window_s`` wide, centred or trailing."""

    def __init__(self, times, volts, window_s=ROLLING_S, centred=True):
        if window_s <= 0:
            raise ValueError(f"A rolling mean needs a window longer than 0 s, not "
                             f"{window_s:g} s.")
        self._T = np.asarray(times, dtype=float)
        volts = np.asarray(volts, dtype=float)
        self.Window_S, self.Centred = float(window_s), centred
        self._Sums = np.concatenate(([0.0], np.cumsum(volts)))
        # the stretches the readings fall in, split where they stop for longer than
        # the window: each reading's stretch, and where its stretch starts and ends
        breaks = np.flatnonzero(np.diff(self._T) > self.Window_S) + 1
        self._Stretch = np.zeros(len(self._T), dtype=np.intp)
        self._Stretch[breaks] = 1
        np.cumsum(self._Stretch, out=self._Stretch)
        bounds = np.concatenate(([0], breaks, [len(self._T)]))
        self._Starts, self._Ends = self._T[bounds[:-1]], self._T[bounds[1:] - 1]

    def At(self, times):
        """``(mean, partial)`` at each of ``times``: the mean of the readings
        within the window, NaN where there are none, and whether the window
        reached past the readings' stretch, so holds one side only."""
        times = np.asarray(times, dtype=float)
        if self.Centred:
            low, high = times - self.Window_S / 2.0, times + self.Window_S / 2.0
        else:
            low, high = times - self.Window_S, times
        first = np.searchsorted(self._T, low, side='left')
        last = np.searchsorted(self._T, high, side='right')
        count = last - first
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = (self._Sums[last] - self._Sums[first]) / count
        partial = count == 0
        if len(self._T):
            stretch = self._Stretch[np.minimum(first, len(self._T) - 1)]
            partial |= (low < self._Starts[stretch]) | (high > self._Ends[stretch])
        return mean, partial


def Rolling_Mean(times, volts, window_s=ROLLING_S, centred=True, at=None):
    """``(mean, partial)`` of :class:`Rolling` at ``at``, or at the readings'
    own times."""
    rolling = Rolling(times, volts, window_s, centred)
    return rolling.At(times if at is None else at)
