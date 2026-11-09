"""The cooler's voltage as the analysis takes it: a rolling mean over time.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The cooler meter is read for 200-500 us every 50 ms, not with the bunch,
so each reading catches the 50 Hz ripple (0.33 V amplitude on the bench)
at a random phase. Correcting each hit by its own reading adds that
scatter to the spectrum rather than taking anything out (A. Raggio,
November 2026). The meeting agreed the analysis takes a rolling mean
instead, about 15 s wide, which follows the slow wander (33 mV rms in
1 s means) and averages the ripple's aliasing away; cls_tools is being
changed to do the same. ORACLE uses it to judge a run as it will be
analysed, and shows it wherever the cooler is plotted over a run.

The window is in seconds, never in readings: a count of readings changes
its meaning with the rate they are taken at (the monitor's 10 Hz now, a
reading per bunch once the meter takes them with the bunch). A
reading's scatter of about 0.3 V falls to about 25 mV over 15 s at
10 Hz, below the wander the mean is to follow.

A mean is a boxcar over the readings within the window of each time:
centred (half the window either side, for a run already taken) or
trailing (the window before it, for a live display, which then lags by
half the window). It never reaches across a gap in the readings longer
than the window, as no window that long can hold readings from both
sides; where it reaches past the readings at either end of the stretch
it is in, it is marked partial, as it then holds one side only. Until
cls_tools' definition is agreed (centred or trailing, edges, gaps),
both forms and the window are parameters.

Depends on: numpy.
"""

import numpy as np

ROLLING_S = 15.0                    # the analysis' window (A. Raggio, November 2026)


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
