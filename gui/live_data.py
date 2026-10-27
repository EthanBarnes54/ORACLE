"""What has been acquired since the last clear, and the histograms drawn from it.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Rows arrive in packets several times a second. Each update histograms
only the rows that came since the last one, so a long scan costs no more
per update than a short one; a change of binning, gate or channels
histograms everything again, once.

The spectrum and the TOF-HFS map count only hits inside the TOF gate;
the TOF spectrum counts every hit on the chosen channels, so the gate can
be placed on it. The map is held only while it is shown, on edges no
finer than MAX_MAP_BINS a side (every few of the spectrum's and the TOF's
own, its ends theirs): it is volts by times, and at their own resolution
a fine scan over a fine TOF is gigabytes.

Depends on: numpy (no Qt, so it is tested headless).
"""

import numpy as np

TIMESTAMP, VOLTS, BUNCH, CHANNEL, TOF, COOLER = range(6)   # run_file.DATA_COLS
MAX_TIME_BINS = 50_000
MAX_MAP_BINS = 2000                 # a side of the TOF-HFS map: a screen's worth


def _Coarse(edges, most):
    """Every few of ``edges``, the last always kept, so that at most ``most``
    bins span the same range."""
    every = int(np.ceil((len(edges) - 1) / most))
    if every <= 1:
        return edges
    coarse = edges[::every]
    return coarse if coarse[-1] == edges[-1] else np.append(coarse, edges[-1])


class Live_Data():

    def __init__(self):
        self.Clear()

    def Clear(self):
        self._Rows = np.empty((4096, 6))
        self.Count = 0
        self._Key = None
        self.Configure(np.array([0.0, 1.0]), np.array([0.0, 1.0]), None, (), False)

    @property
    def Rows(self):
        return self._Rows[:self.Count]

    def Add(self, rows):
        end = self.Count + len(rows)
        if end > len(self._Rows):
            grown = np.empty((max(end, 2 * len(self._Rows)), 6))
            grown[:self.Count] = self._Rows[:self.Count]
            self._Rows = grown
        self._Rows[self.Count:end] = rows
        self.Count = end

    def Configure(self, volt_edges, tof_edges, gate, channels, with_map):
        """Set what the histograms count. Any change starts them again from
        the first row; the same settings again change nothing."""
        key = (volt_edges.tobytes(), tof_edges.tobytes(), gate, tuple(channels), with_map)
        if key == self._Key:
            return
        self._Key = key
        self.Volt_Edges, self.Tof_Edges = volt_edges, tof_edges
        self.Map_Volt_Edges = _Coarse(volt_edges, MAX_MAP_BINS)
        self.Map_Tof_Edges = _Coarse(tof_edges, MAX_MAP_BINS)
        self._Gate, self._Channels, self._With_Map = gate, np.array(channels), with_map
        self.Spectrum = np.zeros(len(volt_edges) - 1)
        self.Tof = np.zeros(len(tof_edges) - 1)
        self.Tof_Map = (np.zeros((len(self.Map_Volt_Edges) - 1, len(self.Map_Tof_Edges) - 1))
                        if with_map else np.zeros((0, 0)))
        self._Done = 0

    def _Select(self, rows, channels, gate):
        rows = rows[np.isin(rows[:, CHANNEL], channels)]
        if gate is None:
            return rows, rows
        inside = (rows[:, TOF] > gate[0]) & (rows[:, TOF] < gate[1])
        return rows, rows[inside]

    def Update(self):
        """Histogram the rows that arrived since the last update; True if any did."""
        new = self._Rows[self._Done:self.Count]
        self._Done = self.Count
        if not len(new):
            return False
        hits, gated = self._Select(new, self._Channels, self._Gate)
        self.Spectrum += np.histogram(gated[:, VOLTS], self.Volt_Edges)[0]
        self.Tof += np.histogram(hits[:, TOF], self.Tof_Edges)[0]
        if self._With_Map:
            self.Tof_Map += np.histogram2d(gated[:, VOLTS], gated[:, TOF],
                                           (self.Map_Volt_Edges, self.Map_Tof_Edges))[0]
        return True

    def Counts_In_Time(self, bin_s, trigger_hz, channels, gate):
        """``(edges in s, counts)`` of the gated hits, binned by bunch number,
        or None with nothing to show."""
        _, rows = self._Select(self.Rows, np.array(channels), gate)
        if not len(rows):
            return None
        bunches = rows[:, BUNCH]
        first, span = bunches.min(), np.ptp(bunches)
        per_bin = max(bin_s * trigger_hz, span / MAX_TIME_BINS, 1.0)
        counts = np.bincount(((bunches - first) // per_bin).astype(int))
        return (first + per_bin * np.arange(len(counts) + 1)) / trigger_hz, counts
