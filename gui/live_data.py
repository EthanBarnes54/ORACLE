"""What has been acquired since the last clear, and the histograms drawn from it.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Each update histograms only the rows since the last, so a long scan
costs no more per update than a short one; a change of binning, gate or
channels histograms everything again, once. The spectrum and the TOF-HFS
map count hits inside the TOF gate; the TOF spectrum counts every hit, so
the gate can be placed on it. The map is held only while shown, at most
MAX_MAP_BINS a side.

Only the four plotted columns are kept, in blocks of BLOCK_ROWS, each as
the narrowest whole-number codes that give every value back exactly (11
bytes a hit against the run file's 48); a value the codes cannot hold
turns its column to float64, and nothing is rounded. ``Version`` moves
whenever anything drawn changes, so a tab redraws only then. As rows
arrive, each PMT's pairs within the dead time are counted on itself and
on the others (cls_acquisition.detectors), an early warning of a tube
that afterpulses.

Depends on: numpy; cls_acquisition.detectors, cls_acquisition.hardware.timetagger
(no Qt, so it is tested headless).
"""

import numpy as np

from cls_acquisition import detectors
from cls_acquisition.hardware.timetagger import BIN_US

TIMESTAMP, VOLTS, BUNCH, CHANNEL, TOF, COOLER = range(6)   # run_file.DATA_COLS
# what the plots read, each as (its codes, the step between them); no step:
# which of the column's distinct values it is
KEPT = {VOLTS: (np.uint16, None), BUNCH: (np.int32, 1.0), CHANNEL: (np.int8, 1.0),
        TOF: (np.int32, BIN_US)}
FIRST_ROWS = 4096                   # the first block's start; it doubles up to BLOCK_ROWS
BLOCK_ROWS = 1 << 20                # 11 MB a block of four columns
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


class _Column():
    """One kept column of every row, in blocks, as codes of ``dtype``: whole
    numbers of ``step``, or with no step indices into the distinct values
    seen. A value they cannot give back exactly turns the column to float64."""

    def __init__(self, dtype, step):
        self._Step = step
        self._Coded = True
        self._Seen = {}                     # value: code, with no step
        self._Values = np.zeros(0)          # by code, with no step
        self._Blocks = []                   # full blocks, each BLOCK_ROWS long
        self._Open = np.empty(FIRST_ROWS, dtype)
        self._Filled = 0                    # rows in the open block

    def _Encode(self, values):
        """``values`` as codes, or as they are once the column is float64."""
        if self._Coded:
            limits = np.iinfo(self._Open.dtype)
            if self._Step is None:
                distinct, which = np.unique(values, return_inverse=True)
                codes = np.array([self._Seen.setdefault(v, len(self._Seen))
                                  for v in distinct.tolist()], dtype=np.int64)
                if len(self._Seen) <= limits.max + 1:
                    if len(self._Values) < len(self._Seen):
                        self._Values = np.array(list(self._Seen), dtype=float)
                    return codes[which.ravel()]
            else:
                steps = np.rint(values / self._Step)
                if np.array_equal(steps * self._Step, values) and (
                        not len(steps) or limits.min <= steps.min() <= steps.max() <= limits.max):
                    return steps
            self._Widen()
        return values

    def _Decode(self, codes):
        if not self._Coded:
            return codes
        return self._Values[codes] if self._Step is None else codes * self._Step

    def _Widen(self):
        """Every row so far, and every one from now on, as float64."""
        self._Blocks = [self._Decode(block) for block in self._Blocks]
        widened = np.empty(len(self._Open))
        widened[:self._Filled] = self._Decode(self._Open[:self._Filled])
        self._Open, self._Coded = widened, False

    def Add(self, values):
        codes = self._Encode(values)
        start = 0
        while start < len(codes):
            if self._Filled == len(self._Open):
                self._Make_Room()
            take = min(len(codes) - start, len(self._Open) - self._Filled)
            self._Open[self._Filled:self._Filled + take] = codes[start:start + take]
            start += take
            self._Filled += take

    def _Make_Room(self):
        """The open block doubled, or once full kept and a new one opened."""
        size = len(self._Open)
        if size == BLOCK_ROWS:
            self._Blocks.append(self._Open)
            self._Open, self._Filled = np.empty(BLOCK_ROWS, self._Open.dtype), 0
            return
        grown = np.empty(min(2 * size, BLOCK_ROWS), self._Open.dtype)
        grown[:size] = self._Open
        self._Open = grown

    def Values(self, start, stop):
        """The values of rows ``start`` to ``stop``, a block at a time."""
        for i, block in enumerate((*self._Blocks, self._Open)):
            low = i * BLOCK_ROWS
            first, last = max(start - low, 0), min(stop - low, BLOCK_ROWS)
            if first < last:
                yield self._Decode(block[first:last])


class Live_Data():

    def __init__(self, dead_time_us=detectors.DEAD_TIME_US):
        self.Version = 0
        self.Dead_Time_Us = dead_time_us
        self.Clear()

    def Clear(self):
        self._Kept = {column: _Column(*codes) for column, codes in KEPT.items()}
        self.Count = 0
        self._Key = None
        self._Pairs = {}                # channel: [counts, pairs on itself, on the others]
        self.Configure(np.array([0.0, 1.0]), np.array([0.0, 1.0]), None, (), False)

    def Add(self, rows):
        """Keep the plotted columns of ``rows``, which are the run file's."""
        if not len(rows):
            return
        for column, kept in self._Kept.items():
            kept.Add(rows[:, column])
        self.Count += len(rows)
        self._Count_Pairs(rows)
        self.Version += 1

    def _Count_Pairs(self, rows):
        keys = detectors.Keys(rows[:, BUNCH], rows[:, TOF])
        order = np.argsort(keys, kind='stable')
        keys, channels = keys[order], rows[order, CHANNEL]
        for channel in np.unique(channels):
            own, mine = keys[channels == channel], channels == channel
            tally = self._Pairs.setdefault(int(channel), [0, 0, 0])
            tally[0] += len(own)
            tally[1] += int(detectors.Counted(own, own, 0.0, self.Dead_Time_Us).sum())
            tally[2] += int(detectors.Counted(own, keys[~mine], 0.0, self.Dead_Time_Us,
                                              after=False).sum())

    def Afterpulse_Ratios(self, least=20):
        """``{channel: ratio}``: each PMT's pairs within the dead time on
        itself over those on the others, scaled by the others' counts over its
        own, so that chance gives 1 and afterpulsing more; None where either
        side has fewer than ``least`` pairs."""
        total = sum(tally[0] for tally in self._Pairs.values())
        ratios = {}
        for channel, (counts, same, other) in sorted(self._Pairs.items()):
            ratios[channel] = (same / other * (total - counts) / counts
                               if min(same, other) >= least else None)
        return ratios

    def _Columns(self, start, *columns):
        """``columns`` of every row from ``start``, a block at a time."""
        return zip(*(self._Kept[column].Values(start, self.Count) for column in columns))

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
        self.Version += 1

    def _Chosen(self, channel, tof, channels, gate):
        """``(on the channels, on them and inside the gate)`` as masks."""
        chosen = np.isin(channel, channels)
        if gate is None:
            return chosen, chosen
        return chosen, chosen & (tof > gate[0]) & (tof < gate[1])

    def Update(self):
        """Histogram the rows that arrived since the last update; True if any did."""
        if self._Done == self.Count:
            return False
        for volts, channel, tof in self._Columns(self._Done, VOLTS, CHANNEL, TOF):
            chosen, gated = self._Chosen(channel, tof, self._Channels, self._Gate)
            self.Spectrum += np.histogram(volts[gated], self.Volt_Edges)[0]
            self.Tof += np.histogram(tof[chosen], self.Tof_Edges)[0]
            if self._With_Map:
                self.Tof_Map += np.histogram2d(volts[gated], tof[gated],
                                               (self.Map_Volt_Edges, self.Map_Tof_Edges))[0]
        self._Done = self.Count
        self.Version += 1
        return True

    def Counts_In_Time(self, bin_s, trigger_hz, channels, gate):
        """``(edges in s, counts)`` of the gated hits, binned by bunch number,
        or None with nothing to show."""
        channels = np.array(channels)
        parts = []
        for bunch, channel, tof in self._Columns(0, BUNCH, CHANNEL, TOF):
            gated = self._Chosen(channel, tof, channels, gate)[1]
            if gated.any():
                parts.append(bunch[gated])
        if not parts:
            return None
        first = min(part.min() for part in parts)
        span = max(part.max() for part in parts) - first
        per_bin = max(bin_s * trigger_hz, span / MAX_TIME_BINS, 1.0)
        # the last bin is the latest bunch's, so every block's count is as long
        bins = int(span // per_bin) + 1
        counts = sum(np.bincount(((part - first) // per_bin).astype(int), minlength=bins)
                     for part in parts)
        return (first + per_bin * np.arange(len(counts) + 1)) / trigger_hz, counts
