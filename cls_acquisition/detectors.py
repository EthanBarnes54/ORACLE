"""The PMTs' afterpulsing: counts that follow a count on the same PMT more often than chance.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A PMT often counts again within a few us of its own count, not of
another's (A. Raggio, November 2026, run 8244), so spectra are noisier
than Poisson. ORACLE measures it; applying a dead time is DENIS's. Each
count is paired with the earlier counts of its bunch on its own PMT and on
the others; chance is the same pairing against later bunches, and the
other PMTs are the control. Every pair within the dead time is judged
against the scatter of the bunches' own contributions. Counts are keyed
by bunch and tagger bin as one integer, so every search is exact.

Depends on: numpy; cls_acquisition.hardware.timetagger.
"""

import numpy as np

from cls_acquisition.hardware.timetagger import BIN_US, TIMESTAMP_BITS

DEAD_TIME_US = 10.0                 # Andrea's, the one the analysis reports at
DEAD_TIMES_US = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
REACH_US = 50.0                     # the pairs' times are followed this far
HISTOGRAM_US = 0.5                  # their histogram's bins
MIXED_BUNCHES = 4                   # the next bunches chance is taken from


def Keys(bunch, time_us):
    """Each count as one integer: its bunch above its time in tagger bins."""
    bins = np.clip(np.rint(np.asarray(time_us) / BIN_US), 0, 2 ** TIMESTAMP_BITS - 1)
    return (np.asarray(bunch).astype(np.int64) << TIMESTAMP_BITS) | bins.astype(np.int64)


def Previous(targets, sources, shift=0):
    """For each of ``targets`` (keys), the time in us back to the latest of
    ``sources`` (keys, in order) before it in the same bunch, ``shift``
    bunches on; NaN where there is none."""
    moved = targets + (np.int64(shift) << TIMESTAMP_BITS)
    if not len(sources):
        return np.full(len(targets), np.nan)
    found = np.searchsorted(sources, moved, side='left') - 1
    before = sources[np.maximum(found, 0)]
    same_bunch = (found >= 0) & ((before >> TIMESTAMP_BITS) == (moved >> TIMESTAMP_BITS))
    return np.where(same_bunch, (moved - before) * BIN_US, np.nan)


def Counted(targets, sources, low_us, high_us, shift=0, after=True):
    """For each of ``targets``, how many of ``sources`` (keys, in order) came
    ``low_us`` to ``high_us`` before it in the same bunch, ``shift`` bunches
    on: strictly before if ``after`` (a PMT's own counts, so not itself)."""
    moved = targets + (np.int64(shift) << TIMESTAMP_BITS)
    start = (moved >> TIMESTAMP_BITS) << TIMESTAMP_BITS
    low = np.maximum(moved - int(round(high_us / BIN_US)), start)
    high = moved - int(round(low_us / BIN_US))
    side = 'left' if after and not low_us else 'right'
    return np.maximum(np.searchsorted(sources, high, side=side)
                      - np.searchsorted(sources, low, side='left'), 0)


class Pairs():
    """One PMT's counts against the previous counts on itself and the others
    (``Same``, ``Other``; NaN where none), and chance: the same against the
    next MIXED_BUNCHES bunches' counts, weighted as if every count had them."""

    def __init__(self, own, others, last_bunch):
        self.Count = len(own)
        self._Own, self._Others = own, others
        self.Same, self.Other = Previous(own, own), Previous(own, others)
        bunch = own >> TIMESTAMP_BITS
        self._Index = (bunch - bunch.min()) if len(own) else bunch
        self._Shifts = []
        for shift in range(1, MIXED_BUNCHES + 1):
            have = bunch + shift <= last_bunch
            if have.any():
                self._Shifts.append((shift, have,
                                     self.Count / (np.count_nonzero(have) * MIXED_BUNCHES)))
        self._Mixed = {name: [(Previous(own[have], sources, shift), weight)
                              for shift, have, weight in self._Shifts]
                       for name, sources in (('Same', own), ('Other', others))}

    def _By_Bunch(self, sources, low_us, high_us, after):
        """Each bunch's pairs with ``sources`` in the window, observed and by
        chance."""
        length = int(self._Index.max()) + 1 if self.Count else 0
        seen = np.bincount(self._Index, Counted(self._Own, sources, low_us, high_us,
                                                after=after), length)
        chance = np.zeros(length)
        for shift, have, weight in self._Shifts:
            chance += weight * np.bincount(self._Index[have], Counted(
                self._Own[have], sources, low_us, high_us, shift, after), length)
        return seen, chance

    def Excess(self, low_us, high_us):
        """``(excess, standard errors, own ratio, others' ratio)``: this PMT's
        pairs in [low, high] us beyond chance as the other PMTs' scale it, its
        significance against the bunches' own scatter, and each side over chance."""
        same, same_chance = self._By_Bunch(self._Own, low_us, high_us, True)
        other, other_chance = self._By_Bunch(self._Others, low_us, high_us, False)
        totals = [float(a.sum()) for a in (same, same_chance, other, other_chance)]
        rise = totals[2] / totals[3] if totals[3] else 1.0
        scale = totals[1] / totals[3] if totals[3] else 0.0
        excess = totals[0] - totals[1] * rise
        # the excess's linear part, bunch by bunch: bunches are independent, pairs not
        parts = same - rise * same_chance - scale * (other - rise * other_chance)
        spread = float(np.sqrt(np.sum(parts ** 2)))
        ratio = lambda seen, chance: seen / chance if chance else np.inf if seen else 1.0
        return (excess, excess / spread if spread else 0.0, ratio(*totals[:2]),
                ratio(*totals[2:]))

    def Ratio(self, which):
        """``(edges, observed / chance)`` of :meth:`Histogram`, NaN where chance has none."""
        edges, observed, chance = self.Histogram(which)
        with np.errstate(divide='ignore', invalid='ignore'):
            return edges, np.where(chance > 0, observed / chance, np.nan)

    def Dead_Time(self, share):
        """The shortest of DEAD_TIMES_US past which no excess is left: at most
        ``share`` of the counts, or within a standard error of none; None if
        some is left past them all."""
        for tau in DEAD_TIMES_US[1:]:
            pairs, errors, _, _ = self.Excess(tau, REACH_US)
            if pairs / self.Count <= share or errors <= 1.0:
                return tau
        return None

    def Histogram(self, which):
        """``(edges, observed, chance)`` over 0 to REACH_US."""
        edges = np.arange(0.0, REACH_US + HISTOGRAM_US / 2, HISTOGRAM_US)
        times = getattr(self, which)
        observed = np.histogram(times[np.isfinite(times)], edges)[0].astype(float)
        chance = np.zeros(len(edges) - 1)
        for t, w in self._Mixed[which]:
            chance += w * np.histogram(t[np.isfinite(t)], edges)[0]
        return edges, observed, chance


def By_Channel(channels, keys, wanted):
    """``{channel: Pairs}`` for each of ``wanted`` that counted, from every
    count's channel and key, the keys in order."""
    last = int(keys[-1] >> TIMESTAMP_BITS) if len(keys) else 0
    found = {}
    for channel in wanted:
        mine = channels == channel
        if mine.any():
            found[channel] = Pairs(keys[mine], keys[~mine], last)
    return found


def Dispersion(cells, shape, channels, wanted, dropped=None):
    """Variance/mean (1: Poisson) of ``wanted`` channels' counts in the cells
    of ``shape`` (scans, setpoints), across scans, pooled; ``cells`` maps each
    count (-1: none), ``dropped`` marks those left out. Returns ``({channel:
    ratio}, the excess every pair shares, which is the beam's, dof)``."""
    scans, setpoints = shape
    kept = cells >= 0 if dropped is None else (cells >= 0) & ~dropped
    counts = {c: np.bincount(cells[kept & (channels == c)], minlength=scans * setpoints)
              .reshape(scans, setpoints).astype(float) for c in wanted}
    ratio, apart, means = {}, {}, {}
    for c, table in counts.items():
        means[c] = table.mean(axis=0)
        apart[c] = table - means[c]
        if means[c].sum():
            ratio[c] = float(np.sum(apart[c] ** 2) / ((scans - 1) * means[c].sum()))
    pairs = [(a, b) for i, a in enumerate(ratio) for b in list(ratio)[i + 1:]]
    common = [float(np.sum(apart[a] * apart[b])
                    / ((scans - 1) * np.sum(np.sqrt(means[a] * means[b])))) for a, b in pairs
              if np.sum(np.sqrt(means[a] * means[b]))]
    dof = (scans - 1) * int(np.count_nonzero(sum(means.values()))) if means else 0
    return ratio, (float(np.mean(common)) if common else 0.0), dof
