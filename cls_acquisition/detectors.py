"""The PMTs' afterpulsing: counts that follow a count on the same PMT more often than chance.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

After a count, the same PMT often counts again within a few us, hundreds
of times more often than chance, while after a count on another PMT
nothing special happens (A. Raggio, November 2026, run 8244): the tube
afterpulses or rings, and the extra counts make the spectrum noisier
than Poisson (variance/mean 2, against 1.03 once counts within 10 us of
one on the same PMT are dropped). ORACLE measures it and says so; the
dead time is the operator's to apply, in DENIS.

For each PMT, the time from each count back to the previous count on the
same PMT, and back to the previous count on any other, in the same bunch
(or DC window: across windows the time is not known). Chance is what the
same pairing gives against the next bunches' counts at the same times in
their bunch, which keeps the bunch's time-of-flight profile: in a bunch,
real photons arrive within a few us of each other, so a flat chance
level would take the light itself for afterpulsing. And light, and the
bunch-to-bunch spread of the number of ions, raise both curves alike,
where afterpulsing raises only the same PMT's: the other PMTs are the
control that the analysis (scan_performance) judges against.

The plotted curves take the time to the previous count, as the slide
does. The judgement counts every pair within the dead time instead: in
a dense bunch a count often has one of its own within 10 us by chance,
so an afterpulse seldom changes which is nearest, but always adds one
pair. Pairs in a bunch are not independent, so how far an excess stands
out is judged against the scatter of the bunches' own contributions to
it, not against Poisson's.

Counts are keyed by bunch and time in the tagger's bins, as one integer,
so every search is exact.

Depends on: numpy; cls_acquisition.hardware.timetagger (BIN_US, TIMESTAMP_BITS).
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
    """One PMT's counts against the counts before them on itself and on the
    others: the time back to the previous of each (``Same``, ``Other``, NaN
    where none), and chance, as the same against the next MIXED_BUNCHES
    bunches' counts at the same times, each weighed as if every count had
    its next bunches."""

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
        """``(excess, standard errors, own ratio, others' ratio)``: the pairs
        on this PMT with a time in [low, high] us beyond chance scaled as the
        other PMTs' pairs stand above theirs, how far that stands out against
        the scatter of the bunches' own contributions to it, and each side's
        pairs over chance."""
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
    """Variance/mean of each of ``wanted`` channels' counts in the cells of
    ``shape`` (scans, setpoints), across scans at each setpoint, pooled:
    1 for Poisson counts. ``cells`` is each count's cell (-1: none),
    ``dropped`` the counts left out (a dead time). Returns ``({channel:
    variance/mean}, the excess common to every pair of channels, degrees
    of freedom)``: what every PMT shares is the beam's, not a tube's."""
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
