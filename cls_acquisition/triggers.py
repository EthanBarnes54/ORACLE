"""The PulseBlaster's trigger patterns, and the time slices that program them.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A pattern is a list of ``[channel, on_ns, off_ns]``, repeated every
period; the channel numbers are the rack's wiring. Calibration has no
pattern: step and confirm takes the PulseBlaster out of it altogether.

Depends on: nothing outside the standard library.
"""

CH_CYCLE = 1                    # high through the whole cycle
CH_BUNCHER = 17                 # buncher release, which also triggers the Keysight
CH_TAGGER = 18                  # opens the time tagger's acquisition window
CH_NEW_BUNCHER = (9, 11, 15)    # the new buncher's cycle inputs
TAGGER_PULSE_NS = 100
DC_PERIOD_NS = 1_000_000        # DC mode: one tagger window per millisecond


def Bunched_Pattern(rep_rate_hz, release_us, tagger_delay_us):
    """One bunch per period: the buncher line goes high ``release_us`` into
    the cycle, and the tagger's window opens ``tagger_delay_us`` in."""
    period = 1e9 / rep_rate_hz
    start = tagger_delay_us * 1000
    pattern = [[CH_CYCLE, 0, period], [CH_BUNCHER, release_us * 1000, period],
               [CH_TAGGER, start, start + TAGGER_PULSE_NS]]
    return pattern + [[channel, 0, period] for channel in CH_NEW_BUNCHER]


def Dc_Pattern():
    pattern = [[CH_CYCLE, 0, DC_PERIOD_NS], [CH_BUNCHER, 0, DC_PERIOD_NS],
               [CH_TAGGER, 0, TAGGER_PULSE_NS]]
    return pattern + [[channel, 0, DC_PERIOD_NS] for channel in CH_NEW_BUNCHER]


def Period_Ns(pattern):
    return max(off for _, _, off in pattern)


def Time_Slices(pattern):
    """``[[duration_ns, bits], ...]``: one slice between each pair of
    neighbouring edges, with the bit of every channel high throughout it
    set (channel n is bit n - 1)."""
    edges = sorted({0} | {t for _, on, off in pattern for t in (on, off)})
    slices = []
    for start, stop in zip(edges, edges[1:]):
        bits = 0
        for channel, on, off in pattern:
            if on <= start and off >= stop:
                bits |= 1 << (channel - 1)
        slices.append([stop - start, bits])
    return slices
