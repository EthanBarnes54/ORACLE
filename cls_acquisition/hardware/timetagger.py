"""The cronologic TimeTagger4, through the rack's compiled wrapper (timetagger4).

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A read gives every ready trigger as rows of [trigger, channel, time]: the
card's number, inputs from 1, times in us, and a MARK row per trigger;
None if none came in time. The old DAQ's wrapper has four faults (inputs 3
and 4 take input 2's edge, hit flags go unread, a failed open is only
printed, no trigger is marked, which this module rebuilds); the corrected
wrapper (timetagger4_wrapper, ORACLE_WRAPPER 2) mends all four. Each run
records its wrapper (:attr:`Time_Tagger.Identity`). Every threshold is
held to the card's range (:func:`Check_Levels`), which the old DAQ's
2.5 V trigger was past: driver 1.13 refuses it, and the old wrapper never
said so.

Depends on: numpy; timetagger4, from the settings' folder
(instruments: tagger_driver) or the import path.
"""

import logging
import os
import sys
import time

import numpy as np

BIN_US = 0.0005                     # one TDC bin
MARK = 0                            # a trigger's own row; inputs are 1 to 4
MARKED_WRAPPER = 2                  # the ORACLE_WRAPPER that marks every trigger
TIMESTAMP_BITS = 24                 # a hit's timestamp; past it the card marks a rollover
MAX_WINDOW_US = (2 ** TIMESTAMP_BITS - 1) * BIN_US
POLL_S = 0.00005                    # between polls of the card while waiting
WINDOW_US = (0.0, 5000.0)           # until Configure says otherwise
# the thresholds (dc_offset) the card's inputs take, as cronologic's driver 1.13
# refuses anything else; the trigger's TTL sits at the top
LEVEL_RANGE_V = (-1.32, 1.18)

log = logging.getLogger('oracle.tagger')


def Check_Window(start_us, stop_us):
    """Raises ValueError unless the acquisition window fits the timestamp range."""
    if not 0.0 <= start_us < stop_us <= MAX_WINDOW_US:
        raise ValueError(f"The tagger's window from {start_us:g} to {stop_us:g} us does not "
                         f"fit its {MAX_WINDOW_US:.1f} us timestamp range; past it the card "
                         f"marks rollovers the wrapper would record as hits.")


def Check_Levels(levels, trigger_level):
    """Raises ValueError unless the trigger's threshold and each input's are
    numbers inside the card's range; an input's must not be 0 V, which has no
    edge (a negative one times falling edges, a positive one rising)."""
    low, high = LEVEL_RANGE_V
    for name, level, signed in (('trigger', trigger_level, False),
                                *((f"input {i}", level, True)
                                  for i, level in enumerate(levels, start=1))):
        if isinstance(level, bool) or not isinstance(level, (int, float)):
            raise ValueError(f"The tagger's {name} threshold is {level!r}, not a number of "
                             f"volts.")
        if not low <= level <= high:
            raise ValueError(f"The tagger's {name} threshold of {level:+.2f} V is outside the "
                             f"{low:+.2f} to {high:+.2f} V its inputs take.")
        if signed and level == 0:
            raise ValueError(f"The tagger's {name} threshold is 0 V, which picks no edge: "
                             f"give it the sign of the input's pulses.")


def Edge_Fault(levels, corrected):
    """The sentence saying which inputs the old wrapper times on another
    input's edge than their thresholds ask for, or None."""
    if corrected or len(levels) < 3:
        return None
    wrong = [i + 1 for i in range(2, len(levels)) if (levels[i] > 0) != (levels[1] > 0)]
    if not wrong:
        return None
    edge = 'rising' if levels[1] > 0 else 'falling'
    return (f"The time tagger's wrapper times input {' and '.join(map(str, wrong))} on input "
            f"2's {edge} edge, not the edge their thresholds ask for; rebuild it from "
            f"cls_acquisition/hardware/timetagger4_wrapper.")


def Unmarked():
    """The sentence for a wrapper that marks no trigger."""
    return ("The time tagger's wrapper marks no trigger, so a read holding only triggers with "
            "no hits is counted as one bunch however many it held; rebuild it from "
            "cls_acquisition/hardware/timetagger4_wrapper.")


def Mark(rows, last):
    """The old wrapper's ``rows`` with each trigger marked: its packet numbers
    count every trigger from 0, so every number after ``last`` up to the
    highest here is one (with no rows, the next). Returns ``(rows, last)``."""
    top = max(int(rows[:, 0].max()) if len(rows) else last + 1, last + 1)
    numbers = np.arange(last + 1, top + 1, dtype=float)
    marks = np.column_stack([numbers, np.full(len(numbers), MARK), np.zeros(len(numbers))])
    return np.vstack([marks, rows]), top


class Time_Tagger():

    def __init__(self, levels, trigger_level, driver_dir=''):
        Check_Levels(levels, trigger_level)
        if driver_dir:
            sys.path.insert(0, driver_dir)
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(driver_dir)
        try:
            import timetagger4
        except (ImportError, OSError) as e:
            raise RuntimeError(f"The time tagger's wrapper (timetagger4) could not be loaded "
                               f"from {driver_dir or 'the import path'}: it is a Windows module, "
                               f"built on the rack PC from cls_acquisition/hardware/"
                               f"timetagger4_wrapper, with the cronologic driver beside it "
                               f"(instruments: tagger_driver) ({e}).") from e
        self._Card_Type = timetagger4.TimeTagger
        self._Levels = list(levels)
        self._Trigger_Level = trigger_level
        self._Window = WINDOW_US
        self._Card = None
        self._Reading = False
        self._Last = -1                    # the old wrapper's last trigger read
        version = getattr(timetagger4, 'ORACLE_WRAPPER', 0)
        self.Identity = {'Wrapper': version, 'Serial': None}
        self._Marked = version >= MARKED_WRAPPER
        self.Warnings = self._Wrapper_Warnings()
        for warning in self.Warnings:
            log.warning("%s", warning)
        self._Open()                    # fails now, at start-up, if the card is absent

    def _Wrapper_Warnings(self):
        """What the loaded wrapper gets wrong with these thresholds."""
        fault = Edge_Fault(self._Levels, bool(self.Identity['Wrapper']))
        return ([fault] if fault else []) + ([] if self._Marked else [Unmarked()])

    def Configure(self, start_us, stop_us):
        Check_Window(start_us, stop_us)
        self._Window = (start_us, stop_us)

    def Set_Levels(self, levels, trigger_level):
        """The thresholds the card is opened with from now on, put to it at
        once, so one it refuses is refused here and the last kept. Raises
        ValueError for a threshold outside its range or a change of inputs."""
        Check_Levels(levels, trigger_level)
        if len(levels) != len(self._Levels):
            raise ValueError(f"{len(levels)} input thresholds were given for the tagger's "
                             f"{len(self._Levels)} inputs.")
        kept = self._Levels, self._Trigger_Level
        self._Levels, self._Trigger_Level = list(levels), trigger_level
        try:
            self._Open()
        except Exception:
            self._Levels, self._Trigger_Level = kept
            self._Open()
            raise
        self.Warnings = self._Wrapper_Warnings()

    def _Open(self):
        self.Stop()
        self._Last = -1                    # a new card counts its packets from 0
        start, stop = (int(round(t / BIN_US)) for t in self._Window)
        options = {'trigger_level': self._Trigger_Level, 'trigger_rising': True}
        for channel, level in enumerate(self._Levels):
            options.update({f'channel_{channel}_used': True,
                            f'channel_{channel}_level': level,
                            f'channel_{channel}_rising': level > 0,
                            f'channel_{channel}_start': start,
                            f'channel_{channel}_stop': stop})
        self._Card = self._Card_Type(**options)

    def Start(self):
        self._Open()
        self._Card.startReading()
        self._Reading = True

    def Read(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status, hits = self._Card.getPackets()
            if status == 0:
                return self._Rows(hits)
            if status != 1:
                raise RuntimeError(f"The time tagger reported status {status} while being "
                                   f"read.")
            time.sleep(POLL_S)
        return None

    def _Rows(self, hits):
        rows = np.array(hits, dtype=float).reshape(-1, 3)
        rows[:, 2] *= BIN_US
        if not self._Marked:
            rows, self._Last = Mark(rows, self._Last)
        return rows

    def Drain(self):
        """Drop every trigger waiting; returns how many there were."""
        dropped = 0
        while self._Card is not None:
            status, hits = self._Card.getPackets()
            if status != 0:
                return dropped
            dropped += int(np.count_nonzero(self._Rows(hits)[:, 1] == MARK))
        return dropped

    def Stop(self):
        if self._Card is None:
            return
        if self._Reading:
            self._Card.stopReading()
            self._Reading = False
        self._Card.stop()
        self._Card = None
