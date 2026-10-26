"""The cronologic TimeTagger4, through the rack's compiled wrapper (timetagger4).

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The wrapper hands back, per trigger, rows of [bunch, channel, time] with
channels counted from 1 and times in TDC bins of 500 ps; this module
hands the loop microseconds. A trigger that carried no hits is an empty
array, and no trigger within the timeout is None.

Three faults in the old DAQ's wrapper (collinear_daq_driver_wrap.cpp)
change what is recorded and cannot be fixed from Python. Inputs 3 and 4
take input 2's edge instead of their own, so with the rack's thresholds
the UV PMTs are timed on their falling edge. It never reads a hit's
flags, so the rollover marks the card sends past the 24-bit timestamp
range would be recorded as hits. And a card that fails to open is
reported only on the console. ORACLE keeps every window inside the
timestamp range, and says at start-up when the edges asked for are not
the edges the old wrapper gives; the corrected wrapper, built from
cls_acquisition/hardware/timetagger4_wrapper, mends all three and marks
itself ORACLE_WRAPPER.

Depends on: numpy; timetagger4, from the folder settings name
(instruments: tagger_driver) or the import path.
"""

import logging
import os
import sys
import time

import numpy as np

BIN_US = 0.0005                     # one TDC bin
TIMESTAMP_BITS = 24                 # a hit's timestamp; past it the card marks a rollover
MAX_WINDOW_US = (2 ** TIMESTAMP_BITS - 1) * BIN_US
POLL_S = 0.00005                    # between polls of the card while waiting
WINDOW_US = (0.0, 5000.0)           # until Configure says otherwise

log = logging.getLogger('oracle.tagger')


def Check_Window(start_us, stop_us):
    """Raises ValueError unless the acquisition window fits the timestamp range."""
    if not 0.0 <= start_us < stop_us <= MAX_WINDOW_US:
        raise ValueError(f"The tagger's window from {start_us:g} to {stop_us:g} us does not "
                         f"fit its {MAX_WINDOW_US:.1f} us timestamp range; past it the card "
                         f"marks rollovers the wrapper would record as hits.")


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


class Time_Tagger():

    def __init__(self, levels, trigger_level, driver_dir=''):
        if driver_dir:
            sys.path.insert(0, driver_dir)
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(driver_dir)
        import timetagger4
        self._Card_Type = timetagger4.TimeTagger
        self._Levels = list(levels)
        self._Trigger_Level = trigger_level
        self._Window = WINDOW_US
        self._Card = None
        self._Reading = False
        fault = Edge_Fault(self._Levels, getattr(timetagger4, 'ORACLE_WRAPPER', False))
        self.Warnings = [fault] if fault else []
        for warning in self.Warnings:
            log.warning("%s", warning)
        self._Open()                    # fails now, at start-up, if the card is absent

    def Configure(self, start_us, stop_us):
        Check_Window(start_us, stop_us)
        self._Window = (start_us, stop_us)

    def _Open(self):
        self.Stop()
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
                rows = np.array(hits, dtype=float).reshape(-1, 3)
                rows[:, 2] *= BIN_US
                return rows
            if status != 1:
                raise RuntimeError(f"The time tagger reported status {status} while being "
                                   f"read.")
            time.sleep(POLL_S)
        return None

    def Drain(self):
        """Drop every packet waiting; returns how many there were."""
        dropped = 0
        while self._Card is not None:
            status, _ = self._Card.getPackets()
            if status != 0:
                return dropped
            dropped += 1
        return dropped

    def Stop(self):
        if self._Card is None:
            return
        if self._Reading:
            self._Card.stopReading()
            self._Reading = False
        self._Card.stop()
        self._Card = None
