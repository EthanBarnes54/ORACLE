"""The cronologic TimeTagger4, through the rack's compiled wrapper (timetagger4).

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The wrapper hands back, per trigger, rows of [bunch, channel, time] with
channels counted from 1 and times in TDC bins of 500 ps; this module
hands the loop microseconds. A trigger that carried no hits is an empty
array, and no trigger within the timeout is None.

Two faults in the wrapper's C++ source (collinear_daq_driver_wrap.cpp)
change what is recorded and cannot be fixed from Python: input 1 is
never enabled, whatever is asked, and inputs 2 and 3 take input 1's edge
setting instead of their own.

Depends on: numpy; timetagger4, from the folder settings name
(instruments: tagger_driver) or the import path.
"""

import os
import sys
import time

import numpy as np

BIN_US = 0.0005                     # one TDC bin
POLL_S = 0.00005                    # between polls of the card while waiting
WINDOW_US = (0.0, 5000.0)           # until Configure says otherwise


class Time_Tagger():

    def __init__(self, levels, trigger_level, driver_dir=''):
        if driver_dir:
            sys.path.insert(0, driver_dir)
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(driver_dir)
        from timetagger4 import TimeTagger
        self._Card_Type = TimeTagger
        self._Levels = list(levels)
        self._Trigger_Level = trigger_level
        self._Window = WINDOW_US
        self._Card = None
        self._Reading = False
        self._Open()                    # fails now, at start-up, if the card is absent

    def Configure(self, start_us, stop_us):
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
