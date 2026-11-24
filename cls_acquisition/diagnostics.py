"""Diagnostics a run can stream beside it (WP9), never into its file.

Date:    2026-11-24
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Ticked on the Scan tab, a run's line meter is streamed for the whole scan
instead of polled: a reading every READBACK_INTERVAL_S at the poll's own
200 us aperture, twenty a mains cycle, to tell whether the mains the
readback carries (76 mV at 200 us, A. Raggio, November 2026) is on the
platform or only in the readback. The readings go to
``<data>/diagnostics/run_NNNN/readback.f32`` (float32 platform volts, a
chunk as it is read, so memory stays bounded; 14 MB an hour), and
``readback.json`` says when each of its records started, at what interval,
and how it ended. A run longer than the meter's longest record goes on in
a new one. The run file and its monitor record are as without it: the
monitor samples the latest streamed reading as it did the poll's. A stream
that fails puts the meter back on its poll and says why; the run goes on.

Depends on: numpy.
"""

import json
import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

FOLDER = 'diagnostics'
READBACK_INTERVAL_S = 0.001         # twenty readings a 50 Hz cycle
READBACK_APERTURE_S = 200e-6        # the poll's, so the monitor's readback is unchanged
READBACK_HOURS = 12.0               # one record at most this long, if the meter holds it
READ_EVERY_S = 0.1                  # as often as the monitor samples the latest
CHUNK = 20000                       # readings per R?
DATA, META = 'readback.f32', 'readback.json'


def Folder(run):
    """Where run ``run``'s diagnostics go, relative to the data folder."""
    return os.path.join(FOLDER, f"run_{run:04d}")


class Readback_Stream():
    """The line ``meter`` (on a ``divider``) streamed into ``folder`` from now
    until :meth:`Stop`, read out by a thread of its own."""

    def __init__(self, meter, divider, folder):
        self.Folder, self._Meter, self._Divider = folder, meter, divider
        self.Taken, self.Error, self.Segments, self.Interval_S = 0, None, [], None
        self._Stop = threading.Event()
        self._Count = self._Left = 0
        os.makedirs(folder, exist_ok=True)
        self._File = open(os.path.join(folder, DATA), 'xb')
        self._Thread = threading.Thread(target=self._Run, daemon=True, name='diagnostics')
        self._Thread.start()

    def _Arm(self):
        """A new record, as long as the meter takes, started now."""
        self._Count = int(min(READBACK_HOURS * 3600.0 / READBACK_INTERVAL_S,
                              self._Meter.Stream_Limit()))
        self.Interval_S = self._Meter.Capture(self._Count, READBACK_INTERVAL_S,
                                              aperture_s=READBACK_APERTURE_S, stream=True)
        self.Segments.append({'Started': time.time(), 'First': self.Taken,
                              'Interval_S': self.Interval_S})
        self._Meter.Arm()
        self._Left = self._Count

    def _Read(self):
        """Everything the meter holds, to the file; a new record once one is full."""
        while self._Left > 0:
            volts = self._Meter.Read_Available(min(CHUNK, self._Left)) * self._Divider
            self._File.write(volts.astype('<f4').tobytes())
            self.Taken += len(volts)
            self._Left -= len(volts)
            if self._Left <= 0 and not self._Stop.is_set():
                self._Arm()
            if len(volts) < CHUNK:
                return

    def _Run(self):
        try:
            self._Arm()
            while not self._Stop.wait(READ_EVERY_S):
                self._Read()
            self._Read()
        except Exception as e:
            self.Error = f"{type(e).__name__}: {e}"
            log.warning("The HV readback stream to %s stopped: %s", self.Folder, self.Error)
        finally:
            self._File.close()
            try:
                self._Meter.Release()
            except Exception:
                log.exception("The line meter did not go back to its poll after the stream")

    def Stop(self):
        """End the stream, the meter back on its poll; writes and returns what
        ``readback.json`` says of it."""
        self._Stop.set()
        self._Thread.join(timeout=10.0)
        meta = {'Readings': self.Taken, 'Volts': 'platform, float32 little-endian',
                'Aperture_S': READBACK_APERTURE_S, 'Records': self.Segments,
                'Ended': time.time(), 'Error': self.Error}
        with open(os.path.join(self.Folder, META), 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=1)
        return meta


def Read(folder):
    """``(times, platform volts)`` of a stream: each reading's time.time() from
    its record's start and interval."""
    with open(os.path.join(folder, META), encoding='utf-8') as fh:
        meta = json.load(fh)
    volts = np.fromfile(os.path.join(folder, DATA), dtype='<f4').astype(float)
    times = np.full(len(volts), np.nan)
    bounds = [r['First'] for r in meta['Records']] + [len(volts)]
    for record, first, last in zip(meta['Records'], bounds, bounds[1:]):
        times[first:last] = record['Started'] + record['Interval_S'] * np.arange(last - first)
    return times, volts
