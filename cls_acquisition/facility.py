"""The facility's services: the ELOG, the EPICS process variables, and the cooler's PVs.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

All remote and optional, so none may freeze the window or hold up a scan.
ELOG posts go out in order from one worker thread, a failure logged; PVs
are monitored, so a read returns the last value pushed, or None. With the
ELOG disabled, posts go to the session log; a simulated rack never posts
and reads fixed PVs. :class:`Cooler_Pv_Server` publishes the old DAQ's
cooler PVs from ORACLE's own poll, so the meter keeps one owner (Mode A).

Depends on: py-elog, pyepics, pcaspy, each imported when first used.
"""

import getpass
import logging
import math
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from functools import partial

SIMULATED_PVS = {'wavenumber': 16956.17, 'mass': 59.0, 'mass_setpoint': 59.0}
COOLER_PVS = {'Readback': {'prec': 4, 'unit': 'V', 'value': 0.0},
              'InstantValue': {'prec': 4, 'unit': 'V', 'value': 0.0}}
COOLER_MEAN_OF = 10                 # readings in Readback's mean, as the old server took
COOLER_PUBLISH_S = 0.1              # between updates; the meter is polled every 50 ms
CA_PROCESS_S = 0.01                 # how long each pass serves Channel Access requests

log = logging.getLogger('oracle.facility')


class Pv_Monitors():
    """The latest value of each named PV, kept current by EPICS callbacks; a
    key with no name is not on EPICS, and has no value."""

    def __init__(self, names, simulated):
        self._Values = dict(SIMULATED_PVS) if simulated else {}
        self._Pvs = []
        if not simulated:
            import epics
            for key, name in names.items():
                if not name:
                    continue
                self._Pvs.append(epics.PV(name, auto_monitor=True,
                                          callback=partial(self._Keep, key)))

    def _Keep(self, key, value=None, **_):
        """An EPICS monitor callback: the PV's new value, by its key."""
        self._Values[key] = value

    def Value(self, key):
        return self._Values.get(key)

    def Close(self):
        for pv in self._Pvs:
            pv.disconnect()


class Elog():
    """The run log on the facility's ELOG: one entry kept open and appended
    to, as the old DAQ did; New_Entry starts the next."""

    def __init__(self, url, author, enabled):
        self.Enabled = enabled
        self.Last_Error = None
        self._Url = url
        self._Author = author or getpass.getuser()
        self._Client = None
        self._Entry = None              # id of the entry being appended to
        self._Jobs = queue.Queue()
        self._Thread = threading.Thread(target=self._Work, name='oracle-elog', daemon=True)
        self._Thread.start()

    def New_Entry(self, experiment):
        self._Jobs.put(('new', '', experiment, None))

    def Append(self, text, experiment, attachment=None):
        """Add ``text`` to the open entry; ``attachment`` is a file the post
        takes over and deletes once sent."""
        self._Jobs.put(('append', text, experiment, attachment))

    def Close(self, timeout=5.0):
        self._Jobs.put(None)
        self._Thread.join(timeout)

    def _Work(self):
        while (job := self._Jobs.get()) is not None:
            attachment = job[3]
            try:
                self._Post(*job)
            except Exception as e:
                self.Last_Error = f"The ELOG post failed: {e}"
                log.warning("%s", self.Last_Error)
            finally:
                if attachment:
                    try:
                        os.remove(attachment)
                    except OSError:
                        pass                # a stray temp file must not stop the worker

    def _Post(self, kind, text, experiment, attachment):
        if not self.Enabled:
            log.info("ELOG (not sent): %s", text or 'new entry')
            return
        if self._Client is None:
            import elog
            self._Client = elog.open(self._Url)
        if kind == 'new' or self._Entry is None:
            self._Entry = self._Client.post(
                datetime.now().strftime('%d/%m/%Y %H:%M:%S') + '\n', Author=self._Author,
                What='Measurement', Experiment=experiment)
            if kind == 'new':
                return
        message, _, _ = self._Client.read(self._Entry)
        self._Entry = self._Client.post(message + '\n' + text, msg_id=self._Entry,
                                        attachments=[attachment] if attachment else None)


class Cooler_Pv_Server():
    """The cooler as EPICS PVs, ``<prefix>Readback`` (the mean of the last
    COOLER_MEAN_OF readings) and ``<prefix>InstantValue``, volts at the
    cooler from ``read_volts()``; a non-number marks both invalid."""

    def __init__(self, prefix, read_volts):
        try:
            import pcaspy
        except ImportError as e:
            raise RuntimeError("Publishing the cooler's PVs needs pcaspy (EPICS's Channel "
                               "Access server for Python), which is not installed here; "
                               "install it, or set cooler_pv: enabled to false.") from e
        self._Pcaspy = pcaspy
        self._Read = read_volts
        self._Recent = deque(maxlen=COOLER_MEAN_OF)
        self._Server = pcaspy.SimpleServer()
        self._Server.createPV(prefix, COOLER_PVS)
        self._Driver = pcaspy.Driver()
        self._Running = True
        self._Thread = threading.Thread(target=self._Serve, name='oracle-cooler-pvs',
                                        daemon=True)
        self._Thread.start()
        log.info("Publishing the cooler's voltage as %sReadback and %sInstantValue", prefix,
                 prefix)

    def _Serve(self):
        published = 0.0
        while self._Running:
            self._Server.process(CA_PROCESS_S)
            if time.monotonic() - published >= COOLER_PUBLISH_S:
                published = time.monotonic()
                try:
                    self._Publish(self._Read())
                except Exception:
                    log.exception("Publishing the cooler's PVs failed")

    def _Publish(self, volts):
        alarm, severity = self._Pcaspy.Alarm, self._Pcaspy.Severity
        if math.isfinite(volts):
            self._Recent.append(volts)
            values = {'Readback': sum(self._Recent) / len(self._Recent), 'InstantValue': volts}
            for reason, value in values.items():
                self._Driver.setParam(reason, value)
                self._Driver.setParamStatus(reason, alarm.NO_ALARM, severity.NO_ALARM)
        else:
            for reason in COOLER_PVS:
                self._Driver.setParamStatus(reason, alarm.READ_ALARM, severity.INVALID_ALARM)
        self._Driver.updatePVs()

    def Close(self):
        self._Running = False
        self._Thread.join(timeout=1.0)
