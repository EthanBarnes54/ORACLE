"""The facility's services: the ELOG, and the EPICS process variables.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Both are remote and both are optional, so neither may freeze the window
or hold up a scan. ELOG posts queue on one worker thread and go out in
order; a failed post is logged and reported, never raised into the DAQ.
PVs are monitored, so reading one returns the last value EPICS pushed,
or None, without waiting on the network.

With the ELOG disabled, each post goes to the session log instead; a
simulated rack never posts, so practice runs cannot reach the real log.
A simulated rack reads fixed PV values.

Depends on: py-elog and pyepics, each imported only when first used.
"""

import getpass
import logging
import os
import queue
import threading
from datetime import datetime
from functools import partial

SIMULATED_PVS = {'wavenumber': 16956.17, 'mass': 59.0, 'mass_setpoint': 59.0}

log = logging.getLogger('oracle.facility')


class Pv_Monitors():
    """The latest value of each named PV, kept current by EPICS callbacks."""

    def __init__(self, names, simulated):
        self._Values = dict(SIMULATED_PVS) if simulated else {}
        self._Pvs = []
        if not simulated:
            import epics
            for key, name in names.items():
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
    """The run log on the facility's ELOG.

    Like the old DAQ, ORACLE keeps one entry open and appends to it: a scan
    adds its parameters and calibration, a stop adds a line, a screenshot is
    attached. New_Entry starts the next one.
    """

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
