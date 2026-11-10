"""One owner thread per instrument: every command to a meter goes through it.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

:meth:`Visa_Session.Transaction` runs a caller's function whole on the
instrument's thread, so command sequences never interleave (the old DAQ's
Mode A). There is one session per resource, closed by its last user;
:func:`Pyvisa_Connect` takes VISA's exclusive lock. I/O failures raise
Instrument_IO_Error, naming the instrument.

Depends on: pyvisa, imported by :func:`Pyvisa_Connect` alone.
"""

import queue
import threading

ERROR_QUEUE_DEPTH = 16              # SYST:ERR? reads before giving up on a drain
LOCK_TIMEOUT_MS = 2000              # for the exclusive lock, held by another program


class Scpi_Error(RuntimeError):
    pass


class Instrument_IO_Error(RuntimeError):
    pass


class _Job():

    def __init__(self, fn):
        self.Fn = fn
        self.Result = self.Error = None
        self.Done = threading.Event()
        self.Abandoned = False


class Instrument_Proxy():
    """The handle a transaction is given, used only on the owner thread."""

    def __init__(self, inst):
        self._Inst = inst

    def Write(self, command):
        self._Inst.write(command)

    def Query(self, command):
        return self._Inst.query(command).strip()

    def Query_Slow(self, command, timeout_s):
        """A query the instrument takes longer than its I/O timeout to answer."""
        previous = self._Inst.timeout
        self._Inst.timeout = round(timeout_s * 1000)
        try:
            return self.Query(command)
        finally:
            self._Inst.timeout = previous

    def Clear(self):
        self._Inst.clear()

    def Errors(self):
        """Drain SYST:ERR? until it reports none, so a rejected command cannot
        pass for one that was obeyed."""
        errors = []
        for _ in range(ERROR_QUEUE_DEPTH):
            text = self.Query("SYST:ERR?")
            if text.startswith(('+0,', '0,')):
                break
            errors.append(text)
        return errors


class Visa_Session():

    _Registry = {}
    _Registry_Lock = threading.Lock()

    @classmethod
    def Open(cls, resource, connect):
        """The session for ``resource``, made with ``connect(resource)`` if
        there is no live one already."""
        with cls._Registry_Lock:
            session = cls._Registry.get(resource)
            if session is None or not session.Alive:
                session = cls(resource, connect(resource))
                cls._Registry[resource] = session
            session._Holders += 1
            return session

    def __init__(self, resource, inst):
        self.Resource = resource
        self.Alive = True
        self._Holders = 0
        self._Inst = inst
        self._Proxy = Instrument_Proxy(inst)
        self._Jobs = queue.Queue()
        self._Thread = threading.Thread(target=self._Run, name=f"visa:{resource[-12:]}",
                                        daemon=True)
        self._Thread.start()

    def _Run(self):
        while self.Alive:
            try:
                job = self._Jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if job.Abandoned:
                continue                    # its caller gave up: do not act on it late
            try:
                job.Result = job.Fn(self._Proxy)
            except BaseException as e:
                job.Error = e
            finally:
                job.Done.set()

    def Transaction(self, fn, timeout=30.0):
        """Run ``fn(proxy)`` on the owner thread and return its result."""
        if not self.Alive:
            raise Instrument_IO_Error(f"The session for {self.Resource} is closed.")
        job = _Job(fn)
        self._Jobs.put(job)
        if not job.Done.wait(timeout):
            job.Abandoned = True
            raise Instrument_IO_Error(f"The instrument at {self.Resource} did not finish a "
                                      f"transaction within {timeout:.0f} s.")
        if job.Error is not None:
            raise job.Error
        return job.Result

    def Close(self):
        with self._Registry_Lock:
            self._Holders -= 1
            if self._Holders > 0:
                return
            self.Alive = False
            self._Registry.pop(self.Resource, None)
        self._Thread.join(timeout=2.0)
        self._Inst.close()


_Manager = None


def Pyvisa_Connect(resource, timeout_ms=5000):
    global _Manager
    try:
        import pyvisa
        if _Manager is None:
            _Manager = pyvisa.ResourceManager()
    except (ImportError, OSError, ValueError) as e:
        # ValueError is pyvisa's "Could not locate a VISA implementation"
        raise Instrument_IO_Error(f"The meters need pyvisa and a VISA library (NI-VISA or "
                                  f"Keysight IO Libraries on Windows, NI-VISA on macOS); "
                                  f"install them, or set rack: simulated ({e}).") from e
    try:
        inst = _Manager.open_resource(resource)
    except pyvisa.errors.VisaIOError as e:
        raise Instrument_IO_Error(f"The instrument at {resource} could not be opened: "
                                  f"{e}") from e
    try:
        inst.lock_excl(timeout=LOCK_TIMEOUT_MS)
    except pyvisa.errors.VisaIOError as e:
        inst.close()
        raise Instrument_IO_Error(f"The instrument at {resource} is held by another program "
                                  f"(the bench calibration tool or the old DAQ?); close it "
                                  f"first: {e}") from e
    inst.timeout = timeout_ms
    inst.read_termination = inst.write_termination = '\n'
    return _Pyvisa_Instrument(inst, resource, pyvisa.errors.VisaIOError)


class _Pyvisa_Instrument():
    """A pyvisa resource whose I/O errors are Instrument_IO_Error, and
    whose lock is let go when it is closed."""

    def __init__(self, inst, resource, io_error):
        self._Inst, self._Resource, self._Io_Error = inst, resource, io_error

    def _Call(self, fn, *args):
        try:
            return fn(*args)
        except self._Io_Error as e:
            raise Instrument_IO_Error(f"The instrument at {self._Resource} did not answer: "
                                      f"{e}") from e

    @property
    def timeout(self):
        return self._Inst.timeout

    @timeout.setter
    def timeout(self, ms):
        self._Inst.timeout = ms

    def write(self, command):
        return self._Call(self._Inst.write, command)

    def query(self, command):
        return self._Call(self._Inst.query, command)

    def clear(self):
        return self._Call(self._Inst.clear)

    def close(self):
        try:
            self._Inst.unlock()
        except self._Io_Error:
            pass                        # closing lets the lock go in any case
        return self._Call(self._Inst.close)
