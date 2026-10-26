"""The acquisition process: the loop that drives the DAC, the triggers and the tagger.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The loop runs in its own process, so the tagger is read on time however
busy the window is. The GUI talks to it through :class:`Rack_Link`. An
instruction is ``(request id, command, parameters)`` and every one gets
exactly one reply carrying the same id, ``(id, ok, payload)``, so a late
reply can never be read as the answer to a later question. A command
the loop does not know is refused, not ignored, and a handler that fails
replies with the failure rather than leave the GUI waiting. What both
sides watch all the time (the mode flags, the voltage on the DAC, the
round, the bunch count) is shared memory, read without a round trip.

Instructions are taken between tagger reads, so a stop lands within one
bunch. Calibration is stepped from the GUI: 'Calibrate', one
'Calibration_Step' per setpoint, then 'Stop_Calibration'; meanwhile the
loop writes the DAC only when told to. The hardware calibrations step
the DAC the same way. A scan or calibration plan beyond the DAC's limit
is refused whole, before its first write.

A scan step is settle-aware. The PulseBlaster, and the buncher and
cooler with it, runs on untouched; what stops is acquisition. At each
step the tagger's window is shut, the bunches still in flight are
dropped (they are past their step's quota), the DAC is written, and the
window stays shut for as long as the settle model says a jump of that
size takes. Whatever arrived meanwhile is dropped too: with no gate
wired, those are the bunches triggered while the platform moved. Only
then does the step's count of bunches begin, so every step holds exactly
its bunches, all taken at a settled voltage, however far apart the
setpoints are.

The bunch column of the data is therefore the loop's own count of
accepted bunches from 0, exactly ``bunches_per_step`` a step, which is
what DENIS assumes when it numbers scans. The tagger's own count is
checked against it on every bunch with hits: a difference means a read
missed a trigger, or timed out with none and was counted anyway (the
collinear DAQ's D5), and is counted and logged.

Data leave on their own queue as ``('scan' | 'stream', rows)`` packets in
the run file's columns, and a scan's end is marked ``('scan_end', None)``
after its last rows, so the GUI closes the run file only once all of
them are in it. If the loop dies, its last word is a reply with id 0
saying why. 'Ping' answers with what the devices found wrong as they
opened (the tagger's wrapper, say), for the window to show.

Depends on: numpy; cls_acquisition.calibration, .triggers, and
cls_acquisition.hardware inside the process.
"""

import logging
import logging.handlers
import multiprocessing
import queue
import threading
import time

import numpy as np

from cls_acquisition import triggers
from cls_acquisition.calibration import Settle_For

SCAN_TYPES = ('Sawtooth', 'Triangle', 'Single scan')
IDLE_WAIT_S = 0.05          # longest an idle loop waits on the instruction queue
TAGGER_TIMEOUT_S = 1.0      # a read with no trigger: counted as a bunch, records nothing
REPLY_TIMEOUT_S = 10.0

log = logging.getLogger('oracle.acquisition')


class Acquisition_Error(RuntimeError):
    pass


def Ramp(start, stop, step_v):
    """The points from ``start`` to ``stop`` at most ``step_v`` apart, ``start``
    itself left out."""
    count = max(int(np.ceil(abs(stop - start) / step_v)), 1)
    return np.linspace(start, stop, count + 1)[1:].tolist()


class _Refused(Exception):
    """An instruction the loop will not carry out, and the sentence saying why."""


class Shared_State():
    """What both processes read at any time without asking."""

    def __init__(self, context):
        self.Scanning = context.Event()
        self.Triggering = context.Event()
        self.Outputting = context.Event()
        self.Calibrating = context.Event()
        self.Voltage = context.Value('d', 0.0)      # on the DAC now; the loop writes it
        self.Requested = context.Value('d', 0.0)    # quick tune's request; the GUI writes it
        self.Cooler = context.Value('d', 0.0)       # cooler meter volts; the GUI writes it
        self.Round = context.Value('i', 0)
        self.Bunches = context.Value('q', 0)
        self.Dropped = context.Value('q', 0)        # bunches dropped while a step settled
        self.Skipped = context.Value('q', 0)        # bunches the tagger's count and ours part by


class Acquisition_Loop():

    def __init__(self, dac, pulses, tagger, gate, instructions, replies, data, state):
        self._Dac, self._Pulses, self._Tagger, self._Gate = dac, pulses, tagger, gate
        self._Instructions, self._Replies, self._Data = instructions, replies, data
        self._State = state
        self._Plan = np.zeros(0)
        self._Index = 0
        self._Scan_Type = SCAN_TYPES[0]
        self._Bunches_Per_Step = 1
        self._Volts = None                  # on the DAC; None before the first write
        self._Last_Write = None
        self._Settle = None                 # the scan's settle model
        self._Settled_At = None             # monotonic time the window may open again
        self._Accepted = 0                  # bunches taken since the scan or stream began
        self._Tagger_Ref = None             # (tagger's bunch, ours) at the last hits
        self._Running = True
        # what the devices found wrong when they opened, for the window to show
        self._Warnings = [warning for device in (dac, pulses, tagger, gate)
                          for warning in getattr(device, 'Warnings', ())]
        self._Handlers = {
            'Ping': lambda: {'Warnings': self._Warnings},
            'Start_Scan': self._Start_Scan,
            'Stop_After_Scan': self._Stop_After_Scan,
            'Stop_Now': self._Stop_Now,
            'Start_Triggering': self._Start_Triggering,
            'Stop_Triggering': self._Stop_Triggering,
            'Output_Volts': self._Output_Volts,
            'Calibrate': self._Calibrate,
            'Calibration_Step': self._Calibration_Step,
            'Stop_Calibration': self._Stop_Calibration,
            'Set_Correction': self._Set_Correction,
            'Shutdown': self._Shutdown,
        }

    def Run(self, parent_alive=lambda: True):
        """Until 'Shutdown', or until the GUI's process is gone: an orphaned
        loop would hold the instruments and the platform's voltage."""
        state = self._State
        while self._Running and parent_alive():
            if state.Scanning.is_set():
                self._Scan_Step()
            elif state.Triggering.is_set():
                self._Take_Instructions()
                self._Hold_Voltage()
                if state.Triggering.is_set():
                    self._Record('stream')
            else:
                self._Take_Instructions(IDLE_WAIT_S)
                self._Hold_Voltage()

    def Close(self):
        """Leave the rack safe: triggers off and the platform at 0 V. Each
        step runs even if one before it failed."""
        for step in (self._Pulses.Stop, self._Tagger.Stop, self._Dac.Zero, self._Pulses.Close,
                     self._Gate.Open):
            try:
                step()
            except Exception:
                log.exception("Shutting down the rack: %s failed", step.__qualname__)

    # ── Instructions ────────────────────────────────────────────

    def _Take_Instructions(self, wait=0.0):
        try:
            request = (self._Instructions.get(timeout=wait) if wait
                       else self._Instructions.get_nowait())
        except queue.Empty:
            return
        while True:
            self._Answer(*request)
            try:
                request = self._Instructions.get_nowait()
            except queue.Empty:
                return

    def _Answer(self, request_id, command, params):
        try:
            handler = self._Handlers.get(command)
            if handler is None:
                raise _Refused(f"'{command}' is not an instruction the acquisition "
                               f"process knows.")
            reply = (request_id, True, handler(**params))
        except _Refused as e:
            reply = (request_id, False, str(e))
        except Exception as e:
            log.exception("'%s' failed in the acquisition process", command)
            reply = (request_id, False, f"'{command}' failed in the acquisition process: {e}")
        self._Replies.put(reply)

    def _Require_Idle(self, action):
        state = self._State
        for flag, what in ((state.Scanning, 'a scan is running'),
                           (state.Triggering, 'the triggers are running'),
                           (state.Calibrating, 'a calibration is running')):
            if flag.is_set():
                raise _Refused(f"Cannot {action} while {what}.")

    def _Require_Within_Limit(self, setpoints):
        try:
            self._Dac.Check(setpoints)
        except ValueError as e:
            raise _Refused(f"{e} Nothing was written.") from None

    def _Start_Scan(self, setpoints, scan_type, bunches_per_step, timing, settle=None):
        self._Require_Idle('start a scan')
        self._Require_Within_Limit(setpoints)
        if scan_type not in SCAN_TYPES:
            raise _Refused(f"'{scan_type}' is not a scan type; use one of "
                           f"{', '.join(SCAN_TYPES)}.")
        self._Plan = np.asarray(setpoints, dtype=float)
        self._Index = 0
        self._Scan_Type = scan_type
        self._Bunches_Per_Step = max(int(bunches_per_step), 1)
        self._Settle = settle
        self._State.Round.value = 1
        self._Start_Triggers(timing)
        self._Step_To(self._Plan[0])
        self._State.Scanning.set()
        return {}

    def _Stop_After_Scan(self):
        self._Scan_Type = 'Single scan'
        return {}

    def _Stop_Now(self):
        if self._State.Scanning.is_set():
            self._End_Scan()
        return {}

    def _Start_Triggering(self, timing):
        self._Require_Idle('start the triggers')
        self._Start_Triggers(timing)
        self._State.Triggering.set()
        return {}

    def _Stop_Triggering(self):
        self._State.Triggering.clear()
        self._Stop_Triggers()
        return {}

    def _Output_Volts(self, on):
        (self._State.Outputting.set if on else self._State.Outputting.clear)()
        return {}

    def _Calibrate(self, setpoints):
        self._Require_Idle('calibrate')
        self._Require_Within_Limit(setpoints)
        self._Plan = np.asarray(setpoints, dtype=float)
        self._State.Calibrating.set()
        return {'Dac_Status': self._Dac.Calibration_Status(),
                'Dac_Correction': self._Dac.Correction_Record(),
                'Correction': self._Dac.Correction}

    def _Calibration_Step(self, index):
        if not self._State.Calibrating.is_set():
            raise _Refused(f"Setpoint {index} was sent with no calibration in progress.")
        if not 0 <= index < len(self._Plan):
            raise _Refused(f"Setpoint {index} is outside the {len(self._Plan)} point "
                           f"calibration plan.")
        volts = float(self._Plan[index])
        channel, dac_volts = self._Write(volts, force=True)
        # stamped once the write has returned: the earliest the HV can settle from
        t_command = time.perf_counter_ns()
        return {'Index': index, 'Volts': volts, 'Channel': channel, 'Dac_V': dac_volts,
                'T_Command_Ns': t_command}

    def _Stop_Calibration(self, ramp_step_v=None, ramp_rate_v_s=None):
        """Out of calibration; a hardware calibration first ramps the platform
        down, since the idle loop would put 0 V on it in one move."""
        if ramp_step_v and self._Volts:
            for volts in Ramp(self._Volts, 0.0, ramp_step_v):
                time.sleep(ramp_step_v / ramp_rate_v_s)
                self._Write(volts)
        self._State.Calibrating.clear()
        return {}

    def _Set_Correction(self, correction):
        self._Require_Idle('change the DAC correction')
        self._Dac.Correction = correction
        self._Volts = None              # the next write goes through the new correction
        return {'Dac_Status': self._Dac.Calibration_Status(),
                'Dac_Correction': self._Dac.Correction_Record()}

    def _Shutdown(self):
        for flag in (self._State.Scanning, self._State.Triggering, self._State.Outputting,
                     self._State.Calibrating):
            flag.clear()
        self._Running = False
        return {}

    # ── Data taking ─────────────────────────────────────────────

    def _Scan_Step(self):
        """Wait out the step's settling, then its bunches, then on to the next
        setpoint."""
        if self._Settled_At is not None:
            remaining = self._Settled_At - time.monotonic()
            if remaining > 0:
                self._Take_Instructions(min(remaining, IDLE_WAIT_S))
                return
            self._Drop()
            self._Gate.Open()
            self._Settled_At = None
        for _ in range(self._Bunches_Per_Step):
            self._Take_Instructions()
            if not self._State.Scanning.is_set():
                return
            self._Record('scan')
        self._Index += 1
        if self._Index == len(self._Plan):
            self._Index = 0
            if self._Scan_Type == 'Single scan':
                self._End_Scan()
                return
            self._State.Round.value += 1
            if self._Scan_Type == 'Triangle':
                self._Plan = self._Plan[::-1]
        self._Step_To(self._Plan[self._Index])

    def _Step_To(self, volts):
        """Shut the window, drop the bunches in flight, write the step, and
        hold the window shut while a jump of this size settles."""
        jump = abs(volts - (self._Volts or 0.0))
        self._Gate.Close()
        self._Drop()
        self._Write(volts)
        self._Settled_At = time.monotonic() + (Settle_For(self._Settle, jump) if jump else 0.0)

    def _Drop(self):
        self._State.Dropped.value += self._Tagger.Drain()
        self._Tagger_Ref = None             # the tagger counted bunches we did not take

    def _End_Scan(self):
        self._State.Scanning.clear()
        self._Settled_At = None
        self._Gate.Open()
        self._Stop_Triggers()
        self._Data.put(('scan_end', None))

    def _Record(self, kind):
        """Read one bunch and send its hits, tagged with the voltage they were
        taken at and the cooler voltage."""
        hits = self._Tagger.Read(TAGGER_TIMEOUT_S)
        self._State.Bunches.value += 1
        bunch, self._Accepted = self._Accepted, self._Accepted + 1
        if hits is None or not len(hits):
            return
        self._Check_Count(int(hits[0, 0]), bunch)
        rows = np.empty((len(hits), 6))
        rows[:, 0] = round(time.time(), 2)
        rows[:, 1] = self._Volts
        rows[:, 2] = bunch
        rows[:, 3:5] = hits[:, 1:]
        rows[:, 5] = self._State.Cooler.value
        self._Data.put((kind, rows))

    def _Check_Count(self, tagger_bunch, bunch):
        """The tagger's count must have moved on as far as ours since the
        last bunch with hits. Further, and a read missed a trigger; less, and
        a read that timed out with none was counted as a bunch."""
        if self._Tagger_Ref is not None:
            parted = (tagger_bunch - self._Tagger_Ref[0]) - (bunch - self._Tagger_Ref[1])
            if parted:
                if not self._State.Skipped.value:
                    log.warning("The tagger's bunch count and the loop's parted by %d; every "
                                "hit stays with the setpoint it was read at", parted)
                self._State.Skipped.value += abs(parted)
        self._Tagger_Ref = (tagger_bunch, bunch)

    def _Write(self, volts, force=False):
        """Put ``volts`` on the platform if it is not there already; returns
        the DAC's (channel, volts) of the last write."""
        if force or volts != self._Volts:
            self._Last_Write = self._Dac.Write(volts)
            self._Volts = volts
            self._State.Voltage.value = volts
        return self._Last_Write

    def _Hold_Voltage(self):
        """Quick tune's voltage while the output is on, 0 V otherwise. A scan
        or a calibration that has just started in this pass owns the DAC, and
        is left alone. A request the DAC refuses turns the output off rather
        than stop the loop."""
        state = self._State
        if state.Scanning.is_set() or state.Calibrating.is_set():
            return
        try:
            self._Write(state.Requested.value if state.Outputting.is_set() else 0.0)
        except ValueError:
            log.exception("The requested voltage was refused; voltage output is off")
            state.Outputting.clear()

    def _Start_Triggers(self, timing):
        """The pulses and the tagger running from a fresh count."""
        for counter in (self._State.Bunches, self._State.Dropped, self._State.Skipped):
            counter.value = 0
        self._Accepted = 0
        self._Tagger_Ref = None
        if timing['bunched']:
            pattern = triggers.Bunched_Pattern(timing['rep_rate_hz'], timing['release_us'],
                                               timing['tagger_delay_us'])
        else:
            pattern = triggers.Dc_Pattern()
        self._Pulses.Program(pattern)
        self._Tagger.Configure(timing['gate_start_us'], timing['gate_stop_us'])
        self._Tagger.Start()
        self._Pulses.Start()

    def _Stop_Triggers(self):
        self._Pulses.Stop()
        self._Tagger.Stop()


def Run_Loop(spec, instructions, replies, data, state, logs):
    """The acquisition process's entry point."""
    root = logging.getLogger('oracle')
    root.setLevel(logging.INFO)
    root.propagate = False
    root.addHandler(logging.handlers.QueueHandler(logs))
    # never block this process's exit on data the window will not read
    data.cancel_join_thread()
    loop = None
    try:
        from cls_acquisition import hardware
        loop = Acquisition_Loop(*hardware.Open_Loop_Devices(spec), instructions, replies,
                                data, state)
        loop.Run(multiprocessing.parent_process().is_alive)
        if loop._Running:
            log.critical("The GUI's process is gone; the rack is being left safe")
    except BaseException as e:
        log.critical("The acquisition process stopped", exc_info=True)
        replies.put((0, False, f"The acquisition process stopped: {e}"))
    finally:
        if loop is not None:
            loop.Close()


class _Forward(logging.Handler):
    """Hands the acquisition process's log records to this process's
    loggers of the same name, so both write one session log."""

    def emit(self, record):
        logging.getLogger(record.name).handle(record)


class Rack_Link():
    """The GUI's end of the acquisition process: starts it, asks it things,
    takes its data and stops it."""

    def __init__(self, spec):
        context = multiprocessing.get_context('spawn')
        self.State = Shared_State(context)
        self.Data = context.Queue()
        self._Instructions, self._Replies, self._Logs = (context.Queue(), context.Queue(),
                                                         context.Queue())
        self._Listener = logging.handlers.QueueListener(self._Logs, _Forward())
        self._Listener.start()
        self._Process = context.Process(
            target=Run_Loop, name='oracle-acquisition', daemon=True,
            args=(spec, self._Instructions, self._Replies, self.Data, self.State, self._Logs))
        self._Process.start()
        # start() lets go of its arguments, and shared memory freed here is
        # gone before the child has opened it; the spec lives as long as the link
        self._Spec = spec
        self._Lock = threading.Lock()
        self._Last_Id = 0
        self._Death = None

    def Send(self, command, timeout=REPLY_TIMEOUT_S, **params):
        """Send one instruction and return its reply's payload. Raises
        Acquisition_Error if it is refused, fails or goes unanswered."""
        with self._Lock:
            if self._Death:
                raise Acquisition_Error(self._Death)
            self._Last_Id += 1
            request_id = self._Last_Id
            self._Instructions.put((request_id, command, params))
            deadline = time.monotonic() + timeout
            while True:
                try:
                    reply_id, ok, payload = self._Replies.get(timeout=0.05)
                except queue.Empty:
                    if not self._Process.is_alive():
                        self._Death = (f"The acquisition process has stopped (exit code "
                                       f"{self._Process.exitcode}); restart ORACLE.")
                        raise Acquisition_Error(self._Death)
                    if time.monotonic() > deadline:
                        raise Acquisition_Error(f"The acquisition process did not answer "
                                                f"'{command}' within {timeout:.0f} s.")
                    continue
                if reply_id == 0:
                    self._Death = payload
                    raise Acquisition_Error(payload)
                if reply_id != request_id:
                    continue                # the answer to an instruction that timed out
                if not ok:
                    raise Acquisition_Error(payload)
                return payload

    def Alive(self):
        return self._Death is None and self._Process.is_alive()

    def Drain(self, budget_s=0.05):
        """The data packets waiting, taking at most ``budget_s`` to collect."""
        packets, deadline = [], time.monotonic() + budget_s
        while time.monotonic() < deadline:
            try:
                packets.append(self.Data.get_nowait())
            except queue.Empty:
                break
        return packets

    def Close(self):
        """Stop the acquisition process, leaving the platform at 0 V."""
        try:
            if self.Alive():
                self.Send('Shutdown', timeout=5.0)
        except Acquisition_Error as e:
            log.warning("%s", e)
        self._Process.join(timeout=5.0)
        if self._Process.is_alive():
            log.warning("The acquisition process did not stop; terminating it")
            self._Process.terminate()
            self._Process.join(timeout=2.0)
        self._Listener.stop()
