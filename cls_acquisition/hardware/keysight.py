"""The two Keysight 34465A meters: the line meter on the HV divider, and the cooler's.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Both are polled on their owner thread for the display, and both are a
small state machine, so an illegal move is refused rather than half
done: 'continuous' while polled, 'capturing' while a hardware
calibration takes timed readings, and 'idle' between. The line meter
also has 'stepping', for a calibration's single readings on command.
Neither stepping nor capturing goes straight back to continuous; the way
back is :meth:`_Meter.Release`, through idle, from any mode and as often
as it is called, and it sets the meter up afresh. The poll reads only in
'continuous' and checks that on the owner thread, so nothing can land
between the check and its READ?.

Every reading is taken on command, in every mode, and never on the bunch
trigger. The line meter's trigger input is wired to PulseBlaster channel
17, which releases the buncher: the old DAQ gated its calibration burst
on it, so the meter sampled on the bunch clock while the software
stepped the DAC on its own, and the two slipped apart (Mode B). Each mode
reads the trigger source back and refuses a meter that is not on
immediate, rather than trust the command was taken.

A meter is reset when it is opened (another program, the bench tool or
the old DAQ, may have left it streaming, in binary or on another
trigger) and must answer *IDN? as a 3446x or 3447x; its *OPT? says how
many readings its memory holds. It is reset again when it is closed, as
the bench tool leaves it, so whichever program opens it next finds it in
its power-on state, not part way through a capture. Every reading is checked for the meter's
overload (+-9.9E37) and invalid (9.91E37) markers, so a bad reading
raises instead of passing for a voltage.

A capture is armed (INIT) and fetched (FETC?) in two transactions, so
the caller can change the voltage while the meter is sampling: that is
how a step response is caught. A capture on the meter's sample timer
uses the interval the meter says it set, never the one asked for, which
it raises to its minimum for the integration time. A streamed capture
is read out and erased while it runs (R?), so it can be longer than the
meter's memory.

A poll that fails logs once and shows NaN until the meter answers again;
it never stops polling.

Depends on: numpy; .visa_session; cls_acquisition.calibration.
"""

import logging
import threading
import time

import numpy as np

from cls_acquisition.calibration import Calibration_Error
from cls_acquisition.hardware.visa_session import Scpi_Error

LINE_CONFIG = ("CONF:VOLT:DC 10,0.00003", "VOLT:APER:ENAB ON", "VOLT:DC:APER 200E-06",
               "VOLT:IMP:AUTO ON", "CALC:SCAL:FUNC SCAL", "CALC:SCAL:GAIN 1",
               "CALC:SCAL:OFFS 0", "CALC:SCAL:STAT OFF", "SAMP:COUN 1")
COOLER_CONFIG = ("CONF:VOLT:DC 10,0.00003", "VOLT:APER:ENAB ON", "VOLT:DC:APER 500E-06",
                 "VOLT:IMP:AUTO ON", "SAMP:COUN 1")
RANGE_V = 10.0                      # both meters' fixed range; they read to 120 % of it
# cooler platform volts per volt at its meter, unless the settings say otherwise
# (hardware: cooler_divider), and for every run that does not record its own
COOLER_DIVIDER = 10000.0
MODELS = ('3446', '3447')           # the Truevolt meters these commands are written for

POLL_PERIOD_S = 0.05
POLL_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 10.0               # one calibration reading, beyond its wait
FETCH_POLL_S = 0.02                 # how often a capture's progress is asked for
AUTOCAL_TIMEOUT_S = 120.0           # CAL:ADC? takes several seconds
STANDARD_MEMORY = 50000             # readings; 2,000,000 with the MEM option
MEM_MEMORY = 2000000
OVERLOAD = 9.9e37                   # the meter's mark for a reading it could not take

log = logging.getLogger('oracle.meters')


def Readings(reply, resource, what):
    """The values in a reply, as volts; raises Scpi_Error naming the meter
    if it marked any as over range or invalid. A block reply (R?) loses its
    IEEE 488.2 header first."""
    if reply.startswith('#'):
        digits = int(reply[1])
        reply = reply[2 + digits:] if digits else reply[2:]
    try:
        values = np.array([float(v) for v in reply.split(',') if v.strip()])
    except ValueError:
        raise Scpi_Error(f"The meter at {resource} answered {what} with {reply[:60]!r}, which "
                         f"is not a list of readings.") from None
    bad = ~np.isfinite(values) | (np.abs(values) >= OVERLOAD)
    if bad.any():
        raise Scpi_Error(f"The meter at {resource} marked {int(bad.sum())} of {len(values)} "
                         f"readings from {what} over range or invalid: its input is beyond "
                         f"its {RANGE_V:g} V range, or the divider is not connected.")
    return values


class _Meter():
    """A meter reset and set up when opened, then polled for the display,
    until it is asked for a capture."""

    _LEGAL = {'idle': ('continuous', 'capturing'),
              'continuous': ('idle', 'capturing'),
              'capturing': ('idle', 'capturing')}

    def __init__(self, session, config):
        self.Voltage = float('nan')
        self.Mode = 'continuous'
        self.Idn = ''
        self.Memory = STANDARD_MEMORY
        self.Capture_Interval_S = None  # the interval the meter set for the last capture
        self._Mode_Lock = threading.Lock()
        self._Session = session
        self._Config = config
        self._Polling = True
        self._Failing = False
        try:
            session.Transaction(self._Open)
        except Exception:
            session.Close()
            raise
        self._Thread = threading.Thread(target=self._Poll_Loop, daemon=True,
                                        name=f"poll:{session.Resource[-12:]}")
        self._Thread.start()

    def _Open(self, inst):
        """Reset, check what the instrument is, then set it up."""
        inst.Clear()
        inst.Write("*RST")
        inst.Write("*CLS")
        self.Idn = inst.Query("*IDN?")
        if not any(model in self.Idn for model in MODELS):
            raise Scpi_Error(f"The instrument at {self._Session.Resource} answers *IDN? with "
                             f"{self.Idn!r}, not a Keysight 3446x or 3447x meter.")
        self.Memory = MEM_MEMORY if 'MEM' in inst.Query("*OPT?") else STANDARD_MEMORY
        self._Setup(inst)

    def _Setup(self, inst):
        inst.Clear()
        inst.Write("*CLS")
        inst.Write("FORM:DATA ASC")
        for command in self._Config:
            inst.Write(command)
        for command in ("SAMP:SOUR IMM", "TRIG:SOUR IMM", "TRIG:COUN 1"):
            inst.Write(command)
        inst.Query("*OPC?")
        self._Refuse_On(inst.Errors(), "rejected its configuration")
        self._Refuse_External(inst)

    def _Refuse_External(self, inst):
        """Raise Scpi_Error unless the meter triggers on command (IMM)."""
        source = inst.Query("TRIG:SOUR?").upper()
        if not source.startswith('IMM'):
            raise Scpi_Error(f"The meter at {self._Session.Resource} reads TRIG:SOUR {source} "
                             f"where IMM was sent, so it would read on its trigger input (the "
                             f"buncher's line), not when asked.")

    def _Refuse_On(self, errors, what):
        """Raise Scpi_Error naming the meter if its error queue held anything."""
        if errors:
            raise Scpi_Error(f"The meter at {self._Session.Resource} {what}: "
                             f"{'; '.join(errors)}.")

    def _Poll_Loop(self):

        def poll(inst):
            if self.Mode == 'continuous':
                self.Voltage = float(Readings(inst.Query("READ?"), self._Session.Resource,
                                              'its poll')[0])

        while self._Polling:
            try:
                self._Session.Transaction(poll, timeout=POLL_TIMEOUT_S)
                if self._Failing:
                    log.info("The meter at %s is answering again", self._Session.Resource)
                    self._Failing = False
            except Exception as e:
                self.Voltage = float('nan')
                if not self._Failing:
                    log.warning("The meter at %s stopped answering its poll: %s",
                                self._Session.Resource, e)
                    self._Failing = True
            time.sleep(POLL_PERIOD_S)

    def _Move(self, mode):
        with self._Mode_Lock:
            if mode not in self._LEGAL[self.Mode]:
                raise RuntimeError(f"The meter at {self._Session.Resource} cannot go from "
                                   f"{self.Mode} to {mode}.")
            self.Mode = mode

    def Capture(self, count, interval_s=None, aperture_s=None, nplc=None, autozero=False,
                stream=False):
        """Set up a capture of ``count`` readings, one every ``interval_s``
        on the meter's timer (back to back if None), each integrating
        ``aperture_s`` or ``nplc`` power line cycles. A streamed capture may
        be longer than the meter's memory, as long as :meth:`Read_Available`
        keeps up. Every setting is read back before the first reading;
        returns the interval the meter set, also kept as
        Capture_Interval_S."""
        if count < 1 or (not stream and count > self.Memory):
            raise ValueError(f"A capture of {count} readings does not fit the meter's "
                             f"memory of {self.Memory}.")
        self._Move('capturing')
        integration = (("VOLT:APER:ENAB OFF", f"VOLT:DC:NPLC {nplc:g}") if nplc
                       else ("VOLT:APER:ENAB ON", f"VOLT:DC:APER {aperture_s:.6g}"))

        def config(inst):
            inst.Write("ABOR")
            if stream and count > int(float(inst.Query("SAMP:COUN? MAX"))):
                raise ValueError(f"A streamed capture of {count} readings is longer than the "
                                 f"meter at {self._Session.Resource} takes in one record.")
            for command in (*integration, f"VOLT:ZERO:AUTO {'ON' if autozero else 'OFF'}",
                            "TRIG:SOUR IMM", "TRIG:COUN 1", "TRIG:DEL 0"):
                inst.Write(command)
            interval = None
            if interval_s is not None:
                inst.Write("SAMP:SOUR TIM")
                shortest = float(inst.Query("SAMP:TIM? MIN"))
                if interval_s < shortest:
                    log.warning("The meter at %s samples every %.3g s at the shortest, not "
                                "%.3g s", self._Session.Resource, shortest, interval_s)
                inst.Write(f"SAMP:TIM {max(interval_s, shortest):.9g}")
                interval = float(inst.Query("SAMP:TIM?"))
            else:
                inst.Write("SAMP:SOUR IMM")
            inst.Write(f"SAMP:COUN {count}")
            inst.Query("*OPC?")
            samples = int(float(inst.Query("SAMP:COUN?")))
            self._Refuse_On(inst.Errors(), "rejected the capture configuration")
            self._Refuse_External(inst)
            if samples != count:
                raise Scpi_Error(f"The meter at {self._Session.Resource} reads SAMP:COUN "
                                 f"{samples} where {count} was sent.")
            return interval

        self.Capture_Interval_S = self._Session.Transaction(config)
        return self.Capture_Interval_S

    def Arm(self):
        """Start one capture; ``perf_counter_ns`` just before INIT goes out."""

        def arm(inst):
            if self.Mode != 'capturing':
                raise RuntimeError(f"The meter at {self._Session.Resource} is {self.Mode}, "
                                   f"not capturing, so it was not armed.")
            stamp = time.perf_counter_ns()
            inst.Write("INIT")
            return stamp

        return self._Session.Transaction(arm)

    def Fetch(self):
        """The armed capture's readings once it is complete, volts at the meter."""

        def fetch(inst):
            return inst.Query("FETC?"), inst.Errors()

        reply, errors = self._Session.Transaction(fetch, timeout=READ_TIMEOUT_S)
        self._Refuse_On(errors, "reported errors on a capture")
        values = Readings(reply, self._Session.Resource, 'a capture')
        self.Voltage = float(values[-1])
        return values

    def Fetch_When_Done(self, count, timeout_s):
        """:meth:`Fetch`, once the meter holds ``count`` readings: a FETC?
        sent sooner would wait inside the bus for the rest, and a capture
        longer than the VISA timeout would end in one."""
        deadline = time.monotonic() + timeout_s
        while True:
            taken = int(float(self._Session.Transaction(lambda inst: inst.Query("DATA:POIN?"))))
            if taken >= count:
                return self.Fetch()
            if time.monotonic() > deadline:
                raise Scpi_Error(f"The meter at {self._Session.Resource} took {taken} of "
                                 f"{count} readings in {timeout_s:.0f} s.")
            time.sleep(FETCH_POLL_S)

    def Read_Available(self, most):
        """Up to ``most`` readings of a streamed capture, read out and erased
        from the meter's memory; none if it has taken none since the last."""

        def read(inst):
            if int(float(inst.Query("DATA:POIN?"))) == 0:
                return None, inst.Errors()
            return inst.Query(f"R? {most}"), inst.Errors()

        reply, errors = self._Session.Transaction(read, timeout=READ_TIMEOUT_S)
        self._Refuse_On(errors, "reported errors on a streamed capture")
        if reply is None:
            return np.zeros(0)
        values = Readings(reply, self._Session.Resource, 'a streamed capture')
        if len(values):
            self.Voltage = float(values[-1])
        return values

    def Autocal(self):
        """The meter's own ADC calibration (CAL:ADC?), which takes out the
        drift since its last; only while capturing."""

        def autocal(inst):
            if self.Mode != 'capturing':
                raise RuntimeError(f"The meter at {self._Session.Resource} is {self.Mode}, "
                                   f"not capturing, so it was not autocalibrated.")
            return inst.Query_Slow("CAL:ADC?", AUTOCAL_TIMEOUT_S), inst.Errors()

        reply, errors = self._Session.Transaction(autocal, timeout=AUTOCAL_TIMEOUT_S + 5.0)
        self._Refuse_On(errors, "reported errors on its autocalibration")
        if int(float(reply)) != 0:
            raise Scpi_Error(f"The meter at {self._Session.Resource} failed its "
                             f"autocalibration (CAL:ADC? answered {reply}).")

    def Release(self):
        """Back to the live poll from any mode, set up afresh; safe to call
        more than once."""
        with self._Mode_Lock:
            if self.Mode == 'continuous':
                return
            self.Mode = 'idle'

        def setup(inst):
            inst.Write("ABOR")
            self._Setup(inst)

        self._Session.Transaction(setup)
        self._Move('continuous')

    def Close(self):
        """Stop the poll, leave the meter reset, and let the session go."""
        self._Polling = False
        self._Thread.join(timeout=1.0)

        def reset(inst):
            for command in ("ABOR", "*RST", "*CLS"):
                inst.Write(command)

        try:
            self._Session.Transaction(reset, timeout=2.0)
        except Exception as e:
            log.warning("The meter at %s was not reset as it closed: %s",
                        self._Session.Resource, e)
        self._Session.Close()


class Cooler_Meter(_Meter):

    def __init__(self, session):
        super().__init__(session, COOLER_CONFIG)


class Line_Meter(_Meter):

    _LEGAL = {'idle': ('continuous', 'stepping', 'capturing'),
              'continuous': ('idle', 'stepping', 'capturing'),
              'stepping': ('idle',),
              'capturing': ('idle', 'capturing')}

    def __init__(self, session):
        super().__init__(session, LINE_CONFIG)

    def Step_Config(self):
        """Single readings on command: immediate trigger, one sample, no
        delay, so a reading's stamp is when it was taken. Every setting is
        read back and the error queue drained before the first reading."""
        self._Move('stepping')

        def config(inst):
            inst.Clear()
            inst.Write("*CLS")
            for command in ("TRIG:SOUR IMM", "TRIG:COUN 1", "SAMP:COUN 1", "TRIG:DEL 0"):
                inst.Write(command)
            inst.Query("*OPC?")
            source = inst.Query("TRIG:SOUR?").upper()
            count = int(float(inst.Query("TRIG:COUN?")))
            samples = int(float(inst.Query("SAMP:COUN?")))
            self._Refuse_On(inst.Errors(), "rejected the stepping configuration")
            if not source.startswith('IMM') or count != 1 or samples != 1:
                raise Scpi_Error(f"The line meter did not take the stepping configuration: "
                                 f"TRIG:SOUR reads {source}, TRIG:COUN {count} and SAMP:COUN "
                                 f"{samples}, where IMM, 1 and 1 were sent.")

        self._Session.Transaction(config)

    def Read_Settled(self, wait_s):
        """Wait ``wait_s``, then take one reading: ``(stamp_ns, volts)``, the
        stamp from perf_counter_ns just before READ? goes out.

        READ? is a synchronous query, so a reading the caller discards leaves
        nothing behind in the instrument.
        """

        def read(inst):
            if self.Mode != 'stepping':
                raise Calibration_Error(f"The line meter is {self.Mode}, not stepping, so it "
                                        f"will not take a calibration reading.")
            time.sleep(wait_s)
            stamp = time.perf_counter_ns()
            reply = inst.Query("READ?")
            return stamp, reply, inst.Errors()

        stamp, reply, errors = self._Session.Transaction(read, timeout=wait_s + READ_TIMEOUT_S)
        self._Refuse_On(errors, "reported errors on a calibration reading")
        values = Readings(reply, self._Session.Resource, 'a calibration reading')
        if len(values) != 1:
            raise Calibration_Error(f"READ? returned {len(values)} values where one was "
                                    f"asked for: the meter is not taking single readings on "
                                    f"command.")
        return stamp, float(values[0])
