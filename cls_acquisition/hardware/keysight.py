"""The two Keysight 34465A meters: the line meter on the HV divider, and the cooler's.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Both are polled on their owner thread for the display, and both are a
small state machine, so an illegal move is refused rather than half
done: 'continuous' while polled, 'capturing' while a hardware
calibration takes a timed block of readings, and 'idle' between. The
line meter also has 'stepping', for a calibration's single readings on
command. Neither stepping nor capturing goes straight back to
continuous; the way back is :meth:`_Meter.Release`, through idle, from
any mode and as often as it is called, and it sets the meter up afresh.
The poll reads only in 'continuous' and checks that on the owner thread,
so nothing can land between the check and its READ?.

A capture is armed (INIT) and fetched (FETC?) in two transactions, so
the caller can change the voltage while the meter is sampling: that is
how a step response is caught.

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
COOLER_DIVIDER = 10000.0            # cooler platform volts per volt at its meter

POLL_PERIOD_S = 0.05
POLL_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 10.0               # one calibration reading, beyond its wait
MAX_SAMPLES = 50000                 # the 34465A's reading memory without the MEM option

log = logging.getLogger('oracle.meters')


class _Meter():
    """A meter set up once and then polled for the display, until it is
    asked for a capture."""

    _LEGAL = {'idle': ('continuous', 'capturing'),
              'continuous': ('idle', 'capturing'),
              'capturing': ('idle',)}

    def __init__(self, session, config):
        self.Voltage = float('nan')
        self.Mode = 'continuous'
        self._Mode_Lock = threading.Lock()
        self._Session = session
        self._Config = config
        self._Polling = True
        self._Failing = False
        try:
            session.Transaction(self._Setup)
        except Exception:
            session.Close()
            raise
        self._Thread = threading.Thread(target=self._Poll_Loop, daemon=True,
                                        name=f"poll:{session.Resource[-12:]}")
        self._Thread.start()

    def _Setup(self, inst):
        inst.Clear()
        inst.Write("*CLS")
        for command in self._Config:
            inst.Write(command)
        for command in ("SAMP:SOUR IMM", "TRIG:SOUR IMM", "TRIG:COUN 1"):
            inst.Write(command)
        inst.Query("*OPC?")
        self._Refuse_On(inst.Errors(), "rejected its configuration")

    def _Refuse_On(self, errors, what):
        """Raise Scpi_Error naming the meter if its error queue held anything."""
        if errors:
            raise Scpi_Error(f"The meter at {self._Session.Resource} {what}: "
                             f"{'; '.join(errors)}.")

    def _Poll_Loop(self):

        def poll(inst):
            if self.Mode == 'continuous':
                self.Voltage = float(inst.Query("READ?"))

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

    def Capture(self, count, interval_s=None, aperture_s=None, nplc=None):
        """Set up blocks of ``count`` readings, one every ``interval_s`` (back
        to back if None), each integrating ``aperture_s`` or ``nplc`` power
        line cycles. Every setting is read back before the first block."""
        if not 1 <= count <= MAX_SAMPLES:
            raise ValueError(f"A capture of {count} readings does not fit the meter's "
                             f"memory of {MAX_SAMPLES}.")
        self._Move('capturing')
        integration = (("VOLT:APER:ENAB OFF", f"VOLT:DC:NPLC {nplc:g}") if nplc
                       else ("VOLT:APER:ENAB ON", f"VOLT:DC:APER {aperture_s:.6g}"))
        timing = (("SAMP:SOUR IMM",) if interval_s is None
                  else ("VOLT:ZERO:AUTO OFF", "SAMP:SOUR TIM", f"SAMP:TIM {interval_s:.6g}"))

        def config(inst):
            for command in (*integration, *timing, f"SAMP:COUN {count}", "TRIG:SOUR IMM",
                            "TRIG:COUN 1", "TRIG:DEL 0"):
                inst.Write(command)
            inst.Query("*OPC?")
            samples = int(float(inst.Query("SAMP:COUN?")))
            self._Refuse_On(inst.Errors(), "rejected the capture configuration")
            if samples != count:
                raise Scpi_Error(f"The meter at {self._Session.Resource} reads SAMP:COUN "
                                 f"{samples} where {count} was sent.")

        self._Session.Transaction(config)

    def Arm(self):
        """Start one block; ``perf_counter_ns`` just before INIT goes out."""

        def arm(inst):
            if self.Mode != 'capturing':
                raise RuntimeError(f"The meter at {self._Session.Resource} is {self.Mode}, "
                                   f"not capturing, so it was not armed.")
            stamp = time.perf_counter_ns()
            inst.Write("INIT")
            return stamp

        return self._Session.Transaction(arm)

    def Fetch(self):
        """The armed block's readings once it is complete, volts at the meter."""

        def fetch(inst):
            return inst.Query("FETC?"), inst.Errors()

        reply, errors = self._Session.Transaction(fetch, timeout=READ_TIMEOUT_S)
        self._Refuse_On(errors, "reported errors on a capture")
        values = np.array([float(v) for v in reply.split(',')])
        self.Voltage = float(values[-1])
        return values

    def Release(self):
        """Back to the live poll from any mode, set up afresh; safe to call
        more than once."""
        with self._Mode_Lock:
            if self.Mode == 'continuous':
                return
            self.Mode = 'idle'
        self._Session.Transaction(self._Setup)
        self._Move('continuous')

    def Close(self):
        self._Polling = False
        self._Thread.join(timeout=1.0)
        self._Session.Close()


class Cooler_Meter(_Meter):

    def __init__(self, session):
        super().__init__(session, COOLER_CONFIG)


class Line_Meter(_Meter):

    _LEGAL = {'idle': ('continuous', 'stepping', 'capturing'),
              'continuous': ('idle', 'stepping', 'capturing'),
              'stepping': ('idle',),
              'capturing': ('idle',)}

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
        values = reply.split(',')
        if len(values) != 1:
            raise Calibration_Error(f"READ? returned {len(values)} values where one was "
                                    f"asked for: the meter is not taking single readings on "
                                    f"command.")
        return stamp, float(values[0])
