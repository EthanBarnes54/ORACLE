"""The simulated rack: a software bench the whole DAQ runs on without hardware.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Only the instruments are simulated. The acquisition loop, the DAC
routing, the meter drivers and their VISA sessions, the calibration and
its gate all run unmodified on top. The converters drive a platform that
settles with a first-order time constant and carries the converters'
seam at 0 V (the 0.437 V step the campaigns measured), so the gate's
polarity check sees what the rack's does. The meters answer the SCPI
the drivers send and report anything else through SYST:ERR?. The
meters also sample timed blocks (INIT, then FETC?) on a thread of their
own, so a block catches whatever the platform does meanwhile. The
PulseBlaster's period clocks a tagger that sees three resonances on a
flat background, the ions in one time-of-flight bunch.

The converters live in the acquisition process and the line meter in
the GUI's, so the platform's state is shared memory (:class:`Platform`),
made by the GUI and handed to the loop when it starts.

Depends on: numpy; cls_acquisition.calibration, .keysight, .mcc_dac and
cls_acquisition.triggers.
"""

import queue
import threading
import time

import numpy as np

from cls_acquisition import triggers
from cls_acquisition.calibration import DIVIDER
from cls_acquisition.hardware.keysight import COOLER_DIVIDER
from cls_acquisition.hardware.mcc_dac import CH_POS

# illustrative values: the rack's settling has never been measured
SETTLE_TAU_S = 0.003
BRANCH_GAIN = (1000.55, 999.45)     # platform volts per converter volt, by channel
BRANCH_OFFSET = (0.2185, -0.2185)   # each amplifier's offset: the seam at 0 V
NOISE_V = 0.175                     # platform volts per reading, the rack's scatter
COOLER_V = 29950.0
READ_TIME_S = 0.002                 # one aperture and the bus

RESONANCES = ((32.0, 1.0), (47.0, 0.6), (71.0, 0.35))  # (platform volts, height)
FWHM_V = 4.0
PEAK_RATE_HZ = 60.0                 # counts per second at the top of the strongest
BACKGROUND_RATE_HZ = 5.0
TOF_US, TOF_WIDTH_US = 52.0, 1.5


class Platform():
    """The platform voltage, shared between processes: where it was, where
    it is going, and since when (time.monotonic, the same in both)."""

    def __init__(self, context):
        self._State = context.Array('d', [0.0, 0.0, time.monotonic()])

    def Set(self, volts):
        with self._State.get_lock():
            now = time.monotonic()
            self._State[:] = [self._At(now), volts, now]

    def Volts(self):
        with self._State.get_lock():
            return self._At(time.monotonic())

    def _At(self, now):
        start, target, since = self._State[:]
        return target + (start - target) * np.exp(-(now - since) / SETTLE_TAU_S)


class Converters():
    """The converter pair behind the DAC driver. A converter driven away from
    zero takes the platform; at 0 V it stays on the last amplifier driven,
    offset and all."""

    def __init__(self, platform):
        self._Platform = platform
        self._Out = [0.0, 0.0]
        self._Live = CH_POS

    def V_Out(self, channel, volts):
        self._Out[channel] = volts
        if volts != 0.0:
            self._Live = channel
        sign = 1.0 if self._Live == CH_POS else -1.0
        self._Platform.Set(sign * BRANCH_GAIN[self._Live] * self._Out[self._Live]
                           + BRANCH_OFFSET[self._Live])


class Keysight():
    """A 3446xA as a VISA session sees it."""

    def __init__(self, source):
        self._Source = source
        self._Settings = {'TRIG:SOUR': 'IMM', 'TRIG:COUN': '1', 'SAMP:COUN': '1',
                          'SAMP:SOUR': 'IMM', 'SAMP:TIM': '0.001'}
        self._Errors = []
        self._Block = None

    def write(self, command):
        head, _, value = command.strip().partition(' ')
        head = head.upper()
        if head in self._Settings:
            self._Settings[head] = value.upper()
        elif head == '*CLS':
            self._Errors = []
        elif head == 'INIT':
            self._Block = []
            threading.Thread(target=self._Sample, args=(self._Block,), daemon=True,
                             name='sim-capture').start()
        elif not head.startswith(('CONF', 'VOLT', 'CALC', 'TRIG:DEL')):
            self._Errors.append('-113,"Undefined header"')

    def _Count(self, key):
        return int(float(self._Settings[key]))

    def query(self, command):
        head = command.strip().upper()
        if head == 'READ?':
            time.sleep(READ_TIME_S)
            count = self._Count('TRIG:COUN') * self._Count('SAMP:COUN')
            return ','.join(f"{self._Source():+.9E}" for _ in range(count)) + '\n'
        if head == 'FETC?':
            count = self._Count('SAMP:COUN')
            while self._Block is not None and len(self._Block) < count:
                time.sleep(READ_TIME_S)
            return ','.join(f"{v:+.9E}" for v in self._Block or []) + '\n'
        if head == '*OPC?':
            return '1\n'
        if head == 'SYST:ERR?':
            return (self._Errors.pop(0) if self._Errors else '+0,"No error"') + '\n'
        if head == 'TRIG:SOUR?':
            return self._Settings['TRIG:SOUR'][:3] + '\n'
        if head in ('TRIG:COUN?', 'SAMP:COUN?'):
            return f"{float(self._Settings[head[:-1]]):+.8E}\n"
        self._Errors.append('-113,"Undefined header"')
        return '0\n'

    def _Sample(self, block):
        """One block of readings, on the meter's sample timer or back to back."""
        timed = self._Settings['SAMP:SOUR'].startswith('TIM')
        interval = float(self._Settings['SAMP:TIM']) if timed else READ_TIME_S
        start = time.perf_counter()
        for i in range(self._Count('SAMP:COUN')):
            time.sleep(max(start + i * interval - time.perf_counter(), 0.0))
            block.append(self._Source())

    def clear(self):
        pass

    def close(self):
        pass


def Line_Meter_Source(platform):
    rng = np.random.default_rng()
    return lambda: (platform.Volts() + rng.normal(0.0, NOISE_V)) / DIVIDER


def Cooler_Meter_Source():
    rng = np.random.default_rng()
    return lambda: (COOLER_V + rng.normal(0.0, 0.5)) / COOLER_DIVIDER


class Tagger():
    """The tagger as the loop sees it, one packet per trigger edge."""

    def __init__(self, platform):
        self._Platform = platform
        self._Rng = np.random.default_rng()
        self._Packets = queue.Queue()
        self._Window = (0.0, 5000.0)
        self._Reading = False
        self._Bunch = 0
        self.Gated = False

    def Configure(self, start_us, stop_us):
        self._Window = (start_us, stop_us)

    def Start(self):
        self._Packets = queue.Queue()
        self._Bunch = 0
        self._Reading = True

    def Stop(self):
        self._Reading = False

    def Read(self, timeout):
        try:
            return self._Packets.get(timeout=timeout)
        except queue.Empty:
            return None

    def Drain(self):
        dropped = 0
        while True:
            try:
                self._Packets.get_nowait()
            except queue.Empty:
                return dropped
            dropped += 1

    def Edge(self, period_s):
        """One trigger: the ions and background of one period, the ions
        timed in their bunch and the background spread over the window. A
        gated window never reaches the tagger, so it is not counted."""
        if not self._Reading or self.Gated:
            return
        self._Bunch += 1
        rng, (start, stop) = self._Rng, self._Window
        volts = self._Platform.Volts()
        shape = sum(height / (1.0 + ((volts - centre) / (0.5 * FWHM_V)) ** 2)
                    for centre, height in RESONANCES)
        times = np.concatenate([
            rng.normal(TOF_US, TOF_WIDTH_US, rng.poisson(PEAK_RATE_HZ * shape * period_s)),
            rng.uniform(start, stop, rng.poisson(BACKGROUND_RATE_HZ * period_s))])
        times = times[(times >= start) & (times < stop)]
        self._Packets.put(np.column_stack([np.full(len(times), float(self._Bunch)),
                                           rng.integers(1, 5, len(times)), times]))


class Window_Gate():
    """The AND unit on the tagger's window."""

    def __init__(self, tagger):
        self._Tagger = tagger

    def Open(self):
        self._Tagger.Gated = False

    def Close(self):
        self._Tagger.Gated = True


class Pulse_Generator():
    """The PulseBlaster's timing: one edge per pattern period, on a fixed grid
    as the crystal-timed original keeps it."""

    def __init__(self, tagger):
        self._Tagger = tagger
        self._Period_S = 0.1
        self._Running = False
        self._Thread = None

    def Program(self, pattern):
        self._Period_S = triggers.Period_Ns(pattern) / 1e9

    def Start(self):
        self.Stop()
        self._Running = True
        self._Thread = threading.Thread(target=self._Run, name='sim-pulses', daemon=True)
        self._Thread.start()

    def _Run(self):
        next_edge = time.monotonic()
        while self._Running:
            next_edge += self._Period_S
            time.sleep(max(next_edge - time.monotonic(), 0.0))
            if self._Running:
                self._Tagger.Edge(self._Period_S)

    def Stop(self):
        self._Running = False
        if self._Thread is not None:
            self._Thread.join(timeout=1.0)
            self._Thread = None

    def Close(self):
        self.Stop()
