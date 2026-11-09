"""The simulated rack: a software bench the whole DAQ runs on without hardware.

Date:    2026-11-09
Version: 0.4.2
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Only the instruments are simulated. The acquisition loop, the DAC
routing and its code rounding, the meter drivers and their VISA
sessions, the calibration and its gate all run unmodified on top.

The rack it imitates is the one the bench tool (cls-cal) measured on
24 and 25 October 2026, so a hardware calibration here finds what it
found there:

* the platform answers a step with a 0.4 ms rise, then overshoots by
  0.18 % of the jump about 6.5 ms later and settles back, inside 0.5 V
  of a 1 kV jump in 17 ms and of a 2 kV jump in 20 ms;
* the two converters drive the amplifier's two inputs, and the platform
  sits where the 1 V-step sweep across +-3 kV found it, readback less
  setpoint on knots 50 V apart: each converter's gain and integral
  non-linearity, and the amplifier's offset of about -1 V, which is
  where 0 V sits whichever converter was driven last;
* the line meter scatters by 0.065 V at the platform at a 200 us
  aperture, less as the integration lengthens;
* the cooler sits at 29884.8 V with the mains ripple and harmonics the
  cooler monitor recorded, a slow drift, and 0.17 V of noise.

The meters answer the SCPI the drivers send, mark a reading beyond their
range as 9.9E37, and report anything else through SYST:ERR?. They also
sample timed captures (INIT, then FETC? or R?) on a thread of their own,
so a capture catches whatever the platform does meanwhile; back-to-back
readings take READ_TIME_S, not their integration time, to keep tests
quick. The PulseBlaster's period clocks a tagger that sees three
resonances on a flat background, the ions in one time-of-flight bunch,
timed in the card's 500 ps bins.

The converters live in the acquisition process and the line meter in
the GUI's, so the platform's state is shared memory (:class:`Platform`),
made by the GUI and handed to the loop when it starts.

Depends on: numpy; cls_acquisition.calibration, .keysight, .mcc_dac,
.timetagger and cls_acquisition.triggers.
"""

import inspect
import queue
import threading
import time

import numpy as np

from cls_acquisition import triggers
from cls_acquisition.calibration import DIVIDER
from cls_acquisition.hardware.keysight import COOLER_DIVIDER, OVERLOAD, RANGE_V
from cls_acquisition.hardware.mcc_dac import AMPLIFIER_GAIN, CH_NEG, CH_POS
from cls_acquisition.hardware.timetagger import BIN_US, MARK, Check_Window

# the step response, fitted to the bench's 1 kV and 2 kV steps
TAU_RISE_S = 0.00018
TAIL_FRACTION, TAU_TAIL_S = 0.03, 0.00093
OVERSHOOT, T_OVERSHOOT_S = 0.00176, 0.0065

# readback - setpoint, from the bench's 1 V-step sweep of 25 October 2026
# (cls-cal 20261025_021414), averaged over 50 V about each knot: each
# converter's gain and non-linearity, and the amplifier's offset, in one
# curve; past +-3 kV it runs on along each end's slope
ERROR_KNOTS_V = np.arange(-3000.0, 3001.0, 50.0)
ERROR_V = (
    1.254, 1.329, 1.438, 1.432, 1.529, 1.588, 1.264, 1.094, 1.253, 1.327,
    1.307, 1.443, 1.375, 0.965, 0.929, 1.034, 1.026, 1.104, 1.237, 1.026,
    0.741, 0.819, 0.922, 0.945, 1.074, 1.084, 0.688, 0.570, 0.691, 0.708,
    0.700, 0.805, 0.628, 0.233, 0.269, 0.363, 0.329, 0.440, 0.504, 0.165,
    -0.046, 0.057, 0.126, 0.111, 0.245, 0.199, -0.252, -0.285, -0.178, -0.200,
    -0.141, -0.089, -0.351, -0.656, -0.567, -0.493, -0.506, -0.350, -0.324, -0.956,
    -1.031, -1.381, -1.450, -1.387, -1.441, -1.531, -1.465, -1.402, -1.588, -1.531,
    -1.524, -1.674, -1.665, -1.539, -1.608, -1.687, -1.599, -1.639, -1.800, -1.778,
    -1.721, -1.923, -1.904, -1.882, -2.033, -2.101, -2.032, -2.081, -2.252, -2.202,
    -2.247, -2.474, -2.503, -2.455, -2.650, -2.634, -2.606, -2.680, -2.841, -2.833,
    -2.871, -3.065, -2.976, -2.923, -3.106, -3.136, -3.087, -3.226, -3.311, -3.260,
    -3.322, -3.442, -3.337, -3.237, -3.395, -3.384, -3.356, -3.514, -3.587, -3.530,
    -3.522,
)
END_SLOPES = (-0.000490, -0.000613)     # d(readback - set)/d(set) past each end

# per reading at a 200 us aperture, in volts at each meter's input
NOISE_APERTURE_S = 0.0002
LINE_NOISE_V = 0.065 / DIVIDER
COOLER_NOISE_V = 0.17 / COOLER_DIVIDER
READ_TIME_S = 0.002                 # one back-to-back reading and the bus
STREAM_MAX = 1000000                # readings the meter takes in one streamed record

COOLER_V = 29884.8
COOLER_DRIFT_V_S = 0.0058 / 60
COOLER_TONES = ((50.0, 0.438), (100.0, 0.014), (150.0, 0.038), (200.0, 0.023))  # (Hz, V)

# every simulated time, in both processes: perf_counter is system-wide on
# Windows and macOS alike, and fine enough for a 0.4 ms rise and a 50 Hz
# ripple, where monotonic ticked every 15.6 ms on Windows before Python 3.13
CLOCK = time.perf_counter

RESONANCES = ((32.0, 1.0), (47.0, 0.6), (71.0, 0.35))  # (platform volts, height)
FWHM_V = 4.0
PEAK_RATE_HZ = 60.0                 # counts per second at the top of the strongest
BACKGROUND_RATE_HZ = 5.0
TOF_US, TOF_WIDTH_US = 52.0, 1.5


def Step_Remaining(t):
    """The share of a step still to come ``t`` seconds after it was
    commanded: 1 at the command, 0 once settled, below 0 while it
    overshoots."""
    t = max(t, 0.0)
    x = t / T_OVERSHOOT_S
    return ((1.0 - TAIL_FRACTION) * np.exp(-t / TAU_RISE_S)
            + TAIL_FRACTION * np.exp(-t / TAU_TAIL_S)
            - OVERSHOOT * x * x * np.exp(2.0 * (1.0 - x)))


class Platform():
    """The platform voltage, shared between processes: where it was, where
    it is going, and since when (CLOCK, the same in both)."""

    def __init__(self, context):
        self._State = context.Array('d', [0.0, 0.0, CLOCK()])

    def Set(self, volts):
        with self._State.get_lock():
            now = CLOCK()
            self._State[:] = [self._At(now), volts, now]

    def Volts(self, t=None):
        """At CLOCK time ``t``, or now."""
        with self._State.get_lock():
            return self._At(CLOCK() if t is None else t)

    def _At(self, now):
        start, target, since = self._State[:]
        return target + (start - target) * Step_Remaining(now - since)


def Platform_Error(setpoint):
    """Where the measured rack puts the platform, less ``setpoint``."""
    low, high = ERROR_KNOTS_V[0], ERROR_KNOTS_V[-1]
    if setpoint < low:
        return ERROR_V[0] + END_SLOPES[0] * (setpoint - low)
    if setpoint > high:
        return ERROR_V[-1] + END_SLOPES[1] * (setpoint - high)
    return float(np.interp(setpoint, ERROR_KNOTS_V, ERROR_V))


class Converters():
    """The converter pair behind the DAC driver, on the amplifier's two
    inputs: the platform follows their difference, so with both at 0 V it
    sits at the amplifier's offset whichever was driven last."""

    def __init__(self, platform):
        self._Platform = platform
        self._Out = [0.0, 0.0]

    def V_Out(self, channel, volts):
        self._Out[channel] = volts
        setpoint = AMPLIFIER_GAIN * (self._Out[CH_POS] - self._Out[CH_NEG])
        self._Platform.Set(setpoint + Platform_Error(setpoint))


class Keysight():
    """A 3446xA as a VISA session sees it. ``source()`` is the voltage at its
    input, or ``source(t)`` at CLOCK time ``t`` if it takes one, as a timed
    capture reads it; ``noise_v`` its scatter per reading at a 200 us
    aperture."""

    DEFAULTS = {'TRIG:SOUR': 'IMM', 'TRIG:COUN': '1', 'SAMP:COUN': '1', 'SAMP:SOUR': 'IMM',
                'SAMP:TIM': '0.001', 'VOLT:APER:ENAB': 'OFF', 'VOLT:DC:APER': '0.0002',
                'VOLT:DC:NPLC': '10', 'VOLT:ZERO:AUTO': 'ON', 'FORM:DATA': 'ASC'}

    def __init__(self, source, noise_v=0.0):
        self._Source, self._Noise_V = source, noise_v
        self._Timed = bool(inspect.signature(source).parameters)
        self._Rng = np.random.default_rng()
        self._Settings = dict(self.DEFAULTS)
        self._Errors = []
        self._Block, self._Taken, self._Read = None, 0, 0
        self.timeout = 5000

    def _Integration_S(self):
        if self._Settings['VOLT:APER:ENAB'] == 'ON':
            return float(self._Settings['VOLT:DC:APER'])
        return float(self._Settings['VOLT:DC:NPLC']) / 50.0

    def _Reading(self, t=None):
        value = self._Source(t) if t is not None and self._Timed else self._Source()
        if self._Noise_V:
            value += self._Rng.normal(0.0, self._Noise_V * np.sqrt(
                NOISE_APERTURE_S / self._Integration_S()))
        return OVERLOAD if abs(value) > 1.2 * RANGE_V else value

    def write(self, command):
        head, _, value = command.strip().partition(' ')
        head = head.upper()
        if head in self._Settings:
            self._Settings[head] = value.upper()
        elif head == '*RST':
            self._Settings = dict(self.DEFAULTS)
            self._Block = None
        elif head == '*CLS':
            self._Errors = []
        elif head == 'ABOR':
            self._Block = None
        elif head == 'INIT':
            self._Block, self._Read = [], 0
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
            return ','.join(f"{self._Reading():+.9E}" for _ in range(count)) + '\n'
        if head == 'FETC?':
            count = self._Count('SAMP:COUN')
            while self._Block is not None and len(self._Block) < count:
                time.sleep(READ_TIME_S)
            return ','.join(f"{v:+.9E}" for v in self._Block or []) + '\n'
        if head.startswith('R?'):
            block = self._Block or []
            end = min(len(block), self._Read + int(head[2:] or len(block)))
            body = ','.join(f"{v:+.9E}" for v in block[self._Read:end])
            self._Read = end
            return f"#{len(str(len(body)))}{len(body)}{body}\n"
        if head == 'DATA:POIN?':
            return f"{len(self._Block or []) - self._Read:d}\n"
        if head == '*OPC?':
            return '1\n'
        if head == '*IDN?':
            return 'Keysight Technologies,34465A,SIMULATED,A.02.17-02.40-02.17-00.52-04-01\n'
        if head == '*OPT?':
            return '0\n'
        if head == 'CAL:ADC?':
            return '+0\n'
        if head == 'SYST:ERR?':
            return (self._Errors.pop(0) if self._Errors else '+0,"No error"') + '\n'
        if head == 'TRIG:SOUR?':
            return self._Settings['TRIG:SOUR'][:3] + '\n'
        if head in ('TRIG:COUN?', 'SAMP:COUN?'):
            return f"{float(self._Settings[head[:-1]]):+.8E}\n"
        if head == 'SAMP:COUN? MAX':
            return f"{float(STREAM_MAX):+.8E}\n"
        if head == 'SAMP:TIM?':
            return f"{float(self._Settings['SAMP:TIM']):+.8E}\n"
        if head == 'SAMP:TIM? MIN':
            return f"{self._Integration_S() + 2e-5:+.8E}\n"
        self._Errors.append('-113,"Undefined header"')
        return '0\n'

    def _Sample(self, block):
        """One capture, on the meter's sample timer or back to back, until its
        count is taken or it is aborted."""
        timed = self._Settings['SAMP:SOUR'].startswith('TIM')
        interval = float(self._Settings['SAMP:TIM']) if timed else READ_TIME_S
        start = time.perf_counter()
        for i in range(self._Count('SAMP:COUN')):
            due = start + i * interval
            time.sleep(max(due - time.perf_counter(), 0.0))
            if self._Block is not block:
                return
            # read at the instant the meter's timer sets, however late this thread
            # woke: the analyses take a record as evenly spaced, as the meter's is
            block.append(self._Reading(due))

    def clear(self):
        pass

    def close(self):
        pass


def Line_Meter_Source(platform):
    return lambda t=None: platform.Volts(t) / DIVIDER


def Cooler_Meter_Source():
    start = CLOCK()

    def cooler(t=None):
        t = CLOCK() if t is None else t
        volts = COOLER_V + COOLER_DRIFT_V_S * (t - start)
        volts += sum(a * np.sin(2.0 * np.pi * f * t) for f, a in COOLER_TONES)
        return volts / COOLER_DIVIDER

    return cooler


class Tagger():
    """The tagger as the loop sees it: one packet per trigger edge, its mark
    and its hits, and a read hands over every packet ready, as the driver
    does, so a loop that falls behind gets several triggers in one."""

    def __init__(self, platform):
        self._Platform = platform
        self._Rng = np.random.default_rng()
        self._Packets = queue.Queue()
        self._Window = (0.0, 5000.0)
        self._Reading = False
        self._Bunch = 0
        self.Gated = False

    def Configure(self, start_us, stop_us):
        Check_Window(start_us, stop_us)
        self._Window = (start_us, stop_us)

    def Start(self):
        self._Packets = queue.Queue()
        self._Bunch = 0
        self._Reading = True

    def Stop(self):
        self._Reading = False

    def Read(self, timeout):
        try:
            packets = [self._Packets.get(timeout=timeout)]
        except queue.Empty:
            return None
        while True:
            try:
                packets.append(self._Packets.get_nowait())
            except queue.Empty:
                return np.vstack(packets)

    def Drain(self):
        dropped = 0
        while (rows := self.Read(0.0)) is not None:
            dropped += int(np.count_nonzero(rows[:, 1] == MARK))
        return dropped

    def Edge(self, period_s):
        """One trigger: the ions and background of one period, the ions
        timed in their bunch and the background spread over the window, each
        in the card's bins as the driver hands real hits on. A gated window
        never reaches the tagger, so it is not counted."""
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
        times = np.rint(times / BIN_US) * BIN_US
        times = times[(times >= start) & (times < stop)]
        hits = np.column_stack([np.full(len(times), float(self._Bunch)),
                                rng.integers(1, 5, len(times)), times])
        self._Packets.put(np.vstack([[self._Bunch, MARK, 0.0], hits]))


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
        next_edge = CLOCK()
        while self._Running:
            next_edge += self._Period_S
            time.sleep(max(next_edge - CLOCK(), 0.0))
            if self._Running:
                self._Tagger.Edge(self._Period_S)

    def Stop(self):
        self._Running = False
        if self._Thread is not None:
            self._Thread.join(timeout=1.0)
            self._Thread = None

    def Close(self):
        self.Stop()
