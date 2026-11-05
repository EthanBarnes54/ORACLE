"""Step-and-confirm calibration, and the gate every calibration must pass.

Date:    2026-11-05
Version: 0.4.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Calibration pairs each commanded platform voltage with what the line
meter reads through the divider. The old DAQ let the meter sample on the
PulseBlaster's clock while the software stepped the DAC on the tagger's,
and lined the two up afterwards: when they slipped, setpoint i was paired
with reading i + k (Mode B), and a race could leave the burst ungated
(Mode A). Here there is one clock. For each setpoint the acquisition
process writes the DAC and confirms, the meter waits SETTLE and takes
exactly one reading on command, and only then does the next setpoint go
out. Reading i cannot be requested before write i is confirmed, so the
lag is zero by construction; the PulseBlaster and the tagger take no part.

The gate (:func:`Quality_Report`) tests properties every healthy
calibration has -- it tracks the command, spans the commanded range, was
read settled, matches the plan, and its last reading is what the platform
reads when held there -- rather than the signatures of faults already
seen, so a new fault is still refused if it breaks one of them. A REJECT
writes no file and starts no scan.

The last of those is the guard on the pairing itself. Once the last point
is read, the DAC stays where it is and the meter reads that setpoint again,
held, well past its settling: readings with no stepping, so with no pairing
to get wrong. Were reading i holding setpoint i - k for any reason, as the
old DAQ's Mode B did, the calibration's last reading would sit k steps
from them; neither tracking nor span can see a shifted ramp, and no lag is
estimated or undone here, only refused.

SETTLE is the DAQ's only definition of settled. An amplifier slower than
it cannot be detected by timing, and on a uniform ramp not in the
readings either: partial settling shifts every reading by the same
fraction of a step. Where the Settle time page has measured the
amplifier, its settle model says how long a jump of a given size takes
(:func:`Settle_For`), and a point whose jump needs longer than SETTLE
waits that long: the first, from where the platform stood, and the
first of each range after a gap are the largest. The wait and the stamp
test use the same number for each point, and the gate checks it again.

Depends on: numpy.
"""

import contextlib
import logging

import numpy as np

# Every number the calibration and its gate decide on.
THRESHOLDS = {
    # healthy archived calibrations track at |r| >= 0.94 and failed ones at
    # <= 0.16 (294 runs, I329/I311/I324)
    'MinTrackingR': 0.90,
    'SpanTolerance': 0.10,          # readback span within 10 % of commanded
    'MaxPlanDeviationV': 1e-6,      # commanded against planned
    'MaxPolarityStepV': 0.10,       # branch offset at 0 V before a WARN
    # the HV amplifier's settling after a write. The bench (cls-cal, 24 Oct
    # 2026) put a 2 kV jump inside 0.5 V 20.5 ms after a 3 ms lag, and at the
    # meter's own scatter by about 30 ms; a jump of the full 4 kV, about 42 ms
    'SettleS': 0.050,
    # one read and two re-requests: a real shortfall is gone after the first
    # re-request, which waits it out, so a third early read is a fault
    'MaxAttemptsPerPoint': 3,
    # early reads tolerated in one calibration. A constant clock offset would
    # otherwise re-request every point and quietly double the run time
    'MaxRetriesPerCalibration': 10,
    'RetryMarginS': 0.005,          # added to a re-request's measured shortfall
    # the last setpoint read again, held, after the calibration: four SETTLEs on,
    # ten readings, against which the calibration's own last reading may sit half
    # a step off (a lag of one is a whole step) or six times their scatter, if
    # more (t on 9 degrees of freedom: 2e-4 of healthy calibrations)
    'HeldWaitS': 0.2,
    'HeldReadings': 10,
    'MaxHeldSteps': 0.5,
    'HeldSigma': 6.0,
    # and they must agree with each other: the bench's readings scatter 0.065 V and
    # the archive's healthy ones 0.175 V about their line, but answers that trail
    # their requests read a held setpoint as a ramp, three steps rms
    'MaxHeldScatterV': 0.5,
}

DIVIDER = 1000.0                    # platform volts per volt at the line meter
SETTLE_MARGIN = 1.25                # on measured settling, for what one test cannot see

PASS, WARN, REJECT = 'PASS', 'WARN', 'REJECT'

log = logging.getLogger('oracle.calibration')


class Calibration_Error(RuntimeError):
    pass


class Calibration():
    """One reading per planned setpoint, in plan order: the step records the
    acquisition process confirmed, the meter's readings (volts at the meter),
    how long after its write each reading was taken and how long its jump
    needed to settle, how many setpoints had to be read again, and the last
    setpoint's readings held."""

    def __init__(self, steps, readings, stamps_ns, settle_ns, retried, dac, held):
        self.Indices = np.array([s['Index'] for s in steps], dtype=int)
        self.Commanded = np.array([s['Volts'] for s in steps], dtype=float)
        self.Readings = np.array(readings, dtype=float)
        self.Gaps_Ns = (np.array(stamps_ns, dtype=np.int64)
                        - np.array([s['T_Command_Ns'] for s in steps], dtype=np.int64))
        self.Settle_Ns = np.array(settle_ns, dtype=np.int64)
        self.Retried = retried
        self.Held = np.array(held, dtype=float)
        self.Dac_Status = dac['Dac_Status']
        self.Dac_Correction = dac['Dac_Correction']


def Settle_For(model, jump_v):
    """How long a jump of ``jump_v`` takes to settle: the model's settling at
    that size times SETTLE_MARGIN, interpolated, held at its first value
    below the smallest jump measured and extended along its last segment
    above the largest. With no model, SETTLE, the only number there is."""
    if not model or not model.get('points'):
        return THRESHOLDS['SettleS']
    jumps, settles = np.array(model['points'], dtype=float).T
    settle = float(np.interp(jump_v, jumps, settles))
    if jump_v > jumps[-1] and len(jumps) > 1:
        slope = (settles[-1] - settles[-2]) / (jumps[-1] - jumps[-2])
        settle = settles[-1] + max(slope, 0.0) * (jump_v - jumps[-1])
    return settle * SETTLE_MARGIN


def _Quietly(cleanup):
    """Run a cleanup while an error is already on its way out: a failure
    here is logged, never raised over the error that matters."""
    try:
        cleanup()
    except Exception:
        log.exception("Cleanup after a failed calibration also failed")


@contextlib.contextmanager
def Calibrating(link, meter, setup, setpoints, stop=None):
    """The DAC and ``meter`` held for a calibration of ``setpoints``: ``setup()``
    readies the meter, the acquisition process enters calibration, and the
    'Calibrate' reply is yielded. However the body ends, the acquisition
    process leaves calibration (by ``stop()`` if given) and the meter goes
    back to its live poll; a cleanup that fails after another failure is
    logged, never raised over it. A 'Calibrate' that fails is stopped too:
    one that timed out may still be carried out, and a stop changes
    nothing in a process that is not calibrating."""
    stop = stop or (lambda: link.Send('Stop_Calibration'))
    try:
        setup()
        try:
            yield link.Send('Calibrate', setpoints=np.asarray(setpoints, dtype=float).tolist())
        except BaseException:
            _Quietly(stop)
            raise
        stop()
    except BaseException:
        _Quietly(meter.Release)
        raise
    meter.Release()


def Run_Calibration(link, meter, plan, progress=None, settle_model=None):
    """Calibrate ``plan`` by step and confirm; returns a :class:`Calibration`.

    ``link`` is the acquisition process, which alone drives the DAC;
    ``meter`` is the line meter; ``progress(index, volts, reading)`` is
    called after each kept point. Every exit leaves the meter on its live
    poll and the acquisition process out of calibration. A point waits
    SETTLE, or as long as ``settle_model`` says its jump takes if that is
    longer: the first, from where the platform stood, and the first of a
    range after a gap usually are.

    The retry rule. A reading stamped less than its point's settling after
    its write was confirmed is inconclusive: it is discarded, and the
    reading (not the write, which would restart the settling) is requested
    again after the measured shortfall plus a margin. Waiting on this side's
    own clock from the confirmation makes that rare; it covers what the wait
    cannot see, since the two stamps come from two processes. A stamp before
    its write means the clocks disagree and aborts at once, as does any
    other fault. Each point is read at most MaxAttemptsPerPoint times and a
    calibration re-requests at most MaxRetriesPerCalibration readings, so
    the retry path terminates and adds at most that many waits of a point's
    settling + margin.
    """
    jumps = np.abs(np.diff(plan.Setpoints, prepend=link.State.Voltage.value))
    needs_s = [max(THRESHOLDS['SettleS'], Settle_For(settle_model, jump)) for jump in jumps]
    steps, readings, stamps, needs_ns = [], [], [], []
    retried = re_requests = 0

    with Calibrating(link, meter, meter.Step_Config, plan.Setpoints) as dac:
        for index in range(len(plan)):
            step = link.Send('Calibration_Step', index=index)
            volts = step['Volts']
            if step['Index'] != index:
                raise Calibration_Error(
                    f"The acquisition process confirmed setpoint {step['Index']} when "
                    f"setpoint {index} was sent, so the calibration stopped rather than "
                    f"pair a reading with the wrong voltage.")
            settle_s = wait_s = needs_s[index]
            settle_ns, attempt = round(settle_s * 1e9), 0
            while True:
                attempt += 1
                stamp, value = meter.Read_Settled(wait_s)
                gap = stamp - step['T_Command_Ns']
                if gap < 0:
                    raise Calibration_Error(
                        f"Setpoint {index} ({volts:.2f} V) was read {-gap / 1e6:.3f} ms "
                        f"before its write was confirmed, which only happens when the "
                        f"two processes' clocks disagree; no reading in this "
                        f"calibration can be trusted.")
                if gap >= settle_ns:
                    break
                if attempt >= THRESHOLDS['MaxAttemptsPerPoint']:
                    raise Calibration_Error(
                        f"Setpoint {index} ({volts:.2f} V) was read too early {attempt} "
                        f"times, the last {gap / 1e6:.1f} ms after its write against a "
                        f"settling time of {settle_s * 1e3:.0f} ms; the limit is "
                        f"{THRESHOLDS['MaxAttemptsPerPoint']} attempts per point.")
                if re_requests >= THRESHOLDS['MaxRetriesPerCalibration']:
                    raise Calibration_Error(
                        f"{re_requests} readings in this calibration came before the "
                        f"voltage had settled, the limit is "
                        f"{THRESHOLDS['MaxRetriesPerCalibration']}, so it stopped at "
                        f"setpoint {index} ({volts:.2f} V): the read timing is not "
                        f"working as designed.")
                re_requests += 1
                retried += attempt == 1
                wait_s = (settle_ns - gap) / 1e9 + THRESHOLDS['RetryMarginS']
            steps.append(step)
            readings.append(value)
            stamps.append(stamp)
            needs_ns.append(settle_ns)
            if progress is not None:
                progress(index, volts, value)
        # the DAC still holds the last setpoint, and the loop leaves it there
        held = [meter.Read_Settled(THRESHOLDS['HeldWaitS'] if i == 0 else 0.0)[1]
                for i in range(THRESHOLDS['HeldReadings'])]
    return Calibration(steps, readings, stamps, needs_ns, retried, dac, held)


def Line_Fit(x, y):
    """``(gradient, offset, residual RMS)`` of the straight line through ``y``
    against ``x``."""
    gradient, offset = np.polyfit(x, y, 1)
    residual = y - (gradient * x + offset)
    return float(gradient), float(offset), float(np.sqrt(np.mean(residual ** 2)))


def Step_At_Zero(commanded, readback):
    """Where a line through the positive branch meets 0 V less where the
    negative branch's does, or None if the scan does not put three points
    either side of zero. 0 V itself is on neither side: both converters sit
    at mid-code there. The platform runs on through 0 V (the bench, 24 and
    25 October 2026), so this is not a step in it but how differently each
    converter's gain and non-linearity bend its branch: what one straight
    line through the whole calibration cannot describe (Mode C)."""
    offsets = []
    for side in (commanded > 0, commanded < 0):
        if side.sum() < 3 or np.ptp(readback[side]) == 0:
            return None
        offsets.append(Line_Fit(commanded[side], readback[side])[1])
    return offsets[0] - offsets[1]


def Quality_Report(cal, plan):
    """Each check's verdict and message, and the run's verdict: the worst."""
    commanded = cal.Commanded
    readback = cal.Readings * DIVIDER
    checks = []

    if len(readback) != len(plan) or len(plan) < 3:
        checks.append(('length', REJECT, f"{len(readback)} readings for a {len(plan)} "
                                         f"point plan."))
        return _Report(checks)
    checks.append(('length', PASS, f"{len(readback)} readings, one per setpoint"))

    moving = bool(np.ptp(readback) > 0)     # exactly 0 for a dead readback, as std is not
    tracking_r = float(abs(np.corrcoef(commanded, readback)[0, 1])) if moving else 0.0
    if tracking_r >= THRESHOLDS['MinTrackingR']:
        checks.append(('tracking', PASS, f"|r| = {tracking_r:.5f}"))
    else:
        checks.append(('tracking', REJECT,
                       f"The readback does not track the commanded voltage "
                       f"(|r| = {tracking_r:.3f}): the meter is measuring something other "
                       f"than the divider output."))

    commanded_span, readback_span = float(np.ptp(commanded)), float(np.ptp(readback))
    if abs(readback_span - commanded_span) > THRESHOLDS['SpanTolerance'] * commanded_span:
        checks.append(('span', REJECT, f"The readback spans {readback_span:.2f} V against "
                                       f"{commanded_span:.2f} V commanded."))
    else:
        checks.append(('span', PASS, f"{readback_span:.2f} V against {commanded_span:.2f} V "
                                     f"commanded"))

    # the divider's gradient: commanded volts per volt of readback
    gradient, _, residual_rms = Line_Fit(readback, commanded) if moving else (0.0, 0.0, 0.0)

    step = Step_At_Zero(commanded, readback)
    if step is None:
        checks.append(('polarity', PASS, "the scan does not cross zero"))
    elif abs(step) > THRESHOLDS['MaxPolarityStepV']:
        checks.append(('polarity', WARN,
                       f"The positive and negative branches differ by {step:+.3f} V at zero; "
                       f"fit each polarity separately."))
    else:
        checks.append(('polarity', PASS, f"the branches agree to {step:+.3f} V"))

    # the stamp test again, on the arrays that will be written
    shortfall = cal.Settle_Ns - cal.Gaps_Ns
    early = int(np.sum(shortfall > 0))
    shortest_ms = float(cal.Gaps_Ns.min()) / 1e6
    if early:
        worst = int(np.argmax(shortfall))
        checks.append(('settle', REJECT,
                       f"{early} readings were taken before their jump had settled, the worst "
                       f"at {commanded[worst]:.2f} V {cal.Gaps_Ns[worst] / 1e6:.1f} ms after "
                       f"its write against the {cal.Settle_Ns[worst] / 1e6:.0f} ms it needs, "
                       f"so the voltage may not have settled."))
    elif cal.Retried:
        checks.append(('settle', WARN,
                       f"{cal.Retried} setpoints were read early and read again; every kept "
                       f"reading is at least {shortest_ms:.1f} ms after its write."))
    else:
        checks.append(('settle', PASS, f"every reading at least {shortest_ms:.1f} ms after "
                                       f"its write"))

    # the last reading against the same setpoint read held: a lag of k sets it k steps off
    held = cal.Held * DIVIDER
    scatter = float(np.std(held, ddof=1))
    apart = float(readback[-1] - np.mean(held))
    allowed = max(THRESHOLDS['MaxHeldSteps'] * plan.Step,
                  THRESHOLDS['HeldSigma'] * scatter * np.sqrt(1.0 + 1.0 / len(held)))
    if scatter > THRESHOLDS['MaxHeldScatterV']:
        checks.append(('held', REJECT,
                       f"{len(held)} readings of the last setpoint ({commanded[-1]:.2f} V), held "
                       f"{THRESHOLDS['HeldWaitS']:g} s on, scatter by {scatter:.3f} V, beyond the "
                       f"{THRESHOLDS['MaxHeldScatterV']:g} V allowed: the meter is not answering "
                       f"the reading asked for (a lag), or the platform is not steady."))
    elif abs(apart) > allowed:
        checks.append(('held', REJECT,
                       f"The last setpoint ({commanded[-1]:.2f} V) read {apart:+.3f} V "
                       f"({apart / plan.Step:+.1f} steps) from {len(held)} readings of it held "
                       f"{THRESHOLDS['HeldWaitS']:g} s on, beyond the {allowed:.2f} V allowed: "
                       f"the readings are not filed against the setpoints they were taken at "
                       f"(a lag), or the voltage had not settled."))
    else:
        checks.append(('held', PASS, f"the last setpoint read {apart:+.3f} V from itself held "
                                     f"(allowed {allowed:.2f} V)"))

    deviation = float(np.max(np.abs(commanded - plan.Setpoints)))
    if not np.array_equal(cal.Indices, np.arange(len(plan))):
        checks.append(('plan', REJECT, "The setpoints confirmed are not every planned "
                                       "setpoint once, in order."))
    elif deviation > THRESHOLDS['MaxPlanDeviationV']:
        checks.append(('plan', REJECT, f"The commanded voltages depart from the plan by up "
                                       f"to {deviation:.6f} V."))
    else:
        checks.append(('plan', PASS, f"commanded matches the plan to {deviation:.2e} V"))

    checks.append(('dac_calibration', *cal.Dac_Status))

    report = _Report(checks)
    # the run keeps the numbers it was judged by, not only the verdict
    report.update(TrackingR=tracking_r, FitGradient=gradient, FitResidualRmsV=residual_rms,
                  ShortestGapMs=shortest_ms, HeldApartV=apart, Thresholds=dict(THRESHOLDS))
    return report


def _Report(checks):
    verdicts = [verdict for _, verdict, _ in checks]
    worst = next((v for v in (REJECT, WARN) if v in verdicts), PASS)
    return {'Checks': [{'Check': c, 'Verdict': v, 'Message': m} for c, v, m in checks],
            'Verdict': worst}


def Rejections(report):
    return "\n".join(f"{c['Check']}: {c['Message']}" for c in report['Checks']
                     if c['Verdict'] == REJECT)


def Summary(report):
    """One line for the ELOG and the log."""
    if 'TrackingR' not in report:
        return f"no usable calibration  {report['Verdict']}"
    return (f"|r| = {report['TrackingR']:.5f}  m = {report['FitGradient']:.2f}  "
            f"RMS = {report['FitResidualRmsV']:.3f} V  "
            f"settled >= {report['ShortestGapMs']:.1f} ms  {report['Verdict']}")
