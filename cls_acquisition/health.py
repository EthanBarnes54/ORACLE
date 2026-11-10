"""The health check: one report on whether the rack is as it should be, at the start of a shift.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

It judges (WP11) the bench table's check, the cooler's 50 Hz ripple, each
PMT's dark counts, the triggers numbered and received against the
PulseBlaster's period, and both meters' trigger sources. Each check is
kept as YAML beside the hardware records and set against the last. It
changes nothing.

Depends on: numpy, PyYAML; cls_acquisition.hardware_calibration.
"""

import glob
import os
from datetime import datetime

import numpy as np
import yaml

from cls_acquisition import hardware_calibration as hc

COOLER_S, COOLER_INTERVAL_S, COOLER_APERTURE_S = 10.0, 0.001, 0.0002
COUNT_S = 10.0                      # of triggers, for the PMTs and the trigger count
RIPPLE_GROWTH = 1.5                 # the ripple this many times the last check's: grown
DARK_CHANGE = 2.0                   # a PMT's rate this many times up or down: changed
TRIGGER_SLIP = 0.01                 # of the triggers expected (two at least): slipped
PASS, WARN, NA = 'PASS', 'WARN', 'N/A'
RANK = {NA: 0, PASS: 1, WARN: 2}


def Run(link, line_meter, cooler_meter, cooler_divider, timing, inputs, progress, stopped):
    """What the rack says of itself: the cooler over COOLER_S, COUNT_S of
    ``timing``'s triggers with each of ``inputs`` counted, and both meters'
    trigger sources; ``progress(text)`` as each starts."""
    progress("Reading the cooler...")
    cooler = hc.Cooler_Monitor(cooler_meter, cooler_divider, COOLER_S, COOLER_INTERVAL_S,
                               COOLER_APERTURE_S, None, lambda _: None, stopped)
    mains = [p for p in cooler['Peaks'] if p['Mains_Harmonic'] == 1]
    # as the cooler carries it: the meter's own aperture takes a little out
    ripple = (mains[0]['Amplitude_V'] / max(mains[0]['Integration_Gain'], 0.1)
              if mains else 0.0)
    progress("Counting the triggers and the PMTs...")
    counted = link.Send('Count', timeout=COUNT_S + 30.0, timing=timing, seconds=COUNT_S)
    return {'When': f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            'Cooler': {'Mean_V': cooler['Mean_V'], 'Ripple_V': float(ripple),
                       'Rms_V': cooler['Rms_Detrended_V']},
            'Triggers': {key: counted[key] for key in ('Seconds', 'Expected', 'Numbered',
                                                       'Received')},
            'Counts': {int(c): int(counted['Counts'].get(c, 0)) for c in inputs},
            'Sources': {'Line meter': line_meter.Trigger_Source(),
                        'Cooler meter': cooler_meter.Trigger_Source()},
            'Trace': (cooler['Times_S'], cooler['Volts'])}


def _Check(name, verdict, rows, note):
    return {'Check': name, 'Verdict': verdict, 'Rows': rows, 'Note': note}


def Judge(found, last=None, table=None):
    """``found`` judged against the ``last`` check (None: the first), with
    the window's table check (``Verdict``, ``Fit``; None if there was
    none): a list of ``{'Check', 'Verdict', 'Rows', 'Note'}``."""
    if table is None:
        checks = [_Check('Bench table', NA, [], "No table is in force, so none was checked.")]
    else:
        good = table['Verdict'] == PASS
        checks = [_Check('Bench table', table['Verdict'], [
            ('Residual', f"{table['Fit']['Rms_V'] * 1e3:.1f} mV rms"),
            ('Points', table['Fit']['Points'])],
            "The converters are as the table says." if good else
            "The table no longer describes the converters: measure it again.")]
    last = last or {}

    cooler, was = found['Cooler'], (last.get('Cooler') or {}).get('Ripple_V')
    grown = bool(was) and cooler['Ripple_V'] > RIPPLE_GROWTH * was
    checks.append(_Check('Cooler', WARN if grown else PASS, [
        ('Mean', f"{cooler['Mean_V']:.1f} V"),
        ('50 Hz ripple', f"{cooler['Ripple_V']:.3f} V" + (f" (last {was:.3f} V)" if was else '')),
        ('Scatter', f"{cooler['Rms_V']:.3f} V rms")],
        "The ripple has grown since the last check: look at the cooler supply." if grown else
        "As at the last check." if was else "The first check here: the next is set against it."))

    seconds, before = found['Triggers']['Seconds'], last.get('Counts') or {}
    then_s = (last.get('Triggers') or {}).get('Seconds') or 1.0
    rows, silent, changed = [], [], []
    for pmt, n in sorted(found['Counts'].items()):
        rate, old = n / seconds, before.get(pmt)
        text = f"{rate:.1f} /s"
        if old is not None:
            expected = old * seconds / then_s
            text += f" (last {old / then_s:.1f} /s)"
            if (n and old and max(n / expected, expected / n) > DARK_CHANGE
                    and abs(n - expected) > 3.0 * np.sqrt(n + expected)):
                changed.append(pmt)
        silent += [pmt] if n == 0 else []
        rows.append((f"PMT {pmt}", text))
    checks.append(_Check('PMTs', WARN if silent or changed else PASS, rows, ' '.join(
        [f"Counted over {seconds:.0f} s of triggers: with the beam off, their dark counts."]
        + ([f"No counts on {', '.join(map(str, silent))}: unwired, unpowered or failed."]
           if silent else [])
        + ([f"{', '.join(map(str, changed))} changed by more than {DARK_CHANGE:g} times "
            f"since the last check: its HV, threshold or light-tightness."] if changed else []))))

    triggers = found['Triggers']
    slipped = abs(triggers['Numbered'] - triggers['Expected']) > max(
        2.0, TRIGGER_SLIP * triggers['Expected'])
    lost = triggers['Numbered'] - triggers['Received']
    checks.append(_Check('Triggers', WARN if slipped or lost > 0 else PASS, [
        ('PulseBlaster', f"{triggers['Expected']:.0f} in {seconds:.1f} s, by its period"),
        ('Tagger', f"{triggers['Numbered']} numbered, {triggers['Received']} reached the loop")],
        ("The tagger did not count the triggers the PulseBlaster gave: their cable, level "
         "or pattern. " if slipped else "")
        + (f"{lost} numbered triggers never reached the loop." if lost > 0 else "")
        or "Every trigger given was counted and read."))

    sources = found['Sources']
    external = [meter for meter, source in sources.items() if not source.startswith('IMM')]
    checks.append(_Check('Meter triggers', WARN if external else PASS, list(sources.items()),
                         f"{' and '.join(external)} would read on its trigger input, not when "
                         f"asked." if external else "Both read when asked (IMM)."))
    return checks


def Verdict(checks):
    return max((c['Verdict'] for c in checks), key=RANK.get, default=NA)


def Save(directory, found, checks):
    """The check as ``health_<time>.yaml`` beside the hardware calibration
    records; returns its path."""
    folder = os.path.join(directory, 'hardware_calibration')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"health_{datetime.now():%Y-%m-%d_%H%M%S}.yaml")
    record = {key: value for key, value in found.items() if key != 'Trace'}
    record['Verdicts'] = {c['Check']: c['Verdict'] for c in checks}
    with open(path, 'x', encoding='utf-8') as fh:
        yaml.safe_dump(record, fh, sort_keys=False, allow_unicode=True)
    return path


def Last(directory):
    """The newest check kept in ``directory``'s records, or None."""
    for path in sorted(glob.glob(os.path.join(directory, 'hardware_calibration',
                                              'health_2*.yaml')), reverse=True):
        try:
            with open(path, encoding='utf-8') as fh:
                found = yaml.safe_load(fh)
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(found, dict):
            return found
    return None


def Elog_Text(checks, when):
    """The check for the ELOG: its verdict, then a line per part."""
    lines = [f"HEALTH CHECK  {Verdict(checks)}  ({when})"]
    for check in checks:
        rows = '; '.join(f"{label} {value}" for label, value in check['Rows'])
        lines.append(f"  {check['Check']}: {check['Verdict']}" + (f", {rows}" if rows else '')
                     + f". {check['Note']}")
    return '\n'.join(lines) + '\n'
