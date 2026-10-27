"""The MCC USB-231 converters that set the platform voltage through the Trek 609E-6.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The two converters drive the amplifier's two inputs, positive setpoints
on one and negative on the other, and the amplifier follows their
difference: with both at 0 V the platform sits at its offset, about
-1 V, whichever was driven last, and it runs on through 0 V (the bench,
24 and 25 October 2026). Each converter has its own gain and integral
non-linearity, so a line through each branch meets 0 V at a different
place, and no single line describes both: Mode C, in all 294 archived
runs. The outgoing converter is released before the incoming one is
driven, so the amplifier never sees both at once, and 0 V goes to the
positive converter.

Each converter is 16 bits across +-10 V, 0.305 V a code at the platform.
A setpoint is written as the code the board itself would round it to
(the bench's records match that rounding in every case), so what each
write returns is what the converter really holds, not what was asked.

The correction ships as identity. A linearity sweep measures each
branch's gain and offset, and, from a dense enough sweep, its integral
non-linearity: a smooth shape that repeated to 0.017 V rms between two
days on the bench, against 0.17 V for the shape itself. Its knots are
25 V apart because the shape repeats every 1024 codes (312.5 V), the
converter's major carry. Each run records which correction was in
force (DacCorrectionApplied, DacCorrection), a measured table past its
stated validity is refused at calibration, and a setpoint outside the
range the non-linearity was measured over is refused rather than
extrapolated.

The same board's digital line can gate the tagger's window (PulseBlaster
channel 18, through an AND unit), so a scan step's settling bunches are
never acquired while the PulseBlaster, and the buncher with it, runs on
untouched. With no line wired the gate does nothing, and the acquisition
loop discards those bunches itself.

Every setpoint is held to one limit, the most the amplifier may be asked
for, whoever asks: a plan beyond it is refused before its first write.

Depends on: numpy; mcculw, imported only when a board is opened.
"""

import time

import numpy as np

CH_POS, CH_NEG = 0, 1
AMPLIFIER_GAIN = 1000.0             # platform volts per converter volt
FULL_SCALE_V = 10.0                 # the converters' range (BIP10VOLTS)
CODES = 65536                       # 16 bits across it
LSB_V = 2 * FULL_SCALE_V / CODES    # one code at the converter
LIMIT_V = 4000.0                    # the Trek 609E-6's rated output, +-4 kV
BOARD_NAME = 'USB-231'
INL_REACH_V = 25.0                  # past the outermost knot a table still covers

CORRECTION = {
    'instrument': 'MCC USB-231',
    'measured': None,               # 'YYYY-MM-DD' once measured
    'valid_days': None,
    'method': 'not yet measured',
    'poly': {CH_POS: [1.0, 0.0], CH_NEG: [1.0, 0.0]},
    'inl': None,                    # {channel: [knots V, platform error V]} once measured
}


class Dac_Error(RuntimeError):
    pass


def Quantised(volts):
    """``(code, volts)`` the converter holds when asked for ``volts``: the
    nearest of its codes, as the board rounds."""
    code = int(np.clip(np.round((volts + FULL_SCALE_V) / LSB_V), 0, CODES - 1))
    return code, code * LSB_V - FULL_SCALE_V


class Dac():
    """``v_out(channel, volts)`` drives one converter; everything else is here."""

    def __init__(self, v_out, correction=CORRECTION, limit_v=LIMIT_V):
        self._V_Out = v_out
        self.Correction = correction
        self.Limit_V = limit_v

    def _Inl(self, channel):
        """``(knots, errors)`` of ``channel``'s non-linearity table, or None."""
        table = (self.Correction.get('inl') or {}).get(channel)
        return None if table is None else np.asarray(table, dtype=float)

    def Check(self, setpoints):
        """Raises ValueError if any of ``setpoints`` is beyond the limit, or
        outside the range the correction's non-linearity was measured over.
        0 V is always allowed: it is where the platform rests, and the idle
        loop must always be able to put it there."""
        setpoints = np.asarray(setpoints, dtype=float)
        worst = float(np.max(np.abs(setpoints), initial=0.0))
        if worst > self.Limit_V:
            raise ValueError(f"{worst:.1f} V is beyond the {self.Limit_V:.0f} V limit on the "
                             f"platform voltage.")
        for channel, side, name in ((CH_POS, setpoints > 0, 'positive'),
                                    (CH_NEG, setpoints < 0, 'negative')):
            table = self._Inl(channel)
            if table is None or not side.any():
                continue
            low, high = table[0].min() - INL_REACH_V, table[0].max() + INL_REACH_V
            outside = setpoints[side & ((setpoints < low) | (setpoints > high))]
            if len(outside):
                raise ValueError(f"{outside[0]:.1f} V is outside the {low:.0f} to {high:.0f} V "
                                 f"over which the DAC correction measured the {name} "
                                 f"converter's non-linearity; sweep that range or clear the "
                                 f"correction.")

    def Write(self, volts):
        """Set the platform to ``volts``; returns the (channel, converter
        volts) written, on the converter's code grid."""
        self.Check([volts])
        live, idle = (CH_POS, CH_NEG) if volts >= 0 else (CH_NEG, CH_POS)
        table = self._Inl(live)
        # command less what the converter's non-linearity will add; near 0 V that
        # can carry the target across zero, which the live converter drives, signed,
        # because the amplifier follows the difference of the two
        target = volts - (np.interp(volts, *table) if table is not None else 0.0)
        drive = (target if live == CH_POS else -target) / AMPLIFIER_GAIN
        wanted = float(np.polyval(self.Correction['poly'][live], drive))
        if abs(wanted) > FULL_SCALE_V:
            raise ValueError(f"{volts:.1f} V needs {wanted:.3f} V from the converter, "
                             f"beyond its {FULL_SCALE_V:.0f} V range.")
        _, dac_volts = Quantised(wanted)
        self._V_Out(idle, 0.0)
        self._V_Out(live, dac_volts)
        return live, dac_volts

    def Zero(self):
        for channel in (CH_NEG, CH_POS):
            self._V_Out(channel, 0.0)

    def Correction_Record(self):
        """``(applied, description)`` for the run file: applied is 0 or 1, a
        plain scalar, because nothing else in a run says whether its
        setpoints were corrected."""
        c = self.Correction
        if c['measured'] is None:
            return 0, "none: both channels identity, uncorrected"
        text = f"measured {c['measured']}, poly pos {c['poly'][CH_POS]} neg {c['poly'][CH_NEG]}"
        for channel, name in ((CH_POS, 'pos'), (CH_NEG, 'neg')):
            table = self._Inl(channel)
            if table is not None:
                text += (f", INL {name} {len(table[0])} knots {table[0].min():g} to "
                         f"{table[0].max():g} V")
        return 1, text

    def Calibration_Status(self, now=None):
        """``(verdict, message)`` for the gate. Unmeasured passes, since a check
        that fires on every run teaches operators to ignore them all; a measured
        table past its validity is worse than none and is refused."""
        c = self.Correction
        if c['measured'] is None:
            return 'PASS', "DAC correction unmeasured, both channels identity"
        if c['valid_days'] is None:
            return 'PASS', f"DAC correction measured {c['measured']}, no expiry declared"
        measured = time.mktime(time.strptime(c['measured'], '%Y-%m-%d'))
        age_days = ((time.time() if now is None else now) - measured) / 86400.0
        if age_days > c['valid_days']:
            return 'REJECT', (f"The DAC correction is {age_days:.0f} days old and valid for "
                              f"{c['valid_days']}; the two converters drift apart.")
        return 'PASS', f"DAC correction measured {c['measured']}, {age_days:.0f} days old"


def _Ul(board):
    """mcculw's ``ul``, its BIP10VOLTS range and a call that names a failure
    of ``board`` in a sentence."""
    try:
        from mcculw import ul
        from mcculw.enums import ULRange
    except (ImportError, OSError) as e:
        # mcculw is a wrapper of InstaCal's cbw64.dll, which exists only on Windows
        raise Dac_Error(f"The MCC DAC needs mcculw and InstaCal (MCC DAQ software), which "
                        f"run only on Windows; install ORACLE there with 'install.bat rack' "
                        f"and InstaCal, or set rack: simulated ({e}).") from e

    def call(what, fn, *args):
        try:
            return fn(*args)
        except ul.ULError as e:
            raise Dac_Error(f"The MCC board {board} failed to {what}: {e.message} (UL error "
                            f"{e.errorcode}).") from e

    return ul, ULRange.BIP10VOLTS, call


def Open_Mcc(board, correction=CORRECTION, limit_v=LIMIT_V):
    """The DAC on InstaCal board ``board``, which must be the USB-231; its
    outputs start at 0 V."""
    ul, bip10, call = _Ul(board)
    name = call('report its name', ul.get_board_name, board)
    if BOARD_NAME not in name:
        raise Dac_Error(f"InstaCal board {board} is a {name}, not the {BOARD_NAME} that drives "
                        f"the HV amplifier; set instruments: dac_board in the settings.")

    def v_out(channel, volts):
        # the code bench-proven on this board: its own conversion, then a_out
        code = call(f"convert {volts:+.5f} V", ul.from_eng_units, board, bip10, volts)
        call(f"write {volts:+.5f} V to output {channel}", ul.a_out, board, channel, bip10, code)

    dac = Dac(v_out, correction, limit_v)
    dac.Zero()
    return dac


class Mcc_Gate():
    """The tagger's window through an AND unit: the line high lets its
    triggers through."""

    def __init__(self, board, bit):
        from mcculw.enums import DigitalIODirection, DigitalPortType
        ul, _, call = _Ul(board)
        call(f"make digital line {bit} an output", ul.d_config_bit, board,
             DigitalPortType.AUXPORT, bit, DigitalIODirection.OUT)
        self._Out = lambda level: call(f"set digital line {bit} to {level}", ul.d_bit_out, board,
                                       DigitalPortType.AUXPORT, bit, level)
        self.Open()

    def Open(self):
        self._Out(1)

    def Close(self):
        self._Out(0)


class No_Gate():
    """No line wired: the window is never shut."""

    def Open(self):
        pass

    def Close(self):
        pass
