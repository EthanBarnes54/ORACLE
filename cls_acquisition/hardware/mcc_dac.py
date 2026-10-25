"""The MCC USB-231 converters that set the platform voltage through the Trek 609E-6.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Positive setpoints, and 0 V, come from one converter and its amplifier;
negative setpoints from the other. Each has its own offset and gain, so
the platform carries a seam at 0 V (Mode C in all 294 archived runs).
Software cannot remove that, only avoid making it worse: 0 V always goes
down the positive branch, and the outgoing converter is released before
the incoming one is driven, so both amplifiers are never live at once.

The per-channel correction ships as identity. No correction is applied
until the two converters are measured on the bench; each run records
which correction was in force (DacCorrectionApplied, DacCorrection), and
a measured table past its stated validity is refused at calibration.
The Hardware Calibration tab's linearity sweep measures it.

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
LIMIT_V = 6000.0                    # the Trek 609E-6's rated output

CORRECTION = {
    'instrument': 'MCC USB-231',
    'measured': None,               # 'YYYY-MM-DD' once measured
    'valid_days': None,
    'method': 'not yet measured',
    'poly': {CH_POS: [1.0, 0.0], CH_NEG: [1.0, 0.0]},
}


class Dac():
    """``v_out(channel, volts)`` drives one converter; everything else is here."""

    def __init__(self, v_out, correction=CORRECTION, limit_v=LIMIT_V):
        self._V_Out = v_out
        self.Correction = correction
        self.Limit_V = limit_v

    def Check(self, setpoints):
        """Raises ValueError if any of ``setpoints`` is beyond the limit."""
        worst = float(np.max(np.abs(setpoints), initial=0.0))
        if worst > self.Limit_V:
            raise ValueError(f"{worst:.1f} V is beyond the {self.Limit_V:.0f} V limit on the "
                             f"platform voltage.")

    def Write(self, volts):
        """Set the platform to ``volts``; returns the (channel, converter volts) used."""
        self.Check([volts])
        live, idle = (CH_POS, CH_NEG) if volts >= 0 else (CH_NEG, CH_POS)
        dac_volts = float(np.polyval(self.Correction['poly'][live],
                                     abs(volts) / AMPLIFIER_GAIN))
        if abs(dac_volts) > FULL_SCALE_V:
            raise ValueError(f"{volts:.1f} V needs {dac_volts:.3f} V from the converter, "
                             f"beyond its {FULL_SCALE_V:.0f} V range.")
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
        return 1, f"measured {c['measured']}, poly pos {c['poly'][CH_POS]} neg {c['poly'][CH_NEG]}"

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


def Open_Mcc(board, correction=CORRECTION, limit_v=LIMIT_V):
    from mcculw import ul
    from mcculw.enums import ULRange

    def v_out(channel, volts):
        ul.v_out(board, channel, ULRange.BIP10VOLTS, volts)

    dac = Dac(v_out, correction, limit_v)
    dac.Zero()
    return dac


class Mcc_Gate():
    """The tagger's window through an AND unit: the line high lets its
    triggers through."""

    def __init__(self, board, bit):
        from mcculw import ul
        from mcculw.enums import DigitalIODirection, DigitalPortType
        ul.d_config_bit(board, DigitalPortType.AUXPORT, bit, DigitalIODirection.OUT)
        self._Out = lambda level: ul.d_bit_out(board, DigitalPortType.AUXPORT, bit, level)
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
