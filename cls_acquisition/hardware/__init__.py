"""The rack: the instruments ORACLE drives, real or simulated.

Date:    2026-11-09
Version: 0.5.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The acquisition process owns the DAC, the PulseBlaster and the tagger
(:func:`Open_Loop_Devices`); the GUI's process owns the two meters
(:meth:`Rack.Open_Meters`), since calibration times the line meter's
readings on the GUI's side. The drivers import their vendor libraries
only when the hardware rack is opened, so a simulated rack needs none.
A platform limit set above the amplifier's rating, a divider ratio, the
cooler's rolling window or the PMTs' dead time that is not a positive
number, or power line cycles the line meter does not take, is refused
before anything opens.

:meth:`Rack.Identity` is the rack as it names itself (the meters'
*IDN?, the DAC board's unique ID, the tagger's wrapper) and as the
settings name the amplifier, which cannot; :meth:`Rack.Dividers` are
the dividers as the settings give them, ratios and all. Both are
recorded with every run and bench record: which runs a hardware change
touched is often learned only later.

Depends on: .keysight, .mcc_dac, .pulseblaster, .simulated, .timetagger,
.visa_session.
"""

from cls_acquisition.hardware import keysight, mcc_dac, pulseblaster, simulated, timetagger
from cls_acquisition.hardware.visa_session import Pyvisa_Connect, Visa_Session


class Rack():

    def __init__(self, settings, context):
        self.Simulated = settings['rack'] != 'hardware'
        self._Instruments = dict(settings['instruments'])
        limit = self._Instruments['hv_limit_v']
        if not 0 < limit <= mcc_dac.LIMIT_V:
            raise ValueError(f"The settings put the platform's limit at {limit:g} V; it must be "
                             f"above 0 and at most the Trek 609E-6's rating of "
                             f"{mcc_dac.LIMIT_V:.0f} V (instruments: hv_limit_v).")
        self._Hardware = settings['hardware']
        for key in ('scan_divider', 'cooler_divider'):
            ratio = self._Hardware[key]['ratio']
            if not (isinstance(ratio, (int, float)) and ratio > 0):
                raise ValueError(f"The settings give the {key.replace('_', ' ')} a ratio of "
                                 f"{ratio!r}; it must be a positive number of platform volts "
                                 f"per volt at its meter (hardware: {key}: ratio).")
        for value, what, key in ((settings['cooler']['rolling_s'],
                                  "the cooler's rolling mean a window", 'cooler: rolling_s'),
                                 (settings['detectors']['dead_time_us'],
                                  "the PMTs a dead time", 'detectors: dead_time_us')):
            if not (isinstance(value, (int, float)) and value > 0):
                raise ValueError(f"The settings give {what} of {value!r}; it must be a "
                                 f"positive number ({key}).")
        nplc = settings['calibration']['nplc']
        if nplc not in keysight.NPLC_CHOICES:
            raise ValueError(f"The settings ask for calibration readings of {nplc!r} power "
                             f"line cycles; the 34465A takes "
                             f"{', '.join(f'{n:g}' for n in keysight.NPLC_CHOICES)} "
                             f"(calibration: nplc).")
        self.Scan_Divider = float(self._Hardware['scan_divider']['ratio'])
        self.Cooler_Divider = float(self._Hardware['cooler_divider']['ratio'])
        self._Correction = settings['dac_correction'] or mcc_dac.CORRECTION
        self._Platform = simulated.Platform(context) if self.Simulated else None

    def Loop_Spec(self):
        """What the acquisition process opens its devices from, handed over
        when it is spawned."""
        return {'simulated': self.Simulated, 'instruments': self._Instruments,
                'correction': self._Correction, 'platform': self._Platform}

    def Open_Meters(self):
        """``(line meter, cooler meter)``."""
        if self.Simulated:
            line = lambda _: simulated.Keysight(
                simulated.Line_Meter_Source(self._Platform, self.Scan_Divider),
                simulated.LINE_NOISE_V, simulated.LINE_MAINS_V)
            cooler = lambda _: simulated.Keysight(
                simulated.Cooler_Meter_Source(self.Cooler_Divider), simulated.COOLER_NOISE_V)
        else:
            line = cooler = Pyvisa_Connect
        line_meter = keysight.Line_Meter(Visa_Session.Open(self._Instruments['line_meter'], line))
        try:
            cooler_meter = keysight.Cooler_Meter(
                Visa_Session.Open(self._Instruments['cooler_meter'], cooler))
        except Exception:
            line_meter.Close()
            raise
        return line_meter, cooler_meter

    def Dividers(self):
        """Both dividers as the settings give them, for a record."""
        return {'Scan': dict(self._Hardware['scan_divider']),
                'Cooler': dict(self._Hardware['cooler_divider'])}

    def Identity(self, loop, line_meter, cooler_meter):
        """The rack, for a record: ``loop`` is the acquisition process's
        'Identity' reply (None if it gave none), the meters as opened."""
        loop = loop or {}
        return {'Kind': 'simulated' if self.Simulated else 'hardware',
                'LineMeter': line_meter.Idn, 'CoolerMeter': cooler_meter.Idn,
                'Dac': loop.get('Dac'), 'Tagger': loop.get('Tagger'),
                'Amplifier': {**self._Hardware['amplifier'], 'Gain': mcc_dac.AMPLIFIER_GAIN,
                              'Wiring': mcc_dac.WIRING}}


def Open_Loop_Devices(spec):
    """``(dac, pulse generator, tagger, window gate)``, in the acquisition process."""
    instruments = spec['instruments']
    dac_args = spec['correction'], instruments['hv_limit_v']
    if spec['simulated']:
        tagger = simulated.Tagger(spec['platform'])
        dac = mcc_dac.Dac(simulated.Converters(spec['platform']).V_Out, *dac_args,
                          {'Board': f"simulated {mcc_dac.BOARD_NAME}", 'UniqueId': None})
        return dac, simulated.Pulse_Generator(tagger), tagger, simulated.Window_Gate(tagger)
    bit = instruments['settle_gate_bit']
    gate = (mcc_dac.No_Gate() if bit is None
            else mcc_dac.Mcc_Gate(instruments['dac_board'], bit))
    return (mcc_dac.Open_Mcc(instruments['dac_board'], *dac_args), pulseblaster.Pulse_Blaster(),
            timetagger.Time_Tagger(instruments['tagger_levels'],
                                   instruments['tagger_trigger_level'],
                                   instruments['tagger_driver']), gate)
