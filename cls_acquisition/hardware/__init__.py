"""The rack: the instruments ORACLE drives, real or simulated.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The acquisition process owns the DAC, the PulseBlaster and the tagger
(:func:`Open_Loop_Devices`); the GUI's process owns the two meters
(:meth:`Rack.Open_Meters`), since calibration times the line meter's
readings on the GUI's side. The drivers import their vendor libraries
only when the hardware rack is opened, so a simulated rack needs none.

Depends on: .keysight, .mcc_dac, .pulseblaster, .simulated, .timetagger,
.visa_session.
"""

from cls_acquisition.hardware import keysight, mcc_dac, pulseblaster, simulated, timetagger
from cls_acquisition.hardware.visa_session import Pyvisa_Connect, Visa_Session


class Rack():

    def __init__(self, settings, context):
        self.Simulated = settings['rack'] != 'hardware'
        self._Instruments = dict(settings['instruments'])
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
            line = lambda _: simulated.Keysight(simulated.Line_Meter_Source(self._Platform))
            cooler = lambda _: simulated.Keysight(simulated.Cooler_Meter_Source())
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


def Open_Loop_Devices(spec):
    """``(dac, pulse generator, tagger, window gate)``, in the acquisition process."""
    instruments = spec['instruments']
    dac_args = spec['correction'], instruments['hv_limit_v']
    if spec['simulated']:
        tagger = simulated.Tagger(spec['platform'])
        dac = mcc_dac.Dac(simulated.Converters(spec['platform']).V_Out, *dac_args)
        return dac, simulated.Pulse_Generator(tagger), tagger, simulated.Window_Gate(tagger)
    bit = instruments['settle_gate_bit']
    gate = (mcc_dac.No_Gate() if bit is None
            else mcc_dac.Mcc_Gate(instruments['dac_board'], bit))
    return (mcc_dac.Open_Mcc(instruments['dac_board'], *dac_args), pulseblaster.Pulse_Blaster(),
            timetagger.Time_Tagger(instruments['tagger_levels'],
                                   instruments['tagger_trigger_level'],
                                   instruments['tagger_driver']), gate)
