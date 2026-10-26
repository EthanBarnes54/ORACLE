"""Spectra from run files: counts against the scan, on any of the data viewer's axes.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The collinear data viewer's analysis, mended. Counts are binned on the
run's own scan plan, one bin per step as the DAQ stepped, never on a
step read off the calibration's first gap; and they are never binned
again. Every other axis is a function of the scan voltage, so it moves
each bin's edges and leaves its counts those of the setpoints they were
taken at. (The viewer re-binned calibrated voltages into even bins,
which empties some bins and doubles others, and replaced voltages in
place one setpoint after another, which could map a voltage twice.)

* Calibrated voltage: the platform as the calibration read it, each
  setpoint's own readback, straight between them and along the
  calibration's line beyond them.
* Beam energy: the charge times the cooler's voltage less the
  calibrated scan voltage.
* Frequency and wavenumber: the laser times its harmonic, Doppler-shifted
  into the ions' frame at that energy, as DENIS computes it, less a
  reference.

A run's hits can be gated in time of flight and in time since its first
hit, and taken from chosen channels (channel 5 is the old DAQ's DC
counter). The hyperfine model is drawn in the ions' frequency, so
:meth:`Run.Model_Axis` carries frequencies back to whichever axis is
shown, through the same calibration.

Depends on: numpy; cls_acquisition.calibration, .run_file, .scan_plan,
.scan_performance and .hardware.keysight.
"""

import os

import numpy as np

from cls_acquisition import run_file
from cls_acquisition.calibration import DIVIDER, Line_Fit
from cls_acquisition.hardware.keysight import COOLER_DIVIDER
from cls_acquisition.scan_performance import C_LIGHT, Ion_Frequency
from cls_acquisition.scan_plan import Scan_Plan, Scan_Plan_Error

AXES = ('Scan voltage', 'Calibrated voltage', 'Beam energy', 'Frequency', 'Wavenumber')
AXIS_LABELS = {'Scan voltage': ('Scan voltage', 'V'),
               'Calibrated voltage': ('Calibrated voltage', 'V'),
               'Beam energy': ('Beam energy', 'eV'),
               'Frequency': ('Frequency - reference (MHz)', None),
               'Wavenumber': ('Wavenumber - reference (cm⁻¹)', None)}
TIMESTAMP, VOLTS, BUNCH, CHANNEL, TOF, COOLER = range(6)   # run_file.DATA_COLS
CM_HZ = C_LIGHT * 100.0             # hertz per inverse centimetre


class Spectrum_Error(ValueError):
    pass


class Run():
    """One run file, read whole: its header, its hits, its plan's bins and
    its calibration."""

    def __init__(self, path):
        self.Path = path
        self.Name = os.path.basename(path)[:-len('.asdf')]
        self.Header, self.Raw, _ = run_file.Read(path)
        header = self.Header
        try:
            plan = Scan_Plan(*zip(*header['ScanningRanges']), header['StepSize'])
        except (Scan_Plan_Error, KeyError, TypeError, ValueError) as e:
            raise Spectrum_Error(f"{self.Name}'s scanning ranges do not make a plan: "
                                 f"{e}") from None
        self.Edges = plan.Bin_Edges()
        commanded = np.asarray(header.get('CalSet', []), dtype=float)
        readback = np.asarray(header.get('CalReadback', []), dtype=float) * DIVIDER
        if len(commanded) != len(readback):
            commanded = readback = np.zeros(0)
        # a setpoint the old DAQ commanded twice is read as the mean of its readings
        self.Cal_Set, inverse = np.unique(np.round(commanded, 6), return_inverse=True)
        self.Cal_Readback = (np.bincount(inverse, readback) / np.bincount(inverse)
                             if len(readback) else readback)
        self.Cooler_V = float(header.get('CoolerVoltage', np.nan)) * COOLER_DIVIDER
        self.Mass_Amu = float(header.get('MassAMU', np.nan))
        self.Laser_Cm = float(header.get('LaserSetpoint', np.nan))
        self.Start = float(self.Raw[:, TIMESTAMP].min()) if len(self.Raw) else 0.0
        self.Channels = np.unique(self.Raw[:, CHANNEL]).astype(int).tolist()

    def Calibrated(self, volts):
        """The platform voltage the calibration says each scan voltage put
        on it; the scan voltage itself for a run with no usable calibration."""
        volts = np.asarray(volts, dtype=float)
        if len(self.Cal_Set) < 2 or not np.ptp(self.Cal_Readback):
            return volts
        inside = np.interp(volts, self.Cal_Set, self.Cal_Readback)
        gain, offset, _ = Line_Fit(self.Cal_Set, self.Cal_Readback)
        outside = (volts < self.Cal_Set[0]) | (volts > self.Cal_Set[-1])
        return np.where(outside, gain * volts + offset, inside)

    def _Ion_Hz(self, volts, laser):
        """The ions' frequency of the laser at each scan voltage."""
        for value, what in ((self.Cooler_V, 'cooler voltage'), (self.Mass_Amu, 'mass'),
                            (self.Laser_Cm, 'laser setpoint')):
            if not np.isfinite(value):
                raise Spectrum_Error(f"{self.Name} holds no {what}, so it has no frequency "
                                     f"axis.")
        beam_v = self.Cooler_V - self.Calibrated(volts)
        return Ion_Frequency(beam_v, self.Laser_Cm * CM_HZ, self.Mass_Amu, laser)

    def Axis(self, volts, axis, laser, reference_cm=0.0):
        """Scan voltages ``volts`` on ``axis`` (one of AXES)."""
        if axis == AXES[0]:
            return np.asarray(volts, dtype=float)
        if axis == AXES[1]:
            return self.Calibrated(volts)
        if axis == AXES[2]:
            if not np.isfinite(self.Cooler_V):
                raise Spectrum_Error(f"{self.Name} holds no cooler voltage, so it has no beam "
                                     f"energy axis.")
            return laser['charge'] * (self.Cooler_V - self.Calibrated(volts))
        ion_hz = self._Ion_Hz(volts, laser) - reference_cm * CM_HZ
        return ion_hz / 1e6 if axis == AXES[3] else ion_hz / CM_HZ

    def Reference_Cm(self, laser):
        """The ions' wavenumber at the middle of the scan: a reference that
        puts the frequency axes about zero."""
        middle = 0.5 * (self.Edges[0] + self.Edges[-1])
        return float(self._Ion_Hz([middle], laser)[0] / CM_HZ)

    def Model_Axis(self, offsets_mhz, axis, laser, reference_cm):
        """Ion frequencies ``offsets_mhz`` from the reference, on ``axis``,
        through this run's calibration and Doppler shift; NaN beyond its scan."""
        volts = np.linspace(self.Edges[0], self.Edges[-1], 2000)
        ion_mhz = self.Axis(volts, AXES[3], laser, reference_cm)
        order = np.argsort(ion_mhz)
        offsets = np.asarray(offsets_mhz, dtype=float)
        scan = np.interp(offsets, ion_mhz[order], volts[order], left=np.nan, right=np.nan)
        return self.Axis(scan, axis, laser, reference_cm)

    def Hits(self, channels, tof_gate=None, time_gate=None):
        """The rows on ``channels`` inside the gates: ``tof_gate`` in us,
        ``time_gate`` in seconds since the run's first hit."""
        rows = self.Raw[np.isin(self.Raw[:, CHANNEL], channels)]
        if tof_gate is not None:
            rows = rows[(rows[:, TOF] > tof_gate[0]) & (rows[:, TOF] < tof_gate[1])]
        if time_gate is not None:
            since = rows[:, TIMESTAMP] - self.Start
            rows = rows[(since >= time_gate[0]) & (since <= time_gate[1])]
        return rows

    def Counts(self, channels, tof_gate=None, time_gate=None):
        """Hits per scan step, on the plan's bins."""
        rows = self.Hits(channels, tof_gate, time_gate)
        return np.histogram(rows[:, VOLTS], self.Edges)[0].astype(float)
