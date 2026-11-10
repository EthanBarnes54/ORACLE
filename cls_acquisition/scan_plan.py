"""The scan plan: which setpoints a scan visits, and in what order.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Each range's points are counted before they are laid down, so a float
step never adds one, and a range ends on its last whole step, never past
its stop (the old DAQ put 478 V on -450 to 477 V at 2 V). A point shared
by two ranges is laid down once. A plan is checked as it is built, and the
same plan drives the calibration and the scan.

Depends on: numpy.
"""

import numpy as np

WHOLE_STEP = 1e-9                   # steps short of a whole one that still count as one


class Scan_Plan_Error(ValueError):
    pass


class Scan_Plan():
    """Setpoints in platform volts, in the order they are commanded."""

    def __init__(self, starts, stops, step):
        self.Step = float(step)
        if not self.Step > 0:
            raise Scan_Plan_Error(f"The step size is {step} V; it must be positive.")
        self.Ranges = [[float(min(a, b)), float(max(a, b))] for a, b in zip(starts, stops)]
        if not self.Ranges:
            raise Scan_Plan_Error("No scan range is ticked.")

        points = []
        for low, high in self.Ranges:
            # a step's float error must not lose the stop itself: (46.2 - 0) / 0.1 is 461.99..
            count = int(np.floor((high - low) / self.Step + WHOLE_STEP)) + 1
            if count < 2:
                raise Scan_Plan_Error(
                    f"The range {low:g} to {high:g} V holds fewer than two points "
                    f"at a {self.Step:g} V step.")
            ladder = low + self.Step * np.arange(count)
            if points and abs(ladder[0] - points[-1]) < 0.5 * self.Step:
                ladder = ladder[1:]
            points.extend(ladder)
        self.Setpoints = np.array(points)

        if len(np.unique(np.round(self.Setpoints, 9))) != len(self.Setpoints):
            raise Scan_Plan_Error(
                "The ranges overlap, so the scan would command some setpoints twice.")

    def __len__(self):
        return len(self.Setpoints)

    def Bin_Edges(self):
        """One spectrum bin per step from the lowest setpoint to the highest,
        each centred on a step of the grid."""
        low, high = self.Setpoints.min(), self.Setpoints.max()
        count = int(round((high - low) / self.Step)) + 1
        return low + self.Step * (np.arange(count + 1) - 0.5)
