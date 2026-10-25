"""The scan plan: which setpoints a scan visits, and in what order.

Date:    2026-10-25
Version: 0.1.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Each range's points are counted before they are laid down, so rounding
in a floating-point step can never add one (an accumulated ``arange``
made 464 points where 463 were planned). Where a range starts on the
last point of the one before, that point is laid down once. The plan is
checked when it is built, so a plan that exists is a valid one, and the
same plan drives both the calibration and the scan.

Depends on: numpy.
"""

import numpy as np


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
            count = int(round((high - low) / self.Step)) + 1
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
