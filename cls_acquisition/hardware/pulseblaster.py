"""The SpinCore PulseBlaster that clocks the buncher and the time tagger.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

Programs a pattern from cls_acquisition.triggers as a loop of time
slices: each slice is one instruction, and the last branches back to the
first. Stopping runs an all-low program, so no output is left high.

Depends on: spinapi; cls_acquisition.triggers.
"""

from cls_acquisition import triggers

CONTINUE, BRANCH = 0, 6             # PulseBlaster instruction codes
PULSE_PROGRAM = 0
CORE_CLOCK_MHZ = 100.0
IDLE_SLICE_NS = 5000


class Pulse_Blaster():

    def __init__(self):
        try:
            from spinapi import spinapi
        except (ImportError, OSError) as e:
            # spinapi loads SpinCore's native library as it is imported
            raise RuntimeError(f"The PulseBlaster needs spinapi and SpinCore's SpinAPI driver; "
                               f"install them on the rack PC ('install.bat rack' and SpinCore's "
                               f"installer), or set rack: simulated ({e}).") from e
        self._Pb = spinapi
        spinapi.pb_select_board(0)
        if spinapi.pb_init() != 0:
            raise RuntimeError(f"The PulseBlaster did not initialise: "
                               f"{spinapi.pb_get_error()}.")
        spinapi.pb_core_clock(CORE_CLOCK_MHZ)

    def _Load(self, slices):
        pb = self._Pb
        pb.pb_stop()
        pb.pb_start_programming(PULSE_PROGRAM)
        first = pb.pb_inst_pbonly(slices[0][1], CONTINUE, 0, slices[0][0])
        for duration, bits in slices[1:-1]:
            pb.pb_inst_pbonly(bits, CONTINUE, 0, duration)
        pb.pb_inst_pbonly(slices[-1][1], BRANCH, first, slices[-1][0])
        pb.pb_stop_programming()

    def Program(self, pattern):
        self._Load(triggers.Time_Slices(pattern))

    def Start(self):
        self._Pb.pb_start()

    def Stop(self):
        self._Load([[IDLE_SLICE_NS, 0], [IDLE_SLICE_NS, 0]])
        self._Pb.pb_start()

    def Close(self):
        self._Pb.pb_close()
