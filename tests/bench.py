"""An acquisition loop on a simulated rack, run on a thread of the test process.

The real Acquisition_Loop, with the simulated devices and plain queues,
so its behaviour can be checked step by step without spawning a process.
"""

import multiprocessing
import queue
import threading
import time

from cls_acquisition.acquisition import Acquisition_Loop, Shared_State
from cls_acquisition.hardware import mcc_dac, simulated


class Loop_Bench():
    """The loop, its devices and both ends of its queues; every converter
    write is recorded in Writes."""

    def __init__(self, parent_alive=lambda: True):
        context = multiprocessing.get_context('spawn')
        self.Platform = simulated.Platform(context)
        self._Converters = simulated.Converters(self.Platform)
        self.Writes = []
        self.Dac = mcc_dac.Dac(self._V_Out)
        self.Tagger = simulated.Tagger(self.Platform)
        self.Pulses = simulated.Pulse_Generator(self.Tagger)
        self.State = Shared_State(context)
        self.Instructions, self.Replies, self.Data = queue.Queue(), queue.Queue(), queue.Queue()
        self.Gate = simulated.Window_Gate(self.Tagger)
        self.Loop = Acquisition_Loop(self.Dac, self.Pulses, self.Tagger, self.Gate,
                                     self.Instructions, self.Replies, self.Data, self.State)
        self.Thread = threading.Thread(target=self.Loop.Run, args=(parent_alive,), daemon=True)
        self.Thread.start()
        self._Id = 0

    def _V_Out(self, channel, volts):
        self.Writes.append((channel, volts))
        self._Converters.V_Out(channel, volts)

    def Send(self, command, **params):
        """``(ok, payload)`` of one instruction."""
        self._Id += 1
        self.Instructions.put((self._Id, command, params))
        reply_id, ok, payload = self.Replies.get(timeout=10)
        assert reply_id == self._Id
        return ok, payload

    def Packets(self, until_end=False, timeout=20.0):
        """Data packets until the scan's end marker, or until the queue is quiet."""
        packets, deadline = [], time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                packets.append(self.Data.get(timeout=0.5))
            except queue.Empty:
                if not until_end:
                    break
                continue
            if until_end and packets[-1][0] == 'scan_end':
                break
        return packets

    def Close(self):
        if self.Thread.is_alive():
            self.Send('Shutdown')
            self.Thread.join(timeout=5)
        self.Loop.Close()
