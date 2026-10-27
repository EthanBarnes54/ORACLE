"""Keeping the PC awake while ORACLE acquires, on Windows and on macOS.

Date:    2026-10-27
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

A PC that goes to sleep during a scan stops it where it stands (Scan
Performance's pauses), and one that sleeps with the platform at high
voltage leaves it there, since the DAC holds its last code. While
anything runs (a scan, the triggers, the voltage output, a calibration)
:class:`Keep_Awake` holds the operating system's own request that the
machine is in use: SetThreadExecutionState on Windows, and on macOS an
NSProcessInfo activity, which also keeps App Nap from slowing the
window's timers while it is hidden. The display may still sleep.

Staying awake helps a run, but is never a reason to refuse one: on any
other system, or if the request fails, it logs once and carries on.
The request belongs to the thread that made it on Windows, so it is
held and let go from the GUI thread only.

Depends on: ctypes and the operating system's own libraries.
"""

import ctypes
import ctypes.util
import logging
import sys

ES_CONTINUOUS = 0x80000000          # Windows: the state stays until it is changed
ES_SYSTEM_REQUIRED = 0x00000001     # Windows: no idle sleep
# macOS: NSActivityUserInitiated, which includes NSActivityIdleSystemSleepDisabled
NS_ACTIVITY_USER_INITIATED = 0x00FFFFFF
REASON = "ORACLE is acquiring data"

log = logging.getLogger('oracle.keep_awake')


class _Objc():
    """The few Objective-C messages macOS's activity takes, through libobjc."""

    def __init__(self):
        self._Lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))
        # NSProcessInfo and NSString live in Foundation, which a headless
        # process has not loaded
        ctypes.cdll.LoadLibrary(ctypes.util.find_library('Foundation'))
        for name, restype in (('objc_getClass', ctypes.c_void_p),
                              ('sel_registerName', ctypes.c_void_p)):
            getattr(self._Lib, name).restype = restype
            getattr(self._Lib, name).argtypes = [ctypes.c_char_p]

    def Class(self, name):
        return self._Lib.objc_getClass(name.encode())

    def Send(self, receiver, selector, *args, argtypes=(), restype=ctypes.c_void_p):
        """``[receiver selector...]``, objc_msgSend cast to this call's own
        signature, as arm64 requires."""
        function = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(
            ctypes.cast(self._Lib.objc_msgSend, ctypes.c_void_p).value)
        return function(receiver, self._Lib.sel_registerName(selector.encode()), *args)


class Keep_Awake():
    """``Hold(True)`` while anything runs, ``Hold(False)`` once nothing does."""

    def __init__(self, platform=sys.platform):
        self.Held = False
        self._Platform = platform
        self._Token = None              # macOS: the activity, retained
        self._Objc = None
        self._Warned = False

    def Hold(self, on):
        """Hold the request while ``on``; the same value again does nothing."""
        on = bool(on)
        if on == self.Held:
            return
        self.Held = on                  # a request that failed is not asked for every tick
        try:
            if self._Platform == 'win32':
                self._Windows(on)
            elif self._Platform == 'darwin':
                self._Mac(on)
        except Exception as e:
            if not self._Warned:
                self._Warned = True
                log.warning("The PC could not be kept awake while acquiring, so it may sleep "
                            "during a run: %s", e)

    def _Windows(self, on):
        state = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(state))

    def _Mac(self, on):
        objc = self._Objc = self._Objc or _Objc()
        info = objc.Send(objc.Class('NSProcessInfo'), 'processInfo')
        if on:
            reason = objc.Send(objc.Class('NSString'), 'stringWithUTF8String:',
                               REASON.encode(), argtypes=(ctypes.c_char_p,))
            token = objc.Send(info, 'beginActivityWithOptions:reason:',
                              NS_ACTIVITY_USER_INITIATED, reason,
                              argtypes=(ctypes.c_uint64, ctypes.c_void_p))
            # the token is autoreleased: kept past the event loop's next pool
            self._Token = objc.Send(token, 'retain')
        elif self._Token is not None:
            objc.Send(info, 'endActivity:', self._Token, argtypes=(ctypes.c_void_p,),
                      restype=None)
            objc.Send(self._Token, 'release', restype=None)
            self._Token = None
