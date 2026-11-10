"""Session logging: every launch leaves a trace, however it ends.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

DENIS's session log. :func:`Install`, first in ``main()``, writes a dated
log under ``<app>/logs`` (the newest few kept), tees stdout and stderr into
it, sends the "oracle" logger (the acquisition process's records too),
warnings and uncaught exceptions on any thread there, and points
faulthandler at it for native crashes. :func:`Install_Qt_Handler` routes
Qt's messages; call it before the QApplication is made.

Depends on: the standard library (PySide6 inside Install_Qt_Handler).
"""

import faulthandler
import logging
import os
import sys
import threading
from datetime import datetime

_LOGGER = 'oracle'


def _Prune(folder, keep):
    try:
        names = [os.path.join(folder, n) for n in os.listdir(folder)
                 if n.startswith('oracle_') and n.endswith('.log')]
    except OSError:
        return
    names.sort(key=os.path.getmtime, reverse=True)
    for path in names[keep:]:
        try:
            os.remove(path)
        except OSError:
            pass


class _Tee():
    """Writes to the original stream, if there is one, and to the log file.
    A failure on either is swallowed: logging must never crash the app."""

    def __init__(self, original, log_file):
        self._Sinks = [s for s in (original, log_file) if s is not None]
        self._Original = original

    def write(self, text):
        for sink in self._Sinks:
            try:
                sink.write(text)
            except Exception:
                pass

    def flush(self):
        for sink in self._Sinks:
            try:
                sink.flush()
            except Exception:
                pass

    def fileno(self):
        if self._Original is None:
            raise OSError("no console stream")
        return self._Original.fileno()

    def isatty(self):
        return False


def Install(app_root, verbose=False, keep=20):
    """Start the session log; returns its path, or None if no file could be made."""
    folder = os.path.join(app_root, 'logs')
    path, log_file = None, None
    try:
        os.makedirs(folder, exist_ok=True)
        _Prune(folder, keep)
        path = os.path.join(folder, f"oracle_{datetime.now():%Y-%m-%d_%H%M%S}.log")
        log_file = open(path, 'w', encoding='utf-8', buffering=1)
        faulthandler.enable(file=log_file, all_threads=True)
    except OSError:
        path = None

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    logger = logging.getLogger(_LOGGER)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    if log_file is not None:
        handler = logging.StreamHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-7s %(name)s: %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)
        logging.captureWarnings(True)
        logging.getLogger('py.warnings').addHandler(handler)

    previous = sys.excepthook

    def crash(kind, value, trace):
        logger.critical("Uncaught exception", exc_info=(kind, value, trace))
        previous(kind, value, trace)

    def thread_crash(args):
        logger.error("Uncaught exception in thread %r", args.thread.name,
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = crash
    threading.excepthook = thread_crash
    return path


def Install_Qt_Handler():
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    logger = logging.getLogger(_LOGGER).getChild('qt')
    levels = {QtMsgType.QtDebugMsg: logging.DEBUG, QtMsgType.QtInfoMsg: logging.INFO,
              QtMsgType.QtWarningMsg: logging.WARNING,
              QtMsgType.QtCriticalMsg: logging.ERROR, QtMsgType.QtFatalMsg: logging.CRITICAL}

    def handler(kind, context, message):
        logger.log(levels.get(kind, logging.INFO), "%s", message)

    qInstallMessageHandler(handler)
