"""ORACLE's core: everything that runs without Qt.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The scan plan, calibration and its gate, the hardware calibrations, Scan
Performance, the acquisition process, run files, facility services and
instrument drivers. Nothing imports PySide6, so the acquisition process
starts without Qt and every module tests headless.

Depends on: numpy, PyYAML, asdf; the vendor libraries (pyvisa, mcculw,
spinapi, timetagger4, pyepics, py-elog, pcaspy) only as the rack or a
service opens.
"""
