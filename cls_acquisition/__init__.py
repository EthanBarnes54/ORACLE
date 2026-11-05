"""ORACLE's core: everything that runs without Qt.

Date:    2026-11-05
Version: 0.4.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The scan plan, the step-and-confirm calibration and its gate, the
hardware calibrations, the scan performance analysis, the acquisition
process, the run file, the facility services (ELOG, EPICS, the cooler's
PVs) and the instrument drivers. The GUI composes these. Nothing here imports
PySide6, so the acquisition process starts without Qt and every module
can be tested headless.

Depends on: numpy, PyYAML, asdf. The vendor libraries (pyvisa, mcculw,
spinapi, timetagger4, pyepics, py-elog, pcaspy) are imported only when
the hardware rack or a facility service is opened.
"""
