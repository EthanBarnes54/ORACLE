# ORACLE

**O**nline **R**eadout & **A**cquisition for **C**ollinear **L**aser spectroscopy **E**xperiments

The data acquisition for collinear laser spectroscopy (CLS) at IGISOL: it steps the platform voltage through a scan, records every ion the time tagger sees against the voltage it was taken at, and calibrates the voltage axis before every run. It keeps the collinear DAQ's layout and file format, brings in the bench calibration tool (cls-cal), and is built so that the old DAQ's faults cannot happen rather than being caught afterwards.

**Version:** 0.6.0
**Developer:** Ethan Barnes (ethan.barnes@postgrad.manchester.ac.uk)
**Framework:** PySide6 (Qt6) + pyqtgraph + numpy + asdf
**Manual:** [`docs/manual/manual.pdf`](docs/manual/manual.pdf)

> **Status.** ORACLE runs end to end on its simulated rack, which imitates the rack the bench tool measured in October 2026. It has not yet driven the beamline: its drivers are tested against stand-ins for the vendor libraries.

---

## Quick Start

```bash
git clone <this repository> ORACLE
cd ORACLE
install.bat full       # Windows   (no argument: a menu; rack = full + instrument libraries)
./install.sh full      # macOS / Linux
uv run python gui.py   # or launch from the shortcut
```

ORACLE starts on the simulated rack. On the beamline PC, install with `rack`, set `rack: hardware` in `settings/settings.yaml`, and build the corrected time-tagger wrapper in `cls_acquisition/hardware/timetagger4_wrapper/` (manual, §2).

## Features

| Tab | What it does |
|---|---|
| **Scan controls** | Up to seven ranges, Sawtooth, Triangle or Single scans, bunched or DC; settle-aware stepping; a scan queue that runs unattended and stops on any fault; live HFS, TOF and TOF-HFS plots with a gate; the ELOG panel; File ▸ Open run brings any run back as taken |
| **Scan Calibration** | Step-and-confirm calibration of every scan's own setpoints, judged by a gate before anything is written; each calibration fitted to the bench table, V = a + b·T |
| **Hardware Calibration** | The bench tool: settle time, linearity and DAC correction, cooler monitor, the bench table (measure, resume, check, compare) and a health check for the start of a shift |
| **Scan Performance** | How the rack shaped each run's spectrum, check by check against counting statistics; afterpulsing per PMT; the run's systematic inputs; nothing is ever cut |
| **Time graphs** | Counts against time and each PMT's live afterpulse ratio |

Behind the window: one owner thread per instrument and every reading on command; a simulated rack that runs the whole DAQ unmodified; every input kept across a crash; the PC kept awake while anything runs; Windows and macOS alike, with the same suite on both.

## Data

Run files are the collinear DAQ's, so cls_tools and DENIS read them unchanged: `<data>/<year>/<month>/<day>/run_NNNN.asdf`, the archive's sixteen header keys and rows of `timestamp, voltage, bunch_number, channel, time, cooler`. One more key, `ORACLE`, holds ORACLE's own record (timing, view, gate, rack identity, dividers, DAC codes, bench table and fit), which readers that look keys up by name pass over. `run_NNNN_monitor.csv` beside each run records the rack every 0.1 s. Every key and file is listed in the manual (§9).

**Proposed for cls_tools** (for A. Raggio's review): the bench table, `<data>/tables/table_<YYYYMMDD>_<id8>.csv`, one row per converter and code (`channel, code, setpoint_v, volts, sigma_v, readings`), with its details as `# key: value` lines above; a run's `DacCodes` and `ScanTable` let its calibration be fitted to it as V = a + b·T(code).

## Repository Layout

```
ORACLE/
|-- gui.py                   # entry point
|-- install.bat, install.sh  # installers (full | rack | uv | shortcut)
|-- cls_acquisition/         # the core, no Qt: plan, calibration and gate, hardware
|   |                        #   calibrations, scan performance, acquisition process,
|   |                        #   run file, facility services, settings
|   |-- hardware/            # meters, DAC, PulseBlaster, tagger, the simulated rack,
|       |-- timetagger4_wrapper/  # the corrected tagger wrapper (C++) and its build
|-- gui/                     # PySide6: main window, five tabs, theme, plots, log
|-- docs/manual/             # LaTeX manual (manual.pdf pre-built; build.sh / build.bat)
|-- settings/                # settings.yaml, inputs.yaml, rack.yaml (all untracked)
|-- tests/                   # headless pytest suite; tests/data/bench: the rack's records
|-- .github/workflows/       # the suite on Windows and macOS
```

## Development

```bash
uv sync
uv run pytest            # about 5 min; no hardware and no window needed
```

The suite (about 430 tests) runs headless on Windows and macOS, and CI runs it on both. It holds every control of the window end to end through a real acquisition process; ORACLE's analyses against the rack's own bench records; the old DAQ's plans, patterns and formats where ORIGINAL_DAQ_STUFF is present; and both systems' clocks, folders and settings. A guard fails the run if a test touched the real settings or left files in the repository.

## Known limits

Nothing has yet run on the beamline. The settle model should be measured there; the settle gate (an AND unit on PulseBlaster channel 18) is not wired; the DAC's 0.305 V codes leave up to ±0.15 V of rounding on a 1 V step; the corrected tagger wrapper is not yet built on the rack. See the manual's Appendix B.

## License

MIT ([LICENSE](LICENSE)). The theme is carried over from DENIS (A. Kayaalp, MIT); icons are from [Lucide](https://lucide.dev/) (ISC). The hardware calibrations follow cls-cal (A. Raggio). Finished runs are viewed and fitted in DENIS.
