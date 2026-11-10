# ORACLE

**O**nline **R**eadout & **A**cquisition for **C**ollinear **L**aser spectroscopy **E**xperiments

The data acquisition for collinear laser spectroscopy (CLS) at IGISOL: it steps the platform voltage through a scan, records every ion the time tagger sees against the voltage it was taken at, and calibrates the voltage axis before every run. ORACLE keeps the collinear DAQ's layout and functions, brings in the bench calibration tool, and is built so that the faults found in that DAQ cannot happen rather than being caught afterwards.

**Version:** 0.6.0
**Developer:** Ethan Barnes (ethan.barnes@postgrad.manchester.ac.uk)
**Framework:** PySide6 (Qt6) + pyqtgraph + numpy + asdf

> **Status.** ORACLE runs end to end on its simulated rack, on any OS, and that rack now imitates the one the bench calibration tool measured on 24 and 25 October 2026. ORACLE itself has not yet driven the beamline: its drivers follow the collinear DAQ and the bench tool, whose code has run on the rack, but are tested against stand-ins for the vendor libraries only. Connecting to the rack, and automating that connection at start-up, is the next stage.

---

## Quick Start

The install path uses [`uv`](https://docs.astral.sh/uv/), which brings its own Python, the project venv and every dependency in one step.

```bash
git clone <this repository> ORACLE
cd ORACLE

# No argument gives a menu. full = uv + dependencies + desktop shortcut
# (on macOS an ORACLE.app in ~/Applications, linked from the Desktop);
# rack = the same plus the instrument libraries, for the beamline PC
install.bat full       # Windows
./install.sh full      # Linux / macOS

# Launch from the shortcut, or:
uv run python gui.py
```

ORACLE starts on the **simulated rack**. On the beamline PC, install with `rack` and set `rack: hardware` in `settings/settings.yaml` (see [Settings](#settings)). Build the corrected time-tagger wrapper there too ([below](#the-time-taggers-wrapper)).

---

## Features

The window is the collinear DAQ's, with the bench calibration tool's measurements and a performance analysis added: five tabs above the quick-tune controls and the experiment information.

### Scan controls
- Up to seven voltage ranges, a step size, and Sawtooth, Triangle or Single scans; bunches per channel (bunched) or seconds per channel (DC)
- **Interaction region** (CEC or LCR) and **PMT HV**, as the operator gives them, recorded with each run: the cooler offset differs between regions, and these say which runs share one. **Reference run** tags a run taken to measure that offset, in its record and its ELOG entry
- Buncher timing: dwell, bunch release, the tagger's trigger delay and its acquisition gate (held inside the tagger's 8.39 ms timestamp range). A timing that does not fit its cycle is refused with a sentence before anything starts: a release past the end of the cycle (the buncher would never fire), a trigger delay past it (the cycle would silently run long) or a gate still open when the next trigger comes. In DC mode the gate stays settable, since DC hits come through the tagger too (the 1 ms DC cycle at its 0 to 200 µs default gate acquires a fifth of the time)
- **Start scan** calibrates the scan's own setpoints, puts the calibration through the gate, and only then writes the run file and starts the scan. **Stop after scan** finishes the round; **Stop now!** ends it within a bunch
- **Scan queue**, the second page of the Scan parameters box (titled as DENIS's optimiser tools are): scans laid out ahead, one per row and one run file each, with their ranges (`20:55, 60:80`), step, bunches (seconds in DC mode), order (Sawtooth or Triangle) and number of scans; the buncher timing, experiment and view are the parameters page's, held for the whole queue. Every row is checked as it is typed, as a scan is when it starts, and the readout gives each one's setpoints and time, or its fault, and when the whole queue would end. **Add current** adds the scan parameters as they stand, **Show** puts a row back on them, rows paste from a spreadsheet, and Delete clears cells. **Automatic** starts each scan as the last ends, so a queue can run overnight unattended; **Manual** asks before each after the first, saying what comes next. Each scan takes exactly its number of scans, ending itself, and its run's record notes its place in the queue. Anything but a clean end (a calibration refused or failed, a file not written, a scan stopped by hand, the acquisition process gone) stops the queue and leaves the platform at rest, its row saying why: an unattended queue never runs on past a fault. **Stop queue after this scan** lets the running scan finish. The rows and their states survive a crash, and a queue started again takes up at the first scan not done
- Live plots: HFS spectrum, TOF spectrum with a gate (Ctrl-drag, or Cmd-drag on macOS; double-click removes it) and its own bin size, and a TOF-HFS map. The spectrum and the map count only hits inside the gate. A TOF bin finer than 50,000 across the gate is widened to that, as the old DAQ's was, and the map is held only while shown, at up to 2000 bins a side, so no setting can ask the window for gigabytes
- **File ▸ Open run...** (Ctrl+O, Cmd+O on macOS) brings a run back as it stood the moment it finished: its hits on these plots and the Time graphs, drawn with its own tagger window and trigger rate; the Scan tab's controls (ranges, step, bunches, scan type, experiment, buncher timing) and its view (TOF bin, gate, channels, map); the calibration gate's verdict and checks on the Scan Calibration tab; its run and scan numbers; and its analysis on Scan Performance. A run older than ORACLE's record of it keeps today's controls for what it lacks, and the Shown line says which. A finished run stays on every tab, drawn as it was taken while the controls are set up for the next, until another run is opened or a scan or the triggers start. Viewing on other axes and fitting are DENIS's
- The plots' view (TOF bin, gate, channels, map, and the Time graphs' bin and channels) is held from Start until the scan's last event, as its parameters are, so the run's record of it holds throughout; the plots still pan and zoom
- ELOG panel: append text, start a new entry, attach the HFS spectrum
- **Settle-aware stepping.** The PulseBlaster, and the buncher and cooler with it, runs on untouched through a scan; what pauses is acquisition. At each step the tagger's window is shut, the DAC is written, and the window stays shut for as long as the settle model says a jump of that size takes; every step then takes its full count of bunches at a settled voltage. With the gate wired (an AND unit on PulseBlaster channel 18, driven by one MCC digital line) no settling bunch is ever acquired; without it the loop drops them in software, good to within the USB latency

### Scan Calibration
- A stand-alone calibration over any range; the plots fill point by point (readback against set, readback minus set, measured step size), and the gate's verdict and every check are listed when it finishes
- **Against the bench table**: with a table in force, every calibration (a scan's own, or one taken here) is fitted to it, V = a + b·T at each point's DAC code, each point weighted by the meter's noise (from the held readings) and the table's own 10 mV repeatability. Reported: a (the run's 0 V level) and b − 1 with their errors, the residual against what those two account for (a WARN, never a refusal, when it is twice that: the table is stale or the hardware changed), how many points fell between the table's codes, and a and b of the last runs on the same table. The fitted table is drawn over the readback's departure from the setpoint. It is recorded with the run (`TableFit`) and in its ELOG entry; CalSet and CalReadback are unchanged

### Hardware Calibration
The bench calibration tool (cls-cal, A. Raggio), in ORACLE: the measurements that answer the briefing's questions about the rack itself. Each drives the DAC by step and confirm, ramps every unmeasured move (rate and step above the pages), ramps back to 0 V on every exit, stop or failure, and writes its record under `<data>/hardware_calibration/` **as it goes**, so a measurement that fails or is stopped keeps what it took; the record ends with its status and results. On the rack each first asks the operator to arm the high voltage, naming the most it will apply.
- **Settle time**: steps the platform out and back while the line meter samples it on its own timer (the interval the meter set, read back), and reports each step's lag from the command, settling time and 10-90 % rise from its 10 % crossing, overshoot and noise, with the slowest and mean of each step; optionally a **drift tail** at a long integration after each step. The slowest step is checked against ORACLE's SETTLE, **Use as scan settle model** makes the scan's settle model from it, and **Use recommended wait for linearity** sets the linearity sweep's wait after each step to three times the slowest step, at least 0.2 s, as the bench tool did
- **Linearity**: a count-based ladder up, down, up then down, or in random order (which separates drift from shape), with a finer grid across 0 V and 0 V read again every few rungs; several readings at a chosen NPLC with autozero, after the meter's own autocalibration if asked. Reported: gain and offset with their uncertainties, residual rms and largest, INL in ppm of full scale, curvature, each branch's line and where the two meet 0 V, the gain mismatch, hysteresis, the drift of the 0 V checks, and from a sweep with at least five rungs in every 25 V the **integral non-linearity** of each converter, apart from what the code rounding leaves. **Use as DAC correction** puts into force, and into the settings, the correction that cancels each branch's gain and offset and that non-linearity, dated, with an expiry (the branches' offsets drift by up to 0.1 V a day; the shape holds); a measured correction past its expiry refuses scans until it is measured again or cleared
- **Cooler monitor**: the cooler streamed off the meter's timer (`R?`), with no gaps and longer than its memory if need be, at a short aperture (ripple) or an NPLC (drift). Reported: level, drift, rms and peak to peak before and after the drift, noise by decade of frequency, the white-noise floor, the oscillations (frequency, amplitude, SNR, mains harmonics named, the meter's own attenuation of each) and how much of the noise they are, and the averaged spectrum, saved as its own record (`cooler_spectrum_*.csv`, the bench tool's `spectrum.csv`)
- **Scan table**: the bench table (A. Raggio, November 2026), the platform voltage each DAC code really gives. **Make from a linearity record...** builds one from a sweep with its 0 V checks: each reading less the 0 V level interpolated between the checks (it warms up by 0.2 V in the first hour), averaged by converter and code. It is kept in the library under `<data>/tables/`, never overwritten, named by its date and ID (its content's hash), and offered for use. **Use a table...** puts one from the library, or from anywhere, in force; **Compare with...** sets another against it code by code, 0 V levels aligned, to see whether the converters have changed. The page shows each branch's shape about a straight line (the non-linearity a straight-line calibration leaves), the 0 V level's drift during the sweep (warning if it was still warming up), and how often 0 V should be checked during a run. A table measured on another rack, or one the settings name that is not in the library, is warned of at start-up, never refused. On the bench tool's own records, the 25 October table predicts the 26 October sweep to 11.6 mV rms once a run's a + b·T is fitted (Raggio: 10.3 mV over twelve runs)
- **Line meter: Read (5 × NPLC 10)**, the bench tool's *Read DMM*: the platform read precisely, against whatever quick tune holds on it, to check the wiring and polarity
- **Open record...** on each page shows a record saved earlier, from this PC or another: its plots, table and analysis, as when it was taken. A linearity record opened this way does not offer its DAC correction, since the correction in force when it was swept is not in the record
- A measurement that fails or is stopped part way is still analysed as far as it went, and its record closes with those results, as the bench tool's `results_partial.json` did

### Scan Performance
How far the rack's behaviour during a run shaped its data, run by run: the last run is analysed as it closes, and any other can be picked from the data folder or from disk. Every check returns PASS, WARN, FAIL or N/A with the number it decided on, judged in scan volts against the run's own step:
- **Voltage scan**: hits carrying a voltage other than the one their bunch was due at; setpoints outside the plan; the calibration's tracking and residual; where the two branches' lines meet 0 V; the platform during the scan (the line meter's live readback) against the start-of-run calibration; the calibration against the runs before it that were calibrated the same way (by ORACLE or by the old DAQ, under the same DAC correction), since across either the two measure different things
- **Timing**: whether reads kept up with the bunches; stalls and pauses (a sleeping PC shows here); bunches taken before their step had settled; triggers the loop missed; the TOF peak's drift beyond its own scatter
- **Beam and spectrum**: beam losses, rate stability and scan-to-scan shape against counting statistics; the strongest peak's centroid scan by scan, its drift and jitter; channel balance
- **Cooler and laser**: the cooler as the analysis takes it, the rolling mean of its readings over `cooler: rolling_s` (15 s), plotted over the readings: how far the mean moved, which the analysis follows, and how far the readings scatter about it, which no mean removes; whether each hit's own reading is what its ions saw: the line's shift against the reading's departure from the mean, by quarters of the hits (1: what the ions saw, so a correction hit by hit would narrow the line; 0: not), calibrated on the run's own hits and flagged only when significant and the narrowing matters, with the slope and its error written to `run_NNNN_performance.yaml` for a campaign's runs to be combined, since one run rarely tells; the laser converted through the Doppler shift at the run's mass and beam energy
- **Detectors**: each PMT's afterpulsing, as A. Raggio's slide 9 shows it: the time to the previous count on itself and on the other PMTs against chance (taken from the next bunches, so the bunch's time-of-flight profile is kept), and its counts' scatter across scans against dead time, the part every PMT shares (the beam) told from its own. A PMT is flagged when its own pairs within `detectors: dead_time_us` stand significantly above the other PMTs' and amount to 0.5 % of its counts or more; a dead time is suggested for DENIS, never applied
- **Systematic inputs**: each source of error as a width at the line, in scan volts and in MHz at the ions, marked as measured on this run (the calibration line's errors, the HV against its calibration, the wavemeter during the run, the cooler's scatter about its mean, afterpulsing, trigger health), declared in the settings (the beam's offset for the run's interaction region, the dividers' accuracy, the 0 V drift until it is checked, the wavemeter's accuracy) or not yet known (the bench table). The A and B Monte Carlo that turns them into errors is DENIS's. A short block (the calibration, the drift, the cooler, the afterpulsing and the largest term) goes to the ELOG as each run is analysed; **Post systematics to ELOG** appends the whole table, and **Export report** writes `run_NNNN_systematics.yaml` beside the run
- **Flagged for review**: what deserves a look, each with its reason. Nothing is cut: the operator decides in DENIS. **Export report** writes `run_NNNN_performance.yaml` beside the run

### Time graphs
- Counts against time for the chosen channels, in bins of 100 s down to 0.1 s at the trigger rate in force
- Each PMT's afterpulsing as the data come in: its pairs within the dead time on itself over those on the other PMTs, against chance (1), shown in the warning colour above 2, an early warning of a tube going bad

### Quick tune and experiment information
- Triggers on and off without a scan; voltage output on and off, set by a slider across a sweep range or by value, both held to the platform's limit. Turning the output on starts from the voltage already on the platform, so it never jumps
- Cooler and readback voltages, the laser wavenumber, run and scan numbers, buncher mode

### Behind the window
- **Simulated rack**: the whole DAQ runs unmodified against software instruments that behave as the bench measured the rack: a 0.4 ms rise and a 0.18 % overshoot that settles back (a 1 kV jump inside 0.5 V in 17 ms, 2 kV in 20 ms); readback less setpoint as the 1 V-step sweep across ±3 kV found it, running on through 0 V at about −1 V; every write on the converter's code grid; the line meter's 0.065 V scatter at a 200 µs aperture; the cooler at 29884.8 V with its 0.44 V mains ripple and harmonics. The window title and status bar say *simulated rack*, and simulated runs go to their own folder and never to the ELOG
- **Windows and macOS alike**: the same code and the same suite on both. The PC is kept awake while anything runs (a scan, the triggers, the voltage output, a calibration), through Windows' execution state or a macOS activity, which also stops App Nap slowing a hidden window; on Windows the taskbar shows ORACLE as itself, not as pythonw. Sub-millisecond timing uses `perf_counter`, one clock across both processes on both systems, where `monotonic` ticked every 15.6 ms on Windows before Python 3.13. A settings file ORACLE cannot read (a Windows folder in double quotes, say) is refused in a window naming the line, never started on defaults; a vendor library a system cannot load (mcculw and the tagger's wrapper are Windows-only) is named with its fix
- **The meters** are reset and identified when opened, and reset again when closed, as the bench tool leaves them (they must answer `*IDN?` as a Keysight 3446x or 3447x), held under VISA's exclusive lock (so the bench tool or the old DAQ cannot open them alongside ORACLE), and every reading is checked for the meter's overload and invalid marks
- **The DAC** must be the USB-231 on the board the settings name, and no setpoint, from anywhere, may exceed the limit: by default the Trek 609E-6's rating of ±4 kV
- **Rack warnings**: a fault of the rack's own that ORACLE cannot mend, such as the old tagger wrapper's edges, is marked in the status bar for the whole session, its sentence in the tooltip and the log; so is a change of hardware since the last session on that kind of rack (a meter, the DAC board, the tagger, the amplifier or a divider, as each names itself or the settings name it), with what changed, since the settle model and the scan table should then be measured again. The last rack of each kind is kept in `settings/rack.yaml`
- **The cooler's PVs** (`FURIOS:Cooler:Readback` and `InstantValue`), which the old DAQ's separate server published, can be published from ORACLE's own readings (settings: `cooler_pv`; needs pcaspy)
- Every plot is handled the same way: drag to pan, right-drag to stretch or squash, the wheel to zoom (over an axis, that axis alone), Shift-drag to zoom to a box. A plot zoomed by hand stops following live data; the ⟲ button in its corner, a double-click, or **View ▸ Autoscale all plots** (Ctrl+0) returns it to the full view
- One instance per user (two would fight over the instruments); a second launch raises the first
- Session log in `logs/oracle_<timestamp>.log`, holding the acquisition process's records too (newest 20 kept; uncaught exceptions and native crashes included)
- Dark theme, wheel guard and plot style shared with [DENIS](https://github.com/EthanBarnes54/DENIS_mac): a spin box or slider changes only once clicked into, so scrolling past quick tune never moves the platform

---

## Calibration: step and confirm

Calibration pairs each commanded platform voltage with what the line meter reads through the ×1000 divider. For every setpoint, in order:

1. the acquisition process writes the DAC, stamps the time once the write has returned, and confirms;
2. the GUI checks the confirmation is for this setpoint, the meter waits SETTLE, and it takes **one** reading on command;
3. the reading is kept only if it was taken at least SETTLE after its write (or longer, below).

Reading *i* cannot be requested before write *i* is confirmed, so the setpoint and the reading can never slip apart: there is one clock, and the PulseBlaster and tagger take no part. Nothing estimates or undoes a lag anywhere; the suite runs the real loop, meter driver and calibration with every reply, write and meter answer delayed at random, and judges each reading by the setpoint the simulated platform really held. As a guard on that construction, the last setpoint is then read again held (the `held` check below): on the beamline's ramp it refuses a lag of any k from 1 to 12 before the scan starts, whether the platform or the meter's answers trail, which the gate could not do before. A reading taken too early is discarded and the reading (not the write) is requested again after the measured shortfall. Retries are capped per point (3 reads) and per calibration (10), and every other fault stops the calibration at once with a sentence naming it.

**SETTLE is 50 ms**, and the bench has now measured what it stands for: a 2 kV jump settled inside 0.5 V 20.5 ms after a 3 ms lag, and to the meter's own scatter in about 30 ms; a jump of the full 4 kV extrapolates to about 42 ms. A point whose jump the settle model says needs longer waits that long: the first, from where the platform stood, and the first of a range after a gap, the two worst cases. The stamp test and the gate hold each point to the same number as its wait.

The gate then checks properties every healthy calibration has. A **REJECT** writes no file and starts no scan; a **WARN** goes to the ELOG with the run.

| Check | Refuses or warns when |
|---|---|
| length | there is not one reading per planned setpoint |
| tracking | the readback does not follow the command (\|r\| < 0.90) |
| span | the readback's span is more than 10 % off the commanded span |
| polarity | (WARN) the two converter branches' lines meet 0 V more than 0.10 V apart |
| settle | a kept reading was taken sooner after its write than its jump takes to settle (SETTLE, or the settle model's longer time); (WARN) setpoints were read again |
| held | the last setpoint, read ten more times held 0.2 s on with no stepping (so with no pairing to get wrong), does not read steady (scatter over 0.5 V), or the calibration's own last reading sits more than half a step (or six of their scatter) from it. A lag of k, by whatever route, puts it k steps off: tracking and span cannot see a shifted ramp |
| plan | the commanded voltages are not the plan, once each, in order |
| dac_calibration | a measured DAC correction is past its stated validity |

All the numbers live in one table, `THRESHOLDS` in `cls_acquisition/calibration.py`, and each run's record keeps the ones it was judged by. A refusal's reasons go to the session log as well as the ELOG, since no run file keeps them; the run number is given again to the next attempt, so a refusal leaves no gap. The suite runs the gate on the ramp the beamline uses (−450 to +475 V in 2 V steps) through the measured rack: uncorrected it warns on polarity, and with a measured correction it passes outright.

### What it closes in the collinear DAQ

| Fault in the collinear DAQ | In ORACLE |
|---|---|
| D1, Mode A: two threads on one meter could leave a burst ungated | One owner thread per instrument; the meter is a state machine; there is no burst to leave ungated, and a reply of more than one value is refused. Every reading, in every mode, is taken on command and never on the meter's trigger input (PulseBlaster channel 17, the buncher's line): each mode reads the trigger source back and refuses anything else, and the PulseBlaster cannot be started while a calibration runs |
| D2, Mode B: setpoint *i* read as reading *i + k* | Step and confirm: the lag is zero by construction |
| D3, Mode C: a fixed 0.95 V shape in every calibration, read as a 0.44 V seam at 0 V | The bench found the platform runs on through 0 V: the shape is each converter's gain and integral non-linearity, which repeats from day to day to 0.017 V. The DAC correction now cancels both: on the measured twin, to the converters' code rounding (0.09 V rms). 0 V goes to the positive converter and the outgoing converter is released first |
| D4: the DC path recorded nothing | DC hits come through the tagger like bunched ones |
| D5: a tagger read that timed out moved the scan on, with no trigger behind it | The tagger takes no part in calibration. In a scan a read that times out is no bunch: a step holds its bunches however long the cycle (the dwell goes to 100 s), and a tagger that sees no trigger for three cycles (two seconds at least) is reported on the status bar, naming the likely cause, while the scan holds its setpoint until triggers return |
| Mode B's root in the scan: the loop counted each tagger read as one bunch, though a read hands over every trigger the card has ready, so a loop running behind merged several triggers' hits under one bunch number and ran its steps long, and the counts' parting was only logged | Each trigger is its own bunch, counted by the card's own number for it; the corrected wrapper marks every trigger, empty ones too, and triggers past a step's quota are dropped. A number the card skipped is a trigger lost on the way, reported as such |
| D6: floating-point plans gained a point, shared endpoints commanded twice | Plans are counted, then laid down; a range that is not a whole number of steps ends on its last whole step, never past its stop; shared endpoints once; checked when built |
| D7: nothing inspected a calibration before the file was written | The gate, before anything is written |
| D8: two sessions on one instrument | One session per resource within ORACLE, and VISA's exclusive lock between programs |
| D9: dead code that disagreed with the live path | Not carried over |
| The window hung forever if the acquisition process died | Every instruction gets one reply carrying its request id, unknown ones are refused, and a dead process is reported |
| A crashed window left the acquisition process holding the rack | The process watches its parent, and leaves the triggers off and the platform at 0 V |
| The tagger wrapper timed inputs 3 and 4 on input 2's edge, recorded rollover marks as hits, ignored a card that failed to open, and left no row for a trigger with no hits | A corrected wrapper (below); until it is built, a rack warning names the inputs affected and says a read of only empty triggers is taken for one, and no window may pass the timestamp range |
| A meter's overload mark (9.9E37) passed for a voltage; a meter left by another program streaming, in binary or on another trigger, was read as it was | Every reading is checked for the marks; each meter is reset and identified when opened |
| The cooler's PV server opened the cooler meter a second time and read it from its own thread | The PVs come from ORACLE's own readings |

### What it closes in the bench tool

| Fault | In ORACLE |
|---|---|
| A sweep that failed or was aborted lost what it took unless it reached its end | Every record is written as the measurement goes, and closed with its status |
| The cooler monitor's blocks left gaps between them | One stream, read out while it runs |

---

## Settings

`settings/settings.yaml` holds what differs from the defaults in `cls_acquisition/settings.py` (it is created on first save and not tracked by git).

Beside it, `settings/inputs.yaml` is ORACLE's memory of every input in the window (each tab's parameters, the plots' view, the hardware pages, quick tune's sweep range, the ELOG draft), written half a second after any change, whole and then renamed into place, so a crash leaves the last complete set. ORACLE opens with those inputs, not the latest run's settings. A file it cannot read costs only its values: every input keeps its default, and the session log says why. It is not tracked by git; delete it to start from the defaults.

| Key | Default | Meaning |
|---|---|---|
| `rack` | `simulated` | `hardware` drives the beamline's instruments |
| `data_directory` | *(empty: `<app>/data`)* | where run files go on the rack. A simulated rack always writes to `<app>/data/simulated` |
| `experiment` | | the experiment name, saved as it is edited |
| `elog` | disabled | `enabled`, `url`, `author` (empty: the login name). Never posts from a simulated rack |
| `pvs` | the wavemeter and mass PVs | EPICS names of the wavenumber, mass and mass setpoint, and of each tagger input's PMT high voltage (`pmt_hv_1` to `pmt_hv_4`; empty: not on EPICS, and the Scan tab's PMT HV box is recorded instead) |
| `hardware` | ratios 1000 and 10000 | what the rack cannot say of itself, recorded with every run and bench record. `scan_divider`, `cooler_divider`: `model`, `serial`, `ratio` (platform volts per meter volt, which scales every readback) and `ppm` (the ratio's stated accuracy). `amplifier`: `model`, `serial` |
| `instruments` | the rack's | VISA resources of the line and cooler meters, the MCC board (`dac_board`, which must be the USB-231), the tagger driver's folder (`tagger_driver`), the tagger thresholds, and `hv_limit_v` (4000: the Trek 609E-6's rating, and the most it may be set to), the most any scan, quick tune or hardware calibration may ask of the platform. A plan beyond it is refused before its first write |
| `instruments: settle_gate_bit` | *(empty: not wired)* | the MCC digital line (AUXPORT bit) gating the tagger's window through an AND unit on PulseBlaster channel 18 |
| `dac_correction` | *(empty: identity)* | the DAC correction a linearity sweep measured, written by **Use as DAC correction**: each branch's polynomial, and its non-linearity table where measured. A setpoint outside the range the table was measured over is refused, never extrapolated. Production runs write uncorrected, since the bench table is applied after collection: a calibration with a correction in force is warned of, lest it be applied twice |
| `scan_table` | *(empty: none)* | the ID of the bench table in force, from the library under `<data>/tables/`, set by the Scan table page; each run records it |
| `settle_model` | *(empty: SETTLE for every step)* | the scan's settling against jump size, written by **Use as scan settle model** on the Settle time page; each step waits 1.25 times it |
| `cooler_pv` | disabled | `enabled`, `prefix` (`FURIOS:Cooler:`): publish the cooler's voltage as EPICS PVs. Needs pcaspy, which has no Windows wheel for Python 3.13 yet; never from a simulated rack |
| `calibration` | `nplc` 1 | power line cycles each calibration reading integrates (20 ms each): whole cycles take out the mains the divider picks up, which a 200 µs reading caught at a random phase (75 mV a point against about 7 mV). The meter's display poll keeps its 200 µs aperture |
| `cooler` | `rolling_s` 15 | the window of the cooler's rolling mean, in seconds: the analysis takes the cooler as the mean of its readings over it, since a reading is not taken with the bunch and alone aliases the mains ripple (A. Raggio, November 2026). Recorded with each run; Scan Performance judges a run by it and plots it over the readings |
| `detectors` | `dead_time_us` 10 | the dead time the PMTs are judged at for afterpulsing (Scan Performance) and counted at as data arrive (the Time graphs tab); recorded with each run. Nothing here applies it: the dead time is the operator's to apply in DENIS |
| `systematics` | no offsets; 0.030 V; 10 MHz | what the systematics table cannot measure: `beam_offset_v`, the beam's offset from the cooler's readback by interaction region as `[offset, error]` in volts from the campaign's reference runs (for example `CEC: [13.0, 8.0]` for Bi); `zero_drift_v`, the 0 V drift assumed until ORACLE checks it during runs; `wavemeter_mhz`, the wavemeter's accuracy per run at the fundamental |
| `laser` | harmonic 4, anti-collinear, charge 1 | how the laser meets the ions, as DENIS takes it: Scan Performance puts the laser into scan volts with it |

---

## Data format

Run files are written exactly as the collinear DAQ wrote them, so cls_tools and DENIS read them unchanged: `<data>/<year>/<month>/<day>/run_NNNN.asdf`, ASDF standard 1.5.0, the same sixteen header keys (`CalSet` and `CalReadback` are the calibration; `DacCorrectionApplied` and `DacCorrection` say which DAC correction was in force), and the events streamed after the header as rows of `timestamp, voltage, bunch_number, channel, time, cooler`. The header's keys are checked when it is written, so the format cannot drift. The cooler is kept as its meter reads it (×10000 for the platform), as the collinear DAQ kept it.

One key more, `ORACLE`, holds ORACLE's own record of the run; readers that look keys up by name pass over it. It is written with the header, before the first event, from settings held for the whole scan, and File ▸ Open run restores the window from it.

**How the run was taken**

| Key | Contents |
|---|---|
| `Version`, `Commit` | ORACLE's version; the git commit it ran from and whether its files differed from it (`Hash`, `Dirty`), empty outside a checkout |
| `ScanType`, `Rounds`, `Queue` | Sawtooth, Triangle or Single scan; the number of scans asked for; its place in the scan queue |
| `Timing` | the trigger timing sent to the rack: `bunched`, `rep_rate_hz`, `release_us`, `tagger_delay_us`, `gate_start_us`, `gate_stop_us` |
| `View` | the plots' TOF bin, gate, channels and map, held for the whole scan |
| `Calibration` | the gate's verdict, each check's verdict and message, and the numbers it decided on |
| `SettleModel`, `Laser`, `Channels` | the settle model in force; the laser's harmonic, geometry and charge; the wired tagger inputs |
| `CoolerRollingS` | the window, in seconds, of the cooler's rolling mean the run is analysed with |
| `DeadTimeUs` | the dead time, in µs, the run's PMTs are judged at for afterpulsing |
| `CalibrationReadings` | the power line cycles each calibration reading integrated (`Nplc`) and how long the calibration took (`DurationS`) |

**What the rack was and what it held**

| Key | Contents |
|---|---|
| `Rack` | `simulated` or `hardware` |
| `RackIdentity` | both meters' `*IDN?`; the DAC board's name and unique ID; the tagger's wrapper version; the amplifier's model, serial, gain and wiring |
| `Dividers` | the scan and cooler dividers as set: `model`, `serial`, `ratio`, `ppm`. Readers scale by these, never by today's settings |
| `DacCodes` | the converter (`Channel`) and `Code` each setpoint was written as, in the order of `CalSet`, so the voltage the DAC held is exact |
| `DacCorrectionFull` | the DAC correction in force, in full; the header's `DacCorrection` is its summary |
| `TaggerLevels` | each input's threshold in volts (negative for a falling edge) and the trigger's |

**What the operator gave**

| Key | Contents |
|---|---|
| `InteractionRegion` | `CEC` or `LCR`; empty if not stated |
| `ScanTable` | the ID of the bench table in force when the run was taken, or empty |
| `TableFit` | the calibration fitted to that table, V = a + b·T: `A_V`, `A_Err_V`, `B`, `B_Err`, `Cov_AB`, `Rms_V`, `Expected_V`, `Chi2`, `Points`, `Interpolated`, `Verdict`; or the sentence saying why it was not fitted |
| `ReferenceRun` | whether the run was taken to measure the beam's offset from the cooler's readback |
| `PmtHv` | the PMTs' high voltage: `Pvs`, read from EPICS where their names are set, and `Entered`, as typed on the Scan tab |

`bunch_number` is the loop's count of bunches taken in the scan, from 0, exactly `BunchesPerChannel` a step: what DENIS assumes when it numbers scans (`bunch // bunches per scan`), now true by construction even when steps pause to settle. The tagger's own count is checked against it live, and a missed trigger is counted.

Beside each run, `run_NNNN_monitor.csv` records what the rack did while it was taken, ten times a second and written once a second: the setpoint, the line meter's live readback, the cooler, the wavemeter, and the loop's counts of bunches taken, dropped while settling, and missed. It is a separate file, so nothing that reads run files sees a change.

A run's number is one more than the highest in the data folder, counted when it starts: numbers survive a restart, and a refused calibration leaves no gap. On the rack an unreachable data folder stops the run rather than restart the numbering.

---

## Formats proposed for cls_tools

For A. Raggio to review (the November 2026 meeting's Q3); ORACLE adapts to whatever is agreed.

**Bench table**, `<data>/tables/table_<YYYYMMDD>_<id8>.csv`: one row per DAC converter and code, `channel, code, setpoint_v, volts, sigma_v, readings`. `channel` is the converter (0 drives the positive branch, 1 the negative); `code` the 16-bit code it held; `volts` the platform voltage it gave relative to the 0 V level at the time, to the microvolt; `sigma_v` its standard error; `setpoint_v` the setpoint that first wrote the code, for a per-setpoint view. Above the columns, `# key: value` lines (each value JSON) give the table's `id` (the SHA-256 of every number in the file, so a changed file is found), when it was measured, the record it came from, the rack it was measured on, the dividers, the 0 V level's drift during it and the 0 V check interval it recommends. `pandas.read_csv(path, comment='#')` reads the rows. A run's codes are in its `ORACLE` record (`DacCodes`, in the order of `CalSet`), and the table it was taken under is `ScanTable`, so a run's calibration can be fitted to its own table as V = a + b·T(code); ORACLE's own fit is `TableFit`, with a and b and their errors and covariance.

## The time tagger's wrapper

The compiled wrapper the collinear DAQ used to read the TimeTagger4 has four faults no Python can mend: inputs 3 and 4 take input 2's edge setting (with the rack's thresholds the UV PMTs are timed on their falling edge), rollover marks are recorded as hits, a card that fails to open is only printed about, and a trigger with no hits leaves no row, so a read of several such triggers cannot say how many it held. `cls_acquisition/hardware/timetagger4_wrapper/` holds the corrected source and its build script; build it on the rack PC and point `instruments: tagger_driver` at that folder (its README says how). ORACLE recognises it, and warns at start-up while the old one is loaded.

---

## Repository Layout

```
ORACLE/
|-- gui.py                   # entry point
|-- pyproject.toml, uv.lock  # uv project + pinned dependencies
|-- install.bat, install.sh  # installers (full | rack | uv | shortcut)
|-- icons/                   # app icon, logo, Lucide icons (ISC)
|-- cls_acquisition/         # the core, no Qt: scan plan, calibration and gate,
|   |                        #   hardware calibrations, scan performance,
|   |                        #   acquisition process, run file,
|   |                        #   ELOG/EPICS/cooler PVs, settings
|   |-- hardware/            # VISA session, Keysight meters, MCC DAC, PulseBlaster,
|       |                    #   time tagger, and the simulated rack
|       |-- timetagger4_wrapper/  # the corrected tagger wrapper (C++) and its build
|-- gui/                     # PySide6: main window, the five tabs, shared widgets,
|                            #   theme, plots, log, the memory of the inputs
|-- settings/                # settings.yaml (yours) and inputs.yaml (the window's), untracked
|-- tests/                   # headless pytest suite (offscreen Qt, simulated rack)
|   |-- data/bench/          # the rack's bench records and the bench tool's results
|-- .github/workflows/       # the suite on Windows and macOS
```

---

## Development

```bash
uv sync
uv run pytest            # about 100 s: no hardware and no window needed
```

The suite (about 240 tests) runs headless on Windows and macOS alike, and `.github/workflows/tests.yml` runs it on both. It keeps, as DENIS's does, a guard that fails the run if a test touched the real settings file or left run files or logs in the repository.

| Tests | What they hold ORACLE to |
|---|---|
| `test_window_wiring.py`, `test_main_window.py` | every control of the window, end to end through a real acquisition process: scans of each type, DC, the ELOG panel, quick tune, triggers, stops, numbering, the hardware calibrations and what they put in force, and what the operator is told when a timing cannot run, a file cannot be written, the acquisition process dies or the window is closed at the wrong moment |
| `test_bench_data.py` | the rack's own bench records (`tests/data/bench`): ORACLE's analyses give the bench tool's answers on them, its converter codes are the board's at all 13,517 points, the correction's table holds from one day to the next, and the simulated rack is the rack they describe |
| `test_parity_original.py` | the programs ORACLE replaces, run from ORIGINAL_DAQ_STUFF (skipped where it is absent): the same plans, trigger patterns, PulseBlaster program, tagger keywords, meter set-up, header keys, facility names and window defaults as the collinear DAQ, the same analyses as the bench tool, the same Doppler shift as the data viewer; and its faults (D6) gone |
| `test_platform.py`, `test_startup.py` | both systems: one clock across processes, keeping the PC awake, vendor libraries a system lacks, settings with Windows folders, line endings, folders with spaces and accents, a core that loads neither Qt nor a vendor library; `main()` starting and refusing to start; the session log; one version everywhere; the house style |
| the rest | each module on its own: the acquisition loop, calibration and gate, drivers on stand-ins for the vendor libraries, meters, run file, scan performance, settings, facility services, live data and plots |

---

## Open questions and known limits

- **SETTLE on the beamline.** The bench measured the amplifier's settling with the rack's converters, amplifier and divider (24 October 2026, up to 2 kV jumps). The Settle time page should repeat it on the beamline across the jumps scans make, including the largest range join, and its settle model be put into force; until then every step waits SETTLE.
- **The settle test's lag is approximate.** Times within a step response are on the meter's clock; the lag from the command to the 10 % crossing assumes the meter's first sample is taken as INIT goes out, and is counted as zero where it comes out below.
- **Clocks across processes.** The write's stamp and the reading's stamp come from two processes. A stamp earlier than its write aborts the calibration; the other direction (the reading's clock ahead) would make a reading look later than it was. The suite checks that a spawned process reads the same `perf_counter` on whichever system it runs on; on the rack PC itself that is still to be seen.
- **The simulated step is linear.** It rises in 0.4 ms whatever the jump, as the bench's 1 kV steps did; the amplifier slews a 2 kV jump in about 1 ms. Its settling matches the bench at both sizes, which is what the settle model uses.
- **The installers on Windows.** `install.bat` now finds the Desktop as Windows has it (OneDrive's, when its folder backup is on) and keeps a `.venv` under OneDrive on the device; it has been read, not yet run, on Windows.
- **The meter's trigger delay.** The collinear DAQ set `TRIG:DEL 0.0099` for its bursts; why is unknown. Stepping uses no delay, and SETTLE covers more than 9.9 ms.
- **The buncher during calibration.** PulseBlaster channel 17 triggers the buncher, which is not pulsed while calibrating. Calibration is treated as a beam-off task.
- **The DAC's resolution.** The USB-231 is 16 bits across ±10 V: 0.305 V a code at the platform. A setpoint between codes is written as the nearer, so a 1 V step carries up to ±0.15 V (0.09 V rms) the correction cannot remove; each calibration reading holds it, so per-point calibration in the analysis sees it, a straight line does not. A step that is a whole number of codes (0.30518 V × n) has none.
- **The DAC correction drifts.** The branches' offsets move by up to 0.1 V a day; the non-linearity's shape held to 0.017 V over a day. Keep the correction's validity short, and re-sweep when it lapses.
- **The settle gate is not wired yet.** Until an AND unit puts one MCC digital line in series with PulseBlaster channel 18 (and `settle_gate_bit` names it), settling bunches are dropped in software, which a bunch read later than its trigger by more than the settle margin can slip past; Scan Performance reports the latency. That the USB-231's AUXPORT bits are individually configurable as outputs is from its documentation, not yet tried on the rack.
- **Settle-aware stepping cannot see the voltage, only time.** A hardware-clocked DAC, or tagger timestamps on the DAC writes (a digital line into a spare tagger input), would tie the voltage axis to the tagger's clock outright.
- **The corrected tagger wrapper** compiles cleanly against the driver's headers but has not been built or run on the rack.
- **Memory.** The live view keeps every hit of the current run in memory, as the collinear DAQ did: the four columns its plots read (voltage, bunch, channel, time of flight), each as the narrowest whole-number codes that give its values back exactly (a time of flight in tagger bins, a voltage as which setpoint it is), 11 bytes a hit, so about 40 MB an hour at 1000 hits a second. A column its codes cannot hold exactly (an old run's times, say) is kept as float64, 8 bytes, instead. The run file holds every column.

---

## License

ORACLE is released under the [MIT License](LICENSE). The theme is carried over from DENIS (A. Kayaalp, MIT); icons are from [Lucide](https://lucide.dev/) (ISC). The hardware calibrations follow the bench calibration tool cls-cal (A. Raggio). Finished runs are viewed and fitted in DENIS.
