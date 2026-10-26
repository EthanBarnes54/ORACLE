# The corrected TimeTagger4 wrapper

`timetagger4.cpp` is the old DAQ's tagger wrapper (`collinear_daq_driver_wrap.cpp`) with the same Python interface and three faults mended:

| In the old wrapper | Here |
|---|---|
| Inputs 3 and 4 took input 2's edge setting (`channel_1_rising`), so with the rack's thresholds the UV PMTs were timed on their falling edge | Every input takes its own edge |
| A hit's flags were never read, so the rollover marks the card sends past the 24-bit timestamp range (8.39 ms) were recorded as hits | Rollover marks are skipped; ORACLE also refuses any window longer than the range |
| A card that failed to open, configure or start printed to the console, and the next call used a dead handle | Each failure raises a RuntimeError naming the step and the driver's message |

It also carries `ORACLE_WRAPPER`, which ORACLE checks at start-up: with the old wrapper loaded, a window warning names the inputs timed on the wrong edge.

## Building it on the rack PC

It needs Visual Studio's C++ build tools and the cronologic TimeTagger4 driver (installed with the card). From this folder:

```bat
uv run --with pybind11 --with setuptools python build_wrapper.py
```

Give `--driver <folder>` if the driver is not in `C:\Program Files\cronologic\TimeTagger4\driver`. The build leaves `timetagger4.<tag>.pyd` and a copy of `xtdc4_driver_64.dll` here (both untracked). Then set, in `settings/settings.yaml`:

```yaml
instruments:
  tagger_driver: <this folder>
```

The source compiles cleanly with `clang++ -std=c++17 -Wall -Wextra` against the driver's headers and pybind11; it has not yet been built or run on the rack.
