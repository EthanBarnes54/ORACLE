"""ORACLE's settings: the rack, the facility services and the data folder.

Date:    2026-11-10
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

``settings/settings.yaml`` (or ``ORACLE_SETTINGS_PATH``) holds what was
set, :data:`DEFAULTS` the rest; the window's inputs live beside it in
``inputs.yaml``. :func:`Update` rewrites only its keys, on a fresh read. A
file that is not a YAML mapping raises Settings_Error with its line (on
Windows, often a quoted folder's backslashes). ``hardware`` holds what the
rack cannot say (dividers, amplifier), recorded with every run, as is
:func:`Commit`.

Depends on: PyYAML; git, if ORACLE runs from a checkout.
"""

import contextlib
import copy
import os
import subprocess
import sys

import yaml

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_ROOT = CODE_ROOT                # where data, logs and settings go; the tests move it

DEFAULTS = {
    # 'simulated' runs the whole DAQ against a software bench; 'hardware'
    # opens the rack's instruments
    'rack': 'simulated',
    'experiment': '',
    # where run files go on the rack; empty means <app>/data. A simulated
    # rack always writes to <app>/data/simulated, never here
    'data_directory': '',
    'elog': {
        # never posts from a simulated rack, whatever this says
        'enabled': False,
        'url': 'http://elog.igisol:8020/Collinear/',
        'author': '',               # empty: the login name
    },
    'pvs': {
        'wavenumber': 'FURIOS:Wavemeter2:Wavenumber1',
        'mass': 'IGISOL:MagnetControl:Mass',
        'mass_setpoint': 'IGISOL:MagnetControl:Target',
        # each tagger input's PMT high voltage, recorded with every run; empty:
        # not on EPICS, and the Scan tab's PMT HV box is recorded instead
        'pmt_hv_1': '', 'pmt_hv_2': '', 'pmt_hv_3': '', 'pmt_hv_4': '',
    },
    'instruments': {
        'line_meter': 'USB0::0x2A8D::0x0101::MY57506790::INSTR',
        'cooler_meter': 'USB0::0x2A8D::0x0101::MY57510428::INSTR',
        'dac_board': 1,
        # the most the HV amplifier may be asked for, by a scan, quick tune or a
        # hardware calibration; at most the Trek 609E-6's rating of 4 kV
        'hv_limit_v': 4000.0,
        # the MCC digital line (AUXPORT bit) that gates the tagger's window
        # through an AND unit on PulseBlaster channel 18; empty: not wired, and
        # the loop discards settling bunches in software instead
        'settle_gate_bit': None,
        'tagger_driver': '',        # folder holding timetagger4 and its DLLs
        'tagger_trigger_level': 2.5,
        # threshold per tagger input in volts; a negative one is a falling edge
        'tagger_levels': [-0.2, -0.6, 0.1, 0.1],
        # the inputs with a detector wired to them; Scan Performance reports one
        # that records nothing as a possible fault
        'tagger_channels': [1, 2, 3, 4],
    },
    # what the rack cannot say of itself, recorded with every run. The meeting of
    # November 2026 heard the two dividers are different models (KV-10 and KV-30)
    # where the slides name the KV-10A for both: set each as its label reads.
    # ppm is the ratio's stated accuracy, for the systematics
    'hardware': {
        'scan_divider': {'model': '', 'serial': '', 'ratio': 1000.0, 'ppm': 100.0},
        'cooler_divider': {'model': '', 'serial': '', 'ratio': 10000.0, 'ppm': 100.0},
        'amplifier': {'model': 'Trek 609E-6', 'serial': ''},
    },
    # each calibration reading integrates this many power line cycles (20 ms each),
    # which takes out the mains on the divider; the poll keeps its 200 us aperture
    'calibration': {'nplc': 1},
    # the cooler as the analysis takes it: the rolling mean of its readings over this
    # window (A. Raggio, November 2026), recorded with each run; Scan Performance
    # judges a run by it and plots it over the readings
    'cooler': {'rolling_s': 15.0},
    # the PMTs as Scan Performance judges them: counts within this dead time of one on
    # the same PMT are tested for afterpulsing, and the dead time DENIS might apply is
    # suggested from them; nothing here applies it
    'detectors': {'dead_time_us': 10.0},
    # the systematics' inputs a run cannot measure (Scan Performance's table):
    # beam_offset_v is the beam's offset from the cooler's readback by interaction
    # region, [offset, its error] in volts from the campaign's reference runs (for
    # example CEC: [13.0, 8.0] for Bi, LCR: [-30.0, 5.0] for Yb+, November 2026);
    # zero_drift_v the 0 V drift assumed until ORACLE checks it during runs; and
    # wavemeter_mhz the wavemeter's accuracy per run at the fundamental
    'systematics': {'beam_offset_v': {}, 'zero_drift_v': 0.030, 'wavemeter_mhz': 10.0},
    # how the laser meets the ions, as DENIS takes it: the Scan Performance tab
    # converts the wavemeter's drift into scan volts with these
    'laser': {
        'harmonic': 4,              # the ions see the wavemeter's value times this
        'geometry': 'anti-collinear',
        'charge': 1,
    },
    # the cooler's voltage published as EPICS PVs, as the old DAQ's server did;
    # needs pcaspy, and never from a simulated rack
    'cooler_pv': {
        'enabled': False,
        'prefix': 'FURIOS:Cooler:',
    },
    # the scan's settling against jump size, from the Settle time page; empty
    # means SETTLE for every step
    'settle_model': None,
    # the DAC correction a linearity sweep measured; empty means none (identity)
    'dac_correction': None,
    # the ID of the bench table runs are fitted to, from the library under
    # <data>/tables (Hardware Calibration, Scan table); empty means none
    'scan_table': None,
}


class Settings_Error(ValueError):
    pass


def Settings_Path():
    override = os.environ.get('ORACLE_SETTINGS_PATH')
    return override or os.path.join(APP_ROOT, 'settings', 'settings.yaml')


def Inputs_Path():
    """Where the window keeps what was last entered in it: beside the settings."""
    return os.path.join(os.path.dirname(Settings_Path()), 'inputs.yaml')


def Last_Rack(kind, rack):
    """The last ``kind`` session's rack, or None; ``rack`` takes its place
    beside the settings. A file that cannot be used is passed over."""
    path = os.path.join(os.path.dirname(Settings_Path()), 'rack.yaml')
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            kept = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        kept = None
    kept = kept if isinstance(kept, dict) else {}
    last = kept.get(kind)
    kept[kind] = rack
    with contextlib.suppress(OSError):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            yaml.safe_dump(kept, fh, default_flow_style=False, sort_keys=False)
    return last if isinstance(last, dict) else None


def _Read():
    path = Settings_Path()
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            stored = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as e:
        raise Settings_Error(f"The settings file {path} is not valid YAML: {e}. A Windows "
                             f"folder must be written bare, in single quotes, or with "
                             f"forward slashes, not in double quotes.") from None
    except UnicodeDecodeError as e:
        raise Settings_Error(f"The settings file {path} is not UTF-8 text ({e}); save it as "
                             f"UTF-8.") from None
    if not isinstance(stored, dict):
        raise Settings_Error(f"The settings file {path} holds a {type(stored).__name__}, not "
                             f"keys and their values.")
    return stored


def _Merged(base, over):
    merged = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _Merged(merged[key], value)
        else:
            merged[key] = value
    return merged


def Load():
    """The settings in force: the file over :data:`DEFAULTS`."""
    return _Merged(DEFAULTS, _Read())


def Update(**values):
    """Set top-level keys in the settings file, leaving the rest as it is."""
    stored = _Read()
    stored.update(values)
    path = Settings_Path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        yaml.safe_dump(stored, fh, default_flow_style=False, sort_keys=False)


def Commit():
    """``{'Hash', 'Dirty'}`` of ORACLE's git commit (Dirty: files differ), or
    None without a checkout or git (the rack PC may have neither)."""
    # no console flashes up on Windows for a git run from a window
    hidden = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}

    def git(*args):
        return subprocess.run(('git', *args), cwd=CODE_ROOT, capture_output=True, text=True,
                              timeout=5.0, check=True, **hidden).stdout.strip()

    try:
        return {'Hash': git('rev-parse', 'HEAD'),
                'Dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}
    except (OSError, subprocess.SubprocessError):
        return None


def Data_Directory(settings):
    """Where run files go. A simulated rack has its own folder, so practice
    runs can never land among, or take numbers from, the real ones."""
    if settings['rack'] != 'hardware':
        return os.path.join(APP_ROOT, 'data', 'simulated')
    return settings['data_directory'] or os.path.join(APP_ROOT, 'data')
