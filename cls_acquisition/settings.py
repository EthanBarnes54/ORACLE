"""ORACLE's settings: the rack, the facility services and the data folder.

Date:    2026-10-30
Version: 0.2.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

``settings/settings.yaml`` holds what has been set; :data:`DEFAULTS`
supplies the rest, so a key added in a later version appears without
anyone editing their file. :func:`Update` rewrites only the keys it is
given, on a fresh read of the file, so it never writes back a stale
copy over an edit made while ORACLE was running.

``ORACLE_SETTINGS_PATH`` overrides the file (the tests point it at a
private copy). What was last entered in the window is kept beside it, in
``inputs.yaml`` (gui.input_memory), so this file stays the operator's.

A file that is not valid YAML, or not a mapping of keys, raises
Settings_Error naming it and the line, rather than starting ORACLE on
defaults the operator did not choose. The likeliest on Windows is a
folder in double quotes, where YAML reads each backslash as an escape.

Depends on: PyYAML.
"""

import copy
import os

import yaml

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
}


class Settings_Error(ValueError):
    pass


def Settings_Path():
    override = os.environ.get('ORACLE_SETTINGS_PATH')
    return override or os.path.join(APP_ROOT, 'settings', 'settings.yaml')


def Inputs_Path():
    """Where the window keeps what was last entered in it: beside the settings."""
    return os.path.join(os.path.dirname(Settings_Path()), 'inputs.yaml')


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


def Data_Directory(settings):
    """Where run files go. A simulated rack has its own folder, so practice
    runs can never land among, or take numbers from, the real ones."""
    if settings['rack'] != 'hardware':
        return os.path.join(APP_ROOT, 'data', 'simulated')
    return settings['data_directory'] or os.path.join(APP_ROOT, 'data')
