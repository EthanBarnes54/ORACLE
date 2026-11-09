"""ORACLE against the programs it replaces: the collinear DAQ, the bench tool and the data viewer.

Where ORACLE keeps what the original did, it must do it exactly so; where
it mends a fault, it must differ only there. Each original is read from
ORIGINAL_DAQ_STUFF, and run as it stands: a module that imports cleanly
(the bench tool's analysis) is imported, and from one that cannot be
(it loads PyQt5, matplotlib or a vendor DLL as it is imported) the
function under test is compiled from its own source, so what runs is
the original's code, not a copy of it. The tests skip where that folder
is absent.
"""

import ast
import os
import sys
import types
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cls_acquisition import hardware_calibration as hc
from cls_acquisition import run_file, scan_performance, settings, triggers
from cls_acquisition.acquisition import SCAN_TYPES
from cls_acquisition.hardware import keysight, mcc_dac
from cls_acquisition.hardware.mcc_dac import CH_NEG, CH_POS
from cls_acquisition.scan_plan import Scan_Plan

ORIGINAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'ORIGINAL_DAQ_STUFF')
DAQ = os.path.join(ORIGINAL, 'collinear_daq')
CLS_CAL = os.path.join(ORIGINAL, 'cls_calibration_tests', 'src')
VIEWER = os.path.join(ORIGINAL, 'collinear_dataviewer')

pytestmark = pytest.mark.skipif(not os.path.isdir(ORIGINAL),
                                reason="ORIGINAL_DAQ_STUFF is not in this checkout")


def _Tree(*parts):
    with open(os.path.join(*parts), encoding='utf-8') as fh:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)     # its Windows paths in strings
            return ast.parse(fh.read())


def _Node(tree, name, kind=ast.FunctionDef):
    return next(n for n in ast.walk(tree) if isinstance(n, kind) and n.name == name)


def _Original(tree, name, **namespace):
    """The original's function ``name``, compiled from its own source and run
    with ``namespace`` as its globals."""
    module = ast.Module(body=[_Node(tree, name)], type_ignores=[])
    namespace.setdefault('np', np)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        exec(compile(module, f"<original {name}>", 'exec'), namespace)
    return namespace[name]


def _Strings(node, method):
    """The constant first arguments of every ``.method(...)`` call in ``node``,
    in order."""
    return [call.args[0].value for call in ast.walk(node)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == method and call.args
            and isinstance(call.args[0], ast.Constant)]


@pytest.fixture(scope='module')
def cls_cal():
    sys.path.insert(0, CLS_CAL)
    try:
        from cls_cal.hardware import daq, dmm
        from cls_cal.routines import analysis, linearity, settle
        yield types.SimpleNamespace(analysis=analysis, linearity=linearity, settle=settle,
                                    daq=daq, dmm=dmm)
    finally:
        sys.path.remove(CLS_CAL)


# ── The collinear DAQ ───────────────────────────────────────────

@pytest.fixture(scope='module')
def daq_tree():
    return _Tree(DAQ, 'collinear_daq.py')


@pytest.mark.parametrize('starts, stops, step', [([0], [100], 1), ([-50], [50], 0.5),
                                                  ([0], [100], 2.5), ([-1000], [1000], 10)])
def test_A_Plain_Scan_Commands_What_The_Old_Daq_Commanded(daq_tree, starts, stops, step):
    make_scan, make_bins = (_Original(daq_tree, name) for name in ('make_scan', 'make_bins'))
    plan = Scan_Plan(starts, stops, step)
    assert plan.Setpoints == pytest.approx(make_scan(starts, stops, step), abs=1e-9)
    # one bin a step, centred on it, as the old DAQ's spectrum had them
    assert plan.Bin_Edges() == pytest.approx(make_bins(starts, stops, step), abs=1e-9)


def test_The_Old_Daqs_Extra_Points_And_Repeated_Endpoints_Are_Gone(daq_tree):
    """D6: the beamline's ramp, where the old arange laid down 464 points for
    463, and two ranges sharing an endpoint, which it commanded twice."""
    make_scan = _Original(daq_tree, 'make_scan')
    assert len(make_scan([-450], [475], 2)) == 464
    assert len(Scan_Plan([-450], [475], 2)) == 463
    old = make_scan([0, 10], [10, 20], 1)
    assert list(old).count(10) == 2
    assert Scan_Plan([0, 10], [10, 20], 1).Setpoints == pytest.approx(
        np.unique(np.round(old, 9)))


def test_The_Run_Header_Keeps_The_Old_Daqs_Keys_In_Its_Order(daq_tree):
    toggle = _Node(daq_tree, 'toggleScan')
    recorder = next(node.value for node in ast.walk(toggle) if isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Attribute)
                    and node.targets[0].attr == 'recorder')
    keys = [key.value for key in recorder.keys]
    assert keys[:-2] == list(run_file.HEADER_KEYS[:len(keys) - 2])
    assert keys[-2:] == ['raw_header', 'raw']
    # the two ORACLE adds say which DAC correction was in force, and come last
    assert run_file.HEADER_KEYS[len(keys) - 2:] == ('DacCorrectionApplied', 'DacCorrection')
    columns = next(node.value for node in daq_tree.body if isinstance(node, ast.Assign)
                   and node.targets[0].id == 'DATA_COLS')
    assert ast.literal_eval(columns) == run_file.DATA_COLS
    assert recorder.values[0].value == run_file.SORTING_UTILITIES


def test_The_Facility_Names_Are_The_Old_Daqs(daq_tree):
    pvs = set(_Strings(daq_tree, 'caget')) | {
        call.args[0].value for call in ast.walk(daq_tree) if isinstance(call, ast.Call)
        and getattr(call.func, 'id', None) == 'caget'}
    # the PMTs' high voltage is not on EPICS until named, and the old DAQ never read it
    assert {name for name in settings.DEFAULTS['pvs'].values() if name} == pvs
    assert _Strings(daq_tree, 'open') == [settings.DEFAULTS['elog']['url']]
    # the ELOG entry is started as the old DAQ started it
    post = next(call for call in ast.walk(_Node(daq_tree, 'ResetElogMessage'))
                if isinstance(call, ast.Call) and getattr(call.func, 'attr', '') == 'post')
    assert {k.arg for k in post.keywords} == {'Author', 'What', 'Experiment'}
    assert next(k.value.value for k in post.keywords if k.arg == 'What') == 'Measurement'


def test_The_Meters_Are_Set_Up_As_The_Old_Daq_Set_Them():
    tree = _Tree(DAQ, 'util', 'keysight_voltmeter.py')
    line = _Node(_Node(tree, 'line_meter', ast.ClassDef), '__init__')
    cooler = _Node(_Node(tree, 'cooler_meter_loop', ast.ClassDef), '__init__')
    assert tuple(_Strings(line, 'write')) == keysight.LINE_CONFIG
    # the old cooler loop's, then one sample a trigger, which its CONF had set anyway
    old = tuple(_Strings(cooler, 'write'))
    assert keysight.COOLER_CONFIG == old + ('SAMP:COUN 1',)
    instruments = settings.DEFAULTS['instruments']
    assert _Strings(line, 'open_resource') == [instruments['line_meter']]
    assert _Strings(cooler, 'open_resource') == [instruments['cooler_meter']]


def test_The_Tagger_Is_Opened_As_The_Old_Daq_Opened_It():
    """The keywords the compiled wrapper takes, and their values for the
    rack's thresholds and a 0 to 200 us window, exactly as the old Tagger
    passed them."""
    loop = _Tree(DAQ, 'util', 'collinear_loop.py')
    setup = _Node(loop, 'setup_tagger')
    levels = ast.literal_eval(next(node.value for node in ast.walk(setup)
                                   if isinstance(node, ast.Assign)
                                   and getattr(node.targets[0], 'id', '') == 'channels'))
    instruments = settings.DEFAULTS['instruments']
    assert [levels[i] for i in range(4)] == instruments['tagger_levels']
    assert _Strings(setup, 'set_trigger_level') == [instruments['tagger_trigger_level']]

    old_kwargs = {}
    init_card = _Original(_Tree(DAQ, 'util', 'timetagger', 'tagger.py'), 'init_card',
                          tg=lambda **kwargs: old_kwargs.update(kwargs))
    init_card(types.SimpleNamespace(
        trigger_level=2.5, trigger_type=True, channels=[True] * 4,
        levels=[levels[i] for i in range(4)], type=[levels[i] > 0 for i in range(4)],
        starts=[0] * 4, stops=[200 * 2000] * 4, card=None))

    ours = {}
    stub = types.ModuleType('timetagger4')
    stub.TimeTagger = type('Card', (), {'__init__': lambda self, **kwargs: ours.update(kwargs),
                                        'stop': lambda self: None})
    sys.modules['timetagger4'] = stub
    try:
        from cls_acquisition.hardware.timetagger import Time_Tagger
        tagger = Time_Tagger(instruments['tagger_levels'], instruments['tagger_trigger_level'])
        ours.clear()
        tagger.Configure(0.0, 200.0)
        tagger._Open()
    finally:
        del sys.modules['timetagger4']
    assert ours == old_kwargs


def test_The_Trigger_Patterns_Are_The_Old_Daqs():
    tree = _Tree(DAQ, 'util', 'blaster.py')
    programmed = []
    blaster = types.SimpleNamespace(setPattern=programmed.append)
    bunched = _Original(tree, 'set_triggers_bunched', blaster=blaster)
    dc = _Original(tree, 'set_triggers_DC', blaster=blaster)
    for rep_rate, release, delay in ((10.0, 100, 0), (20.0, 250, 30), (1.0, 5000, 120)):
        bunched({'rep_rate': rep_rate, 'buncher_sig_length': release,
                 'card_trigger_offset': delay})
        assert sorted(triggers.Bunched_Pattern(rep_rate, release, delay)) == sorted(
            programmed.pop())
    dc()
    assert sorted(triggers.Dc_Pattern()) == sorted(programmed.pop())


def test_The_PulseBlaster_Is_Programmed_As_The_Old_Daq_Programmed_It(monkeypatch):
    tree = _Tree(DAQ, 'util', 'pulseblaster.py')
    old_calls, new_calls = [], []

    def recorder(calls):
        lib = types.SimpleNamespace()
        for name in ('pb_stop', 'pb_start_programming', 'pb_stop_programming', 'pb_start',
                     'pb_select_board', 'pb_core_clock', 'pb_close'):
            setattr(lib, name, lambda *args, name=name: calls.append((name, *args)))
        lib.pb_init = lambda: 0
        lib.pb_inst_pbonly = lambda *args: calls.append(('pb_inst_pbonly', *args)) or 0
        return lib

    transform = _Original(tree, 'transformLinesToTimeSlices')
    program = _Original(tree, 'programTimeSlices',
                        spinapi=types.SimpleNamespace(spinapi=recorder(old_calls)),
                        print=lambda *_: None)
    old_self = types.SimpleNamespace(CONTINUE=0, BRANCH=6, PULSE_PROGRAM=0)
    pattern = triggers.Bunched_Pattern(10.0, 100, 20)
    old_slices = transform(None, pattern)
    assert [s[:2] for s in old_slices] == triggers.Time_Slices(pattern)
    program(old_self, old_slices)

    package = types.ModuleType('spinapi')
    package.spinapi = recorder(new_calls)
    monkeypatch.setitem(sys.modules, 'spinapi', package)
    monkeypatch.setitem(sys.modules, 'spinapi.spinapi', package.spinapi)
    from cls_acquisition.hardware.pulseblaster import Pulse_Blaster
    blaster = Pulse_Blaster()
    new_calls.clear()
    blaster.Program(pattern)
    assert new_calls == old_calls


def test_The_Collinear_Daqs_Window_Defaults_Are_Kept(qapp):
    """Control for control, the old window's starting values and ranges."""
    from gui.calibration_tab import Calibration_Tab
    from gui.main_window import SLIDER_STEPS
    from gui.scan_tab import RANGES, Scan_Tab
    from gui.time_tab import BIN_SECONDS, Time_Tab
    ui = ET.parse(os.path.join(DAQ, 'UI_files', 'collinear_daq_elog.ui'))

    def widget(name):
        node = next(w for w in ui.iter('widget') if w.get('name') == name)
        props = {}
        for prop in node.findall('property'):
            value = list(prop)[0]
            props[prop.get('name')] = value.text
        return props, [''.join(item.itertext()).strip() for item in node.findall('item')]

    def number(name, key='value', default=0.0):
        return float(widget(name)[0].get(key) or default)

    limit = settings.DEFAULTS['instruments']['hv_limit_v']
    scan, calibration, time = Scan_Tab(limit), Calibration_Tab(limit), Time_Tab()
    assert widget('scantypeBox')[1] == list(SCAN_TYPES)
    assert len(scan.Range_Checks) == RANGES == 7
    assert [c.isChecked() for c in scan.Range_Checks] == [
        widget(f'scanCheck{i}')[0].get('checked') == 'true' for i in range(1, 8)]
    for i in range(7):
        for box, name in ((scan.From_Boxes[i], f'fromSpinBox{i + 1}'),
                          (scan.To_Boxes[i], f'toSpinBox{i + 1}')):
            assert box.value() == number(name)
            assert (box.minimum(), box.maximum()) == (number(name, 'minimum'),
                                                      number(name, 'maximum'))
    assert scan.Step_Box.value() == number('stepsizeBox')
    assert scan.Step_Box.minimum() == number('stepsizeBox', 'minimum')
    assert (scan.Dwell_Box.value(), scan.Dwell_Box.minimum(), scan.Dwell_Box.maximum()) == (
        number('dwellBox'), number('dwellBox', 'minimum'), number('dwellBox', 'maximum'))
    assert scan.Release_Box.value() == number('coolerLength')
    assert scan.Tagger_Delay_Box.value() == number('DAQTriggerDelay')
    assert scan.Gate_Start_Box.value() == number('tagStartBox')
    assert scan.Gate_Length_Box.value() == number('tagStopBox')
    assert scan.Bunched() and scan.Per_Step_Box.value() == 1     # buncherDCChanged's value
    assert scan.Per_Step_Box.maximum() == number('bunchPerStepBox', 'maximum')
    assert [c.isChecked() for c in scan.Channel_Checks] == [True] * 4
    assert (calibration.From_Box.value(), calibration.To_Box.value(),
            calibration.Step_Box.value()) == (number('fromCalSpinBox'), number('toCalSpinBox'),
                                              number('stepsizeCalBox'))
    assert list(BIN_SECONDS) == widget('timeCombo')[1]
    assert [c.isChecked() for c in time.Channel_Checks] == [
        widget(f'ch{c}_2')[0].get('checked') == 'true' for c in (1, 2, 3, 4)]
    assert SLIDER_STEPS == number('voltSlider', 'maximum')
    for box in (scan, calibration, time):
        box.deleteLater()


# ── The bench tool ──────────────────────────────────────────────

@pytest.mark.parametrize('order, direction', [('up', hc.DIRECTIONS[0]),
                                              ('updown', hc.DIRECTIONS[2]),
                                              ('random', hc.DIRECTIONS[3])])
def test_A_Linearity_Sweep_Visits_The_Bench_Tools_Points(cls_cal, order, direction):
    config = cls_cal.linearity.LinearityConfig(min_kv=-1.0, max_kv=1.0, step_kv=0.1,
                                               fine_span_kv=0.02, fine_step_kv=0.005,
                                               order=order, zero_check_every=7, seed=11)
    theirs = [(p['kind'], p['direction'], p['set_kv'] * 1e3)
              for p in cls_cal.linearity.make_points(config)]
    # a ramp step wider than the sweep, so every write is a measured point
    plan = hc.Linearity_Plan(-1000.0, 1000.0, 100.0, direction, 1e4, 20.0, 5.0, 7, 11)
    ours = [plan.Marks[i] for i in sorted(plan.Marks)]
    assert [(k, w) for k, w, _ in ours] == [(k, w) for k, w, _ in theirs]
    assert [v for _, _, v in ours] == pytest.approx([v for _, _, v in theirs], abs=1e-6)


def test_The_Linearity_Analysis_Is_The_Bench_Tools(cls_cal):
    rng = np.random.default_rng(3)
    set_kv = np.round(np.linspace(-1.5, 1.5, 61), 9)
    sweep = np.concatenate([set_kv, set_kv[-2::-1]])
    ways = ['up'] * len(set_kv) + ['down'] * (len(set_kv) - 1)
    # each branch its own gain and offset, a bow, hysteresis and noise
    read = np.where(sweep > 0, 0.9992 * sweep - 0.0010, 0.9988 * sweep - 0.0006)
    read = read + 4e-5 * sweep ** 2 + np.where(np.array(ways) == 'down', 2e-5, 0.0)
    read = read + rng.normal(0, 3e-5, len(sweep))
    theirs_points = [{'kind': 'sweep', 'direction': w, 'set_kv': s, 'mean_kv': r, 't_s': 10 * i}
                     for i, (s, r, w) in enumerate(zip(sweep, read, ways))]
    theirs_points += [{'kind': 'zero', 'direction': '', 'set_kv': 0.0,
                       'mean_kv': -0.0008 + 1e-6 * i, 't_s': 100.0 * i} for i in range(5)]
    theirs = cls_cal.analysis.analyze_linearity(theirs_points)
    ours = hc.Linearity_Analysis([
        {'Kind': p['kind'], 'Direction': p['direction'], 'Set_V': p['set_kv'] * 1e3,
         'Read_V': p['mean_kv'] * 1e3, 'Sigma_V': 0.0,
         'Channel': CH_POS if p['set_kv'] >= 0 else CH_NEG, 'Dac_V': 0.0, 'Time_S': p['t_s']}
        for p in theirs_points])
    assert ours['Gain'] == pytest.approx(theirs['gain'], rel=1e-12)
    assert ours['Offset_V'] == pytest.approx(theirs['offset_v'], rel=1e-9)
    assert ours['Residual_Rms_V'] == pytest.approx(theirs['residual_rms_v'], rel=1e-9)
    assert ours['Quadratic_V_Per_Kv2'] == pytest.approx(theirs['quadratic_v_per_kv2'], rel=1e-9)
    assert ours['Seam_V'] == pytest.approx(theirs['branches']['zero_jump_v'], rel=1e-9)
    for channel, name in hc.BRANCHES:
        assert ours['Fits'][channel]['Gain'] == pytest.approx(theirs['branches'][name]['gain'],
                                                              rel=1e-12)
    assert ours['Hysteresis']['Max_V'] == pytest.approx(theirs['hysteresis']['max_abs_v'],
                                                        rel=1e-9)
    assert ours['Zero_Drift']['Slope_V_Per_H'] == pytest.approx(
        theirs['zero_drift']['slope_v_per_h'], rel=1e-9)


def test_The_Noise_Analysis_Is_The_Bench_Tools(cls_cal):
    """The bench tool's own test signal: a slow oscillation, the mains, a
    drift and white noise."""
    rng = np.random.default_rng(1)
    t = np.arange(60000) * 1e-3
    volts = (30e3 + 0.01 * t + 0.8 * np.sin(2 * np.pi * 2.7 * t + 1)
             + 0.4 * np.sin(2 * np.pi * 50.03 * t) + rng.normal(0, 0.2, len(t)))
    theirs = cls_cal.analysis.analyze_noise(volts, 1e-3, integration_s=200e-6)
    ours = hc.Noise_Analysis(volts, 1e-3, 200e-6)
    for key, name in (('Rms_Detrended_V', 'rms_detrended_v'),
                      ('Random_Rms_V', 'random_noise_rms_v'),
                      ('Noise_Floor_V_Rt_Hz', 'noise_floor_v_per_rthz')):
        assert ours[key] == pytest.approx(theirs[name], rel=1e-12)
    assert ours['Drift_V_Per_Min'] == pytest.approx(theirs['drift_v_per_min'], rel=1e-6)
    assert [(p['Freq_Hz'], p['Amplitude_V'], p['Mains_Harmonic']) for p in ours['Peaks']] == [
        (pytest.approx(p['freq_hz'], rel=1e-12), pytest.approx(p['amplitude_v'], rel=1e-12),
         p['mains_harmonic']) for p in theirs['peaks']]


def test_A_Step_Is_Timed_As_The_Bench_Tool_Times_It(cls_cal):
    t = np.arange(0, 0.5, 1e-4)
    tau, at = 5e-3, 0.05
    kv = np.where(t < at, 0.0, 1.0 - np.exp(-np.clip(t - at, 0, None) / tau))
    theirs = cls_cal.analysis.analyze_step(t, kv, 0, 1, baseline_s=0.04, tolerance_kv=1e-3)
    ours = hc.Step_Response(t, kv * 1e3, 1.0)
    assert ours['T10_S'] == pytest.approx(theirs.t10_s, abs=1e-9)
    assert ours['Rise_S'] == pytest.approx(theirs.rise_10_90_s, abs=1e-9)
    assert ours['Settle_S'] == pytest.approx(theirs.settle_s, abs=1e-4)


def test_The_Recommended_Wait_Is_The_Bench_Tools(cls_cal):
    rows = [{'step': s, 'mode': 'fast', 'from_kv': 0.0, 'to_kv': 1.0, 'settle_s': settle,
             'rise_10_90_s': 4e-4, 'overshoot_pct': 0.2}
            for s, settle in ((0, 0.016), (1, 0.021), (2, 0.11))]
    theirs = cls_cal.settle.summarize(rows, 0.5)['recommended_wait_s']
    ours = hc.Recommended_Wait([{'Lag_S': 0.0, 'Settle_S': r['settle_s']} for r in rows])
    assert ours == pytest.approx(theirs)
    assert hc.Recommended_Wait([{'Lag_S': 0.0, 'Settle_S': 0.01}]) == pytest.approx(
        cls_cal.settle.summarize(rows[:1], 0.5)['recommended_wait_s'])


def test_Readings_Are_Parsed_And_Refused_As_The_Bench_Tool_Does(cls_cal):
    dmm = cls_cal.dmm
    for reply in ('+1.0E+00,+2.5E-03,-7.0E-01', '#226+1.0E+00,+2.5E-03,-7.0E-01'):
        assert keysight.Readings(reply, 'SIM', 'a test') == pytest.approx(
            dmm.parse_readings(dmm.strip_block_header(reply), 'R?'))
    for bad in ('+1.0E+00,+9.9E+37', '+9.91E+37'):
        with pytest.raises(dmm.DmmError):
            dmm.check_readings(dmm.parse_readings(bad, 'FETC?'), 10.0, 1000.0)
        with pytest.raises(keysight.Scpi_Error):
            keysight.Readings(bad, 'SIM', 'a test')


def test_The_Converters_Are_Driven_As_The_Bench_Tool_Drives_Them(cls_cal):
    """Positive setpoints on the positive converter, negative on the other,
    the outgoing converter released first; ORACLE's land on the board's codes."""
    config = cls_cal.daq.DaqConfig(mode='split', max_abs_kv=4.0)

    class Recorder(cls_cal.daq.HvOutput):
        def __init__(self):
            super().__init__(config)
            self.Writes = []

        def _write_channel(self, channel, volts):
            self.Writes.append((channel, volts))
            return 0

    theirs, ours = Recorder(), []
    dac = mcc_dac.Dac(lambda channel, volts: ours.append((channel, volts)))
    for volts in (500.0, -250.0, 3000.0, -3999.0, 1.0):
        theirs.Writes.clear()
        ours.clear()
        theirs.set_kv(volts / 1e3)
        dac.Write(volts)
        assert [c for c, _ in ours] == [c for c, _ in theirs.Writes]
        assert [v for _, v in ours] == pytest.approx([v for _, v in theirs.Writes],
                                                     abs=mcc_dac.LSB_V / 2)


# ── The data viewer ─────────────────────────────────────────────

def test_The_Frequency_Axis_Is_The_Viewers_Doppler_Shift():
    """Scan Performance puts the laser into the ions' frame as the data viewer
    did with its dopplershift, anti-collinear, at the beam energy the cooler
    less the scan gave."""
    tree = _Tree(VIEWER, 'doppler.py')
    constants = {node.targets[0].id: eval(compile(ast.Expression(node.value), 'doppler',
                                                  'eval'))
                 for node in tree.body if isinstance(node, ast.Assign)}
    factor = _Original(tree, 'dopplerfactor', **constants)
    shift = _Original(tree, 'dopplershift', dopplerfactor=factor)
    required_v = _Original(tree, 'calculate_required_v', **constants)
    laser = {'harmonic': 2, 'geometry': 'anti-collinear', 'charge': 1}
    nu = 16956.17 * scan_performance.C_LIGHT * 100.0
    for beam_v, mass in ((29884.8 - 40.0, 58.93), (30000.0, 131.9), (25000.0, 7.016)):
        ours = scan_performance.Ion_Frequency(beam_v, nu, mass, laser)
        theirs = shift(nu * laser['harmonic'], beam_v, mass, False, False)
        # to the viewer's constants (CODATA 2014 for the mass unit and e)
        assert ours == pytest.approx(theirs, rel=5e-11)
        # and the viewer's inverse puts the model back on the same beam energy, to its
        # mass constant (CODATA 2014 against ORACLE's 2018: 7 parts in 10^9 of m c^2)
        assert required_v(nu * laser['harmonic'], ours, mass) == pytest.approx(beam_v, rel=2e-8)

