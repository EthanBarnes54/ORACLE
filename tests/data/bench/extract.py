"""Make the bench fixtures in this folder from the bench tool's own records.

Date:    2026-11-09
Version: 0.6.0
Author:  Ethan Barnes <ethan.barnes@postgrad.manchester.ac.uk>

The bench calibration tool (cls-cal, A. Raggio) measured the CLS rack on
24 to 26 October 2026 and wrote its records and its own results under
ORIGINAL_DAQ_STUFF/cls_calibration_tests/data. This takes from them only
what the regression tests need, so the suite runs without that folder:
the readings as compact arrays (*.npz), and what cls-cal itself made of
them (reference.json), which ORACLE's analyses must reproduce.

Run from the repository root, once; the fixtures are committed:

    .venv/bin/python tests/data/bench/extract.py

Depends on: numpy.
"""

import csv
import json
import os

import numpy as np

SOURCE = os.path.join('ORIGINAL_DAQ_STUFF', 'cls_calibration_tests', 'data')
HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = {'sweep': 0, 'zero': 1}
DIRECTIONS = {'': 0, 'up': 1, 'down': 2, 'random': 3}


def _Rows(run, name):
    with open(os.path.join(SOURCE, run, name), newline='') as fh:
        return list(csv.DictReader(fh))


def _Json(run, name):
    with open(os.path.join(SOURCE, run, name)) as fh:
        return json.load(fh)


def _Linearity(run, out, sweep_only=False):
    rows = [r for r in _Rows(run, 'points.csv') if not sweep_only or r['kind'] == 'sweep']
    np.savez_compressed(
        os.path.join(HERE, out),
        kind=np.array([KINDS[r['kind']] for r in rows], dtype=np.int8),
        direction=np.array([DIRECTIONS[r['direction']] for r in rows], dtype=np.int8),
        set_v=np.array([float(r['set_kv']) * 1e3 for r in rows]),
        read_v=np.array([float(r['mean_kv']) * 1e3 for r in rows]),
        sigma_v=np.array([float(r['std_kv']) * 1e3 for r in rows]),
        time_s=np.array([float(r['t_s']) for r in rows]),
        # what the board was asked for on each converter, and the code it took
        dac_v=np.array([[float(r['ao0_v']), float(r['ao1_v'])] for r in rows]),
        dac_code=np.array([[int(r['ao0_code']), int(r['ao1_code'])] for r in rows],
                          dtype=np.int32))


def _Zero_Checks(run, out):
    """The 0 V checks a sweep took between its points, which the bench table
    is corrected by: when, and where the platform read."""
    rows = [r for r in _Rows(run, 'points.csv') if r['kind'] == 'zero']
    np.savez_compressed(
        os.path.join(HERE, out),
        time_s=np.array([float(r['t_s']) for r in rows]),
        read_v=np.array([float(r['mean_kv']) * 1e3 for r in rows]),
        sigma_v=np.array([float(r['std_kv']) * 1e3 for r in rows]))


def main():
    reference = {'source': SOURCE, 'tool': 'cls-cal 0.1.0 (A. Raggio)'}

    run = '20261024_184702_linearity'
    _Linearity(run, 'linearity_1kv.npz')
    result = _Json(run, 'results.json')
    reference['linearity_1kv'] = {
        'run': run, 'config': _Json(run, 'meta.json')['config']['linearity'],
        **{key: result[key] for key in ('gain', 'gain_err', 'offset_v', 'offset_err_v',
                                        'residual_rms_v', 'residual_max_abs_v', 'inl_ppm_fs',
                                        'quadratic_v_per_kv2')},
        'branches': result['branches'],
        'hysteresis_max_abs_v': result['hysteresis']['max_abs_v'],
        'zero_drift_span_v': result['zero_drift']['span_v'],
        'zero_drift_slope_v_per_h': result['zero_drift']['slope_v_per_h']}

    _Linearity('20261025_021414_linearity_3kvspan', 'linearity_3kv_25oct.npz', True)
    _Linearity('20261026_111717_linearity', 'linearity_3kv_26oct.npz', True)
    _Zero_Checks('20261025_021414_linearity_3kvspan', 'zero_3kv_25oct.npz')
    _Zero_Checks('20261026_111717_linearity', 'zero_3kv_26oct.npz')

    run = '20261024_142142_settle_LCR'
    steps = _Rows(run, 'steps.csv')
    traces = {}
    for r in _Rows(run, 'traces_fast.csv'):
        traces.setdefault((int(r['step']), int(r['repeat'])), []).append(float(r['hv_kv']) * 1e3)
    keys = [(int(s['step']), int(s['repeat'])) for s in steps]
    np.savez_compressed(
        os.path.join(HERE, 'settle.npz'), hv_v=np.array([traces[key] for key in keys]),
        from_v=np.array([float(s['from_kv']) * 1e3 for s in steps]),
        to_v=np.array([float(s['to_kv']) * 1e3 for s in steps]),
        command_s=np.array([float(s['t_cmd_s']) for s in steps]),
        interval_s=float(steps[0]['interval_s']))
    reference['settle'] = {
        'run': run, 'config': _Json(run, 'meta.json')['config']['settle'],
        'steps': [{'step': int(s['step']), 'repeat': int(s['repeat']),
                   **{key: float(s[key]) for key in ('t10_s', 'rise_10_90_s', 'overshoot_pct',
                                                     'settle_s')}} for s in steps],
        'summary': _Json(run, 'results.json')['summary']}

    run = '20261024_141540_cooler'
    rows = _Rows(run, 'readings.csv')
    np.savez_compressed(os.path.join(HERE, 'cooler.npz'),
                        cooler_v=np.array([float(r['cooler_v']) for r in rows]),
                        interval_s=float(rows[1]['t_s']) - float(rows[0]['t_s']))
    result = _Json(run, 'results.json')
    reference['cooler'] = {
        'run': run, 'config': _Json(run, 'meta.json')['config']['cooler_monitor'],
        **{key: value for key, value in result.items() if not isinstance(value, (list, dict))},
        'peaks': result['peaks'], 'bands': result['bands']}

    with open(os.path.join(HERE, 'reference.json'), 'w') as fh:
        json.dump(reference, fh, indent=1)


if __name__ == '__main__':
    main()
