"""The health check's judgement: each part against the last check, and its record and ELOG text."""

import pytest

from cls_acquisition import health

GOOD = {'When': '2026-11-10 08:00:00',
        'Cooler': {'Mean_V': 29884.8, 'Ripple_V': 0.44, 'Rms_V': 0.33},
        'Triggers': {'Seconds': 10.0, 'Expected': 100.0, 'Numbered': 100, 'Received': 100},
        'Counts': {1: 50, 2: 60, 3: 40, 4: 55},
        'Sources': {'Line meter': 'IMM', 'Cooler meter': 'IMM'}}


def _Verdicts(checks):
    return {c['Check']: c['Verdict'] for c in checks}


def _Found(**changes):
    return {key: {**value, **changes.get(key, {})} if isinstance(value, dict) else value
            for key, value in GOOD.items()}


def test_A_Rack_As_It_Was_Passes_And_A_First_Check_Is_Set_Against_Nothing():
    first = health.Judge(GOOD)
    assert _Verdicts(first) == {'Bench table': 'N/A', 'Cooler': 'PASS', 'PMTs': 'PASS',
                                'Triggers': 'PASS', 'Meter triggers': 'PASS'}
    assert 'first check' in first[1]['Note'] and health.Verdict(first) == 'PASS'
    again = health.Judge(GOOD, GOOD, {'Verdict': 'PASS', 'Fit': {'Rms_V': 0.009,
                                                                 'Points': 250}})
    assert set(_Verdicts(again).values()) == {'PASS'} and 'last 0.440 V' in str(again[1])


@pytest.mark.parametrize('changes, part, words', [
    ({'Cooler': {'Ripple_V': 0.9}}, 'Cooler', 'ripple has grown'),
    ({'Counts': {1: 0}}, 'PMTs', 'No counts on 1'),
    ({'Counts': {2: 400}}, 'PMTs', '2 changed'),
    ({'Triggers': {'Numbered': 90, 'Received': 90}}, 'Triggers', 'did not count'),
    ({'Triggers': {'Received': 97}}, 'Triggers', '3 numbered triggers never reached'),
    ({'Sources': {'Cooler meter': 'EXT'}}, 'Meter triggers', 'Cooler meter would read'),
])
def test_What_Changed_Since_The_Last_Check_Is_Warned_Of_With_Its_Cause(changes, part, words):
    checks = health.Judge(_Found(**changes), GOOD)
    assert _Verdicts(checks)[part] == 'WARN' and health.Verdict(checks) == 'WARN'
    assert words in next(c['Note'] for c in checks if c['Check'] == part)
    assert [c['Check'] for c in checks if c['Verdict'] == 'WARN'] == [part]


def test_A_Few_More_Dark_Counts_Is_No_Change():
    """Twice as many, but too few to tell from chance."""
    assert _Verdicts(health.Judge(_Found(Counts={3: 6}), _Found(Counts={3: 2})))['PMTs'] == \
        'PASS'


def test_A_Check_Is_Kept_And_Is_What_The_Next_Is_Set_Against(tmp_path):
    assert health.Last(str(tmp_path)) is None
    found = dict(GOOD, Trace=([0.0, 0.001], [1.0, 2.0]))
    checks = health.Judge(found, None, {'Verdict': 'WARN', 'Fit': {'Rms_V': 0.03,
                                                                    'Points': 250}})
    path = health.Save(str(tmp_path), found, checks)
    last = health.Last(str(tmp_path))
    assert last['Counts'] == GOOD['Counts'] and 'Trace' not in last
    assert last['Verdicts']['Bench table'] == 'WARN' and path.endswith('.yaml')
    text = health.Elog_Text(checks, GOOD['When'])
    assert text.startswith('HEALTH CHECK  WARN  (2026-11-10 08:00:00)')
    assert '  Bench table: WARN, Residual 30.0 mV rms; Points 250. The table no longer' in text
