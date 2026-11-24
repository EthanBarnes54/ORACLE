"""The HV readback streamed beside a run (WP9), on the simulated line meter."""

import itertools
import json
import os
import time

import numpy as np
import pytest

from cls_acquisition import diagnostics
from cls_acquisition.hardware import keysight, simulated
from cls_acquisition.hardware.visa_session import Visa_Session

_Names = (f"SIM::DIAGNOSTICS::{i}" for i in itertools.count())


@pytest.fixture
def meter():
    inst = simulated.Keysight(lambda: 1.25)          # 1250 V through the 1000:1 divider
    line = keysight.Line_Meter(Visa_Session.Open(next(_Names), lambda _: inst))
    yield line
    line.Close()


def _Stream(meter, folder, seconds):
    stream = diagnostics.Readback_Stream(meter, 1000.0, str(folder))
    time.sleep(seconds)
    return stream, stream.Stop()


def test_The_Readback_Streams_Every_Millisecond_Beside_The_Run(meter, tmp_path):
    stream, meta = _Stream(meter, tmp_path / 'run_0007', 0.5)
    times, volts = diagnostics.Read(stream.Folder)
    assert meta['Error'] is None and meta['Readings'] == len(volts) > 300
    # on the meter's timer, in time.time() seconds: resolved to 0.24 us at today's epoch
    assert np.allclose(volts, 1250.0) and np.allclose(np.diff(times), 0.001, rtol=0, atol=1e-6)
    assert meta['Aperture_S'] == 200e-6 and len(meta['Records']) == 1
    assert os.path.getsize(os.path.join(stream.Folder, diagnostics.DATA)) == 4 * len(volts)
    # the meter back on its poll, its latest reading the stream's, as the monitor takes it
    assert meter.Mode == 'continuous' and meter.Voltage == pytest.approx(1.25)


def test_A_Run_Longer_Than_A_Record_Goes_On_In_A_New_One(meter, tmp_path, monkeypatch):
    monkeypatch.setattr(meter, 'Stream_Limit', lambda: 120)
    stream, meta = _Stream(meter, tmp_path / 'run_0008', 0.6)
    times, volts = diagnostics.Read(stream.Folder)
    assert len(meta['Records']) >= 3 and meta['Error'] is None
    assert [r['First'] for r in meta['Records'][:3]] == [0, 120, 240]
    assert np.all(np.diff(times) > 0)                  # each record after the last


def test_A_Stream_That_Fails_Leaves_The_Meter_Polling_And_Says_Why(meter, tmp_path, monkeypatch):
    def refuse(*_, **__):
        raise keysight.Scpi_Error("the meter at SIM refused the capture")
    monkeypatch.setattr(meter, 'Capture', refuse)
    stream, meta = _Stream(meter, tmp_path / 'run_0009', 0.1)
    assert meta['Readings'] == 0 and 'refused the capture' in meta['Error']
    assert meter.Mode == 'continuous'
    with open(os.path.join(stream.Folder, diagnostics.META), encoding='utf-8') as fh:
        assert json.load(fh)['Error'] == meta['Error']
