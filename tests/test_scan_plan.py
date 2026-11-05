"""The scan plan: counted, de-duplicated, and refused when it cannot be right."""

import numpy as np
import pytest

from cls_acquisition.scan_plan import Scan_Plan, Scan_Plan_Error


def test_Points_Are_Counted_Not_Accumulated():
    # 463 points, the production ramp an accumulated arange made 464 of
    assert len(Scan_Plan([0.0], [46.2], 0.1)) == 463
    assert len(Scan_Plan([-2310.0], [2310.0], 10.0)) == 463


def test_Shared_Endpoint_Is_Commanded_Once():
    plan = Scan_Plan([0, 100], [100, 200], 1)
    assert len(plan) == 201
    assert np.all(np.diff(plan.Setpoints) > 0)


def test_Ranges_Are_Ascending_Whichever_Way_Entered():
    assert Scan_Plan([10], [0], 1).Setpoints[0] == 0.0


@pytest.mark.parametrize('starts, stops, step', [
    ([0, 50], [100, 150], 1),       # overlap: setpoints commanded twice
    ([0], [0.5], 1),                # fewer than two points
    ([], [], 1),                    # nothing ticked
    ([0], [10], 0),                 # no step
])
def test_Bad_Plans_Are_Refused(starts, stops, step):
    with pytest.raises(Scan_Plan_Error):
        Scan_Plan(starts, stops, step)


def test_Bin_Edges_Centre_On_Setpoints():
    plan = Scan_Plan([0], [10], 2)
    edges = plan.Bin_Edges()
    assert np.allclose(0.5 * (edges[1:] + edges[:-1]), plan.Setpoints)


@pytest.mark.parametrize('stop, last', [(474, 474), (475, 474), (476, 476), (477, 476)])
def test_A_Range_Never_Runs_Past_Its_Stop(stop, last):
    # counted to the nearest step, -450 to 477 V at 2 V ended at 478 V, the old DAQ's
    # spurious final point back, while -450 to 475 V ended at 474 V
    plan = Scan_Plan([-450], [stop], 2)
    assert plan.Setpoints[-1] == last and len(plan) == (last + 450) // 2 + 1
