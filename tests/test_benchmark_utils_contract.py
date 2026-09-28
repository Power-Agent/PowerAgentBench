from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from poweragentbench.benchmark_utils import (
    _assert_step,
    compute_action_cost,
    compute_violation_score,
    load_actioncost,
    load_actionspace,
    load_solution,
    validate_action_values,
)


ROOT = Path(__file__).resolve().parents[1]
LEVEL1 = ROOT / "benchmarks" / "steady" / "level_1"


def test_published_level1_contracts_are_loadable() -> None:
    actionspace = load_actionspace(LEVEL1 / "actionspace.json")
    actioncost = load_actioncost(LEVEL1 / "actioncost.json")

    assert actionspace["actions"]
    assert actionspace["contingencies"]
    assert actionspace["operating_limits"]
    assert set(actionspace["action_index"]) == {
        action["id"] for action in actionspace["actions"]
    }
    assert set(actioncost["action_index"]) == {
        action["id"] for action in actioncost["actions"]
    }


def test_validate_action_values_accepts_empty_solution() -> None:
    actionspace = load_actionspace(LEVEL1 / "actionspace.json")

    values = validate_action_values({}, actionspace)

    assert len(values) == len(actionspace["actions"])
    assert all(value == 0.0 for value in values.values())


def test_validate_action_values_rejects_unknown_action() -> None:
    actionspace = load_actionspace(LEVEL1 / "actionspace.json")

    with pytest.raises(ValueError, match="Unknown action"):
        validate_action_values({"does_not_exist": 1.0}, actionspace)


def test_validate_action_values_rejects_out_of_range_action() -> None:
    actionspace = load_actionspace(LEVEL1 / "actionspace.json")
    action = actionspace["actions"][0]
    action_id = action["id"]
    too_large = float(action["max_value"]) + float(action["step"])

    with pytest.raises(ValueError, match="outside"):
        validate_action_values({action_id: too_large}, actionspace)


def test_validate_action_values_rejects_off_grid_step() -> None:
    actionspace = load_actionspace(LEVEL1 / "actionspace.json")
    action = next(
        action for action in actionspace["actions"] if float(action["step"]) > 0
    )
    action_id = action["id"]
    min_value = float(action["min_value"])
    step = float(action["step"])
    candidate = min_value + 0.5 * step
    if candidate > float(action["max_value"]):
        pytest.skip("Published action has no interior half-step candidate")

    with pytest.raises(ValueError, match="align with step size"):
        validate_action_values({action_id: candidate}, actionspace)


def test_assert_step_handles_floating_point_roundoff() -> None:
    _assert_step(0.30000000000000004, 0.1, 0.1, "roundoff")


def test_action_cost_matches_published_per_step_values() -> None:
    actionspace = load_actionspace(LEVEL1 / "actionspace.json")
    actioncost = load_actioncost(LEVEL1 / "actioncost.json")

    one_step = validate_action_values(
        {"redispatch_G7_up": 20.0},
        actionspace,
    )
    assert compute_action_cost(one_step, actionspace, actioncost) == pytest.approx(50.0)

    multiple_steps = validate_action_values(
        {"redispatch_G9_up": 100.0},
        actionspace,
    )
    assert compute_action_cost(multiple_steps, actionspace, actioncost) == pytest.approx(240.0)

    negative_permitted = validate_action_values(
        {"phase_shift_T9": -2.0},
        actionspace,
    )
    assert compute_action_cost(
        negative_permitted, actionspace, actioncost
    ) == pytest.approx(10.0)


def test_action_cost_combines_multiple_actions() -> None:
    actionspace = load_actionspace(LEVEL1 / "actionspace.json")
    actioncost = load_actioncost(LEVEL1 / "actioncost.json")
    values = validate_action_values(
        {
            "redispatch_G7_up": 20.0,
            "redispatch_G9_up": 100.0,
            "phase_shift_T9": -2.0,
        },
        actionspace,
    )

    assert compute_action_cost(values, actionspace, actioncost) == pytest.approx(300.0)


def test_violation_score_is_zero_inside_limits() -> None:
    line = pd.Series({"L1": 50.0, "L2": 80.0})
    voltage = pd.Series({"B1": 0.98, "B2": 1.02})
    assert compute_violation_score(line, voltage, 100.0, 0.95, 1.05) == pytest.approx(0.0)


def test_violation_score_line_overload_is_exact() -> None:
    line = pd.Series({"L1": 110.0})
    voltage = pd.Series({"B1": 1.00})
    assert compute_violation_score(line, voltage, 100.0, 0.95, 1.05) == pytest.approx(0.1)


def test_violation_score_undervoltage_is_exact() -> None:
    line = pd.Series({"L1": 50.0})
    voltage = pd.Series({"B1": 0.90})
    assert compute_violation_score(line, voltage, 100.0, 0.95, 1.05) == pytest.approx(0.5)


def test_violation_score_overvoltage_is_exact() -> None:
    line = pd.Series({"L1": 50.0})
    voltage = pd.Series({"B1": 1.10})
    assert compute_violation_score(line, voltage, 100.0, 0.95, 1.05) == pytest.approx(0.5)


def test_violation_score_combines_line_and_voltage_terms() -> None:
    line = pd.Series({"L1": 110.0})
    voltage = pd.Series({"B1": 0.90})
    assert compute_violation_score(line, voltage, 100.0, 0.95, 1.05) == pytest.approx(0.6)


def test_violation_score_is_zero_on_operating_limits() -> None:
    line = pd.Series({"L1": 100.0})
    voltage = pd.Series({"B1": 0.95, "B2": 1.05})
    assert compute_violation_score(line, voltage, 100.0, 0.95, 1.05) == pytest.approx(0.0)


def test_solution_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    payload = {
        "actions": [
            {"id": "x", "value": 1},
            {"id": "x", "value": 2},
        ]
    }
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate action"):
        load_solution(path)


def test_solution_loader_rejects_non_numeric_values(tmp_path: Path) -> None:
    payload = {"actions": [{"id": "x", "value": "1"}]}
    path = tmp_path / "bad_value.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="numeric"):
        load_solution(path)
