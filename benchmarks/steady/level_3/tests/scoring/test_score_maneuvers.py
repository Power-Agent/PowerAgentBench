# ABOUTME: Tests the standalone benchmark scorer against real dataset scenarios.
# ABOUTME: Covers scorer loop semantics, invalid actions, budget capping, and batch aggregation.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandapower as pp
import pytest

from restorebench.schemas.power_flow import NRDiagnostics, PowerFlowResult
from restorebench.corpus.augment import build_augmented_base


DATASET_FULL = Path("dataset/ieee118/full")
DATASET_INDEX = Path("dataset/ieee118/evaluation_manifest.json")

# Every resolving sequence below is the scenario's own curation witness, copied from
# dataset/ieee118/private/witnesses.json. Scoring one is the strictest end-to-end check the
# scorer has: the corpus itself asserts the sequence resolves.
RESOLVING_SCENARIO = "S0008"
RESOLVING_MANEUVER: dict[str, Any] = {"type": "GEN_V_SETPOINT", "gen_id": 33, "new_vm_pu": 0.96}

SUCCESS_FIXTURES: list[dict[str, Any]] = [
    {
        "scenario_id": "S0008",
        "maneuvers": [{"type": "GEN_V_SETPOINT", "gen_id": 33, "new_vm_pu": 0.96}],
    },
    {
        "scenario_id": "S0014",
        "maneuvers": [{"type": "GEN_V_SETPOINT", "gen_id": 31, "new_vm_pu": 0.981}],
    },
    {
        "scenario_id": "S0019",
        "maneuvers": [{"type": "GEN_V_SETPOINT", "gen_id": 2, "new_vm_pu": 1.0}],
    },
    {
        "scenario_id": "S0039",
        "maneuvers": [{"type": "GEN_V_SETPOINT", "gen_id": 10, "new_vm_pu": 1.04}],
    },
    {
        "scenario_id": "S0029",
        "maneuvers": [{"type": "GEN_V_SETPOINT", "gen_id": 11, "new_vm_pu": 1.025}],
    },
]


def _without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_runtime(item)
            for key, item in value.items()
            if key not in {"scoring_runtime_seconds", "mean_scoring_time_seconds"}
        }
    if isinstance(value, list):
        return [_without_runtime(item) for item in value]
    return value


def _write_attempt(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _first_tappable_trafo_id(net: Any) -> int:
    mask = net.trafo["in_service"].astype(bool) & net.trafo["tap_pos"].notna()
    return int(net.trafo.index[mask][0])


def test_known_resolving_sequence_succeeds_with_atomic_maneuvers() -> None:
    from restorebench.scoring.score_maneuvers import score_attempt

    report = score_attempt(
        {
            "scenario_id": RESOLVING_SCENARIO,
            "source": "pytest real-data fixture",
            "maneuvers": [RESOLVING_MANEUVER],
        },
        data_dir=DATASET_FULL,
    )

    assert report["status"] == "SUCCESS"
    assert report["converged"] is True
    assert report["n_maneuvers"] == 1
    assert report["n_proposed"] == 1
    assert report["n_invalid"] == 0
    assert report["steps"][-1]["outcome"] == "APPLIED_CONVERGED"
    assert report["quality"] is not None
    assert report["solver_settings"] == {"init": "dc", "max_iteration": 30}


def test_empty_maneuver_list_exhausts_budget_without_crashing() -> None:
    from restorebench.scoring.score_maneuvers import score_attempt

    report = score_attempt({"scenario_id": RESOLVING_SCENARIO, "maneuvers": []}, data_dir=DATASET_FULL)

    assert report["status"] == "BUDGET_EXHAUSTED"
    assert report["converged"] is False
    assert report["n_maneuvers"] == 0
    assert report["n_invalid"] == 0
    assert report["steps"] == []
    assert report["quality"] is None


def test_schema_invalid_actions_burn_slots_and_continue_to_success() -> None:
    from restorebench.scoring.score_maneuvers import score_attempt

    report = score_attempt(
        {
            "scenario_id": RESOLVING_SCENARIO,
            "maneuvers": [
                {"type": "GEN_V_SETPOINT", "gen_id": 11, "new_vm_pu": 1.2},
                {"type": "TAP_ADJUSTMENT", "trafo_id": 2, "new_tap_pos": 5},
                {"type": "LINE_SWITCH", "line_id": 0, "target_state": False},
                {"type": "GEN_V_SETPOINT", "gen_id": 11},
                RESOLVING_MANEUVER,
            ],
        },
        data_dir=DATASET_FULL,
    )

    assert report["status"] == "SUCCESS"
    assert report["n_invalid"] == 4
    assert report["n_maneuvers"] == 1
    outcomes = [step["outcome"] for step in report["steps"]]
    assert outcomes[:4] == ["INVALID_ACTION"] * 4
    assert outcomes[-1] == "APPLIED_CONVERGED"


def test_applicability_invalid_actions_discard_sandbox_and_continue(tmp_path: Path) -> None:
    from restorebench.scoring.score_maneuvers import score_attempt

    data_dir = tmp_path / "full"
    data_dir.mkdir()
    net = pp.from_json(str(DATASET_FULL / f"{RESOLVING_SCENARIO}.json"))
    trafo_id = _first_tappable_trafo_id(net)
    shunt_id = int(net.shunt.index.max()) + 1000
    net.trafo.at[trafo_id, "tap_max"] = 1
    pp.to_json(net, str(data_dir / "S8008.json"))

    report = score_attempt(
        {
            "scenario_id": "S8008",
            "maneuvers": [
                {"type": "SHUNT_STEP", "shunt_id": shunt_id, "new_step": 0},
                {"type": "GEN_V_SETPOINT", "gen_id": int(net.gen.index.max()) + 1000, "new_vm_pu": 1.0},
                {"type": "TAP_ADJUSTMENT", "trafo_id": trafo_id, "new_tap_pos": 2},
                RESOLVING_MANEUVER,
            ],
        },
        data_dir=data_dir,
    )

    assert report["status"] == "SUCCESS"
    assert report["n_invalid"] == 3
    assert report["n_maneuvers"] == 1
    outcomes = [step["outcome"] for step in report["steps"]]
    assert outcomes[:3] == ["INVALID_ACTION"] * 3
    assert outcomes[-1] == "APPLIED_CONVERGED"


def test_q_saturated_voltage_raise_is_invalid_in_standalone_scorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from restorebench.scoring import score_maneuvers

    net = build_augmented_base()
    gen_id = int(net.gen.index[0])
    net.gen.at[gen_id, "vm_pu"] = 1.0
    baseline = PowerFlowResult(
        converged=False,
        iterations=30,
        tolerance_used=1e-6,
        runtime_ms=1.0,
        error_message="did not converge",
        diagnostics=NRDiagnostics(
            iterations_attempted=30,
            worst_bus=1,
            lowest_vm_pu=0.7,
            lowest_vm_bus=1,
            gens_at_q_limit=[gen_id],
            error_message="did not converge",
            diagnostics_source="local_nose",
        ),
    )
    pf_inputs: list[Any] = []

    def fake_run_ac_pf(grid: Any) -> PowerFlowResult:
        pf_inputs.append(grid)
        return baseline

    monkeypatch.setattr(score_maneuvers, "_load_scenario", lambda *_args: net)
    monkeypatch.setattr(score_maneuvers, "run_ac_pf", fake_run_ac_pf)

    report = score_maneuvers.score_attempt(
        {
            "scenario_id": "S9001",
            "maneuvers": [
                {"type": "GEN_V_SETPOINT", "gen_id": gen_id, "new_vm_pu": 1.01},
            ],
        },
        data_dir=DATASET_FULL,
    )

    assert report["status"] == "BUDGET_EXHAUSTED"
    assert report["n_invalid"] == 1
    assert report["n_maneuvers"] == 0
    assert len(pf_inputs) == 1


def test_scenario_that_converges_on_load_fails_loudly(tmp_path: Path) -> None:
    from restorebench.scoring.score_maneuvers import CorpusIntegrityError, score_attempt

    data_dir = tmp_path / "full"
    data_dir.mkdir()
    pp.to_json(build_augmented_base(), str(data_dir / "S9000.json"))

    with pytest.raises(CorpusIntegrityError, match="converges on load"):
        score_attempt({"scenario_id": "S9000", "maneuvers": []}, data_dir=data_dir)


def test_eleventh_maneuver_is_never_attempted() -> None:
    from restorebench.scoring.score_maneuvers import MANEUVER_BUDGET, score_attempt

    invalid_entries = [{"type": "GEN_V_SETPOINT", "gen_id": 11, "new_vm_pu": 1.2} for _ in range(MANEUVER_BUDGET)]
    report = score_attempt(
        {
            "scenario_id": RESOLVING_SCENARIO,
            "maneuvers": [
                *invalid_entries,
                {"type": "GEN_V_SETPOINT", "gen_id": 11, "new_vm_pu": 1.025},
            ],
        },
        data_dir=DATASET_FULL,
    )

    assert report["status"] == "BUDGET_EXHAUSTED"
    assert report["n_proposed"] == MANEUVER_BUDGET + 1
    assert report["n_invalid"] == MANEUVER_BUDGET
    assert report["n_maneuvers"] == 0
    assert len(report["steps"]) == MANEUVER_BUDGET
    assert all(step["outcome"] == "INVALID_ACTION" for step in report["steps"])


def test_batch_report_aggregates_real_scenarios_and_is_deterministic(tmp_path: Path) -> None:
    from restorebench.scoring.score_maneuvers import score_batch

    batch_dir = tmp_path / "attempts"
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    batch_dir.mkdir()
    for index, payload in enumerate(SUCCESS_FIXTURES, start=1):
        _write_attempt(batch_dir / f"attempt_{index}.json", payload)

    first = score_batch(batch_dir, out_dir=first_out, data_dir=DATASET_FULL, index_path=DATASET_INDEX)
    second = score_batch(batch_dir, out_dir=second_out, data_dir=DATASET_FULL, index_path=DATASET_INDEX)

    assert (first_out / "benchmark_report.json").exists()
    attempt_reports = sorted(first_out.glob("attempt_*_score.json"))
    assert len(attempt_reports) == len(SUCCESS_FIXTURES)
    assert json.loads(attempt_reports[0].read_text(encoding="utf-8"))["scenario_id"] == "S0008"
    assert first["n_attempts"] == 5
    assert first["success_rate"] == 1.0
    assert first["n_maneuvers_success"]["mean"] == 1.0
    assert first["n_maneuvers_success"]["distribution"] == {"1": 5}
    assert first["quality_clean_rate"] == 0.0
    assert first["invalid_action_rate"] == 0.0
    assert {
        split: row["n_attempts"]
        for split, row in first["by_memory_split"].items()
    } == {"held_out": 5}
    assert _without_runtime(first) == _without_runtime(second)


def test_cli_data_dir_selects_corpus_for_single_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from restorebench.scoring import score_maneuvers

    attempt = tmp_path / "attempt.json"
    output = tmp_path / "score.json"
    _write_attempt(attempt, SUCCESS_FIXTURES[0])
    calls: list[tuple[Path, Path]] = []

    def fake_score_attempt_file(path: Path, *, data_dir: Path) -> dict[str, Any]:
        calls.append((path, data_dir))
        return {"scenario_id": "S0001", "status": "SUCCESS"}

    monkeypatch.setattr(score_maneuvers, "score_attempt_file", fake_score_attempt_file)

    score_maneuvers.main([
        str(attempt),
        "--data-dir",
        "dataset/pegase89",
        "--out",
        str(output),
    ])

    assert calls == [(attempt, Path("dataset/pegase89/full"))]
    assert output.is_file()


def test_cli_data_dir_selects_matching_corpus_and_index_for_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from restorebench.scoring import score_maneuvers

    batch = tmp_path / "attempts.jsonl"
    batch.write_text("", encoding="utf-8")
    output = tmp_path / "scores"
    calls: list[tuple[Path, Path, Path, Path]] = []

    def fake_score_batch(
        batch_path: Path,
        *,
        out_dir: Path,
        data_dir: Path,
        index_path: Path,
    ) -> dict[str, Any]:
        calls.append((batch_path, out_dir, data_dir, index_path))
        return {"n_attempts": 0}

    monkeypatch.setattr(score_maneuvers, "score_batch", fake_score_batch)

    score_maneuvers.main([
        "--batch",
        str(batch),
        "--data-dir",
        "dataset/pegase89/full",
        "--out",
        str(output),
    ])

    assert calls == [(
        batch,
        output,
        Path("dataset/pegase89/full"),
        Path("dataset/pegase89/evaluation_manifest.json"),
    )]


def test_published_s0081_witness_converges_at_primary_tolerance() -> None:
    from restorebench.scoring.score_maneuvers import score_attempt

    witnesses = json.loads(Path("dataset/ieee118/private/witnesses.json").read_text(encoding="utf-8"))
    witness = next(row for row in witnesses if row["scenario_id"] == "S0081")

    report = score_attempt(
        {"scenario_id": "S0081", "maneuvers": witness["maneuvers"]},
        data_dir=DATASET_FULL,
    )

    assert report["status"] == "SUCCESS"
    assert report["steps"][-1]["tolerance_used"] == 1e-8


def test_aggregate_reports_tolerates_success_with_null_quality() -> None:
    # A converged attempt can carry quality=None (the key is present, so a plain
    # .get('quality', {}) returns None and None.get('clean') raises AttributeError).
    from restorebench.scoring.score_maneuvers import _aggregate_reports

    reports = [
        {
            "scenario_id": "S0001",
            "status": "SUCCESS",
            "steps": [],
            "n_invalid": 0,
            "n_maneuvers": 1,
            "quality": None,
            "scoring_runtime_seconds": 0.5,
        }
    ]

    aggregate = _aggregate_reports(reports)

    assert aggregate["quality_clean_rate"] == 0.0
