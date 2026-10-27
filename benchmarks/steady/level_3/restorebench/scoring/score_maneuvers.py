# ABOUTME: Scores proposed maneuver lists against frozen non-converging pandapower scenarios.
# ABOUTME: Reuses the existing sandbox and power-flow tool layer without LLM or network calls.
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandapower as pp
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from restorebench.schemas.actions import Action, Maneuver
from restorebench.schemas.errors import CorpusIntegrityError, InvalidActionError, ToolFailureError
from restorebench.schemas.power_flow import PowerFlowResult
from restorebench.tools.power_flow import INIT, MAX_ITERATION, run_ac_pf
from restorebench.tools.sandbox import (
    apply_maneuver,
    create_sandbox,
    discard_sandbox,
    promote_sandbox,
    resolve_net,
)


ROOT = Path(__file__).resolve().parents[2]
MANEUVER_BUDGET = 10
DEFAULT_DATA_DIR = ROOT / "dataset/ieee118/full"
DEFAULT_DATASET_ROOT = ROOT / "dataset/ieee118"
DEFAULT_BATCH_OUT = ROOT / "results/manual_scoring"
ACTION_ADAPTER: TypeAdapter[Any] = TypeAdapter(Action)
SOLVER_SETTINGS = {"init": INIT, "max_iteration": MAX_ITERATION}


class AttemptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    maneuvers: list[Any]
    source: str | None = None


def score_attempt(payload: Mapping[str, Any], *, data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    started = time.perf_counter()
    attempt = AttemptInput.model_validate(payload)
    working = _load_scenario(attempt.scenario_id, data_dir)
    baseline = run_ac_pf(working)
    if baseline.converged:
        raise CorpusIntegrityError(f"{attempt.scenario_id} converges on load; corpus integrity failed")

    applied: list[Maneuver] = []
    steps: list[dict[str, Any]] = []
    final_result: PowerFlowResult | None = None
    latest_result = baseline
    converged = False

    for entry in attempt.maneuvers[:MANEUVER_BUDGET]:
        try:
            maneuver = parse_maneuver_entry(entry)
        except ValidationError as exc:
            steps.append(_invalid_step(entry, str(exc)))
            continue

        sandbox = create_sandbox(working)
        try:
            diagnostics = latest_result.diagnostics
            saturated_gens = frozenset(diagnostics.gens_at_q_limit) if diagnostics is not None else frozenset()
            apply_maneuver(sandbox, maneuver, saturated_gens=saturated_gens)
        except (InvalidActionError, ToolFailureError) as exc:
            discard_sandbox(sandbox)
            steps.append(_invalid_step(entry, str(exc)))
            continue

        final_result = run_ac_pf(sandbox)
        latest_result = final_result
        promote_sandbox(sandbox)
        working = resolve_net(sandbox)
        discard_sandbox(sandbox)
        applied.append(maneuver)
        outcome = "APPLIED_CONVERGED" if final_result.converged else "STILL_DIVERGED"
        steps.append(_result_step(maneuver, outcome, final_result))
        if final_result.converged:
            converged = True
            break

    return _attempt_report(attempt, applied, steps, final_result, converged, started)


def parse_maneuver_entry(entry: Any) -> Maneuver:
    action_payload = _action_payload(entry)
    action = ACTION_ADAPTER.validate_python(action_payload)
    return Maneuver(action=action, diagnosed_cause=None, rationale="")


def score_attempt_file(path: Path, *, data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    return score_attempt(json.loads(path.read_text(encoding="utf-8")), data_dir=data_dir)


def score_batch(
    batch_path: Path,
    *,
    out_dir: Path = DEFAULT_BATCH_OUT,
    data_dir: Path = DEFAULT_DATA_DIR,
    index_path: Path | None = None,
) -> dict[str, Any]:
    attempts = _load_batch_attempts(batch_path)
    reports = [score_attempt(attempt, data_dir=data_dir) for attempt in attempts]
    resolved_index_path = index_path or _dataset_paths(data_dir)[1]
    scenario_index = _load_scenario_index(resolved_index_path)
    report = _aggregate_batch(reports, scenario_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, attempt_report in enumerate(reports, start=1):
        _write_json(out_dir / f"attempt_{index:04d}_score.json", attempt_report)
    _write_json(out_dir / "benchmark_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    data_dir, index_path = _dataset_paths(args.data_dir)
    if args.batch is not None:
        out_dir = args.out or DEFAULT_BATCH_OUT
        report = score_batch(
            args.batch,
            out_dir=out_dir,
            data_dir=data_dir,
            index_path=index_path,
        )
        print(_json_dumps(report))
        return

    if args.input is None:
        raise SystemExit("one maneuver JSON file is required unless --batch is used")
    report = score_attempt_file(args.input, data_dir=data_dir)
    _write_json(_single_report_path(args.input, args.out), report)
    print(_json_dumps(report))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score maneuver proposals against frozen benchmark scenarios.")
    parser.add_argument("input", nargs="?", type=Path, help="single maneuver JSON attempt")
    parser.add_argument("--batch", type=Path, help="directory, JSON file, or JSONL file of maneuver attempts")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="dataset root or its full/ scenario directory (default: dataset/ieee118)",
    )
    parser.add_argument("--out", type=Path, help="output directory, or output JSON file for a single attempt")
    return parser.parse_args(argv)


def _dataset_paths(data_dir: Path) -> tuple[Path, Path]:
    """Resolve one corpus selection for both scenario loading and batch stratification."""
    dataset_root = data_dir.parent if data_dir.name == "full" else data_dir
    full_dir = data_dir if data_dir.name == "full" else data_dir / "full"
    return full_dir, dataset_root / "evaluation_manifest.json"


def _load_scenario(scenario_id: str, data_dir: Path) -> Any:
    return pp.from_json(str(data_dir / f"{scenario_id}.json"))


def _load_batch_attempts(batch_path: Path) -> list[dict[str, Any]]:
    if batch_path.is_dir():
        paths = sorted(path for path in batch_path.glob("*.json") if path.name != "benchmark_report.json")
        return [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if batch_path.suffix == ".jsonl":
        return [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    loaded = json.loads(batch_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, list) else [loaded]


def _load_scenario_index(index_path: Path) -> dict[str, dict[str, Any]]:
    loaded = json.loads(index_path.read_text(encoding="utf-8"))
    return {str(label["scenario_id"]): label for label in loaded["scenarios"]}


def _action_payload(entry: Any) -> Any:
    if isinstance(entry, Mapping) and isinstance(entry.get("action"), Mapping):
        return entry["action"]
    return entry


def _invalid_step(entry: Any, detail: str) -> dict[str, Any]:
    return {
        "action": _jsonable(_action_payload(entry)),
        "outcome": "INVALID_ACTION",
        "detail": detail,
        "iterations": None,
        "tolerance_used": None,
    }


def _result_step(maneuver: Maneuver, outcome: str, result: PowerFlowResult) -> dict[str, Any]:
    detail = "converged" if result.converged else result.error_message or "power flow did not converge"
    return {
        "action": maneuver.action.model_dump(mode="json"),
        "outcome": outcome,
        "detail": detail,
        "iterations": result.iterations,
        "tolerance_used": result.tolerance_used,
    }


def _attempt_report(
    attempt: AttemptInput,
    applied: list[Maneuver],
    steps: list[dict[str, Any]],
    final_result: PowerFlowResult | None,
    converged: bool,
    started: float,
) -> dict[str, Any]:
    quality = (
        final_result.quality.model_dump(mode="json") if converged and final_result and final_result.quality else None
    )
    return {
        "scenario_id": attempt.scenario_id,
        "status": "SUCCESS" if converged else "BUDGET_EXHAUSTED",
        "converged": converged,
        "n_maneuvers": len(applied),
        "n_proposed": len(attempt.maneuvers),
        "n_invalid": sum(step["outcome"] == "INVALID_ACTION" for step in steps),
        "steps": steps,
        "quality": quality,
        "scoring_runtime_seconds": round(time.perf_counter() - started, 6),
        "solver_settings": SOLVER_SETTINGS,
        "source": attempt.source,
    }


def _aggregate_batch(reports: list[dict[str, Any]], scenario_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = [(report, _metadata_for(report, scenario_index)) for report in reports]
    aggregate = _aggregate_reports(reports)
    aggregate["solver_settings"] = SOLVER_SETTINGS
    aggregate["by_memory_split"] = _aggregate_by(
        pairs,
        lambda metadata: str(metadata["memory_split"]),
    )
    return aggregate


def _metadata_for(report: Mapping[str, Any], scenario_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scenario_id = str(report["scenario_id"])
    try:
        return scenario_index[scenario_id]
    except KeyError as exc:
        raise KeyError(f"{scenario_id} missing from split manifest") from exc


def _aggregate_by(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report, metadata in pairs:
        groups[key_fn(metadata)].append(report)
    return {key: _aggregate_reports(groups[key]) for key in sorted(groups)}


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [report for report in reports if report["status"] == "SUCCESS"]
    attempted_slots = sum(len(report["steps"]) for report in reports)
    invalid = sum(int(report["n_invalid"]) for report in reports)
    # `quality` is always present but can be None on a converged attempt; a plain
    # .get("quality", {}) would return None and crash on .get("clean").
    clean = sum(1 for report in successes if (report.get("quality") or {}).get("clean") is True)
    return {
        "n_attempts": len(reports),
        "success_rate": _rate(len(successes), len(reports)),
        "n_maneuvers_success": _maneuver_stats(successes),
        "mean_scoring_time_seconds": _mean([float(report["scoring_runtime_seconds"]) for report in reports]),
        "quality_clean_rate": _rate(clean, len(successes)) if successes else None,
        "invalid_action_rate": _rate(invalid, attempted_slots),
    }


def _maneuver_stats(successes: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(int(report["n_maneuvers"]) for report in successes)
    return {
        "mean": _mean([float(report["n_maneuvers"]) for report in successes]),
        "distribution": {str(key): counts[key] for key in sorted(counts)},
    }


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else round(sum(values) / len(values), 6)


def _single_report_path(input_path: Path, out: Path | None) -> Path:
    if out is not None and out.suffix == ".json":
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    directory = out or input_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{input_path.stem}_score.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(_json_dumps(payload) + "\n", encoding="utf-8")


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


if __name__ == "__main__":
    main()
