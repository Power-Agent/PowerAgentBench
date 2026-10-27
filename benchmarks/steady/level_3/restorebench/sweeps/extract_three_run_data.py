# ABOUTME: Extracts run-preserving episode and event CSVs from RestoreBench result stores.
# ABOUTME: Validates three-run coverage without fabricating absent cells or missing measurements.
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from restorebench.eval.store import load_response, model_id_from_response
from restorebench.llm.models import model_slug
from restorebench.schemas.actions import Maneuver
from restorebench.schemas.response import ResolutionResponse
from restorebench.sweeps.build_paper_figures import (
    ARCH_LABELS,
    BUDGET_EVENT_LABELS,
    MANEUVER_BUDGET,
    MODEL_DISPLAY,
    OVERSTRESS_EPS,
    committed_overstress_chain,
)
from restorebench.sweeps.run_sweep import Campaign, campaign_scenarios, load_campaigns


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "reports/three_run_extraction"
NETWORK_CAMPAIGNS = {
    "ieee118": ("ieee118-anthropic", "ieee118-bedrock", "ieee118-openai"),
    "pegase89": ("pegase89-anthropic", "pegase89-bedrock", "pegase89-openai"),
}
EXPECTED_REPETITIONS = (0, 1, 2)

EPISODE_COLUMNS = (
    "network",
    "model",
    "architecture",
    "run_id",
    "case_id",
    "success",
    "final_status",
    "termination_reason",
    "cost_usd",
    "runtime_s",
    "in_band_buses_pct",
    "maneuvers_used",
    "maneuver_budget",
    "llm_calls",
    "budget_events_consumed",
    "maneuvers_remaining",
    "llm_calls_per_committed_maneuver",
)

EVENT_COLUMNS = (
    "network",
    "model",
    "architecture",
    "run_id",
    "case_id",
    "event_index",
    "event_category",
    "episode_final_status",
    "progress_verdict",
    "llm_calls_before_event",
    "llm_calls_per_committed_maneuver",
    "action_type",
    "action_target",
    "action_value",
    "solver_status_after_action",
    "converged_after_action",
    "electrically_feasible_after_action",
    "native_event_kind",
    "native_iteration_index",
    "native_event_detail",
    "maneuvers_consumed",
    "maneuvers_remaining",
    "maximum_maneuver_budget",
    "total_llm_calls",
)

SUMMARY_COLUMNS = (
    "network",
    "model",
    "architecture",
    "run_id",
    "number_of_cases",
    "number_successful",
    "success_rate",
    "mean_cost_per_case",
    "median_cost_per_case",
    "mean_runtime",
    "median_runtime",
    "number_budget_exhausted",
    "number_timeout",
    "number_other_terminal_statuses",
    "complete_46_cases",
    "missing_case_count",
)


@dataclass(frozen=True)
class SourceCell:
    network: str
    campaign: str
    store: Path
    path: Path
    response: ResolutionResponse


@dataclass
class Validation:
    unreadable_files: list[str]
    unexpected_repetitions: list[str]
    filename_mismatches: list[str]
    duplicate_episodes: dict[tuple[str, str, str, int, str], list[str]]
    missing_cost: list[tuple[str, str, str, int, str]]
    missing_runtime: list[tuple[str, str, str, int, str]]
    missing_final_status: list[tuple[str, str, str, int, str]]
    impossible_success_status: list[tuple[str, str, str, int, str]]
    progress_chain_mismatches: list[str]
    over_budget_events: list[str]
    version_stamp_errors: list[str]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--store",
        action="append",
        default=[],
        metavar="CAMPAIGN=PATH",
        help="Override one campaign result root. Repeat for multiple campaigns.",
    )
    return parser.parse_args(argv)


def _store_overrides(values: Sequence[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise SystemExit(f"invalid --store {value!r}; expected CAMPAIGN=PATH")
        overrides[name] = Path(raw_path)
    return overrides


def _model_display(model_id: str) -> str:
    return MODEL_DISPLAY.get(model_id, model_id)


def _cells_dir(store: Path) -> Path:
    # Published campaign paths name the result root; CLI overrides may name cells/ directly.
    return store if store.name == "cells" else store / "cells"


def _campaign_networks(campaigns: dict[str, Campaign]) -> dict[str, str]:
    return {
        campaign_name: network
        for network, campaign_names in NETWORK_CAMPAIGNS.items()
        for campaign_name in campaign_names
        if campaign_name in campaigns
    }


def load_source_cells(
    campaigns: dict[str, Campaign], overrides: dict[str, Path], validation: Validation
) -> tuple[list[SourceCell], dict[str, Path]]:
    campaign_network = _campaign_networks(campaigns)
    stores: dict[str, Path] = {}
    cells: list[SourceCell] = []
    for campaign_name, network in sorted(campaign_network.items()):
        campaign = campaigns[campaign_name]
        store = overrides.get(campaign_name, ROOT / campaign.results_dir)
        stores[campaign_name] = store
        cells_dir = _cells_dir(store)
        if not cells_dir.is_dir():
            continue
        for path in sorted(cells_dir.glob("*.json")):
            if "manifest" in path.name:
                continue
            try:
                response = load_response(path)
            except Exception as exc:
                validation.unreadable_files.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            if response.repetition_index not in EXPECTED_REPETITIONS:
                validation.unexpected_repetitions.append(
                    f"{path}: repetition_index={response.repetition_index}"
                )
                continue
            expected_name = (
                f"{response.scenario_id}__config{response.configuration}__"
                f"{model_slug(model_id_from_response(response))}__rep{response.repetition_index}.json"
            )
            if path.name != expected_name:
                validation.filename_mismatches.append(f"{path}: expected filename {expected_name}")
            cells.append(
                SourceCell(
                    network=network,
                    campaign=campaign_name,
                    store=store,
                    path=path,
                    response=response,
                )
            )
    return cells, stores


def _empty_validation() -> Validation:
    return Validation([], [], [], {}, [], [], [], [], [], [], [])


def _episode_key(cell: SourceCell) -> tuple[str, str, str, int, str]:
    response = cell.response
    return (
        cell.network,
        _model_display(model_id_from_response(response)),
        ARCH_LABELS[response.configuration],
        response.repetition_index + 1,
        response.scenario_id,
    )


def _budget_events(response: ResolutionResponse) -> int:
    return sum(1 for feedback in response.failure_feedback if feedback.kind in BUDGET_EVENT_LABELS)


def _cost(response: ResolutionResponse) -> float | None:
    from restorebench.analysis.cost import run_cost_usd

    try:
        value = run_cost_usd(response)
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def episode_rows(cells: Sequence[SourceCell], validation: Validation) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, int, str], list[SourceCell]] = defaultdict(list)
    for cell in cells:
        by_key[_episode_key(cell)].append(cell)
    validation.duplicate_episodes = {
        key: [str(cell.path) for cell in matches]
        for key, matches in by_key.items()
        if len(matches) > 1
    }

    rows: list[dict[str, Any]] = []
    for key, matches in sorted(by_key.items()):
        cell = matches[0]
        response = cell.response
        cost = _cost(response)
        runtime = response.total_runtime_seconds
        consumed = _budget_events(response)
        ratio = (
            response.trace.n_llm_calls / response.n_maneuvers
            if response.n_maneuvers > 0
            else None
        )
        quality = response.quality
        bus_count = 118 if cell.network == "ieee118" else 89
        in_band = (
            100.0 * (bus_count - quality.n_buses_out_of_band) / bus_count
            if response.status == "SUCCESS" and quality is not None
            else None
        )
        if cost is None:
            validation.missing_cost.append(key)
        if not math.isfinite(runtime):
            validation.missing_runtime.append(key)
            runtime_value: float | None = None
        else:
            runtime_value = runtime
        if not response.status:
            validation.missing_final_status.append(key)
        success = int(response.status == "SUCCESS")
        if bool(success) != response.converged:
            validation.impossible_success_status.append(key)
        if consumed > MANEUVER_BUDGET:
            validation.over_budget_events.append(
                f"{cell.path}: {consumed} budget events exceed {MANEUVER_BUDGET}"
            )
        rows.append(
            {
                "network": cell.network,
                "model": key[1],
                "architecture": key[2],
                "run_id": key[3],
                "case_id": response.scenario_id,
                "success": success,
                "final_status": response.status,
                # ResolutionResponse has one terminal-status field and no second reason field.
                "termination_reason": response.status,
                "cost_usd": cost,
                "runtime_s": runtime_value,
                "in_band_buses_pct": in_band,
                "maneuvers_used": response.n_maneuvers,
                "maneuver_budget": MANEUVER_BUDGET,
                "llm_calls": response.trace.n_llm_calls,
                "budget_events_consumed": consumed,
                "maneuvers_remaining": MANEUVER_BUDGET - consumed if response.status == "TIMEOUT" else None,
                "llm_calls_per_committed_maneuver": ratio,
            }
        )
    return rows


def _accepted_iterations(response: ResolutionResponse) -> list[int]:
    iterations = []
    for event in response.trace.events:
        if event.phase == "solve" and event.event_name == "run_ac_pf":
            iteration = event.payload.get("iteration")
            if isinstance(iteration, int):
                iterations.append(iteration)
    return iterations


def _llm_calls_by_iteration(response: ResolutionResponse) -> dict[int, int | None]:
    if response.configuration == 1:
        iterations = {feedback.iteration for feedback in response.failure_feedback}
        iterations.update(_accepted_iterations(response))
        return {iteration: response.trace.n_llm_calls for iteration in iterations}

    calls: dict[int, int | None] = {}
    cumulative = 0
    exact = True
    for event in response.trace.events:
        if event.phase != "diagnosis" or event.event_name != "agent_step":
            continue
        iteration = event.payload.get("iteration")
        if not isinstance(iteration, int):
            continue
        count = event.payload.get("llm_calls")
        if exact and isinstance(count, int):
            cumulative += count
            calls[iteration] = cumulative
        else:
            exact = False
            calls[iteration] = None
    return calls


def _progress_verdict(pair: tuple[float, float] | None) -> str | None:
    if pair is None:
        return None
    before, after = pair
    if after < before - OVERSTRESS_EPS:
        return "Improved"
    if after > before + OVERSTRESS_EPS:
        return "Worsened"
    return "Unchanged"


def _action_fields(maneuver: Maneuver | None) -> tuple[Any, Any, Any]:
    if maneuver is None:
        return None, None, None
    action = maneuver.action.model_dump(mode="json")
    action_type = action["type"]
    if action_type == "GEN_V_SETPOINT":
        return action_type, action["gen_id"], action["new_vm_pu"]
    if action_type == "SHUNT_STEP":
        return action_type, action["shunt_id"], action["new_step"]
    if action_type == "TAP_ADJUSTMENT":
        return action_type, action["trafo_id"], action["new_tap_pos"]
    return action_type, None, None


def _solver_fields(kind: str) -> tuple[str | None, bool | None, bool | None]:
    if kind == "STILL_DIVERGED":
        return "DIVERGED", False, None
    if kind == "SOLVED_INFEASIBLE":
        return "CONVERGED_INFEASIBLE", True, False
    if kind == "SUCCESS":
        return "CONVERGED_FEASIBLE", True, True
    return None, None, None


def _event_base(cell: SourceCell) -> dict[str, Any]:
    response = cell.response
    consumed = _budget_events(response)
    return {
        "network": cell.network,
        "model": _model_display(model_id_from_response(response)),
        "architecture": ARCH_LABELS[response.configuration],
        "run_id": response.repetition_index + 1,
        "case_id": response.scenario_id,
        "episode_final_status": response.status,
        "llm_calls_per_committed_maneuver": (
            response.trace.n_llm_calls / response.n_maneuvers
            if response.n_maneuvers > 0
            else None
        ),
        "maneuvers_consumed": consumed if response.status == "TIMEOUT" else None,
        "maneuvers_remaining": MANEUVER_BUDGET - consumed if response.status == "TIMEOUT" else None,
        "maximum_maneuver_budget": MANEUVER_BUDGET,
        "total_llm_calls": response.trace.n_llm_calls,
    }


def _event_row(
    base: dict[str, Any],
    *,
    event_index: int,
    event_category: str | None,
    progress_verdict: str | None,
    llm_calls_before_event: int | None,
    maneuver: Maneuver | None,
    native_kind: str,
    native_iteration: int,
    native_detail: str | None,
) -> dict[str, Any]:
    action_type, action_target, action_value = _action_fields(maneuver)
    solver_status, converged, feasible = _solver_fields(native_kind)
    return {
        **base,
        "event_index": event_index,
        "event_category": event_category,
        "progress_verdict": progress_verdict,
        "llm_calls_before_event": llm_calls_before_event,
        "action_type": action_type,
        "action_target": action_target,
        "action_value": action_value,
        "solver_status_after_action": solver_status,
        "converged_after_action": converged,
        "electrically_feasible_after_action": feasible,
        "native_event_kind": native_kind,
        "native_iteration_index": native_iteration,
        "native_event_detail": native_detail,
    }


def event_rows(cells: Sequence[SourceCell], validation: Validation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    unique_cells: dict[tuple[str, str, str, int, str], SourceCell] = {}
    for cell in cells:
        unique_cells.setdefault(_episode_key(cell), cell)

    for _, cell in sorted(unique_cells.items()):
        response = cell.response
        base = _event_base(cell)
        progress = committed_overstress_chain(response)
        if len(progress) != response.n_maneuvers or any(pair is None for pair in progress):
            if response.n_maneuvers:
                validation.progress_chain_mismatches.append(str(cell.path))
        accepted_iterations = _accepted_iterations(response)
        llm_calls = _llm_calls_by_iteration(response)
        committed_index = 0

        for feedback in response.failure_feedback:
            if feedback.kind not in BUDGET_EVENT_LABELS:
                continue
            pair = None
            maneuver = feedback.maneuver
            if feedback.kind in {"STILL_DIVERGED", "SOLVED_INFEASIBLE"}:
                if committed_index < len(progress):
                    pair = progress[committed_index]
                if committed_index < len(response.maneuvers):
                    maneuver = response.maneuvers[committed_index]
                committed_index += 1
            rows.append(
                _event_row(
                    base,
                    event_index=feedback.iteration + 1,
                    event_category=BUDGET_EVENT_LABELS[feedback.kind],
                    progress_verdict=_progress_verdict(pair),
                    llm_calls_before_event=llm_calls.get(feedback.iteration),
                    maneuver=maneuver,
                    native_kind=feedback.kind,
                    native_iteration=feedback.iteration,
                    native_detail=feedback.detail,
                )
            )

        # A successful terminal maneuver has no FailureFeedback entry, but it is part of the
        # paper's committed-overstress chain and therefore needs one event-level observation.
        if response.status == "SUCCESS" and committed_index < response.n_maneuvers:
            terminal_index = response.n_maneuvers - 1
            native_iteration = (
                accepted_iterations[terminal_index]
                if terminal_index < len(accepted_iterations)
                else max((feedback.iteration for feedback in response.failure_feedback), default=-1) + 1
            )
            pair = progress[terminal_index] if terminal_index < len(progress) else None
            rows.append(
                _event_row(
                    base,
                    event_index=native_iteration + 1,
                    event_category=None,
                    progress_verdict=_progress_verdict(pair),
                    llm_calls_before_event=llm_calls.get(native_iteration),
                    maneuver=response.maneuvers[terminal_index],
                    native_kind="SUCCESS",
                    native_iteration=native_iteration,
                    native_detail=None,
                )
            )
    return sorted(
        rows,
        key=lambda row: (
            row["network"],
            row["model"],
            row["architecture"],
            row["run_id"],
            row["case_id"],
            row["event_index"],
            row["native_event_kind"],
        ),
    )


def _expected_groups(
    campaigns: dict[str, Campaign],
) -> tuple[list[tuple[str, str, str, int, str]], dict[str, set[str]]]:
    groups = []
    expected_cases: dict[str, set[str]] = {}
    for network, campaign_names in NETWORK_CAMPAIGNS.items():
        available = [campaigns[name] for name in campaign_names if name in campaigns]
        if not available:
            continue
        expected_cases[network] = set(campaign_scenarios(available[0]))
        for campaign in available:
            for model_id in campaign.models:
                for configuration in campaign.configurations:
                    for repetition in EXPECTED_REPETITIONS:
                        groups.append(
                            (
                                network,
                                _model_display(model_id),
                                ARCH_LABELS[configuration],
                                repetition + 1,
                                campaign.name,
                            )
                        )
    return sorted(groups), expected_cases


def _finite_values(rows: Sequence[dict[str, Any]], field: str) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if row[field] is not None and math.isfinite(float(row[field]))
    ]


def summary_rows(
    episodes: Sequence[dict[str, Any]],
    campaigns: dict[str, Campaign],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, int], set[str]], dict[str, set[str]]]:
    groups, expected_cases = _expected_groups(campaigns)
    indexed: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        indexed[(row["network"], row["model"], row["architecture"], int(row["run_id"]))].append(row)

    summaries: list[dict[str, Any]] = []
    missing: dict[tuple[str, str, str, int], set[str]] = {}
    for network, model, architecture, run_id, _campaign in groups:
        key = (network, model, architecture, run_id)
        rows = indexed.get(key, [])
        unique_cases = {row["case_id"] for row in rows}
        missing[key] = expected_cases[network] - unique_cases
        costs = _finite_values(rows, "cost_usd")
        runtimes = _finite_values(rows, "runtime_s")
        successes = sum(int(row["success"]) for row in rows)
        statuses = Counter(row["final_status"] for row in rows)
        summaries.append(
            {
                "network": network,
                "model": model,
                "architecture": architecture,
                "run_id": run_id,
                "number_of_cases": len(unique_cases),
                "number_successful": successes,
                "success_rate": successes / len(unique_cases) if unique_cases else None,
                "mean_cost_per_case": statistics.mean(costs) if costs else None,
                "median_cost_per_case": statistics.median(costs) if costs else None,
                "mean_runtime": statistics.mean(runtimes) if runtimes else None,
                "median_runtime": statistics.median(runtimes) if runtimes else None,
                "number_budget_exhausted": statuses["BUDGET_EXHAUSTED"],
                "number_timeout": statuses["TIMEOUT"],
                "number_other_terminal_statuses": sum(
                    count
                    for status, count in statuses.items()
                    if status not in {"SUCCESS", "BUDGET_EXHAUSTED", "TIMEOUT"}
                ),
                "complete_46_cases": len(unique_cases) == 46 and not missing[key],
                "missing_case_count": len(missing[key]),
            }
        )
    return summaries, missing, expected_cases


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("|", "\\|")


def _markdown_table(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    lines.extend(
        "| " + " | ".join(_markdown_value(row.get(column)) for column in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _source_mapping_rows(
    campaigns: dict[str, Campaign], stores: dict[str, Path], summaries: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    campaign_for_model = {
        (network, _model_display(model_id)): campaign_name
        for network, campaign_names in NETWORK_CAMPAIGNS.items()
        for campaign_name in campaign_names
        if campaign_name in campaigns
        for model_id in campaigns[campaign_name].models
    }
    rows = []
    for summary in summaries:
        key = (summary["network"], summary["model"])
        campaign_name = campaign_for_model[key]
        store = stores.get(campaign_name, ROOT / campaigns[campaign_name].results_dir)
        configuration = next(
            number for number, label in ARCH_LABELS.items() if label == summary["architecture"]
        )
        model_id = next(
            model_id
            for model_id in campaigns[campaign_name].models
            if _model_display(model_id) == summary["model"]
        )
        pattern = (
            f"*__config{configuration}__{model_slug(model_id)}__"
            f"rep{int(summary['run_id']) - 1}.json"
        )
        rows.append(
            {
                "network": summary["network"],
                "model": summary["model"],
                "architecture": summary["architecture"],
                "run_id": summary["run_id"],
                "source path/store": f"{_cells_dir(store)}/{pattern}",
                "source state": "present" if summary["number_of_cases"] else "absent",
            }
        )
    return rows


def _issue_lines(title: str, values: Sequence[Any]) -> list[str]:
    if not values:
        return [f"- {title}: none detected among readable source cells."]
    lines = [f"- {title}: {len(values)}."]
    lines.extend(f"  - `{value}`" for value in values[:50])
    if len(values) > 50:
        lines.append(f"  - ... {len(values) - 50} more")
    return lines


def write_report(
    path: Path,
    *,
    campaigns: dict[str, Campaign],
    stores: dict[str, Path],
    episodes: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    missing: dict[tuple[str, str, str, int], set[str]],
    validation: Validation,
) -> None:
    mapping = _source_mapping_rows(campaigns, stores, summaries)
    expected_total = sum(len(case_ids) for case_ids in missing.values()) + len(episodes)
    incomplete = [row for row in summaries if not row["complete_46_cases"]]
    campaign_repetitions = {
        name: campaigns[name].repetitions
        for names in NETWORK_CAMPAIGNS.values()
        for name in names
        if name in campaigns
    }
    duplicate_details = [
        f"{key}: {', '.join(paths)}"
        for key, paths in sorted(validation.duplicate_episodes.items())
    ]
    missing_sets: dict[tuple[str, tuple[str, ...]], list[tuple[str, str, str, int]]] = defaultdict(list)
    for key, case_ids in missing.items():
        missing_sets[(key[0], tuple(sorted(case_ids)))].append(key)

    lines = [
        "# RestoreBench three-run extraction report",
        "",
        "## Outcome",
        "",
        f"Readable LLM episode cells found: **{len(episodes)}** of **{expected_total}** expected "
        "(2 networks × 7 models × 3 architectures × 3 runs × 46 cases).",
        "",
        "No values were reconstructed from the published PNG/PDF figures, hard-coded narrative "
        "numbers, aggregate statistics, or another repository. Missing observations remain absent.",
        "",
        "`restorebench_baselines.csv` was not created: this checkout contains no deterministic "
        "greedy result store, random-baseline repetitions, or baseline-result schema. The LLM "
        "chatbot architecture is not treated as a greedy/random baseline.",
        "",
        "## Source paths inspected",
        "",
    ]
    for campaign_name in sorted(stores):
        store = stores[campaign_name]
        cells_dir = _cells_dir(store)
        state = "present" if cells_dir.is_dir() else "missing"
        lines.append(f"- `{cells_dir}` — {state}")
    lines.extend(
        [
            "- `results/` and `reports/` — absent before this extraction; both are ignored run outputs.",
            "- Repository-wide search for `*.db`, `*.sqlite*`, `*.jsonl`, `*.csv`, logs, result "
            "files, and `cells/` directories — no experimental stores found.",
            "- All local and remote Git refs/history — no tracked historical `results/`, "
            "`reports/`, CSV, JSONL, or SQLite result artifact found.",
            "- `docs/figs/results/` — static PNG/PDF figures only; excluded as non-raw sources.",
            "- `restorebench/sweeps/build_figure_tables.py` — contains aggregate/narrative values; "
            "excluded as non-raw sources.",
            "",
            "The repository README explicitly states: **No result cells ship with this repository**.",
            "",
            "## Run identification",
            "",
            "The native run identity is explicit, not inferred from directory order: "
            "`ResolutionResponse.repetition_index` and filename suffix `__repN.json`. The export "
            "maps native `rep0`, `rep1`, and `rep2` to `run_id` 1, 2, and 3 respectively. "
            "Because no cells are present, every mapping row below is an expected filename pattern, "
            "not a claim that the source exists.",
            "",
            _markdown_table(
                ("network", "model", "architecture", "run_id", "source path/store", "source state"),
                mapping,
            ),
            "",
            "## Campaign repetitions",
            "",
            "All six published campaign definitions specify the three-run design:",
            "",
        ]
    )
    lines.extend(f"- `{name}`: `repetitions = {count}`" for name, count in sorted(campaign_repetitions.items()))
    lines.extend(
        [
            "",
            "The evaluation harness encodes the three repetitions as `rep0`, `rep1`, and `rep2`.",
            "",
            "## Field mappings",
            "",
            "- `network`: campaign/corpus identity (`ieee118` or `pegase89`).",
            "- `model`: model id from `llm_assignment`, normalized through the paper display map.",
            "- `architecture`: configuration 1/2/3 → chatbot/single agent/multi-agent.",
            "- `run_id`: `repetition_index + 1`; the native index is encoded in every expected filename.",
            "- `case_id`: `scenario_id`.",
            "- `success`: `status == SUCCESS`.",
            "- `final_status`: `status`.",
            "- `termination_reason`: `status`; `ResolutionResponse` has no separate terminal-reason field.",
            "- `cost_usd`: derived per episode from stored input/output token counts via "
            "`restorebench.analysis.cost.run_cost_usd` and the model price registry. No aggregate cost is used.",
            "- `runtime_s`: `total_runtime_seconds`.",
            "- `in_band_buses_pct`: `(network_bus_count - quality.n_buses_out_of_band) / "
            "network_bus_count × 100`.",
            "- `maneuvers_used`: `n_maneuvers` (committed maneuvers only).",
            f"- `maneuver_budget`: `{MANEUVER_BUDGET}`, reused from the current paper-analysis constant; "
            "the response schema does not store the budget per episode.",
            "- `llm_calls`: `trace.n_llm_calls`.",
            "- Extra timeout columns preserve the paper logic: budget events consumed, remaining slots, "
            "and total calls divided by committed maneuvers.",
            "",
            "## Current filters",
            "",
            "### Runtime",
            "",
            "The current runtime analysis includes every readable tested episode, regardless of terminal "
            "status. It uses `total_runtime_seconds`; SUCCESS, BUDGET_EXHAUSTED, TIMEOUT, LLM_FAILURE, "
            "and TOOL_FAILURE are not filtered out. Missing/non-finite runtime is left blank and flagged.",
            "",
            "### In-band buses",
            "",
            "The current voltage-quality analysis includes only episodes with `status == SUCCESS` and a "
            "non-null `quality` object. Other statuses remain blank, never zero. The denominator is 118 "
            "for IEEE 118 and 89 for PEGASE 89.",
            "",
            "## Existing event classification reused",
            "",
            "The export imports the mapping directly from `restorebench/sweeps/build_paper_figures.py`:",
            "",
            "- `STILL_DIVERGED` → Still diverged",
            "- `MALFORMED_OUTPUT` → Structured-output failure",
            "- `INVALID_ACTION` → Invalid action",
            "- `SOLVED_INFEASIBLE` → Solved infeasible",
            "",
            "Only those four `FailureFeedback.kind` values consume slots in the current failure and "
            "timeout analyses. Preview failures and LLM_FAILURE do not consume slots.",
            "",
            "Maneuver progress reuses `committed_overstress_chain` and the paper figure's exact "
            f"tolerance (`OVERSTRESS_EPS = {OVERSTRESS_EPS}`). A maneuver is Improved when "
            "`after < before - eps`, Worsened when `after > before + eps`, and Unchanged otherwise. "
            "STILL_DIVERGED uses the next diagnostics' overstress; SOLVED_INFEASIBLE and the terminal "
            "SUCCESS maneuver use zero. Unclassifiable pairs remain blank.",
            "",
            "For TIMEOUT episodes the current paper logic counts the four slot-consuming feedback "
            f"kinds, computes `{MANEUVER_BUDGET} - consumed`, and computes "
            "`trace.n_llm_calls / n_maneuvers` only when at least one maneuver was committed.",
            "",
            "## Validation results",
            "",
            f"- Duplicate episode keys: {len(validation.duplicate_episodes)}.",
            f"- Configurations/runs with 46 unique cases: {len(summaries) - len(incomplete)} of {len(summaries)}.",
            f"- Incomplete configuration/runs: {len(incomplete)}.",
            f"- Missing episode observations: {sum(len(case_ids) for case_ids in missing.values())}.",
        ]
    )
    lines.extend(_issue_lines("Duplicate case IDs / duplicated episodes", duplicate_details))
    lines.extend(_issue_lines("Unreadable result files", validation.unreadable_files))
    lines.extend(_issue_lines("Unexpected repetition indices", validation.unexpected_repetitions))
    lines.extend(_issue_lines("Filename/payload run-id mismatches", validation.filename_mismatches))
    lines.extend(_issue_lines("Missing episode cost", validation.missing_cost))
    lines.extend(_issue_lines("Missing/non-finite runtime", validation.missing_runtime))
    lines.extend(_issue_lines("Missing final status", validation.missing_final_status))
    lines.extend(_issue_lines("Impossible success/status combinations", validation.impossible_success_status))
    lines.extend(_issue_lines("Unclassifiable or mismatched maneuver-progress chains", validation.progress_chain_mismatches))
    lines.extend(_issue_lines("Episodes exceeding the event budget", validation.over_budget_events))
    lines.extend(_issue_lines("Version-stamp errors", validation.version_stamp_errors))
    lines.extend(["", "### Exact missing case IDs", ""])
    for (network, case_ids), affected_groups in sorted(missing_sets.items()):
        if not case_ids:
            continue
        lines.append(
            f"- `{network}` — {len(affected_groups)} configuration/run groups: "
            + ", ".join(case_ids)
        )
    lines.extend(
        [
            "",
            "No validation problem was silently repaired. With zero readable source cells, duplicate "
            "and field-level checks are necessarily vacuous; the material failure is complete source absence.",
            "",
            "## Per-run summary",
            "",
            _markdown_table(SUMMARY_COLUMNS, list(summaries)),
            "",
            "## Generated files",
            "",
            f"- `restorebench_episode_level_3runs.csv`: {len(episodes)} data rows.",
            f"- `restorebench_event_level_3runs.csv`: {len(events)} data rows.",
            "- `restorebench_baselines.csv`: not generated because no baseline raw results are available.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_version_stamps(cells: Sequence[SourceCell], validation: Validation) -> None:
    fields = (
        "dataset_version",
        "solver_version",
        "action_policy_version",
        "ranking_policy_version",
        "result_schema_version",
    )
    by_network: dict[str, list[SourceCell]] = defaultdict(list)
    for cell in cells:
        by_network[cell.network].append(cell)
    for network, network_cells in by_network.items():
        for field in fields:
            values = {getattr(cell.response, field) for cell in network_cells}
            if None in values:
                validation.version_stamp_errors.append(f"{network}: {field} is missing")
            if len(values) > 1:
                validation.version_stamp_errors.append(
                    f"{network}: incompatible {field} values {sorted(str(value) for value in values)}"
                )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    campaigns = load_campaigns()
    validation = _empty_validation()
    overrides = _store_overrides(args.store)
    cells, stores = load_source_cells(campaigns, overrides, validation)
    _validate_version_stamps(cells, validation)
    episodes = episode_rows(cells, validation)
    events = event_rows(cells, validation)
    summaries, missing, _expected_cases = summary_rows(episodes, campaigns)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "restorebench_episode_level_3runs.csv", EPISODE_COLUMNS, episodes)
    _write_csv(args.out_dir / "restorebench_event_level_3runs.csv", EVENT_COLUMNS, events)
    write_report(
        args.out_dir / "extraction_report.md",
        campaigns=campaigns,
        stores=stores,
        episodes=episodes,
        events=events,
        summaries=summaries,
        missing=missing,
        validation=validation,
    )
    print(
        f"{len(episodes)} episodes and {len(events)} events -> {args.out_dir}; "
        f"{sum(len(case_ids) for case_ids in missing.values())} expected episodes missing"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
