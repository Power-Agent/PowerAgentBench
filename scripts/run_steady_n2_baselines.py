from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from poweragentbench.steady_state_agentic import (
    BaseLoadingAgent,
    DegreeAgent,
    HybridMitigationAgent,
    HybridToolAgent,
    LODFAgent,
    NoValidationHeuristicAgent,
    RandomSearchAgent,
    aggregate_metrics,
    contingency_space,
    evaluate_contingencies,
    latex_result_row,
    load_case39_dc,
    make_synthetic_case,
    score_agent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PowerAgentBench-SS N-2 baseline agents.")
    parser.add_argument("--case-source", choices=["case39", "synthetic"], default="case39")
    parser.add_argument("--network", type=Path, default=None, help="Optional PyPSA netCDF path for case39.")
    parser.add_argument("--cases", type=int, default=8, help="Number of deterministic operating-point variants.")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--report-k", type=int, default=20)
    parser.add_argument("--rating-scale", type=float, default=0.85)
    parser.add_argument("--random-trials", type=int, default=50, help="Number of random-search repeats per case.")
    parser.add_argument("--danger-threshold", type=float, default=None, help="Optional absolute severity threshold for dangerous cases.")
    parser.add_argument("--danger-quantile", type=float, default=0.95, help="Empirical severity quantile used when --danger-threshold is not set.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/steady_n2"))
    return parser.parse_args()


def make_case(args: argparse.Namespace, seed: int):
    if args.case_source == "synthetic":
        return make_synthetic_case(seed=seed, n_bus=24, n_line=36, n_gen=5)
    return load_case39_dc(network_path=args.network, rating_scale=args.rating_scale, variant_seed=seed)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_records: List[Dict[str, Any]] = []
    for i in range(args.cases):
        seed = args.seed_start + i
        case = make_case(args, seed)
        candidates = contingency_space(case, args.k)
        oracle = evaluate_contingencies(case, candidates)
        case_records.append(
            {
                "seed": seed,
                "case": case,
                "candidates": candidates,
                "oracle": oracle,
            }
        )

    agents = [
        NoValidationHeuristicAgent(report_k=args.report_k),
        DegreeAgent(budget=args.budget, report_k=args.report_k),
        BaseLoadingAgent(budget=args.budget, report_k=args.report_k),
        LODFAgent(budget=args.budget, report_k=args.report_k),
        HybridToolAgent(budget=args.budget, report_k=args.report_k),
        HybridMitigationAgent(budget=args.budget, report_k=args.report_k),
    ]

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    if args.random_trials > 0:
        random_rows: List[Dict[str, Any]] = []
        for trial in range(args.random_trials):
            for record in case_records:
                seed = int(record["seed"])
                case = record["case"]
                candidates = record["candidates"]
                oracle = record["oracle"]

                agent = RandomSearchAgent(
                    budget=args.budget,
                    report_k=args.report_k,
                    seed=seed * 100000 + trial,
                )
                out = agent.run(case, candidates)
                metrics = score_agent(
                    case,
                    out,
                    oracle,
                    top_m=args.report_k,
                    danger_threshold=args.danger_threshold,
                    danger_quantile=args.danger_quantile,
                )
                metrics["case_seed"] = seed
                metrics["n_candidates"] = len(candidates)
                metrics["random_trial"] = trial

                random_rows.append(metrics)
                all_rows.append(metrics)

        random_summary = aggregate_metrics(random_rows)
        summaries.append(random_summary)
        print(json.dumps(random_summary, indent=2))
        print(latex_result_row(random_summary))

    for agent in agents:
        rows: List[Dict[str, Any]] = []
        for record in case_records:
            seed = int(record["seed"])
            case = record["case"]
            candidates = record["candidates"]
            oracle = record["oracle"]

            out = agent.run(case, candidates)
            metrics = score_agent(
                case,
                out,
                oracle,
                top_m=args.report_k,
                danger_threshold=args.danger_threshold,
                danger_quantile=args.danger_quantile,
            )
            metrics["case_seed"] = seed
            metrics["n_candidates"] = len(candidates)
            metrics["random_trial"] = -1

            rows.append(metrics)
            all_rows.append(metrics)

        summary = aggregate_metrics(rows)
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
        print(latex_result_row(summary))

    write_csv(args.output_dir / "baseline_per_case.csv", all_rows)
    write_csv(args.output_dir / "baseline_summary.csv", summaries)


if __name__ == "__main__":
    main()
