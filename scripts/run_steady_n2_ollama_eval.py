from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from poweragentbench.llm_agent_adapter import LLMToolAgent, load_prompt_template, parse_json_command
from poweragentbench.ollama_client import OllamaGenerateClient
from poweragentbench.steady_state_agentic import (
    aggregate_metrics,
    contingency_space,
    evaluate_contingencies,
    latex_result_row,
    latex_result_row_with_diagnostics,
    load_case39_dc,
    make_synthetic_case,
    score_agent,
)

DEFAULT_ENV_FILES = [
    Path(".env"),
    Path("benchmarks/steady/level_2/.env"),
]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def load_env_file(path: Path, *, override: bool = False) -> None:
    """Load KEY=VALUE pairs without adding a python-dotenv dependency."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_model_list(value: str | None) -> List[str] | None:
    if not value:
        return None
    # Accept either comma-separated or whitespace-separated lists.
    normalized = value.replace(",", " ")
    models = [m.strip() for m in normalized.split() if m.strip()]
    return models or None


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=Path, default=None, help="Optional .env file with Ollama settings.")
    pre_args, _ = pre.parse_known_args()
    for path in DEFAULT_ENV_FILES:
        load_env_file(path)
    if pre_args.env_file is not None:
        load_env_file(pre_args.env_file, override=True)

    env_models = parse_model_list(os.getenv("POWERAGENTBENCH_OLLAMA_MODELS"))
    parser = argparse.ArgumentParser(
        description="Evaluate one or more Ollama LLM agents on PowerAgentBench-SS N-2.",
        parents=[pre],
    )
    parser.add_argument(
        "--url",
        default=os.getenv("POWERAGENTBENCH_OLLAMA_URL"),
        help="Ollama endpoint. Can also be set with POWERAGENTBENCH_OLLAMA_URL in .env.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=env_models,
        help="Ollama model names. Can also be set with POWERAGENTBENCH_OLLAMA_MODELS in .env.",
    )
    parser.add_argument("--case-source", choices=["case39", "synthetic"], default="case39")
    parser.add_argument("--network", type=Path, default=None, help="Optional PyPSA netCDF path for case39.")
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--report-k", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--rating-scale", type=float, default=0.85)
    parser.add_argument(
        "--api-mode",
        choices=["generate", "chat"],
        default=os.getenv("POWERAGENTBENCH_OLLAMA_API_MODE", "generate"),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("POWERAGENTBENCH_OLLAMA_TEMPERATURE", "0.0")),
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=int(os.getenv("POWERAGENTBENCH_OLLAMA_NUM_CTX", "16384")),
        help="Ollama context length passed as options.num_ctx.",
    )
    parser.add_argument("--no-schema-format", action="store_true", default=not env_bool("POWERAGENTBENCH_OLLAMA_SCHEMA_FORMAT", True))
    parser.add_argument("--think", choices=["true", "false", "none"], default=os.getenv("POWERAGENTBENCH_OLLAMA_THINK", "false"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/steady_n2"))
    parser.add_argument("--prompt-template", type=Path, default=None, help="Optional JSON prompt template with a system field.")
    parser.add_argument(
        "--require-submit",
        action="store_true",
        help="If set, agents that do not call submit receive an empty reported list instead of auto-finalization.",
    )
    args = parser.parse_args()
    if not args.url:
        raise SystemExit("Missing Ollama endpoint. Set --url or POWERAGENTBENCH_OLLAMA_URL in .env.")
    if not args.models:
        raise SystemExit("Missing model names. Set --models or POWERAGENTBENCH_OLLAMA_MODELS in .env.")
    return args


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
    think_value = None if args.think == "none" else (args.think == "true")
    system_prompt = load_prompt_template(args.prompt_template)
    all_summaries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for model in args.models:
        client = OllamaGenerateClient(
            url=args.url,
            model=model,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            api_mode=args.api_mode,
            schema_format=not args.no_schema_format,
            think=think_value,
        )
        agent_name = f"{model}-Ollama"
        # Interface check before spending the benchmark run.
        probe = client([
            {"role": "system", "content": "Return exactly one JSON command."},
            {"role": "user", "content": json.dumps({"allowed_tools": ["case_summary"], "example": {"tool": "case_summary", "args": {}}})},
        ])
        parse_json_command(probe)

        rows: List[Dict[str, Any]] = []
        tool_log_path = args.output_dir / f"{slug(model)}_tool_logs.jsonl"
        debug_path = args.output_dir / f"{slug(model)}_api_debug.jsonl"
        with tool_log_path.open("w", encoding="utf-8") as log_fh, debug_path.open("w", encoding="utf-8") as debug_fh:
            for i in range(args.cases):
                seed = args.seed_start + i
                case = make_case(args, seed)
                candidates = contingency_space(case, args.k)
                oracle = evaluate_contingencies(case, candidates)
                agent = LLMToolAgent(
                    llm=client,
                    validation_budget=args.budget,
                    report_k=args.report_k,
                    max_turns=args.max_turns,
                    name=agent_name,
                    system_prompt=system_prompt,
                    require_submit=args.require_submit,
                )
                out = agent.run(case, candidates)
                metrics = score_agent(case, out, oracle, top_m=args.report_k)
                metrics["case_seed"] = seed
                metrics["n_candidates"] = len(candidates)
                rows.append(metrics)
                all_rows.append(metrics)
                log_fh.write(json.dumps({"model": model, "case_seed": seed, "tool_log": out.tool_log}) + "\n")
                # Do not record endpoint URL or other local secrets in debug logs.
                debug_fh.write(json.dumps({"model": model, "case_seed": seed, "last_api_debug": client.last_debug}) + "\n")
                print(json.dumps(metrics, indent=2))
        summary = aggregate_metrics(rows)
        summary["agent"] = agent_name
        all_summaries.append(summary)
        write_csv(args.output_dir / f"{slug(model)}_per_case.csv", rows)
        write_csv(args.output_dir / f"{slug(model)}_summary.csv", [summary])
        print("Aggregate summary:")
        print(json.dumps(summary, indent=2))
        print("LaTeX row:")
        print(latex_result_row(summary, label=agent_name))
        print("LaTeX row with diagnostics:")
        print(latex_result_row_with_diagnostics(summary, label=agent_name))
    write_csv(args.output_dir / "ollama_all_per_case.csv", all_rows)
    write_csv(args.output_dir / "ollama_all_summary.csv", all_summaries)


if __name__ == "__main__":
    main()
