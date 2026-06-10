from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from poweragentbench.llm_agent_adapter import LLMToolAgent, load_prompt_template, parse_json_command
from poweragentbench.openai_client import OpenAIResponsesClient
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
    normalized = value.replace(",", " ")
    models = [m.strip() for m in normalized.split() if m.strip()]
    return models or None


def parse_optional_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"", "none", "null", "omit", "default"}:
        return None
    return float(text)


def parse_optional_int(value: str | int | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null", "omit", "default"}:
        return None
    return int(text)


def env_optional_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if value.lower() in {"", "none", "null", "omit", "default"}:
        return None
    return value


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=Path, default=None, help="Optional .env file with OpenAI settings.")
    pre_args, _ = pre.parse_known_args()
    for path in DEFAULT_ENV_FILES:
        load_env_file(path)
    if pre_args.env_file is not None:
        load_env_file(pre_args.env_file, override=True)

    env_models = parse_model_list(os.getenv("POWERAGENTBENCH_OPENAI_MODELS"))
    parser = argparse.ArgumentParser(
        description="Evaluate one or more OpenAI/ChatGPT LLM agents on PowerAgentBench-SS N-2.",
        parents=[pre],
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("POWERAGENTBENCH_OPENAI_API_KEY"),
        help="OpenAI API key. Prefer POWERAGENTBENCH_OPENAI_API_KEY in .env.",
    )
    parser.add_argument(
        "--url",
        default=os.getenv("POWERAGENTBENCH_OPENAI_URL", "https://api.openai.com/v1/responses"),
        help="OpenAI Responses API endpoint.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=env_models or [os.getenv("POWERAGENTBENCH_OPENAI_MODEL", "gpt-5.5")],
        help="OpenAI model names. Can also be set with POWERAGENTBENCH_OPENAI_MODELS in .env.",
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
        "--temperature",
        default=os.getenv("POWERAGENTBENCH_OPENAI_TEMPERATURE", "none"),
        help="Optional OpenAI temperature. Use 'none' to omit. Some reasoning models reject this parameter.",
    )
    parser.add_argument(
        "--max-output-tokens",
        default=os.getenv("POWERAGENTBENCH_OPENAI_MAX_OUTPUT_TOKENS", "4096"),
        help="Max output tokens. Use 'none' to omit.",
    )
    parser.add_argument(
        "--structured-outputs",
        action="store_true",
        default=env_bool("POWERAGENTBENCH_OPENAI_STRUCTURED_OUTPUTS", True),
        help="Use Responses API text.format json_schema.",
    )
    parser.add_argument(
        "--no-structured-outputs",
        action="store_false",
        dest="structured_outputs",
        help="Use JSON object mode instead of json_schema structured outputs.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=env_optional_str("POWERAGENTBENCH_OPENAI_REASONING_EFFORT", None),
        help="Optional reasoning effort, e.g. low, medium, high. Omit for model default.",
    )
    parser.add_argument(
        "--reasoning-summary",
        default=env_optional_str("POWERAGENTBENCH_OPENAI_REASONING_SUMMARY", None),
        help="Optional reasoning summary setting. Summaries are not parsed or scored.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("POWERAGENTBENCH_OPENAI_TIMEOUT", "300")),
        help="Per-request HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("POWERAGENTBENCH_OPENAI_MAX_RETRIES", "3")),
        help="Retries for transient API errors and read timeouts.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=float(os.getenv("POWERAGENTBENCH_OPENAI_RETRY_BACKOFF", "2.0")),
        help="Initial retry backoff in seconds. Uses exponential backoff.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=env_bool("POWERAGENTBENCH_OPENAI_CONTINUE_ON_ERROR", False),
        help="Log API/runtime failures and continue with later cases. Aggregates only successful cases.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/steady_n2_openai"))
    parser.add_argument("--prompt-template", type=Path, default=None, help="Optional JSON prompt template with a system field.")
    parser.add_argument(
        "--require-submit",
        action="store_true",
        help="If set, agents that do not call submit receive an empty reported list instead of auto-finalization.",
    )
    args = parser.parse_args()
    args.temperature = parse_optional_float(args.temperature)
    args.max_output_tokens = parse_optional_int(args.max_output_tokens, default=4096)
    if not args.api_key:
        raise SystemExit("Missing OpenAI API key. Set --api-key or POWERAGENTBENCH_OPENAI_API_KEY in .env.")
    if not args.models:
        raise SystemExit("Missing model names. Set --models or POWERAGENTBENCH_OPENAI_MODELS in .env.")
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


def write_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = load_prompt_template(args.prompt_template)
    all_summaries: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for model in args.models:
        client = OpenAIResponsesClient(
            api_key=args.api_key,
            url=args.url,
            model=model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            structured_outputs=args.structured_outputs,
            reasoning_effort=args.reasoning_effort,
            reasoning_summary=args.reasoning_summary,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )
        agent_name = f"{model}-OpenAI"
        agent_slug = slug(agent_name)
        rows: List[Dict[str, Any]] = []
        tool_log_path = args.output_dir / f"{agent_slug}_tool_logs.jsonl"
        debug_path = args.output_dir / f"{agent_slug}_api_debug.jsonl"
        error_path = args.output_dir / f"{agent_slug}_errors.jsonl"

        # Interface check before spending the benchmark run.
        probe = client([
            {"role": "system", "content": "Return exactly one JSON command."},
            {"role": "user", "content": json.dumps({"allowed_tools": ["case_summary"], "example": {"tool": "case_summary", "args": {}}})},
        ])
        parse_json_command(probe)

        for i in range(args.cases):
            seed = args.seed_start + i
            try:
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
                write_jsonl(tool_log_path, {"model": model, "case_seed": seed, "tool_log": out.tool_log})
                # Do not record API keys, raw output text, or reasoning text in debug logs.
                write_jsonl(debug_path, {"model": model, "case_seed": seed, "last_api_debug": client.sanitized_debug()})
                print(json.dumps(metrics, indent=2))
            except Exception as exc:
                write_jsonl(
                    error_path,
                    {
                        "model": model,
                        "case_seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "last_api_debug": client.sanitized_debug(),
                    },
                )
                # Preserve successful rows before stopping.
                write_csv(args.output_dir / f"{agent_slug}_per_case.csv", rows)
                if rows:
                    partial_summary = aggregate_metrics(rows)
                    partial_summary["agent"] = agent_name
                    partial_summary["completed_cases"] = len(rows)
                    partial_summary["requested_cases"] = args.cases
                    write_csv(args.output_dir / f"{agent_slug}_summary_partial.csv", [partial_summary])
                if not args.continue_on_error:
                    raise
                print(json.dumps({"agent": agent_name, "case_seed": seed, "error": str(exc)}, indent=2))

        if not rows:
            continue
        summary = aggregate_metrics(rows)
        summary["agent"] = agent_name
        summary["completed_cases"] = len(rows)
        summary["requested_cases"] = args.cases
        all_summaries.append(summary)
        write_csv(args.output_dir / f"{agent_slug}_per_case.csv", rows)
        write_csv(args.output_dir / f"{agent_slug}_summary.csv", [summary])
        print("Aggregate summary:")
        print(json.dumps(summary, indent=2))
        print("LaTeX row:")
        print(latex_result_row(summary, label=agent_name))
        print("LaTeX row with diagnostics:")
        print(latex_result_row_with_diagnostics(summary, label=agent_name))

    write_csv(args.output_dir / "openai_all_per_case.csv", all_rows)
    write_csv(args.output_dir / "openai_all_summary.csv", all_summaries)


if __name__ == "__main__":
    main()
