"""Manual Anthropic tool-use loop for one DMQ benchmark run."""
import json
import os
import random
import time
from pathlib import Path

import anthropic

import config
from agent_tools import TOOLS, RunContext

SYSTEM_PROMPT = """\
You are a senior power-systems engineer performing a dynamic model quality \
review of a solar photovoltaic plant model for interconnection. The plant uses \
the WECC second-generation renewable generic models (REGCAU1 + REECAU1 + \
REPCAU1) at bus 91003, machine ID S1, in a PSS/e dynamic case.

The submitted model fails the system operator's model-quality test suite. The \
suspected defect is in the REECAU1 electrical-control gains, which appear too \
aggressive and drive an unstable reactive/voltage response.

ACTION SPACE (strictly enforced): You may modify ONLY the four REECAU1 control \
gains:
  - Kqp : reactive-power-control proportional gain
  - Kqi : reactive-power-control integral gain
  - Kvp : voltage-control proportional gain
  - Kvi : voltage-control integral gain
Each must lie in the range (0, 100]. Every other model parameter - structure, \
bus numbers, current limits, voltage thresholds, protection flags, plant \
controller settings - is LOCKED and cannot be changed.

THE TEST SUITE (8 DMView tests). A model passes only if every test passes:
  - Test01_FS    Flat start, no disturbance: output must hold steady (small \
P/Q drift). Large drift means the model does not even initialize stably.
  - Test02_VOLTDOWN / Test03_VOLTUP  Voltage reference step: reactive and active \
power must settle without sustained oscillation, and active power must recover.
  - Test04_FRQDOWN / Test05_FRQUP  Frequency step: active power must respond \
without collapsing and without sustained oscillation.
  - Test06_HVRT  High-voltage ride-through: ride through the overvoltage and \
recover with no sustained oscillation.
  - Test07_LVRT  Low-voltage ride-through: a deep fault drives voltage to ~0; \
the plant must recover power promptly after the voltage recovers. A long \
"momentary cessation" (power stays near zero long after voltage returns) is a \
failure.
  - Test08_SCR2  Weak-grid stability: the grid short-circuit ratio is stepped \
down (5 -> 3 -> 1.5); the plant must remain stable (no growing voltage/reactive \
oscillation) at each level.

PROCEDURE:
1. You are given the BASELINE test report (the failing model as submitted).
2. You have a budget of {budget} test-suite runs. Each iteration: reason about \
which gains to change and why (cite the specific failing metrics), call \
update_dyr_gains, then call run_test_suite to see the effect.
3. High proportional/integral gains generally cause oscillation and \
overshoot; reducing them damps the response, but gains that are too low can \
slow recovery. Use the metrics (peak-to-peak oscillation, drift, cessation \
duration, per-SCR-level stability) as evidence for each change.
4. Stop early if all 8 tests pass. When done (all pass, or budget exhausted), \
call submit_final_report with your verdict, the final gains, and an \
evidence-based engineering summary.

Always ground your decisions in the actual tool results - never invent test \
outcomes.
"""


def _client():
    return anthropic.Anthropic(max_retries=4, timeout=300.0)


def _create_with_backoff(client, **kwargs):
    for attempt in range(4):
        try:
            return client.messages.create(**kwargs)
        except (anthropic.APIConnectionError, anthropic.InternalServerError,
                anthropic.RateLimitError) as e:
            if attempt == 3:
                raise
            delay = min(2 ** attempt + random.random(), 30)
            ra = getattr(getattr(e, "response", None), "headers", {}) or {}
            try:
                delay = max(delay, float(ra.get("retry-after", 0)))
            except (TypeError, ValueError):
                pass
            time.sleep(delay)


def _log(transcript: Path, kind: str, payload):
    with open(transcript, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "kind": kind, "payload": payload}, default=str) + "\n")


def run_agent(model_id: str, run_dir: Path) -> dict:
    """Execute one full agent run; returns a result summary dict."""
    run_dir = Path(run_dir)
    transcript = run_dir / "transcript.jsonl"
    ctx = RunContext(run_dir)
    ctx.setup()

    baseline = ctx.run_baseline()
    _log(transcript, "baseline_report", baseline)

    system = SYSTEM_PROMPT.format(budget=config.ITERATION_BUDGET)
    user0 = (
        "Here is the BASELINE DMView test report for the submitted model "
        "(all values per-unit). Diagnose the failures and repair the model "
        "within your action space.\n\n"
        + json.dumps(ctx._compact(baseline), indent=2)
    )
    messages = [{"role": "user", "content": user0}]

    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "api_calls": 0}
    error = None

    for turn in range(config.MAX_TURNS):
        if ctx.done:
            break
        kwargs = dict(model=model_id, max_tokens=config.MAX_TOKENS,
                      system=system, tools=TOOLS, messages=messages)
        if model_id in config.ADAPTIVE_THINKING_MODELS:
            kwargs["thinking"] = {"type": "adaptive"}
        try:
            resp = _create_with_backoff(_client(), **kwargs)
        except Exception as e:  # noqa: BLE001
            error = f"api_error: {type(e).__name__}: {e}"
            _log(transcript, "api_error", error)
            break

        usage["api_calls"] += 1
        for k in ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens"):
            usage[k] += getattr(resp.usage, k, 0) or 0
        _log(transcript, "assistant", resp.to_dict())

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "max_tokens":
            messages.append({"role": "user", "content": "continue"})
            continue
        if resp.stop_reason != "tool_use":
            # model ended its turn without a tool call - nudge once toward closing
            messages.append({"role": "user", "content":
                "If you are finished, call submit_final_report. Otherwise "
                "continue the review with update_dyr_gains / run_test_suite."})
            continue

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            result, is_error = ctx.dispatch(block.name, block.input or {})
            _log(transcript, "tool", {"name": block.name, "input": block.input,
                                      "result": result, "is_error": is_error})
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
                **({"is_error": True} if is_error else {}),
            })
        messages.append({"role": "user", "content": tool_results})

    usage["est_cost_usd"] = round(
        config.est_cost_usd(model_id, usage["input_tokens"],
                            usage["output_tokens"]), 4)

    if ctx.final_report and ctx.final_report["verdict"] == "fixed":
        final_report = ctx.final_report
    elif ctx.final_report:
        final_report = ctx.final_report
    else:
        final_report = None

    first_pass = next((h["iteration"] for h in ctx.history
                       if h["overall_pass"]), None)

    result = {
        "model": model_id,
        "error": error,
        "final_report_submitted": ctx.final_report is not None,
        "final_report": final_report,
        "iterations_used": ctx.iterations_used,
        "history": ctx.history,
        "first_pass_iteration": first_pass,
        "overall_pass": first_pass is not None,
        "final_gains": ctx.history[-1]["gains"] if ctx.history else None,
        "final_n_pass": ctx.history[-1]["n_pass"] if ctx.history else 0,
        "usage": usage,
    }
    if final_report:
        (run_dir / "final_report.md").write_text(
            f"# Final Report ({model_id})\n\n"
            f"Verdict: {final_report.get('verdict', '-')}\n\n"
            f"Final gains: {final_report.get('final_gains', '-')}\n\n"
            f"## Reasoning\n{final_report.get('reasoning_summary', '')}\n\n"
            f"## Per-test summary\n{final_report.get('per_test_summary', '')}\n",
            encoding="utf-8")
    return result
