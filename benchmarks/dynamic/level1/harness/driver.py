"""Benchmark driver: run the 3-model x N-run matrix, with resume.

Usage (PSS/e-bound Python 3.11):
    python driver.py --all
    python driver.py --model claude-haiku-4-5 --runs 1 --budget 1   # smoke
    python driver.py --model claude-opus-4-8 --runs 10

Runs are strictly sequential (PSS/e is single-instance). Each run writes a
checkpoint.json; rerunning the driver skips completed runs and restarts any
partial run from scratch (the old folder is kept as run<NN>_attempt<k>).
"""
import argparse
import json
import time
from pathlib import Path

import agent_loop
import config
import envfile

envfile.load_env(config.ENV_FILE)


def _run_dir(model_id: str, run_id: int) -> Path:
    return config.RUNS_DIR / model_id / f"run{run_id:02d}"


def _checkpoint(run_dir: Path) -> dict | None:
    cp = run_dir / "checkpoint.json"
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _write_checkpoint(run_dir: Path, data: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    cp = run_dir / "checkpoint.json"
    tmp = cp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(cp)


def _archive_partial(run_dir: Path):
    k = 1
    while (run_dir.parent / f"{run_dir.name}_attempt{k}").exists():
        k += 1
    run_dir.rename(run_dir.parent / f"{run_dir.name}_attempt{k}")


def run_one(model_id: str, run_id: int, budget: int) -> dict:
    run_dir = _run_dir(model_id, run_id)
    cp = _checkpoint(run_dir)
    if cp and cp.get("status") == "completed":
        print(f"  [skip] {model_id} run{run_id:02d} already completed")
        return cp
    if run_dir.exists():
        _archive_partial(run_dir)

    config.ITERATION_BUDGET = budget
    t0 = time.time()
    _write_checkpoint(run_dir, {"model": model_id, "run_id": run_id,
                                "status": "in_progress",
                                "iteration_budget": budget,
                                "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(f"  [run]  {model_id} run{run_id:02d} (budget={budget}) ...")
    try:
        result = agent_loop.run_agent(model_id, run_dir)
        status = "failed" if result.get("error") else "completed"
    except Exception as e:  # noqa: BLE001
        result = {"error": f"{type(e).__name__}: {e}"}
        status = "failed"

    cp_out = {
        "model": model_id, "run_id": run_id, "status": status,
        "iteration_budget": budget,
        "error": result.get("error"),
        "overall_pass": result.get("overall_pass"),
        "first_pass_iteration": result.get("first_pass_iteration"),
        "iterations_used": result.get("iterations_used"),
        "final_gains": result.get("final_gains"),
        "final_n_pass": result.get("final_n_pass"),
        "final_report_submitted": result.get("final_report_submitted"),
        "history": result.get("history"),
        "usage": result.get("usage"),
        "elapsed_s": round(time.time() - t0, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_checkpoint(run_dir, cp_out)
    u = result.get("usage", {})
    print(f"         -> status={status} pass={cp_out['overall_pass']} "
          f"iters={cp_out['iterations_used']} "
          f"first_pass={cp_out['first_pass_iteration']} "
          f"cost=${u.get('est_cost_usd', 0)} ({cp_out['elapsed_s']}s)")
    return cp_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--runs", type=int, default=config.RUNS_PER_MODEL)
    ap.add_argument("--budget", type=int, default=config.ITERATION_BUDGET)
    args = ap.parse_args()

    if args.all:
        models = list(config.MODELS.values())
    elif args.model:
        models = [args.model]
    else:
        ap.error("specify --all or --model")

    for model_id in models:
        print(f"=== model {model_id} ===")
        for run_id in range(1, args.runs + 1):
            run_one(model_id, run_id, args.budget)


if __name__ == "__main__":
    main()
