"""Tool schemas and dispatcher for the DMQ agent loop.

A RunContext owns one run's working folder, iteration counter, latest test
report, and logs. The tools are the agent's entire action surface - there is
no filesystem or shell access, so the locked action space (only the four
REECAU1 gains) is enforced structurally.
"""
import json
import shutil
from pathlib import Path

import config
import dmview_runner
import dyr_tools
import evaluator

TOOLS = [
    {
        "name": "get_dyr_parameters",
        "description": "Return the current REECAU1 control gains (Kqp, Kqi, "
                       "Kvp, Kvi) in the working .dyr model, plus the locked "
                       "context (bus, machine id, allowed range).",
        "input_schema": {"type": "object", "properties": {},
                         "additionalProperties": False},
    },
    {
        "name": "update_dyr_gains",
        "description": "Replace the four REECAU1 gains in the working .dyr "
                       "model. Only these four values may be changed; every "
                       "other model parameter is locked. Each gain must be in "
                       "the open-closed range (0, 100]. This does NOT run the "
                       "tests - call run_test_suite afterwards.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kqp": {"type": "number", "description": "Reactive-power-control proportional gain"},
                "kqi": {"type": "number", "description": "Reactive-power-control integral gain"},
                "kvp": {"type": "number", "description": "Voltage-control proportional gain"},
                "kvi": {"type": "number", "description": "Voltage-control integral gain"},
            },
            "required": ["kqp", "kqi", "kvp", "kvi"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_test_suite",
        "description": "Run the full DMView model-quality test suite (8 tests) "
                       "on the current working .dyr model and return a "
                       "structured pass/fail report with per-test metrics. "
                       "Limited by the iteration budget.",
        "input_schema": {"type": "object", "properties": {},
                         "additionalProperties": False},
    },
    {
        "name": "read_log",
        "description": "Return the tail of the raw DMView .log file for a given "
                       "test from the most recent suite run, for detailed "
                       "diagnostics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_name": {"type": "string", "enum": config.TEST_NAMES},
                "max_chars": {"type": "integer", "default": 4000},
            },
            "required": ["test_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_final_report",
        "description": "Submit the final engineering report and end the review. "
                       "Call this when all tests pass, or when the iteration "
                       "budget is exhausted.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["fixed", "not_fixed"]},
                "final_gains": {
                    "type": "object",
                    "properties": {
                        "kqp": {"type": "number"}, "kqi": {"type": "number"},
                        "kvp": {"type": "number"}, "kvi": {"type": "number"},
                    },
                    "required": ["kqp", "kqi", "kvp", "kvi"],
                    "additionalProperties": False,
                },
                "reasoning_summary": {"type": "string"},
                "per_test_summary": {"type": "string"},
            },
            "required": ["verdict", "final_gains", "reasoning_summary",
                         "per_test_summary"],
            "additionalProperties": False,
        },
    },
]


class RunContext:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.workdir = self.run_dir / "work"
        self.dyr = self.workdir / "Solar.dyr"
        self.change_log = self.run_dir / "dyr_changes.log"
        self.iterations_used = 0          # agent-triggered run_test_suite calls
        self.latest_report = None         # most recent report dict
        self.latest_archive = None        # Path to most recent iter folder
        self.history = []                 # list of compact iteration summaries
        self.final_report = None          # set by submit_final_report
        self.done = False

    # ---- setup ----------------------------------------------------------
    def setup(self):
        if self.workdir.exists():
            shutil.rmtree(self.workdir)
        self.workdir.mkdir(parents=True)
        for fn in ("Solar.sav", "Solar.dyr"):
            shutil.copy2(config.PRISTINE_DIR / fn, self.workdir / fn)

    def run_baseline(self):
        """Iteration 0 - run by the harness, not counted against the budget."""
        return self._run_and_eval(0)

    # ---- helpers --------------------------------------------------------
    def _run_and_eval(self, iteration: int) -> dict:
        archive = self.run_dir / f"iter{iteration}"
        res = dmview_runner.run_suite(self.workdir, archive)
        gains = dyr_tools.read_gains(self.dyr)
        report = evaluator.evaluate_suite(
            archive, gains, iteration,
            suite_status=res.status, elapsed_s=res.elapsed_s)
        (archive / "report.json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
        self.latest_report = report
        self.latest_archive = archive
        self.history.append({
            "iteration": iteration, "gains": gains,
            "n_pass": report["n_pass"], "overall_pass": report["overall_pass"],
            "suite_status": report["suite_status"],
        })
        return report

    def _compact(self, report: dict) -> dict:
        """Token-lean version of a report for the model."""
        return {
            "iteration": report["iteration"],
            "gains": report["gains"],
            "suite_status": report["suite_status"],
            "overall_pass": report["overall_pass"],
            "n_pass": report["n_pass"],
            "n_fail": report["n_fail"],
            "tests": {tn: {"status": e["status"], "metrics": e["metrics"],
                           "thresholds": e["thresholds"], "notes": e["notes"]}
                      for tn, e in report["tests"].items()},
        }

    # ---- dispatch -------------------------------------------------------
    def dispatch(self, name: str, args: dict):
        """Return (result_obj, is_error)."""
        if name == "get_dyr_parameters":
            return {
                "gains": dyr_tools.read_gains(self.dyr),
                "model": "REECAU1", "bus": 91003, "machine_id": "S1",
                "editable": list(config.GAIN_NAMES),
                "allowed_range": f"({config.GAIN_MIN}, {config.GAIN_MAX}]",
                "locked": "all other .dyr parameters",
            }, False

        if name == "update_dyr_gains":
            gains = {"Kqp": args["kqp"], "Kqi": args["kqi"],
                     "Kvp": args["kvp"], "Kvi": args["kvi"]}
            err = dyr_tools.validate_gains(gains)
            if err:
                return {"error": err}, True
            res = dyr_tools.write_gains(self.dyr, gains, self.change_log,
                                        self.iterations_used + 1)
            return {"updated": True, "old": res["old"], "new": res["new"]}, False

        if name == "run_test_suite":
            if self.iterations_used >= config.ITERATION_BUDGET:
                return {"error": f"Iteration budget of {config.ITERATION_BUDGET} "
                                 "suite runs is exhausted. Call "
                                 "submit_final_report now."}, True
            self.iterations_used += 1
            report = self._run_and_eval(self.iterations_used)
            out = self._compact(report)
            out["iterations_used"] = self.iterations_used
            out["iterations_remaining"] = config.ITERATION_BUDGET - self.iterations_used
            return out, False

        if name == "read_log":
            test = args["test_name"]
            max_chars = args.get("max_chars", 4000)
            if self.latest_archive is None:
                return {"error": "no suite has been run yet"}, True
            results = self.latest_archive / "RESULTs"
            stem = test.split("_", 1)[1]
            logs = [p for p in results.rglob("*.log") if stem.lower() in p.name.lower()]
            if not logs:
                return {"error": f"no log found for {test}"}, True
            log = min(logs, key=lambda p: len(p.parts))
            text = log.read_text(encoding="utf-8", errors="replace")
            return {"test_name": test, "log_tail": text[-max_chars:]}, False

        if name == "submit_final_report":
            self.final_report = args
            self.done = True
            return {"received": True}, False

        return {"error": f"unknown tool {name}"}, True
