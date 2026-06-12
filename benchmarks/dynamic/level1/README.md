# Level 1: WECC Solar Dynamic Model Quality Review (DMQ)

This folder defines a reproducible level-1 **dynamic** benchmark for the
PowerAgentBench-Dyn *Dynamic Model Quality Review* task. An agent must repair a
modified WECC solar photovoltaic dynamic model so that it passes the full DMView
model-quality test suite, while modifying **only** the four allowed REECAU1
controller gains.

It is the dynamic-domain analogue of `benchmarks/steady/level_1`: a small,
fully published, constrained-action task suitable for rapid iteration.

## Task

The submitted `Solar.dyr` has an injected error — an overly aggressive REECAU1
gain set `(Kqp, Kqi, Kvp, Kvi) = (10, 50, 10, 50)` — that drives an unstable
reactive/voltage (Q/V) control loop. At the baseline the model fails all eight
DMView tests (see `baseline_summary.json`).

The agent connects the model to the DMView test system, runs the suite, reads
the pass/fail report, and proposes constrained updates to the four gains,
re-running until all tests pass or the **five-iteration budget** is exhausted.

### Action space (strictly enforced)
Only the four REECAU1 gains may change, each in the open-closed range `(0, 100]`:
`Kqp`, `Kqi`, `Kvp`, `Kvi` (data tokens 24–27 of the REECAU1 record at bus
91003, machine `S1`). Every other `.dyr` parameter and the other models
(REGCAU1, REPCAU1) are locked. The harness enforces this structurally — the only
mutating tool rewrites exactly those four tokens.

### Test suite (8 tests)
Flat start; voltage step down/up; frequency step down/up; HVRT; LVRT; and a
graded weak-grid SCR test (`5 → 3 → 1.5 → 1.2`). A model passes only if every
test passes. For SCR, graded levels 5/3/1.5 must be stable; 1.2 is informational
only (not stabilizable by the reference solution).

## Files

- `actionspace.json` — published action contract: the four gains, bounds, model
  location, test suite, and iteration budget.
- `actioncost.json` — per-simulation cost and budget used for ranking.
- `baseline_summary.json` — reference no-fix evaluation (corrupted model, 0/8)
  plus the known-good reference solution `(1, 5, 1, 5)`.
- `solution_template.json` — minimal valid solution file to copy and edit.
- `harness/` — runnable harness (DMView automation + evaluator + agentic loop).

Case data lives in `cases/solar_wecc/psse/` (`Solar.sav`, `Solar.dyr`).

## Reference results

Over 10 independent runs per model (Anthropic Claude), iteration budget 5:

| Agent | Success rate | Median iters | Median passing gains |
|-------|--------------|--------------|----------------------|
| Opus 4.8 | 10/10 | 1 | (1, 5, 1, 5) |
| Sonnet 4.6 | 10/10 | 1 | (0.5, 2, 0.5, 3.5) |
| Haiku 4.5 | 9/10 | 2 | (1.2, 2, 1.5, 4) |

## Prerequisites

Unlike the steady-state benchmarks (which run on open-source PyPSA), this
dynamic benchmark drives commercial / licensed tooling that you must install
first:

1. **PSS/E 36.2** (Siemens) — the dynamic simulation engine. A valid license is
   required. The Python bindings must be available to the interpreter (this
   benchmark was validated with PSS/E 36.2 + Python 3.11).
2. **DMView 3.4** (ERCOT dynamic-model review tool) —
   <https://sites.google.com/view/dmview/home>. Install per its manual; it must
   run on a Python version it supports (3.8/3.9/3.11/3.13) that also has the
   PSS/E bindings — in practice **Python 3.11**. Set `PSSE_VER = 36` in the
   project INI (the harness does this automatically).
3. **Python 3.11** with `anthropic`, `numpy<2` (PSS/E needs numpy 1.x),
   `openpyxl`, and `matplotlib`.
4. An `ANTHROPIC_API_KEY` in `PowerAgentBench/.env`.

Then point the harness at your local install by editing `harness/config.py`:

- `DMVIEW_ROOT` — the DMView 3.4 install directory.
- `PY311` — the Python 3.11 executable that has the PSS/E bindings.

(The case and output paths are derived automatically from this benchmark's
location.)

## Running the harness

```bash
cd benchmarks/dynamic/level1/harness
# calibrate the evaluator (optional - thresholds are frozen in criteria.json)
python calibrate.py bad
python calibrate.py good
python run_calibration_eval.py

# run the benchmark matrix (resumable)
python driver.py --all --runs 10 --budget 5
# smoke test:
python driver.py --model claude-haiku-4-5 --runs 1 --budget 1

# aggregate -> results/aggregate.{csv,json}, dmq_table.tex, summary.md
python aggregate.py
```

Outputs are written under this folder: `runs/<model>/run<NN>/` (per-run
transcripts, checkpoints, `dyr_changes.log`, final reports) and `results/`.

### Note on paths
`harness/config.py` derives `PRISTINE_DIR` (the case) and the output folders
from this benchmark's location, but `DMVIEW_ROOT` and `PY311` are absolute paths
to the local DMView/PSS-E install and must be adjusted for a different machine.
