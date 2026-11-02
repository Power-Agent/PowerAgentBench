# ANDES dynamic simulation — Active power setpoint change

Instructions to run the TDS simulation on the `test_grid.json` network,
with step changes of the BESS active power setpoint and export of P and
Q. This test is used to tune the **active power branch** of the BESS
control (REPCA1 + REECA1).

## Order of the tests (consecutive tuning)

Tuning is done **in sequence**, not in parallel:

1. **First the P test** (this file): all tuning runs of the active power
   branch. During the P test the V/Q branch parameters stay at the
   original values of `test_grid.json`.
2. **Closing the P test**: choose the best P run with the criterion in
   DESCRIPTION.txt and record it in `<RUN>/steps/steps.md`. From this
   moment the PARAM_P of the best run are **frozen**: you never go back to
   the P test for tuning.
3. **Then the V test** (`SIMULATION_setpoint_V.md`): every V run, baseline
   included, uses the frozen PARAM_P of the best P run.
4. **At the end, two out-of-budget verification runs** with the complete
   final configuration (best PARAM_P + best PARAM_V). See the section
   "Final verification run" below and in `SIMULATION_setpoint_V.md`.

## Reference network

Case: `test_grid.json`

| Model | No. of instances | Role |
|---|---|---|
| Bus | 2 | Gen Bus (BESS), Infinite Bus |
| PV | 1 | Static operating point of the BESS, `p0 = 0.5 pu`, `v0 ≈ 0.9982 pu` (zero initial Q) |
| Slack | 1 | Slack bus (Infinite Bus) |
| Line | 1 | Connection line (x = 0.12 pu) |
| BusFreq | 1 | Frequency measurement at the Gen Bus (for REPCA1) |
| REGCA1 | 1 | Converter–grid interface (Ip/Iq current injection) |
| REECA1 | 1 | Electrical control (Q/V loops, current limits and P ramp limits) |
| REPCA1 | 1 | Plant control (V and P regulation at the point of connection) |

## Setpoint profile to reproduce

| Time | P setpoint value |
|---|---|
| t = 0 s | 0.5 pu |
| t = 5 s | → 0.7 pu |
| t = 35 s | → 0.5 pu |
| t = 65 s | → 0.2 pu |
| t = 95 s | → 0.5 pu |
| t = 130 s | end of simulation |

## How the step is applied: `Pline0` of REPCA1

The plant power reference of **REPCA1** (`Plant_pref`) is tied to the
value **`Pline0`** (line power measured at the power flow, 0.5 pu). An
`Alter` event on `Pline0` with `method = "="` directly sets the **new
absolute setpoint** (not the difference with respect to the base value):

| t [s] | Pline0 (absolute) |
|---|---|
| 5 | 0.7 |
| 35 | 0.5 |
| 65 | 0.2 |
| 95 | 0.5 |

## MCP call sequence (ANDES tools) — one run

`setup` can be called only once per loaded case: **each run restarts
from `load_case`**.

1. `load_case` — loads `test_grid.json`.
2. `set_param` — (only from run 1 onwards) sets the PARAM_P of the run:
   `model = "REPCA1"` or `"REECA1"`, `idx = 1`, `param = <name>`,
   `value = <value>`. Must be done **before `setup`**. During P tuning the
   PARAM_V are not modified.
3. `add_alter` — **before `setup`**, four calls, one per step:
   - `model = "REPCA1"`, `dev = 1`, `src = "Pline0"`, `method = "="`
   - `(t = 5.0, amount = 0.7)`, `(t = 35.0, amount = 0.5)`,
     `(t = 65.0, amount = 0.2)`, `(t = 95.0, amount = 0.5)`
4. `setup` — initializes the system.
5. `run_power_flow` — must converge before the TDS.
6. `configure_tds` — `tf = 130`, `fixt = false` (adaptive step).
7. `run_tds` — runs the time-domain simulation (**counts as 1 run of the
   budget**).
8. `export_results`:
   - `variables = ["REGCA1.Pe", "REGCA1.Qe"]`
   - `path = "<RUN>/P/run_<i>/output.csv"` (`<RUN>` = `RUN_FOLDER` defined in DESCRIPTION.txt, e.g. `RUN1`)
   - `plot = false`

## Exported variables and comparison with the target

| Exported variable | Column in `Target_P.csv` | Meaning |
|---|---|---|
| `REGCA1.Pe` | `REGCA1.Pe` | Active power delivered by the BESS [pu] |
| `REGCA1.Qe` | `REGCA1.Qe` | Reactive power delivered by the BESS [pu] |

Time is in the `t_s` column (0–130 s, step 0.01 s).

## Final verification run (out of budget)

After closing the V test, run **one** P verification run with the
complete final configuration:

- same call sequence described above, but at step 2 use `set_param` to set
  **both** the PARAM_P of the best P run **and** the PARAM_V of the best V
  run;
- `export_results` with `path = "<RUN>/P/final_check/output.csv"`;
- also save in `<RUN>/P/final_check/` `params.json` (read with
  `get_device_params` after `set_param`) and `metrics.json` (same metrics
  as the other P runs: rmse_P, rmse_Q, maxerr_P, maxerr_Q, r2_P, r2_Q).

This run:
- **does not count** towards the tuning budget (the 15 runs) and is in
  addition to them;
- **does not take part** in the choice of the best run;
- must be reported in `steps.md`, in `summary.csv` (`final_check` row), in
  `best_runs.json`, in the final report and in the dashboard, comparing it
  with the best P run, so as to show the effect of the tuned PARAM_V on
  the P test (in particular on Q).

Technical note: the MCP server rejects `run_tds` calls beyond the
`ANDES_MCP_MAX_TDS` limit (in `.mcp.json`), set to **17** (15 for tuning +
2 for verification). The tuning budget remains 15: calls #16 and #17 are
reserved exclusively for the two verification runs.
