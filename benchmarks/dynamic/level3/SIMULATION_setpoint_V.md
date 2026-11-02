# ANDES dynamic simulation — Voltage setpoint change

Instructions to run the TDS simulation on the `test_grid.json` network,
with a step of the BESS plant voltage reference and export of P, Q and
V. This test is used to tune the **voltage / reactive power branch** of
the BESS control (REPCA1 + REECA1).

## Prerequisite: P test closed (consecutive tuning)

The V test is run **after** closing the P test
(`SIMULATION_setpoint_P.md`, section "Order of the tests"):

- the best P run must already be chosen and recorded in
  `<RUN>/steps/steps.md`;
- its PARAM_P (those modified with respect to the baseline) are
  **frozen** and must be applied in **every** V run, baseline included;
- during the V test only the PARAM_V are tuned; the PARAM_P are not
  touched any more.

Consequently **run 0 of the V test** is not the original case of
`test_grid.json`: it is the case with the PARAM_P of the best P run and
the PARAM_V still at their original values. It is the reference with
respect to which the improvements of the V test are computed.

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

| Time | Vref value (REPCA1) |
|---|---|
| t = 0 s | initialization value (≈ 0.9982 pu, Gen Bus voltage at the power flow) |
| t = 5 s | → 0.97 pu |
| t = 30 s | end of simulation |

## How the step is applied: `Vref0` of REPCA1

**REPCA1** regulates the Gen Bus voltage (`VCFlag = 0`, `Kc = 0`,
`RefFlag = 1`) and its reference (`Vref`) is tied to **`Vref0`**. An
`Alter` event on `Vref0` with `method = "="` (absolute value) directly
sets the new voltage reference.

## MCP call sequence (ANDES tools) — one run

`setup` can be called only once per loaded case: **each run restarts
from `load_case`**.

1. `load_case` — loads `test_grid.json`.
2. `set_param` — **before `setup`**, in this order:
   - **in every run, run 0 included**: set the PARAM_P of the best run of
     the P test (those modified with respect to the baseline);
   - **only from run 1 onwards**: set the PARAM_V of the run
     (`model = "REPCA1"` or `"REECA1"`, `idx = 1`, `param = <name>`,
     `value = <value>`).
   In `params.json` list separately the PARAM_P inherited from the P test
   and the PARAM_V modified in this run.
3. `add_alter` — **before `setup`**, adds the step:
   - `model = "REPCA1"`, `dev = 1`, `src = "Vref0"`
   - `t = 5.0`, `amount = 0.97`, `method = "="`
4. `setup` — initializes the system.
5. `run_power_flow` — must converge before the TDS.
6. `configure_tds` — `tf = 30`, `fixt = false` (adaptive step).
7. `run_tds` — runs the time-domain simulation (**counts as 1 run of the
   budget**).
8. `export_results`:
   - `variables = ["REGCA1.Pe", "REGCA1.Qe", "REGCA1.v"]`
   - `path = "<RUN>/V/run_<i>/output.csv"` (`<RUN>` = `RUN_FOLDER` defined in DESCRIPTION.txt, e.g. `RUN1`)
   - `plot = false`

## Exported variables and comparison with the target

| Exported variable | Column in `Target_V.csv` | Meaning |
|---|---|---|
| `REGCA1.Pe` | `REGCA1.Pe` | Active power delivered by the BESS [pu] |
| `REGCA1.Qe` | `REGCA1.Qe` | Reactive power delivered by the BESS [pu] |
| `REGCA1.v`  | `REGCA1.v`  | Gen Bus voltage [pu] |

Time is in the `t_s` column (0–30 s, step 1/30 s).

## Final verification runs (out of budget)

Once the V test is closed (best V run chosen and recorded in `steps.md`),
run the **two verification runs** with the complete final configuration,
i.e. PARAM_P of the best P run + PARAM_V of the best V run:

1. **P verification**: complete P test (`Pline0` steps, `tf = 130`),
   output in `<RUN>/P/final_check/` — see `SIMULATION_setpoint_P.md`,
   section "Final verification run".
2. **V verification**: same call sequence as in this file (`Vref0` step,
   `tf = 30`), with `set_param` of the best PARAM_P + best PARAM_V;
   `export_results` with `path = "<RUN>/V/final_check/output.csv"`; also
   save `params.json` and `metrics.json` (rmse, maxerr, r2 of P, Q, V).

Since the V test already uses the best PARAM_P, the V verification should
reproduce the best V run (it serves as a consistency check); the P
verification instead shows the effect of the tuned PARAM_V on the P test.

The two verification runs:
- **do not count** towards the tuning budget (15 runs) and are in
  addition to them;
- **do not take part** in the choice of the best runs;
- are the metrics of the **final configuration** and must be reported in
  `steps.md` (global run_tds numbers #16 and #17, if the whole budget was
  used), in the `summary.csv` files (`final_check` row), in
  `best_runs.json`, in the final report and in the dashboard
  ("final_check" curve).

Technical note: the MCP server rejects `run_tds` calls beyond the
`ANDES_MCP_MAX_TDS` limit (in `.mcp.json`), set to **17** (15 for tuning +
2 for verification). The tuning budget remains 15: do not use calls #16
and #17 for tuning.
