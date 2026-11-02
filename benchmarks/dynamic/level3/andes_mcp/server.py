"""
MCP server for ANDES (https://github.com/CURENT/andes).

Exposes as MCP tools the typical operations of an RMS study with ANDES:
  - loading a grid case from file
  - inspecting/modifying model (device) parameters
  - adding disturbance events (Alter, Fault, Toggle)
  - running power flow and time-domain simulation (TDS)
  - exporting simulation results (CSV / PNG plot)

Single session: only one ANDES System lives in the server process at a
time (see andes_mcp.session).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

import andes
import pandas as pd
from mcp.server.mcpserver import MCPServer

from andes_mcp.session import (
    STATE,
    AndesSessionError,
    get_model,
    new_system_from_file,
    require_pflow,
    require_setup,
    require_system,
    run_capturing_output,
)

mcp = MCPServer("andes")

# Simulation budget and audit (configurable from .mcp.json):
#   ANDES_MCP_MAX_TDS  maximum number of run_tds calls per session (0 = unlimited)
#   ANDES_MCP_TDS_LOG  JSON lines file where every run_tds call is logged
TDS_MAX_RUNS = int(os.environ.get("ANDES_MCP_MAX_TDS", "0") or 0)
TDS_LOG_PATH = os.environ.get("ANDES_MCP_TDS_LOG") or None


def _err(msg: str) -> dict:
    return {"ok": False, "error": msg}


def _ok(**payload: Any) -> dict:
    return {"ok": True, **payload}


# ---------------------------------------------------------------------
# Case loading and session state
# ---------------------------------------------------------------------

@mcp.tool()
def load_case(path: str) -> dict:
    """Load an ANDES grid case from file (xlsx, json, PSS/E raw, MATPOWER,
    etc.) and make it the active system of the session. Replaces any
    previously loaded system. Does not run setup() or the power flow yet:
    use 'setup' after optionally adding disturbance events, then
    'run_power_flow'.
    """
    try:
        ss = new_system_from_file(path)
    except AndesSessionError as e:
        return _err(str(e))

    STATE.ss = ss
    STATE.case_path = path
    STATE.setup_done = False
    STATE.pflow_done = False
    STATE.pflow_converged = False
    STATE.tds_done = False
    STATE.tds_converged = False
    STATE.pending_events = []

    model_counts = {
        name: model.n for name, model in ss.models.items() if model.n > 0
    }
    return _ok(
        case_path=path,
        models=model_counts,
        message="Case loaded. Call 'setup' to initialize the system.",
    )


@mcp.tool()
def get_system_summary() -> dict:
    """Return the current state of the session: loaded case, whether the
    system has been initialized (setup), whether the power flow has
    converged, whether the TDS simulation has been run."""
    return _ok(
        case_path=STATE.case_path,
        setup_done=STATE.setup_done,
        pflow_done=STATE.pflow_done,
        pflow_converged=STATE.pflow_converged,
        tds_done=STATE.tds_done,
        tds_converged=STATE.tds_converged,
        pending_events=STATE.pending_events,
    )


# ---------------------------------------------------------------------
# Model and device inspection
# ---------------------------------------------------------------------

@mcp.tool()
def list_models() -> dict:
    """List the models (device types) present in the loaded system with
    their number of instances. Useful to know which models are available
    before querying their parameters."""
    try:
        ss = require_system()
    except AndesSessionError as e:
        return _err(str(e))

    counts = {name: model.n for name, model in ss.models.items() if model.n > 0}
    return _ok(models=counts)


@mcp.tool()
def list_devices(model: str) -> dict:
    """List all devices of a model with their current parameters (one per
    row). Example: model='REGCA1' lists all the converter interfaces with
    their parameters (Tg, Rrpwr, Brkpt, ...)."""
    try:
        ss = require_system()
        m = get_model(ss, model)
    except AndesSessionError as e:
        return _err(str(e))

    if m.n == 0:
        return _ok(model=model, devices=[])

    df = m.as_df()
    return _ok(model=model, devices=df.to_dict(orient="records"))


@mcp.tool()
def get_device_params(model: str, idx: Any) -> dict:
    """Return all the parameters of a single device identified by
    model + idx (e.g. model='REECA1', idx=1)."""
    try:
        ss = require_system()
        m = get_model(ss, model)
    except AndesSessionError as e:
        return _err(str(e))

    df = m.as_df()
    idx_col = df["idx"] if "idx" in df.columns else None
    if idx_col is None:
        return _err(f"Model '{model}' does not expose an 'idx' column.")

    matches = df[idx_col.astype(str) == str(idx)]
    if matches.empty:
        return _err(f"No device with idx={idx!r} in model '{model}'.")
    return _ok(model=model, idx=idx, params=matches.iloc[0].to_dict())


@mcp.tool()
def list_variables(model: str) -> dict:
    """List the state and algebraic variables (states, algebs) of a
    dynamic model: these are the names usable with 'export_results' to
    extract the time series of a TDS simulation (e.g. model='REGCA1'
    -> 'Pe', 'Qe', 'v', ...)."""
    try:
        ss = require_system()
        m = get_model(ss, model)
    except AndesSessionError as e:
        return _err(str(e))

    return _ok(
        model=model,
        states=list(m.states.keys()),
        algebs=list(m.algebs.keys()),
        states_ext=list(m.states_ext.keys()),
        algebs_ext=list(m.algebs_ext.keys()),
    )


# ---------------------------------------------------------------------
# Parameter changes and device/event addition
# ---------------------------------------------------------------------

@mcp.tool()
def set_param(model: str, idx: Any, param: str, value: float) -> dict:
    """Modify the value of a parameter of an existing device
    (e.g. model='REECA1', idx=1, param='Kvp', value=0.6).

    Can be called either before 'setup' (modifies the static case data)
    or after power flow/TDS (updates the current value, equivalent to an
    instantaneous 'Alter' event); for a step applied at a precise instant
    during the TDS use 'add_alter' before setup instead."""
    try:
        ss = require_system()
        m = get_model(ss, model)
    except AndesSessionError as e:
        return _err(str(e))

    if param not in m.params:
        return _err(
            f"'{param}' is not a parameter of model '{model}'. "
            f"Available parameters: {sorted(m.params.keys())}"
        )

    try:
        m.set(src=param, idx=idx, attr="v", value=value)
    except Exception as e:  # ANDES can raise heterogeneous exceptions
        return _err(f"Unable to set {model}.{param}[{idx}] = {value}: {e}")

    return _ok(model=model, idx=idx, param=param, value=value)


@mcp.tool()
def add_device(model: str, params: dict) -> dict:
    """Add a new device to the system (e.g. a new Bus, Line, PV, dynamic
    generator, etc.) BEFORE calling 'setup'. 'params' is a dictionary of
    the parameters required by the model (must include 'idx' if an
    explicit identifier is wanted). Not usable after 'setup': for
    disturbance events use 'add_alter' / 'add_fault' / 'add_toggle',
    which also add devices but with dedicated checks."""
    try:
        ss = require_system()
    except AndesSessionError as e:
        return _err(str(e))

    if STATE.setup_done:
        return _err(
            "The system has already been initialized (setup). New devices "
            "cannot be added: reload a case with 'load_case' to start over."
        )

    try:
        idx = ss.add(model, params)
    except Exception as e:
        return _err(f"Unable to add device '{model}': {e}")

    return _ok(model=model, idx=idx, params=params)


@mcp.tool()
def add_alter(
    model: str,
    dev: Any,
    src: str,
    t: float,
    amount: float,
    attr: str = "v",
    method: str = "=",
) -> dict:
    """Add an 'Alter' event: at time t it modifies the parameter/variable
    'src' of device 'dev' of model 'model' (e.g. a step of the plant
    voltage reference: model='REPCA1', dev=1, src='Vref0', t=5.0,
    amount=0.97). 'method' can be '=', '+', '-', '*', '/'.
    Must be called BEFORE 'setup'."""
    return add_device(
        "Alter",
        dict(model=model, dev=dev, src=src, attr=attr, t=t, method=method, amount=amount),
    )


@mcp.tool()
def add_fault(bus: Any, tf: float, tc: float, xf: float = 0.0, rf: float = 0.0) -> dict:
    """Add a fault event (short circuit) at the given bus, applied at time
    tf and cleared at time tc. xf/rf are the fault reactance/resistance
    (default: bolted fault to ground).
    Must be called BEFORE 'setup'."""
    return add_device("Fault", dict(bus=bus, tf=tf, tc=tc, xf=xf, rf=rf))


@mcp.tool()
def add_toggle(model: str, dev: Any, t: float) -> dict:
    """Add a 'Toggle' event: disables/re-enables device 'dev' of model
    'model' at time t (e.g. opening a line: model='Line', dev=1, t=2.0).
    Must be called BEFORE 'setup'."""
    return add_device("Toggle", dict(model=model, dev=dev, t=t))


# ---------------------------------------------------------------------
# Setup, power flow, TDS
# ---------------------------------------------------------------------

@mcp.tool()
def setup() -> dict:
    """Initialize the system (ss.setup()): finalizes indices and matrices
    and sizes the problem. Must be called only once, after loading the
    case and optionally adding devices/events, and before running the
    power flow."""
    try:
        ss = require_system()
    except AndesSessionError as e:
        return _err(str(e))

    if STATE.setup_done:
        return _err("The system has already been initialized.")

    try:
        result, log = run_capturing_output(ss.setup)
    except Exception as e:
        return _err(f"Error during setup: {e}")

    if result is False:
        return _err(f"Setup failed.\n{log}")

    STATE.setup_done = True
    return _ok(message="System initialized.", log=log)


@mcp.tool()
def run_power_flow() -> dict:
    """Run the power flow (ss.PFlow.run()). Requires 'setup' to have been
    called already. Returns the convergence status and a summary of the
    bus voltages."""
    try:
        ss = require_setup()
    except AndesSessionError as e:
        return _err(str(e))

    try:
        converged, log = run_capturing_output(ss.PFlow.run)
    except Exception as e:
        return _err(f"Error during the power flow: {e}")

    STATE.pflow_done = True
    STATE.pflow_converged = bool(converged)

    if not converged:
        return _err(f"Power flow did not converge.\n{log}")

    bus_df = pd.DataFrame(
        {
            "idx": ss.Bus.idx.v,
            "name": ss.Bus.name.v,
            "v_pu": ss.Bus.v.v,
            "angle_rad": ss.Bus.a.v,
        }
    )
    return _ok(converged=True, buses=bus_df.to_dict(orient="records"), log=log)


@mcp.tool()
def configure_tds(tf: float = 20.0, fixt: bool = False, tstep: Optional[float] = None) -> dict:
    """Configure the time-domain simulation (TDS) parameters before running
    it with 'run_tds': tf = final simulation time [s], fixt = fixed (True)
    or adaptive (False) time step, tstep = time step [s] (used if
    fixt=True or as the maximum step)."""
    try:
        ss = require_setup()
    except AndesSessionError as e:
        return _err(str(e))

    ss.TDS.config.tf = tf
    ss.TDS.config.fixt = fixt
    if tstep is not None:
        ss.TDS.config.tstep = tstep

    return _ok(tf=tf, fixt=fixt, tstep=tstep)


@mcp.tool()
def run_tds() -> dict:
    """Run the time-domain simulation (ss.TDS.run()). Requires the power
    flow to have converged already. Use 'configure_tds' before this tool
    to set the simulation duration and step (default: 20 s, adaptive
    step)."""
    try:
        ss = require_pflow()
    except AndesSessionError as e:
        return _err(str(e))

    if TDS_MAX_RUNS and STATE.tds_count >= TDS_MAX_RUNS:
        return _err(
            f"Simulation budget exhausted: {STATE.tds_count}/{TDS_MAX_RUNS} "
            f"run_tds calls already executed in this session."
        )

    STATE.tds_count += 1
    try:
        converged, log = run_capturing_output(ss.TDS.run)
    except Exception as e:
        _log_tds_run(ss, converged=False, error=str(e))
        return _err(f"Error during the TDS simulation: {e}")

    STATE.tds_done = True
    STATE.tds_converged = bool(converged)
    _log_tds_run(ss, converged=bool(converged))

    n_points = len(ss.dae.ts.t) if hasattr(ss.dae.ts, "t") else 0
    budget = dict(tds_run=STATE.tds_count, tds_max=TDS_MAX_RUNS or None)
    if not converged:
        return _err(f"TDS simulation did not complete successfully ({budget}).\n{log}")

    return _ok(converged=True, n_points=n_points, t_final=ss.TDS.config.tf, log=log, **budget)


def _log_tds_run(ss: Any, converged: bool, error: Optional[str] = None) -> None:
    """Append to the audit log (ANDES_MCP_TDS_LOG, JSON lines) one line
    per run_tds call: sequential number, Alter events and controller
    parameters used. Used to verify the actual budget externally."""
    if not TDS_LOG_PATH:
        return

    def _params(name: str) -> dict:
        m = ss.models.get(name)
        if m is None or m.n == 0:
            return {}
        out = {}
        for k, p in m.params.items():
            try:
                out[k] = float(p.v[0])
            except (TypeError, ValueError, IndexError):
                continue
        return out

    alters = []
    if ss.Alter.n:
        for i in range(ss.Alter.n):
            alters.append(dict(
                model=ss.Alter.model.v[i], src=ss.Alter.src.v[i],
                t=float(ss.Alter.t.v[i]), amount=float(ss.Alter.amount.v[i]),
            ))

    entry = dict(
        time=datetime.now().isoformat(timespec="seconds"),
        tds_run=STATE.tds_count,
        case=STATE.case_path,
        tf=float(ss.TDS.config.tf),
        converged=converged,
        error=error,
        alters=alters,
        REGCA1=_params("REGCA1"),
        REECA1=_params("REECA1"),
        REPCA1=_params("REPCA1"),
    )
    try:
        with open(TDS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------
# Results export
# ---------------------------------------------------------------------

def _resolve_ts_variable(ss: Any, spec: str):
    """Convert a 'Model.var' string into the corresponding ANDES variable
    object, usable with ss.dae.ts.get_data."""
    if "." not in spec:
        raise AndesSessionError(
            f"Invalid variable '{spec}': use the format 'Model.variable' "
            f"(e.g. 'REGCA1.Pe')."
        )
    model_name, var_name = spec.split(".", 1)
    m = get_model(ss, model_name)
    # External variables (e.g. REGCA1.v = voltage of the bus the generator
    # is connected to) can be exported too.
    if not any(var_name in d for d in (m.states, m.algebs, m.states_ext, m.algebs_ext)):
        raise AndesSessionError(
            f"'{var_name}' is not a state/algebraic variable of model "
            f"'{model_name}'. Use 'list_variables' for the valid list."
        )
    return getattr(m, var_name)


@mcp.tool()
def export_results(variables: list[str], path: str, plot: bool = False) -> dict:
    """Export the time series of the TDS simulation to a CSV file.
    'variables' is a list of 'Model.variable' strings (e.g.
    ['REGCA1.Pe', 'REGCA1.Qe', 'REPCA1.Vref']); use 'list_variables' to
    discover the names available for each model. 'path' is the path of
    the destination CSV file. If plot=True a PNG plot is also generated
    (same name as the CSV, .png extension) with one subplot per variable."""
    try:
        ss = require_system()
        if not STATE.tds_done or not STATE.tds_converged:
            return _err(
                "No successfully completed TDS simulation to export. "
                "Use 'run_tds' before 'export_results'."
            )
        var_objs = [_resolve_ts_variable(ss, v) for v in variables]
    except AndesSessionError as e:
        return _err(str(e))

    try:
        data = ss.dae.ts.get_data(var_objs)
    except Exception as e:
        return _err(f"Unable to extract the requested variables: {e}")

    t = ss.dae.ts.t
    columns = []
    for spec, var in zip(variables, var_objs):
        n = getattr(var, "n", 1) or 1
        if n <= 1:
            columns.append(spec)
        else:
            columns.extend(f"{spec}[{i}]" for i in range(n))

    df = pd.DataFrame(data, columns=columns)
    df.insert(0, "t_s", t)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False)

    result = _ok(csv_path=path, n_rows=len(df), columns=columns)

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        png_path = os.path.splitext(path)[0] + ".png"
        fig, axes = plt.subplots(len(columns), 1, sharex=True, figsize=(9, 2.2 * len(columns)))
        if len(columns) == 1:
            axes = [axes]
        for ax, col in zip(axes, columns):
            ax.plot(df["t_s"], df[col])
            ax.set_ylabel(col)
            ax.grid(True)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        result["plot_path"] = png_path

    return result


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
