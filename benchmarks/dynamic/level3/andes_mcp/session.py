"""
ANDES session state shared by the MCP tools.

Single session: only one andes.System object lives in the server process
at a time. All operations (case loading, parameter changes, power flow,
TDS, export) act on this global state.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any, Optional

import andes


class AndesSessionError(RuntimeError):
    """Application-level session error (invalid state, wrong parameter, etc.)."""


@dataclass
class SessionState:
    ss: Optional[Any] = None
    case_path: Optional[str] = None
    setup_done: bool = False
    pflow_done: bool = False
    pflow_converged: bool = False
    tds_done: bool = False
    tds_converged: bool = False
    # Number of run_tds calls executed in the server session (not reset
    # by load_case): used for the simulation budget.
    tds_count: int = 0
    # Models added manually (Alter/Fault/Toggle/others) before setup,
    # kept only for user-side reference/summary.
    pending_events: list = field(default_factory=list)


STATE = SessionState()


def run_capturing_output(fn, *args, **kwargs):
    """Run fn capturing stdout/stderr (ANDES logs a lot) and return
    (result, text_log)."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        result = fn(*args, **kwargs)
    log = (buf_out.getvalue() + buf_err.getvalue()).strip()
    return result, log


def require_system() -> Any:
    if STATE.ss is None:
        raise AndesSessionError(
            "No case loaded. Use the 'load_case' tool before proceeding."
        )
    return STATE.ss


def require_setup() -> Any:
    ss = require_system()
    if not STATE.setup_done:
        raise AndesSessionError(
            "The system has not been initialized with 'setup'. "
            "Add any events (Alter/Fault/Toggle) and then call 'setup'."
        )
    return ss


def require_pflow() -> Any:
    ss = require_setup()
    if not STATE.pflow_done or not STATE.pflow_converged:
        raise AndesSessionError(
            "The power flow has not been run successfully. Use 'run_power_flow'."
        )
    return ss


def get_model(ss: Any, model_name: str) -> Any:
    if model_name not in ss.models:
        raise AndesSessionError(
            f"Model '{model_name}' not found. Use 'list_models' for the valid list."
        )
    return ss.models[model_name]


def reset_session() -> None:
    """Reset STATE to its initial state by mutating it in place, so that
    references already imported elsewhere (e.g. server.py) remain valid."""
    fresh = SessionState()
    for field_name in fresh.__dataclass_fields__:
        setattr(STATE, field_name, getattr(fresh, field_name))


def new_system_from_file(path: str) -> Any:
    if not os.path.isfile(path):
        raise AndesSessionError(f"Case file not found: {path}")
    ss, log = run_capturing_output(andes.load, path, setup=False)
    if ss is None:
        raise AndesSessionError(f"Unable to load case '{path}'.\n{log}")
    return ss
