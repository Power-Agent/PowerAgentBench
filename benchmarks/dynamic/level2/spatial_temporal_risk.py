"""Additional risk metrics for transient-stability voltage trajectories.
These metrics complement the existing Integral of Violation (IoV) score.
They quantify spatial concentration and temporal persistence rather than
averaging every bus and time sample into one scalar.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

def spatial_temporal_risk(voltages: pd.DataFrame, times: pd.Series, *,
    undervoltage_threshold: float = 0.95, undervoltage_floor: float = 0.90,
    overvoltage_threshold: float = 1.05, overvoltage_ceiling: float = 1.10,
    tail_fraction: float = 0.10) -> dict[str, Any]:
    """Compute exposure, concentration, tail, and persistence diagnostics.

    Rows are time samples, columns are buses, and time units are seconds.
    Results are descriptive diagnostics, not protection criteria.
    """
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if not 0.0 <= undervoltage_floor < undervoltage_threshold:
        raise ValueError("undervoltage_floor must be below undervoltage_threshold")
    if not 0.0 < overvoltage_threshold < overvoltage_ceiling:
        raise ValueError("overvoltage_threshold must be below overvoltage_ceiling")
    if len(voltages) != len(times):
        raise ValueError("voltages and times must have the same number of samples")
    if voltages.empty or voltages.shape[1] == 0:
        return _empty_result(tail_fraction)
    values = voltages.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("voltages must contain only finite numeric values")
    t = np.asarray(times, dtype=float)
    if not np.isfinite(t).all():
        raise ValueError("times must contain only finite numeric values")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    window = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    under = np.clip((undervoltage_threshold - values) / (undervoltage_threshold - undervoltage_floor), 0.0, 1.0)
    over = np.clip((values - overvoltage_threshold) / (overvoltage_ceiling - overvoltage_threshold), 0.0, 1.0)
    if len(t) < 2 or window <= 0:
        under_iov = np.zeros(values.shape[1]); over_iov = np.zeros(values.shape[1])
        under_sample = np.zeros(len(t)); over_sample = np.zeros(len(t))
        under_duration = np.zeros(values.shape[1]); over_duration = np.zeros(values.shape[1])
    else:
        # Trapezoidal integration accounts for nonuniform solver sampling.
        under_iov = np.trapezoid(under, t, axis=0) / window
        over_iov = np.trapezoid(over, t, axis=0) / window
        under_sample = under.mean(axis=1); over_sample = over.mean(axis=1)
        intervals = np.diff(t)
        under_duration = ((under[:-1] > 0) & (under[1:] > 0)).T @ intervals
        over_duration = ((over[:-1] > 0) & (over[1:] > 0)).T @ intervals
    return {
        "sample_count": int(len(t)), "bus_count": int(values.shape[1]),
        "window_seconds": _r(window),
        "mean_iov_undervoltage": _r(float(under_iov.mean())),
        "mean_iov_overvoltage": _r(float(over_iov.mean())),
        "worst_bus_iov_undervoltage": _r(float(under_iov.max())),
        "worst_bus_iov_overvoltage": _r(float(over_iov.max())),
        "affected_bus_fraction_undervoltage": _r(float((under > 0).any(axis=0).mean())),
        "affected_bus_fraction_overvoltage": _r(float((over > 0).any(axis=0).mean())),
        "tail_mean_undervoltage": _r(_tail_mean(under_sample, tail_fraction)),
        "tail_mean_overvoltage": _r(_tail_mean(over_sample, tail_fraction)),
        "longest_violation_seconds_undervoltage": _r(float(under_duration.max(initial=0.0))),
        "longest_violation_seconds_overvoltage": _r(float(over_duration.max(initial=0.0))),
        "tail_fraction": float(tail_fraction),
    }

def _tail_mean(values: np.ndarray, fraction: float) -> float:
    if values.size == 0: return 0.0
    count = max(1, int(np.ceil(values.size * fraction)))
    return float(np.sort(values)[-count:].mean())

def _r(value: float) -> float:
    return round(value, 8)

def _empty_result(fraction: float) -> dict[str, Any]:
    keys = ("mean_iov_undervoltage", "mean_iov_overvoltage",
        "worst_bus_iov_undervoltage", "worst_bus_iov_overvoltage",
        "affected_bus_fraction_undervoltage", "affected_bus_fraction_overvoltage",
        "tail_mean_undervoltage", "tail_mean_overvoltage",
        "longest_violation_seconds_undervoltage", "longest_violation_seconds_overvoltage")
    result = {key: 0.0 for key in keys}
    result.update(sample_count=0, bus_count=0, window_seconds=0.0, tail_fraction=float(fraction))
    return result
