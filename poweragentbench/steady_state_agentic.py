"""Agentic steady-state contingency benchmark utilities.

This module implements the PowerAgentBench-SS Level 2 N-k task used in the
paper.  It is intentionally small but follows the public/hidden split:
scripted or LLM agents see only public tools, while the runner computes a
hidden exhaustive oracle for scoring.

The default case loader uses the existing IEEE 39-bus case already distributed
with the repository.  Synthetic cases are kept as a development option.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Sequence, Tuple

import numpy as np

Contingency = Tuple[int, ...]


@dataclass
class GridCase:
    """Linear DC representation of a steady-state transmission case."""

    n_bus: int
    from_bus: np.ndarray
    to_bus: np.ndarray
    x: np.ndarray
    rating: np.ndarray
    load: np.ndarray
    gen_buses: np.ndarray
    gen_p: np.ndarray
    gen_min: np.ndarray
    gen_max: np.ndarray
    gen_cost: np.ndarray
    slack: int = 0
    name: str = "case"
    branch_names: List[str] = field(default_factory=list)
    bus_names: List[str] = field(default_factory=list)

    @property
    def n_line(self) -> int:
        return int(len(self.from_bus))

    @property
    def n_gen(self) -> int:
        return int(len(self.gen_buses))

    @property
    def base_injection(self) -> np.ndarray:
        p = -self.load.astype(float).copy()
        for bus, gen in zip(self.gen_buses, self.gen_p):
            p[int(bus)] += float(gen)
        p[self.slack] -= float(p.sum())
        return p

    def with_generation(self, gen_p: np.ndarray) -> "GridCase":
        return GridCase(
            n_bus=self.n_bus,
            from_bus=self.from_bus.copy(),
            to_bus=self.to_bus.copy(),
            x=self.x.copy(),
            rating=self.rating.copy(),
            load=self.load.copy(),
            gen_buses=self.gen_buses.copy(),
            gen_p=np.asarray(gen_p, dtype=float).copy(),
            gen_min=self.gen_min.copy(),
            gen_max=self.gen_max.copy(),
            gen_cost=self.gen_cost.copy(),
            slack=self.slack,
            name=self.name,
            branch_names=list(self.branch_names),
            bus_names=list(self.bus_names),
        )


@dataclass
class PFResult:
    feasible: bool
    flows: np.ndarray
    loading: np.ndarray
    severity: float
    island_penalty: float
    outage: Contingency


@dataclass
class AgentOutput:
    name: str
    validated: Dict[Contingency, float]
    reported: List[Contingency]
    mitigated_case: GridCase | None = None
    action_cost: float = 0.0
    tool_log: List[Dict[str, Any]] = field(default_factory=list)
    invalid_tool_calls: float = 0.0
    raw_responses: List[str] = field(default_factory=list)
    schema_repairs: float = 0.0
    type_coercions: float = 0.0
    duplicate_validation_requests: float = 0.0
    submitted_explicitly: float = 1.0
    auto_finalized: float = 0.0
    validation_budget: float = 0.0


def _connected_components(n_bus: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n_bus)]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    seen = np.zeros(n_bus, dtype=bool)
    comps: List[List[int]] = []
    for start in range(n_bus):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(int(u))
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def dc_power_flow(
    case: GridCase,
    outage: Iterable[int] = (),
    emergency_limit: float = 1.10,
    island_weight: float = 10.0,
) -> PFResult:
    """Run DC power flow and return a normalized thermal severity score."""
    outage_tuple: Contingency = tuple(sorted(int(i) for i in outage))
    out = set(outage_tuple)
    active = np.array([i not in out for i in range(case.n_line)], dtype=bool)
    edges = [(int(case.from_bus[i]), int(case.to_bus[i])) for i in range(case.n_line) if active[i]]
    p = case.base_injection.copy()

    island_penalty = 0.0
    feasible = True
    for comp in _connected_components(case.n_bus, edges):
        imbalance = float(p[comp].sum())
        if abs(imbalance) > 1e-7:
            feasible = False
            island_penalty += island_weight * abs(imbalance) / max(1.0, float(case.load.sum()))
            p[comp[0]] -= imbalance

    bbus = np.zeros((case.n_bus, case.n_bus), dtype=float)
    for i in range(case.n_line):
        if not active[i]:
            continue
        a = int(case.from_bus[i])
        b = int(case.to_bus[i])
        bij = 1.0 / max(1e-6, float(case.x[i]))
        bbus[a, a] += bij
        bbus[b, b] += bij
        bbus[a, b] -= bij
        bbus[b, a] -= bij

    keep = [i for i in range(case.n_bus) if i != case.slack]
    theta = np.zeros(case.n_bus, dtype=float)
    try:
        theta[keep] = np.linalg.solve(bbus[np.ix_(keep, keep)], p[keep])
    except np.linalg.LinAlgError:
        feasible = False
        theta[keep] = np.linalg.lstsq(bbus[np.ix_(keep, keep)], p[keep], rcond=None)[0]
        island_penalty += island_weight

    flows = np.zeros(case.n_line, dtype=float)
    for i in range(case.n_line):
        if active[i]:
            flows[i] = (theta[int(case.from_bus[i])] - theta[int(case.to_bus[i])]) / max(1e-6, float(case.x[i]))

    loading = np.zeros(case.n_line, dtype=float)
    loading[active] = np.abs(flows[active]) / np.maximum(1e-6, case.rating[active])
    overload = np.maximum(loading[active] - emergency_limit, 0.0)
    severity = float(np.sum(overload) + island_penalty)
    return PFResult(feasible=feasible, flows=flows, loading=loading, severity=severity, island_penalty=island_penalty, outage=outage_tuple)


def make_synthetic_case(seed: int, n_bus: int = 24, n_line: int = 36, n_gen: int = 5) -> GridCase:
    """Generate a deterministic synthetic case for local development."""
    rng = np.random.default_rng(seed)
    edges: List[Tuple[int, int]] = []
    for b in range(1, n_bus):
        edges.append((int(rng.integers(0, b)), b))
    existing = {tuple(sorted(e)) for e in edges}
    while len(edges) < n_line:
        a, b = int(rng.integers(0, n_bus)), int(rng.integers(0, n_bus))
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in existing:
            continue
        existing.add(key)
        edges.append((a, b))

    from_bus = np.array([a for a, _ in edges], dtype=int)
    to_bus = np.array([b for _, b in edges], dtype=int)
    x = rng.uniform(0.05, 0.22, size=n_line)
    load = rng.lognormal(mean=0.0, sigma=0.55, size=n_bus)
    load[0] *= 0.2
    load *= 1000.0 / load.sum()
    gen_buses = np.array(sorted(set([0] + list(rng.choice(np.arange(1, n_bus), size=n_gen - 1, replace=False)))), dtype=int)
    n_gen = len(gen_buses)
    gen_p = rng.dirichlet(np.ones(n_gen) * 1.4) * load.sum()
    gen_min = np.maximum(0.0, gen_p - 0.35 * load.sum() / n_gen - rng.uniform(20, 80, size=n_gen))
    gen_max = gen_p + 0.35 * load.sum() / n_gen + rng.uniform(20, 80, size=n_gen)
    gen_cost = rng.uniform(8, 30, size=n_gen)

    tmp = GridCase(n_bus, from_bus, to_bus, x, np.full(n_line, 1e4), load, gen_buses, gen_p, gen_min, gen_max, gen_cost)
    base = dc_power_flow(tmp)
    max_abs = np.abs(base.flows) + 1e-3
    for i in range(n_line):
        max_abs = np.maximum(max_abs, np.abs(dc_power_flow(tmp, outage=(i,)).flows))
    rating = np.maximum(15.0, 0.62 * max_abs + rng.uniform(4, 22, size=n_line))
    rating = np.maximum(rating, np.abs(base.flows) / 0.94 + 5.0)
    return GridCase(
        n_bus,
        from_bus,
        to_bus,
        x,
        rating,
        load,
        gen_buses,
        gen_p,
        gen_min,
        gen_max,
        gen_cost,
        name=f"synthetic_{seed}",
        branch_names=[f"B{i}" for i in range(n_line)],
        bus_names=[f"bus{i}" for i in range(n_bus)],
    )


def _series_value(row: Any, names: Sequence[str], default: float) -> float:
    for name in names:
        if name in row and row[name] == row[name]:
            try:
                return float(row[name])
            except Exception:
                pass
    return float(default)


def _pypsa_case_to_gridcase(network: Any, name: str, rating_scale: float = 0.85, include_transformers: bool = True) -> GridCase:
    bus_names = [str(x) for x in list(network.buses.index)]
    bus_index = {bus: i for i, bus in enumerate(bus_names)}

    from_bus: List[int] = []
    to_bus: List[int] = []
    reactance: List[float] = []
    rating: List[float] = []
    branch_names: List[str] = []

    def add_branch(component: str, idx: Any, row: Any) -> None:
        b0, b1 = str(row["bus0"]), str(row["bus1"])
        if b0 not in bus_index or b1 not in bus_index:
            return
        x = abs(_series_value(row, ["x", "x_pu", "x_pu_eff"], 0.05))
        if x <= 1e-8:
            x = 0.01
        r = abs(_series_value(row, ["s_nom", "s_nom_extendable", "rateA"], 0.0))
        if r <= 1e-8:
            r = 1e4
        from_bus.append(bus_index[b0])
        to_bus.append(bus_index[b1])
        reactance.append(x)
        rating.append(max(1.0, rating_scale * r))
        branch_names.append(f"{component}:{idx}")

    for idx, row in network.lines.iterrows():
        if bool(row.get("active", True)):
            add_branch("line", idx, row)
    if include_transformers and hasattr(network, "transformers"):
        for idx, row in network.transformers.iterrows():
            if bool(row.get("active", True)):
                add_branch("transformer", idx, row)

    n_bus = len(bus_names)
    load = np.zeros(n_bus, dtype=float)
    for _, row in network.loads.iterrows():
        bus = str(row["bus"])
        if bus in bus_index:
            load[bus_index[bus]] += max(0.0, _series_value(row, ["p_set", "p"], 0.0))

    gen_buses: List[int] = []
    gen_p: List[float] = []
    gen_min: List[float] = []
    gen_max: List[float] = []
    gen_cost: List[float] = []
    for _, row in network.generators.iterrows():
        bus = str(row["bus"])
        if bus not in bus_index:
            continue
        p = _series_value(row, ["p_set", "p"], 0.0)
        p_nom = max(abs(p), _series_value(row, ["p_nom"], abs(p)))
        p_min_pu = _series_value(row, ["p_min_pu"], 0.0)
        p_max_pu = _series_value(row, ["p_max_pu"], 1.0)
        gmin = min(p, p_min_pu * p_nom)
        gmax = max(p, p_max_pu * p_nom, p + 1.0)
        gen_buses.append(bus_index[bus])
        gen_p.append(p)
        gen_min.append(max(0.0, gmin))
        gen_max.append(max(gmax, p + 1.0))
        gen_cost.append(max(1.0, _series_value(row, ["marginal_cost"], 10.0)))

    if not gen_buses:
        raise ValueError("PyPSA network contains no generators that can be converted to a DC case.")

    case = GridCase(
        n_bus=n_bus,
        from_bus=np.asarray(from_bus, dtype=int),
        to_bus=np.asarray(to_bus, dtype=int),
        x=np.asarray(reactance, dtype=float),
        rating=np.asarray(rating, dtype=float),
        load=load,
        gen_buses=np.asarray(gen_buses, dtype=int),
        gen_p=np.asarray(gen_p, dtype=float),
        gen_min=np.asarray(gen_min, dtype=float),
        gen_max=np.asarray(gen_max, dtype=float),
        gen_cost=np.asarray(gen_cost, dtype=float),
        slack=0,
        name=name,
        branch_names=branch_names,
        bus_names=bus_names,
    )
    return _replace_missing_ratings(case)


def _replace_missing_ratings(case: GridCase) -> GridCase:
    if np.all(case.rating < 1e4):
        return case
    tmp = case.with_generation(case.gen_p)
    tmp.rating = np.full(case.n_line, 1e6)
    base = dc_power_flow(tmp)
    rating = case.rating.copy()
    missing = rating >= 1e4
    rating[missing] = np.maximum(20.0, np.abs(base.flows[missing]) / 0.8 + 10.0)
    case.rating = rating
    return case


def load_case39_dc(
    network_path: str | Path | None = None,
    rating_scale: float = 0.85,
    include_transformers: bool = True,
    variant_seed: int | None = None,
    load_sigma: float = 0.04,
) -> GridCase:
    """Load the repository's existing IEEE 39-bus case and convert it to DC.

    If ``variant_seed`` is supplied, the topology and equipment are unchanged but
    a deterministic small load and dispatch perturbation is applied.  This gives
    multiple test instances while still using the same existing network.
    """
    try:
        import pypsa  # type: ignore
        from poweragentbench.benchmark_utils import NETWORK_FILE, load_or_build_scenario
    except Exception as exc:
        raise RuntimeError("load_case39_dc requires the repository's PyPSA dependencies.") from exc

    if network_path is None:
        network_path = NETWORK_FILE
    path = Path(network_path)
    network = pypsa.Network(path) if path.exists() else load_or_build_scenario(path)
    case = _pypsa_case_to_gridcase(network, name="case39", rating_scale=rating_scale, include_transformers=include_transformers)
    if variant_seed is not None:
        case = perturb_operating_point(case, variant_seed, load_sigma=load_sigma)
        case.name = f"case39_seed_{variant_seed}"
    return case


def perturb_operating_point(case: GridCase, seed: int, load_sigma: float = 0.04) -> GridCase:
    rng = np.random.default_rng(seed)
    load_factor = np.clip(rng.normal(1.0, load_sigma, size=case.n_bus), 0.85, 1.18)
    load = case.load * load_factor
    total = float(load.sum())
    old_total = float(case.gen_p.sum())
    if old_total <= 1e-9:
        return case
    gen_p = case.gen_p * (total / old_total)
    # Add a balanced random dispatch shift inside generator limits.
    shift = rng.normal(0.0, 0.03 * total / max(1, case.n_gen), size=case.n_gen)
    shift -= shift.mean()
    gen_p = np.clip(gen_p + shift, case.gen_min, case.gen_max)
    gen_p *= total / max(1e-9, gen_p.sum())
    out = case.with_generation(gen_p)
    out.load = load
    return out


def contingency_space(case: GridCase, k: int = 2) -> List[Contingency]:
    return [tuple(c) for c in combinations(range(case.n_line), k)]


def evaluate_contingencies(case: GridCase, contingencies: Sequence[Contingency]) -> Dict[Contingency, float]:
    return {tuple(c): dc_power_flow(case, c).severity for c in contingencies}


def compute_ptdf(case: GridCase) -> np.ndarray:
    n = case.n_bus
    m = case.n_line
    bbus = np.zeros((n, n), dtype=float)
    for ell in range(m):
        a, b = int(case.from_bus[ell]), int(case.to_bus[ell])
        bij = 1.0 / max(1e-6, float(case.x[ell]))
        bbus[a, a] += bij
        bbus[b, b] += bij
        bbus[a, b] -= bij
        bbus[b, a] -= bij
    keep = [i for i in range(n) if i != case.slack]
    binv = np.linalg.pinv(bbus[np.ix_(keep, keep)])
    ptdf = np.zeros((m, m), dtype=float)
    for col in range(m):
        inj = np.zeros(n)
        inj[int(case.from_bus[col])] = 1.0
        inj[int(case.to_bus[col])] = -1.0
        theta = np.zeros(n)
        theta[keep] = binv @ inj[keep]
        for row in range(m):
            ptdf[row, col] = (theta[int(case.from_bus[row])] - theta[int(case.to_bus[row])]) / max(1e-6, float(case.x[row]))
    return ptdf


def lodf_matrix(case: GridCase) -> np.ndarray:
    ptdf = compute_ptdf(case)
    lodf = np.zeros_like(ptdf)
    for out in range(case.n_line):
        denom = 1.0 - ptdf[out, out]
        if abs(denom) < 1e-6:
            lodf[:, out] = 0.0
            lodf[out, out] = -1.0
        else:
            lodf[:, out] = ptdf[:, out] / denom
            lodf[out, out] = -1.0
    return lodf


def predicted_nk_severity(case: GridCase, contingency: Contingency, lodf: np.ndarray | None = None) -> float:
    if lodf is None:
        lodf = lodf_matrix(case)
    base = dc_power_flow(case, outage=()).flows
    pred = base.copy()
    for out in contingency:
        pred += lodf[:, int(out)] * base[int(out)]
    for out in contingency:
        pred[int(out)] = 0.0
    loading = np.abs(pred) / np.maximum(1e-6, case.rating)
    return float(np.maximum(loading - 1.10, 0.0).sum())


def base_loading_score(case: GridCase, contingency: Contingency) -> float:
    base = dc_power_flow(case, outage=()).loading
    return float(sum(base[int(i)] for i in contingency))


def degree_score(case: GridCase, contingency: Contingency) -> float:
    degree = np.zeros(case.n_bus, dtype=float)
    for a, b in zip(case.from_bus, case.to_bus):
        degree[int(a)] += 1
        degree[int(b)] += 1
    score = 0.0
    for line in contingency:
        score += degree[int(case.from_bus[int(line)])] + degree[int(case.to_bus[int(line)])]
        score += base_loading_score(case, (int(line),))
    return float(score)


def redispatch_cost(case: GridCase, new_gen: np.ndarray) -> float:
    return float(np.sum(np.abs(new_gen - case.gen_p) * case.gen_cost) / 100.0)


def mitigation_objective(case: GridCase, contingencies: Sequence[Contingency], cost_weight: float = 0.015) -> float:
    score = dc_power_flow(case, ()).severity
    for contingency in contingencies:
        score += dc_power_flow(case, contingency).severity
    return float(score + cost_weight * redispatch_cost(case, case.gen_p))


def greedy_preventive_redispatch(
    case: GridCase,
    contingencies: Sequence[Contingency],
    max_steps: int = 40,
    step_mw: float = 15.0,
) -> Tuple[GridCase, float]:
    if not contingencies:
        return case, 0.0
    current = case
    current_obj = mitigation_objective(current, contingencies)
    for _ in range(max_steps):
        best_case = current
        best_obj = current_obj
        g = current.gen_p
        for up in range(current.n_gen):
            for down in range(current.n_gen):
                if up == down:
                    continue
                trial = g.copy()
                step = min(step_mw, current.gen_max[up] - trial[up], trial[down] - current.gen_min[down])
                if step <= 1e-6:
                    continue
                trial[up] += step
                trial[down] -= step
                candidate = current.with_generation(trial)
                obj = mitigation_objective(candidate, contingencies)
                if obj + 1e-9 < best_obj:
                    best_obj = obj
                    best_case = candidate
        if best_case is current:
            break
        current = best_case
        current_obj = best_obj
    return current, redispatch_cost(case, current.gen_p)


class NoValidationHeuristicAgent:
    def __init__(self, report_k: int):
        self.report_k = int(report_k)

    def run(self, case: GridCase, candidates: Sequence[Contingency]) -> AgentOutput:
        ranked = sorted(candidates, key=lambda c: base_loading_score(case, c), reverse=True)[: self.report_k]
        return AgentOutput("No-validation", {}, ranked, validation_budget=0.0)


class RandomSearchAgent:
    def __init__(self, budget: int, report_k: int, seed: int = 0):
        self.budget = int(budget)
        self.report_k = int(report_k)
        self.seed = int(seed)

    def run(self, case: GridCase, candidates: Sequence[Contingency]) -> AgentOutput:
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(candidates), size=min(self.budget, len(candidates)), replace=False)
        selected = [candidates[int(i)] for i in idx]
        values = evaluate_contingencies(case, selected)
        reported = sorted(values, key=lambda c: values[c], reverse=True)[: self.report_k]
        return AgentOutput("Random", values, reported, validation_budget=float(self.budget))


class DegreeAgent:
    def __init__(self, budget: int, report_k: int):
        self.budget = int(budget)
        self.report_k = int(report_k)

    def run(self, case: GridCase, candidates: Sequence[Contingency]) -> AgentOutput:
        selected = sorted(candidates, key=lambda c: degree_score(case, c), reverse=True)[: self.budget]
        values = evaluate_contingencies(case, selected)
        reported = sorted(values, key=lambda c: values[c], reverse=True)[: self.report_k]
        return AgentOutput("Topology", values, reported, validation_budget=float(self.budget))


class BaseLoadingAgent:
    def __init__(self, budget: int, report_k: int):
        self.budget = int(budget)
        self.report_k = int(report_k)

    def run(self, case: GridCase, candidates: Sequence[Contingency]) -> AgentOutput:
        selected = sorted(candidates, key=lambda c: base_loading_score(case, c), reverse=True)[: self.budget]
        values = evaluate_contingencies(case, selected)
        reported = sorted(values, key=lambda c: values[c], reverse=True)[: self.report_k]
        return AgentOutput("Base-loading", values, reported, validation_budget=float(self.budget))


class LODFAgent:
    def __init__(self, budget: int, report_k: int):
        self.budget = int(budget)
        self.report_k = int(report_k)

    def run(self, case: GridCase, candidates: Sequence[Contingency]) -> AgentOutput:
        lodf = lodf_matrix(case)
        selected = sorted(candidates, key=lambda c: predicted_nk_severity(case, c, lodf), reverse=True)[: self.budget]
        values = evaluate_contingencies(case, selected)
        reported = sorted(values, key=lambda c: values[c], reverse=True)[: self.report_k]
        return AgentOutput("LODF-screen", values, reported, validation_budget=float(self.budget))


class HybridToolAgent:
    def __init__(self, budget: int, report_k: int, base_share: float = 0.55, lodf_share: float = 0.35):
        self.budget = int(budget)
        self.report_k = int(report_k)
        self.base_share = float(base_share)
        self.lodf_share = float(lodf_share)

    def run(self, case: GridCase, candidates: Sequence[Contingency]) -> AgentOutput:
        selected: List[Contingency] = []
        seen: set[Contingency] = set()

        def add_many(items: Sequence[Contingency]) -> None:
            for item in items:
                if item not in seen and len(selected) < self.budget:
                    selected.append(item)
                    seen.add(item)

        n_base = max(1, int(round(self.base_share * self.budget)))
        n_lodf = max(1, int(round(self.lodf_share * self.budget)))
        add_many(sorted(candidates, key=lambda c: base_loading_score(case, c), reverse=True)[:n_base])
        lodf = lodf_matrix(case)
        add_many(sorted(candidates, key=lambda c: predicted_nk_severity(case, c, lodf), reverse=True)[:n_lodf])
        add_many(sorted(candidates, key=lambda c: degree_score(case, c), reverse=True))
        add_many(sorted(candidates, key=lambda c: base_loading_score(case, c), reverse=True))
        values = evaluate_contingencies(case, selected[: self.budget])
        reported = sorted(values, key=lambda c: values[c], reverse=True)[: self.report_k]
        return AgentOutput("Hybrid-tool", values, reported, validation_budget=float(self.budget))


class HybridMitigationAgent(HybridToolAgent):
    def __init__(self, budget: int, report_k: int, search_steps: int = 40, step_mw: float = 15.0):
        super().__init__(budget, report_k)
        self.search_steps = int(search_steps)
        self.step_mw = float(step_mw)

    def run(self, case: GridCase, candidates: Sequence[Contingency]) -> AgentOutput:
        base = super().run(case, candidates)
        focus = base.reported[: min(10, len(base.reported))]
        mitigated, cost = greedy_preventive_redispatch(case, focus, max_steps=self.search_steps, step_mw=self.step_mw)
        post_values = evaluate_contingencies(mitigated, list(base.validated.keys()))
        reported = sorted(post_values, key=lambda c: post_values[c], reverse=True)[: self.report_k]
        return AgentOutput("Hybrid+redispatch", post_values, reported, mitigated_case=mitigated, action_cost=cost, validation_budget=float(self.budget))


def normalize_contingency_list(raw: Any) -> List[Contingency]:
    """Extract N-2 branch contingencies from common unambiguous formats.

    Canonical format is ``[[i, j], ...]``.  The function also accepts common
    wrappers such as ``{"outage": [i, j]}``, but it deliberately rejects
    ambiguous branch/bus dictionaries such as ``{"branch": 22, "bus": 26}``.
    """
    return extract_contingencies(raw).contingencies


@dataclass
class ContingencyParseResult:
    contingencies: List[Contingency]
    schema_repairs: int = 0
    malformed_items: int = 0
    raw_count: int = 0
    type_coercions: int = 0


def _raw_list(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in ("reported", "contingencies", "outages", "candidates", "focus"):
            if key in raw:
                return _raw_list(raw[key])
        return [raw]
    if isinstance(raw, (str, bytes)):
        return [raw]
    try:
        return list(raw)
    except TypeError:
        return [raw]


def _coerce_branch_id(value: Any) -> Tuple[int | None, int]:
    """Return (branch_id, type_coercions). Strings are accepted but counted."""
    if isinstance(value, bool):
        return None, 0
    if isinstance(value, (int, np.integer)):
        return int(value), 0
    if isinstance(value, float) and float(value).is_integer():
        return int(value), 1
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        return int(value), 1
    return None, 0


def _parse_one_contingency(item: Any) -> Tuple[Contingency | None, int, int, int]:
    """Return (contingency, schema_repairs, malformed_items, type_coercions)."""
    repairs = 0
    type_coercions = 0
    if isinstance(item, dict):
        for key in ("outage", "contingency", "lines", "branches"):
            if key in item:
                item = item[key]
                repairs += 1
                break
        else:
            # Reject ambiguous branch/bus dictionaries. They often mix a branch
            # id with a bus id and are not a valid N-2 branch-pair schema.
            if {"branch", "bus"} & set(item.keys()):
                return None, repairs, 1, type_coercions
            if "branch_i" in item and "branch_j" in item:
                item = [item["branch_i"], item["branch_j"]]
                repairs += 1
            elif "line_i" in item and "line_j" in item:
                item = [item["line_i"], item["line_j"]]
                repairs += 1
            elif "i" in item and "j" in item:
                item = [item["i"], item["j"]]
                repairs += 1
            elif "a" in item and "b" in item:
                item = [item["a"], item["b"]]
                repairs += 1
            else:
                return None, repairs, 1, type_coercions
    if isinstance(item, str):
        parts_raw = re.findall(r"\d+", item)
        repairs += 1
    else:
        try:
            parts_raw = list(item)
        except Exception:
            return None, repairs, 1, type_coercions
    if len(parts_raw) < 2:
        return None, repairs, 1, type_coercions
    parts: List[int] = []
    for raw in parts_raw[:2]:
        branch_id, coerced = _coerce_branch_id(raw)
        type_coercions += coerced
        if branch_id is None:
            return None, repairs, 1, type_coercions
        parts.append(branch_id)
    return tuple(sorted(parts)), repairs, 0, type_coercions


def extract_contingencies(raw: Any) -> ContingencyParseResult:
    items = _raw_list(raw)
    contingencies: List[Contingency] = []
    repairs = 0
    malformed = 0
    type_coercions = 0
    for item in items:
        contingency, r, bad, coerced = _parse_one_contingency(item)
        repairs += r
        malformed += bad
        type_coercions += coerced
        if contingency is not None:
            contingencies.append(contingency)
    return ContingencyParseResult(
        contingencies=contingencies,
        schema_repairs=repairs,
        malformed_items=malformed,
        raw_count=len(items),
        type_coercions=type_coercions,
    )


def extract_tool_contingencies(args: MutableMapping[str, Any] | None, primary_key: str) -> ContingencyParseResult:
    """Extract contingencies from tool args with controlled alias support.

    The canonical key is ``primary_key``.  Common aliases are repaired and counted
    so results are fair but schema deviations remain visible in the CSV metrics.
    """
    args = args or {}
    aliases = {
        "contingencies": ["contingencies", "outages", "candidates"],
        "reported": ["reported", "contingencies", "outages", "candidates"],
        "focus": ["focus", "contingencies", "outages", "candidates"],
    }
    keys = aliases.get(primary_key, [primary_key])
    for key in keys:
        if key in args:
            result = extract_contingencies(args[key])
            if key != primary_key:
                result.schema_repairs += 1
            return result
    return ContingencyParseResult(contingencies=[], raw_count=0)


def format_contingencies(values: Dict[Contingency, float], limit: int = 20) -> List[Dict[str, Any]]:
    ranked = sorted(values, key=lambda c: values[c], reverse=True)[:limit]
    return [{"outage": list(c), "severity": round(float(values[c]), 6)} for c in ranked]


@dataclass
class ToolState:
    validation_budget: int
    report_k: int
    validated: Dict[Contingency, float] = field(default_factory=dict)
    mitigated_case: GridCase | None = None
    action_cost: float = 0.0
    tool_log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def remaining_validations(self) -> int:
        return max(0, self.validation_budget - len(self.validated))


class SteadyN2ToolServer:
    """Public tool server used by LLM agents."""

    def __init__(self, case: GridCase, candidates: Sequence[Contingency], validation_budget: int, report_k: int, max_rank_return: int = 120):
        self.case = case
        self.candidates = list(candidates)
        self.candidate_set = set(candidates)
        self.state = ToolState(validation_budget=int(validation_budget), report_k=int(report_k))
        self.max_rank_return = int(max_rank_return)

    def execute(self, tool: str, args: MutableMapping[str, Any] | None) -> Tuple[Dict[str, Any], bool, List[Contingency] | None]:
        args = args or {}
        tool = tool.lower().strip()
        eval_case = self.state.mitigated_case if self.state.mitigated_case is not None else self.case
        if tool == "case_summary":
            base = dc_power_flow(eval_case, ())
            return {
                "case": self.case.name,
                "n_bus": self.case.n_bus,
                "n_branch": self.case.n_line,
                "n_gen": self.case.n_gen,
                "candidate_count": len(self.candidates),
                "base_severity": round(float(base.severity), 6),
                "base_max_loading": round(float(base.loading.max()), 6),
                "remaining_validations": self.state.remaining_validations,
            }, False, None
        if tool == "rank_base_loading":
            top_n = min(int(args.get("top_n", 80)), self.max_rank_return)
            ranked = sorted(self.candidates, key=lambda c: base_loading_score(eval_case, c), reverse=True)[:top_n]
            return {"ranked": [{"outage": list(c), "score": round(base_loading_score(eval_case, c), 6)} for c in ranked], "remaining_validations": self.state.remaining_validations}, False, None
        if tool == "rank_lodf":
            top_n = min(int(args.get("top_n", 80)), self.max_rank_return)
            lodf = lodf_matrix(eval_case)
            ranked = sorted(self.candidates, key=lambda c: predicted_nk_severity(eval_case, c, lodf), reverse=True)[:top_n]
            return {"ranked": [{"outage": list(c), "score": round(predicted_nk_severity(eval_case, c, lodf), 6)} for c in ranked], "remaining_validations": self.state.remaining_validations}, False, None
        if tool == "validate":
            parsed = extract_tool_contingencies(args, "contingencies")
            if parsed.raw_count == 0:
                return {
                    "error": "validate requires args.contingencies as a non-empty list of branch-id pairs, e.g. [[22, 26], [15, 26]].",
                    "malformed_items": parsed.malformed_items,
                    "schema_repairs": parsed.schema_repairs,
                    "type_coercions": parsed.type_coercions,
                    "remaining_validations": self.state.remaining_validations,
                }, False, None
            if not parsed.contingencies:
                return {
                    "error": "validate requires args.contingencies as a list of branch-id pairs, e.g. [[22, 26], [15, 26]].",
                    "malformed_items": parsed.malformed_items,
                    "schema_repairs": parsed.schema_repairs,
                    "type_coercions": parsed.type_coercions,
                    "remaining_validations": self.state.remaining_validations,
                }, False, None
            requested = parsed.contingencies
            invalid_candidates = [c for c in requested if c not in self.candidate_set]
            duplicate_requests = [c for c in requested if c in self.state.validated]
            new: List[Contingency] = []
            seen_new: set[Contingency] = set()
            for contingency in requested:
                if contingency in self.candidate_set and contingency not in self.state.validated and contingency not in seen_new:
                    new.append(contingency)
                    seen_new.add(contingency)
            new = new[: self.state.remaining_validations]
            values = evaluate_contingencies(eval_case, new)
            self.state.validated.update(values)
            observation = {
                "validated_now": format_contingencies(values, len(values)),
                "best_validated_so_far": format_contingencies(self.state.validated, min(20, self.state.report_k)),
                "remaining_validations": self.state.remaining_validations,
                "schema_repairs": parsed.schema_repairs,
                "type_coercions": parsed.type_coercions,
                "duplicate_validation_requests": len(duplicate_requests),
                "invalid_candidate_count": len(invalid_candidates),
                "malformed_items": parsed.malformed_items,
            }
            if parsed.malformed_items > 0 or invalid_candidates:
                observation["error"] = "Some requested contingencies were malformed or outside the candidate set."
            elif requested and not values and not duplicate_requests:
                observation["error"] = "No valid new contingencies were validated."
            return observation, False, None
        if tool == "redispatch":
            parsed = extract_tool_contingencies(args, "focus")
            focus = parsed.contingencies
            if parsed.raw_count > 0 and not focus:
                return {
                    "error": "redispatch focus must be a list of branch-id pairs, e.g. [[22, 26], [15, 26]].",
                    "malformed_items": parsed.malformed_items,
                    "schema_repairs": parsed.schema_repairs,
                    "type_coercions": parsed.type_coercions,
                    "remaining_validations": self.state.remaining_validations,
                }, False, None
            if not focus:
                focus = sorted(self.state.validated, key=lambda c: self.state.validated[c], reverse=True)[:10]
            focus = [c for c in focus if c in self.candidate_set]
            mitigated, cost = greedy_preventive_redispatch(self.case, focus)
            self.state.mitigated_case = mitigated
            self.state.action_cost = float(cost)
            post_values = evaluate_contingencies(mitigated, focus)
            for c, v in post_values.items():
                if c in self.state.validated:
                    self.state.validated[c] = v
            obs = {
                "action": "greedy_preventive_redispatch",
                "cost": round(float(cost), 6),
                "post_focus": format_contingencies(post_values, len(post_values)),
                "remaining_validations": self.state.remaining_validations,
                "schema_repairs": parsed.schema_repairs,
                "type_coercions": parsed.type_coercions,
                "malformed_items": parsed.malformed_items,
            }
            if parsed.malformed_items > 0:
                obs["error"] = "Some redispatch focus contingencies were malformed."
            return obs, False, None
        if tool == "submit":
            parsed = extract_tool_contingencies(args, "reported")
            reported = [c for c in parsed.contingencies if c in self.candidate_set]
            fallback_used = False
            if not reported:
                reported = sorted(self.state.validated, key=lambda c: self.state.validated[c], reverse=True)[: self.state.report_k]
                fallback_used = True
            obs = {
                "submitted": [list(c) for c in reported[: self.state.report_k]],
                "diagnosis": str(args.get("diagnosis", ""))[:1000],
                "validated_calls": len(self.state.validated),
                "action_cost": round(float(self.state.action_cost), 6),
                "schema_repairs": parsed.schema_repairs,
                "type_coercions": parsed.type_coercions,
                "malformed_items": parsed.malformed_items,
                "fallback_to_validated": fallback_used,
            }
            if parsed.raw_count > 0 and not parsed.contingencies:
                obs["error"] = "submit reported must be a list of branch-id pairs, e.g. [[22, 26], [15, 26]]."
            elif parsed.malformed_items > 0:
                obs["error"] = "Some submitted contingencies were malformed and ignored."
            return obs, True, reported
        return {"error": f"Unknown tool '{tool}'.", "allowed_tools": ["case_summary", "rank_base_loading", "rank_lodf", "validate", "redispatch", "submit"]}, False, None


def score_agent(
    original_case: GridCase,
    output: AgentOutput,
    oracle_values: Dict[Contingency, float],
    top_m: int = 20,
    danger_threshold: float = 0.05,
) -> Dict[str, float | str]:
    ranked_oracle = sorted(oracle_values, key=lambda c: oracle_values[c], reverse=True)
    oracle_top = set(ranked_oracle[:top_m])
    dangerous = {c for c, v in oracle_values.items() if v >= danger_threshold}
    found = set(output.validated.keys())
    reported = set(output.reported)
    best_oracle = oracle_values[ranked_oracle[0]] if ranked_oracle else 0.0
    best_validated = max((oracle_values.get(c, 0.0) for c in found), default=0.0)
    best_reported = max((oracle_values.get(c, 0.0) for c in reported), default=0.0)
    found_top = len(found & oracle_top) / max(1, len(oracle_top))
    validated_top = len(reported & found & oracle_top) / max(1, len(oracle_top))
    reported_top = len(reported & oracle_top) / max(1, len(oracle_top))
    danger_recall = len(found & dangerous) / max(1, len(dangerous))
    reported_precision = len(reported & dangerous) / max(1, len(reported))
    evidence_rate = len(reported & found) / max(1, len(reported))

    eval_case = output.mitigated_case if output.mitigated_case is not None else original_case
    pre_top = float(np.mean([dc_power_flow(original_case, c).severity for c in ranked_oracle[:top_m]]))
    post_top = float(np.mean([dc_power_flow(eval_case, c).severity for c in ranked_oracle[:top_m]]))
    reduction = 0.0 if pre_top <= 1e-12 else (pre_top - post_top) / pre_top
    return {
        "agent": output.name,
        "validated_calls": float(len(output.validated)),
        "reported_top20_recall": reported_top,
        "validated_top20_recall": validated_top,
        "found_top20_recall": found_top,
        "danger_recall": danger_recall,
        "reported_precision": reported_precision,
        "evidence_rate": evidence_rate,
        "best_capture_reported": 0.0 if best_oracle <= 1e-12 else best_reported / best_oracle,
        "best_capture_validated": 0.0 if best_oracle <= 1e-12 else best_validated / best_oracle,
        "severity_regret": best_oracle - best_validated,
        "pre_top20_violation": pre_top,
        "post_top20_violation": post_top,
        "violation_reduction": reduction,
        "action_cost": float(output.action_cost),
        "invalid_tool_calls": float(output.invalid_tool_calls),
        "schema_repairs": float(output.schema_repairs),
        "type_coercions": float(output.type_coercions),
        "duplicate_validation_requests": float(output.duplicate_validation_requests),
        "submitted_explicitly": float(output.submitted_explicitly),
        "auto_finalized": float(output.auto_finalized),
        "validation_budget_used": (float(len(output.validated)) / float(output.validation_budget)) if output.validation_budget else 0.0,
    }


def aggregate_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    out: Dict[str, Any] = {"agent": rows[0].get("agent", "agent")}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for key in keys:
        arr = np.array([float(r[key]) for r in rows], dtype=float)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(arr.std(ddof=0))
    return out


def latex_result_row(summary: Dict[str, Any], label: str | None = None) -> str:
    name = label or str(summary.get("agent", "agent"))
    return (
        f"{name} & {summary.get('validated_calls_mean', 0.0):.1f} & "
        f"{summary.get('reported_top20_recall_mean', 0.0):.3f} & "
        f"{summary.get('validated_top20_recall_mean', 0.0):.3f} & "
        f"{summary.get('evidence_rate_mean', 0.0):.3f} & "
        f"{summary.get('best_capture_validated_mean', 0.0):.3f} & "
        f"{summary.get('severity_regret_mean', 0.0):.3f} & "
        f"{summary.get('post_top20_violation_mean', 0.0):.3f} & "
        f"{summary.get('violation_reduction_mean', 0.0):.3f} & "
        f"{summary.get('action_cost_mean', 0.0):.3f} & "
        f"{summary.get('invalid_tool_calls_mean', 0.0):.1f} \\\\" 
    )


def latex_result_row_with_diagnostics(summary: Dict[str, Any], label: str | None = None) -> str:
    """LaTeX row including submit, auto-finalize, duplicate, and repair diagnostics."""
    name = label or str(summary.get("agent", "agent"))
    return (
        f"{name} & {summary.get('validated_calls_mean', 0.0):.1f} & "
        f"{summary.get('reported_top20_recall_mean', 0.0):.3f} & "
        f"{summary.get('validated_top20_recall_mean', 0.0):.3f} & "
        f"{summary.get('found_top20_recall_mean', 0.0):.3f} & "
        f"{summary.get('best_capture_validated_mean', 0.0):.3f} & "
        f"{summary.get('severity_regret_mean', 0.0):.3f} & "
        f"{summary.get('post_top20_violation_mean', 0.0):.3f} & "
        f"{summary.get('violation_reduction_mean', 0.0):.3f} & "
        f"{summary.get('submitted_explicitly_mean', 0.0):.2f} & "
        f"{summary.get('auto_finalized_mean', 0.0):.2f} & "
        f"{summary.get('duplicate_validation_requests_mean', 0.0):.1f} & "
        f"{summary.get('schema_repairs_mean', 0.0):.1f} \\\\"
    )
