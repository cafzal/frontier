"""Structured viz_data builders — the chart payloads the web UI renders.

Split from explorer.py (which keeps the analytics and ASCII renderers): every
explore view attaches one of these JSON payloads alongside its ASCII chart, and
the TypeScript types in ui/lib/viz-data.ts mirror them field-for-field.

Two small helpers live here but are shared with explorer's analytics:
``_dominates_min`` (also used by certify_against_exact) and ``_marginal_window``
(also used by the ASCII marginal-rates renderer) — explorer imports them back,
keeping the import direction explorer -> viz only.
"""

from __future__ import annotations

import numpy as np

from engine.models import Direction, Problem, Run, _content_signature
from solvers import is_exact_solver


def _dominates_min(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> bool:
    """In minimize-space, does point ``a`` strictly (Pareto) dominate ``b``? — no worse on every
    axis and strictly better on at least one."""
    return bool(np.all(a <= b + eps) and np.any(a < b - eps))


def _marginal_window(n: int, inflection: dict | None, max_rows: int | None) -> tuple[int, int]:
    """[start, end) slice of rate rows to show, centered on the inflection.

    Returns (0, n) when no cap applies. Shared by the ASCII and viz_data
    renderers so both show the same window around the knee.
    """
    if not max_rows or n <= max_rows:
        return 0, n
    center = inflection["position"] if inflection else n // 2
    half = max_rows // 2
    start = max(0, center - half)
    end = min(n, start + max_rows)
    start = max(0, end - max_rows)
    return start, end


def _scatter_exact_layer(problem: Problem, run: Run, objectives: list,
                         heuristic_solutions: list, scenario: str | None):
    """Provenance + optional exact-overlay for the frontier scatter.

    `provenance` labels the rendered frontier — `kind` heuristic vs exact, the precise `solver`,
    and `exact_certified` (every point carries an optimality certificate: always on the
    continuous LP/QP path, on MILP only at a zero gap — `solvers.run_is_certified` is the
    single classifier). `exact_overlay` is attached only when a
    *heuristic* base-case frontier is shown while an exact-solver overlay exists for the same
    problem: it carries the exact-certified points to draw on top, plus the heuristic points the
    exact front strictly dominates (`dominated_ids`) — so the chart can show "these looked
    efficient; an exact solve beats them at their own cost." Mirrors the scenario / run_id guards
    in `_frontier_provenance`, so the overlay never leaks onto a scenario frontier (scenario runs
    are NSGA-only; the base-case exact overlay doesn't apply to them).
    """
    from solvers import is_exact_solver, run_is_certified

    is_exact = is_exact_solver(run.solver)
    provenance = {
        "kind": "exact" if is_exact else "heuristic",
        "solver": run.solver,
        "exact_certified": run_is_certified(run, problem.approach),
    }
    exact_run = problem.exact_run
    if (scenario is not None or is_exact or exact_run is None
            or not exact_run.solutions or exact_run.run_id == run.run_id):
        return provenance, None

    names = [o.name for o in objectives]
    # Minimize-space sign per objective (+1 minimize, -1 maximize) so plain ≤ is "no worse" —
    # the same convention `certify_against_exact` uses for its dominance audit.
    sign = np.array([-1.0 if o.direction == Direction.maximize else 1.0 for o in objectives])

    def _matrix(sols):
        return np.array([[s.objective_values.get(n, 0.0) for n in names] for s in sols], dtype=float)

    N = _matrix(heuristic_solutions) * sign       # heuristic frontier, minimize-space
    E = _matrix(exact_run.solutions) * sign       # exact overlay, minimize-space
    # Non-dominated subset of the exact points (integer rounding can leave a few dominated ones).
    exact_front = [k for k in range(len(E))
                   if not any(_dominates_min(E[j], E[k]) for j in range(len(E)) if j != k)]
    dominated_ids = [heuristic_solutions[i].solution_id
                     for i in range(len(N))
                     if any(_dominates_min(E[k], N[i]) for k in exact_front)]
    overlay = {
        "solver": exact_run.solver,
        # Which overlay produced this — certify reports the same id as `exact_run_id`, so a
        # reader can tell a scatter rendered against an older overlay from the current certificate.
        "exact_run_id": exact_run.run_id,
        "exact_certified": run_is_certified(exact_run, problem.approach),
        "points": [{"solution_id": s.solution_id, "values": dict(s.objective_values)}
                   for s in exact_run.solutions],
        "dominated_ids": dominated_ids,
    }
    return provenance, overlay


def _viz_data_tradeoffs(solutions: list, objectives: list, result: dict,
                        curated: list | None = None, problem: Problem | None = None,
                        run: Run | None = None, scenario: str | None = None) -> dict:
    """Scatter-matrix payload: every solution as a point across all objectives.

    Each point carries `name` = the curated custom_name when the solution has been
    curated (matched by content_signature), else None — so the UI can show
    id + name on selection.

    When `problem`/`run` are supplied (the real `explore tradeoffs` path), the payload also
    carries `provenance` (heuristic vs exact-certified) and, on a heuristic base-case frontier
    that has an exact overlay, an `exact_overlay` block (the certified points + the heuristic
    `dominated_ids` they beat), so the chart can denote certification. See `_scatter_exact_layer`.
    """
    provenance, overlay = (None, None)
    if problem is not None and run is not None:
        provenance, overlay = _scatter_exact_layer(problem, run, objectives, solutions, scenario)

    obj_meta = []
    for o in objectives:
        lo = result["objective_ranges"][o.name]["min"]
        hi = result["objective_ranges"][o.name]["max"]
        if overlay:
            # Widen the axis so a sharpened exact corner (which can sit past the heuristic
            # range) renders in-frame rather than clipped at the plot edge.
            ovals = [p["values"][o.name] for p in overlay["points"] if o.name in p["values"]]
            if ovals:
                lo, hi = min(lo, min(ovals)), max(hi, max(ovals))
        obj_meta.append({
            "name": o.name,
            "direction": o.direction.value,
            "min": lo,
            "max": hi,
        })
    name_by_sig = {
        cs.content_signature: cs.custom_name
        for cs in (curated or [])
        if cs.custom_name
    }
    points = [
        {
            "solution_id": s.solution_id,
            "values": dict(s.objective_values),
            "name": name_by_sig.get(
                s.content_signature or _content_signature(s.selected_options, s.allocations)
            ),
        }
        for s in solutions
    ]
    extremes = {}
    for o in objectives:
        reverse = o.direction.value == "maximize"
        best = (max if reverse else min)(solutions, key=lambda s: s.objective_values[o.name])
        worst = (min if reverse else max)(solutions, key=lambda s: s.objective_values[o.name])
        extremes[o.name] = {"best_id": best.solution_id, "worst_id": worst.solution_id}
    inflection_ids = [c["solution_id"] for c in result.get("inflection_point_candidates", [])]
    out = {
        "type": "scatter",
        "objectives": obj_meta,
        "points": points,
        "extremes": extremes,
        "balanced_id": result["balanced_solution"]["solution_id"],
        "inflection_ids": inflection_ids,
    }
    if provenance is not None:
        out["provenance"] = provenance
    if overlay is not None:
        out["exact_overlay"] = overlay
    return out


def _viz_data_parallel_coords(
    solutions_data: list[dict], objectives: list, labels: dict
) -> dict:
    """Parallel-coordinates payload for `compare` and `compare_curated`.

    Each entry in `solutions_data` carries either `solution_id` (run frontier) or
    `content_signature` (curated set); we expose the present id under a generic `id`
    field so the renderer doesn't have to branch.
    """
    axes = []
    for o in objectives:
        vals = [s["objective_values"][o.name] for s in solutions_data]
        axes.append({
            "name": o.name,
            "direction": o.direction.value,
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
        })
    series = []
    for s in solutions_data:
        sid = s.get("solution_id") if "solution_id" in s else s.get("content_signature")
        series.append({
            "id": sid,
            "label": labels.get(sid, str(sid)),
            "values": dict(s["objective_values"]),
        })
    return {"type": "parallel_coords", "axes": axes, "series": series}


def _viz_data_marginal_rates(
    rates: list[dict], obj_a, obj_b, inflection: dict | None, max_rows: int | None = None
) -> dict:
    """Marginal-rate bar payload: cost-per-unit between adjacent solutions.

    The full rate list is one row per Pareto transition — hundreds long on a
    large frontier, which renders an unusably tall chart. max_rows windows the
    rows around the inflection (the decision-relevant region) and re-indexes the
    inflection marker into the slice. Omit max_rows (detail mode) for all rows.
    """
    start, end = _marginal_window(len(rates), inflection, max_rows)
    sliced = rates[start:end]
    if inflection is not None:
        pos = inflection["position"] - start
        inflection = {**inflection, "position": pos if 0 <= pos < len(sliced) else -1}
    return {
        "type": "marginal_rates",
        "from_objective": {"name": obj_a.name, "direction": obj_a.direction.value},
        "to_objective": {"name": obj_b.name, "direction": obj_b.direction.value},
        "rates": sliced,
        "inflection": inflection,
    }


def _viz_data_scenario_parcoords(
    scenario_runs: dict, objectives: list, max_per_scenario: int = 80, problem: Problem | None = None
) -> dict:
    """Parallel-coords payload overlaying each scenario's frontier, colored by scenario.

    Field lines (each scenario's frontier) are evenly sampled per scenario (cap
    max_per_scenario) so the overlay stays readable. When the problem carries curated
    picks, each pick is also evaluated under every scenario and returned in ``curated``
    so the UI can render it as a bold, labelled line over the faint field — the
    emphasis channel for curation (weight/opacity), leaving colour free for scenario.

    A pick whose objective vector is identical across scenarios (e.g. constraint-only
    scenarios leave a fixed slate's scores unchanged) collapses to one ``invariant``
    line; a pick that shifts across scenarios (score-based scenarios) draws one line
    per scenario. This is fully data-driven — no per-problem assumptions.
    """
    obj_names = [o.name for o in objectives]
    scenario_names = list(scenario_runs.keys())
    name_to_idx = {n: i for i, n in enumerate(scenario_names)}

    lines = []
    for idx, name in enumerate(scenario_names):
        sols = scenario_runs[name].solutions
        if len(sols) > max_per_scenario:
            step = len(sols) / max_per_scenario
            sols = [sols[int(i * step)] for i in range(max_per_scenario)]
        for s in sols:
            lines.append({
                "scenario": idx,
                "values": {n: s.objective_values.get(n, 0.0) for n in obj_names},
            })

    curated = []
    if problem is not None and problem.curated_solutions and problem.scenario_config:
        from . import optimizer
        for cs in problem.curated_solutions:
            per = []      # (scenario_idx, values) — pick's profile under each scenario
            present = []  # scenario idxs where the slate is feasible (lives there)
            for sc in problem.scenario_config.scenarios:
                sidx = name_to_idx.get(sc.name)
                if sidx is None:
                    continue
                r = optimizer.score_slate(problem, cs.selected_options, cs.allocations, scenario=sc)
                per.append((sidx, {n: r["values"].get(n, 0.0) for n in obj_names}))
                if r["feasible"]:
                    present.append(sidx)
            if not per:
                continue
            first = per[0][1]
            invariant = all(v == first for _, v in per)
            clines = (
                [{"scenario": -1, "values": first}]
                if invariant
                else [{"scenario": sidx, "values": v} for sidx, v in per]
            )
            label = cs.custom_name or (cs.content_signature[:8] if cs.content_signature else "pick")
            # present: which scenarios this pick is feasible in — the UI colors a
            # multi-scenario pick distinctly (robust/shared) vs a single-scenario one.
            curated.append({"name": label, "invariant": invariant, "lines": clines, "present": present})

    # Axis ranges span field + curated so bold pick lines are never clipped.
    axes = []
    for o in objectives:
        vals = [l["values"].get(o.name, 0.0) for l in lines]
        for c in curated:
            vals += [cl["values"].get(o.name, 0.0) for cl in c["lines"]]
        axes.append({
            "name": o.name,
            "direction": o.direction.value,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
        })

    return {
        "type": "scenario_parcoords",
        "axes": axes,
        "scenarios": scenario_names,
        "lines": lines,
        "curated": curated,
    }


def _viz_data_scenario_summary(result: dict) -> dict:
    """Scenario summary payload — option robustness tiers, per-objective risk, regret."""
    return {
        "type": "scenario_summary",
        # Staleness rides along with the numbers it qualifies — a chart surface that
        # rendered these ranges unmarked would invite exactly the reading the payload's
        # marker exists to prevent. Absent when the runs match the current model.
        **({"scenario_results_stale": True,
            "scenario_stale_note": result["scenario_stale_note"]}
           if result.get("scenario_results_stale") else {}),
        # Scenario names are the keys of per_scenario (there is no top-level
        # "scenarios" key); the panel keys its visibility off this list, so an
        # empty list hides the whole panel — derive it from per_scenario.
        "scenarios": list(result.get("per_scenario", {})),
        "option_robustness": result.get("option_robustness", []),
        "expected_values": result.get("expected_values", {}),
        "scenario_risk": result.get("scenario_risk", {}),
        # Minimax-regret lens (per-solution, distinct from the per-objective risk
        # table) — already computed onto result above; carry it so the panel can
        # render it. {"available": False} when there's no base run to regret against.
        "regret": result.get("regret", {}),
    }
