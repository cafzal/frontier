"""Explorer: navigate the Pareto frontier after solving."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from .models import (
    Aggregation,
    Approach,
    CuratedSolution,
    Direction,
    Problem,
    Run,
    _content_signature,
    scenario_base_mismatch,
    scenario_results_stale,
    stale_scenario_overrides,
)
from .viz import (
    _dominates_min,
    _marginal_window,
    _scatter_exact_layer,
    _viz_data_marginal_rates,
    _viz_data_parallel_coords,
    _viz_data_scenario_parcoords,
    _viz_data_scenario_summary,
    _viz_data_tradeoffs,
)


def get_tradeoffs(problem: Problem, scenario: str | None = None, source: str | None = None) -> dict:
    """Frontier overview: ranges, correlations, extremes, balanced solution."""
    run = _require_run(problem, scenario, source)
    solutions = run.solutions
    obj_names = [o.name for o in problem.objectives]

    # Objective ranges
    obj_ranges = {}
    for name in obj_names:
        vals = [s.objective_values[name] for s in solutions]
        obj_ranges[name] = {"min": min(vals), "max": max(vals)}

    # Key tradeoffs: direction-normalized Pearson correlation + normalized MI.
    # Each objective's raw values are flipped when its direction is minimize, so
    # positive r means "both get better together" (redundancy candidate) and
    # negative r means "improving one hurts the other" (genuine tradeoff).
    # MI is invariant to monotonic transforms, so normalization doesn't shift it,
    # but we keep the interpretation consistent with r below.
    if len(solutions) >= 3 and len(obj_names) >= 2:
        matrix = np.array([
            [s.objective_values[name] for name in obj_names]
            for s in solutions
        ])
        dir_signs = np.array([
            -1.0 if o.direction.value == "minimize" else 1.0
            for o in problem.objectives
        ])
        norm_matrix = matrix * dir_signs
        corr = np.corrcoef(norm_matrix.T)
        n = len(solutions)
        mi_reliable = n >= 15
        key_tradeoffs = []
        for i in range(len(obj_names)):
            for j in range(i + 1, len(obj_names)):
                r = float(corr[i, j])
                mi = _normalized_mi(norm_matrix[:, i], norm_matrix[:, j]) if mi_reliable else None
                entry = {
                    "objectives": [obj_names[i], obj_names[j]],
                    "correlation": round(r, 2),
                }
                if mi is not None:
                    entry["mutual_info_normalized"] = round(mi, 2)
                key_tradeoffs.append(entry)
    else:
        key_tradeoffs = []

    # Extreme solutions: top performer per objective
    extreme_solutions = {}
    for obj in problem.objectives:
        name = obj.name
        reverse = obj.direction.value == "maximize"
        best = max(solutions, key=lambda s: s.objective_values[name]) if reverse else min(solutions, key=lambda s: s.objective_values[name])
        extreme_solutions[f"extreme_{name}"] = {
            "solution_id": best.solution_id,
            "value": best.objective_values[name],
            "selected_options": best.selected_options,
        }

    # Balanced solution: min normalized distance to ideal point
    balanced = _find_balanced(solutions, problem.objectives)

    # Inflection-point candidates: solutions where marginal tradeoff cost jumps
    inflection_candidates = []
    for inf in _find_inflection_solutions(solutions, problem.objectives, problem):
        s = inf["solution"]
        if s.solution_id != balanced.solution_id:  # Don't duplicate balanced
            inflection_candidates.append({
                "solution_id": s.solution_id,
                "objective_values": s.objective_values,
                "selected_options": s.selected_options,
                "inflection_pair": inf["pair"],
                "jump_factor": inf["jump_factor"],
                **({"co_improvement": True} if inf["co_improvement"] else {}),
                "rationale": inf["rationale"],
            })

    result = {
        "total_solutions": len(solutions),
        "objective_ranges": obj_ranges,
        "key_tradeoffs": key_tradeoffs,
        "extreme_solutions": extreme_solutions,
        "balanced_solution": {
            "solution_id": balanced.solution_id,
            "objective_values": balanced.objective_values,
            "selected_options": balanced.selected_options,
        },
        "inflection_point_candidates": inflection_candidates,
    }

    if problem.reference_points:
        result["balanced_vs_references"] = _compute_reference_analysis(
            balanced.objective_values, problem.reference_points, problem.objectives,
        )

    # Frontier shape classification per conflicting objective pair
    result["frontier_shape"] = _classify_frontier_shapes(solutions, problem.objectives)

    # Shadow-price guidance for binding constraints
    result["binding_analysis"] = _binding_analysis(problem, solutions)

    # Objective redundancy: classify each pair using Pearson + MI, flag disagreement
    result["objective_redundancy"] = _objective_redundancy(key_tradeoffs, len(solutions))

    # First-class option data: how often each option appears across the frontier
    # (consensus vs distinctive). Deeper composition/patterns via `explore composition`.
    result["option_selection"] = option_selection_stats(solutions, problem.approach.value)

    # Cue the agent to layer scenario_risk into narration when scenarios exist.
    # Without this, narration from tradeoffs alone forgets scenario data on disk.
    if scenario is None and problem.scenario_run and problem.scenario_run.scenario_runs:
        result["scenarios_available"] = {
            "scenario_names": list(problem.scenario_run.scenario_runs.keys()),
            "hint": "Scenario data is on disk. Layer scenario_risk (CVaR, worst_case) and option_robustness into the narration via `explore scenario_results`.",
        }

    result["visualization"] = _render_tradeoffs_viz(result, problem.objectives, solutions)
    result["viz_data"] = _viz_data_tradeoffs(
        solutions, problem.objectives, result, problem.curated_solutions,
        problem=problem, run=run, scenario=scenario,
    )
    result["frontier_source"] = _frontier_provenance(problem, run, scenario)
    return result


def compare_solutions(problem: Problem, solution_ids: list[int], scenario: str | None = None,
                      source: str | None = None, detail: bool = False) -> dict:
    """Side-by-side comparison of specific solutions.

    ``detail`` works as it does on the curated form (``compare_curated``) and on
    ``get_solutions``: default compact — ``{solution_id, objective_values,
    content_signature}`` per solution, since `shared_options` / `differentiating_options`
    already carry the structural difference; ``detail=True`` returns the full dump
    (`selected_options`, `allocations`). On proportional problems the per-option
    `allocation_comparison` table rides both modes — it IS the comparison.
    """
    run = _require_run(problem, scenario, source)
    sol_map = {s.solution_id: s for s in run.solutions}

    selected = []
    for sid in solution_ids:
        if sid not in sol_map:
            raise ValueError(f"Solution {sid} not found in current run.")
        selected.append(sol_map[sid])

    # Shared and differentiating options
    option_sets = [set(s.selected_options) for s in selected]
    shared = set.intersection(*option_sets) if option_sets else set()
    all_options = set.union(*option_sets) if option_sets else set()
    differentiating = all_options - shared

    # Tradeoff summary per objective
    tradeoff_summary = {}
    for obj in problem.objectives:
        name = obj.name
        vals = {s.solution_id: s.objective_values[name] for s in selected}
        best_id = max(vals, key=vals.get) if obj.direction.value == "maximize" else min(vals, key=vals.get)
        worst_id = min(vals, key=vals.get) if obj.direction.value == "maximize" else max(vals, key=vals.get)
        all_vals = list(vals.values())
        tradeoff_summary[name] = {
            "leads": best_id,
            "trails": worst_id,
            "range": [min(all_vals), max(all_vals)],
        }

    result = {
        "solutions": [
            s.model_dump() if detail else {
                "solution_id": s.solution_id,
                "objective_values": s.objective_values,
                "content_signature": s.content_signature,
            }
            for s in selected
        ],
        "shared_options": sorted(shared),
        "differentiating_options": sorted(differentiating),
        "tradeoff_summary": tradeoff_summary,
        "detail": detail,
    }

    # Allocation comparison for proportional mode
    if any(s.allocations for s in selected):
        alloc_comparison = {}
        for opt in sorted(all_options):
            alloc_comparison[opt] = {
                f"solution_{s.solution_id}": (s.allocations or {}).get(opt, 0)
                for s in selected
            }
        result["allocation_comparison"] = alloc_comparison

    # Parallel coordinates visualization
    sol_dicts = [{"solution_id": s.solution_id, "objective_values": s.objective_values}
                 for s in selected]
    labels = {s.solution_id: f"[{s.solution_id}]" for s in selected}
    result["visualization"] = _render_parallel_coords(sol_dicts, problem.objectives, labels)
    result["viz_data"] = _viz_data_parallel_coords(sol_dicts, problem.objectives, labels)
    result["frontier_source"] = _frontier_provenance(problem, run, scenario)

    return result


def get_solutions(problem: Problem, scenario: str | None = None, detail: bool = False, source: str | None = None) -> dict:
    """Pareto frontier, sorted by first objective.

    Default (detail=False): compact — `{solution_id, objective_values, content_signature}` per solution.
    detail=True: full dump including `selected_options` and `allocations`.

    For full single-solution detail prefer `get_solution(problem, solution_id)`.
    """
    run = _require_run(problem, scenario, source)
    if detail:
        sols = [s.model_dump() for s in run.solutions]
    else:
        sols = [
            {
                "solution_id": s.solution_id,
                "objective_values": s.objective_values,
                "content_signature": s.content_signature,
            }
            for s in run.solutions
        ]
    result = {
        "run_id": run.run_id,
        "total_solutions": len(run.solutions),
        "detail": detail,
        "solutions": sols,
    }
    # Parallel-coordinates over the full frontier — ASCII for chat/coding-agent
    # surfaces, viz_data for chart-rendering hosts (the web UI draws D3 from it).
    sol_dicts = [
        {"solution_id": s.solution_id, "objective_values": s.objective_values}
        for s in run.solutions
    ]
    labels = {s.solution_id: f"[{s.solution_id}]" for s in run.solutions}
    result["visualization"] = _render_parallel_coords(sol_dicts, problem.objectives, labels)
    result["viz_data"] = _viz_data_parallel_coords(sol_dicts, problem.objectives, labels)
    result["option_selection"] = option_selection_stats(run.solutions, problem.approach.value)
    result["frontier_source"] = _frontier_provenance(problem, run, scenario)
    return result


def get_solution(problem: Problem, solution_id: int, scenario: str | None = None, source: str | None = None) -> dict:
    """Single solution detail by ID."""
    run = _require_run(problem, scenario, source)
    for s in run.solutions:
        if s.solution_id == solution_id:
            result = s.model_dump()
            if problem.reference_points:
                result["vs_references"] = _compute_reference_analysis(
                    s.objective_values, problem.reference_points, problem.objectives,
                )
            # First-class option data: how common each of this solution's options is
            # across the frontier (consensus vs distinctive pick).
            stats = {e["option"]: e["selection_pct"]
                     for e in option_selection_stats(run.solutions, problem.approach.value)}
            present = _present_options(s, problem.approach.value == "proportional" or bool(s.allocations))
            result["option_context"] = {opt: stats.get(opt, 0.0) for opt in sorted(present)}
            result["frontier_source"] = _frontier_provenance(problem, run, scenario)
            return result
    raise ValueError(f"Solution {solution_id} not found in current run.")


def resolve_signature(problem: Problem, content_signature: str,
                      scenario: str | None = None, source: str | None = None) -> int:
    """``content_signature`` -> ``solution_id`` within the run a read would select.

    The certificate hands out signatures, not ids: ``per_pick.dominated_by`` names the
    certified plan that beats a finalist, and `certify`'s own ``next_steps`` says to go
    match it and curate it in its place. Without this, doing that means listing the whole
    exact frontier and scanning by eye — which overflows the response cap on a few hundred
    points, so the answer the certificate produced ends up back on disk. That is precisely
    the round trip per-pick verdicts exist to remove.

    Signatures are the identifier that survives a re-solve (ids get reassigned), so they
    are also the right thing to hold across the curate/certify loop. Resolution is scoped
    to the selected run: the same plan carries the same signature in the heuristic run and
    the exact overlay, and the caller's ``source``/``scenario`` says which one it means.
    """
    run = _require_run(problem, scenario, source)
    for s in run.solutions:
        sig = s.content_signature or _content_signature(s.selected_options, s.allocations)
        if sig == content_signature:
            return s.solution_id
    where = f'source="{source}"' if source else "the current run"
    raise ValueError(
        f"No solution with content_signature '{content_signature}' in {where}"
        + (f' (scenario "{scenario}")' if scenario else "")
        + ". Signatures are scoped to a run — a certified point's signature resolves "
          'against source="exact".')


# ─── Solution composition & pattern analysis (knowledge-discovery pillar) ───


def _present_options(s, proportional: bool) -> set:
    """Options actually used by a solution — non-zero allocation (proportional) or selected (binary)."""
    if proportional and s.allocations:
        return {o for o, p in s.allocations.items() if p > 0}
    return set(s.selected_options or [])


def option_selection_stats(solutions: list, approach: str) -> list[dict]:
    """Per-option selection frequency across a solution set — first-class option data.

    Binary: selection_count / selection_pct (fraction of solutions including the option).
    Proportional: inclusion count/pct (non-zero allocation) plus mean allocation weight.
    Sorted by selection_pct descending. Mirrors the per-scenario Counter pattern in
    get_scenario_results, but over a single frontier (or curated subset).
    """
    n = len(solutions)
    if n == 0:
        return []
    proportional = approach == "proportional" or any(s.allocations for s in solutions)
    counts: dict[str, int] = {}
    weight_sum: dict[str, float] = {}
    for s in solutions:
        if proportional and s.allocations:
            for opt, pct in s.allocations.items():
                if pct > 0:
                    counts[opt] = counts.get(opt, 0) + 1
                    weight_sum[opt] = weight_sum.get(opt, 0.0) + pct
        else:
            for opt in (s.selected_options or []):
                counts[opt] = counts.get(opt, 0) + 1
    stats = []
    for opt, c in counts.items():
        entry = {"option": opt, "selection_count": c, "selection_pct": round(c / n, 3)}
        if proportional:
            entry["mean_weight"] = round(weight_sum.get(opt, 0.0) / n, 1)
            entry["mean_weight_if_included"] = round(weight_sum.get(opt, 0.0) / c, 1) if c else 0.0
        stats.append(entry)
    stats.sort(key=lambda e: e["selection_pct"], reverse=True)
    return stats


def _co_occurrence(solutions: list, option_selection: list[dict], proportional: bool, top: int = 8) -> list[dict]:
    """Option-pair lift: lift(A,B) = P(A∧B)/(P(A)·P(B)). >1 complements, <1 substitutes.

    Ranked by departure from independence (|lift-1|), top-N — no value cutoff, the
    consumer judges salience from the ranked list.
    """
    n = len(solutions)
    if n == 0:
        return []
    presence = [_present_options(s, proportional) for s in solutions]
    p_single = {e["option"]: e["selection_count"] / n for e in option_selection}
    # Pairs only meaningful for options that sometimes-but-not-always appear.
    opts = [o for o in p_single if 0 < p_single[o] < 1]
    pairs = []
    for i in range(len(opts)):
        for j in range(i + 1, len(opts)):
            a, b = opts[i], opts[j]
            both = sum(1 for ps in presence if a in ps and b in ps) / n
            denom = p_single[a] * p_single[b]
            if denom == 0:
                continue
            lift = both / denom
            pairs.append({
                "options": [a, b],
                "lift": round(lift, 2),
                "relation": "complement" if lift > 1 else "substitute",
            })
    pairs.sort(key=lambda x: abs(x["lift"] - 1), reverse=True)
    return pairs[:top]


def _design_principles(solutions: list, option_selection: list[dict], co_occ: list[dict],
                       problem: Problem, proportional: bool) -> list[dict]:
    """Statements that hold across the set (innovization-lite): always/never present,
    co-occurs/substitutes, and region-bound (presence tracks an objective)."""
    n = len(solutions)
    principles = []
    for e in option_selection:
        if e["selection_count"] == n:
            principles.append({"type": "always", "options": [e["option"]], "support": 1.0,
                               "detail": f"Every solution on this set includes {e['option']}."})
    present_opts = {e["option"] for e in option_selection}
    for o in problem.options:
        if o.name not in present_opts:
            principles.append({"type": "never", "options": [o.name], "support": 1.0,
                               "detail": f"{o.name} appears in no solution on this set."})
    # region_bound: option presence correlates with an objective value (point-biserial)
    if n >= 5:
        varying = [e["option"] for e in option_selection if 0 < e["selection_count"] < n]
        for opt in varying:
            pres = np.array([1.0 if opt in _present_options(s, proportional) else 0.0 for s in solutions])
            if pres.std() == 0:
                continue
            best = None
            for ob in problem.objectives:
                vals = np.array([s.objective_values[ob.name] for s in solutions])
                if vals.std() == 0:
                    continue
                r = float(np.corrcoef(pres, vals)[0, 1])
                if best is None or abs(r) > abs(best[1]):
                    best = (ob.name, r)
            if best and abs(best[1]) >= 0.6:
                where = "high" if best[1] > 0 else "low"
                principles.append({"type": "region_bound", "options": [opt], "support": round(abs(best[1]), 2),
                                   "detail": f"{opt} appears mainly in {where}-{best[0]} solutions (r={best[1]:+.2f})."})
    for pair in co_occ[:3]:
        if pair["relation"] == "complement" and pair["lift"] >= 1.5:
            principles.append({"type": "co_occurs", "options": pair["options"], "support": round(min(pair["lift"], 9.99), 2),
                               "detail": f"{pair['options'][0]} and {pair['options'][1]} tend to appear together (lift {pair['lift']})."})
        elif pair["relation"] == "substitute" and pair["lift"] <= 0.5:
            principles.append({"type": "substitutes", "options": pair["options"], "support": round(1 - pair["lift"], 2),
                               "detail": f"{pair['options'][0]} and {pair['options'][1]} rarely appear together (lift {pair['lift']})."})
    return principles


def _cut_by_largest_gap(merge_d, n: int) -> int:
    """Pick cluster count from the largest jump in merge distance — the natural break
    where tightly-grouped solutions stop merging. Clamped to a legible handful."""
    if len(merge_d) < 2:
        return 1
    diffs = np.diff(merge_d)
    if diffs.size == 0:
        return 1
    gap_idx = int(np.argmax(diffs))
    k = len(merge_d) - gap_idx  # merges after the gap aren't applied → that many clusters
    return max(2, min(k, 6, n))


def _decision_space_clusters(solutions: list, problem: Problem, proportional: bool) -> list[dict]:
    """Cluster solutions in DECISION space (composition), not objective space. Reports
    each family's defining options and its objective spread — surfacing the survey's
    point that good decision-space clusters need not be good objective-space clusters."""
    n = len(solutions)
    if n < 4:
        return []
    opt_names = [o.name for o in problem.options]
    if proportional:
        X = np.array([[(s.allocations or {}).get(o, 0) for o in opt_names] for s in solutions], dtype=float)
        metric = "euclidean"
    else:
        X = np.array([[1.0 if o in set(s.selected_options) else 0.0 for o in opt_names] for s in solutions], dtype=float)
        metric = "jaccard"
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
    except Exception:
        return []
    d = pdist(X, metric=metric)
    if d.size == 0 or not np.isfinite(d).any() or float(np.nanmax(d)) == 0.0:
        return []
    d = np.nan_to_num(d, nan=0.0)
    Z = linkage(d, method="average")
    k = _cut_by_largest_gap(Z[:, 2], n)
    if k < 2:
        return []
    labels = fcluster(Z, t=k, criterion="maxclust")
    member_ids_by_cluster: dict[int, list] = {}
    for i, lab in enumerate(labels):
        member_ids_by_cluster.setdefault(int(lab), []).append(solutions[i])
    clusters = []
    for cid, members in member_ids_by_cluster.items():
        rep = _cluster_medoid(members, opt_names, proportional)
        defining = _defining_options(members, solutions, proportional)
        sig = {}
        for ob in problem.objectives:
            vals = [m.objective_values[ob.name] for m in members]
            sig[ob.name] = {"min": round(min(vals), 2), "max": round(max(vals), 2),
                            "mean": round(sum(vals) / len(vals), 2)}
        clusters.append({
            "cluster_id": cid,
            "size": len(members),
            "representative_solution_id": rep.solution_id,
            "defining_options": defining,
            "objective_signature": sig,
        })
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters


def _clusters_summary(clusters: list[dict]) -> dict | None:
    """Partition quality beside the families themselves: how many, and how much of the set
    the largest one holds. Without it a 104/1/1/1 split reads exactly like a meaningful
    155/19 — same block, same field names, a very different decision."""
    if not clusters:
        return None
    sizes = [c["size"] for c in clusters]
    total = sum(sizes)
    share = round(max(sizes) / total, 2) if total else 0.0
    summary = {"n_families": len(clusters), "largest_family_share": share}
    if share > 0.9:
        summary["note"] = ("one strategy with corner variants, not a menu of families — "
                           "present the family, not a menu")
    return summary


def _cluster_medoid(members: list, opt_names: list[str], proportional: bool):
    if len(members) == 1:
        return members[0]
    if proportional:
        M = np.array([[(m.allocations or {}).get(o, 0) for o in opt_names] for m in members], dtype=float)
    else:
        M = np.array([[1.0 if o in set(m.selected_options) else 0.0 for o in opt_names] for m in members], dtype=float)
    centroid = M.mean(axis=0)
    return members[int(np.argmin(np.linalg.norm(M - centroid, axis=1)))]


def _defining_options(members: list, all_solutions: list, proportional: bool, top: int = 4) -> list[str]:
    """Options notably more common inside the cluster than outside it."""
    member_ids = {id(m) for m in members}
    others = [s for s in all_solutions if id(s) not in member_ids]
    ni, no = len(members), max(len(others), 1)

    def freq(sols, opt):
        return sum(1 for s in sols if opt in _present_options(s, proportional))

    opt_universe = set()
    for s in members:
        opt_universe |= _present_options(s, proportional)
    scored = []
    for opt in opt_universe:
        fi = freq(members, opt) / ni
        fo = freq(others, opt) / no if others else 0.0
        if fi >= 0.5 and (fi - fo) > 0.2:
            scored.append((opt, fi - fo))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [o for o, _ in scored[:top]]


_THIN_EVIDENCE_FLOOR = 3   # rated solutions per side below which a rule is a hypothesis


def _feedback_rules(problem: Problem, run_solutions: list, proportional: bool) -> dict:
    """Induce rules separating liked (rating ≥4) from disliked (≤2) solutions.

    No fixed count gate — a rule surfaces only when it MEASURABLY separates the two
    classes (separation × coverage), so weak/no-signal cases return cleanly. A clean
    separator is a candidate latent constraint → route to problem_framing.

    Every block carries `n_liked` / `n_disliked`, the rated solutions the rules were
    induced from: separation 1.0 and coverage 1.0 off ONE liked solution is arithmetic,
    not evidence, and the scores alone read identically to a rule learned from twenty.
    Below `_THIN_EVIDENCE_FLOOR` on either side the block says so in its note.
    """
    if not problem.feedback:
        return {"available": False, "n_liked": 0, "n_disliked": 0,
                "note": "No feedback yet — rate solutions (explore feedback) to learn liked/disliked patterns."}
    sig_to_sol = {s.content_signature: s for s in run_solutions}
    id_to_sol = {s.solution_id: s for s in run_solutions}
    rated: dict[str, int] = {}
    for fb in problem.feedback:
        if fb.rating is None:
            continue
        sig = fb.content_signature
        if sig is None and fb.solution_id is not None and fb.solution_id in id_to_sol:
            sig = id_to_sol[fb.solution_id].content_signature
        if sig:
            rated[sig] = fb.rating  # appended in order → latest wins
    liked = [sig for sig, r in rated.items() if r >= 4 and sig in sig_to_sol]
    disliked = [sig for sig, r in rated.items() if r <= 2 and sig in sig_to_sol]
    evidence = {"n_liked": len(liked), "n_disliked": len(disliked)}
    # Ratings on plans this frontier is not carrying (rated on another run, or the plan
    # left the frontier on a re-solve) can't feed rule induction — but omitting them
    # silently reads as "your ratings vanished" (observed live). Name them.
    off_frontier = sum(1 for sig in rated if sig not in sig_to_sol)
    if off_frontier:
        evidence["n_rated_off_frontier"] = off_frontier
    off_tail = (f" {off_frontier} rated solution(s) sit on a frontier this composition "
                "is not reading — pass the source/scenario they were rated on to use them."
                if off_frontier else "")
    if not liked or not disliked:
        return {"available": False, **evidence,
                "note": (f"Need rated solutions on both sides in the current frontier; "
                         f"have {len(liked)} liked, {len(disliked)} disliked.{off_tail}")}
    pos = [_present_options(sig_to_sol[s], proportional) for s in liked]
    neg = [_present_options(sig_to_sol[s], proportional) for s in disliked]
    rules = _greedy_separating_rules([o.name for o in problem.options], pos, neg)
    if not rules:
        return {"available": True, "rules": [], **evidence,
                "note": "Rated solutions don't separate cleanly on option composition — no reliable rule."}
    note = ("These are candidate latent constraints — confirm with the user, then route to "
            "problem_framing.")
    thin = [side for side, n in (("liked", len(liked)), ("disliked", len(disliked)))
            if n < _THIN_EVIDENCE_FLOOR]
    if thin:
        # Self-contained: appending the confident "candidate latent constraints, route to
        # problem_framing" tail would read straight against the point being made here.
        note = (f"Thin evidence: {len(liked)} liked, {len(disliked)} disliked "
                f"({' and '.join(thin)} under {_THIN_EVIDENCE_FLOOR}). A rule induced from "
                "so few rated solutions separates them perfectly by arithmetic, so its "
                "separation and coverage carry no weight yet — treat it as a hypothesis to "
                "confirm (rate more solutions, or corroborate against design_principles "
                "region_bound) before routing anything to problem_framing.")
    return {"available": True, "rules": rules, **evidence, "note": note}


def _greedy_separating_rules(opt_names: list[str], pos: list[set], neg: list[set], max_rules: int = 3) -> list[dict]:
    npos, nneg = len(pos), len(neg)
    candidates = []
    for opt in opt_names:
        for present in (True, False):
            cov_pos = sum(1 for f in pos if (opt in f) == present) / npos
            cov_neg = sum(1 for f in neg if (opt in f) == present) / nneg
            # "liked" rule: holds for liked, fails for disliked (and vice-versa)
            candidates.append((opt, present, "liked", cov_pos, cov_pos - cov_neg))
            candidates.append((opt, present, "disliked", cov_neg, cov_neg - cov_pos))
    # Surface only measurably clean separators — strength and coverage both high.
    candidates = [c for c in candidates if c[4] >= 0.6 and c[3] >= 0.6]
    candidates.sort(key=lambda c: (c[4], c[3]), reverse=True)
    rules, seen = [], set()
    for opt, present, side, cov, sep in candidates:
        if (opt, present) in seen:
            continue
        seen.add((opt, present))
        verb = "includes" if present else "excludes"
        rules.append({"condition": f"solution {verb} {opt}", "separates": side,
                      "separation": round(sep, 2), "coverage": round(cov, 2)})
        if len(rules) >= max_rules:
            break
    return rules


def analyze_composition(problem: Problem, solution_ids: list[int] | None = None,
                        signatures: list[str] | None = None, source: str | None = None,
                        detail: bool = False) -> dict:
    """Mine the solution set: per-option selection rates, co-occurrence, design
    principles, decision-space strategy families, and feedback-driven rules.

    Operates over the active Pareto frontier, or a curated subset when solution_ids /
    signatures are given. The knowledge-discovery pillar, complementing visualization
    (tradeoffs) and uncertainty exploration (scenario_results).
    """
    run = _require_run(problem, None, source)
    solutions = run.solutions
    set_kind = "frontier"
    if solution_ids:
        wanted = set(solution_ids)
        solutions = [s for s in solutions if s.solution_id in wanted]
        set_kind = "curated"
    elif signatures:
        wanted = set(signatures)
        solutions = [s for s in solutions if s.content_signature in wanted]
        set_kind = "curated"
    if not solutions:
        return {"error": "No solutions matched the requested subset."}

    approach = problem.approach.value
    proportional = approach == "proportional" or any(s.allocations for s in solutions)
    option_selection = option_selection_stats(solutions, approach)
    co_occ = _co_occurrence(solutions, option_selection, proportional, top=(9999 if detail else 8))

    clusters = _decision_space_clusters(solutions, problem, proportional)
    clusters_summary = _clusters_summary(clusters)

    result = {
        "scope": {"set": set_kind, "n_solutions": len(solutions), "approach": approach},
        "option_selection": option_selection,
        "co_occurrence": co_occ,
        "design_principles": _design_principles(solutions, option_selection, co_occ, problem, proportional),
        "clusters": clusters,
        **({"clusters_summary": clusters_summary} if clusters_summary else {}),
        "feedback_rules": _feedback_rules(problem, run.solutions, proportional),
    }
    result["visualization"] = _render_composition_viz(result)
    result["frontier_source"] = _frontier_provenance(problem, run, None)
    return result


def _render_composition_viz(result: dict) -> str:
    lines = []
    sel = result.get("option_selection", [])
    if sel:
        lines.append("─── Option selection across the set ───")
        for e in sel[:12]:
            pct = e["selection_pct"]
            bar = "█" * int(round(pct * 20))
            wt = f"  ~{e['mean_weight']}%" if "mean_weight" in e else ""
            lines.append(f"  {e['option'][:20]:20s} {bar:<20} {int(round(pct * 100)):3d}%{wt}")
    dp = result.get("design_principles", [])
    if dp:
        lines.append("")
        lines.append("─── Design principles ───")
        for p in dp[:8]:
            lines.append(f"  • {p['detail']}")
    cl = result.get("clusters", [])
    if cl:
        cs = result.get("clusters_summary") or {}
        share = cs.get("largest_family_share")
        head = f" — largest holds {int(round(share * 100))}%" if share is not None else ""
        lines.append("")
        lines.append(f"─── Strategy families ({len(cl)}){head} ───")
        for c in cl:
            opts = ", ".join(c["defining_options"]) or "(mixed)"
            lines.append(f"  [{c['cluster_id']}] {c['size']} solutions — defining: {opts}")
        if cs.get("note"):
            lines.append(f"  ({cs['note']})")
    fr = result.get("feedback_rules", {})
    if fr.get("rules"):
        lines.append("")
        # The evidence size sits in the header: sep/cov alone read the same off 1 rated
        # solution as off 20, and the first number is what calibrates the rest.
        lines.append(f"─── Learned from your feedback ({fr.get('n_liked', 0)} liked / "
                     f"{fr.get('n_disliked', 0)} disliked) ───")
        for r in fr["rules"]:
            lines.append(f"  • {r['condition']} → {r['separates']} (sep {r['separation']}, cov {r['coverage']})")
        if (min(fr.get("n_liked", 0), fr.get("n_disliked", 0)) < _THIN_EVIDENCE_FLOOR):
            lines.append("  (thin evidence — hypotheses to confirm, not constraints to apply)")
    return "\n".join(lines) if lines else "No composition patterns to show on this set."


_SCENARIO_REMEDY = ("Pick that scenario's plan from its own frontier (explore tradeoffs "
                    "scenario=<name>) or fold its constraints into the base model and re-solve.")


def _round_regret(v: float) -> float:
    """Regret rounded for display, with 1.0 reserved for the exact clamp: a raw 0.9996
    displays as 0.999, so every rendered 100% — including the saturation-note scenario
    naming — means total regret (infeasible or fully dominated), matching the raw
    saturated check."""
    r = round(v, 3)
    return 0.999 if r >= 1.0 and v < 1.0 else r


def scenario_regret(problem: Problem) -> dict:
    """Scenario minimax-regret over the base frontier.

    Scenarios are solved as independent frontiers, so a base solution carries no
    cross-scenario values — we recompute each base solution's value under each scenario
    via optimizer.score_slate. regret = best-achievable-in-scenario minus this solution,
    normalized per objective range. max over (scenario, objective) = the solution's
    max_regret; the minimax pick minimizes it. Scenario-infeasible solutions are flagged
    and assigned worst-case regret (surfaced, not hidden).
    """
    if not problem.scenario_run or not problem.scenario_run.scenario_runs:
        return {"available": False}
    # Stale guard: regret re-scores base solutions per scenario over the CURRENT
    # objectives — a missing objective would silently contribute zero regret.
    if problem.run is not None and problem.run.solutions:
        _consistent(problem, problem.run)
        for _sname, _srun in problem.scenario_run.scenario_runs.items():
            _consistent(problem, _srun, remedy="solve run_scenarios")
    if problem.run is None or not problem.run.solutions:
        return {"available": False}
    # Cross-run guard. Every other scenario read serves its stored run with a stale marker —
    # each is internally coherent. This one is not: regret re-scores the CURRENT base
    # solutions against the STORED scenario frontiers, so when the two were solved against
    # different base models the join is silently incoherent (fresh solution_ids ranked
    # against a previous model's achievable ranges). Decline the join and name the re-solve
    # rather than serve a confidently wrong ranking.
    if scenario_base_mismatch(problem):
        return {
            "available": False,
            "reason": "base_scenario_mismatch",
            "note": ("the base frontier and the scenario frontiers were solved against "
                     "DIFFERENT versions of the model, so regret — which re-scores base "
                     "solutions against each scenario's achievable range — would join two "
                     "models: current solution ids against a previous model's scenario "
                     "ranges. Withheld. Re-run `solve run_scenarios` so both sides share "
                     "one model, then read regret."),
        }
    from .optimizer import make_slate_scorer

    scenarios = {s.name: s for s in (problem.scenario_config.scenarios if problem.scenario_config else [])}
    scen_runs = problem.scenario_run.scenario_runs
    # One scoring context per scenario (overrides applied, matrices built), reused across
    # every base solution — the per-slate rebuild was this loop's dominant cost.
    scorers = {name: make_slate_scorer(problem, scenarios.get(name)) for name in scen_runs}
    objs = problem.objectives
    ideal: dict[tuple, float] = {}
    span: dict[tuple, float] = {}
    for name, run in scen_runs.items():
        for ob in objs:
            vals = [s.objective_values[ob.name] for s in run.solutions if ob.name in s.objective_values]
            if not vals:
                continue
            mx, mn = max(vals), min(vals)
            ideal[(name, ob.name)] = mx if ob.direction.value == "maximize" else mn
            span[(name, ob.name)] = (mx - mn) or 1.0

    base = problem.run.solutions
    cache: dict[tuple, dict] = {}

    def regret_for(s, name, ob):
        ev = cache.get((s.content_signature, name))
        if ev is None or not ev["feasible"]:
            return 1.0  # infeasible → worst-case
        v = ev["values"].get(ob.name)
        key = (name, ob.name)
        if v is None or key not in ideal:
            return 0.0
        gap = (ideal[key] - v) if ob.direction.value == "maximize" else (v - ideal[key])
        # Clamp to [0,1]: regret is the fraction of the scenario's achievable range given
        # up; a solution worse than the scenario's own frontier caps at 1.0 (as bad as
        # infeasible), keeping the scale interpretable.
        return min(1.0, max(0.0, gap) / span[key])

    per_solution = []
    raw_by_sig: dict[str, dict] = {}
    for s in base:
        for name in scen_runs:
            ck = (s.content_signature, name)
            if ck not in cache:
                cache[ck] = scorers[name](s.selected_options, s.allocations)
        raw = {name: max((regret_for(s, name, ob) for ob in objs), default=0.0) for name in scen_runs}
        raw_by_sig[s.content_signature] = raw
        per_solution.append({
            "solution_id": s.solution_id, "content_signature": s.content_signature,
            "by_scenario": {name: _round_regret(v) for name, v in raw.items()},
            "feasible_in_all": all(cache[(s.content_signature, name)]["feasible"] for name in scen_runs),
        })

    # Per-scenario survivor counts, derived from the same feasibility cache the rows use
    # (one source of truth). A total-wipeout scenario (zero survivors) makes regret
    # uniformly 1.0 there — folding it into the ranking would saturate the whole metric
    # and hide real differences on the surviving scenarios, so wipeouts are excluded from
    # the ranking (still visible in by_scenario, named in wipeout_note). When EVERY
    # scenario is a wipeout nothing is "excluded" — the all-saturated presentation owns it.
    survivors = {name: sum(1 for s in base if cache[(s.content_signature, name)]["feasible"])
                 for name in scen_runs}
    wipeouts = [name for name in scen_runs if survivors[name] == 0]
    ranked = [name for name in scen_runs if name not in wipeouts]
    excluded = wipeouts if ranked else []
    if not ranked:
        ranked = list(scen_runs)
    raw_max: dict[str, float] = {}
    for row in per_solution:
        raw = raw_by_sig[row["content_signature"]]
        vals = [raw[name] for name in ranked]
        raw_max[row["content_signature"]] = max(vals, default=0.0)
        row["max_regret"] = _round_regret(raw_max[row["content_signature"]])
        row["mean_regret"] = _round_regret(sum(vals) / len(vals)) if vals else None
        row["feasible_in_ranked"] = all(cache[(row["content_signature"], name)]["feasible"]
                                        for name in ranked)
    per_solution.sort(key=lambda x: raw_max[x["content_signature"]])

    per_objective = {}
    for ob in objs:
        best_sid, best_val = None, None
        for s in base:
            worst = max((regret_for(s, name, ob) for name in ranked), default=0.0)
            if best_val is None or worst < best_val:
                best_val, best_sid = worst, s.solution_id
        if best_sid is not None:
            per_objective[ob.name] = {"min_max_regret": _round_regret(best_val), "achieved_by_solution_id": best_sid}

    # Curated lens: the user's shortlisted picks re-scored under each scenario with the
    # SAME scorers, normalization, and ranked-scenario set as the base rows, so the two
    # minimax numbers compare directly. Curated pins carry selected_options/allocations
    # (everything score_slate needs) and are content_signature-keyed, so an
    # exact-overlay finalist scores here even though it never appears in the base run.
    # Absent when nothing is curated — zero noise.
    curated_block = None
    if problem.curated_solutions:
        curated_rows = []
        raw_max_cur: dict[str, float] = {}
        for cs in problem.curated_solutions:
            for name in scen_runs:
                ck = (cs.content_signature, name)
                if ck not in cache:
                    cache[ck] = scorers[name](cs.selected_options, cs.allocations)
            raw = {name: max((regret_for(cs, name, ob) for ob in objs), default=0.0)
                   for name in scen_runs}
            vals = [raw[name] for name in ranked]
            raw_max_cur[cs.content_signature] = max(vals, default=0.0)
            curated_rows.append({
                "content_signature": cs.content_signature,
                "custom_name": cs.custom_name,
                "by_scenario": {name: _round_regret(v) for name, v in raw.items()},
                "max_regret": _round_regret(raw_max_cur[cs.content_signature]),
                "mean_regret": _round_regret(sum(vals) / len(vals)) if vals else None,
                "feasible_in_all": all(cache[(cs.content_signature, name)]["feasible"]
                                       for name in scen_runs),
                "feasible_in_ranked": all(cache[(cs.content_signature, name)]["feasible"]
                                          for name in ranked),
            })
        curated_rows.sort(key=lambda r: raw_max_cur[r["content_signature"]])
        cur_saturated = bool(curated_rows) and raw_max_cur[curated_rows[0]["content_signature"]] >= 1.0
        curated_block = {
            "rows": curated_rows,
            "minimax_choice": ({"content_signature": curated_rows[0]["content_signature"],
                                "custom_name": curated_rows[0]["custom_name"],
                                "max_regret": curated_rows[0]["max_regret"]}
                               if curated_rows and not cur_saturated else None),
            "note": ("the curated picks re-scored under each scenario — same normalization "
                     "and ranked-scenario set as per_solution, so max_regret here compares "
                     "directly against minimax_choice."
                     + (" Saturated: every curated pick hits max regret over the ranked "
                        "scenarios, so no curated minimax is nominated." if cur_saturated else "")),
        }

    # per_solution is sorted ascending, so the metric is saturated exactly when the BEST
    # solution already hits total regret — every base solution is infeasible or fully
    # dominated in some RANKED scenario, and "minimizes worst-case regret" would pick among
    # an all-1.0 tie. The check runs on unrounded values, and round_regret keeps the
    # displayed values consistent with it: 1.0 anywhere in the payload means the clamp.
    saturated = bool(per_solution) and raw_max[per_solution[0]["content_signature"]] >= 1.0
    minimax = per_solution[0] if per_solution and not saturated else None
    scope_note = (
        " max/mean/minimax and the per-objective regrets are computed over the ranked "
        "scenarios only — wipeout scenarios are excluded (see wipeout_note)." if excluded else "")
    result = {
        "available": True,
        "method": "scenario_minimax",
        "normalization": "per_objective_range",
        "per_objective": per_objective,
        "per_solution": per_solution[:20],
        "per_solution_total": len(per_solution),
        "survivors_by_scenario": survivors,
        "minimax_choice": ({"solution_id": minimax["solution_id"],
                            "content_signature": minimax["content_signature"],
                            "max_regret": minimax["max_regret"]} if minimax else None),
        **({"curated": curated_block} if curated_block else {}),
        "note": ("regret = best-achievable-in-scenario minus this solution re-evaluated there, "
                 "normalized per objective range; minimax_choice minimizes worst-case regret. "
                 "per_solution lists the lowest-max-regret solutions "
                 "(see per_solution_total for the full count)." + scope_note),
    }
    if saturated:
        sat_scens = [name for name in ranked
                     if all(ps["by_scenario"].get(name) == 1.0 for ps in per_solution)]
        result["saturated"] = True
        result["saturation_note"] = (
            "minimax not informative — every base-frontier solution hits max regret "
            "(infeasible or fully dominated) under: "
            f"{', '.join(sat_scens) or 'the ranked scenarios (the saturating scenario varies by plan)'}. "
            "minimax_choice is omitted. " + _SCENARIO_REMEDY
        )
    if excluded:
        result["wipeout_scenarios"] = excluded
        result["wipeout_note"] = (
            f"no base-frontier solution is feasible under: {', '.join(excluded)} — the base "
            "frontier cannot hedge that scenario, so it is excluded from the minimax ranking "
            "(its regret is uniformly total; see by_scenario). " + _SCENARIO_REMEDY
        )
    return result


def _compute_reference_analysis(solution_obj_values: dict, reference_points: list, objectives: list) -> list[dict]:
    """Compute distance from a solution to each reference point, per objective."""
    if not reference_points:
        return []

    obj_dirs = {o.name: o.direction.value for o in objectives}
    results = []
    for rp in reference_points:
        per_obj = {}
        for obj_name, ref_val in rp.objective_values.items():
            if obj_name not in solution_obj_values:
                continue
            sol_val = solution_obj_values[obj_name]
            diff = sol_val - ref_val
            direction = obj_dirs.get(obj_name, "maximize")
            # Positive = better than reference for maximize, worse for minimize
            if direction == "minimize":
                diff = -diff
            pct = round(diff / abs(ref_val) * 100, 1) if ref_val != 0 else 0.0
            per_obj[obj_name] = {
                "solution_value": sol_val,
                "reference_value": ref_val,
                "difference": round(diff, 4),
                "percent_vs_reference": pct,
                "better": diff > 0,
            }
        results.append({
            "reference": rp.name or rp.type,
            "type": rp.type,
            "objectives": per_obj,
        })
    return results


def compare_runs(problem: Problem, run_ids: list[str]) -> dict:
    """Compare two or more runs: constraint diffs, frontier diffs, option coverage."""
    all_runs = {r.run_id: r for r in problem.runs}
    if problem.run:
        all_runs[problem.run.run_id] = problem.run
    if problem.exact_run:
        all_runs[problem.exact_run.run_id] = problem.exact_run

    selected_runs = []
    for rid in run_ids:
        if rid not in all_runs:
            raise ValueError(f"Run {rid} not found. Available: {list(all_runs.keys())}")
        selected_runs.append(all_runs[rid])

    obj_names = [o.name for o in problem.objectives]

    # Criteria diff: compare constraint snapshots between runs
    criteria_diffs = []
    for i in range(1, len(selected_runs)):
        prev_snap = selected_runs[i - 1].constraints_snapshot
        curr_snap = selected_runs[i].constraints_snapshot
        prev_set = {_constraint_key(c) for c in prev_snap}
        curr_set = {_constraint_key(c) for c in curr_snap}
        added = [c for c in curr_snap if _constraint_key(c) not in prev_set]
        removed = [c for c in prev_snap if _constraint_key(c) not in curr_set]
        criteria_diffs.append({
            "from_run": selected_runs[i - 1].run_id,
            "to_run": selected_runs[i].run_id,
            "added": added,
            "removed": removed,
        })

    # Frontier diff: solution count and objective range changes
    frontier_diffs = []
    for run in selected_runs:
        ranges = {}
        for name in obj_names:
            vals = [s.objective_values.get(name, 0) for s in run.solutions]
            if vals:
                ranges[name] = {"min": min(vals), "max": max(vals)}
            else:
                ranges[name] = {"min": None, "max": None}
        frontier_diffs.append({
            "run_id": run.run_id,
            "mode": run.mode.value,
            "solution_count": len(run.solutions),
            "objective_ranges": ranges,
            "created_at": run.created_at.isoformat(),
        })

    # Option coverage diff: how many solutions include each option per run
    option_coverage = {}
    opt_names = [o.name for o in problem.options]
    for run in selected_runs:
        coverage = {}
        for opt in opt_names:
            coverage[opt] = sum(1 for s in run.solutions if opt in s.selected_options)
        option_coverage[run.run_id] = coverage

    # One row per option PER RUN, so this block scales with options × runs — the widest
    # part of the response at portfolio scale. Keep the options that actually moved (the
    # question compare_runs answers) and summarize the rest, the same elide-and-say-so
    # treatment solve's own option_coverage gets (metrics._MAX_COVERAGE_RETURNED).
    option_coverage, coverage_elided = _elide_coverage_diff(option_coverage, opt_names)

    return {
        "runs_compared": [r.run_id for r in selected_runs],
        "criteria_diffs": criteria_diffs,
        "frontier_diffs": frontier_diffs,
        "option_coverage": option_coverage,
        **({"option_coverage_elided": coverage_elided} if coverage_elided else {}),
    }


def _elide_coverage_diff(option_coverage: dict[str, dict[str, int]],
                         opt_names: list[str]) -> tuple[dict, dict | None]:
    """Keep the top options by SPREAD in selection count across the compared runs.

    Spread (max − min) rather than raw count is what the action is for: an option holding
    steady at 40 plans in every run says nothing about the edit, while one that moved from
    0 to 12 is the finding. Generalizes past a pair, as `run_ids` does — with two runs the
    spread IS the delta. Returns ``(coverage, elided-summary or None)``."""
    from engine.metrics import _MAX_COVERAGE_RETURNED

    if len(opt_names) <= _MAX_COVERAGE_RETURNED:
        return option_coverage, None

    def delta(opt: str) -> int:
        counts = [cov.get(opt, 0) for cov in option_coverage.values()]
        return max(counts) - min(counts)

    ranked = sorted(opt_names, key=lambda o: (-delta(o), o))
    kept = ranked[:_MAX_COVERAGE_RETURNED]
    tail_max_delta = delta(ranked[_MAX_COVERAGE_RETURNED])
    trimmed = {rid: {o: cov[o] for o in kept if o in cov} for rid, cov in option_coverage.items()}
    return trimmed, {
        "shown": len(kept),
        "total_options": len(opt_names),
        "ranked_by": "spread in selection count across the compared runs (max − min)",
        "tail_max_delta": tail_max_delta,
        "note": ("absence below the head is elision, NOT a coverage of zero — "
                 + ("every elided option holds the same count in all compared runs. "
                    if tail_max_delta == 0 else
                    f"an elided option's count spans at most {tail_max_delta} plan(s) "
                    "across the compared runs. ")
                 + "Full per-option selection rates for the CURRENT frontier: "
                   "explore composition"),
    }


def _shared_box_masks(parts: list[np.ndarray], seed: int = 42):
    """Joint-normalize stacked minimize-space fronts into one shared box and return a
    ``mask(front)`` function over ONE Monte-Carlo draw — the same sampler and conventions
    as ``optimizer._approx_hypervolume``, so a front's normalized hypervolume is exactly
    ``mask(front).mean()`` and unions compose as mask OR (per-union cost O(|front| × samples),
    not O(|everything| × samples)). Shared samples make ``union ≥ part`` exact, never a
    sampling artifact. Returns ``None`` when the joint box is degenerate (a flat axis),
    where hypervolume is undefined."""
    P = np.vstack(parts)
    f_min = P.min(axis=0)
    spread = P.max(axis=0) - f_min
    if not np.all(spread > 0):
        return None
    m = P.shape[1]
    ref = np.ones(m) * 1.1
    samples = np.random.default_rng(seed).uniform(0, ref, size=(10000, m))

    def mask(F: np.ndarray) -> np.ndarray:
        out = np.zeros(len(samples), dtype=bool)
        for i in range(0, len(F), 256):  # chunk the broadcast: bounded temp at large fronts
            Fn = (F[i:i + 256] - f_min) / spread
            out |= ((samples[:, None, :] >= Fn[None, :, :]).all(axis=2)).any(axis=1)
        return out

    return f_min, spread, mask


def _nondominated_idx_min(E: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated subset in minimize-space — the vectorized form of
    filtering by ``_dominates_min`` (same ±1e-9 strict-dominance epsilon)."""
    if len(E) == 0:
        return np.array([], dtype=int)
    le = (E[None, :, :] <= E[:, None, :] + 1e-9).all(axis=2)  # le[i, j]: E[j] no worse than E[i]
    lt = (E[None, :, :] < E[:, None, :] - 1e-9).any(axis=2)   # lt[i, j]: E[j] strictly better somewhere
    return np.where(~(le & lt).any(axis=1))[0]


def _coverage_gain(N: np.ndarray, E: np.ndarray, seed: int = 42) -> dict | None:
    """Hypervolume the exact overlay reclaims over the NSGA frontier alone — the *magnitude*
    companion to the dominance *count*, and the coverage value that grows with problem size.

    Both fronts arrive in minimize-space. ``_shared_box_masks`` normalizes the *combined* set
    into one shared box so a single reference point is fair to both (hypervolume is
    reference-point dependent) and draws one shared sample set, so ``combined ≥ nsga`` exactly —
    the reclaimed volume is never a sampling artifact. Returns ``None`` when the combined front
    is degenerate (a flat axis), where coverage is undefined. Volumes are normalized to the box
    (the engine's ``hypervolume_normalized`` convention), so they read on the same 0–1 scale as
    a run's quality.
    """
    ctx = _shared_box_masks([N, E], seed=seed)
    if ctx is None:
        return None
    _, _, mask = ctx
    m_nsga = mask(N)
    hv_nsga = float(m_nsga.mean())
    hv_comb = float((m_nsga | mask(E)).mean())
    reclaimed = max(0.0, hv_comb - hv_nsga)  # ≥ 0 by construction; the clamp only absorbs FP noise
    return {
        "nsga_hypervolume": round(hv_nsga, 4),
        "combined_hypervolume": round(hv_comb, 4),
        "exact_reclaims": round(reclaimed, 4),
        "reclaimed_fraction": round(reclaimed / hv_comb, 4) if hv_comb > 0 else 0.0,
    }


_COMPLETENESS_NOISE = 0.005  # normalized-HV tolerance for "complete": the shared-seed MC trick
                             # makes combined ≥ exact exact (never a sampling artifact), so this
                             # floor only absorbs rounding/display-precision noise. Stated openly
                             # on the payload so the verdict is auditable.
_WITNESS_RTOL = 1e-6  # per-objective coverage tolerance, relative to the joint front's spread —
                      # objective values are stored rounded and exact backends re-solve within
                      # solver tolerance, so an absolute epsilon would mint phantom witnesses on
                      # large-magnitude objectives (and the fill would chase them forever).


def _signed_matrix(problem: Problem, run: Run) -> tuple[np.ndarray, np.ndarray]:
    """(natural, minimize-space) objective-value matrices for a run — the one construction
    certify, the completeness analysis, and the fill work-list all share, so no consumer can
    drift on sign convention or missing-value fill."""
    objs = problem.objectives
    names = [o.name for o in objs]
    sign = np.array([-1.0 if o.direction == Direction.maximize else 1.0 for o in objs])
    nat = np.array([[s.objective_values.get(n, 0.0) for n in names] for s in run.solutions],
                   dtype=float).reshape(len(run.solutions), len(names))
    return nat, nat * sign


def _completeness_analysis(problem: Problem, nsga_run: Run, exact_run: Run) -> dict:
    """Shared gap detection for the certificate's ``completeness`` block AND the targeted
    fill (``optimizer.fill_gaps``) — ONE criterion decides both the verdict and the fill
    work-list, so a "complete" certificate and a non-no-op fill can never disagree.

    Witnesses are NSGA points no exact point covers (equal-or-better on every objective,
    within a per-objective tolerance relative to the joint spread); ``under`` is True only
    when witnesses exist AND the hypervolume they reclaim clears ``_COMPLETENESS_NOISE``
    (when the joint box is degenerate, hypervolume is undefined and witnesses alone decide).
    """
    if not nsga_run.solutions or not exact_run.solutions:
        witness_idx = list(range(len(nsga_run.solutions)))
        return {"witness_idx": witness_idx, "under": bool(witness_idx), "reclaims": None,
                "hv_exact": None, "hv_comb": None, "nat_N": None, "N": None, "Eref": None,
                "box": None}
    nat_N, N = _signed_matrix(problem, nsga_run)
    _, E = _signed_matrix(problem, exact_run)
    Eref = E[_nondominated_idx_min(E)]
    joint = np.vstack([N, Eref])
    tol = (joint.max(axis=0) - joint.min(axis=0)) * _WITNESS_RTOL + 1e-9
    covered = ((Eref[None, :, :] <= N[:, None, :] + tol).all(axis=2)).any(axis=1)
    witness_idx = list(np.where(~covered)[0])
    box = _shared_box_masks([N, Eref])
    if box is None:
        return {"witness_idx": witness_idx, "under": bool(witness_idx), "reclaims": None,
                "hv_exact": None, "hv_comb": None, "nat_N": nat_N, "N": N, "Eref": Eref,
                "box": None}
    f_min, spread_box, mask = box
    mask_e = mask(Eref)
    hv_exact = float(mask_e.mean())
    hv_comb = float((mask_e | mask(N)).mean())
    reclaims = max(0.0, hv_comb - hv_exact)
    return {"witness_idx": witness_idx,
            "under": bool(witness_idx) and reclaims > _COMPLETENESS_NOISE,
            "reclaims": reclaims, "hv_exact": hv_exact, "hv_comb": hv_comb,
            "nat_N": nat_N, "N": N, "Eref": Eref, "box": (f_min, spread_box)}


def gap_witness_solutions(problem: Problem, nsga_run: Run, exact_run: Run) -> list:
    """The NSGA solutions inside detected coverage gaps of the exact overlay.

    The read side of ``completeness`` and the work-list for the targeted fill
    (``solve scope="fill_gaps"`` → ``optimizer.fill_gaps``): re-solving exactly these
    points' scalarizations closes the gaps the certificate names. Noise-gated on the SAME
    criterion as the certificate's verdict — a "complete" overlay yields an empty
    work-list, so the fill's documented no-op contract can never disagree with the
    certificate. Pure detection — no solver touched."""
    a = _completeness_analysis(problem, nsga_run, exact_run)
    if not a["under"]:
        return []
    return [nsga_run.solutions[i] for i in a["witness_idx"]]


def _completeness_block(problem: Problem, nsga_run: Run, exact_run: Run) -> dict | None:
    """The mirror of ``_coverage_gain``: the exact overlay audited for completeness by NSGA.

    ``coverage`` measures what exact reclaims over the heuristic; this measures the inverse —
    hypervolume the heuristic run already covers that the exact scalarization sweep never
    sampled (supported-points-only gaps on non-convex shapes). Same joint normalization and
    shared-seed Monte-Carlo, roles swapped. ``gap_regions`` localizes the reclaim: witnesses
    clustered by nearest exact point in the normalized box — boxes from nondominated
    witnesses, not ordered chords, so any objective count works — each region naming its
    witness solution ids (self-certifying), largest first. Returns ``None`` when the joint
    front is degenerate (hypervolume undefined)."""
    a = _completeness_analysis(problem, nsga_run, exact_run)
    if a["reclaims"] is None:
        return None
    names = [o.name for o in problem.objectives]
    under = a["under"]
    block: dict = {
        "exact_hypervolume": round(a["hv_exact"], 4),
        "combined_hypervolume": round(a["hv_comb"], 4),
        "heuristic_reclaims": round(a["reclaims"], 4),
        "reclaimed_fraction_over_exact": round(a["reclaims"] / a["hv_comb"], 4) if a["hv_comb"] > 0 else 0.0,
        "noise_floor": _COMPLETENESS_NOISE,
        "verdict": "under_covered" if under else "complete",
        "mirror": "coverage",
        "note": ("the heuristic holds hypervolume the exact sweep never sampled — fill the "
                 "named regions with solve scope='fill_gaps'" if under else
                 "the exact overlay covers everything the heuristic reached (reclaim at or "
                 "below the noise floor)"),
    }
    if not under:
        return block

    # Localize: cluster witnesses by nearest exact point in the shared normalized box,
    # largest region first. Witness count ranks the regions — a per-region hypervolume
    # share would cost a Monte-Carlo pass per cluster for a number that is only
    # indicative anyway (regions overlap in dominated volume), and the fill consumes
    # ALL witnesses regardless.
    nat_N, N, Eref = a["nat_N"], a["N"], a["Eref"]
    f_min, spread = a["box"]
    Nn = (N - f_min) / spread
    En = (Eref - f_min) / spread
    clusters: dict[int, list[int]] = {}
    for i in a["witness_idx"]:
        anchor = int(np.argmin(((En - Nn[i]) ** 2).sum(axis=1)))
        clusters.setdefault(anchor, []).append(i)
    regions = []
    for _, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        bbox = {names[j]: [round(float(nat_N[members, j].min()), 4),
                           round(float(nat_N[members, j].max()), 4)]
                for j in range(len(names))}
        ids = [nsga_run.solutions[i].solution_id for i in members]
        regions.append({
            "witness_solution_ids": ids[:10],
            "size": len(members),
            "bounding_box": bbox,
        })
    if len(regions) > 8:  # payload compactness; the fill consumes ALL witnesses regardless
        block["regions_truncated"] = len(regions) - 8
        regions = regions[:8]
    block["gap_regions"] = regions
    return block


def _objective_bound_dips(problem: Problem, run: Run) -> list[dict]:
    """Points whose *recorded* objective values sit a hair outside a stated ``objective_bound``.

    Whole-percent rounding of a continuous (proportional) optimum can dip a point just
    under a floor the LP/QP itself honors — name those points instead of letting the
    certified table silently contradict the model. Display artifact, not an infeasible
    plan; binary selections are integer by construction and never dip.

    Reads the value the solution **stores**, which is what the certified table prints. Its
    sibling ``optimizer.verify_run`` re-derives the value from the stored *allocations* and so
    catches the complementary case — a recorded value that honors the bound while the recorded
    plan, re-aggregated, does not. Both are rounding effects on the same proportional path;
    keeping them separate keeps each one's claim exact."""
    if problem.approach != Approach.proportional:
        return []
    dips = []
    for c in problem.constraints or []:
        if getattr(c, "type", "") != "objective_bound":
            continue
        op = getattr(c.operator, "value", c.operator)
        bound = float(c.value)
        for s in run.solutions:
            v = s.objective_values.get(c.objective)
            if v is None:
                continue
            if (v < bound - 1e-9) if op == "min" else (v > bound + 1e-9):
                dips.append({"solution_id": s.solution_id, "objective": c.objective,
                             "bound": f"{op} {bound}", "value": v})
    return dips


def _frontier_resolution_block(problem: Problem, front_sols: list) -> dict | None:
    """Resolution bound on the TRUE frontier, from the exact points' own duals — the
    sandwich read of the overlay.

    ``coverage``/``completeness`` audit the two runs against *each other*; this block bounds
    the reported frontier against the frontier itself. On a 2-objective continuous LP/QP
    overlay the tradeoff curve is convex in minimize-space, so it is sandwiched between two
    piecewise-linear envelopes built from data already on board (no extra solves):

    - **above** — the chords between adjacent certified points (convexity: the curve lies
      below its chords);
    - **below** — each point's supporting tangent, whose slope is the ε-floor's own
      ``shadow_price`` (envelope theorem: the dual IS the local frontier slope), plus a
      monotonicity floor (between adjacent points the curve can never beat the better
      endpoint) that holds even where a dual is absent — an unpriced anchor or a
      degenerate corner loosens the bound, never voids it.

    The widest chord-to-envelope gap is the certified resolution: between the anchors, no
    feasible plan improves the primary over straight-line interpolation of the certified
    points by more than that gap. The frontier-side sibling of ``explore audit``'s
    whole-region guarantee — a claim about every feasible plan at every swept level in the
    span, not about the sampled points. Present as ``null`` (matching ``completeness``)
    when the shape can't carry the claim: 3+ objectives, an integer overlay (no duals), an
    ambiguous ε-primary, or objectives not in tension."""
    objs = problem.objectives
    if len(objs) != 2 or len(front_sols) < 2:
        return None

    # The ε-constraint primary, named by the duals themselves — one consistent answer or
    # nothing (an explainability payload never guesses; same rule as `explore sensitivity`).
    sol_primary = [_optimized_objective(problem, s.sensitivity) if s.sensitivity else None
                   for s in front_sols]
    primaries = {p for p in sol_primary if p}
    if len(primaries) != 1:
        return None
    primary_name = primaries.pop()
    swept = next(o for o in objs if o.name != primary_name)
    sign = {o.name: -1.0 if o.direction == Direction.maximize else 1.0 for o in objs}

    # Minimize-space points (u = swept, v = primary) with the tangent slope where priced.
    # Cost-sense dual (tightening the floor hurts) → dv/du = −|λ|; same lever read as
    # `_pair_dual_rates`, but per-point tolerant (the monotonicity floor covers gaps).
    pts = []
    for s, prim in zip(front_sols, sol_primary):
        lam = _floor_dual(s.sensitivity, swept.name) if prim == primary_name else None
        pts.append((sign[swept.name] * s.objective_values.get(swept.name, 0.0),
                    sign[primary_name] * s.objective_values.get(primary_name, 0.0),
                    None if lam is None else -lam, s.solution_id))
    pts.sort(key=lambda t: (t[0], t[1]))
    dd = [p for k, p in enumerate(pts) if k == 0 or p[0] - pts[k - 1][0] > 1e-9]
    if len(dd) < 2:
        return None
    v_spread = max(p[1] for p in dd) - min(p[1] for p in dd)
    if v_spread <= 1e-12:
        return None  # objectives not in tension — no resolution to certify

    # Per segment: gap = max over [u1,u2] of chord(u) − envelope(u). Every line is affine,
    # so chord − max(lines) is concave piecewise-linear — its max sits at an endpoint or a
    # pairwise intersection of envelope lines. Clamped ≥ 0: whole-percent rounding and
    # solver tolerance can nudge a tangent a hair above the chord.
    worst = None
    tightened = 0
    for k in range(len(dd) - 1):
        (u1, v1, m1, _), (u2, v2, m2, _) = dd[k], dd[k + 1]
        lines = [(0.0, v2)]  # monotonicity floor: v(u) ≥ v2 for u ≤ u2, dual-free
        if m1 is not None:
            lines.append((m1, v1 - m1 * u1))
        if m2 is not None:
            lines.append((m2, v2 - m2 * u2))
        if len(lines) > 1:
            tightened += 1
        chord_m = (v2 - v1) / (u2 - u1)
        chord_b = v1 - chord_m * u1
        cands = [u1, u2]
        for (ma, ba), (mb, bb) in combinations(lines, 2):
            if abs(ma - mb) > 1e-15:
                ux = (bb - ba) / (ma - mb)
                if u1 < ux < u2:
                    cands.append(ux)
        gap = max(0.0, max((chord_m * u + chord_b) - max(m * u + b for m, b in lines)
                           for u in cands))
        if worst is None or gap > worst[0]:
            worst = (gap, k)

    gap, k = worst
    s_sign = sign[swept.name]
    seg = sorted([s_sign * dd[k][0], s_sign * dd[k + 1][0]])
    span = sorted([s_sign * dd[0][0], s_sign * dd[-1][0]])
    frac = gap / v_spread
    return {
        "basis": "solver_exact_duals",
        "points_used": len(dd),
        "segments_dual_tightened": tightened,  # of points_used − 1 segments
        "span": {swept.name: [round(span[0], 4), round(span[1], 4)]},
        "max_gap": {primary_name: round(gap, 6),
                    "fraction_of_frontier_range": round(frac, 4)},
        "widest_segment": {swept.name: [round(seg[0], 4), round(seg[1], 4)],
                           "solution_ids": [dd[k][3], dd[k + 1][3]]},
        "claim": (f"between {swept.name} {span[0]:g} and {span[1]:g}, no feasible plan "
                  f"improves {primary_name} over straight-line interpolation of the "
                  f"certified points by more than {gap:.4g} ({frac:.1%} of the frontier's "
                  f"{primary_name} span)"),
        "note": ("bounds the true continuous frontier, not just the sampled points — chords "
                 "above (convexity) and floor-dual tangents below (envelope theorem) are "
                 "model-class facts of the continuous LP/QP, independent of how the sampled "
                 "frontier_shape heuristic reads; a monotonicity floor covers segments "
                 "without a dual. Exact to solver tolerance and the whole-percent rounding "
                 "of reported values; the widest gap is named in widest_segment — a targeted "
                 "exact solve there tightens it."),
    }


def _per_pick_verdicts(problem: Problem, names: list[str], sign: np.ndarray, N: np.ndarray,
                       E: np.ndarray, exact_front: np.ndarray, exact_run: Run) -> dict:
    """The curated finalists judged one by one against the certified frontier, keyed by
    ``content_signature`` (the identifier curation already uses).

    ``dominance_audit`` counts how many NSGA points exact beats — a frontier-level fact. The
    user's question at CERTIFY is narrower and personal (*are the finalists **I** curated
    optimal?*), and answering it from the aggregate meant reading the persisted exact-run JSON
    off disk. Same criterion as the audit (``_dominates_min`` against the cleaned exact front),
    applied per pick, so the two readings can never disagree.

    ``gap`` is the distance to the nearest certified point that beats the pick on every
    objective: the worst per-objective shortfall, normalized by that objective's spread across
    both fronts (the box the certificate already reasons in), so it reads on the same 0–1 scale
    as ``regret``. An ``optimal`` pick has no certifying point above it — nothing to close, so
    its gap is 0. A pin whose stored values predate a current objective can't be placed in this
    objective space at all: it is reported ``inconclusive`` with the objectives it lacks, rather
    than scored against a fabricated 0.0. Absent (not empty) when nothing is curated, matching
    ``regret.curated``.

    **A pick is judged at its slate's value under the CURRENT scores**, not at the values
    stored when it was pinned. The certified frontier is built from today's scores; a pin
    carries yesterday's. Judging the stored vector against a fresh frontier certified an
    obsolete pick as ``optimal, gap 0.0`` after a score edit — its stale value sat above the
    new ceiling, so nothing on the honest frontier could dominate it (observed live:
    budget_allocation, ROI 26.2 stored vs 24.0 under revised scores, new ceiling 24.2). So
    each pick's option set is re-scored via the optimizer's own ``score_slate`` — the same
    machinery ``regret.curated`` trusts — and the dominance check runs in one score regime.
    When the recomputed values differ from the stored ones the entry says so
    (``stored_values_stale`` + ``values_now``); a slate no longer feasible under the current
    constraints is reported ``infeasible`` rather than ranked (an invalid plan has no place
    on any frontier). A pin with no recorded slate falls back to its stored values — all a
    hand-built pin has.
    """
    if not problem.curated_solutions:
        return {}
    joint = np.vstack([N, E[exact_front]])
    spread = joint.max(axis=0) - joint.min(axis=0)
    spread = np.where(spread > 1e-12, spread, 1.0)  # a flat axis can carry no shortfall

    def _sig(s) -> str:
        return s.content_signature or _content_signature(s.selected_options, s.allocations)

    scorer = None  # built once, only if some pick carries a slate

    def _revalue(cs):
        """(values under current scores, feasible) — or (None, None) when re-scoring is
        impossible: a slate-less hand-built pin, or a model the scorer itself cannot evaluate
        (e.g. a quadratic objective missing its matrix — solve validate flags that upstream).
        Falling back to stored values is the honest floor; it is exactly what the block did
        for every pin before the regime check existed."""
        nonlocal scorer
        if not cs.selected_options and not cs.allocations:
            return None, None
        try:
            if scorer is None:
                from .optimizer import make_slate_scorer
                scorer = make_slate_scorer(problem, None)
            r = scorer(cs.selected_options, cs.allocations)
        except Exception:
            return None, None
        return r["values"], r["feasible"]

    per_pick = {}
    for cs in problem.curated_solutions:
        now, feasible = _revalue(cs)
        source = now if now is not None else cs.objective_values
        missing = [n for n in names if n not in source]
        if missing:
            per_pick[cs.content_signature] = {
                "verdict": "inconclusive", "custom_name": cs.custom_name,
                "missing_objectives": missing,   # self-certifying: names why it can't be judged
            }
            continue
        # Staleness is measured, not assumed: material drift between stored and current
        # values means a score edit happened after the pin. The threshold must clear the
        # whole-percent rounding floor — on proportional problems the search records the
        # continuous objective while score_slate re-derives it from the integer
        # allocations, a ~0.4%-of-spread artifact that made EVERY NSGA-sourced pin read
        # stale seconds after pinning (observed live at 1e-4). 0.5% of the joint spread
        # sits above that floor and far below any decision-bearing edit (the live case
        # moved a pin 60% of spread); the verdict itself always uses values_now, so an
        # edit small enough to hide under the flag still gets judged honestly.
        stale = now is not None and any(
            abs(now[n] - cs.objective_values[n]) > max(1e-9, 5e-3 * spread[i])
            for i, n in enumerate(names) if n in cs.objective_values)
        stale_fields = ({"stored_values_stale": True,
                         "values_now": {n: round(now[n], 4) for n in names}} if stale else {})
        if now is not None and not feasible:
            per_pick[cs.content_signature] = {
                "verdict": "infeasible", "custom_name": cs.custom_name, **stale_fields,
                "note": "this slate violates the current constraints — re-curate from the "
                        "current frontier rather than ranking an invalid plan",
            }
            continue
        pick = np.array([source[n] for n in names], dtype=float) * sign
        nearest = None
        for k in exact_front:
            if not _dominates_min(E[k], pick):
                continue
            d = float(np.max(np.maximum(pick - E[k], 0.0) / spread))
            if nearest is None or d < nearest[0]:
                nearest = (d, int(k))
        if nearest is None:
            per_pick[cs.content_signature] = {
                "verdict": "optimal", "custom_name": cs.custom_name, "gap": 0.0, **stale_fields}
        else:
            gap, k = nearest
            per_pick[cs.content_signature] = {
                "verdict": "dominated", "custom_name": cs.custom_name, "gap": round(gap, 4),
                "dominated_by": _sig(exact_run.solutions[k]), **stale_fields,
            }
    return per_pick


def certify_against_exact(problem: Problem, nsga_run: Run, exact_run: Run) -> dict:
    """Audit an approximate (NSGA) frontier against an exact-solver frontier — the
    explore-then-certify workflow made measurable. **Solver-agnostic:** the exact run can come
    from either first-class exact backend (HiGHS on CPU, cuOpt on GPU); the engine that produced
    it is reported, never assumed.

    An exact inner solve is optimal for its scalarization (to a 0.1% gap; a certified zero gap with
    ``exact=True``), so overlaying exact points on the heuristic frontier can only **confirm or
    improve** it, never worsen it — exact acts as an *auditor*. This certificate makes that concrete
    on four axes:

    - **Dominance audit** — how many NSGA points the exact frontier strictly dominates: heuristic
      slack, points presented as efficient that an exact solver beats at their own cost.
    - **Per-pick verdicts** — the same dominance criterion applied to the *curated* finalists
      (``per_pick``), so the phase's own question — are the picks the user kept optimal? — is
      answered on the certificate instead of in the persisted exact-run JSON.
    - **Coverage** — the hypervolume the exact overlay reclaims over the NSGA frontier alone: the
      *magnitude* behind the dominance count, and the coverage value that grows with problem size
      (near-zero on a small instance the heuristic already covers — exact confirms, doesn't expand).
    - **The invariant** — NSGA should dominate *no* exact point. A violation isn't a heuristic win
      (the exact point is optimal for its scalarization); it flags that the exact run *under-sampled*
      that region on its EA budget, so it's reported as a budget signal, not a defeat.
    - **Corner sharpening** — per objective, the exact optimum vs NSGA's best. Strongest at the
      convex risk/variance corner (a flat bowl where heuristics wobble and a QP is exact); where
      exact looks *short* of NSGA on a corner, that's the same budget artifact (a targeted exact
      solve would match it), flagged ``under-sampled``, not a capability limit.
    - **Completeness (the mirror audit)** — the exact overlay audited *by* NSGA: hypervolume the
      heuristic already covers that the scalarization sweep never sampled, localized as
      ``gap_regions`` with witness solution ids. The auditor framing's missing half — each engine
      audits the other; an under-covered verdict routes to the targeted fill
      (``solve scope="fill_gaps"``), which re-solves only the witnesses and merges.
    - **Frontier resolution (the sandwich bound)** — where coverage/completeness audit the two
      runs against each other, this bounds the reported frontier against the TRUE frontier: on a
      2-objective continuous overlay, the certified points' chords (above) and their floor-dual
      tangents (below) sandwich the convex tradeoff curve, and the widest gap is a guarantee —
      between the anchors, no feasible plan improves the primary over interpolation of the
      certified points by more than that gap. None on shapes that can't carry the claim
      (see ``_frontier_resolution_block``).

    Returns a JSON-friendly dict (the MCP ``explore certify`` payload).
    """
    if not nsga_run.solutions or not exact_run.solutions:
        raise ValueError("certify needs both runs to have solutions (run NSGA and an exact solver first).")
    # Stale guard: a run missing a current objective would certify in a fabricated
    # objective space (the value matrix fills absent names with 0.0).
    _consistent(problem, nsga_run)
    _consistent(problem, exact_run, remedy='the exact solve (solve solver="highs"|"cuopt")')

    objs = problem.objectives
    names = [o.name for o in objs]
    # Minimize-space sign per objective (+1 minimize, -1 maximize) so plain ≤ is "no worse".
    sign = np.array([-1.0 if o.direction == Direction.maximize else 1.0 for o in objs])

    def _matrix(run: Run) -> np.ndarray:
        return np.array([[s.objective_values.get(n, 0.0) for n in names] for s in run.solutions], dtype=float)

    nat_N, nat_E = _matrix(nsga_run), _matrix(exact_run)   # natural (reported) values
    N, E = nat_N * sign, nat_E * sign                       # minimize-space

    # Exact frontier reference = the non-dominated subset of the exact points (clean by
    # construction, but integer rounding can introduce a few dominated ones — re-filter).
    exact_front = _nondominated_idx_min(E)
    Eref = E[exact_front]

    # Dominance audit: NSGA points strictly dominated by the exact frontier (heuristic slack).
    dominated_idx = [i for i in range(len(N)) if any(_dominates_min(e, N[i]) for e in Eref)]
    examples = []
    for i in dominated_idx[:3]:
        beater = next(k for k in exact_front if _dominates_min(E[k], N[i]))
        examples.append({
            "nsga_point": {n: round(float(v), 4) for n, v in zip(names, nat_N[i])},
            "dominated_by_exact": {n: round(float(v), 4) for n, v in zip(names, nat_E[beater])},
        })

    # Invariant: exact points strictly dominated by any NSGA point (expected 0).
    exact_dominated = sum(1 for k in range(len(E)) if any(_dominates_min(N[i], E[k]) for i in range(len(N))))

    # Corner sharpening: per objective, exact optimum vs NSGA best (direction-aware natural terms).
    risk_name = next((o.name for o in objs
                      if o.aggregation == Aggregation.quadratic and o.direction == Direction.minimize), None)
    corners = {}
    for j, o in enumerate(objs):
        if o.direction == Direction.maximize:
            nb, eb = float(nat_N[:, j].max()), float(nat_E[:, j].max())
            improvement = eb - nb
        else:
            nb, eb = float(nat_N[:, j].min()), float(nat_E[:, j].min())
            improvement = nb - eb
        corners[o.name] = {
            "nsga_best": round(nb, 4),
            "exact_best": round(eb, 4),
            "improvement": round(improvement, 4),   # >0 exact sharpens; <0 exact under-samples (budget)
            "direction": o.direction.value,
            "status": "sharpened" if improvement > 1e-6 else ("matched" if improvement > -1e-6 else "under-sampled"),
            "is_risk_corner": o.name == risk_name,
        }

    # Headline: the risk corner if exact sharpened it, else the objective with the largest gain.
    sharpened = {n: c for n, c in corners.items() if c["status"] == "sharpened"}
    if risk_name and corners[risk_name]["status"] == "sharpened":
        headline = risk_name
    elif sharpened:
        headline = max(sharpened, key=lambda n: sharpened[n]["improvement"])
    else:
        headline = None

    completeness = _completeness_block(problem, nsga_run, exact_run)
    frontier_resolution = _frontier_resolution_block(
        problem, [exact_run.solutions[i] for i in exact_front])

    # The curated finalists, judged individually against the same certified front.
    per_pick = _per_pick_verdicts(problem, names, sign, N, E, exact_front, exact_run)
    dominated_picks = [sig for sig, v in per_pick.items() if v["verdict"] == "dominated"]
    inconclusive_picks = [sig for sig, v in per_pick.items() if v["verdict"] == "inconclusive"]
    infeasible_picks = [sig for sig, v in per_pick.items() if v["verdict"] == "infeasible"]

    parts = []
    if per_pick:
        # Lead with it: "are my finalists optimal?" is the question the phase exists to answer.
        # Count optimal directly — deriving it by subtraction silently absorbs any new verdict.
        n_opt = sum(1 for v in per_pick.values() if v["verdict"] == "optimal")
        tally = f"curated picks: {n_opt} of {len(per_pick)} optimal at their own tradeoff"
        if dominated_picks:
            tally += f", {len(dominated_picks)} dominated"
        if infeasible_picks:
            tally += f", {len(infeasible_picks)} infeasible under current constraints"
        if inconclusive_picks:
            tally += f", {len(inconclusive_picks)} inconclusive"
        parts.append(tally + " (see per_pick)")
    if dominated_idx:
        parts.append(f"exact certifies {len(dominated_idx)}/{len(N)} NSGA points as dominated (heuristic slack)")
    if headline:
        c = corners[headline]
        corner = "risk corner" if c["is_risk_corner"] else "corner"
        parts.append(f"sharpens the {headline} {corner} {c['nsga_best']}→{c['exact_best']}")
    if not parts:
        parts.append("NSGA already matches the exact frontier here — exact confirms, adds no new points")
    under = [n for n, c in corners.items() if c["status"] == "under-sampled"]
    if under:
        parts.append(f"under-samples {', '.join(under)} (this run's sweep sampling, not a "
                     "capability limit — a re-run with a higher budget or a targeted exact "
                     "solve closes it)")
    if completeness and completeness["verdict"] == "under_covered":
        share = completeness["reclaimed_fraction_over_exact"]
        parts.append(f"the overlay under-covers {len(completeness['gap_regions'])} region(s) the "
                     f"heuristic reached ({share:.1%} of combined hypervolume; see "
                     "completeness.gap_regions) — close them with solve scope='fill_gaps'")
    if frontier_resolution:
        sw, (lo, hi) = next(iter(frontier_resolution["span"].items()))
        frac = frontier_resolution["max_gap"]["fraction_of_frontier_range"]
        parts.append(f"resolution-certified across {sw} {lo:g}–{hi:g}: no feasible plan "
                     f"beats the reported frontier by more than {frac:.1%} of its span "
                     "(frontier_resolution.claim)")
    recommendation = "; ".join(parts) + "."

    from solvers import run_is_certified   # local, matching _frontier_provenance's pattern

    certified = run_is_certified(exact_run, problem.approach)

    # Hand the agent onward (the certificate is a step, not a terminus). Shape-branched: a
    # continuous/QP overlay also carries solver duals (`explore sensitivity`); a MILP overlay
    # is integer, so duals don't exist — point at the frontier-derived binding_analysis instead.
    anchor = f"the sharpened {headline} corner" if headline else "the certified frontier"
    if problem.approach.value == "binary":
        # Calibrated to the certificate: "optimal" without a hedge only when exact_certified.
        # A default bounded run proves each point to a 0.1% gap UNLESS the inner solve's
        # safety deadline (not the gap criterion) stopped it — the run doesn't record which
        # fired, so the bounded prose must hedge for both.
        optimality = ("every point is now optimal, not heuristic" if certified else
                      "every point is now exact-solver optimal to a 0.1% gap, not heuristic "
                      "(a point whose inner solve hit its safety deadline can carry a looser "
                      "incumbent; re-run with exact=true for zero-gap certificates)")
        next_steps = (
            f"Present {anchor} to the user as the decision anchor — {optimality}. "
            "Integer/MILP solutions carry no exact duals; use `binding_analysis` "
            "from `explore tradeoffs` for shadow-price intuition. Read this certificate with the "
            "`solution_interpreter` skill ('Reading the Certificate')."
        )
    else:
        next_steps = (
            f"Present {anchor} to the user as the decision anchor, and navigate the exact overlay "
            "with `explore … source=\"exact\"`. On this continuous/QP problem, `explore sensitivity` "
            "adds solver-exact shadow prices + reduced costs (the explainability layer) — anchor it "
            "on a solution_id or content_signature taken from `explore solutions source=\"exact\"`, "
            "not from a per_pick entry: an exact re-solve re-allocates, so even an `optimal` pick's "
            "signature is often absent from the exact run (the verdict is a comparison, not a "
            "guarantee the plan is a point on that frontier). Read this certificate with the "
            "`solution_interpreter` skill ('Reading the Certificate')."
        )
    if dominated_picks:
        # Routing, not teaching: a dominated finalist has a certified replacement waiting, and
        # `per_pick` names it by the signature `explore solutions` echoes.
        next_steps = (
            "Re-curate first: per_pick names a certifying replacement for each dominated pick "
            f"({len(dominated_picks)} of {len(per_pick)}) — pass its `dominated_by` signature "
            "straight to `explore curate content_signature=<sig> source=\"exact\"` (and "
            "`explore solutions content_signature=<sig> source=\"exact\"` to read it first), "
            "then unpin the dominated one with `remove=true`. " + next_steps)
    if infeasible_picks:
        # An infeasible pin has no dominating replacement to name — the route is the current
        # frontier itself.
        next_steps = (
            f"{len(infeasible_picks)} pinned plan(s) violate the CURRENT constraints (see "
            "per_pick verdict=infeasible) — unpin with `remove=true` and pick a replacement "
            "from the current frontier; every plan there honors the rules as they now stand. "
            + next_steps)

    coverage = _coverage_gain(N, Eref)  # the cleaned exact front (as the dominance audit uses), so integer-rounding artifacts can't rescale the shared box
    if coverage is not None:
        coverage["mirror"] = "completeness"  # the inverse audit rides the same certificate

    # Quality gate over the certified points: an exact optimum can still be a degenerate plan
    # (empty, pinned, concentrated) — the certificate says "optimal for the model as written",
    # the gate says whether the model as written asks for something actionable.
    flagged = []
    for s in exact_run.solutions:
        q = solution_quality(problem, s.selected_options, s.allocations)
        if q["status"] != "GOOD":
            flagged.append({"solution_id": s.solution_id, "status": q["status"],
                            "checks": [f["check"] for f in q["flags"]]})

    bound_dips = _objective_bound_dips(problem, exact_run)

    return {
        "quality_gates": {
            "flagged": flagged,
            # Diagnosis guidance lives in 'Reading the Certificate' (the guidance_pointer target).
            "note": ("degenerate or pinned certified plans — surfaced for the user, never dropped"
                     if flagged else "no degenerate plans among the certified points"),
        },
        "nsga_run_id": nsga_run.run_id,
        "exact_run_id": exact_run.run_id,
        "exact_solver": exact_run.solver,
        "exact_certified": certified,
        **({"rounding_bound_dips": {
            "dips": bound_dips,
            "note": ("whole-percent rounding of the continuous optimum — the LP/QP "
                     "solution honors the bound; a display artifact of the integer "
                     "allocation, not an infeasible plan"),
        }} if bound_dips else {}),
        "nsga_count": len(N),
        "exact_count": len(E),
        "dominance_audit": {
            "nsga_dominated_by_exact": len(dominated_idx),
            "nsga_dominated_fraction": round(len(dominated_idx) / len(N), 4) if len(N) else 0.0,
            "examples": examples,
        },
        # The curated lens on that audit — keyed by content_signature, absent (not empty)
        # when nothing is curated, the same way `regret.curated` behaves.
        **({"per_pick": per_pick} if per_pick else {}),
        "coverage": coverage,
        "completeness": completeness,
        "frontier_resolution": frontier_resolution,
        "invariant": {
            "holds": exact_dominated == 0,
            "exact_dominated_by_nsga": exact_dominated,
            # A violation has a different honest explanation per shape: proportional exact
            # optima are rounded to whole-percent allocations (rounding artifact); a binary
            # run without exact=True accepts gap/time-bounded incumbents a full NSGA search
            # can legitimately beat (bounded-incumbent artifact, cured by exact=True).
            "note": ((("a few exact points edged out by NSGA — this bounded MILP run accepts "
                       "gap/time-limited incumbents, so that is an uncertified incumbent, not a "
                       "heuristic beating a certified optimum; re-run with exact=True to certify"
                       if problem.approach == Approach.binary and not exact_run.exact else
                       "a few exact points edged out by NSGA — the integer rounding of the "
                       "continuous LP/QP optimum to whole-percent allocations, not a heuristic "
                       "beating the exact solve (certified MILP corners, integer by "
                       "construction, never show this)"))
                     if exact_dominated else "NSGA dominates no exact point — exact can only confirm or improve"),
        },
        "corner_sharpening": corners,
        "headline_corner": headline,
        "recommendation": recommendation,
        "next_steps": next_steps,
    }


def _audit_framing(verdict: str) -> dict:
    """Per-verdict recommendation + next-step for the `explore audit` payload. The deep
    presentation playbook lives in solution_interpreter → 'Reading the Audit (explore audit)';
    this is the one-line frame the agent leads with."""
    read = "Read this with the `solution_interpreter` skill ('Reading the Audit (explore audit)')."
    text = {
        "feasible": "A feasible plan exists — the current constraints are satisfiable. The witness is "
                    "one concrete example; proceed to solve() for the frontier.",
        "no_feasible_plan": "No feasible plan exists — the constraints jointly over-constrain the problem "
                            "(the exact version of validate's pre-solve check). The attached `conflicts` "
                            "names a minimal set that cannot all hold together: relaxing any one member "
                            "clears that conflict, though other independent conflicts may remain — relax "
                            "one and re-probe to confirm. Which member to relax is the user's call.",
        "violated": "The property does NOT hold across the feasible space — the witness is a concrete "
                    "counterexample. Treat the property as an aspiration, not a guarantee; encode it as a "
                    "hard constraint if it must hold.",
        "holds": "Proven: the property holds for EVERY feasible plan — the negation is infeasible across "
                 "the whole feasible region, not a sampled subset. A guarantee you can put in front of a "
                 "stakeholder, not a spot-check.",
        "holds_vacuously": "The property holds only vacuously — there are no feasible plans for it to apply "
                           "to. The attached `conflicts` names the contradiction; relax one member and "
                           "re-probe feasibility (audit with no property).",
        "inconclusive": "INCONCLUSIVE — the solve stopped (time limit or solver error; see the attached "
                        "status) without proving anything. Not evidence either way; do not read this "
                        "as a pass.",
    }
    # An unfit shape / missing backend raises ValueError upstream (→ a tool error), never reaching
    # here — so every verdict in `text` is an analytical outcome, framed the same way.
    return {"recommendation": text.get(verdict, ""), "next_steps": read}


def audit_property(problem: Problem, property_dict: dict | list | None) -> dict:
    """MCP payload for `explore audit` — the witness / feasibility auditor (sibling of certify).

    Parses the optional property (a ``Constraint`` dict, or a *list* of them for a compound
    guarantee — the conjunction holds iff every conjunct holds — same vocabulary as model
    constraints), runs the engine audit over the whole feasible space, and frames the verdict for
    presentation. No prior solve required: audit reasons about the model's feasible region
    directly, so it can feasibility-probe *before* spending a solve."""
    from engine import optimizer

    # The auditable constraint vocabulary + canonical shape, named once so every rejection
    # path (non-dict, missing/unknown `type`, bad fields) points the caller at the same
    # discoverable example instead of a cryptic discriminated-union error.
    _AUDIT_TYPES = ("objective_bound", "force_include", "force_exclude", "cardinality",
                    "group_limit", "exclusion_pair", "dependency")
    _SHAPE_HINT = (
        'give a "type" field — one of: ' + ", ".join(_AUDIT_TYPES) + '. Most common: '
        '{"type": "objective_bound", "objective": "Cost", "operator": "max", "value": 100} '
        '(operator is "max" or "min"; the numeric field is "value").')

    def _parse_one(d) -> "Constraint":
        if not isinstance(d, dict):
            raise ValueError(
                "audit `property` must be a constraint object (or a list of them) — " + _SHAPE_HINT)
        from typing import Annotated

        from pydantic import Field, TypeAdapter, ValidationError

        from engine.models import Constraint

        # A floor-only guarantee ("at least N from this group") is a natural audit ask;
        # the model vocabulary requires a cap on group_limit, so default it to the vacuous
        # one (the group size) instead of rejecting the property.
        if d.get("type") == "group_limit" and "max" not in d and d.get("options"):
            d = {**d, "max": len(d["options"])}
        try:
            # Discriminate on `type` so a validation error names the intended constraint's
            # missing/invalid fields rather than an arbitrary union member.
            return TypeAdapter(Annotated[Constraint, Field(discriminator="type")]).validate_python(d)
        except ValidationError as e:
            errs = e.errors()
            # A missing/invalid `type` discriminator yields a cryptic union error — name the
            # vocabulary + canonical shape instead (this is the discoverability trap: a dict
            # without `type`, or with an unknown one). Field-level errors (valid type, wrong
            # fields) keep pydantic's precise per-field detail, plus the shape reminder.
            if any(err.get("type") in ("union_tag_not_found", "union_tag_invalid") for err in errs):
                raise ValueError("audit `property` needs a valid constraint `type` — " + _SHAPE_HINT)
            detail = "; ".join(
                f"{'.'.join(str(loc) for loc in err.get('loc', ())) or 'property'}: {err.get('msg', '')}"
                for err in errs[:3]) or str(e)
            raise ValueError(f"audit `property` isn't a valid constraint: {detail}. Shape — {_SHAPE_HINT}")

    prop = None
    if isinstance(property_dict, (list, tuple)):
        if not property_dict:
            raise ValueError(
                "audit `property` list is empty — pass one or more constraint objects, or omit "
                "the property to probe feasibility.")
        prop = [_parse_one(d) for d in property_dict]
    elif property_dict is not None:
        prop = _parse_one(property_dict)

    result = optimizer.audit(problem, prop=prop, solver="highs")
    result["audited"] = ("feasibility of the current constraint set" if prop is None else property_dict)
    result.update(_audit_framing(result["verdict"]))
    return result


def _constraint_key(c: dict) -> str:
    """Stable string key for a constraint dict, for comparison.

    Keys the *rule*, never its annotations: `motivated_by` documents why a constraint exists
    and changing it must not read as a model change. The typed branches below already ignore
    unlisted fields; the unknown-type fallback strips the annotation explicitly so a future
    constraint type inherits that property instead of re-acquiring the bug."""
    ctype = c.get("type", "")
    if ctype == "cardinality":
        return f"cardinality:{c.get('min')}:{c.get('max')}"
    elif ctype in ("force_include", "force_exclude"):
        return f"{ctype}:{c.get('option')}"
    elif ctype == "objective_bound":
        return f"objective_bound:{c.get('objective')}:{c.get('operator')}:{c.get('value')}"
    elif ctype == "exclusion_pair":
        return f"exclusion_pair:{c.get('option_a')}:{c.get('option_b')}"
    elif ctype == "dependency":
        return f"dependency:{c.get('if_option')}:{c.get('then_option')}"
    elif ctype == "group_limit":
        return f"group_limit:{','.join(c.get('options', []))}:{c.get('min', 0)}:{c.get('max')}"
    elif ctype == "max_allocation":
        return f"max_allocation:{c.get('max')}"
    elif ctype == "allocation_bound":
        return f"allocation_bound:{c.get('option')}:{c.get('min', 0)}:{c.get('max')}"
    return str({k: v for k, v in sorted(c.items()) if k != "motivated_by"})


_QUALITY_SEVERITY = {"GOOD": 0, "WARNING": 1, "DEGENERATE": 2}


def solution_quality(problem: Problem, selected_options: list[str], allocations: dict | None) -> dict:
    """Deterministic degenerate-plan checks over one solution/finalist. An OPTIMAL solver status
    certifies optimality for the model as written, not that the plan is actionable — an empty or
    pinned plan usually signals a missing or mis-scaled constraint. Flags travel WITH the finalist,
    never drop it: selection stays the user's call.

    Returns ``{"status": GOOD | WARNING | DEGENERATE, "flags": [{check, severity, message}]}``,
    each message naming the triggering check in the user's terms."""
    flags = []
    alloc = {k: v for k, v in (allocations or {}).items() if v}

    if not alloc and not selected_options:
        flags.append({
            "check": "empty_selection", "severity": "DEGENERATE",
            "message": "this plan selects nothing — an optimal empty plan usually means a missing "
                       "must-cover requirement (e.g. a cardinality minimum or an objective floor)",
        })

    # Distribution checks apply where the shape implies a spread (proportional allocations);
    # a one-option binary selection is often a legitimate answer, so it is not flagged.
    if problem.approach == Approach.proportional and alloc:
        n = len(problem.options)
        total = sum(alloc.values())
        top_name, top_val = max(alloc.items(), key=lambda kv: kv[1])
        if n > 1 and top_val >= 0.9 * total:
            flags.append({
                "check": "single_option_concentration", "severity": "WARNING",
                "message": f"'{top_name}' takes {round(100 * top_val / total)}% of the allocation — "
                           "if you expected a spread, add a max_allocation cap or revisit the "
                           "scores/interactions that let one option dominate",
            })
        # The APPLIED cap (the tightest row), so the at-a-bound test measures allocations
        # against the same edge the solver placed them on.
        from .optimizer import merged_max_allocation
        cap = merged_max_allocation(problem.constraints) or 100
        if n >= 3:
            at_bounds = sum(1 for o in problem.options if alloc.get(o.name, 0) in (0, cap))
            if at_bounds >= 0.9 * n:
                flags.append({
                    "check": "allocations_at_bounds", "severity": "WARNING",
                    "message": f"{at_bounds} of {n} allocations sit at a bound (0 or the {cap}% "
                               "cap) — the optimizer is pinned to the box edges, so the interior "
                               "tradeoff space is unused; check whether the cap is tighter than "
                               "intended",
                })

    status = "GOOD"
    for f in flags:
        if _QUALITY_SEVERITY[f["severity"]] > _QUALITY_SEVERITY[status]:
            status = f["severity"]
    return {"status": status, "flags": flags}


def curate_solution(problem: Problem, solution_id: int, custom_name: str = "", notes: str = "", scenario: str | None = None, source: str | None = None) -> dict:
    """Add a solution from the current frontier to the curated set."""
    run = _require_run(problem, scenario, source)
    sol = None
    for s in run.solutions:
        if s.solution_id == solution_id:
            sol = s
            break
    if sol is None:
        raise ValueError(f"Solution {solution_id} not found in current run.")

    sig = sol.content_signature or _content_signature(sol.selected_options, sol.allocations)

    # Duplicate check
    for cs in problem.curated_solutions:
        if cs.content_signature == sig:
            return {"error": f"Solution already curated as '{cs.custom_name or sig}'."}

    # Pull in any existing feedback matching this signature
    existing_feedback = [
        fb for fb in problem.feedback if fb.content_signature == sig
    ]

    curated = CuratedSolution(
        content_signature=sig,
        custom_name=custom_name,
        selected_options=sol.selected_options,
        allocations=sol.allocations,
        objective_values=sol.objective_values,
        source_run_id=run.run_id,
        notes=notes,
        feedback=existing_feedback,
    )
    problem.curated_solutions.append(curated)
    return {
        "curated": True,
        "content_signature": sig,
        "custom_name": custom_name,
        "total_curated": len(problem.curated_solutions),
        # Quality gate rides along, never blocks: a flagged finalist is curated with its flag.
        "quality": solution_quality(problem, sol.selected_options, sol.allocations),
    }


def uncurate_solution(problem: Problem, content_signature: str) -> dict:
    """Remove a solution from the curated set."""
    before = len(problem.curated_solutions)
    problem.curated_solutions = [
        cs for cs in problem.curated_solutions if cs.content_signature != content_signature
    ]
    if len(problem.curated_solutions) == before:
        raise ValueError(f"No curated solution with signature '{content_signature}'.")
    return {"removed": content_signature, "total_curated": len(problem.curated_solutions)}


def rename_curated(problem: Problem, content_signature: str, custom_name: str) -> dict:
    """Update the name of a curated solution."""
    for cs in problem.curated_solutions:
        if cs.content_signature == content_signature:
            cs.custom_name = custom_name
            return {"renamed": content_signature, "custom_name": custom_name}
    raise ValueError(f"No curated solution with signature '{content_signature}'.")


def export_curated(problem: Problem, format: str = "markdown") -> dict:
    """Export curated solutions as raw formatted text for handoff.

    format: "markdown" (pipe-aligned table) or "csv".
    Returns {"format": ..., "content": "<string>", "total_curated": N}.
    """
    fmt = (format or "markdown").lower()
    if fmt not in ("markdown", "csv"):
        raise ValueError(f"Unknown format '{format}'. Use 'markdown' or 'csv'.")

    curated = problem.curated_solutions
    if not curated:
        return {"format": fmt, "content": "", "total_curated": 0}

    obj_names = [o.name for o in problem.objectives]
    opt_names = [o.name for o in problem.options]
    use_allocations = any(cs.allocations for cs in curated)

    # Column layout: name, signature, quality gate, objective values, then either selected
    # options or allocations. Quality travels into the handoff so a flagged finalist stays flagged.
    headers = ["name", "content_signature", "quality", *obj_names]
    if use_allocations:
        headers += [f"alloc:{opt}" for opt in opt_names]
    else:
        headers += ["selected_options"]

    rows: list[list[str]] = []
    for cs in curated:
        q = solution_quality(problem, cs.selected_options, cs.allocations)
        row = [cs.custom_name or "", cs.content_signature, q["status"]]
        for obj in obj_names:
            val = cs.objective_values.get(obj)
            row.append("" if val is None else f"{val}")
        if use_allocations:
            alloc = cs.allocations or {}
            for opt in opt_names:
                row.append(str(alloc.get(opt, 0)))
        else:
            row.append("; ".join(cs.selected_options))
        rows.append(row)

    if fmt == "csv":
        import csv
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        writer.writerows(rows)
        content = buf.getvalue().rstrip("\n")
    else:
        # Markdown pipe table
        def _esc(v: str) -> str:
            return v.replace("|", "\\|")
        header_line = "| " + " | ".join(_esc(h) for h in headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = [
            "| " + " | ".join(_esc(c) for c in row) + " |"
            for row in rows
        ]
        content = "\n".join([header_line, separator, *body_lines])

    return {
        "format": fmt,
        "content": content,
        "total_curated": len(curated),
    }


def list_curated(problem: Problem) -> dict:
    """List all curated solutions with survival status against current run."""
    current_sigs = set()
    frontier = problem.run or problem.exact_run  # exact-only solves carry the frontier in exact_run
    if frontier:
        for s in frontier.solutions:
            sig = s.content_signature or _content_signature(s.selected_options, s.allocations)
            current_sigs.add(sig)

    curated = []
    for cs in problem.curated_solutions:
        entry = cs.model_dump()
        entry["in_current_frontier"] = cs.content_signature in current_sigs
        entry["quality"] = solution_quality(problem, cs.selected_options, cs.allocations)
        entry["feedback_count"] = len(cs.feedback)
        if cs.feedback:
            ratings = [fb.rating for fb in cs.feedback if fb.rating is not None]
            entry["avg_rating"] = round(sum(ratings) / len(ratings), 1) if ratings else None
        else:
            entry["avg_rating"] = None
        curated.append(entry)

    return {
        "total_curated": len(curated),
        "curated_solutions": curated,
    }


def compare_curated(problem: Problem, signatures: list[str], detail: bool = False) -> dict:
    """Side-by-side comparison of curated solutions.

    Default (detail=False): compact — omits per-solution `selected_options` and
    `allocations`. The `shared_options` and `differentiating_options` summary
    fields already convey the structural differences.

    detail=True: include full `selected_options` and `allocations` per solution.
    Use for artifact assembly or deep dives; for large proportional problems,
    pair with the `full_result_path` file written by solve instead.
    """
    sig_map = {cs.content_signature: cs for cs in problem.curated_solutions}

    selected = []
    for sig in signatures:
        if sig not in sig_map:
            raise ValueError(f"No curated solution with signature '{sig}'.")
        selected.append(sig_map[sig])

    # Shared and differentiating options
    option_sets = [set(cs.selected_options) for cs in selected]
    shared = set.intersection(*option_sets) if option_sets else set()
    all_options = set.union(*option_sets) if option_sets else set()
    differentiating = all_options - shared

    # Tradeoff summary per objective
    tradeoff_summary = {}
    for obj in problem.objectives:
        name = obj.name
        vals = {cs.content_signature: cs.objective_values.get(name, 0) for cs in selected}
        if not vals:
            continue
        best_sig = max(vals, key=vals.get) if obj.direction.value == "maximize" else min(vals, key=vals.get)
        worst_sig = min(vals, key=vals.get) if obj.direction.value == "maximize" else max(vals, key=vals.get)
        all_vals = list(vals.values())
        tradeoff_summary[name] = {
            "leads": best_sig,
            "trails": worst_sig,
            "range": [min(all_vals), max(all_vals)],
        }

    def _sol_entry(cs):
        entry = {
            "content_signature": cs.content_signature,
            "custom_name": cs.custom_name,
            "objective_values": cs.objective_values,
        }
        if detail:
            entry["selected_options"] = cs.selected_options
            entry["allocations"] = cs.allocations
        return entry

    result = {
        "solutions": [_sol_entry(cs) for cs in selected],
        "shared_options": sorted(shared),
        "differentiating_options": sorted(differentiating),
        "tradeoff_summary": tradeoff_summary,
        "detail": detail,
    }

    if problem.reference_points:
        for sol_dict in result["solutions"]:
            sol_dict["vs_references"] = _compute_reference_analysis(
                sol_dict["objective_values"], problem.reference_points, problem.objectives,
            )

    # Parallel coordinates visualization
    sol_dicts = [{"content_signature": cs.content_signature, "objective_values": cs.objective_values}
                 for cs in selected]
    labels = {cs.content_signature: (cs.custom_name or cs.content_signature[:8]) for cs in selected}
    result["visualization"] = _render_parallel_coords(sol_dicts, problem.objectives, labels)
    result["viz_data"] = _viz_data_parallel_coords(sol_dicts, problem.objectives, labels)

    return result


def _scenario_changes(sc) -> dict:
    """Deterministic restatement of what a scenario varies vs. inherits — the 'held fixed' line.
    Sweep discipline: a scenario varies exactly what it names; everything else anchors to the
    base model, and the reader sees that restated rather than assuming it."""
    varies = []
    if sc.score_overrides:
        by_obj: dict[str, set] = {}
        for s in sc.score_overrides:
            by_obj.setdefault(s.objective, set()).add(s.option)
        for obj, opts in sorted(by_obj.items()):
            varies.append(f"scores: '{obj}' for {len(opts)} option(s)")
    for adj in sc.score_adjustments:
        parts = []
        if adj.multiply is not None:
            parts.append(f"×{adj.multiply:g}")
        if adj.add is not None:
            parts.append(f"{adj.add:+g}")
        varies.append(f"scores: all '{adj.objective}' {' '.join(parts)}")
    if sc.constraint_overrides:
        varies.append(f"constraints: REPLACED wholesale by {len(sc.constraint_overrides)} "
                      "override(s) — base constraints do not carry over")
    if sc.interaction_matrix_overrides:
        varies.append(f"interactions: {len(sc.interaction_matrix_overrides)} matrix override(s)")
    return {
        "varies": varies or ["nothing — identical to the base model"],
        "held_fixed": "every parameter not listed in `varies` inherits the base model unchanged",
    }


def get_scenario_results(problem: Problem, cvar_alpha: float | None = None) -> dict:
    """Analyze per-scenario results: robust options, scenario-specific options, expected value.

    cvar_alpha: tail fraction in (0, 1) for Conditional Value-at-Risk. Default 0.2
        (worst 20% of scenarios by probability mass). CVaR of per-objective
        best-in-scenario value across scenarios — diagnostic only, not an
        optimization target.
    """
    if not problem.scenario_run or not problem.scenario_run.scenario_runs:
        raise ValueError("No scenario runs found. Use solve run_scenarios first.")
    if not problem.scenario_config or not problem.scenario_config.scenarios:
        raise ValueError("No scenario config found.")
    for _srun in problem.scenario_run.scenario_runs.values():
        _consistent(problem, _srun, remedy="solve run_scenarios")

    scenario_runs = problem.scenario_run.scenario_runs
    scenarios = {s.name: s for s in problem.scenario_config.scenarios}
    opt_names = [o.name for o in problem.options]
    obj_names = [o.name for o in problem.objectives]

    # Which scenarios' overrides were authored against a different base constraint set than
    # the current one. `varies` already discloses that overrides replace the base rules
    # wholesale; this says the replacement is FROZEN AT AN OLDER MODEL — the difference
    # between a disclosed convention and a live warning (a frozen pre-edit rule set is how a
    # scenario "wipeout" gets read as a finding).
    from .optimizer import constraints_fingerprint

    override_stale = set(stale_scenario_overrides(
        problem, constraints_fingerprint(problem.constraints)))

    # Per-scenario summaries
    per_scenario = {}
    for name, run in scenario_runs.items():
        ranges = {}
        for obj in obj_names:
            vals = [s.objective_values.get(obj, 0) for s in run.solutions]
            if vals:
                ranges[obj] = {"min": min(vals), "max": max(vals)}
        entry = {
            "solution_count": len(run.solutions),
            "objective_ranges": ranges,
        }
        sc = scenarios.get(name)
        if sc is not None:
            entry.update(_scenario_changes(sc))     # what varies + the held-fixed line
            if sc.motivated_by:
                entry["motivated_by"] = sc.motivated_by   # cite the lever that seeded this scenario
            if name in override_stale:
                entry["constraint_overrides_stale"] = True
                entry["constraint_overrides_note"] = (
                    "this scenario's constraint overrides were authored against an EARLIER base "
                    "constraint set — they replace the base rules wholesale, so later base-"
                    "constraint edits never reached this scenario. Read its results as a "
                    "question about the frozen rules, not a finding about the current model; "
                    "restate the overrides if the edit was meant to apply here.")
        per_scenario[name] = entry

    # Robust options: frequency + allocation-weighted across scenarios
    from collections import Counter

    all_scenario_names = list(scenario_runs.keys())
    # Per-scenario: count how often each option appears + total allocation weight
    per_scenario_freq = {}  # {scenario: {opt: frequency 0-1}}
    per_scenario_avg_weight = {}  # {scenario: {opt: avg_weight when present}}
    for name, run in scenario_runs.items():
        n_sols = len(run.solutions)
        opt_count: Counter = Counter()
        opt_weight_sum: Counter = Counter()
        for sol in run.solutions:
            for opt in sol.selected_options:
                opt_count[opt] += 1
                if sol.allocations:
                    opt_weight_sum[opt] += sol.allocations.get(opt, 0)
        per_scenario_freq[name] = {
            opt: opt_count[opt] / n_sols if n_sols else 0 for opt in opt_count
        }
        per_scenario_avg_weight[name] = {
            opt: opt_weight_sum[opt] / opt_count[opt] if opt_count[opt] else 0
            for opt in opt_count
        }

    # Aggregate across scenarios: avg frequency, avg weight, importance score.
    # Weight-derived fields only exist where allocations do — on binary selections
    # avg_weight/importance would read 0.0 for every option (meaningless next to a
    # "core" tier), so the rows omit them and rank by frequency alone.
    proportional = problem.approach == Approach.proportional
    option_robustness = []
    scenario_specific = {}
    for opt in opt_names:
        freqs = [per_scenario_freq[s].get(opt, 0) for s in all_scenario_names]
        present_in = [s for s in all_scenario_names if per_scenario_freq[s].get(opt, 0) > 0]
        avg_freq = sum(freqs) / len(freqs) if freqs else 0

        if avg_freq > 0:
            # Tier keyed on the WORST per-scenario frequency (see tier_rule in the payload):
            # core (>50% in every future), common (>25% worst-case or present in all),
            # marginal (the rest).
            min_freq = min(freqs)
            if min_freq > 0.5:
                tier = "core"
            elif min_freq > 0.25 or len(present_in) == len(all_scenario_names):
                tier = "common"
            else:
                tier = "marginal"

            row = {
                "option": opt,
                "avg_frequency": round(avg_freq, 3),
                "tier": tier,
                "scenarios_present": len(present_in),
            }
            if proportional:
                weights = [per_scenario_avg_weight[s].get(opt, 0) for s in all_scenario_names]
                avg_weight = sum(w for w in weights if w > 0) / max(len([w for w in weights if w > 0]), 1)
                row["avg_weight"] = round(avg_weight, 1)
                row["importance"] = round(avg_freq * avg_weight, 2)
            option_robustness.append(row)

        if 0 < len(present_in) < len(all_scenario_names):
            scenario_specific[opt] = present_in

    # Sort by importance where it exists (proportional), then frequency — the frequency
    # key alone carries the binary ranking.
    option_robustness.sort(key=lambda x: (x.get("importance", 0.0), x["avg_frequency"]), reverse=True)
    robust_options = [r["option"] for r in option_robustness if r["tier"] == "core"]

    # Size-aware trim: at portfolio scale (hundreds of options) the per-option table
    # dwarfs everything else in the payload — and is copied again into viz_data — which
    # pushes the MCP response past the inline result cap and clients get a file dump
    # instead of readable JSON. Ship the ranked head; summarize the elided tail. Small
    # problems (every bundled example except capital-300) pass through whole.
    _ROBUSTNESS_TABLE_CAP = 60
    option_robustness_elided = None
    if len(option_robustness) > _ROBUSTNESS_TABLE_CAP:
        total = len(option_robustness)
        tail = option_robustness[_ROBUSTNESS_TABLE_CAP:]
        option_robustness = option_robustness[:_ROBUSTNESS_TABLE_CAP]
        # The ranking named here must match the rows above it: binary rows carry no
        # importance field, so the note claims only the key the table actually ranks by.
        ranking = ("importance and cross-scenario frequency" if proportional
                   else "cross-scenario frequency")
        option_robustness_elided = {
            "shown": _ROBUSTNESS_TABLE_CAP,
            "total_options": total,
            "tiers": dict(Counter(t["tier"] for t in tail)),
            "note": (f"ranked table truncated to the top {_ROBUSTNESS_TABLE_CAP} by "
                     f"{ranking}; the elided tail ranks below everything shown. Per-plan "
                     "detail: explore solutions solution_id=<id>; cross-frontier selection "
                     "rates: explore composition."),
        }

    # Expected value: probability-weighted if probabilities provided, else equal-weight
    has_probabilities = all(
        scenarios[name].probability is not None for name in scenario_runs
    )
    n_scenarios = len(scenario_runs)

    expected_values = {}
    for obj in obj_names:
        ev = 0.0
        for name, run in scenario_runs.items():
            if has_probabilities:
                weight = scenarios[name].probability
            else:
                weight = 1.0 / n_scenarios  # equal weight
            vals = [s.objective_values.get(obj, 0) for s in run.solutions]
            if vals:
                direction = next(o.direction.value for o in problem.objectives if o.name == obj)
                best = max(vals) if direction == "maximize" else min(vals)
                ev += weight * best
        expected_values[obj] = round(ev, 4)

    # Risk: worst-case + CVaR across scenarios for each objective.
    # Uses best-in-scenario values (same basis as expected_values).
    alpha = 0.2 if cvar_alpha is None else float(cvar_alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"cvar_alpha must be in (0, 1); got {cvar_alpha}")

    scenario_risk = {}
    for obj in obj_names:
        direction = next(o.direction.value for o in problem.objectives if o.name == obj)
        per_scenario_best = []  # list of (value, probability)
        for name, run in scenario_runs.items():
            vals = [s.objective_values.get(obj, 0) for s in run.solutions]
            if not vals:
                continue
            best = max(vals) if direction == "maximize" else min(vals)
            prob = scenarios[name].probability if has_probabilities else 1.0 / n_scenarios
            per_scenario_best.append((best, prob))
        if not per_scenario_best:
            continue

        values = [v for v, _ in per_scenario_best]
        worst = min(values) if direction == "maximize" else max(values)
        best_across = max(values) if direction == "maximize" else min(values)

        # CVaR: mean of worst α-fraction by probability mass
        # Sort ascending for maximize (lower = worse), descending for minimize (higher = worse)
        sorted_pairs = sorted(per_scenario_best, key=lambda vp: vp[0], reverse=(direction == "minimize"))
        cum = 0.0
        cvar_num = 0.0
        cvar_den = 0.0
        for v, p in sorted_pairs:
            remaining = alpha - cum
            if remaining <= 0:
                break
            take = min(p, remaining)
            cvar_num += v * take
            cvar_den += take
            cum += take
        cvar = round(cvar_num / cvar_den, 4) if cvar_den > 0 else None

        scenario_risk[obj] = {
            "expected": expected_values.get(obj),
            "worst_case": round(worst, 4),
            "best_case": round(best_across, 4),
            f"cvar_{int(round(alpha * 100))}": cvar,
            "range": [round(min(values), 4), round(max(values), 4)],
        }

    result = {
        # Leads the payload: a reader must meet the staleness before the numbers, not after.
        **_scenario_stale_block(problem),
        "per_scenario": per_scenario,
        "robust_options": sorted(robust_options),
        "option_robustness": option_robustness,
        "tier_rule": ("tiers key on the WORST per-scenario frequency, not the average shown: "
                      "core = in >50% of plans in EVERY future; common = >25% worst-case (or "
                      "present in all); marginal = the rest — an option can average high yet "
                      "tier lower because one future benches it"),
        **({"option_robustness_elided": option_robustness_elided}
           if option_robustness_elided else {}),
        "scenario_specific_options": scenario_specific,
        "expected_values": expected_values,
        "scenario_risk": scenario_risk,
        # With one scenario there is no spread to take a tail of: worst_case, best_case and
        # CVaR all collapse onto that scenario's own best value. Three numbers agreeing reads
        # as a robustness finding when it is an echo, so the payload says so rather than
        # leaving the reader to notice n=1 (present only in that case). `expected` is left out
        # of the claim: it is probability-weighted, so a lone scenario carrying a probability
        # scales it by that probability and it does NOT match the other three.
        **({"scenario_risk_note": (
            "single scenario — risk spread degenerates: worst_case, best_case and CVaR are "
            "all that one scenario's value, so their agreement carries no tail information. "
            "Add scenarios for a tail reading.")} if n_scenarios == 1 else {}),
        "cvar_alpha": alpha,
        "weighting": "probability" if has_probabilities else "equal",
    }
    # Minimax-regret robustness lens, beside CVaR/worst-case (a distinct question:
    # "how much worse than the best I could have chosen, in hindsight?").
    result["regret"] = scenario_regret(problem)
    # Lift per-scenario survivor counts into the per_scenario layer — "N of M base plans
    # survive scenario X" is a per-scenario fact every consumer should get without digging
    # into the regret lens (and per_solution is truncated to 20, so it isn't derivable there).
    base_total = len(problem.run.solutions) if problem.run else 0
    for name, count in (result["regret"].get("survivors_by_scenario") or {}).items():
        if name in result["per_scenario"]:
            result["per_scenario"][name]["base_plans_feasible"] = count
            result["per_scenario"][name]["base_plans_total"] = base_total
    result["visualization"] = _render_scenario_viz(result)
    result["viz_data"] = _viz_data_scenario_summary(result)
    return result


def get_scenario_frontiers(problem: Problem) -> dict:
    """Per-scenario Pareto frontiers overlaid as parallel coordinates, colored by scenario.

    Each scenario's frontier is a separate non-dominated set; overlaying them shows how
    the achievable tradeoffs shift across futures (recession / inflation / rate_cuts …).
    """
    if not problem.scenario_run or not problem.scenario_run.scenario_runs:
        raise ValueError("No scenario runs found. Use solve run_scenarios first.")
    scenario_runs = problem.scenario_run.scenario_runs
    counts = {name: len(run.solutions) for name, run in scenario_runs.items()}
    return {
        **_scenario_stale_block(problem),
        "scenarios": list(scenario_runs.keys()),
        "solution_counts": counts,
        "visualization": _render_scenario_frontiers(scenario_runs, problem.objectives),
        "viz_data": _viz_data_scenario_parcoords(scenario_runs, problem.objectives, problem=problem),
        "note": (
            "Per-scenario frontiers overlaid as parallel coordinates, colored by scenario. "
            "Narrate how the achievable tradeoffs shift across scenarios."
        ),
    }


def _render_scenario_frontiers(scenario_runs: dict, objectives: list) -> str:
    """Per-scenario frontier ranges as a readable table — the ASCII/MD equivalent of
    the colored overlay, for chat / coding-agent surfaces that don't render charts."""
    names = list(scenario_runs.keys())
    lines = ["─── Per-Scenario Frontiers (objective ranges) ───", ""]
    name_w = max(12, max((len(o.name) for o in objectives), default=12) + 2)
    col_w = max(16, max((len(n) for n in names), default=10) + 2)
    header = "objective".ljust(name_w) + "| " + "| ".join(n.ljust(col_w) for n in names)
    lines.append(header)
    lines.append("-" * len(header))
    for o in objectives:
        cells = []
        for n in names:
            vals = [s.objective_values[o.name] for s in scenario_runs[n].solutions if o.name in s.objective_values]
            cells.append((f"{min(vals):.2f}–{max(vals):.2f}" if vals else "—").ljust(col_w))
        arrow = "↑" if o.direction.value == "maximize" else "↓"
        lines.append(f"{(o.name + ' ' + arrow).ljust(name_w)}| " + "| ".join(cells))
    lines.append("")
    lines.append("Each cell is the [min–max] achievable range for that objective under that scenario.")
    return "\n".join(lines)


def marginal_analysis(problem: Problem, scenario: str | None = None, detail: bool = False, source: str | None = None) -> dict:
    """Marginal rate analysis: cost-per-unit improvement between adjacent Pareto solutions.

    For each negatively-correlated objective pair, sorts solutions by one objective
    and computes the marginal rate of exchange. Detects inflection points where the rate
    jumps sharply — the point of diminishing returns.

    Default (detail=False): summary per pair — inflection, stats, top-5 steepest rates.
    detail=True: includes full rates array and untruncated visualization.
    """
    run = _require_run(problem, scenario, source)
    solutions = run.solutions
    objectives = problem.objectives
    obj_names = [o.name for o in objectives]

    if len(solutions) < 3:
        return {"pairs": [], "note": "Need at least 3 solutions for marginal analysis.",
                "frontier_source": _frontier_provenance(problem, run, scenario)}

    pairs = []
    for i, j, r in _conflicting_pair_indices(solutions, objectives):
        obj_a = objectives[i]
        obj_b = objectives[j]

        # Traverse in A's IMPROVING direction (worst-A first), so "beyond this point each unit
        # of A costs more B" reads forward and a convex elbow shows as a rate jump the detector
        # can fire on. (Best-A-first traversal shows the same elbow as a rate *drop*, which the
        # up-jump detector misses entirely.)
        reverse_a = obj_a.direction.value == "minimize"
        sorted_sols = sorted(
            solutions,
            key=lambda s: s.objective_values[obj_a.name],
            reverse=reverse_a,
        )

        # Marginal rates between adjacent solutions — solver-exact duals where available
        raw_rates, rate_basis, floors = _pair_rates_with_basis(sorted_sols, obj_a, obj_b, problem)
        rates = [
            {
                "from_id": rr["from_sol"].solution_id,
                "to_id": rr["to_sol"].solution_id,
                f"delta_{obj_a.name}": round(rr["delta_a"], 4),
                f"delta_{obj_b.name}": round(rr["delta_b"], 4),
                "rate": round(rr["rate"], 4),
                **({"degenerate": True} if rr.get("degenerate") else {}),
                **({"flat": True} if rr.get("flat") else {}),
                **({"co_improvement": True} if rr.get("co_improvement") else {}),
            }
            for rr in raw_rates
        ]
        trustworthy = _trustworthy_rates(raw_rates)

        # Detect inflection: largest jump in marginal rate, over trustworthy transitions only
        detected = _detect_inflection([rr["rate"] for rr in raw_rates], eligible=trustworthy)
        if detected is None:
            inflection = None
        else:
            landing = raw_rates[detected["position"]]
            co_gain = bool(landing.get("co_improvement"))
            inflection = {
                "solution_id": landing["from_sol"].solution_id,
                "position": detected["position"],
                "jump_factor": detected["jump_factor"],
                **({"co_improvement": True} if co_gain else {}),
                "rationale": _knee_rationale(obj_a, obj_b, detected["jump_factor"], co_gain),
            }

        pair_result = {
            "objectives": [obj_a.name, obj_b.name],
            "correlation": round(r, 2),
            "rate_basis": rate_basis,
            "rate_guard": _rate_guard(rates, obj_a, obj_b, floors),
            "inflection": inflection,
        }

        if detail:
            # Full output: all rates + untruncated viz
            pair_result["rates"] = rates
            pair_result["visualization"] = _render_marginal_rates(
                rates, obj_a, obj_b, inflection,
            )
            pair_result["viz_data"] = _viz_data_marginal_rates(rates, obj_a, obj_b, inflection)
        else:
            # Summary: stats + top-5 steepest + truncated viz. Stats span EVERY transition —
            # they describe how this frontier is sampled, so the median read is the same
            # number with or without the guard; the headline lists below are the filtered ones.
            rate_values = [r["rate"] for r in rates]
            pair_result["summary"] = {
                "total_transitions": len(rates),
                "rate_min": round(min(rate_values), 4) if rate_values else 0,
                "rate_max": round(max(rate_values), 4) if rate_values else 0,
                "rate_median": round(float(np.median(rate_values)), 4) if rate_values else 0,
            }
            # Top-5 steepest transitions (the most expensive tradeoff steps) — over the
            # transitions a rate can be trusted on, so the headline is a price and never a
            # tie's artifact. Both flags are filtered, on either rate basis: a `flat` row is
            # harmless on secants (a step that bought ~nothing divides out to a ~0 rate, which
            # cannot rank), but on `solver_exact_duals` the rate is the endpoint's shadow price
            # and carries no memory of the step — a flat row measured at 900.0 topped this list
            # while buying 0.0001. The number is a real slope; naming that transition the most
            # expensive step is the misread. It keeps its place, flagged, in the detail rows.
            top_rates = sorted(
                (r for r, ok in zip(rates, trustworthy) if ok),
                key=lambda r: r["rate"], reverse=True,
            )[:5]
            pair_result["steepest_transitions"] = top_rates
            # Truncated viz: ~20 rows around inflection
            pair_result["visualization"] = _render_marginal_rates(
                rates, obj_a, obj_b, inflection, max_rows=20,
            )
            pair_result["viz_data"] = _viz_data_marginal_rates(
                rates, obj_a, obj_b, inflection, max_rows=20,
            )

        pairs.append(pair_result)

    return {"pairs": pairs, "frontier_source": _frontier_provenance(problem, run, scenario)}


def cap_marginal_detail(problem: Problem, result: dict, max_rows: int) -> dict:
    """A token-capped view of a ``detail=true`` marginal_analysis payload.

    detail=true returns one row per adjacent-solution transition per pair — O(frontier size),
    which runs to hundreds of rows and tens of thousands of tokens at portfolio scale. This
    returns a new payload keeping at most ``max_rows`` transitions per pair, windowed on the
    inflection exactly as the summary view is, with the visualization and viz_data re-rendered
    to match. Nothing is mutated: the caller keeps the full payload for the on-disk dump the
    capped response points at.
    """
    by_name = {o.name: o for o in problem.objectives}
    capped = []
    for pair in result.get("pairs", []):
        rates = pair.get("rates") or []
        if len(rates) <= max_rows or not all(n in by_name for n in pair["objectives"]):
            capped.append(pair)
            continue
        obj_a, obj_b = (by_name[n] for n in pair["objectives"])
        inflection = pair.get("inflection")
        start, end = _marginal_window(len(rates), inflection, max_rows)
        capped.append({
            **pair,
            "rates": rates[start:end],
            # `position` on the inflection indexes the FULL list, which is what the on-disk
            # dump holds; the window says which slice of it is inline.
            "truncated": {"total_transitions": len(rates), "shown": end - start,
                          "window": [start, end]},
            "visualization": _render_marginal_rates(rates, obj_a, obj_b, inflection,
                                                    max_rows=max_rows),
            "viz_data": _viz_data_marginal_rates(rates, obj_a, obj_b, inflection,
                                                 max_rows=max_rows),
        })
    return {**result, "pairs": capped}


def cap_composition_detail(result: dict, max_rows: int) -> dict:
    """A token-capped view of a ``detail=true`` composition payload.

    detail=true lifts the co-occurrence top-N from a summary handful to every
    sometimes-but-not-always option pair — O(options²), which runs to thousands of rows
    and hundreds of KB at portfolio scale. This returns a new payload keeping the top
    ``max_rows`` pairs: the list is already ranked by departure from independence, so the
    head IS the summary view scaled up, and no re-rendering is needed (the visualization
    reads option_selection / principles / clusters, never the pair list). Nothing is
    mutated: the caller keeps the full payload for the on-disk dump the capped response
    points at.
    """
    pairs = result.get("co_occurrence") or []
    if len(pairs) <= max_rows:
        return result
    return {**result,
            "co_occurrence": pairs[:max_rows],
            "truncated": {"total_pairs": len(pairs), "shown": max_rows}}


def _rate_guard(rates: list[dict], obj_a, obj_b, floors: tuple[float, float]) -> dict:
    """What the unsigned `rate` column hides, echoed so the pair's inflection is auditable.

    Every count is a per-response fact (this frontier's sampling), and each names a way a
    rate misleads: a tied denominator's quotient reads as a cliff, a tied numerator's reads
    as one too (the jump against it explodes) while nothing was actually bought, and a mutual
    gain reads as a price. The reading is in `solution_interpreter` → Marginal Analysis
    Interpretation.
    """
    floor_a, floor_b = floors
    degenerate = sum(1 for r in rates if r.get("degenerate"))
    flat = sum(1 for r in rates if r.get("flat"))
    co_improving = sum(1 for r in rates if r.get("co_improvement"))
    notes = []
    if degenerate:
        notes.append(f"`degenerate` rows tie on {obj_a.name} (step below denominator_floor), so "
                     f"their rate is a division artifact — excluded from inflection and "
                     f"steepest_transitions, still counted in summary")
    if flat:
        notes.append(f"`flat` rows tie on {obj_b.name} (step below numerator_floor) — the step "
                     f"bought essentially none of it, so its rate prices nothing; excluded from "
                     f"inflection (as both the jump's baseline and its landing) and from "
                     f"steepest_transitions, still counted in summary")
    if co_improving:
        notes.append(f"`co_improvement` rows GAINED {obj_b.name} rather than paying it, so "
                     f"their rate is a ratio of two gains, not a price")
    return {
        "denominator": obj_a.name,
        "denominator_floor": round(floor_a, 6),
        "degenerate_transitions": degenerate,
        "numerator": obj_b.name,
        "numerator_floor": round(floor_b, 6),
        "flat_transitions": flat,
        "co_improvement_transitions": co_improving,
        # Caption only where there is something to caption — a clean pair says so by its counts.
        **({"note": "; ".join(notes)} if notes else {}),
    }


def _render_marginal_rates(rates: list[dict], obj_a, obj_b, inflection: dict | None,
                           max_rows: int | None = None) -> str:
    """ASCII bar chart of marginal rates between adjacent solutions.

    max_rows: if set, truncate to max_rows centered on inflection point.
    """
    if not rates:
        return ""

    lines = []
    lines.append(f"─── Marginal Rates: {obj_b.name} cost per unit {obj_a.name} ───")
    lines.append("")

    # Scale the bars to the trustworthy rates: a near-tie's quotient can be orders of
    # magnitude above every real rate, and letting it set full scale flattens the rest to
    # nothing. Degenerate bars then peg at full width, which is the honest picture of them.
    trusted = [r["rate"] for r in rates if not r.get("degenerate")]
    rate_values = trusted or [r["rate"] for r in rates]
    max_rate = max(rate_values) if rate_values else 1.0
    if max_rate == 0:
        max_rate = 1.0

    # Determine which rows to show (shared window logic with viz_data)
    start, end = _marginal_window(len(rates), inflection, max_rows)
    show_range = range(start, end)
    truncated = (start, end) != (0, len(rates))

    BAR_W = 30
    if truncated and show_range.start > 0:
        lines.append(f"  ... {show_range.start} earlier transitions omitted")
    for idx in show_range:
        r = rates[idx]
        from_id = r["from_id"]
        to_id = r["to_id"]
        rate = r["rate"]
        filled = max(0, min(BAR_W, round(rate / max_rate * BAR_W)))
        bar = "█" * filled + "░" * (BAR_W - filled)
        marker = " ◀ INFLECTION" if inflection and idx == inflection["position"] else ""
        # The same caveats the payload flags, inline where the number is read.
        if r.get("degenerate"):
            marker += f"  ≈ tie on {obj_a.name}"
        if r.get("flat"):
            marker += f"  ≈ flat on {obj_b.name}"
        if r.get("co_improvement"):
            marker += f"  ↑ {obj_b.name} gained, not paid"
        lines.append(f"  [{from_id}]→[{to_id}]  |{bar}| {_fmt(rate)}{marker}")
    if truncated and show_range.stop < len(rates):
        lines.append(f"  ... {len(rates) - show_range.stop} later transitions omitted")

    lines.append("")
    if inflection:
        lines.append(f"  ◀ Inflection: marginal cost jumps {inflection['jump_factor']:.1f}x at solution {inflection['solution_id']}")
    else:
        lines.append("  No significant inflection detected — marginal costs change gradually.")

    return "\n".join(lines)


# ─── ASCII Visualization Helpers ───


def _normalize(value: float, lo: float, hi: float, direction: str) -> float:
    """Normalize to 0.0–1.0 where 1.0 is 'better'.

    For maximize: higher raw value → higher normalized.
    For minimize: lower raw value → higher normalized (flipped).
    """
    if hi == lo:
        return 0.5
    raw = (value - lo) / (hi - lo)
    return (1.0 - raw) if direction == "minimize" else raw


def _bar(value: float, lo: float, hi: float, width: int = 30) -> str:
    """Render a single normalized bar."""
    if hi == lo:
        filled = width // 2
    else:
        filled = max(0, min(width, round((value - lo) / (hi - lo) * width)))
    return "█" * filled + "░" * (width - filled)


def _fmt(v: float) -> str:
    """Format a number compactly."""
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _render_scatter(solutions, objectives, key_tradeoffs, extreme_solutions,
                    balanced_solution, inflection_candidates=None) -> str:
    """2D ASCII scatter plot of the most conflicting objective pair.

    Grid-bins all solutions as · then overlays labeled points for extremes (●),
    balanced (⚖), and inflection points (◆). Shows frontier shape and clustering.
    """
    WIDTH, HEIGHT = 50, 16

    # Find most negatively correlated pair
    if not key_tradeoffs:
        return "  (scatter requires 2+ objectives with correlation data)"
    best_pair = min(key_tradeoffs, key=lambda t: t["correlation"])
    x_name, y_name = best_pair["objectives"]
    r_val = best_pair["correlation"]

    obj_map = {o.name: o for o in objectives}
    x_obj, y_obj = obj_map[x_name], obj_map[y_name]
    x_dir, y_dir = x_obj.direction.value, y_obj.direction.value
    x_unit = f" ({x_obj.unit})" if x_obj.unit else ""
    y_unit = f" ({y_obj.unit})" if y_obj.unit else ""

    # Compute ranges
    x_vals = [s.objective_values[x_name] for s in solutions]
    y_vals = [s.objective_values[y_name] for s in solutions]
    x_lo, x_hi = min(x_vals), max(x_vals)
    y_lo, y_hi = min(y_vals), max(y_vals)

    # Build grid (row 0 = top = high normalized y)
    grid = [[" "] * WIDTH for _ in range(HEIGHT)]

    def _to_grid(xv, yv):
        nx = _normalize(xv, x_lo, x_hi, x_dir)
        ny = _normalize(yv, y_lo, y_hi, y_dir)
        col = max(0, min(WIDTH - 1, int(nx * (WIDTH - 1))))
        row = max(0, min(HEIGHT - 1, HEIGHT - 1 - int(ny * (HEIGHT - 1))))
        return row, col

    # Plot all solutions as dots
    for s in solutions:
        r, c = _to_grid(s.objective_values[x_name], s.objective_values[y_name])
        if grid[r][c] == " ":
            grid[r][c] = "·"

    # Collect labeled points
    labels_right = []  # (row, col, label)

    # Extremes for the plotted pair
    x_ext_key = f"extreme_{x_name}"
    y_ext_key = f"extreme_{y_name}"
    bal_id = balanced_solution["solution_id"]
    bal_vals = balanced_solution["objective_values"]

    inflection_ids = {k["solution_id"] for k in (inflection_candidates or [])}

    for s in solutions:
        sid = s.solution_id
        xv, yv = s.objective_values[x_name], s.objective_values[y_name]
        r, c = _to_grid(xv, yv)
        if x_ext_key in extreme_solutions and sid == extreme_solutions[x_ext_key]["solution_id"]:
            grid[r][c] = "●"
            labels_right.append((r, c, f"[{sid}] Top {x_name[:12]}"))
        elif y_ext_key in extreme_solutions and sid == extreme_solutions[y_ext_key]["solution_id"]:
            grid[r][c] = "●"
            labels_right.append((r, c, f"[{sid}] Top {y_name[:12]}"))
        elif sid == bal_id:
            grid[r][c] = "⚖"
            labels_right.append((r, c, f"[{sid}] Balanced"))
        elif sid in inflection_ids:
            grid[r][c] = "◆"
            labels_right.append((r, c, f"[{sid}] Inflection"))

    # Render
    lines = []
    lines.append(f"─── Frontier Scatter: {x_name} vs {y_name} (r={r_val:+.2f}) ───")
    lines.append("")

    # Y-axis label
    y_better = "↑ better" if y_dir == "maximize" else "↑ better (lower)"
    y_top = _fmt(y_hi if y_dir == "maximize" else y_lo)
    y_bot = _fmt(y_lo if y_dir == "maximize" else y_hi)
    y_label = f"{y_name}{y_unit}"
    lines.append(f"  {y_label} {y_better}")

    # Build label lookup by row for right-side labels
    row_labels = {}
    for r, c, lbl in labels_right:
        row_labels[r] = lbl

    for r in range(HEIGHT):
        row_str = "".join(grid[r])
        y_tick = ""
        if r == 0:
            y_tick = f"{y_top:>6s}"
        elif r == HEIGHT - 1:
            y_tick = f"{y_bot:>6s}"
        else:
            y_tick = "      "
        label = f"  {row_labels[r]}" if r in row_labels else ""
        lines.append(f"  {y_tick} |{row_str}|{label}")

    # X-axis
    x_better = "→ better" if x_dir == "maximize" else "→ better (lower)"
    x_left = _fmt(x_lo if x_dir == "maximize" else x_hi)
    x_right = _fmt(x_hi if x_dir == "maximize" else x_lo)
    lines.append(f"         +{'─' * WIDTH}+ {x_better}")
    lines.append(f"         {x_left}{' ' * (WIDTH - len(x_left) - len(x_right))}{x_right}")
    x_label = f"{x_name}{x_unit}"
    pad = (WIDTH - len(x_label)) // 2
    lines.append(f"         {' ' * pad}{x_label}")
    lines.append("")
    lines.append("  ● extreme  ⚖ balanced  ◆ inflection  · frontier")

    return "\n".join(lines)


def _render_tradeoffs_viz(result: dict, objectives: list, solutions=None) -> str:
    """Scatter plot of most conflicting pair + correlation summary."""
    parts = []

    # Scatter plot (needs full solutions list)
    if solutions and len(objectives) >= 2 and result.get("key_tradeoffs"):
        parts.append(_render_scatter(
            solutions, objectives,
            result["key_tradeoffs"],
            result["extreme_solutions"],
            result["balanced_solution"],
            inflection_candidates=result.get("inflection_point_candidates"),
        ))

    # Correlation summary — raw values, LLM interprets via solution_interpreter skill
    if result.get("key_tradeoffs"):
        lines = ["", "─── Correlations ───", ""]
        for t in result["key_tradeoffs"]:
            r = t["correlation"]
            if abs(r) < 0.3:
                continue
            o1, o2 = t["objectives"]
            lines.append(f"  {o1} vs {o2}: r={r:+.2f}")
        parts.append("\n".join(lines))

    return "\n".join(parts)


def _render_parallel_coords(solutions_data: list[dict], objectives: list,
                            labels: dict | None = None) -> str:
    """Parallel coordinates: one row per solution, one column per objective.

    Each objective gets a 12-char bar with a 2-char marker showing the
    solution's normalized position. Direction-aware: right = better.

    solutions_data: list of dicts with at least 'objective_values' key.
    labels: maps an identifier to a display label (e.g. {0: "[0] Best Ret"}).
    """
    if not solutions_data or not objectives:
        return ""

    BAR_W = 12
    labels = labels or {}

    # Determine which objectives to show (cap at 6 most differentiating)
    obj_list = list(objectives)
    if len(obj_list) > 6:
        # Pick 6 with highest variance across compared solutions
        variances = []
        for obj in obj_list:
            vals = [sd["objective_values"].get(obj.name, 0) for sd in solutions_data]
            v = max(vals) - min(vals) if vals else 0
            variances.append((v, obj))
        variances.sort(reverse=True)
        obj_list = [o for _, o in variances[:6]]
        omitted = len(objectives) - 6
    else:
        omitted = 0

    # Compute ranges across compared solutions
    ranges = {}
    for obj in obj_list:
        vals = [sd["objective_values"].get(obj.name, 0) for sd in solutions_data]
        ranges[obj.name] = (min(vals), max(vals))

    # Build header
    lines = []
    n_sols = len(solutions_data)
    n_objs = len(obj_list)
    lines.append(f"─── Parallel Coordinates: {n_sols} solutions x {n_objs} objectives ───")
    lines.append("")

    # Objective names row
    name_row = f"  {'Solution':<18s}"
    for obj in obj_list:
        arrow = "→" if obj.direction.value == "maximize" else "←"
        name = obj.name[:10]
        name_row += f"  {name}{arrow:>{BAR_W - len(name)}s}"
    lines.append(name_row)

    # Scale row
    scale_row = f"  {'':<18s}"
    for obj in obj_list:
        lo, hi = ranges[obj.name]
        if obj.direction.value == "minimize":
            scale_row += f"  {_fmt(hi):>{BAR_W // 2}s}─{_fmt(lo):<{BAR_W // 2}s}"
        else:
            scale_row += f"  {_fmt(lo):>{BAR_W // 2}s}─{_fmt(hi):<{BAR_W // 2}s}"
    lines.append(scale_row)

    lines.append(f"  {'─' * 18}{'─' * ((BAR_W + 2) * n_objs)}")

    # Solution rows
    for sd in solutions_data:
        # Determine label
        sid = sd.get("solution_id", sd.get("content_signature", ""))
        label = labels.get(sid, str(sid)[:15])
        label = label[:18]

        row = f"  {label:<18s}"
        for obj in obj_list:
            lo, hi = ranges[obj.name]
            val = sd["objective_values"].get(obj.name, 0)
            n = _normalize(val, lo, hi, obj.direction.value)
            pos = max(0, min(BAR_W - 2, int(n * (BAR_W - 2))))
            bar = list("·" * BAR_W)
            bar[pos] = "█"
            bar[min(pos + 1, BAR_W - 1)] = "█"
            row += f"  {''.join(bar)}"
        lines.append(row)

    if omitted:
        lines.append(f"  ({omitted} more objectives omitted)")

    return "\n".join(lines)


def _render_scenario_viz(result: dict) -> str:
    """ASCII side-by-side range comparison per scenario."""
    lines = []
    per_scenario = result["per_scenario"]
    scenario_names = list(per_scenario.keys())
    if not scenario_names:
        return ""

    # Get all objectives
    obj_names = list(per_scenario[scenario_names[0]]["objective_ranges"].keys())

    # Global min/max per objective across scenarios
    global_ranges = {}
    for obj in obj_names:
        all_lo = min(per_scenario[s]["objective_ranges"][obj]["min"] for s in scenario_names)
        all_hi = max(per_scenario[s]["objective_ranges"][obj]["max"] for s in scenario_names)
        global_ranges[obj] = (all_lo, all_hi)

    lines.append("─── Scenario Range Comparison ───")
    lines.append("")

    for obj in obj_names:
        lo, hi = global_ranges[obj]
        lines.append(f"  {obj}  [{lo:.2f} — {hi:.2f}]")
        for s_name in scenario_names:
            s_lo = per_scenario[s_name]["objective_ranges"][obj]["min"]
            s_hi = per_scenario[s_name]["objective_ranges"][obj]["max"]
            bar_lo = _bar(s_lo, lo, hi, width=30)
            bar_hi = _bar(s_hi, lo, hi, width=30)
            # Show range as a bracket
            if hi == lo:
                pos_lo, pos_hi = 15, 15
            else:
                pos_lo = max(0, min(30, round((s_lo - lo) / (hi - lo) * 30)))
                pos_hi = max(0, min(30, round((s_hi - lo) / (hi - lo) * 30)))
            range_bar = "·" * pos_lo + "█" * max(1, pos_hi - pos_lo) + "·" * (30 - pos_hi)
            lines.append(f"    {s_name:30s} |{range_bar}| {s_lo:.2f}–{s_hi:.2f}")
        lines.append("")

    # Option robustness (frequency + allocation-weighted; binary rows carry frequency only)
    robustness = result.get("option_robustness", [])
    weighted = bool(robustness) and "avg_weight" in robustness[0]
    lines.append("─── Option Robustness (frequency × avg weight) ───" if weighted
                 else "─── Option Robustness (selection frequency) ───")
    if robustness:
        wt_cols = f" {'Wt%':>6s} {'Score':>7s}" if weighted else ""
        lines.append(f"  {'Option':20s} {'Tier':10s} {'Freq':>6s}{wt_cols} {'In':>4s}")
        lines.append(f"  {'─'*20} {'─'*10} {'─'*6}" + (f" {'─'*6} {'─'*7}" if weighted else "") + f" {'─'*4}")
        for r in robustness[:15]:  # Top 15
            wt = f" {r['avg_weight']:5.1f}% {r['importance']:7.1f}" if weighted else ""
            lines.append(
                f"  {r['option']:20s} {r['tier']:10s} "
                f"{r['avg_frequency']:6.1%}{wt} {r['scenarios_present']:3d}/{len(per_scenario)}"
            )
        if len(robustness) > 15:
            lines.append(f"  ... and {len(robustness) - 15} more")
    else:
        lines.append("  No options found in solutions.")
    lines.append("")

    specific = result.get("scenario_specific_options", {})
    if specific:
        lines.append(f"  Scenario-specific: {len(specific)} options (appear in some but not all)")
        for opt, scenarios in sorted(specific.items()):
            lines.append(f"    {opt}: {', '.join(scenarios)}")
        lines.append("")

    # Expected values
    ev = result.get("expected_values", {})
    if ev:
        lines.append("─── Expected Values (probability-weighted best-per-scenario) ───")
        lines.append("  ⚠ These are ideal-point values — no single solution achieves all simultaneously.")
        for obj, val in ev.items():
            lines.append(f"  {obj:25s} {val:.2f}")

    return "\n".join(lines)


def _consistent(problem: Problem, run: Run, remedy: str = "solve run") -> Run:
    """The iterate loop edits the model and peeks at previous results. Serving a run whose
    solutions lack a current objective would KeyError deep inside a consumer — decline in
    words instead, naming the re-solve that actually clears the decline (``remedy`` differs
    by run kind: base solve, run_scenarios, or the exact solver)."""
    if run.solutions:
        have = set(run.solutions[0].objective_values)
        missing = [o.name for o in problem.objectives if o.name not in have]
        if missing:
            raise ValueError(
                f"results are stale — the model's objectives changed since this run "
                f"(no values for: {', '.join(missing)}). Re-run {remedy} to explore the "
                "updated model.")
    return run


def _scenario_stale_block(problem: Problem) -> dict:
    """Stale marker for any read served off the stored scenario runs.

    The base-run convention is a fingerprint comparison (`results_stale`); this is its
    scenario-side twin. A scenario run stamped against an earlier model is still internally
    coherent — its own frontier under its own overrides — so the read is served rather than
    refused, but never as a current finding: the marker leads the payload and names the
    re-solve. (The one read that is NOT internally coherent, the base↔scenario regret join,
    declines instead — see `scenario_regret`.) Empty when the runs match the current model,
    or when they predate the stamp and staleness is unknowable."""
    if not scenario_results_stale(problem):
        return {}
    return {
        "scenario_results_stale": True,
        "scenario_stale_note": (
            "STALE — these scenario runs were solved against an earlier version of the model "
            "(it has been edited since). They are shown as the last solved state, not as "
            "findings about the current model; re-run `solve run_scenarios` before reading "
            "anything from them. Scenario `constraint_overrides` replace the base rules "
            "wholesale, so check whether a base-constraint edit needs restating per scenario."),
    }


def _require_run(problem: Problem, scenario: str | None = None, source: str | None = None) -> Run:
    if scenario:
        # Scenario runs are NSGA-only, so `source` doesn't apply to them.
        if not problem.scenario_run or not problem.scenario_run.scenario_runs:
            raise ValueError("No scenario runs found. Use solve run_scenarios first.")
        if scenario not in problem.scenario_run.scenario_runs:
            available = list(problem.scenario_run.scenario_runs.keys())
            raise ValueError(f"Scenario '{scenario}' not found. Available: {available}")
        run = problem.scenario_run.scenario_runs[scenario]
        if not run.solutions:
            raise ValueError(f"Scenario '{scenario}' has no solutions.")
        return _consistent(problem, run, remedy="solve run_scenarios")
    if source == "exact":
        # Explicitly target the exact-solver frontier (e.g. when both an
        # exploratory `run` and an `exact_run` overlay exist).
        if problem.exact_run is not None and problem.exact_run.solutions:
            return _consistent(problem, problem.exact_run,
                               remedy='the exact solve (solve solver="highs"|"cuopt")')
        raise ValueError("No exact_run found. Solve with an exact solver "
                         "(solver=\"highs\" or \"cuopt\") first.")
    if source not in (None, "run"):
        raise ValueError(f"Unknown source '{source}'. Use 'run' (default) or 'exact'.")
    if problem.run is None:
        # A problem solved only with an exact solver carries its frontier in
        # exact_run (the exact overlay) with no exploratory run — fall back to
        # it so those results are explorable rather than reported as "no run".
        if problem.exact_run is not None and problem.exact_run.solutions:
            return _consistent(problem, problem.exact_run,
                               remedy='the exact solve (solve solver="highs"|"cuopt")')
        raise ValueError("No run found. Use solve first.")
    if not problem.run.solutions:
        raise ValueError("Run has no solutions.")
    return _consistent(problem, problem.run)


def _frontier_provenance(problem: Problem, run: Run, scenario: str | None = None) -> dict:
    """Provenance label for the frontier an explore result was computed over.

    Makes heuristic-vs-exact unambiguous so a heuristic frontier is never silently passed
    off as the exact overlay — e.g. when an explore call's ``source`` is omitted, or stripped
    before it reaches the engine (a stale MCP server/schema can do this). ``kind`` is the
    category (``heuristic`` vs ``exact``); ``solver`` is the precise engine
    (nsga-ii/nsga-iii/highs/cuopt). When the heuristic frontier is served while a base-case
    exact overlay also exists, the label advertises it, so ``source="exact"`` is discoverable
    rather than assumed.
    """
    from solvers import is_exact_solver

    is_exact = is_exact_solver(run.solver)
    prov = {
        "run_id": run.run_id,
        "solver": run.solver,
        "kind": "exact" if is_exact else "heuristic",
    }
    exact_run = problem.exact_run
    if (scenario is None and not is_exact and exact_run is not None
            and exact_run.solutions and exact_run.run_id != run.run_id):
        prov["exact_overlay_available"] = True
        prov["hint"] = ('Heuristic NSGA frontier. An exact-solver overlay exists for this '
                        'problem — pass source="exact" to analyze it.')
    # A scenario frontier served from a run that predates the current model gets the same
    # marker the aggregate scenario reads carry — provenance is exactly where "which model
    # is this?" belongs, so every per-scenario read (tradeoffs / compare / solutions /
    # marginal_analysis) inherits it from one place.
    if scenario is not None:
        prov.update(_scenario_stale_block(problem))
    return prov


def _classify_frontier_shapes(solutions, objectives) -> list[dict]:
    """Classify tradeoff shape (linear/concave/convex/discontinuous) per conflicting pair.

    Only analyzes pairs with negative correlation and >= 5 solutions.
    """
    from scipy.stats import spearmanr

    obj_names = [o.name for o in objectives]
    if len(solutions) < 5 or len(obj_names) < 2:
        return []

    matrix = np.array([
        [s.objective_values[name] for name in obj_names]
        for s in solutions
    ])
    dir_signs = np.array([
        -1.0 if o.direction.value == "minimize" else 1.0
        for o in objectives
    ])
    norm_matrix = matrix * dir_signs
    corr = np.corrcoef(norm_matrix.T)

    shapes = []
    for i in range(len(obj_names)):
        for j in range(i + 1, len(obj_names)):
            r = float(corr[i, j])
            if r >= -0.2:
                continue  # Only conflicting pairs

            obj_a, obj_b = objectives[i], objectives[j]
            reverse_a = obj_a.direction.value == "maximize"
            sorted_sols = sorted(
                solutions,
                key=lambda s: s.objective_values[obj_a.name],
                reverse=reverse_a,
            )

            # Compute marginal rates between adjacent solutions
            rates = []
            spacings = []
            for k in range(len(sorted_sols) - 1):
                s1, s2 = sorted_sols[k], sorted_sols[k + 1]
                delta_a = s2.objective_values[obj_a.name] - s1.objective_values[obj_a.name]
                delta_b = s2.objective_values[obj_b.name] - s1.objective_values[obj_b.name]
                if obj_a.direction.value == "minimize":
                    delta_a = -delta_a
                if obj_b.direction.value == "minimize":
                    delta_b = -delta_b
                rate = abs(delta_b / delta_a) if abs(delta_a) > 1e-9 else 0.0
                rates.append(rate)
                spacing = abs(delta_a)
                spacings.append(spacing)

            if len(rates) < 3:
                continue

            rates_arr = np.array(rates)
            spacings_arr = np.array(spacings)

            # Check for discontinuous: large gaps in objective space (> 3× median spacing)
            median_spacing = np.median(spacings_arr)
            if median_spacing > 1e-9:
                max_gap = spacings_arr.max()
                if max_gap > 3.0 * median_spacing:
                    shapes.append({
                        "objectives": [obj_a.name, obj_b.name],
                        "shape": "discontinuous",
                        "confidence": round(min(float(max_gap / median_spacing) / 10.0, 1.0), 2),
                    })
                    continue

            # Rate trend: Spearman correlation of rate vs position
            positions = np.arange(len(rates_arr))
            rho, _ = spearmanr(positions, rates_arr)

            # Rate stability: coefficient of variation
            rate_mean = rates_arr.mean()
            rate_cv = float(rates_arr.std() / rate_mean) if rate_mean > 1e-9 else 0.0

            if rate_cv < 0.3:
                shape = "linear"
                confidence = round(1.0 - rate_cv, 2)
            elif rho > 0.3:
                shape = "concave"  # rates increasing → diminishing returns
                confidence = round(min(abs(rho), 1.0), 2)
            elif rho < -0.3:
                shape = "convex"  # rates decreasing → diminishing sacrifice
                confidence = round(min(abs(rho), 1.0), 2)
            else:
                shape = "linear"
                confidence = round(max(0.3, 1.0 - rate_cv), 2)

            shapes.append({
                "objectives": [obj_a.name, obj_b.name],
                "shape": shape,
                "confidence": confidence,
            })

    return shapes


def _normalized_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized mutual information in [0, 1] via quantile-binned histogram.

    Uses sqrt(n) bins (rounded, min 3, max 10) and normalizes MI by sqrt(H(X)*H(Y))
    so 0 = independent and 1 = deterministic relationship. Suitable for small
    samples (15-200). Returns 0.0 if either variable is degenerate.
    """
    n = len(x)
    if n < 3:
        return 0.0
    # Degenerate check
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0

    n_bins = int(max(3, min(10, round(np.sqrt(n)))))

    # Quantile binning: each bin gets ~equal count, handles skewed distributions
    def _bin(v: np.ndarray) -> np.ndarray:
        edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        # Nudge uniques to avoid empty bins when many duplicates exist
        edges = np.unique(edges)
        if len(edges) < 3:
            # Not enough distinct values to bin — fall back to binary above/below median
            return (v >= np.median(v)).astype(int)
        # np.digitize places values into bins 1..len(edges); clamp to [0, len(edges)-2]
        idx = np.digitize(v, edges[1:-1], right=False)
        return idx

    bx = _bin(x)
    by = _bin(y)

    # Joint histogram
    joint, _, _ = np.histogram2d(bx, by, bins=[np.unique(bx).size, np.unique(by).size])
    joint = joint / joint.sum() if joint.sum() > 0 else joint
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)

    # MI and entropies (natural log, nats); normalization cancels the unit
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((joint > 0) & (px > 0) & (py > 0), joint / (px * py), 1.0)
        mi = float(np.sum(joint * np.log(ratio)))
        hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
        hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))

    denom = np.sqrt(max(hx, 0.0) * max(hy, 0.0))
    if denom <= 0:
        return 0.0
    nmi = mi / denom
    return float(np.clip(nmi, 0.0, 1.0))


def _objective_redundancy(key_tradeoffs: list[dict], n_solutions: int) -> list[dict]:
    """Classify each objective pair using direction-normalized Pearson + MI.

    Input `correlation` is direction-normalized (see get_tradeoffs): positive r
    means both objectives improve together (redundancy candidate), negative r
    means improving one worsens the other (genuine tradeoff — NOT redundant).

    Classifications:
    - `linear_redundant`: r ≥ +0.7 (both improve together, linearly)
    - `strong_tradeoff`: r ≤ -0.7 (the optimizer's job — don't flag as redundant)
    - `redundant`: MI ≥ 0.7 catches strongly coupled pairs that aren't linearly aligned
    - `nonlinear_dependent`: |r| < 0.3 but MI ≥ 0.4 — non-linear dependence Pearson misses
    - `independent`: otherwise
    """
    out: list[dict] = []
    for entry in key_tradeoffs:
        r = entry.get("correlation")
        mi = entry.get("mutual_info_normalized")
        if r is None:
            continue

        flags: list[str] = []
        classification = "independent"

        if r >= 0.7:
            classification = "linear_redundant"
            flags.append("high_pearson_aligned")
        elif r <= -0.7:
            classification = "strong_tradeoff"
            flags.append("high_pearson_opposed")

        if mi is not None and mi >= 0.7 and classification not in ("linear_redundant", "strong_tradeoff"):
            classification = "redundant"
            flags.append("high_mi")
        if mi is not None and abs(r) < 0.3 and mi >= 0.4:
            classification = "nonlinear_dependent"
            flags.append("pearson_mi_disagreement")

        out.append({
            "objectives": entry["objectives"],
            "classification": classification,
            "pearson": round(r, 2),
            "mutual_info_normalized": mi,
            "flags": flags,
            "mi_reliable": n_solutions >= 15,
        })
    return out


def _binding_analysis(problem: Problem, solutions: list) -> list[dict]:
    """Shadow-price guidance for binding constraints.

    Detects constraints that are binding on a material fraction of the frontier,
    then estimates how much each other objective would shift per unit of slack
    relaxation — the shadow price. Rates are derived from the existing frontier;
    no hardcoded relaxation amounts.

    Supported constraint types: objective_bound, cardinality, group_limit.
    """
    from .models import (
        BoundOperator,
        CardinalityConstraint,
        GroupLimitConstraint,
        ObjectiveBoundConstraint,
    )

    if len(solutions) < 3:
        return []

    out: list[dict] = []
    for c in problem.constraints:
        if isinstance(c, ObjectiveBoundConstraint):
            entry = _binding_objective_bound(c, solutions, problem.objectives)
        elif isinstance(c, CardinalityConstraint):
            entry = _binding_cardinality(c, solutions, problem.objectives)
        elif isinstance(c, GroupLimitConstraint):
            entry = _binding_group_limit(c, solutions, problem.objectives)
        else:
            entry = None
        if entry is not None:
            entries = entry if isinstance(entry, list) else [entry]
            # Echo the constraint's own motive beside its price — "this cap costs X per unit"
            # is a stronger prompt when it can also say what the cap was protecting.
            if getattr(c, "motivated_by", ""):
                for e in entries:
                    e["motivated_by"] = c.motivated_by
            out.extend(entries)
    return out


def _binding_objective_bound(c, solutions, objectives) -> dict | None:
    from .models import BoundOperator

    obj_name = c.objective
    bound = c.value
    values = np.array([s.objective_values.get(obj_name, 0.0) for s in solutions], dtype=float)
    if len(values) < 3:
        return None

    # Near-binding mask: within 5% of bound in the binding direction
    if c.operator == BoundOperator.max:
        threshold = bound * 0.95 if bound > 0 else bound - abs(bound) * 0.05 - 1e-9
        mask = values >= threshold
        op_str = "≤"
    else:
        threshold = bound * 1.05 if bound > 0 else bound + abs(bound) * 0.05 + 1e-9
        mask = values <= threshold
        op_str = "≥"

    binding_count = int(mask.sum())
    if binding_count < 2:
        return None
    binding_fraction = round(binding_count / len(values), 3)

    # Shadow price: regress each other objective on X across near-binding solutions.
    # Slope = dY/dX — how fast Y changes as X changes near the bound.
    # When max-bound: increasing X is relaxation; when min-bound: decreasing X is relaxation.
    x = values[mask]
    shadow_prices: list[dict] = []
    if binding_count >= 3 and x.std() > 1e-9:
        for obj in objectives:
            if obj.name == obj_name:
                continue
            y = np.array([s.objective_values.get(obj.name, 0.0) for s in solutions], dtype=float)[mask]
            if y.std() < 1e-12:
                continue
            slope = float(np.polyfit(x, y, 1)[0])
            # Reframe slope in terms of "per unit of relaxation"
            # max-bound: relax = +ΔX; min-bound: relax = -ΔX (so flip sign)
            if c.operator == BoundOperator.min:
                slope = -slope
            shadow_prices.append({
                "objective": obj.name,
                "slope_per_unit_relaxed": round(slope, 6),
            })

    return {
        "constraint": f"{obj_name} {op_str} {bound}",
        "constraint_type": "objective_bound",
        "binding_fraction": binding_fraction,
        "near_binding_count": binding_count,
        "shadow_prices": shadow_prices,
        "note": None if shadow_prices else "insufficient variation in near-binding solutions to estimate shadow prices",
    }


def _binding_cardinality(c, solutions, objectives) -> dict | None:
    counts = np.array([len(s.selected_options) for s in solutions])
    if len(counts) < 2:
        return None

    # Bounds may be one-sided — an absent side is unbounded, so nothing binds on it.
    at_max = counts == c.max if c.max is not None else np.zeros(len(counts), dtype=bool)
    at_min = counts == c.min if c.min is not None else np.zeros(len(counts), dtype=bool)
    # Determine which side is binding
    if at_max.sum() > 0:
        binding_level, adjacent_level = c.max, c.max - 1
        side = "max"
        op_str = "≤"
    elif at_min.sum() > 0 and c.min > 1:
        binding_level, adjacent_level = c.min, c.min + 1
        side = "min"
        op_str = "≥"
    else:
        return None

    mask_binding = counts == binding_level
    mask_adjacent = counts == adjacent_level
    if mask_binding.sum() == 0:
        return None
    binding_fraction = round(float(mask_binding.sum() / len(counts)), 3)

    shadow_prices: list[dict] = []
    if mask_adjacent.sum() > 0:
        for obj in objectives:
            vals = np.array([s.objective_values.get(obj.name, 0.0) for s in solutions], dtype=float)
            is_max = obj.direction.value == "maximize"
            best_binding = vals[mask_binding].max() if is_max else vals[mask_binding].min()
            best_adjacent = vals[mask_adjacent].max() if is_max else vals[mask_adjacent].min()
            # "Per +1 slot of relaxation": max-bound is +1; min-bound is -1
            delta_per_slot = (best_binding - best_adjacent) if side == "max" else (best_adjacent - best_binding)
            # Flip sign when minimize so positive = improvement
            if not is_max:
                delta_per_slot = -delta_per_slot
            shadow_prices.append({
                "objective": obj.name,
                "gain_per_additional_slot": round(float(delta_per_slot), 4),
            })

    return {
        "constraint": f"cardinality {op_str} {binding_level}",
        "constraint_type": "cardinality",
        "binding_fraction": binding_fraction,
        "near_binding_count": int(mask_binding.sum()),
        "shadow_prices": shadow_prices,
        "note": None if shadow_prices else "no adjacent-cardinality solutions on frontier — cannot estimate gain from +1 slot",
    }


def _binding_group_limit(c, solutions, objectives) -> list[dict] | None:
    """A group can bind at its cap (frontier plans pinned at max) or — with a floor set —
    at its min (plans held up at the floor); report whichever side binds, cap first."""
    group_set = set(c.options)
    counts = np.array([len(group_set.intersection(s.selected_options)) for s in solutions])
    g_min = int(getattr(c, "min", 0) or 0)

    def _entry(bound: int, adjacent: int, label: str, gain_key: str, cap_side: bool) -> dict | None:
        mask_binding = counts == bound
        if mask_binding.sum() == 0:
            return None
        mask_adjacent = counts == adjacent
        shadow_prices: list[dict] = []
        if mask_adjacent.sum() > 0:
            for obj in objectives:
                vals = np.array([s.objective_values.get(obj.name, 0.0) for s in solutions], dtype=float)
                is_max = obj.direction.value == "maximize"
                best_binding = vals[mask_binding].max() if is_max else vals[mask_binding].min()
                best_adjacent = vals[mask_adjacent].max() if is_max else vals[mask_adjacent].min()
                # Cap side: what the last allowed slot bought (binding vs one-below) — the
                # existing "+1 slot" estimate. Floor side: what one member above the floor
                # achieves vs pinned-at-floor — positive means the floor isn't pinching.
                delta = best_binding - best_adjacent if cap_side else best_adjacent - best_binding
                if not is_max:
                    delta = -delta
                shadow_prices.append({"objective": obj.name, gain_key: round(float(delta), 4)})
        return {
            "constraint": label,
            "constraint_type": "group_limit",
            "binding_fraction": round(float(mask_binding.sum() / len(counts)), 3),
            "near_binding_count": int(mask_binding.sum()),
            "shadow_prices": shadow_prices,
            "note": None if shadow_prices else (
                "no adjacent-count solutions on frontier — cannot estimate the gain from "
                "relaxing this bound by one"),
        }

    # Name a handful of members, count the rest — at real problem scale a group can hold
    # hundreds of options, and the full roster would swamp every payload citing this label.
    opts = (", ".join(c.options) if len(c.options) <= 6
            else f"{', '.join(c.options[:5])}, +{len(c.options) - 5} more")
    entries = [_entry(c.max, c.max - 1, f"group_limit({opts}) ≤ {c.max}",
                      "gain_per_additional_slot", cap_side=True)]
    if g_min > 0:
        # Both sides can bind across one frontier (some plans pinned at the cap, others held
        # up at the floor — or min == max pins every plan at both). Report each side that
        # binds; metrics' binding check reports both, and the two surfaces must agree.
        entries.append(_entry(g_min, g_min + 1, f"group_limit({opts}) ≥ {g_min}",
                              "gain_per_extra_member", cap_side=False))
    return [e for e in entries if e is not None] or None


# One tolerance for "is this dual / reduced cost priced" across the sensitivity surface —
# the solver layer rounds both to 6 decimals, so anything nonzero after rounding clears it.
_DUAL_EPS = 1e-9

# The one handoff instruction every scenario seed carries — a single copy so the four
# seed variants (binding, shadow-price, cap, floor) can't drift apart.
_SCENARIO_HOW = ("model update → scenarios (copy this `motivated_by` onto the scenario so "
                 "scenario_results cites it) → solve run_scenarios → explore scenario_results")


def _suggested_scenarios_from_binding(binding: list[dict]) -> list[dict]:
    """Scenario seeds from the frontier-inferred binding analysis — the same duals-rank /
    scenarios-quantify handoff on runs without solver duals. Deterministic seeds only: the
    engine names the lever and the discipline; the increment is the user's call."""
    suggested = []
    for b in sorted(binding, key=lambda x: -x.get("binding_fraction", 0))[:2]:
        suggested.append({
            "motivated_by": f"binding_constraint:{b['constraint']}",
            "vary": (f"only '{b['constraint']}' — note a scenario's constraint_overrides replace "
                     "the WHOLE base constraint set, so restate the unchanged constraints in it"),
            "why": (f"binding on {b['binding_fraction']:.0%} of the frontier — the tightest lever; "
                    "a scenario re-solve quantifies a finite relaxation, not just the local rate"),
            "how": _SCENARIO_HOW,
        })
    return suggested


# --------------------------------------------------------------------------- #
# Sensitivity analysis from exact-solver duals (explainability)
# --------------------------------------------------------------------------- #
def sensitivity_analysis(problem: Problem, solution_id: int | None = None,
                         scenario: str | None = None, source: str | None = None,
                         content_signature: str | None = None) -> dict:
    """Post-optimal sensitivity from exact-solver duals — the explainability view.

    Two reads, in decision language:
      * **where_to_invest** — constraint shadow prices for a reference solution, ranked by
        magnitude, in the solver's cost sense: the price a binding constraint charges the
        optimized objective per unit of tightening (positive = tighter hurts, whichever
        direction the objective optimizes); the largest is where renegotiating buys the most.
      * **near_misses** — options the optimizer left at zero, ranked by reduced cost (smallest
        first = closest to entering). ``capped_options`` lists any option pinned at its
        allocation cap (a negative reduced cost — the cap binds).

    Prefers solver-exact duals (attached to a continuous LP/QP exact run); when none are
    present it falls back to the frontier-inferred binding analysis, clearly tagged, so the
    action always answers. Integer/MILP solutions carry no exact duals."""
    eff_source = source or ("exact" if (problem.exact_run is not None
                                        and problem.exact_run.solutions) else None)
    run = _require_run(problem, scenario, eff_source)
    solutions = run.solutions
    exact = [s for s in solutions
             if getattr(s, "sensitivity", None) and s.sensitivity.source == "solver_exact"]

    if not exact:
        binding = _binding_analysis(problem, solutions)
        out = {
            "source": "frontier_inferred",
            "solver": run.solver,
            "frontier_source": _frontier_provenance(problem, run, scenario),
            "scope": "frontier-regression estimates — no solver-exact duals on this run",
            # Estimate-vs-dual claim calibration lives in the skill section the
            # guidance_pointer retargets ('Binding Analysis') — this note is the upgrade path.
            "note": ("Run solve(solver='highs' or 'cuopt') on a continuous/proportional (QP) "
                     "problem for exact shadow prices + reduced costs. Integer/MILP solutions "
                     "have no exact duals. Showing the frontier-inferred binding analysis below."),
            "binding_analysis": binding,
        }
        suggested = _suggested_scenarios_from_binding(binding)
        if suggested:
            out["suggested_scenarios"] = suggested
        return out

    if solution_id is not None:
        ref = next((s for s in exact if s.solution_id == solution_id), None)
        if ref is None:
            return {"error": f"Solution {solution_id} not found or has no solver-exact sensitivity."}
    elif content_signature is not None:
        # Honor an explicit anchor by content_signature — don't silently fall back to the
        # balanced default (that returns duals for the WRONG solution with no warning).
        ref = next((s for s in exact if s.content_signature == content_signature), None)
        if ref is None:
            return {"error": f"No solver-exact solution matches content_signature "
                    f"'{content_signature[:12]}…'. An exact re-solve changes allocations, so an "
                    "NSGA point's signature often isn't in the exact run — take the anchor's "
                    "solution_id or content_signature from explore(action=\"solutions\", "
                    "source=\"exact\") (or explore(action=\"tradeoffs\", source=\"exact\")) and "
                    "pass that."}
    else:
        ref = _find_balanced(exact, problem.objectives)

    optimized = _optimized_objective(problem, ref.sensitivity)
    from .optimizer import merged_allocation_bounds
    # Merged per option: an option carrying a floor row and a separate cap row is held at
    # the floor the intersection actually applies, not at whichever row came last.
    floors = {opt: lo for opt, (lo, _hi)
              in merged_allocation_bounds(problem.constraints).items() if lo > 0}
    where, near, capped, floored = _format_solution_sensitivity(ref, optimized, floors)
    scope = ("Exact LP/QP duals (continuous path). Shadow price = the price a binding "
             "constraint charges the optimized objective per unit of tightening, in that "
             "constraint's own unit (cost sense: positive means tighter hurts, whichever "
             "direction the objective optimizes; rates on different levers aren't directly "
             "comparable); reduced cost = roughly how far an unheld option must improve to "
             "become competitive (a marginal rate). Undefined for integer/MILP.")
    if optimized:
        scope += (f" The optimized objective here is '{optimized}' — the ε-constraint primary; "
                  "the other objectives enter as floors.")
    # The sensitivity→scenario handoff: duals RANK the levers (instantaneous rates at this
    # optimum); the suggested scenarios QUANTIFY the top ones with a finite-change re-solve.
    # Only PRICED levers earn a suggestion — at a corner/degenerate anchor every epsilon
    # floor can carry a 0.0 dual, and a scenario "motivated by" a zero rate with
    # highest-|rate| framing is self-contradicting; the capped/floored reduced-cost levers
    # below then carry the handoff instead.
    suggested = []
    for w in where:
        if (w["role"] in _FLOOR_ROLES and len(suggested) < 2
                and abs(w["shadow_price"]) > _DUAL_EPS):
            suggested.append({
                "motivated_by": f"shadow_price:{w['lever']}",
                "rate": w["shadow_price"],
                "vary": f"'{w['lever']}' score estimates only (e.g. a ±10% score_adjustment) — "
                        "hold every other anchor fixed",
                "why": (f"'{w['lever']}' is the highest-|rate| lever at this optimum, so the mix is "
                        "most sensitive to its estimates; the dual is a marginal rate, and only a "
                        "scenario re-solve gives the realized effect of a finite change"),
                "how": _SCENARIO_HOW,
            })
    if capped and len(suggested) < 2:
        capped_names = ", ".join(c["option"] for c in capped[:3])
        suggested.append({
            "motivated_by": "reduced_cost:allocation_cap",
            "vary": "the max_allocation cap only (constraint_overrides in a scenario)",
            "why": (f"pinned at the cap: {capped_names} — the cap binds, and a finite raise "
                    "re-solve shows who takes more and what it buys"),
            "how": _SCENARIO_HOW,
        })
    if floored and len(suggested) < 2:
        dearest = floored[0]   # sorted dearest-first; reduced_cost > _DUAL_EPS by construction
        suggested.append({
            "motivated_by": f"reduced_cost:allocation_floor:{dearest['option']}",
            "rate": dearest["reduced_cost"],
            "vary": (f"'{dearest['option']}'s contractual floor only "
                     "(constraint_overrides in a scenario)"),
            "why": (f"'{dearest['option']}' is the dearest floor-pinned commitment at this "
                    f"optimum (~{dearest['reduced_cost']:.4g} per allocation point); a finite "
                    "floor-change re-solve shows what renegotiating the commitment actually buys"),
            "how": _SCENARIO_HOW,
        })

    trend, trend_elided = _shadow_price_trend(exact)
    return {
        "source": "solver_exact",
        "solver": run.solver,
        "frontier_source": _frontier_provenance(problem, run, scenario),
        **({"suggested_scenarios": suggested} if suggested else {}),
        **({"optimized_objective": optimized} if optimized else {}),
        "scope": scope,
        "reference_solution": {
            "solution_id": ref.solution_id,
            "objective_values": ref.objective_values,
            "allocations": ref.allocations,
        },
        "where_to_invest": where,
        "near_misses": near,
        "capped_options": capped,
        **({"floored_options": floored} if floored else {}),
        "frontier_shadow_price_trend": trend,
        **({"frontier_shadow_price_trend_elided": trend_elided} if trend_elided else {}),
        # Narration guidance lives in the skill section the guidance_pointer names
        # ('Exact Sensitivity') — this note is the field legend only.
        "note": ("Shadow prices and reduced costs are reported at the reference solution; the "
                 "trend shows how the swept-constraint shadow price changes along the frontier "
                 "(rising = diminishing returns)."),
    }


_FLOOR_ROLES = ("return_floor", "linear_floor")  # the ε-floor lever roles, QP and LP paths


def _optimized_objective(problem: Problem, sens) -> str | None:
    """Name the ε-constraint primary: the one objective the duals don't floor.

    Each exact scalarization optimizes a single objective (the QP risk term, or
    the LP first objective) and floors the rest — so the floor levers name every
    objective except the optimized one. Returns None when that inference isn't
    unambiguous: an explainability payload should never guess."""
    floored = {sp.name for sp in sens.shadow_prices if sp.role in _FLOOR_ROLES}
    free = [o.name for o in problem.objectives if o.name not in floored]
    return free[0] if len(free) == 1 else None


def _floor_dual(sens, floor_name: str) -> float | None:
    """|shadow_price| of the ``floor_name`` ε-floor lever on one solution's duals — the
    per-point read shared by the marginal-rate basis and the resolution sandwich. Callers
    keep their own policy for a missing lever (all-or-nothing vs monotonicity fallback)."""
    if sens is None:
        return None
    sp = next((x for x in sens.shadow_prices
               if x.role in _FLOOR_ROLES and x.name == floor_name), None)
    return None if sp is None else abs(sp.shadow_price)


def _shadow_interpretation(sp, objective: str | None) -> str:
    # Duals are reported in the solver's COST sense (the minimization it actually ran), so a
    # positive price on a binding row always reads "tightening this hurts the optimized
    # objective" — for a maximize primary that means the value comes off it, never onto it.
    target = f"'{objective}'" if objective else "the optimized objective"
    if sp.role == "budget":
        return (f"price of the total-budget row on {target} "
                "(allocations are normalized to 100%; cost sense — positive means tighter hurts)")
    if sp.role in _FLOOR_ROLES:
        # Fold the sign into the verb — "worsens by ~-1.3" reads backwards next to the
        # cost-sense convention, so a (rare) negative price renders as an improvement.
        verb = "worsens" if sp.shadow_price >= 0 else "improves"
        return (f"marginal cost of '{sp.name}': raising the {sp.name} requirement by one unit "
                f"{verb} {target} by ~{abs(sp.shadow_price):.4g} (in {target}'s own units)")
    return (f"price of this constraint on {target} per unit it is tightened "
            "(cost sense — positive means tighter hurts)")


def _format_solution_sensitivity(s, objective: str | None = None, floors: dict | None = None):
    """(where_to_invest, near_misses, capped, floored) for one solution's solver-exact duals.
    ``floors`` maps option → allocation_bound minimum; an option held AT its floor with a
    positive reduced cost is being carried above what it would earn — the floor's price."""
    sens = s.sensitivity
    floors = floors or {}
    target = f"'{objective}'" if objective else "the optimal mix"
    # `+ 0.0` folds IEEE -0.0 to 0.0 for display (also cleans runs captured before the
    # solver layer normalized).
    where = [{
        "lever": sp.name,
        "role": sp.role,
        "shadow_price": sp.shadow_price + 0.0,
        "interpretation": _shadow_interpretation(sp, objective),
    } for sp in sorted(sens.shadow_prices, key=lambda x: -abs(x.shadow_price))]

    near = [{
        "option": rc.option,
        "reduced_cost": rc.reduced_cost + 0.0,
        "interpretation": (f"unheld — would enter the optimal mix if its marginal contribution "
                           f"to {target} improved by ~{abs(rc.reduced_cost):.4g}"),
    } for rc in sorted((r for r in sens.reduced_costs
                        if r.eligible and r.allocation == 0 and abs(r.reduced_cost) > _DUAL_EPS),
                       key=lambda x: abs(x.reduced_cost))]

    capped = [{
        "option": rc.option,
        "allocation": rc.allocation,
        "reduced_cost": rc.reduced_cost,
        "interpretation": "at its allocation cap — the cap binds; it would take more if allowed",
    } for rc in sorted((r for r in sens.reduced_costs
                        if r.allocation > 0 and r.reduced_cost < -_DUAL_EPS),
                       key=lambda x: x.reduced_cost)]

    floored = [{
        "option": rc.option,
        "allocation": rc.allocation,
        "floor": floors[rc.option],
        "reduced_cost": rc.reduced_cost,
        "interpretation": (f"pinned at its {floors[rc.option]}% contractual floor — held above "
                           f"what it would earn; the floor costs ~{abs(rc.reduced_cost):.4g} per "
                           "unit of allocation (the price of the commitment)"),
    } for rc in sorted((r for r in sens.reduced_costs
                        # +1 tolerance: integer rounding can park a floor-pinned option one
                        # percent above its floor without changing what binds.
                        if r.option in floors and r.allocation <= floors[r.option] + 1
                        and r.reduced_cost > _DUAL_EPS),
                       key=lambda x: -x.reduced_cost)]
    return where, near, capped, floored


# Frontier-length outputs get the same size-aware treatment as scenario_results'
# robustness table: past this many points the trend is downsampled, never dumped whole.
_TREND_MAX_POINTS = 40


def _shadow_price_trend(solutions) -> tuple[list[dict], dict | None]:
    """The swept-objective shadow price across the frontier, by solution_id — the
    diminishing-returns curve made exact. The swept constraint is the QP path's ``return_floor``
    or, on the pure-linear LP path (no quadratic primary), the first objective ``linear_floor``.

    Returns ``(trend, elided)``: at portfolio scale (one row per frontier point) the curve is
    evenly downsampled to ``_TREND_MAX_POINTS`` with both endpoints kept — the shape survives,
    the row dump doesn't — and ``elided`` says what was thinned; small frontiers pass whole."""
    trend = []
    for s in sorted(solutions, key=lambda x: x.solution_id):
        sp = next((x for x in s.sensitivity.shadow_prices
                   if x.role in _FLOOR_ROLES), None)
        if sp is not None:
            trend.append({"solution_id": s.solution_id, "lever": sp.name,
                          "shadow_price": sp.shadow_price + 0.0})
    if len(trend) <= _TREND_MAX_POINTS:
        return trend, None
    total = len(trend)
    keep = sorted({round(i * (total - 1) / (_TREND_MAX_POINTS - 1))
                   for i in range(_TREND_MAX_POINTS)})
    elided = {
        "total_points": total,
        "shown": len(keep),
        "note": ("trend evenly downsampled (endpoints kept) — the diminishing-returns shape "
                 "is intact; ask sensitivity at a specific solution_id for any thinned point"),
    }
    return [trend[i] for i in keep], elided


def _compute_pair_rates(sorted_sols, obj_a, obj_b) -> list[dict]:
    """Compute direction-normalized marginal rates between adjacent sorted solutions.

    Returns list of dicts with from_sol, to_sol, delta_a, delta_b, rate — rich
    enough for both marginal_analysis display and inflection detection.

    ``rate`` is unsigned (|delta_b| per unit delta_a), so the signs stay on the deltas:
    a step where ``delta_b`` is also positive gained B rather than paying it, and
    ``_mark_rate_geometry`` labels that (and a near-tie on either side) before the payload
    presents any rate as a cost.
    """
    rates = []
    for k in range(len(sorted_sols) - 1):
        s1, s2 = sorted_sols[k], sorted_sols[k + 1]
        delta_a = s2.objective_values[obj_a.name] - s1.objective_values[obj_a.name]
        delta_b = s2.objective_values[obj_b.name] - s1.objective_values[obj_b.name]
        if obj_a.direction.value == "minimize":
            delta_a = -delta_a
        if obj_b.direction.value == "minimize":
            delta_b = -delta_b
        rate = abs(delta_b / delta_a) if abs(delta_a) > 1e-9 else 0.0
        rates.append({
            "from_sol": s1,
            "to_sol": s2,
            "delta_a": delta_a,
            "delta_b": delta_b,
            "rate": rate,
        })
    return rates


# A marginal rate is a quotient, so it misreads at BOTH ends: it inflates without bound as
# its denominator step goes to zero, and a jump measured against a ~zero numerator inflates
# the same way. Each side gets a floor — the step below which two adjacent points are a tie
# rather than a move — but the two are derived separately, because the two sides' steps are
# different quantities (see `_flat_floor`).
#
# DENOMINATOR. Two scales, the larger winning: a fixed slice of the objective's range across
# the analyzed frontier, and a fraction of the typical step between adjacent points. The
# range term is the backstop that survives a frontier of mostly-ties; the spacing term binds
# on a densely sampled frontier, where every step is a small slice of the range and only the
# steps far below the typical one are artifacts. Both hold because denominator steps are the
# frontier's SAMPLING — near-uniform, so mean ≈ median and neither term runs away.
_DEGENERACY_RANGE_FRACTION = 0.001
_DEGENERACY_SPACING_FRACTION = 0.25

# NUMERATOR. Numerator steps are the frontier's SHAPE: they swing by orders of magnitude
# between the cheap and expensive stretches, and that swing is the very thing marginal
# analysis measures. Two consequences, both measured:
#
#   * The range term does NOT transfer. `range / spacing = 0.001·n·mean / (frac·median)`, and
#     a skewed distribution has mean/median of 5–50×, so `0.001 × span` overtakes the spacing
#     term at a handful of transitions and then GROWS with sampling density — a fixed slice of
#     a fixed span against steps that shrink as span/n. On a smooth convex frontier it flagged
#     41% of transitions at n=300 and 67% at n=1000. Dropped: the numerator floor carries no
#     range term.
#   * The typical step is the wrong reference on its own. A quarter of it is an ordinary cheap
#     move (the flat side of any honest elbow); a twentieth is "this step bought essentially
#     nothing". And where the cheap side is the MINORITY, the median itself sits in the steep
#     cluster, so even a twentieth of it condemns the whole cheap regime once the cliff is
#     steeper than ~20:1 — which is where the real receipts live (25×–17038×).
#
# So the numerator floor is a twentieth of the typical step, CAPPED at half the pair's own
# tenth-percentile step — read as the k-th smallest step, never the smallest itself (a lone
# artifact would otherwise set the cap that protects it). The cap reads the small end of this
# pair's distribution rather than its centre, so a genuine cheap regime — down to a tenth of
# the transitions — sets its own scale and survives, while an artifact far below even the
# bottom decile is still caught. It also bounds the flag structurally: nothing above half the
# tenth-percentile step can carry it, whatever the sampling density.
#
# BOTH DIRECTIONS OF THAT TENTH, deliberately. A flat patch under a tenth of the transitions
# is an artifact and gets flagged; a cheap stretch over a tenth is a REGIME — the flat side of
# a real elbow — and keeps its rates. The converse is the cost of the same line: a genuine
# notch of m steps goes unflagged once m reaches k, so on a 20-transition frontier a two-step
# notch reads as ordinary. That is the intended trade. Under a tenth is an artifact; over a
# tenth is a shape the user should see, and a guard that erased it would be the worse error.
#
# Both statistics are taken over the steps that MOVED. Exact ties are common on an integer
# objective (adjacent points tie on one axis and pay on another), and counting them would hand
# the bottom of the distribution — and with it the cap — to the very artifact being tested for.
_FLAT_SPACING_FRACTION = 0.05
_FLAT_TAIL_PERCENTILE = 10
_FLAT_TAIL_FRACTION = 0.5


def _degeneracy_floor(rates: list[dict], sorted_sols, obj_a) -> float:
    """Smallest denominator step on which a marginal rate for this pair is trustworthy."""
    vals = [s.objective_values[obj_a.name] for s in sorted_sols]
    span = (max(vals) - min(vals)) if vals else 0.0
    steps = [abs(rr["delta_a"]) for rr in rates]
    typical = float(np.median(steps)) if steps else 0.0
    return max(_DEGENERACY_RANGE_FRACTION * span, _DEGENERACY_SPACING_FRACTION * typical)


def _flat_floor(rates: list[dict]) -> float:
    """Smallest numerator step that still counts as buying something, for this pair.

    A twentieth of the typical step, capped at half the tenth-percentile one. The percentile
    is the k-th smallest step with ``k = max(2, ceil(n/10))`` — never the smallest itself, so
    a single artifact can't set the cap that would spare it, and a cheap regime holding a
    tenth of the transitions still sets the cap from inside its own cluster. Both terms are
    scale-free in the sampling density, so the flag can't grow with frontier size.

    Both are taken over the steps that MOVED. Exact ties belong to no distribution of prices —
    they are the thing being tested for — and letting them fill the bottom would collapse the
    cap onto the artifact beside them, so a frontier with a tenth of its steps tied would stop
    flagging artifacts exactly where the flat stretch is widest.

    Returns 0.0 only when NOTHING moved — no rates, or every step an exact tie. Neither
    reaches here through ``marginal_analysis``: it needs three solutions, and a numerator that
    never moves makes the pair non-conflicting, so it is never analyzed. The branch is kept so
    the function is total, and 0.0 means "every transition is a tie" rather than standing in
    for "no artifacts" — the caller's ``max(floor, 1e-9)`` then flags them all, which is the
    honest reading of a frontier where the pair bought nothing anywhere.
    """
    moved = sorted(s for s in (abs(rr["delta_b"]) for rr in rates) if s > 0.0)
    if not moved:
        return 0.0
    k = max(2, math.ceil(len(moved) * _FLAT_TAIL_PERCENTILE / 100))
    tail = moved[min(k, len(moved)) - 1]
    typical = float(np.median(moved))
    return min(_FLAT_SPACING_FRACTION * typical, _FLAT_TAIL_FRACTION * tail)


def _mark_rate_geometry(rates: list[dict], sorted_sols, obj_a, obj_b) -> tuple[float, float]:
    """Tag each transition with what an unsigned rate hides, and return ``(floor_a, floor_b)``.

    ``degenerate`` — the two points effectively tie on the DENOMINATOR objective (step below
    its floor), so the rate is a division artifact: the price looks like it exploded.
    ``flat`` — the two points effectively tie on the NUMERATOR objective, so the step bought
    essentially nothing of it: the rate is ~0 and any jump measured against it explodes just
    as unboundedly. The mirror of ``degenerate``, and the reader's distinction is which one
    happened — the price exploded, or nothing was bought.
    ``co_improvement`` — the step improved BOTH objectives (possible on a frontier of three or
    more objectives, where the pair's conflict is paid elsewhere), so its rate is a ratio of
    two gains, not a price. All three are set only when true, so a plain row is the ordinary case.

    The two floors are derived separately — a denominator step is sampling, a numerator step
    is shape — so ``obj_b`` enters through ``_flat_floor``'s own construction.
    """
    floor_a = _degeneracy_floor(rates, sorted_sols, obj_a)
    floor_b = _flat_floor(rates)
    for rr in rates:
        if abs(rr["delta_a"]) <= max(floor_a, 1e-9):
            rr["degenerate"] = True
        if abs(rr["delta_b"]) <= max(floor_b, 1e-9):
            rr["flat"] = True
        if rr["delta_b"] > 0:
            rr["co_improvement"] = True
    return floor_a, floor_b


def _trustworthy_rates(rates: list[dict]) -> list[bool]:
    """Which transitions may anchor an inflection — either end of the quotient a real move.

    A rate whose denominator tied (``degenerate``) or whose numerator tied (``flat``) is an
    artifact in BOTH detector roles: as the jump's baseline it manufactures the jump, and as
    its landing it plants a stop-here marker where the user has nothing to stop for.
    """
    return [not (rr.get("degenerate") or rr.get("flat")) for rr in rates]


def _pair_dual_rates(sorted_sols, obj_a, obj_b, problem) -> list[float] | None:
    """Solver-exact slopes for one objective pair: |shadow price| of the ``obj_a`` floor at each
    frontier point — valid when every point carries that dual and ``obj_b`` is the one objective
    the scalarization optimized (then the dual IS d(obj_b)/d(obj_a) at that point). Returns None
    when the exact basis isn't available; callers fall back to secants rather than guessing."""
    if problem is None:
        return None
    vals = []
    for s in sorted_sols:
        if s.sensitivity is None or _optimized_objective(problem, s.sensitivity) != obj_b.name:
            return None
        lam = _floor_dual(s.sensitivity, obj_a.name)
        if lam is None:
            return None
        vals.append(lam)
    return vals


def _pair_rates_with_basis(sorted_sols, obj_a, obj_b, problem=None):
    """Marginal rates for one conflicting pair, on the sharpest basis available: solver-exact
    duals (the slope AT each transition's left endpoint) when the exact path attached them,
    else secants between adjacent points. Returns ``(rates, basis, floors)`` with basis ∈
    'solver_exact_duals' | 'secant' so the payload states its epistemic footing, and ``floors``
    the ``(denominator, numerator)`` steps below which a transition is a tie on that side (see
    ``_mark_rate_geometry``).

    The tie tests ride on the deltas, so they hold on both bases: a dual is an exact slope, but
    an inflection is a marker telling the user where to stop, and two points that tie — on
    either objective — are not two stopping places to choose between.
    """
    rates = _compute_pair_rates(sorted_sols, obj_a, obj_b)
    floors = _mark_rate_geometry(rates, sorted_sols, obj_a, obj_b)
    duals = _pair_dual_rates(sorted_sols, obj_a, obj_b, problem)
    if duals is None:
        return rates, "secant", floors
    for k, rr in enumerate(rates):
        rr["rate"] = duals[k]
    return rates, "solver_exact_duals", floors


def _knee_rationale(obj_a, obj_b, jump_factor: float, co_improvement: bool = False) -> str:
    """One-line, number-anchored knee framing in the user's objective names.

    The price framing holds where the step pays {obj_b} for {obj_a}. Where the step gained
    both (the pair's conflict is settled in some third objective), the jump is in the
    exchange rate's magnitude and the sentence says so instead of naming a price.
    """
    if co_improvement:
        return (f"the exchange rate between {obj_a.name} and {obj_b.name} steepens ~{jump_factor}× "
                f"here — read it as where the pair's relationship turns, not as a price: this step "
                f"GAINS {obj_b.name} alongside {obj_a.name}, so the tradeoff is being paid elsewhere")
    return (f"the defensible default for this pair: beyond this point each extra unit of "
            f"{obj_a.name} costs ~{jump_factor}× more {obj_b.name} per unit than before")


def _detect_inflection(rate_values: list[float], threshold: float = 2.0,
                       eligible: list[bool] | None = None) -> dict | None:
    """Find the largest jump in marginal rate across adjacent rates.

    Returns {position, jump_factor} if max jump >= threshold, else None.
    Position is the index into the rate list where the jump lands (i.e. k+1).

    ``eligible`` (default: all) drops transitions whose rate is a near-tie artifact from BOTH
    roles — the jump's baseline and its landing point — so the detector compares the nearest
    trustworthy neighbours instead and an inflection marks a real cliff. Positions still index
    the full rate list, so callers keep their alignment.
    """
    idxs = [k for k in range(len(rate_values)) if eligible is None or eligible[k]]
    if len(idxs) < 2:
        return None
    max_jump, inflection_idx = 0.0, idxs[0]
    for prev_k, next_k in zip(idxs, idxs[1:]):
        prev_rate = rate_values[prev_k]
        next_rate = rate_values[next_k]
        if prev_rate > 1e-9:
            jump = next_rate / prev_rate
        else:
            jump = next_rate if next_rate > 0 else 0.0
        if jump > max_jump:
            max_jump = jump
            inflection_idx = next_k
    if max_jump >= threshold:
        return {"position": inflection_idx, "jump_factor": round(max_jump, 1)}
    return None


def _conflicting_pair_indices(solutions, objectives) -> list[tuple[int, int, float]]:
    """Return (i, j, correlation) for each conflicting objective pair.

    Direction-normalized: flips minimize objectives so higher = better before
    computing correlation. Only pairs with negative correlation are returned.
    """
    obj_names = [o.name for o in objectives]
    if len(solutions) < 3 or len(obj_names) < 2:
        return []
    matrix = np.array([
        [s.objective_values[name] for name in obj_names]
        for s in solutions
    ])
    dir_signs = np.array([
        -1.0 if o.direction.value == "minimize" else 1.0
        for o in objectives
    ])
    norm_matrix = matrix * dir_signs
    corr = np.corrcoef(norm_matrix.T)
    pairs = []
    for i in range(len(obj_names)):
        for j in range(i + 1, len(obj_names)):
            r = float(corr[i, j])
            if r < 0:
                pairs.append((i, j, r))
    return pairs


def _find_inflection_solutions(solutions, objectives, problem=None) -> list[dict]:
    """Find inflection-point solutions from marginal rate analysis.

    Returns a list of {solution, pair, jump_factor, rationale} for each conflicting
    objective pair that has a detected inflection. Shares rate computation
    and inflection detection with marginal_analysis; passing ``problem`` upgrades
    the slopes to solver-exact duals where the exact path attached them.
    """
    knees = []
    seen_ids = set()
    for i, j, _r in _conflicting_pair_indices(solutions, objectives):
        obj_a, obj_b = objectives[i], objectives[j]
        # Improving-A traversal — see marginal_analysis for why the sort direction matters.
        reverse_a = obj_a.direction.value == "minimize"
        sorted_sols = sorted(
            solutions,
            key=lambda s: s.objective_values[obj_a.name],
            reverse=reverse_a,
        )
        rates, _basis, _floors = _pair_rates_with_basis(sorted_sols, obj_a, obj_b, problem)
        inflection = _detect_inflection(
            [r["rate"] for r in rates], eligible=_trustworthy_rates(rates))
        if inflection is None:
            continue
        landing = rates[inflection["position"]]
        inflection_sol = landing["from_sol"]
        if inflection_sol.solution_id in seen_ids:
            continue
        seen_ids.add(inflection_sol.solution_id)
        co_gain = bool(landing.get("co_improvement"))
        knees.append({
            "solution": inflection_sol,
            "pair": f"{obj_a.name} vs {obj_b.name}",
            "jump_factor": inflection["jump_factor"],
            "co_improvement": co_gain,
            "rationale": _knee_rationale(obj_a, obj_b, inflection["jump_factor"], co_gain),
        })
    return knees


def _find_balanced(solutions, objectives) -> object:
    """Find the solution with minimum normalized Euclidean distance to ideal."""
    obj_names = [o.name for o in objectives]
    directions = [o.direction.value for o in objectives]

    # Build matrix
    matrix = np.array([
        [s.objective_values[name] for name in obj_names]
        for s in solutions
    ])

    # Ideal point: best value per objective
    ideal = np.zeros(len(obj_names))
    for j, d in enumerate(directions):
        ideal[j] = matrix[:, j].max() if d == "maximize" else matrix[:, j].min()

    # Normalize
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    spread = col_max - col_min
    spread[spread == 0] = 1.0  # avoid division by zero

    norm_matrix = (matrix - col_min) / spread
    norm_ideal = (ideal - col_min) / spread

    # For maximize objectives, distance = (ideal - actual); for minimize, (actual - ideal)
    # After normalization, ideal for maximize = 1.0, for minimize = 0.0
    distances = np.zeros(len(solutions))
    for j, d in enumerate(directions):
        if d == "maximize":
            distances += (norm_ideal[j] - norm_matrix[:, j]) ** 2
        else:
            distances += (norm_matrix[:, j] - norm_ideal[j]) ** 2

    distances = np.sqrt(distances)
    best_idx = int(np.argmin(distances))
    return solutions[best_idx]


# ─── Structured viz_data builders (D3-friendly payloads) ─────────────────────
#
# Sibling to the ASCII `_render_*` helpers. Each returns a dict the web UI can
# render as a D3 chart. ASCII still emits to `result["visualization"]` for
# chat/coding-agent surfaces; structured data lands in `result["viz_data"]`.
# Hosts that don't render D3 simply ignore the field.


