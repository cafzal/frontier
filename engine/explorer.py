"""Explorer: navigate the Pareto frontier after solving."""

from __future__ import annotations

import numpy as np

from .models import Aggregation, CuratedSolution, Direction, Problem, Run, _content_signature
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
    for inf in _find_inflection_solutions(solutions, problem.objectives):
        s = inf["solution"]
        if s.solution_id != balanced.solution_id:  # Don't duplicate balanced
            inflection_candidates.append({
                "solution_id": s.solution_id,
                "objective_values": s.objective_values,
                "selected_options": s.selected_options,
                "inflection_pair": inf["pair"],
                "jump_factor": inf["jump_factor"],
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


def compare_solutions(problem: Problem, solution_ids: list[int], scenario: str | None = None, source: str | None = None) -> dict:
    """Side-by-side comparison of specific solutions."""
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
        "solutions": [s.model_dump() for s in selected],
        "shared_options": sorted(shared),
        "differentiating_options": sorted(differentiating),
        "tradeoff_summary": tradeoff_summary,
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


def _feedback_rules(problem: Problem, run_solutions: list, proportional: bool) -> dict:
    """Induce rules separating liked (rating ≥4) from disliked (≤2) solutions.

    No fixed count gate — a rule surfaces only when it MEASURABLY separates the two
    classes (separation × coverage), so weak/no-signal cases return cleanly. A clean
    separator is a candidate latent constraint → route to problem_framing.
    """
    if not problem.feedback:
        return {"available": False,
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
    if not liked or not disliked:
        return {"available": False,
                "note": (f"Need rated solutions on both sides in the current frontier; "
                         f"have {len(liked)} liked, {len(disliked)} disliked.")}
    pos = [_present_options(sig_to_sol[s], proportional) for s in liked]
    neg = [_present_options(sig_to_sol[s], proportional) for s in disliked]
    rules = _greedy_separating_rules([o.name for o in problem.options], pos, neg)
    if not rules:
        return {"available": True, "rules": [],
                "note": "Rated solutions don't separate cleanly on option composition — no reliable rule."}
    return {"available": True, "rules": rules,
            "note": "These are candidate latent constraints — confirm with the user, then route to problem_framing."}


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

    result = {
        "scope": {"set": set_kind, "n_solutions": len(solutions), "approach": approach},
        "option_selection": option_selection,
        "co_occurrence": co_occ,
        "design_principles": _design_principles(solutions, option_selection, co_occ, problem, proportional),
        "clusters": _decision_space_clusters(solutions, problem, proportional),
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
        lines.append("")
        lines.append(f"─── Strategy families ({len(cl)}) ───")
        for c in cl:
            opts = ", ".join(c["defining_options"]) or "(mixed)"
            lines.append(f"  [{c['cluster_id']}] {c['size']} solutions — defining: {opts}")
    fr = result.get("feedback_rules", {})
    if fr.get("rules"):
        lines.append("")
        lines.append("─── Learned from your feedback ───")
        for r in fr["rules"]:
            lines.append(f"  • {r['condition']} → {r['separates']} (sep {r['separation']}, cov {r['coverage']})")
    return "\n".join(lines) if lines else "No composition patterns to show on this set."


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
    if problem.run is None or not problem.run.solutions:
        return {"available": False}
    from .optimizer import score_slate

    scenarios = {s.name: s for s in (problem.scenario_config.scenarios if problem.scenario_config else [])}
    scen_runs = problem.scenario_run.scenario_runs
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
    for s in base:
        feasible_all = True
        for name in scen_runs:
            ck = (s.content_signature, name)
            if ck not in cache:
                cache[ck] = score_slate(problem, s.selected_options, s.allocations, scenarios.get(name))
            if not cache[ck]["feasible"]:
                feasible_all = False
        by_scen = {name: round(max((regret_for(s, name, ob) for ob in objs), default=0.0), 3) for name in scen_runs}
        max_reg = max(by_scen.values(), default=0.0)
        mean_reg = round(sum(by_scen.values()) / len(by_scen), 3) if by_scen else None
        per_solution.append({
            "solution_id": s.solution_id, "content_signature": s.content_signature,
            "max_regret": round(max_reg, 3), "mean_regret": mean_reg,
            "by_scenario": by_scen, "feasible_in_all": feasible_all,
        })
    per_solution.sort(key=lambda x: x["max_regret"])

    per_objective = {}
    for ob in objs:
        best_sid, best_val = None, None
        for s in base:
            worst = max((regret_for(s, name, ob) for name in scen_runs), default=0.0)
            if best_val is None or worst < best_val:
                best_val, best_sid = worst, s.solution_id
        if best_sid is not None:
            per_objective[ob.name] = {"min_max_regret": round(best_val, 3), "achieved_by_solution_id": best_sid}

    # per_solution is sorted ascending by max_regret, so the metric is saturated exactly
    # when the BEST solution already hits 1.0 — every base solution is infeasible or fully
    # dominated in some scenario, and "minimizes worst-case regret" picks among an all-1.0
    # tie. Don't nominate a winner from a meaningless tie; name the saturating scenarios.
    saturated = bool(per_solution) and per_solution[0]["max_regret"] >= 1.0
    minimax = per_solution[0] if per_solution and not saturated else None
    result = {
        "available": True,
        "method": "scenario_minimax",
        "normalization": "per_objective_range",
        "per_objective": per_objective,
        "per_solution": per_solution[:20],
        "per_solution_total": len(per_solution),
        "minimax_choice": ({"solution_id": minimax["solution_id"],
                            "content_signature": minimax["content_signature"],
                            "max_regret": minimax["max_regret"]} if minimax else None),
        "note": ("regret = best-achievable-in-scenario minus this solution re-evaluated there, "
                 "normalized per objective range; minimax_choice minimizes worst-case regret. "
                 "per_solution lists the lowest-max-regret solutions "
                 "(see per_solution_total for the full count)."),
    }
    if saturated:
        sat_scens = [name for name in scen_runs
                     if all(ps["by_scenario"].get(name) == 1.0 for ps in per_solution)]
        result["saturated"] = True
        result["saturation_note"] = (
            "minimax not informative — every base-frontier solution hits max regret "
            f"(infeasible or fully dominated) under: {', '.join(sat_scens) or 'some scenario'}. "
            "minimax_choice is omitted. Pick a hedge from that scenario's own frontier "
            "(explore tradeoffs scenario=<name>) or fold its constraints into the base "
            "model and re-solve."
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

    return {
        "runs_compared": [r.run_id for r in selected_runs],
        "criteria_diffs": criteria_diffs,
        "frontier_diffs": frontier_diffs,
        "option_coverage": option_coverage,
    }


def _coverage_gain(N: np.ndarray, E: np.ndarray, seed: int = 42) -> dict | None:
    """Hypervolume the exact overlay reclaims over the NSGA frontier alone — the *magnitude*
    companion to the dominance *count*, and the coverage value that grows with problem size.

    Both fronts arrive in minimize-space. Normalize the *combined* set into one shared [0, 1] box
    so a single reference point is fair to both (hypervolume is reference-point dependent), then
    Monte-Carlo the dominated volume of NSGA alone vs NSGA∪exact against the same reference and
    sample. Same seed → identical samples → ``combined ≥ nsga`` exactly, so the reclaimed volume
    is never a sampling artifact. Returns ``None`` when the combined front is degenerate (a flat
    axis), where coverage is undefined. Volumes are normalized to the box (the engine's
    ``hypervolume_normalized`` convention), so they read on the same 0–1 scale as a run's quality.
    """
    from engine.optimizer import _approx_hypervolume

    P = np.vstack([N, E])
    f_min = P.min(axis=0)
    spread = P.max(axis=0) - f_min
    if not np.all(spread > 0):
        return None
    ref = np.ones(P.shape[1]) * 1.1
    hv_box = float(np.prod(ref))
    hv_nsga = _approx_hypervolume((N - f_min) / spread, ref, seed=seed) / hv_box
    hv_comb = _approx_hypervolume((P - f_min) / spread, ref, seed=seed) / hv_box
    reclaimed = max(0.0, hv_comb - hv_nsga)  # combined ≥ nsga by construction (shared seed/samples); the clamp only absorbs FP noise
    return {
        "nsga_hypervolume": round(hv_nsga, 4),
        "combined_hypervolume": round(hv_comb, 4),
        "exact_reclaims": round(reclaimed, 4),
        "reclaimed_fraction": round(reclaimed / hv_comb, 4) if hv_comb > 0 else 0.0,
    }


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

    Returns a JSON-friendly dict (the MCP ``explore certify`` payload).
    """
    if not nsga_run.solutions or not exact_run.solutions:
        raise ValueError("certify needs both runs to have solutions (run NSGA and an exact solver first).")

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
    exact_front = np.array([i for i in range(len(E))
                            if not any(_dominates_min(E[j], E[i]) for j in range(len(E)) if j != i)])
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

    parts = []
    if dominated_idx:
        parts.append(f"exact audits {len(dominated_idx)}/{len(N)} NSGA points as dominated (heuristic slack)")
    if headline:
        c = corners[headline]
        corner = "risk corner" if c["is_risk_corner"] else "corner"
        parts.append(f"sharpens the {headline} {corner} {c['nsga_best']}→{c['exact_best']}")
    if not parts:
        parts.append("NSGA already matches the exact frontier here — exact confirms, adds no new points")
    under = [n for n, c in corners.items() if c["status"] == "under-sampled"]
    if under:
        parts.append(f"under-samples {', '.join(under)} (EA budget, not a limit — raise the budget to close it)")
    recommendation = "; ".join(parts) + "."

    # Hand the agent onward (the certificate is a step, not a terminus). Shape-branched: a
    # continuous/QP overlay also carries solver duals (`explore sensitivity`); a MILP overlay
    # is integer, so duals don't exist — point at the frontier-derived binding_analysis instead.
    anchor = f"the sharpened {headline} corner" if headline else "the certified frontier"
    if problem.approach.value == "binary":
        next_steps = (
            f"Present {anchor} to the user as the decision anchor — every point is now optimal, "
            "not heuristic. Integer/MILP solutions carry no exact duals; use `binding_analysis` "
            "from `explore tradeoffs` for shadow-price intuition. Read this certificate with the "
            "`solution_interpreter` skill ('Reading the Certificate')."
        )
    else:
        next_steps = (
            f"Present {anchor} to the user as the decision anchor, and navigate the exact overlay "
            "with `explore … source=\"exact\"`. On this continuous/QP problem, `explore sensitivity` "
            "adds solver-exact shadow prices + reduced costs (the explainability layer). Read this "
            "certificate with the `solution_interpreter` skill ('Reading the Certificate')."
        )

    coverage = _coverage_gain(N, Eref)  # the cleaned exact front (as the dominance audit uses), so integer-rounding artifacts can't rescale the shared box

    return {
        "nsga_run_id": nsga_run.run_id,
        "exact_run_id": exact_run.run_id,
        "exact_solver": exact_run.solver,
        "exact_certified": bool(exact_run.exact),
        "nsga_count": len(N),
        "exact_count": len(E),
        "dominance_audit": {
            "nsga_dominated_by_exact": len(dominated_idx),
            "nsga_dominated_fraction": round(len(dominated_idx) / len(N), 4) if len(N) else 0.0,
            "examples": examples,
        },
        "coverage": coverage,
        "invariant": {
            "holds": exact_dominated == 0,
            "exact_dominated_by_nsga": exact_dominated,
            "note": ("a few exact points edged out by NSGA — the integer rounding of the continuous QP "
                     "optimum to whole-percent allocations, not a heuristic beating the exact solve "
                     "(MILP corners, integer by construction, never show this)"
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
                            "(the exact version of validate's pre-solve check). Relax the tightest "
                            "constraint and re-probe before solving.",
        "violated": "The property does NOT hold across the feasible space — the witness is a concrete "
                    "counterexample. Treat the property as an aspiration, not a guarantee; encode it as a "
                    "hard constraint if it must hold.",
        "holds": "Proven: the property holds for EVERY feasible plan — the negation is infeasible across "
                 "the whole feasible region, not a sampled subset. A guarantee you can put in front of a "
                 "stakeholder, not a spot-check.",
        "holds_vacuously": "The property holds only vacuously — there are no feasible plans for it to apply "
                           "to. Probe feasibility first (audit with no property).",
        "inconclusive": "INCONCLUSIVE — the solve hit its time limit without a verdict. Do not read this "
                        "as a pass.",
    }
    # An unfit shape / missing backend raises ValueError upstream (→ a tool error), never reaching
    # here — so every verdict in `text` is an analytical outcome, framed the same way.
    return {"recommendation": text.get(verdict, ""), "next_steps": read}


def audit_property(problem: Problem, property_dict: dict | None) -> dict:
    """MCP payload for `explore audit` — the witness / feasibility auditor (sibling of certify).

    Parses the optional property (a ``Constraint`` dict — same vocabulary as model constraints),
    runs the engine audit over the whole feasible space, and frames the verdict for presentation.
    No prior solve required: audit reasons about the model's feasible region directly, so it can
    feasibility-probe *before* spending a solve."""
    from engine import optimizer

    prop = None
    if property_dict is not None:
        if not isinstance(property_dict, dict):
            raise ValueError(
                "audit `property` must be a constraint object, e.g. "
                '{"type": "objective_bound", "objective": "Cost", "operator": "max", "value": 100}.')
        from pydantic import TypeAdapter, ValidationError

        from engine.models import Constraint
        try:
            prop = TypeAdapter(Constraint).validate_python(property_dict)
        except ValidationError as e:
            errs = e.errors()
            raise ValueError(
                f"audit `property` isn't a valid constraint: {errs[0].get('msg', e) if errs else e}")

    result = optimizer.audit(problem, prop=prop, solver="highs")
    result["audited"] = ("feasibility of the current constraint set" if prop is None else property_dict)
    result.update(_audit_framing(result["verdict"]))
    return result


def _constraint_key(c: dict) -> str:
    """Stable string key for a constraint dict, for comparison."""
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
        return f"group_limit:{','.join(c.get('options', []))}:{c.get('max')}"
    return str(c)


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

    # Column layout: name, signature, objective values, then either selected options or allocations
    headers = ["name", "content_signature", *obj_names]
    if use_allocations:
        headers += [f"alloc:{opt}" for opt in opt_names]
    else:
        headers += ["selected_options"]

    rows: list[list[str]] = []
    for cs in curated:
        row = [cs.custom_name or "", cs.content_signature]
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

    scenario_runs = problem.scenario_run.scenario_runs
    scenarios = {s.name: s for s in problem.scenario_config.scenarios}
    opt_names = [o.name for o in problem.options]
    obj_names = [o.name for o in problem.objectives]

    # Per-scenario summaries
    per_scenario = {}
    for name, run in scenario_runs.items():
        ranges = {}
        for obj in obj_names:
            vals = [s.objective_values.get(obj, 0) for s in run.solutions]
            if vals:
                ranges[obj] = {"min": min(vals), "max": max(vals)}
        per_scenario[name] = {
            "solution_count": len(run.solutions),
            "objective_ranges": ranges,
        }

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

    # Aggregate across scenarios: avg frequency, avg weight, importance score
    option_robustness = []
    scenario_specific = {}
    for opt in opt_names:
        freqs = [per_scenario_freq[s].get(opt, 0) for s in all_scenario_names]
        weights = [per_scenario_avg_weight[s].get(opt, 0) for s in all_scenario_names]
        present_in = [s for s in all_scenario_names if per_scenario_freq[s].get(opt, 0) > 0]
        avg_freq = sum(freqs) / len(freqs) if freqs else 0
        avg_weight = sum(w for w in weights if w > 0) / max(len([w for w in weights if w > 0]), 1)
        importance = round(avg_freq * avg_weight, 2)

        if avg_freq > 0:
            # Tier: core (>50% freq in all scenarios), common (>25% or in all),
            # marginal (<25% or missing from some)
            min_freq = min(freqs)
            if min_freq > 0.5:
                tier = "core"
            elif min_freq > 0.25 or len(present_in) == len(all_scenario_names):
                tier = "common"
            else:
                tier = "marginal"

            option_robustness.append({
                "option": opt,
                "avg_frequency": round(avg_freq, 3),
                "avg_weight": round(avg_weight, 1),
                "importance": importance,
                "tier": tier,
                "scenarios_present": len(present_in),
            })

        if 0 < len(present_in) < len(all_scenario_names):
            scenario_specific[opt] = present_in

    # Sort by importance descending
    option_robustness.sort(key=lambda x: x["importance"], reverse=True)
    robust_options = [r["option"] for r in option_robustness if r["tier"] == "core"]

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
        "per_scenario": per_scenario,
        "robust_options": sorted(robust_options),
        "option_robustness": option_robustness,
        "scenario_specific_options": scenario_specific,
        "expected_values": expected_values,
        "scenario_risk": scenario_risk,
        "cvar_alpha": alpha,
        "weighting": "probability" if has_probabilities else "equal",
    }
    # Minimax-regret robustness lens, beside CVaR/worst-case (a distinct question:
    # "how much worse than the best I could have chosen, in hindsight?").
    result["regret"] = scenario_regret(problem)
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

        # Sort solutions by objective A (in "better" direction)
        reverse_a = obj_a.direction.value == "maximize"
        sorted_sols = sorted(
            solutions,
            key=lambda s: s.objective_values[obj_a.name],
            reverse=reverse_a,
        )

        # Compute marginal rates between adjacent solutions
        raw_rates = _compute_pair_rates(sorted_sols, obj_a, obj_b)
        rates = [
            {
                "from_id": rr["from_sol"].solution_id,
                "to_id": rr["to_sol"].solution_id,
                f"delta_{obj_a.name}": round(rr["delta_a"], 4),
                f"delta_{obj_b.name}": round(rr["delta_b"], 4),
                "rate": round(rr["rate"], 4),
            }
            for rr in raw_rates
        ]

        # Detect inflection: largest jump in marginal rate
        detected = _detect_inflection([rr["rate"] for rr in raw_rates])
        if detected is None:
            inflection = None
        else:
            inflection = {
                "solution_id": raw_rates[detected["position"]]["from_sol"].solution_id,
                "position": detected["position"],
                "jump_factor": detected["jump_factor"],
            }

        pair_result = {
            "objectives": [obj_a.name, obj_b.name],
            "correlation": round(r, 2),
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
            # Summary: stats + top-5 steepest + truncated viz
            rate_values = [r["rate"] for r in rates]
            pair_result["summary"] = {
                "total_transitions": len(rates),
                "rate_min": round(min(rate_values), 4) if rate_values else 0,
                "rate_max": round(max(rate_values), 4) if rate_values else 0,
                "rate_median": round(float(np.median(rate_values)), 4) if rate_values else 0,
            }
            # Top-5 steepest transitions (most expensive tradeoff steps)
            top_rates = sorted(rates, key=lambda r: r["rate"], reverse=True)[:5]
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

    rate_values = [r["rate"] for r in rates]
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

    # Option robustness (frequency + allocation-weighted)
    robustness = result.get("option_robustness", [])
    lines.append("─── Option Robustness (frequency × avg weight) ───")
    if robustness:
        lines.append(f"  {'Option':20s} {'Tier':10s} {'Freq':>6s} {'Wt%':>6s} {'Score':>7s} {'In':>4s}")
        lines.append(f"  {'─'*20} {'─'*10} {'─'*6} {'─'*6} {'─'*7} {'─'*4}")
        for r in robustness[:15]:  # Top 15
            lines.append(
                f"  {r['option']:20s} {r['tier']:10s} "
                f"{r['avg_frequency']:6.1%} {r['avg_weight']:5.1f}% "
                f"{r['importance']:7.1f} {r['scenarios_present']:3d}/{len(per_scenario)}"
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
        return run
    if source == "exact":
        # Explicitly target the exact-solver frontier (e.g. when both an
        # exploratory `run` and an `exact_run` overlay exist).
        if problem.exact_run is not None and problem.exact_run.solutions:
            return problem.exact_run
        raise ValueError("No exact_run found. Solve with an exact solver "
                         "(solver=\"highs\" or \"cuopt\") first.")
    if source not in (None, "run"):
        raise ValueError(f"Unknown source '{source}'. Use 'run' (default) or 'exact'.")
    if problem.run is None:
        # A problem solved only with an exact solver carries its frontier in
        # exact_run (the exact overlay) with no exploratory run — fall back to
        # it so those results are explorable rather than reported as "no run".
        if problem.exact_run is not None and problem.exact_run.solutions:
            return problem.exact_run
        raise ValueError("No run found. Use solve first.")
    if not problem.run.solutions:
        raise ValueError("Run has no solutions.")
    return problem.run


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
            out.append(entry)
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

    at_max = counts == c.max
    at_min = counts == c.min
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


def _binding_group_limit(c, solutions, objectives) -> dict | None:
    group_set = set(c.options)
    counts = np.array([len(group_set.intersection(s.selected_options)) for s in solutions])
    mask_binding = counts == c.max
    if mask_binding.sum() == 0:
        return None
    mask_adjacent = counts == c.max - 1
    binding_fraction = round(float(mask_binding.sum() / len(counts)), 3)

    shadow_prices: list[dict] = []
    if mask_adjacent.sum() > 0:
        for obj in objectives:
            vals = np.array([s.objective_values.get(obj.name, 0.0) for s in solutions], dtype=float)
            is_max = obj.direction.value == "maximize"
            best_binding = vals[mask_binding].max() if is_max else vals[mask_binding].min()
            best_adjacent = vals[mask_adjacent].max() if is_max else vals[mask_adjacent].min()
            delta = best_binding - best_adjacent
            if not is_max:
                delta = -delta
            shadow_prices.append({
                "objective": obj.name,
                "gain_per_additional_slot": round(float(delta), 4),
            })

    return {
        "constraint": f"group_limit({', '.join(c.options)}) ≤ {c.max}",
        "constraint_type": "group_limit",
        "binding_fraction": binding_fraction,
        "near_binding_count": int(mask_binding.sum()),
        "shadow_prices": shadow_prices,
        "note": None if shadow_prices else "no adjacent-count solutions on frontier — cannot estimate gain from +1 slot",
    }


# --------------------------------------------------------------------------- #
# Sensitivity analysis from exact-solver duals (explainability)
# --------------------------------------------------------------------------- #
def sensitivity_analysis(problem: Problem, solution_id: int | None = None,
                         scenario: str | None = None, source: str | None = None) -> dict:
    """Post-optimal sensitivity from exact-solver duals — the explainability view.

    Two reads, in decision language:
      * **where_to_invest** — constraint shadow prices for a reference solution, ranked by
        magnitude. A shadow price is the marginal change in the optimized objective per unit a
        binding constraint is relaxed; the largest is where relaxing buys the most.
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
        return {
            "source": "frontier_inferred",
            "solver": run.solver,
            "frontier_source": _frontier_provenance(problem, run, scenario),
            "scope": "frontier-regression estimates — no solver-exact duals on this run",
            "note": ("Run solve(solver='highs' or 'cuopt') on a continuous/proportional (QP) "
                     "problem for exact shadow prices + reduced costs. Integer/MILP solutions "
                     "have no exact duals. Showing the frontier-inferred binding analysis below."),
            "binding_analysis": _binding_analysis(problem, solutions),
            "next_steps": ("Present this as a frontier-inferred estimate, not a solver dual — read it "
                           "with the `solution_interpreter` skill ('Binding Analysis')."),
        }

    if solution_id is not None:
        ref = next((s for s in exact if s.solution_id == solution_id), None)
        if ref is None:
            return {"error": f"Solution {solution_id} not found or has no solver-exact sensitivity."}
    else:
        ref = _find_balanced(exact, problem.objectives)

    optimized = _optimized_objective(problem, ref.sensitivity)
    where, near, capped = _format_solution_sensitivity(ref, optimized)
    scope = ("Exact LP/QP duals (continuous path). Shadow price = marginal change in the "
             "optimized objective per unit a binding constraint is relaxed; reduced cost = "
             "how far an unheld option must improve to enter. Undefined for integer/MILP.")
    if optimized:
        scope += (f" The optimized objective here is '{optimized}' — the ε-constraint primary; "
                  "the other objectives enter as floors.")
    return {
        "source": "solver_exact",
        "solver": run.solver,
        "frontier_source": _frontier_provenance(problem, run, scenario),
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
        "frontier_shadow_price_trend": _shadow_price_trend(exact),
        "note": ("Shadow prices and reduced costs are reported at the reference solution; the "
                 "trend shows how the swept-constraint shadow price changes along the frontier "
                 "(rising = diminishing returns)."),
        "next_steps": ("Read these duals with the `solution_interpreter` skill ('Exact Sensitivity'): "
                       "the top `where_to_invest` shadow price is the highest-leverage constraint to "
                       "renegotiate; the smallest `near_misses` reduced cost is the option closest to "
                       "entering (a re-scoring / cap-relaxation prompt). Anchor every number in the "
                       "reference solution; route a persistent near-miss back to problem_framing."),
    }


def _optimized_objective(problem: Problem, sens) -> str | None:
    """Name the ε-constraint primary: the one objective the duals don't floor.

    Each exact scalarization optimizes a single objective (the QP risk term, or
    the LP first objective) and floors the rest — so the floor levers name every
    objective except the optimized one. Returns None when that inference isn't
    unambiguous: an explainability payload should never guess."""
    floored = {sp.name for sp in sens.shadow_prices
               if sp.role in ("return_floor", "linear_floor")}
    free = [o.name for o in problem.objectives if o.name not in floored]
    return free[0] if len(free) == 1 else None


def _shadow_interpretation(sp, objective: str | None) -> str:
    target = f"'{objective}'" if objective else "the optimized objective"
    if sp.role == "budget":
        return (f"marginal change in {target} per unit of total budget "
                "(allocations are normalized to 100%)")
    if sp.role in ("return_floor", "linear_floor"):
        return (f"marginal cost of '{sp.name}': raising the {sp.name} requirement by one unit "
                f"shifts {target} by ~{sp.shadow_price:.4g}")
    return f"marginal change in {target} per unit this constraint is relaxed"


def _format_solution_sensitivity(s, objective: str | None = None):
    """(where_to_invest, near_misses, capped) for one solution's solver-exact duals."""
    sens = s.sensitivity
    target = f"'{objective}'" if objective else "the optimal mix"
    where = [{
        "lever": sp.name,
        "role": sp.role,
        "shadow_price": sp.shadow_price,
        "interpretation": _shadow_interpretation(sp, objective),
    } for sp in sorted(sens.shadow_prices, key=lambda x: -abs(x.shadow_price))]

    near = [{
        "option": rc.option,
        "reduced_cost": rc.reduced_cost,
        "interpretation": (f"unheld — would enter the optimal mix if its marginal contribution "
                           f"to {target} improved by ~{abs(rc.reduced_cost):.4g}"),
    } for rc in sorted((r for r in sens.reduced_costs
                        if r.eligible and r.allocation == 0 and abs(r.reduced_cost) > 1e-9),
                       key=lambda x: abs(x.reduced_cost))]

    capped = [{
        "option": rc.option,
        "allocation": rc.allocation,
        "reduced_cost": rc.reduced_cost,
        "interpretation": "at its allocation cap — the cap binds; it would take more if allowed",
    } for rc in sorted((r for r in sens.reduced_costs
                        if r.allocation > 0 and r.reduced_cost < -1e-9),
                       key=lambda x: x.reduced_cost)]
    return where, near, capped


def _shadow_price_trend(solutions) -> list[dict]:
    """The swept-objective shadow price across the frontier, by solution_id — the
    diminishing-returns curve made exact. The swept constraint is the QP path's ``return_floor``
    or, on the pure-linear LP path (no quadratic primary), the first objective ``linear_floor``."""
    trend = []
    for s in sorted(solutions, key=lambda x: x.solution_id):
        sp = next((x for x in s.sensitivity.shadow_prices
                   if x.role in ("return_floor", "linear_floor")), None)
        if sp is not None:
            trend.append({"solution_id": s.solution_id, "lever": sp.name,
                          "shadow_price": sp.shadow_price})
    return trend


def _compute_pair_rates(sorted_sols, obj_a, obj_b) -> list[dict]:
    """Compute direction-normalized marginal rates between adjacent sorted solutions.

    Returns list of dicts with from_sol, to_sol, delta_a, delta_b, rate — rich
    enough for both marginal_analysis display and inflection detection.
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


def _detect_inflection(rate_values: list[float], threshold: float = 2.0) -> dict | None:
    """Find the largest jump in marginal rate across adjacent rates.

    Returns {position, jump_factor} if max jump >= threshold, else None.
    Position is the index into the rate list where the jump lands (i.e. k+1).
    """
    if len(rate_values) < 2:
        return None
    max_jump, inflection_idx = 0.0, 0
    for k in range(len(rate_values) - 1):
        prev_rate = rate_values[k]
        next_rate = rate_values[k + 1]
        if prev_rate > 1e-9:
            jump = next_rate / prev_rate
        else:
            jump = next_rate if next_rate > 0 else 0.0
        if jump > max_jump:
            max_jump = jump
            inflection_idx = k + 1
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


def _find_inflection_solutions(solutions, objectives) -> list[dict]:
    """Find inflection-point solutions from marginal rate analysis.

    Returns a list of {solution, pair, jump_factor} for each conflicting
    objective pair that has a detected inflection. Shares rate computation
    and inflection detection with marginal_analysis.
    """
    knees = []
    seen_ids = set()
    for i, j, _r in _conflicting_pair_indices(solutions, objectives):
        obj_a, obj_b = objectives[i], objectives[j]
        reverse_a = obj_a.direction.value == "maximize"
        sorted_sols = sorted(
            solutions,
            key=lambda s: s.objective_values[obj_a.name],
            reverse=reverse_a,
        )
        rates = _compute_pair_rates(sorted_sols, obj_a, obj_b)
        inflection = _detect_inflection([r["rate"] for r in rates])
        if inflection is None:
            continue
        inflection_sol = rates[inflection["position"]]["from_sol"]
        if inflection_sol.solution_id in seen_ids:
            continue
        seen_ids.add(inflection_sol.solution_id)
        knees.append({
            "solution": inflection_sol,
            "pair": f"{obj_a.name} vs {obj_b.name}",
            "jump_factor": inflection["jump_factor"],
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


