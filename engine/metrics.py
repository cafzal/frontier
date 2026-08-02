"""Lightweight eval metrics for Frontier Phase 0.

Two layers:
  1. Automated checkpoints — deterministic signals from problem state
  2. Diagnostics — patterns that suggest structural issues

These are computed from Problem state, not LLM-generated.
The MCP server includes them in tool responses at natural checkpoints.
"""

from __future__ import annotations

from collections import Counter
from statistics import variance

from .models import (
    SCORE_MATRIX_MSG,
    BoundOperator,
    CardinalityConstraint,
    Direction,
    ForceExcludeConstraint,
    GroupLimitConstraint,
    Objective,
    ObjectiveBoundConstraint,
    Problem,
    Run,
    Solution,
    ValidationResult,
)

_MAX_MISSING_SCORES_RETURNED = 20
_MAX_DOMINATED_RETURNED = 20  # echoed on every structural model update — cap like missing_scores
_MAX_NEAR_DUPLICATES_RETURNED = 10  # pairs; same every-update echo, so capped like dominated_options
_NEAR_DUPLICATE_REL_TOL = 0.02  # twins = "pick either" — well under the 5%-of-range solution-clustering convention
_MAX_COVERAGE_RETURNED = 60   # option_coverage rides every solve response — ranked head at portfolio scale
_MAX_NEVER_SELECTED_LISTED = 20  # beyond this, per-option diagnostics collapse to one summary entry


def compute_metrics(problem: Problem) -> dict:
    """Compute all workflow health metrics from current problem state."""
    return {
        "framing": framing_metrics(problem),
        "data": data_metrics(problem),
        "solve": solve_metrics(problem),
        "outcome": outcome_metrics(problem),
        "diagnostics": diagnostics(problem),
    }


# --- Layer 1: Automated checkpoints ---


def framing_metrics(problem: Problem) -> dict:
    """Metrics about problem structure (objectives, options, constraints)."""
    return {
        "objective_count": len(problem.objectives),
        "option_count": len(problem.options),
        "constraint_count": len(problem.constraints),
        "directions_set": all(
            obj.direction is not None for obj in problem.objectives
        ),
    }


def data_metrics(problem: Problem) -> dict:
    """Metrics about score matrix completeness and quality."""
    obj_names = {o.name for o in problem.objectives}
    opt_names = {o.name for o in problem.options}
    max_scores = len(obj_names) * len(opt_names)

    if max_scores == 0:
        return {
            "score_completeness": 0.0,
            "missing_scores": [],
            "score_variance_by_objective": {},
            "dominated_options": [],
            "near_duplicate_options": [],
        }

    # Single pass over scores: build filled set, per-objective values, and score map
    filled: set[tuple[str, str]] = set()
    scores_by_obj: dict[str, list[float]] = {}
    score_map: dict[str, dict[str, float]] = {}
    for s in problem.scores:
        filled.add((s.option, s.objective))
        if s.objective in obj_names:
            scores_by_obj.setdefault(s.objective, []).append(s.value)
        score_map.setdefault(s.option, {})[s.objective] = s.value

    # Missing scores (capped to avoid bloating tool responses)
    missing = []
    for opt in opt_names:
        for obj in obj_names:
            if (opt, obj) not in filled:
                missing.append({"option": opt, "objective": obj})
                if len(missing) >= _MAX_MISSING_SCORES_RETURNED:
                    break
        if len(missing) >= _MAX_MISSING_SCORES_RETURNED:
            break

    completeness = len(filled) / max_scores

    # Per-objective variance
    variance_by_obj = {}
    for obj_name, values in scores_by_obj.items():
        if len(values) >= 2:
            variance_by_obj[obj_name] = round(variance(values), 4)

    # Dominated options (reuse score_map built above)
    dominated = _find_dominated_options(problem, score_map)
    near_duplicates = _find_near_duplicate_options(problem, score_map)

    missing_total = max_scores - len(filled)
    result = {
        "score_completeness": round(completeness, 4),
        "missing_scores": missing,
        "score_variance_by_objective": variance_by_obj,
        "dominated_options": dominated[:_MAX_DOMINATED_RETURNED],
        "near_duplicate_options": near_duplicates[:_MAX_NEAR_DUPLICATES_RETURNED],
    }
    if missing_total > _MAX_MISSING_SCORES_RETURNED:
        result["missing_scores_total"] = missing_total
    if len(dominated) > _MAX_DOMINATED_RETURNED:
        result["dominated_options_total"] = len(dominated)
    if len(near_duplicates) > _MAX_NEAR_DUPLICATES_RETURNED:
        result["near_duplicate_options_total"] = len(near_duplicates)

    return result


def solve_metrics(problem: Problem, run: Run | None = None) -> dict:
    """Metrics about optimization results. ``run`` selects which frontier to describe
    (an exact overlay, a scenario run); default is the exploratory ``problem.run`` —
    callers returning a specific run must pass it, or the block silently describes a
    different frontier than the response it rides in."""
    run = run if run is not None else problem.run
    if run is None:
        return {
            "solve_success": False,
            "solution_count": 0,
            "hypervolume": None,
            "spacing_cv": None,
            "objective_variation": {},
            "option_coverage": {},
        }

    solutions = run.solutions

    # Objective variation across solutions
    obj_variation = {}
    for obj in problem.objectives:
        values = _obj_values(solutions, obj.name)
        if values:
            obj_variation[obj.name] = {
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "range": round(max(values) - min(values), 4),
            }

    # Option coverage: how many solutions include each option?
    opt_names = [o.name for o in problem.options]
    coverage = {name: 0 for name in opt_names}
    for sol in solutions:
        for opt in sol.selected_options:
            if opt in coverage:
                coverage[opt] += 1

    result = {
        "solve_success": len(solutions) > 0,
        "solution_count": len(solutions),
        "hypervolume": run.quality.hypervolume_normalized,
        "spacing_cv": run.quality.spacing_cv,
        "objective_variation": obj_variation,
        "option_coverage": coverage,
    }
    # At portfolio scale the per-option dict dwarfs the rest of the solve response —
    # ship the ranked head and summarize the tail (the scenario_results treatment).
    if len(coverage) > _MAX_COVERAGE_RETURNED:
        ranked = sorted(coverage.items(), key=lambda kv: (-kv[1], kv[0]))
        head = dict(ranked[:_MAX_COVERAGE_RETURNED])
        tail_max = ranked[_MAX_COVERAGE_RETURNED][1]
        result["option_coverage"] = head
        result["option_coverage_elided"] = {
            "shown": len(head),
            "total_options": len(coverage),
            "tail_max_count": tail_max,
            "note": ("ranked by selection count; absence below the head is elision, NOT zero — "
                     f"an elided option may still appear in up to {tail_max} plan(s). Full "
                     "per-option selection rates: explore composition"),
        }
    return result


def frontier_quality(
    solutions: list[Solution],
    objectives: list[Objective],
    spacing_cv: float | None = None,
) -> dict:
    """Classify a returned frontier with progressive gates and a unified status.

    Gates are evaluated in order — failure short-circuits the rest:
      1. frontier_returned: ≥1 solution
      2. non_trivial: ≥2 solutions AND at least one objective varies (≥1% relative range)
      3. diverse: spacing CV bounded, no extreme allocation concentration, every
         objective participates in the tradeoff (none flat)

    Status mapping:
      POOR    — frontier_returned or non_trivial fails (frontier is empty or degenerate)
      WARNING — passes non_trivial but fails diverse (uneven coverage, single-winner
                allocation, or a dead objective axis)
      GOOD    — all gates pass

    Returns: {"status", "gates", "issues"}.
    """
    issues: list[str] = []

    if len(solutions) == 0:
        return {
            "status": "POOR",
            "gates": {"frontier_returned": False, "non_trivial": False, "diverse": False},
            "issues": ["No solutions returned — frontier is empty."],
        }

    non_trivial = True
    flat_objs: list[str] = []
    if len(solutions) < 2:
        non_trivial = False
        issues.append("Only 1 solution — frontier is degenerate (likely over-constrained or a single-feasible-point problem).")
    else:
        for obj in objectives:
            vals = [s.objective_values.get(obj.name, 0.0) for s in solutions]
            v_range = max(vals) - min(vals)
            scale = max(abs(max(vals)), abs(min(vals)), 1e-9)
            if v_range / scale < 0.01:
                flat_objs.append(obj.name)
        if objectives and len(flat_objs) == len(objectives):
            non_trivial = False
            issues.append("All objectives flat (<1% variation) — frontier collapsed to a single point in objective space.")

    if not non_trivial:
        return {
            "status": "POOR",
            "gates": {"frontier_returned": True, "non_trivial": False, "diverse": False},
            "issues": issues,
        }

    diverse = True

    if spacing_cv is not None and spacing_cv > 1.5:
        diverse = False
        issues.append(f"Spacing CV {spacing_cv:.2f} (>1.5) — frontier coverage is uneven; solutions cluster in some regions.")

    # A flat axis on an otherwise-varying frontier: the plans differ, but this objective
    # is <1% of the choice — epsilon spreads keep such points technically non-dominated,
    # so without this gate they arrive looking like a healthy tradeoff.
    if flat_objs:
        diverse = False
        names = ", ".join(f"'{o}'" for o in flat_objs)
        subject = (f"Objective {names} is flat" if len(flat_objs) == 1
                   else f"Objectives {names} are flat")
        issues.append(
            f"{subject} across the frontier (<1% relative variation) — not differentiating "
            "these plans; the tradeoff runs among the remaining objectives."
        )

    # Pairwise objective alignment is deliberately NOT gated here: among mutually
    # non-dominated points any pair is anti-correlated by construction on a 2-objective
    # frontier, and with ≥3 objectives a dominant tradeoff axis induces strong positive
    # pair correlations on perfectly healthy frontiers (measured 0.91–0.98 across the
    # bundled examples). All-objectives-aligned collapse already lands in gate 2 via the
    # dominance filter; the graded pairwise read is `explore tradeoffs` →
    # objective_redundancy.

    if solutions[0].allocations is not None:
        worst_concentration = 0
        worst_sol_id: int | None = None
        worst_opt: str | None = None
        for sol in solutions:
            if not sol.allocations:
                continue
            top_opt = max(sol.allocations, key=lambda k: sol.allocations[k])
            top_alloc = sol.allocations[top_opt]
            if top_alloc > worst_concentration:
                worst_concentration = top_alloc
                worst_sol_id = sol.solution_id
                worst_opt = top_opt
        if worst_concentration >= 80:
            diverse = False
            issues.append(
                f"Solution #{worst_sol_id} allocates {worst_concentration}% to '{worst_opt}' — "
                "consider a per-option upper bound or a concave objective if single-winner solutions are unwanted."
            )

    status = "GOOD" if diverse else "WARNING"
    return {
        "status": status,
        "gates": {"frontier_returned": True, "non_trivial": True, "diverse": diverse},
        "issues": issues,
    }


def readiness(problem: Problem, vr: ValidationResult) -> dict:
    """Classify validate's findings into a readiness verdict in the workflow's terms.

    A deterministic classifier over the existing issues (code decides what is true;
    skills teach the vocabulary). Verdicts:
      ready       — structure and scores complete; solving is next
      framing_gap — the model's structure needs decisions (objectives, options, rules)
      data_gap    — structure is set; the score matrix has holes

    When both kinds of gap exist, framing wins (scores can't settle a structure that
    isn't). ``ready`` on the ValidationResult stays the single go/no-go — this block
    names which kind of work remains, rolls score gaps up per objective, and points
    at the one next step.
    """
    if vr.ready:
        block = {
            "verdict": "ready",
            "summary": "Structure and scores are complete.",
            "next_steps": "solve run",
        }
        advisories = sum(1 for i in vr.issues if i.severity == "warning")
        if advisories:
            block["summary"] += f" {advisories} advisory issue(s) noted."
        return block

    framing_errors = [
        i for i in vr.issues
        if i.severity == "error" and not i.message.startswith(SCORE_MATRIX_MSG)
    ]

    score_gaps: list[dict] = []
    if vr.missing_scores:
        # validate scans unique option names, so the rollup denominator must too —
        # duplicate-named options (their own framing error) would overstate it.
        n_opts = len({o.name for o in problem.options})
        by_obj = Counter(m["objective"] for m in vr.missing_scores)
        for obj_name, k in sorted(by_obj.items(), key=lambda kv: (-kv[1], kv[0])):
            gap = {"objective": obj_name, "missing": k, "of": n_opts}
            if k == n_opts:
                gap["unscored"] = True  # the whole axis is missing, not scattered cells
            score_gaps.append(gap)

    if framing_errors:
        block = {
            "verdict": "framing_gap",
            "summary": f"{len(framing_errors)} structural issue(s) to resolve before scoring and solving (see issues).",
            "next_steps": "model update to resolve the issues; problem_framing guides the judgment calls",
        }
    else:
        fully_unscored = [g["objective"] for g in score_gaps if g.get("unscored")]
        detail = (" — " + ", ".join(f"'{o}'" for o in fully_unscored) + " entirely unscored"
                  if fully_unscored else "")
        block = {
            "verdict": "data_gap",
            "summary": (f"Structure is set; {len(vr.missing_scores)} option–objective "
                        f"score(s) still to fill{detail}."),
            "next_steps": "model update scores=… to fill the gaps; data_collection guides elicitation",
        }
    if score_gaps:
        block["score_gaps"] = score_gaps
    return block


def outcome_metrics(problem: Problem) -> dict:
    """Metrics about user engagement and decision progress.

    Uses the latest feedback entry that has both solution_id and rating,
    falling back to the latest of each independently.
    """
    feedback_list = problem.feedback

    # Prefer the latest feedback that has both fields paired
    selected = None
    rating = None
    for fb in reversed(feedback_list):
        if fb.solution_id is not None and fb.rating is not None:
            selected = fb.solution_id
            rating = fb.rating
            break

    # Fall back to latest of each if no paired entry exists
    if selected is None:
        for fb in reversed(feedback_list):
            if fb.solution_id is not None:
                selected = fb.solution_id
                break
    if rating is None:
        for fb in reversed(feedback_list):
            if fb.rating is not None:
                rating = fb.rating
                break

    return {
        "user_selected_solution": selected,
        "user_rating": rating,
        "feedback_count": len(feedback_list),
    }


# --- Diagnostics ---


def diagnostics(problem: Problem, run: Run | None = None) -> list[dict]:
    """Detect structural patterns that suggest issues or opportunities.

    Returns a list of diagnostic dicts with 'pattern', 'message', and 'severity'.
    ``run`` selects the frontier to diagnose (same contract as ``solve_metrics``).
    """
    results: list[dict] = []
    run = run if run is not None else problem.run

    if run is None or not run.solutions:
        if run is not None:
            results.append({
                "pattern": "zero_solutions",
                "severity": "error",
            })
        return results

    solutions = run.solutions

    # Pre-compute per-objective stats (single pass, shared by clustering + dominance checks)
    obj_stats = _compute_obj_stats(problem, solutions)

    # 1. Clustered solutions: all solutions within 5% of each other. Two solutions
    # can cluster too — a 2-point near-identical frontier used to fall between this
    # check (which wanted ≥3) and the <1% all-flat POOR gate, presenting as clean.
    if len(solutions) >= 2:
        _check_clustering(obj_stats, results)

    # 2. One objective dominates: <10% variation
    _check_low_variation(obj_stats, results)

    # 3. Option never selected
    _check_option_coverage(problem, solutions, results)

    # 4. Binding constraints + low diversity
    _check_binding_constraints(problem, solutions, results)

    return results


# --- Diagnostic helpers ---


def _compute_obj_stats(
    problem: Problem, solutions: list[Solution],
) -> dict[str, dict]:
    """Compute per-objective stats once for use by multiple diagnostic checks."""
    stats = {}
    for obj in problem.objectives:
        values = _obj_values(solutions, obj.name)
        if len(values) < 2:
            continue
        val_min = min(values)
        val_max = max(values)
        val_range = val_max - val_min
        val_mean = sum(values) / len(values)
        # Zero-safe scale: a mean-straddling-zero axis (net P&L, carbon delta) has real
        # spread that a |mean| denominator would collapse to rel_range 0.0 — reading a
        # 100-unit swing as "no variation" for the clustering/low-variation checks.
        scale = abs(val_mean) if abs(val_mean) > 1e-9 else max(abs(val_max), abs(val_min), 1e-9)
        rel_range = val_range / scale
        stats[obj.name] = {
            "min": val_min,
            "max": val_max,
            "range": val_range,
            "mean": val_mean,
            "relative_range": rel_range,
        }
    return stats


def _check_clustering(obj_stats: dict, results: list[dict]):
    """Detect if solutions are highly clustered (within 5% on all objectives)."""
    if not obj_stats:
        return
    for stat in obj_stats.values():
        if stat["relative_range"] > 0.05:
            return  # At least one objective has spread — not clustered

    results.append({
        "pattern": "clustered_solutions",
        "severity": "warning",
    })


def _check_low_variation(obj_stats: dict, results: list[dict]):
    """Detect objectives with <10% variation across solutions."""
    for obj_name, stat in obj_stats.items():
        if stat["relative_range"] < 0.10:
            results.append({
                "pattern": "low_variation_objective",
                "severity": "info",
                "objective": obj_name,
                "relative_range": round(stat["relative_range"], 4),
            })


def _check_option_coverage(problem, solutions, results):
    """Detect options that never appear in any solution."""
    opt_names = {o.name for o in problem.options}
    for c in problem.constraints:
        if isinstance(c, ForceExcludeConstraint):
            opt_names.discard(c.option)

    selected_ever = set()
    for sol in solutions:
        selected_ever.update(sol.selected_options)

    never_selected = sorted(opt_names - selected_ever)
    if len(never_selected) <= _MAX_NEVER_SELECTED_LISTED:
        for opt in never_selected:
            results.append({
                "pattern": "option_never_selected",
                "severity": "info",
                "option": opt,
            })
    else:
        # A ~170-entry flood of identical info rows (capital-300) buries the diagnostics
        # that matter — collapse to one summary entry past the cap.
        results.append({
            "pattern": "option_never_selected",
            "severity": "info",
            "count": len(never_selected),
            "options": never_selected[:_MAX_NEVER_SELECTED_LISTED],
            "note": (f"{len(never_selected)} options never appear in any solution; first "
                     f"{_MAX_NEVER_SELECTED_LISTED} listed — full selection rates: "
                     "explore composition"),
        })


def _check_binding_constraints(problem, solutions, results):
    """Detect constraints that are binding on any solution."""
    for constraint in problem.constraints:
        if isinstance(constraint, ObjectiveBoundConstraint):
            _check_binding_objective_bound(constraint, solutions, results)
        elif isinstance(constraint, CardinalityConstraint):
            _check_binding_cardinality(constraint, solutions, results)
        elif isinstance(constraint, GroupLimitConstraint):
            _check_binding_group_limit(constraint, solutions, results)


def _check_binding_objective_bound(constraint, solutions, results):
    obj_name = constraint.objective
    bound_val = constraint.value
    values = _obj_values(solutions, obj_name)
    if not values:
        return

    if constraint.operator == BoundOperator.max:
        extreme = max(values)
        if extreme >= bound_val * 0.95:
            results.append({
                "pattern": "binding_constraint",
                "severity": "info",
                "constraint": f"{obj_name} ≤ {bound_val}",
                "extreme_value": round(extreme, 2),
            })
    elif constraint.operator == BoundOperator.min:
        extreme = min(values)
        if extreme <= bound_val * 1.05:
            results.append({
                "pattern": "binding_constraint",
                "severity": "info",
                "constraint": f"{obj_name} ≥ {bound_val}",
                "extreme_value": round(extreme, 2),
            })


def _check_binding_cardinality(constraint, solutions, results):
    for sol in solutions:
        count = len(sol.selected_options)
        if count == constraint.max:
            results.append({
                "pattern": "binding_constraint",
                "severity": "info",
                "constraint": f"cardinality ≤ {constraint.max}",
                "actual_value": count,
            })
            return
        if count == constraint.min and constraint.min > 1:
            results.append({
                "pattern": "binding_constraint",
                "severity": "info",
                "constraint": f"cardinality ≥ {constraint.min}",
                "actual_value": count,
            })
            return


def _check_binding_group_limit(constraint, solutions, results):
    group_set = set(constraint.options)
    g_min = int(getattr(constraint, "min", 0) or 0)
    cap_hit = floor_hit = False
    for sol in solutions:
        count = len(group_set.intersection(sol.selected_options))
        if count == constraint.max and not cap_hit:
            cap_hit = True
            results.append({
                "pattern": "binding_constraint",
                "severity": "info",
                "constraint": f"group_limit({', '.join(constraint.options)}) ≤ {constraint.max}",
                "actual_value": count,
            })
        if g_min > 0 and count == g_min and not floor_hit:
            floor_hit = True
            results.append({
                "pattern": "binding_constraint",
                "severity": "info",
                "constraint": f"group_limit({', '.join(constraint.options)}) ≥ {g_min}",
                "actual_value": count,
            })
        if cap_hit and (floor_hit or g_min == 0):
            return


# --- Shared helpers ---


def _obj_values(solutions: list[Solution], obj_name: str) -> list[float]:
    """Extract values for a single objective across all solutions."""
    return [s.objective_values.get(obj_name, 0) for s in solutions]


def _find_dominated_options(
    problem: Problem,
    score_map: dict[str, dict[str, float]] | None = None,
) -> list[str]:
    """Find options dominated on all objectives by at least one other option."""
    obj_names = [o.name for o in problem.objectives]
    opt_names = [o.name for o in problem.options]
    directions = {o.name: o.direction for o in problem.objectives}

    if not obj_names or not opt_names:
        return []

    if score_map is None:
        score_map = {}
        for s in problem.scores:
            score_map.setdefault(s.option, {})[s.objective] = s.value

    complete_opts = _complete_options(obj_names, opt_names, score_map)

    dominated = []
    for opt_a in complete_opts:
        for opt_b in complete_opts:
            if opt_a == opt_b:
                continue
            b_dominates = True
            b_strictly_better = False
            for obj in obj_names:
                a_val = score_map[opt_a][obj]
                b_val = score_map[opt_b][obj]
                if directions[obj] == Direction.maximize:
                    if b_val < a_val:
                        b_dominates = False
                        break
                    if b_val > a_val:
                        b_strictly_better = True
                else:
                    if b_val > a_val:
                        b_dominates = False
                        break
                    if b_val < a_val:
                        b_strictly_better = True
            if b_dominates and b_strictly_better:
                dominated.append(opt_a)
                break

    return dominated


def _complete_options(obj_names: list, opt_names: list, score_map: dict) -> list[str]:
    """Options scored on every objective — the only ones the dominance and twin checks
    can compare (one definition, two detectors)."""
    return [opt for opt in opt_names
            if all(obj in score_map.get(opt, {}) for obj in obj_names)]


def _find_near_duplicate_options(
    problem: Problem,
    score_map: dict[str, dict[str, float]],
) -> list[dict]:
    """Pairs of complete-scored options within ``_NEAR_DUPLICATE_REL_TOL`` of each
    objective's score spread on EVERY objective — the same bet entered twice.

    Strict dominance misses these (an exact tie dominates nothing), yet twins cost real
    search budget: NSGA spreads coverage across interchangeable options, and post-solve
    stats (option_coverage, substitution principles) split between them. Pairs, not
    transitive groups — a chain A≈B≈C can put A and C outside tolerance. Advisory only:
    the engine flags, the user merges or confirms the options are genuinely distinct.
    A dominated-by-a-hair twin appears in both lists; the twin read ("pick either") is
    the truthful framing there. Sorted closest-first; capped by the caller."""
    obj_names = [o.name for o in problem.objectives]
    complete_opts = _complete_options(obj_names, [o.name for o in problem.options], score_map)
    if not obj_names or len(complete_opts) < 2:
        return []

    # Per-objective spread over complete options; a flat axis distinguishes nothing.
    spread_by_obj = {}
    for obj in obj_names:
        vals = [score_map[opt][obj] for opt in complete_opts]
        spread_by_obj[obj] = max(vals) - min(vals)

    pairs = []
    for i, opt_a in enumerate(complete_opts):
        for opt_b in complete_opts[i + 1:]:
            max_gap_fraction = 0.0
            twin = True
            for obj in obj_names:
                spread = spread_by_obj[obj]
                if spread <= 0:
                    continue  # flat axis — every option ties on it
                gap = abs(score_map[opt_a][obj] - score_map[opt_b][obj])
                if gap > spread * _NEAR_DUPLICATE_REL_TOL:
                    twin = False
                    break
                max_gap_fraction = max(max_gap_fraction, gap / spread)
            if twin:
                pairs.append({"options": sorted([opt_a, opt_b]),
                              "max_gap_fraction": round(max_gap_fraction, 4)})

    pairs.sort(key=lambda p: (p["max_gap_fraction"], p["options"]))
    return pairs
