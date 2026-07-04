"""Tests for curation quality gates (solution_quality) and knee detection with rationale.

Quality gates: an OPTIMAL status certifies optimality for the model as written, not that the
plan is actionable — degenerate finalists (empty, concentrated, pinned to bounds) are flagged
in the user's terms and surfaced with the finalist, never dropped.

Knee detection: the frontier point just before the largest ratio jump in marginal tradeoff
rate, traversed in the improving direction of objective A so a convex elbow fires the up-jump
detector; a near-linear frontier yields NO knee (threshold-gated, never a fake one). Where the
exact path attached solver duals, they replace secants as the slopes (rate_basis says which).
"""
import pytest

from engine.explorer import (
    certify_against_exact,
    curate_solution,
    export_curated,
    get_tradeoffs,
    list_curated,
    marginal_analysis,
    solution_quality,
)
from engine.models import (
    MaxAllocationConstraint,
    Objective,
    Option,
    Problem,
    Run,
    Score,
    ShadowPrice,
    Solution,
    SolutionSensitivity,
)


def _problem(approach="binary", n_options=5, constraints=None):
    names = ["A", "B", "C", "D", "E"][:n_options]
    scores = []
    for k, o in enumerate(names):
        scores.append(Score(option=o, objective="Value", value=5 + k))
        scores.append(Score(option=o, objective="Cost", value=1 + k))
    return Problem(
        name="quality-t", approach=approach,
        objectives=[Objective(name="Value", direction="maximize"),
                    Objective(name="Cost", direction="minimize")],
        options=[Option(name=o) for o in names],
        scores=scores, constraints=constraints or [],
    )


def _run(points, allocations=None, sensitivities=None):
    """Synthetic run: points = [(value, cost), ...] in Value-ascending order."""
    sols = []
    for i, (v, c) in enumerate(points):
        sols.append(Solution(
            solution_id=i,
            selected_options=["A"] if v or c else [],
            objective_values={"Value": float(v), "Cost": float(c)},
            allocations=(allocations[i] if allocations else None),
            sensitivity=(sensitivities[i] if sensitivities else None),
        ))
    return Run(solutions=sols)


# ─── solution_quality: the checks, in the user's terms ───

def test_empty_selection_is_degenerate():
    q = solution_quality(_problem(), [], None)
    assert q["status"] == "DEGENERATE"
    assert [f["check"] for f in q["flags"]] == ["empty_selection"]
    assert "selects nothing" in q["flags"][0]["message"]


def test_all_zero_allocations_are_degenerate():
    q = solution_quality(_problem(approach="proportional"), [], {"A": 0, "B": 0})
    assert q["status"] == "DEGENERATE"
    assert q["flags"][0]["check"] == "empty_selection"


def test_single_option_concentration_warns():
    q = solution_quality(_problem(approach="proportional"), ["A", "B"], {"A": 95, "B": 5})
    assert q["status"] == "WARNING"
    assert [f["check"] for f in q["flags"]] == ["single_option_concentration"]
    assert "'A'" in q["flags"][0]["message"] and "95%" in q["flags"][0]["message"]


def test_allocations_pinned_at_bounds_warn():
    p = _problem(approach="proportional", constraints=[MaxAllocationConstraint(max=50)])
    q = solution_quality(p, ["A", "B"], {"A": 50, "B": 50, "C": 0, "D": 0, "E": 0})
    assert q["status"] == "WARNING"
    assert [f["check"] for f in q["flags"]] == ["allocations_at_bounds"]
    assert "50%" in q["flags"][0]["message"]


def test_healthy_spread_has_no_flags():
    q = solution_quality(_problem(approach="proportional"), list("ABCDE"),
                         {"A": 30, "B": 25, "C": 20, "D": 15, "E": 10})
    assert q == {"status": "GOOD", "flags": []}


def test_single_option_binary_selection_is_not_flagged():
    # A one-option binary pick is often the right answer; distribution checks are
    # proportional-only.
    q = solution_quality(_problem(), ["A"], None)
    assert q["status"] == "GOOD"


# ─── The gate rides the curate surfaces, never drops a finalist ───

def test_curate_and_list_surface_quality_and_keep_finalist():
    p = _problem()
    p.run = _run([(0, 0), (5, 1), (11, 3)])  # solution 0 is the empty plan
    out = curate_solution(p, solution_id=0, custom_name="empty")
    assert out["curated"] is True                       # flagged, still curated
    assert out["quality"]["status"] == "DEGENERATE"
    listed = list_curated(p)["curated_solutions"]
    assert listed[0]["quality"]["status"] == "DEGENERATE"


def test_export_carries_quality_column():
    p = _problem()
    p.run = _run([(0, 0), (5, 1), (11, 3)])
    curate_solution(p, solution_id=0, custom_name="empty")
    curate_solution(p, solution_id=2, custom_name="full")
    md = export_curated(p, format="markdown")["content"]
    assert "quality" in md.splitlines()[0]
    assert "DEGENERATE" in md and "GOOD" in md
    csv = export_curated(p, format="csv")["content"]
    assert "quality" in csv.splitlines()[0] and "DEGENERATE" in csv


def test_certify_readback_flags_degenerate_exact_point():
    p = _problem()
    nsga = _run([(5, 1), (11, 3)])
    exact = _run([(0, 0), (5, 1), (12, 3)])             # exact includes the empty plan
    cert = certify_against_exact(p, nsga, exact)
    gates = cert["quality_gates"]
    assert [f["solution_id"] for f in gates["flagged"]] == [0]
    assert gates["flagged"][0]["status"] == "DEGENERATE"
    assert "checks" in gates["flagged"][0]
    assert "optimal" in gates["note"]


def test_certify_readback_clean_when_no_degenerate_points():
    p = _problem()
    cert = certify_against_exact(p, _run([(5, 1), (11, 3)]), _run([(5, 1), (12, 3)]))
    assert cert["quality_gates"]["flagged"] == []


# ─── Knee detection: elbow fires with rationale, flat frontier stays quiet ───

ELBOW = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 8), (5, 13)]   # rate jumps 1 → 5 after (3, 3)


def test_marginal_analysis_finds_elbow_with_rationale():
    p = _problem()
    p.run = _run(ELBOW)
    pair = marginal_analysis(p, detail=True)["pairs"][0]
    assert pair["rate_basis"] == "secant"
    inflection = pair["inflection"]
    assert inflection["solution_id"] == 3               # the point just before the jump
    assert inflection["jump_factor"] == 5.0
    assert "5.0×" in inflection["rationale"].replace("5.0x", "5.0×") or "5.0" in inflection["rationale"]
    assert "Value" in inflection["rationale"] and "Cost" in inflection["rationale"]


def test_near_linear_frontier_yields_no_knee():
    p = _problem()
    p.run = _run([(0, 0), (1, 1), (2, 2.1), (3, 3), (4, 4.2), (5, 5.1)])
    pair = marginal_analysis(p, detail=True)["pairs"][0]
    assert pair["inflection"] is None


def test_tradeoffs_candidates_carry_rationale():
    p = _problem()
    p.run = _run(ELBOW)
    for cand in get_tradeoffs(p)["inflection_point_candidates"]:
        assert "rationale" in cand and "jump_factor" in cand


# ─── Exact duals replace secants as slopes where the exact path attached them ───

def _sens(dual):
    return SolutionSensitivity(shadow_prices=[
        ShadowPrice(name="Value", role="linear_floor", shadow_price=dual)])


def test_dual_slopes_used_when_every_point_carries_them():
    p = _problem(approach="proportional")
    duals = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
    p.run = _run(ELBOW, sensitivities=[_sens(d) for d in duals])
    pair = marginal_analysis(p, detail=True)["pairs"][0]
    assert pair["rate_basis"] == "solver_exact_duals"
    # Transition rates = dual at each left endpoint (exact slopes, not secants).
    assert [r["rate"] for r in pair["rates"]] == [1.0, 1.0, 1.0, 5.0, 5.0]
    assert pair["inflection"]["solution_id"] == 3


def test_dual_slopes_fall_back_to_secants_when_any_point_lacks_them():
    p = _problem(approach="proportional")
    sens = [_sens(1.0)] * 5 + [None]
    p.run = _run(ELBOW, sensitivities=sens)
    pair = marginal_analysis(p, detail=True)["pairs"][0]
    assert pair["rate_basis"] == "secant"
