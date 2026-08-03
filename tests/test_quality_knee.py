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
    # Caption, not teaching: diagnosis prose lives in 'Reading the Certificate'.
    assert "degenerate or pinned" in gates["note"]


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


# ─── Degeneracy guard: a near-tie denominator is a tie, not a cliff ───
#
# The live defect these reconstruct: adjacent frontier points that all but tie on the
# denominator objective divide a real numerator by ~0, and the resulting rate took the
# inflection headline away from the genuine elbow (supplier_selection and
# community_program_funding both led with a jump factor driven by a 1e-3/1e-4-scale step).

# A frontier whose elbow (solution 3) and ideal-closest point (solution 4) are different
# solutions, so the `tradeoffs` de-duplication doesn't hide the candidate under test.
ELBOW_LONG = ELBOW + [(6, 18), (7, 23)]
# ...plus one near-tie step at the end: Value moves 0.0001 while Cost moves 1.5, for a rate
# of 15000 — the largest jump in the sequence by three orders of magnitude, and pure
# division artifact. This is the shape of both live receipts.
TIE_AFTER_ELBOW = ELBOW_LONG + [(7.0001, 24.5)]


def _pair(points, **kwargs):
    p = _problem()
    p.run = _run(points)
    return marginal_analysis(p, **kwargs)["pairs"][0]


def test_near_tie_denominator_loses_the_inflection_to_the_real_elbow():
    pair = _pair(TIE_AFTER_ELBOW, detail=True)
    # The elbow, not the tie: solution 3 at 5x, not solution 5 at ~3000x.
    assert pair["inflection"]["solution_id"] == 3
    assert pair["inflection"]["jump_factor"] == 5.0


def test_near_tie_transition_is_kept_in_detail_and_flagged():
    pair = _pair(TIE_AFTER_ELBOW, detail=True)
    tie = pair["rates"][-1]
    assert tie["delta_Value"] == 0.0001 and tie["degenerate"] is True
    assert tie["rate"] == 15000.0
    assert len(pair["rates"]) == len(TIE_AFTER_ELBOW) - 1      # flagged, never dropped
    assert [r for r in pair["rates"][:-1] if r.get("degenerate")] == []


def test_rate_guard_echoes_the_floor_that_excluded_it():
    guard = _pair(TIE_AFTER_ELBOW, detail=True)["rate_guard"]
    assert guard["denominator"] == "Value"
    assert guard["degenerate_transitions"] == 1
    # Floor = a quarter of the typical (median) step of 1.0, well above the 0.0001 tie.
    assert guard["denominator_floor"] == 0.25


def test_degenerate_transition_stays_out_of_the_steepest_headline():
    pair = _pair(TIE_AFTER_ELBOW)                              # summary mode
    assert all(not r.get("degenerate") for r in pair["steepest_transitions"])
    assert max(r["rate"] for r in pair["steepest_transitions"]) == 5.0


def test_summary_stats_span_every_transition_so_the_median_read_is_untouched():
    # Filtering shapes the headline lists; it never moves the distribution stats. Rates are
    # [1, 1, 1, 5, 5, 5, 5, 15000] — median 5.0, and the artifact stays visible as rate_max.
    summary = _pair(TIE_AFTER_ELBOW)["summary"]
    assert summary["total_transitions"] == 8
    assert summary["rate_median"] == 5.0
    assert summary["rate_max"] == 15000.0


def test_tradeoffs_candidates_are_pre_filtered_for_degeneracy():
    p = _problem()
    p.run = _run(TIE_AFTER_ELBOW)
    candidates = get_tradeoffs(p)["inflection_point_candidates"]
    assert [c["solution_id"] for c in candidates] == [3]
    assert candidates[0]["jump_factor"] == 5.0


def test_genuine_cliff_over_a_healthy_step_still_fires():
    # No tie anywhere: the guard must stay out of the way of a real elbow.
    pair = _pair(ELBOW_LONG, detail=True)
    assert pair["inflection"]["solution_id"] == 3
    assert pair["rate_guard"]["degenerate_transitions"] == 0


def test_floor_tracks_spacing_on_a_densely_sampled_frontier():
    # The community_program_funding regime: 40 points across a range of 0.39, so EVERY step
    # is a fraction of a percent of the range and a range-only floor would catch none of
    # them. The floor follows the typical step instead, so the 1e-4 tie is still the tie.
    points = [(round(0.01 * k, 4), 0.5 * k) for k in range(40)]
    points.insert(20, (points[19][0] + 0.0001, points[19][1] + 0.4))
    pair = _pair(points, detail=True)
    guard = pair["rate_guard"]
    assert guard["degenerate_transitions"] == 1
    assert 0.0001 < guard["denominator_floor"] < 0.01
    assert pair["rates"][19]["degenerate"] is True


def test_exact_ties_are_degenerate_even_when_they_are_the_typical_step():
    # Majority-tie frontier: the median step is 0, so the spacing term vanishes and the
    # range term is what keeps the floor above zero.
    points = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (2, 9)]
    pair = _pair(points, detail=True)
    assert pair["rate_guard"]["denominator_floor"] == pytest.approx(0.002)   # 0.1% of a span of 2
    assert [bool(r.get("degenerate")) for r in pair["rates"]] == [True, True, True, False, False]


# ─── Direction honesty: an unsigned rate is not always a price ───


def _three_objective_problem():
    """Three objectives, so a pair can improve together while the conflict is paid on the third."""
    names = ["A", "B", "C"]
    scores = [Score(option=o, objective=obj, value=1.0)
              for o in names for obj in ("Value", "Cost", "Risk")]
    return Problem(
        name="co-improve-t", approach="binary",
        objectives=[Objective(name="Value", direction="maximize"),
                    Objective(name="Cost", direction="minimize"),
                    Objective(name="Risk", direction="minimize")],
        options=[Option(name=o) for o in names], scores=scores, constraints=[],
    )


# Value climbs throughout; the last step also DROPS Cost (Risk pays for it), so that
# transition's rate is a ratio of two gains and is where the largest rate jump lands.
CO_IMPROVING = [(0, 0, 9), (1, 1, 8), (2, 2, 7), (3, 3, 6), (4, 4, 5), (5, 0, 40)]


def _run3(points):
    return Run(solutions=[
        Solution(solution_id=i, selected_options=["A"],
                 objective_values={"Value": float(v), "Cost": float(c), "Risk": float(k)})
        for i, (v, c, k) in enumerate(points)
    ])


def test_a_step_that_gains_both_objectives_is_labelled_not_priced():
    # Value up and Cost DOWN across the last step — the pair improves together (Risk pays for
    # it), so the rate is a ratio of two gains and must not read as cost-per-unit-gained.
    p = _three_objective_problem()
    p.run = _run3(CO_IMPROVING)
    pair = next(x for x in marginal_analysis(p, detail=True)["pairs"]
                if x["objectives"] == ["Value", "Cost"])
    assert pair["rates"][-1]["co_improvement"] is True
    assert [r for r in pair["rates"][:-1] if r.get("co_improvement")] == []
    assert pair["rate_guard"]["co_improvement_transitions"] == 1


def test_a_co_improving_inflection_says_so_instead_of_naming_a_price():
    p = _three_objective_problem()
    p.run = _run3(CO_IMPROVING)
    pair = next(x for x in marginal_analysis(p, detail=True)["pairs"]
                if x["objectives"] == ["Value", "Cost"])
    inflection = pair["inflection"]
    assert inflection["co_improvement"] is True
    assert "GAINS Cost" in inflection["rationale"]
    assert "costs ~" not in inflection["rationale"]


def test_an_ordinary_tradeoff_inflection_keeps_the_price_framing():
    pair = _pair(ELBOW, detail=True)
    assert "co_improvement" not in pair["inflection"]
    assert "costs ~5.0× more Cost" in pair["inflection"]["rationale"]


def test_the_ascii_marks_both_caveats_where_the_number_is_read():
    assert "≈ tie on Value" in _pair(TIE_AFTER_ELBOW, detail=True)["visualization"]
    p = _three_objective_problem()
    p.run = _run3(CO_IMPROVING)
    pair = next(x for x in marginal_analysis(p, detail=True)["pairs"]
                if x["objectives"] == ["Value", "Cost"])
    assert "↑ Cost gained, not paid" in pair["visualization"]


# ─── Flatness guard: a near-tie NUMERATOR is a flat stretch, not a cliff ───
#
# The mirror of the degeneracy guard, and the live defect it reconstructs (channel_budget,
# Conversions vs BrandLift): the marker sat at the END of an anomalously flat stretch — the
# numerator objective moved 0.15 over a perfectly healthy 1.74 denominator step, so the rate
# was ~0 and the jump measured against the even flatter step before it read as a 25.5x cliff.
# It is the opposite of a cliff: nothing was bought there. The bundled example carries the
# same shape at 1e-4 scale (BrandLift moves 0.0002 across a 1.301 Conversions step, for a
# headline jump of 17038x).


def _steps(dv, dc):
    """Frontier points from per-transition (Value, Cost) steps — deltas are what's under test."""
    v, c, pts = 0.0, 0.0, [(0.0, 0.0)]
    for a, b in zip(dv, dc):
        v, c = round(v + a, 6), round(c + b, 6)
        pts.append((v, c))
    return pts


# The live receipt, reconstructed: a flat stretch (Cost moves 0.005, then 0.15) inside a
# frontier whose steps otherwise run 4-6. Without the guard the largest jump in the whole
# sequence is the 0.15 step landing on the 0.00507 one — exactly 25.5x, on a step that
# bought 0.15 of Cost.
FLAT_STRETCH = _steps([1.0, 1.0, 1.5, 1.74, 3.0, 1.0],
                      [4.0, 4.4, 0.00507, 0.15, 6.0, 5.0])

# The genuine-kink control: the step INTO the marker is small on both sides (3.02 Value buys
# 0.08 Cost) but this pair's steps run ~0.5, so 0.08 is a real move, not a non-move — and the
# step OUT of it is a real price (0.29 buys 1.29). The kink survives untouched.
GENUINE_KINK = _steps([0.5, 0.5, 3.02, 0.29, 0.5, 0.5],
                      [0.5, 0.6, 0.08, 1.29, 0.55, 0.5])


def test_flat_numerator_step_is_flagged_and_loses_the_marker():
    pair = _pair(FLAT_STRETCH, detail=True)
    flat = pair["rates"][3]
    assert flat["delta_Cost"] == -0.15 and flat["delta_Value"] == 1.74   # healthy denominator
    assert flat["flat"] is True and "degenerate" not in flat
    # The marker moves off it — to the one real steepening left in the sequence.
    assert pair["inflection"]["solution_id"] == 5
    assert pair["inflection"]["jump_factor"] == 2.5


def test_the_flat_stretch_is_what_the_unguarded_detector_would_have_led_with():
    # Both ends of the stretch are ties, so neither can anchor a jump: the 25.5x the raw
    # quotients offer (0.0862 / 0.00338) is the number the guard exists to keep off the page.
    def secant(k):     # unrounded, straight off the fixture's points
        (v0, c0), (v1, c1) = FLAT_STRETCH[k], FLAT_STRETCH[k + 1]
        return (c1 - c0) / (v1 - v0)
    assert round(secant(3) / secant(2), 1) == 25.5
    rows = _pair(FLAT_STRETCH, detail=True)["rates"]
    assert [i for i, r in enumerate(rows) if r.get("flat")] == [2, 3]


def test_a_genuine_kink_over_a_small_but_real_step_still_fires():
    pair = _pair(GENUINE_KINK, detail=True)
    assert pair["rate_guard"]["flat_transitions"] == 0
    assert pair["inflection"]["solution_id"] == 3
    assert pair["inflection"]["jump_factor"] == 167.9


def test_the_numerator_floor_is_that_pair_s_own_scale_not_an_absolute():
    # Why the two receipts above split: 0.15 is a tie where the typical step is 4.2, while a
    # SMALLER 0.08 is a real move where the typical step is 0.55. The floor is relative, so
    # the same absolute step reads differently in two pairs — as it should.
    flat_guard = _pair(FLAT_STRETCH)["rate_guard"]
    kink_guard = _pair(GENUINE_KINK)["rate_guard"]
    assert flat_guard["numerator_floor"] > 0.15 > kink_guard["numerator_floor"]
    assert kink_guard["numerator_floor"] < 0.08


def test_rate_guard_echoes_the_numerator_side_too():
    guard = _pair(FLAT_STRETCH, detail=True)["rate_guard"]
    assert guard["numerator"] == "Cost"
    assert guard["numerator_floor"] == pytest.approx(0.21)   # a twentieth of the 4.2 median
    assert guard["flat_transitions"] == 2
    assert "`flat` rows tie on Cost" in guard["note"]


def test_flat_transitions_are_kept_in_detail_and_counted_in_summary():
    detail = _pair(FLAT_STRETCH, detail=True)
    assert len(detail["rates"]) == len(FLAT_STRETCH) - 1          # flagged, never dropped
    # Filtering shapes the headline; it never moves the distribution stats. Rates are
    # [4.0, 4.4, 0.0034, 0.0862, 2.0, 5.0] — median 3.0, unchanged by the guard.
    summary = _pair(FLAT_STRETCH)["summary"]
    assert summary["total_transitions"] == 6
    assert summary["rate_median"] == 3.0
    assert summary["rate_min"] == 0.0034


def test_tradeoffs_candidates_are_pre_filtered_for_flat_steps():
    p = _problem()
    p.run = _run(FLAT_STRETCH)
    candidates = get_tradeoffs(p)["inflection_point_candidates"]
    assert [c["solution_id"] for c in candidates] == [5]


def test_an_elbow_s_own_cheap_side_is_not_a_flat_tie():
    # The numerator's typical step is the frontier's SHAPE, so a quarter of it (the
    # denominator's tie fraction) would call every cheap step of an honest elbow a non-move.
    # ELBOW_LONG's cheap side moves 1.0 against a 5.0 median — real, and the elbow survives.
    pair = _pair(ELBOW_LONG, detail=True)
    assert pair["rate_guard"]["flat_transitions"] == 0
    assert pair["rate_guard"]["numerator_floor"] < 1.0
    assert pair["inflection"]["solution_id"] == 3


def test_a_step_that_ties_on_both_sides_carries_both_flags():
    # Degeneracy and flatness are separate reads — "the price exploded" vs "nothing was
    # bought" — and a step that barely moves at all is honestly both.
    pair = _pair(ELBOW_LONG + [(7.0001, 23.0001)], detail=True)
    both = pair["rates"][-1]
    assert both["degenerate"] is True and both["flat"] is True
    assert pair["rate_guard"]["degenerate_transitions"] == 1
    assert pair["rate_guard"]["flat_transitions"] == 1


def test_the_ascii_marks_the_flat_caveat_where_the_number_is_read():
    viz = _pair(FLAT_STRETCH, detail=True)["visualization"]
    assert "≈ flat on Cost" in viz
    assert "≈ tie on Value" not in viz     # this pair's denominators are all healthy
