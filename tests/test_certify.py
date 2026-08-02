"""Tests for the exact-solver audit/certification helper (engine/explorer.py:certify_against_exact).

Unit tests build hand-crafted NSGA/exact runs so the dominance audit, the invariant check, and
corner sharpening are exercised deterministically (no solver needed). One integration test
(skipped without highspy) loads a bundled MILP example, runs NSGA + an exact backend, and
certifies — confirming the invariant holds strictly when the exact points are integer by
construction. Solver-agnostic: the helper treats HiGHS and cuOpt identically.
"""

import tempfile

import numpy as np
import pytest

import mcp_server.server as srv
from engine.explorer import _dominates_min, certify_against_exact
from engine.models import (
    Approach,
    CardinalityConstraint,
    CuratedSolution,
    Objective,
    Option,
    Problem,
    Run,
    Score,
    Solution,
)
from engine.store import Store


def _problem(objectives):
    names = ["A", "B", "C", "D"]
    return Problem(
        name="t", approach="proportional", objectives=objectives,
        options=[Option(name=n) for n in names],
        scores=[Score(option=n, objective=o.name, value=1.0) for n in names for o in objectives],
    )


def _run(points, objectives, solver="nsga-ii", exact=False):
    """A Run whose solutions carry the given (per-objective) value dicts."""
    sols = [Solution(solution_id=i, selected_options=["A"], objective_values=dict(zip([o.name for o in objectives], p)))
            for i, p in enumerate(points)]
    return Run(solutions=sols, solver=solver, exact=exact)


# Return ↑, Risk ↓ (quadratic) — a mean-variance shape.
_OBJS = [Objective(name="Return", direction="maximize", aggregation="avg"),
         Objective(name="Risk", direction="minimize", aggregation="quadratic")]


def _binary_milp():
    """Small binary selection with additive (sum) objectives — the end-to-end MILP certify
    fixture. Tiny so the real exact solve stays fast (decoupled from showcase examples). Binary
    `sum` is the only exact-MILP-representable aggregation (see solvers.exact_solver_fits)."""
    names = ["A", "B", "C", "D", "E", "F"]
    table = {"Value": [9, 7, 8, 4, 6, 5], "Cost": [5, 3, 6, 2, 4, 3]}
    scores = [Score(option=n, objective=o, value=table[o][i])
              for i, n in enumerate(names) for o in table]
    return Problem(
        name="milp", approach="binary",
        objectives=[Objective(name="Value", direction="maximize"),
                    Objective(name="Cost", direction="minimize")],
        options=[Option(name=n) for n in names], scores=scores,
        constraints=[CardinalityConstraint(min=2, max=4)],
    )


# ─── dominance primitive ───

def test_dominates_min_strict():
    assert _dominates_min(np.array([1.0, 1.0]), np.array([2.0, 2.0]))      # better on both
    assert _dominates_min(np.array([1.0, 2.0]), np.array([1.0, 3.0]))      # equal + better
    assert not _dominates_min(np.array([1.0, 3.0]), np.array([2.0, 2.0]))  # trade-off, neither
    assert not _dominates_min(np.array([2.0, 2.0]), np.array([2.0, 2.0]))  # equal → not strict


# ─── dominance audit ───

def test_audit_counts_nsga_points_dominated_by_exact():
    prob = _problem(_OBJS)
    # Exact point: Return 10, Risk 2. NSGA has one dominated point (Return 8, Risk 3) and one
    # genuine trade-off (Return 12, Risk 5) the exact set doesn't dominate.
    exact = _run([(10.0, 2.0)], _OBJS, solver="highs")
    nsga = _run([(8.0, 3.0), (12.0, 5.0)], _OBJS)
    c = certify_against_exact(prob, nsga, exact)
    assert c["dominance_audit"]["nsga_dominated_by_exact"] == 1
    assert c["dominance_audit"]["nsga_dominated_fraction"] == 0.5
    assert c["exact_solver"] == "highs"
    assert c["dominance_audit"]["examples"][0]["nsga_point"] == {"Return": 8.0, "Risk": 3.0}


def test_invariant_holds_when_exact_undominated():
    prob = _problem(_OBJS)
    exact = _run([(10.0, 2.0), (14.0, 6.0)], _OBJS, solver="cuopt")
    nsga = _run([(9.0, 3.0), (11.0, 4.0)], _OBJS)
    c = certify_against_exact(prob, nsga, exact)
    assert c["invariant"]["holds"] is True
    assert c["invariant"]["exact_dominated_by_nsga"] == 0
    assert c["exact_solver"] == "cuopt"


def test_invariant_violation_is_flagged_not_celebrated():
    prob = _problem(_OBJS)
    # An exact point (Return 8, Risk 4) that an NSGA point (Return 10, Risk 2) dominates — the
    # rounding/under-sampling artifact. Reported as a violation with a not-a-heuristic-win note.
    exact = _run([(8.0, 4.0)], _OBJS, solver="highs")
    nsga = _run([(10.0, 2.0)], _OBJS)
    c = certify_against_exact(prob, nsga, exact)
    assert c["invariant"]["holds"] is False
    assert c["invariant"]["exact_dominated_by_nsga"] == 1
    assert "not a heuristic" in c["invariant"]["note"]


# ─── per-pick verdicts (the curated finalists) ───

def _curated_fixture():
    """A certificate whose curated set spans every verdict the block can return.

    Minimize-space (Return negated): NSGA [(-8,3), (-11,2.2), (-9,4.5)], exact [(-10,2), (-12,4)],
    every exact point non-dominated. Joint spread = (4, 2.5), so the gaps below are hand-checkable.
    """
    p = _problem(_OBJS)
    nsga = _run([(8.0, 3.0), (11.0, 2.2), (9.0, 4.5)], _OBJS)
    exact = _run([(10.0, 2.0), (12.0, 4.0)], _OBJS, solver="highs")
    for s, sig in zip(exact.solutions, ["exact-a", "exact-b"]):
        s.content_signature = sig
    p.curated_solutions = [
        # An NSGA pick one exact point beats: gap max(2/4, 1/2.5) = 0.5 against exact-a.
        CuratedSolution(content_signature="pick-beaten", custom_name="Balanced",
                        objective_values={"Return": 8.0, "Risk": 3.0}),
        # A pick pinned off the exact overlay itself — nothing certified beats it.
        CuratedSolution(content_signature="pick-safe", custom_name="Aggressive",
                        objective_values={"Return": 12.0, "Risk": 4.0}),
        # Beaten by BOTH exact points: exact-a by max(1/4, 2.5/2.5) = 1.0, exact-b by
        # max(3/4, 0.5/2.5) = 0.75 — the nearer certifying point is the one reported.
        CuratedSolution(content_signature="pick-two-beaters", custom_name="Hedge",
                        objective_values={"Return": 9.0, "Risk": 4.5}),
        # An NSGA pick the exact overlay does NOT beat — the audit leaves it alone too.
        CuratedSolution(content_signature="pick-survivor", custom_name="Lean",
                        objective_values={"Return": 11.0, "Risk": 2.2}),
    ]
    return certify_against_exact(p, nsga, exact)


def test_per_pick_names_the_gap_and_the_certifying_point():
    """A dominated finalist reports how far it sits from the certified frontier and which
    certified point proves it — the answer that previously needed the exact-run JSON on disk."""
    pp = _curated_fixture()["per_pick"]
    assert pp["pick-beaten"] == {"verdict": "dominated", "custom_name": "Balanced",
                                 "gap": 0.5, "dominated_by": "exact-a"}


def test_per_pick_reports_the_nearest_certifying_point():
    """When several certified points beat a pick, the gap is the distance to the NEAREST one —
    the smallest move that reaches a plan better on every objective."""
    assert _curated_fixture()["per_pick"]["pick-two-beaters"] == {
        "verdict": "dominated", "custom_name": "Hedge", "gap": 0.75, "dominated_by": "exact-b"}


def test_per_pick_marks_an_undominated_finalist_optimal():
    """No certified point above it → optimal at its own tradeoff, nothing to close (gap 0), and
    no `dominated_by` to name."""
    assert _curated_fixture()["per_pick"]["pick-safe"] == {
        "verdict": "optimal", "custom_name": "Aggressive", "gap": 0.0}


def test_per_pick_agrees_with_the_dominance_audit():
    """Same criterion, two readings: a pin sitting on an NSGA point is `dominated` exactly when
    the frontier-level audit counts that point. The two can never tell different stories."""
    c = _curated_fixture()
    beaten = {sig for sig, v in c["per_pick"].items() if v["verdict"] == "dominated"}
    assert beaten == {"pick-beaten", "pick-two-beaters"}          # the two beaten NSGA pins
    assert c["per_pick"]["pick-survivor"]["verdict"] == "optimal"  # the NSGA pin the audit spares
    assert c["dominance_audit"]["nsga_dominated_by_exact"] == 2    # (8,3) and (9,4.5) of the three
    assert c["invariant"]["holds"] is True


def test_per_pick_absent_when_nothing_is_curated():
    """Absent, not empty — the same way `regret.curated` behaves. Nothing curated, no noise."""
    c = certify_against_exact(_problem(_OBJS), _run([(8.0, 3.0)], _OBJS),
                              _run([(10.0, 2.0)], _OBJS, solver="highs"))
    assert "per_pick" not in c


def test_per_pick_declines_a_pin_that_predates_a_current_objective():
    """A pin curated before an objective existed can't be placed in this objective space. Report
    it inconclusive and name what it lacks — scoring it against a fabricated 0.0 would read as a
    spurious `optimal`, the exact misreport this block exists to prevent."""
    p = _problem(_OBJS)
    p.curated_solutions = [CuratedSolution(content_signature="pick-stale", custom_name="Old pin",
                                           objective_values={"Return": 9.0})]
    entry = certify_against_exact(p, _run([(8.0, 3.0)], _OBJS),
                                  _run([(10.0, 2.0)], _OBJS, solver="highs"))["per_pick"]["pick-stale"]
    assert entry["verdict"] == "inconclusive" and entry["missing_objectives"] == ["Risk"]
    assert "gap" not in entry


def test_certificate_prose_leads_with_the_curated_verdict():
    """The tally rides the headline `recommendation`, and a dominated finalist routes the agent to
    re-curate before anything is presented — the certificate's own next move."""
    c = _curated_fixture()
    assert c["recommendation"].startswith("curated picks: 2 of 4 optimal")
    assert "2 dominated" in c["recommendation"]
    assert c["next_steps"].startswith("Re-curate first:") and 'source="exact"' in c["next_steps"]


def test_per_pick_rides_the_explore_certify_wire(srv_tmp_store):
    """Contract gate: the verdicts must reach the agent through `explore certify` itself. The
    defect this fixes was exactly a per-finalist answer that existed only in the exact-run JSON
    on disk, so an engine-level assertion alone wouldn't have caught it."""
    p = _problem(_OBJS)
    p.run = _run([(8.0, 3.0)], _OBJS)
    p.exact_run = _run([(10.0, 2.0)], _OBJS, solver="highs")
    p.exact_run.solutions[0].content_signature = "exact-a"
    p.curated_solutions = [CuratedSolution(content_signature="pick-beaten", custom_name="Balanced",
                                           objective_values={"Return": 8.0, "Risk": 3.0})]
    srv.store.save(p)
    out = srv.explore(action="certify", problem_id=p.problem_id)
    assert out["per_pick"]["pick-beaten"]["verdict"] == "dominated"
    assert out["per_pick"]["pick-beaten"]["dominated_by"] == "exact-a"


def test_certificate_prose_stays_quiet_without_curation():
    """No curated set → no per-pick clause anywhere; the certificate reads exactly as before."""
    c = certify_against_exact(_problem(_OBJS), _run([(8.0, 3.0)], _OBJS),
                              _run([(10.0, 2.0)], _OBJS, solver="highs"))
    assert "curated picks" not in c["recommendation"]
    assert "Re-curate" not in c["next_steps"]


# ─── corner sharpening ───

def test_corner_sharpening_marks_risk_corner_and_status():
    prob = _problem(_OBJS)
    # Exact reaches a lower (better) Risk minimum (1.8 vs NSGA 2.0) → sharpened risk corner.
    # NSGA reaches a higher Return (14 vs exact 11) → exact under-samples that linear corner.
    exact = _run([(11.0, 1.8), (9.0, 3.0)], _OBJS, solver="highs")
    nsga = _run([(14.0, 2.0), (10.0, 2.5)], _OBJS)
    c = certify_against_exact(prob, nsga, exact)
    risk = c["corner_sharpening"]["Risk"]
    assert risk["is_risk_corner"] is True and risk["status"] == "sharpened"
    assert risk["nsga_best"] == 2.0 and risk["exact_best"] == 1.8 and risk["improvement"] == 0.2
    ret = c["corner_sharpening"]["Return"]
    assert ret["is_risk_corner"] is False and ret["status"] == "under-sampled"
    assert c["headline_corner"] == "Risk"           # risk corner is the headline when sharpened
    assert "under-samples Return" in c["recommendation"]


def test_only_quadratic_minimize_is_the_risk_corner():
    # A maximize-quadratic (e.g. Reach) is NOT a risk corner — the convex-bowl argument is
    # minimize-variance only.
    objs = [Objective(name="Return", direction="maximize", aggregation="avg"),
            Objective(name="Reach", direction="maximize", aggregation="quadratic")]
    prob = _problem(objs)
    c = certify_against_exact(prob, _run([(10.0, 5.0)], objs, solver="highs"), _run([(9.0, 4.0)], objs))
    assert all(not v["is_risk_corner"] for v in c["corner_sharpening"].values())


def test_empty_run_raises():
    prob = _problem(_OBJS)
    with pytest.raises(ValueError):
        certify_against_exact(prob, _run([], _OBJS), _run([(1.0, 1.0)], _OBJS))


# ─── coverage (hypervolume the exact overlay reclaims) ───

def test_coverage_reclaims_volume_when_exact_extends_the_front():
    """When the exact overlay reaches a point that dominates/extends the NSGA front, it reclaims
    hypervolume — the magnitude companion to the dominance count."""
    prob = _problem(_OBJS)
    nsga = _run([(10.0, 5.0)], _OBJS)                           # Return 10, Risk 5
    exact = _run([(12.0, 3.0)], _OBJS, solver="highs")         # better on both → extends the front
    c = certify_against_exact(prob, nsga, exact)
    cov = c["coverage"]
    assert cov is not None
    assert 0.0 <= cov["reclaimed_fraction"] <= 1.0
    assert cov["exact_reclaims"] > 0.0                          # exact expands the covered region
    assert cov["combined_hypervolume"] >= cov["nsga_hypervolume"]


def test_coverage_is_zero_when_exact_adds_nothing():
    """When every exact point is already dominated by NSGA, the overlay reclaims no volume — the
    honest small-instance result (exact confirms, doesn't expand). Here the 0 is geometric: NSGA
    already dominates the exact point, so the union adds nothing."""
    prob = _problem(_OBJS)
    nsga = _run([(12.0, 3.0)], _OBJS)                           # dominates the exact point below
    exact = _run([(10.0, 5.0)], _OBJS, solver="highs")
    c = certify_against_exact(prob, nsga, exact)
    assert c["coverage"]["exact_reclaims"] == 0.0
    assert c["coverage"]["reclaimed_fraction"] == 0.0


def test_coverage_none_on_degenerate_front():
    """A combined front with a flat axis (no spread) has undefined coverage — reported as None,
    not a divide-by-zero."""
    prob = _problem(_OBJS)
    c = certify_against_exact(prob, _run([(10.0, 5.0)], _OBJS),
                              _run([(10.0, 5.0)], _OBJS, solver="highs"))
    assert c["coverage"] is None


def test_coverage_partial_reclaim_on_multi_point_fronts():
    """The realistic regime: exact extends *part* of a multi-point NSGA front, so the reclaimed
    fraction lands strictly between 0 and 1 — single-point fronts collapse to the trivial
    box-origin case, so this locks the multi-point normalization the metric is built for."""
    prob = _problem(_OBJS)
    nsga = _run([(10.0, 5.0), (8.0, 4.0), (6.0, 2.0)], _OBJS)
    exact = _run([(12.0, 3.0), (11.0, 2.5), (9.0, 1.8)], _OBJS, solver="highs")
    cov = certify_against_exact(prob, nsga, exact)["coverage"]
    assert cov is not None
    assert 0.0 < cov["reclaimed_fraction"] < 1.0
    assert cov["combined_hypervolume"] > cov["nsga_hypervolume"]


def test_coverage_zero_reclaim_on_multi_point_fronts_when_nsga_dominates():
    """Multi-point counterpart to the zero case: when the NSGA front dominates the whole exact
    set, the union reclaims no volume — a genuine-geometry 0, not the single-point box-origin
    degeneracy."""
    prob = _problem(_OBJS)
    nsga = _run([(12.0, 3.0), (11.0, 2.5), (9.0, 1.8)], _OBJS)   # dominates every exact point below
    exact = _run([(10.0, 5.0), (8.0, 4.0), (6.0, 2.0)], _OBJS, solver="highs")
    cov = certify_against_exact(prob, nsga, exact)["coverage"]
    assert cov is not None
    assert cov["exact_reclaims"] == 0.0 and cov["reclaimed_fraction"] == 0.0


def test_coverage_handles_three_objectives():
    """Coverage normalization is per-axis, so it must be dimension-agnostic — a 3-objective front
    yields a valid in-[0,1] reclaim, not an indexing error."""
    objs = [Objective(name="Return", direction="maximize", aggregation="avg"),
            Objective(name="Risk", direction="minimize", aggregation="quadratic"),
            Objective(name="Cost", direction="minimize", aggregation="sum")]
    prob = _problem(objs)
    nsga = _run([(10.0, 5.0, 9.0), (8.0, 4.0, 7.0)], objs)
    exact = _run([(12.0, 3.0, 6.0), (11.0, 2.5, 8.0)], objs, solver="highs")
    cov = certify_against_exact(prob, nsga, exact)["coverage"]
    assert cov is not None
    assert 0.0 <= cov["reclaimed_fraction"] <= 1.0
    assert cov["combined_hypervolume"] >= cov["nsga_hypervolume"]


# ─── integration: real solver, MILP invariant is strict ───

def test_certify_milp_invariant_strict():
    """On a binary MILP the exact points are integer by construction, so the invariant holds
    strictly (no rounding artifact) and the audit runs end-to-end through a real exact solver."""
    pytest.importorskip("highspy")
    from engine.optimizer import optimize

    prob = _binary_milp()
    nsga = optimize(prob, seed=42)
    exact = optimize(prob, seed=42, solver="highs")
    c = certify_against_exact(prob, nsga, exact)

    assert c["exact_solver"] == "highs"
    assert c["invariant"]["holds"] is True                      # MILP: integer, never rounding-dominated
    assert c["invariant"]["exact_dominated_by_nsga"] == 0
    assert 0.0 <= c["dominance_audit"]["nsga_dominated_fraction"] <= 1.0
    assert set(c["corner_sharpening"]) == {o.name for o in prob.objectives}
    assert isinstance(c["recommendation"], str) and c["recommendation"]
    assert "binding_analysis" in c["next_steps"]                # MILP overlay → no exact duals


# ─── journey wiring: the certificate hands off (Pillar 1) ───

def test_certify_next_steps_qp_points_to_sensitivity():
    """A continuous/QP overlay's certificate points onward to `explore sensitivity` (duals) and
    the exact-overlay navigation — turning certify from a dead-end into a guided step."""
    prob = _problem(_OBJS)                                      # approach="proportional" (QP)
    c = certify_against_exact(prob, _run([(8.0, 3.0)], _OBJS),
                              _run([(10.0, 2.0)], _OBJS, solver="highs"))
    assert "next_steps" in c
    assert "sensitivity" in c["next_steps"] and 'source="exact"' in c["next_steps"]


def test_certify_next_steps_milp_points_to_binding_analysis():
    """A binary/MILP overlay's certificate points to binding_analysis — integer solutions carry
    no exact duals, so it must NOT send the agent to `explore sensitivity`."""
    objs = [Objective(name="Value", direction="maximize", aggregation="sum"),
            Objective(name="Cost", direction="minimize", aggregation="sum")]
    names = ["A", "B", "C", "D"]
    prob = Problem(name="t", approach="binary", objectives=objs,
                   options=[Option(name=n) for n in names],
                   scores=[Score(option=n, objective=o.name, value=1.0) for n in names for o in objs])
    c = certify_against_exact(prob, _run([(8.0, 3.0)], objs),
                              _run([(10.0, 2.0)], objs, solver="highs", exact=True))
    assert "binding_analysis" in c["next_steps"] and "sensitivity" not in c["next_steps"]


# ─── journey wiring: the read-side skill injects once per problem (regression) ───


@pytest.fixture
def srv_tmp_store(monkeypatch):
    """Isolated temp store + cleared skill-injection tracking, for driving the certify handler
    in mcp_server.server directly (mirrors the fixture in tests/test_server.py)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(srv, "store", Store(tmpdir))
        srv._injected_skills.clear()
        yield


def test_certify_injects_solution_interpreter_only_once_per_problem(srv_tmp_store):
    """Regression: the certify branch must MARK solution_interpreter injected after surfacing it,
    so a second certify on the same problem does not re-inject the full ~49KB skill body.

    Reproduces the post-re-arm state where certify is the first action to inject the skill — a
    structural model/update clears the injection flag while preserving both runs (server.py
    ~line 583). That is the only path where the double-inject surfaced; in the normal journey the
    first solve already injects+marks it, so certify suppresses correctly. Before the fix, both
    certify responses carried `_skill_guidance`; after it, only the first does.
    """
    prob = _problem(_OBJS)                                                   # proportional mean-variance QP
    prob.run = _run([(8.0, 3.0), (12.0, 5.0)], _OBJS)                        # exploratory NSGA frontier
    prob.exact_run = _run([(10.0, 2.0)], _OBJS, solver="highs")             # exact overlay
    srv.store.save(prob)
    pid = prob.problem_id

    assert not srv._was_injected(pid, "solution_interpreter")               # flag clear at certify time

    first = srv.explore(action="certify", problem_id=pid)
    assert "error" not in first
    assert first.get("_skill_guidance", {}).get("skill") == "solution_interpreter"  # surfaced once...

    second = srv.explore(action="certify", problem_id=pid)
    assert "error" not in second
    assert "_skill_guidance" not in second                                  # ...and never again


# ── Rounding bound dips + certification flag ──


def test_rounding_dip_under_objective_bound_is_flagged():
    """Whole-percent rounding of a continuous optimum can dip a point a hair under an
    objective_bound the LP itself honors (live case: StrategicValue 4.784 vs ≥ 4.8).
    Certify names the point instead of silently contradicting the model."""
    from engine.models import ObjectiveBoundConstraint

    p = _problem(_OBJS)
    p.constraints = [ObjectiveBoundConstraint(objective="Return", operator="min", value=4.8)]
    nsga = _run([(5.0, 3.0)], _OBJS)
    exact = _run([(4.784, 2.0), (4.85, 2.5)], _OBJS, solver="highs")
    out = certify_against_exact(p, nsga, exact)
    dips = out["rounding_bound_dips"]
    assert [d["solution_id"] for d in dips["dips"]] == [0]
    assert dips["dips"][0]["bound"] == "min 4.8"
    assert "rounding" in dips["note"]


def test_no_dips_key_when_all_points_honor_the_bounds():
    from engine.models import ObjectiveBoundConstraint

    p = _problem(_OBJS)
    p.constraints = [ObjectiveBoundConstraint(objective="Return", operator="min", value=4.8)]
    nsga = _run([(5.0, 3.0)], _OBJS)
    exact = _run([(4.9, 2.0)], _OBJS, solver="highs")
    assert "rounding_bound_dips" not in certify_against_exact(p, nsga, exact)


def test_binary_selections_never_report_rounding_dips():
    """Binary selections are integer by construction — no rounding, no dips key, even
    when a bounded MILP incumbent sits outside an objective_bound for other reasons."""
    from engine.models import ObjectiveBoundConstraint

    p = _problem(_OBJS)
    p.approach = Approach.binary
    p.constraints = [ObjectiveBoundConstraint(objective="Return", operator="min", value=4.8)]
    nsga = _run([(5.0, 3.0)], _OBJS)
    exact = _run([(4.784, 2.0)], _OBJS, solver="highs")
    assert "rounding_bound_dips" not in certify_against_exact(p, nsga, exact)


def test_exact_certified_true_on_continuous_path_without_exact_flag():
    """The LP/QP scalarization is exact by construction: a proportional overlay is
    certified even when exact=True was never requested (it's a MILP-only knob) —
    the field must not read as a failed certification (live-test confusion)."""
    p = _problem(_OBJS)
    nsga = _run([(5.0, 3.0)], _OBJS)
    exact = _run([(6.0, 2.0)], _OBJS, solver="highs", exact=False)
    assert certify_against_exact(p, nsga, exact)["exact_certified"] is True


def test_exact_certified_on_binary_requires_zero_gap():
    """A default binary MILP run accepts 0.1%-gap incumbents — certified only with
    exact=True."""
    p = _problem(_OBJS)
    p.approach = Approach.binary
    nsga = _run([(5.0, 3.0)], _OBJS)
    bounded = _run([(6.0, 2.0)], _OBJS, solver="highs", exact=False)
    zero_gap = _run([(6.0, 2.0)], _OBJS, solver="highs", exact=True)
    assert certify_against_exact(p, nsga, bounded)["exact_certified"] is False
    assert certify_against_exact(p, nsga, zero_gap)["exact_certified"] is True


# ─── frontier resolution (the sandwich bound) ───


def _dual_run(points, prices, objectives, solver="highs"):
    """An exact Run whose solutions carry return_floor duals for the swept objective.
    ``prices[i]`` is the cost-sense |λ| at point i (None = no sensitivity attached)."""
    from engine.models import ShadowPrice, SolutionSensitivity
    sols = []
    for i, p in enumerate(points):
        sens = None
        if prices[i] is not None:
            sens = SolutionSensitivity(shadow_prices=[
                ShadowPrice(name="Return", role="return_floor", shadow_price=prices[i])])
        sols.append(Solution(solution_id=i, selected_options=["A"], sensitivity=sens,
                             objective_values=dict(zip([o.name for o in objectives], p))))
    return Run(solutions=sols, solver=solver)


def test_frontier_resolution_none_without_duals():
    """No solver-exact duals on the overlay (MILP / hand-built) → no resolution claim.
    The block is None-and-omitted, matching coverage/completeness on shapes that can't
    carry it — never a guessed bound."""
    p = _problem(_OBJS)
    nsga = _run([(1.0, 1.0)], _OBJS)
    exact = _run([(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)], _OBJS, solver="highs")
    assert certify_against_exact(p, nsga, exact)["frontier_resolution"] is None


def test_frontier_resolution_sandwich_geometry_exact_values():
    """Hand-checkable parabola: Risk = Return² at Return 0, 1, 2, duals = |dv/du| = 0, 2, 4.
    Each segment's chord-to-tangent gap is 0.5 (tangent-intersection midpoints), so the
    certified resolution is 0.5 Risk = 12.5% of the frontier's Risk span."""
    p = _problem(_OBJS)
    nsga = _run([(1.0, 1.0)], _OBJS)
    exact = _dual_run([(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)], [0.0, 2.0, 4.0], _OBJS)
    fr = certify_against_exact(p, nsga, exact)["frontier_resolution"]

    assert fr is not None
    assert fr["points_used"] == 3 and fr["segments_dual_tightened"] == 2
    assert fr["span"] == {"Return": [0.0, 2.0]}
    assert fr["max_gap"]["Risk"] == pytest.approx(0.5)
    assert fr["max_gap"]["fraction_of_frontier_range"] == pytest.approx(0.125)
    assert "Risk" in fr["claim"] and fr["basis"] == "solver_exact_duals"


def test_frontier_resolution_monotonicity_floor_survives_missing_dual():
    """An unpriced point (no sensitivity — an anchor or degenerate corner) loosens the
    bound but never voids it: the monotonicity floor still bounds every segment. Same
    parabola with the middle point unpriced → max gap grows 0.5 → 1.0."""
    p = _problem(_OBJS)
    nsga = _run([(1.0, 1.0)], _OBJS)
    exact = _dual_run([(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)], [0.0, None, 4.0], _OBJS)
    fr = certify_against_exact(p, nsga, exact)["frontier_resolution"]

    assert fr is not None
    assert fr["max_gap"]["Risk"] == pytest.approx(1.0)
    # Both segments still carry at least one dual tangent from their priced endpoint.
    assert fr["segments_dual_tightened"] == 2


def test_frontier_resolution_names_widest_segment():
    """The widest gap is localized with its endpoint solution ids — self-certifying, and a
    future targeted-solve handle. Asymmetric duals put the widest gap on the flat side."""
    p = _problem(_OBJS)
    nsga = _run([(1.0, 1.0)], _OBJS)
    exact = _dual_run([(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)], [0.0, None, 4.0], _OBJS)
    fr = certify_against_exact(p, nsga, exact)["frontier_resolution"]

    seg = fr["widest_segment"]
    assert seg["Return"] == [0.0, 1.0] and seg["solution_ids"] == [1, 0]
    assert fr["max_gap"]["Risk"] == pytest.approx(1.0)  # the widest segment IS max_gap


def test_frontier_resolution_none_on_three_objectives():
    """The sandwich is a 2-objective read (one swept axis, one primary) — a 3-objective
    overlay returns None rather than a bound over a surface it didn't measure."""
    objs = _OBJS + [Objective(name="Effort", direction="minimize")]
    p = _problem(objs)
    nsga = _run([(1.0, 1.0, 1.0)], objs)
    exact = _run([(0.0, 0.0, 0.0), (2.0, 4.0, 1.0)], objs, solver="highs")
    assert certify_against_exact(p, nsga, exact)["frontier_resolution"] is None


def test_frontier_resolution_none_when_objectives_not_in_tension():
    """A flat primary axis (no tradeoff) has no resolution to certify."""
    p = _problem(_OBJS)
    nsga = _run([(1.0, 1.0)], _OBJS)
    exact = _dual_run([(0.0, 2.0), (2.0, 2.0)], [0.0, 0.0], _OBJS)
    assert certify_against_exact(p, nsga, exact)["frontier_resolution"] is None


def test_frontier_resolution_rides_the_recommendation():
    """A present bound earns one recommendation clause pointing at the claim."""
    p = _problem(_OBJS)
    nsga = _run([(1.0, 1.0)], _OBJS)
    exact = _dual_run([(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)], [0.0, 2.0, 4.0], _OBJS)
    cert = certify_against_exact(p, nsga, exact)
    assert "resolution-certified" in cert["recommendation"]


def test_frontier_resolution_end_to_end_qp():
    """Real explore-then-certify on a tiny 2-objective mean-variance QP: NSGA run, lean
    exact overlay (duals attached by the continuous path), certificate carries a sane
    bound. Inline fixture — every bundled example is 3+ objectives, and the sandwich is
    deliberately a 2-objective read (see `_frontier_resolution_block`)."""
    from engine.optimizer import certify_curated, optimize

    from engine.models import InteractionMatrix

    names = ["A", "B", "C", "D", "E"]
    table = {"Return": [8.0, 6.5, 5.0, 7.2, 4.0], "Risk": [9.0, 5.0, 2.0, 7.0, 1.5]}
    var = {"A": 9.0, "B": 5.0, "C": 2.0, "D": 7.0, "E": 1.5}
    cov = {a: {b: (var[a] if a == b else 0.6 * min(var[a], var[b]))
               for b in names} for a in names}
    p = Problem(
        name="mv", approach="proportional",
        objectives=[Objective(name="Return", direction="maximize", aggregation="avg"),
                    Objective(name="Risk", direction="minimize", aggregation="quadratic")],
        options=[Option(name=n) for n in names],
        scores=[Score(option=n, objective=o, value=table[o][i])
                for i, n in enumerate(names) for o in table],
        interaction_matrices=[InteractionMatrix(objective="Risk", entries=cov)],
    )
    nsga = optimize(p, seed=42)
    exact = certify_curated(p, nsga, solver="highs")
    fr = certify_against_exact(p, nsga, exact)["frontier_resolution"]

    assert fr is not None and fr["points_used"] >= 2
    frac = fr["max_gap"]["fraction_of_frontier_range"]
    assert 0.0 <= frac < 1.0
    assert fr["segments_dual_tightened"] >= 1
    assert "no feasible plan improves" in fr["claim"]
