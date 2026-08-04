"""Tests for the explorer — tradeoffs, compare, get_solutions, get_solution."""

import pytest

from engine.explorer import (
    _frontier_provenance,
    analyze_composition,
    compare_solutions,
    get_solution,
    get_solutions,
    get_tradeoffs,
    marginal_analysis,
    option_selection_stats,
)
from engine.models import (
    CardinalityConstraint,
    Feedback,
    Objective,
    ObjectiveBoundConstraint,
    Option,
    Problem,
    Run,
    ScenarioRun,
    Score,
    Solution,
    _content_signature,
)
from engine.optimizer import optimize


@pytest.fixture
def solved_problem():
    p = Problem(
        objectives=[
            Objective(name="Revenue", direction="maximize", unit="$"),
            Objective(name="Effort", direction="minimize", unit="weeks"),
        ],
        options=[Option(name=n) for n in ["A", "B", "C", "D", "E"]],
        scores=[
            Score(option="A", objective="Revenue", value=8),
            Score(option="A", objective="Effort", value=5),
            Score(option="B", objective="Revenue", value=6),
            Score(option="B", objective="Effort", value=3),
            Score(option="C", objective="Revenue", value=9),
            Score(option="C", objective="Effort", value=7),
            Score(option="D", objective="Revenue", value=4),
            Score(option="D", objective="Effort", value=2),
            Score(option="E", objective="Revenue", value=7),
            Score(option="E", objective="Effort", value=4),
        ],
        constraints=[CardinalityConstraint(min=2, max=3)],
    )
    p.run = optimize(p)
    return p


class TestGetTradeoffs:
    def test_returns_expected_keys(self, solved_problem):
        result = get_tradeoffs(solved_problem)
        assert "total_solutions" in result
        assert "objective_ranges" in result
        assert "key_tradeoffs" in result
        assert "extreme_solutions" in result
        assert "balanced_solution" in result

    def test_objective_ranges(self, solved_problem):
        result = get_tradeoffs(solved_problem)
        assert "Revenue" in result["objective_ranges"]
        assert "Effort" in result["objective_ranges"]
        rev_range = result["objective_ranges"]["Revenue"]
        assert rev_range["min"] <= rev_range["max"]

    def test_extreme_solutions(self, solved_problem):
        result = get_tradeoffs(solved_problem)
        assert "extreme_Revenue" in result["extreme_solutions"]
        assert "extreme_Effort" in result["extreme_solutions"]

    def test_balanced_solution_has_options(self, solved_problem):
        result = get_tradeoffs(solved_problem)
        balanced = result["balanced_solution"]
        assert "selected_options" in balanced
        assert len(balanced["selected_options"]) > 0

    def test_key_tradeoffs_correlations(self, solved_problem):
        result = get_tradeoffs(solved_problem)
        for t in result["key_tradeoffs"]:
            assert "correlation" in t
            assert -1.0 <= t["correlation"] <= 1.0

    def test_no_run_raises(self):
        p = Problem()
        with pytest.raises(ValueError, match="No run found"):
            get_tradeoffs(p)


class TestCompare:
    def test_compare_two_solutions(self, solved_problem):
        result = compare_solutions(solved_problem, [1, 2])
        assert "solutions" in result
        assert "shared_options" in result
        assert "differentiating_options" in result
        assert "tradeoff_summary" in result
        assert len(result["solutions"]) == 2

    def test_compare_nonexistent_solution(self, solved_problem):
        with pytest.raises(ValueError, match="not found"):
            compare_solutions(solved_problem, [1, 9999])

    def test_tradeoff_summary_has_all_objectives(self, solved_problem):
        result = compare_solutions(solved_problem, [1, 2])
        assert "Revenue" in result["tradeoff_summary"]
        assert "Effort" in result["tradeoff_summary"]


class TestGetSolutions:
    def test_returns_all_solutions(self, solved_problem):
        result = get_solutions(solved_problem)
        assert result["total_solutions"] == len(solved_problem.run.solutions)
        assert len(result["solutions"]) == result["total_solutions"]

    def test_has_run_id(self, solved_problem):
        result = get_solutions(solved_problem)
        assert "run_id" in result

    def test_compact_by_default(self, solved_problem):
        """Default response is compact: no selected_options or allocations fields."""
        result = get_solutions(solved_problem)
        assert result["detail"] is False
        sol = result["solutions"][0]
        assert "objective_values" in sol
        assert "solution_id" in sol
        assert "selected_options" not in sol

    def test_detail_returns_full(self, solved_problem):
        """detail=True returns full Solution dump including selected_options."""
        result = get_solutions(solved_problem, detail=True)
        assert result["detail"] is True
        sol = result["solutions"][0]
        assert "selected_options" in sol

    def test_includes_parallel_coords_viz(self, solved_problem):
        """Listing carries a parallel-coords payload so chart hosts (web UI) can draw it."""
        result = get_solutions(solved_problem)
        assert "visualization" in result
        viz = result["viz_data"]
        assert viz["type"] == "parallel_coords"
        assert len(viz["series"]) == result["total_solutions"]
        assert len(viz["axes"]) == len(solved_problem.objectives)

    def test_no_run_raises(self):
        p = Problem()
        with pytest.raises(ValueError, match="No run found"):
            get_solutions(p)


class TestGetSolution:
    def test_returns_single_solution(self, solved_problem):
        result = get_solution(solved_problem, 1)
        assert result["solution_id"] == 1
        assert "selected_options" in result
        assert "objective_values" in result

    def test_nonexistent_id_raises(self, solved_problem):
        with pytest.raises(ValueError, match="not found"):
            get_solution(solved_problem, 9999)


class TestFrontierShape:
    def test_tradeoffs_include_frontier_shape(self, solved_problem):
        result = get_tradeoffs(solved_problem)
        assert "frontier_shape" in result

    def test_shape_entries_have_required_fields(self, solved_problem):
        result = get_tradeoffs(solved_problem)
        for entry in result["frontier_shape"]:
            assert "objectives" in entry
            assert "shape" in entry
            assert entry["shape"] in ("linear", "concave", "convex", "discontinuous")
            assert "confidence" in entry
            assert 0.0 <= entry["confidence"] <= 1.0


class TestObjectiveRedundancy:
    def test_tradeoffs_include_redundancy(self, solved_problem):
        result = get_tradeoffs(solved_problem)
        assert "objective_redundancy" in result
        assert isinstance(result["objective_redundancy"], list)

    def test_redundancy_entries_schema(self, solved_problem):
        result = get_tradeoffs(solved_problem)
        for entry in result["objective_redundancy"]:
            assert "objectives" in entry
            assert "classification" in entry
            assert entry["classification"] in (
                "independent", "linear_redundant", "strong_tradeoff",
                "redundant", "nonlinear_dependent",
            )
            assert "pearson" in entry
            assert -1.0 <= entry["pearson"] <= 1.0
            assert "flags" in entry
            assert "mi_reliable" in entry

    def test_redundant_same_direction_flagged(self):
        # Two maximize objectives perfectly aligned (Profit = 0.8 * Revenue) should
        # be `linear_redundant` — both getting better together.
        p = Problem(
            objectives=[
                Objective(name="Revenue", direction="maximize"),
                Objective(name="Profit", direction="maximize"),
                Objective(name="Effort", direction="minimize"),
            ],
            options=[Option(name=n) for n in ["A", "B", "C", "D", "E", "F"]],
            scores=[
                Score(option="A", objective="Revenue", value=10),
                Score(option="A", objective="Profit", value=8),
                Score(option="A", objective="Effort", value=3),
                Score(option="B", objective="Revenue", value=6),
                Score(option="B", objective="Profit", value=4.8),
                Score(option="B", objective="Effort", value=2),
                Score(option="C", objective="Revenue", value=9),
                Score(option="C", objective="Profit", value=7.2),
                Score(option="C", objective="Effort", value=5),
                Score(option="D", objective="Revenue", value=4),
                Score(option="D", objective="Profit", value=3.2),
                Score(option="D", objective="Effort", value=1),
                Score(option="E", objective="Revenue", value=7),
                Score(option="E", objective="Profit", value=5.6),
                Score(option="E", objective="Effort", value=4),
                Score(option="F", objective="Revenue", value=5),
                Score(option="F", objective="Profit", value=4.0),
                Score(option="F", objective="Effort", value=2),
            ],
            constraints=[CardinalityConstraint(min=2, max=4)],
        )
        p.run = optimize(p)
        result = get_tradeoffs(p)
        rp = [
            e for e in result["objective_redundancy"]
            if set(e["objectives"]) == {"Revenue", "Profit"}
        ]
        assert len(rp) == 1
        entry = rp[0]
        assert entry["pearson"] >= 0.7
        assert entry["classification"] in ("linear_redundant", "redundant")

    def test_mixed_direction_pair_not_redundant(self):
        # Revenue (max) and Effort (min) that covary in raw values are a genuine
        # tradeoff, NOT redundancy. Direction-normalized r should be negative.
        p = Problem(
            objectives=[
                Objective(name="Revenue", direction="maximize"),
                Objective(name="Effort", direction="minimize"),
                Objective(name="Satisfaction", direction="maximize"),
            ],
            options=[Option(name=n) for n in ["A", "B", "C", "D", "E", "F"]],
            scores=[
                # High-revenue options are also high-effort
                Score(option="A", objective="Revenue", value=100), Score(option="A", objective="Effort", value=10), Score(option="A", objective="Satisfaction", value=5),
                Score(option="B", objective="Revenue", value=20), Score(option="B", objective="Effort", value=2), Score(option="B", objective="Satisfaction", value=3),
                Score(option="C", objective="Revenue", value=80), Score(option="C", objective="Effort", value=8), Score(option="C", objective="Satisfaction", value=7),
                Score(option="D", objective="Revenue", value=30), Score(option="D", objective="Effort", value=3), Score(option="D", objective="Satisfaction", value=4),
                Score(option="E", objective="Revenue", value=60), Score(option="E", objective="Effort", value=6), Score(option="E", objective="Satisfaction", value=6),
                Score(option="F", objective="Revenue", value=40), Score(option="F", objective="Effort", value=4), Score(option="F", objective="Satisfaction", value=4),
            ],
            constraints=[CardinalityConstraint(min=2, max=4)],
        )
        p.run = optimize(p)
        result = get_tradeoffs(p)
        re = [
            e for e in result["objective_redundancy"]
            if set(e["objectives"]) == {"Revenue", "Effort"}
        ]
        assert len(re) == 1
        entry = re[0]
        # Direction-normalized r must be negative (genuine tradeoff)
        assert entry["pearson"] < 0
        # Must NOT be flagged as redundant
        assert entry["classification"] != "linear_redundant"
        assert entry["classification"] != "redundant"


class TestBindingAnalysis:
    def test_tradeoffs_include_binding_analysis_key(self, solved_problem):
        result = get_tradeoffs(solved_problem)
        assert "binding_analysis" in result
        assert isinstance(result["binding_analysis"], list)

    def test_cardinality_binding_has_shadow_prices(self, solved_problem):
        # The fixture uses CardinalityConstraint(min=2, max=3), likely binding.
        result = get_tradeoffs(solved_problem)
        cardinality_entries = [
            e for e in result["binding_analysis"]
            if e["constraint_type"] == "cardinality"
        ]
        assert len(cardinality_entries) >= 1
        entry = cardinality_entries[0]
        assert 0.0 <= entry["binding_fraction"] <= 1.0
        assert entry["near_binding_count"] >= 1
        # Either shadow prices populated or a note explaining why not
        assert entry["shadow_prices"] or entry["note"]
        for sp in entry["shadow_prices"]:
            assert "objective" in sp
            assert "gain_per_additional_slot" in sp

    def test_objective_bound_binding_is_detected(self):
        p = Problem(
            objectives=[
                Objective(name="Revenue", direction="maximize"),
                Objective(name="Effort", direction="minimize"),
            ],
            options=[Option(name=n) for n in ["A", "B", "C", "D", "E"]],
            scores=[
                Score(option="A", objective="Revenue", value=10),
                Score(option="A", objective="Effort", value=8),
                Score(option="B", objective="Revenue", value=7),
                Score(option="B", objective="Effort", value=3),
                Score(option="C", objective="Revenue", value=9),
                Score(option="C", objective="Effort", value=6),
                Score(option="D", objective="Revenue", value=5),
                Score(option="D", objective="Effort", value=2),
                Score(option="E", objective="Revenue", value=8),
                Score(option="E", objective="Effort", value=5),
            ],
            constraints=[
                CardinalityConstraint(min=2, max=3),
                ObjectiveBoundConstraint(objective="Effort", operator="max", value=10),
            ],
        )
        p.run = optimize(p)
        result = get_tradeoffs(p)
        # binding_analysis must not crash with objective_bound constraint
        assert "binding_analysis" in result
        ob_entries = [
            e for e in result["binding_analysis"]
            if e["constraint_type"] == "objective_bound"
        ]
        # May or may not be binding depending on the frontier, but if present
        # the schema should be valid
        for entry in ob_entries:
            assert "constraint" in entry
            assert "binding_fraction" in entry
            assert entry["shadow_prices"] or entry["note"]


class TestFrontierProvenance:
    """`_frontier_provenance` labels which frontier a result came from, so a heuristic
    frontier is never silently passed off as the exact overlay — the failure mode when
    `explore`'s `source` is omitted or dropped before it reaches the engine.
    """

    @staticmethod
    def _run(solver, rid):
        return Run(
            run_id=rid, solver=solver,
            solutions=[Solution(solution_id=1, selected_options=["A"],
                                objective_values={"x": 1.0})],
        )

    def test_exact_run_labeled_exact(self):
        run = self._run("highs", "exact-1")
        prov = _frontier_provenance(Problem(exact_run=run), run)
        assert prov["kind"] == "exact"
        assert prov["solver"] == "highs"
        assert prov["run_id"] == "exact-1"
        assert "exact_overlay_available" not in prov  # nothing better to point at

    def test_heuristic_with_overlay_advertises_it(self):
        nsga = self._run("nsga-iii", "nsga-1")
        exact = self._run("highs", "exact-1")
        prov = _frontier_provenance(Problem(run=nsga, exact_run=exact), nsga)
        assert prov["kind"] == "heuristic"
        assert prov["solver"] == "nsga-iii"
        assert prov["run_id"] == "nsga-1"
        assert prov["exact_overlay_available"] is True
        assert 'source="exact"' in prov["hint"]

    def test_overlay_hint_suppressed_for_scenarios(self):
        # Scenario runs are NSGA-only and the base-case exact overlay doesn't apply to them,
        # so the "exact overlay available" hint must not leak onto scenario results.
        nsga = self._run("nsga-iii", "nsga-1")
        exact = self._run("highs", "exact-1")
        prov = _frontier_provenance(
            Problem(run=nsga, exact_run=exact), nsga, scenario="budget_cut")
        assert prov["kind"] == "heuristic"
        assert "exact_overlay_available" not in prov

    def test_heuristic_without_overlay_has_no_hint(self):
        nsga = self._run("nsga-ii", "nsga-1")
        prov = _frontier_provenance(Problem(run=nsga), nsga)
        assert prov["kind"] == "heuristic"
        assert "exact_overlay_available" not in prov

    def test_every_labeled_action_carries_provenance(self, solved_problem):
        # The label is a per-site convention, so enumerate every analytics action that resolves
        # a frontier — a future call site shipping without frontier_source fails CI here, not in
        # production. (solutions/solution return the dict directly; tradeoffs/compare/
        # marginal_analysis flow through the server's _format_explore passthrough.)
        p = solved_problem
        ids = [s["solution_id"] for s in get_solutions(p)["solutions"]]
        results = {
            "tradeoffs": get_tradeoffs(p),
            "solutions": get_solutions(p),
            "solution": get_solution(p, ids[0]),
            "compare": compare_solutions(p, ids[:2]),
            "marginal_analysis": marginal_analysis(p),
        }
        for action, result in results.items():
            assert "frontier_source" in result, f"{action} dropped frontier_source"
            assert result["frontier_source"]["kind"] == "heuristic", action
            assert result["frontier_source"]["run_id"] == p.run.run_id, action


class TestScatterCertification:
    """The frontier scatter `viz_data` carries certification: `provenance` (heuristic vs
    exact-certified) on every frontier, and an `exact_overlay` (the certified points + the
    heuristic ids they dominate) when a heuristic base-case frontier has an exact overlay — so the
    chart can denote what's certified. Reuses the certify dominance logic and mirrors the
    `_frontier_provenance` scenario guard, so the overlay never leaks onto a scenario frontier.
    """

    @staticmethod
    def _run(points, names, solver="nsga-ii", exact=False):
        """A Run whose solutions carry the given per-objective value tuples (id = list index)."""
        sols = [Solution(solution_id=i, selected_options=["A"],
                         objective_values=dict(zip(names, p))) for i, p in enumerate(points)]
        return Run(solutions=sols, solver=solver, exact=exact)

    @staticmethod
    def _mean_variance_problem():
        # get_tradeoffs reads objectives + run + exact_run; scores aren't needed for the scatter.
        objs = [Objective(name="Return", direction="maximize", aggregation="avg"),
                Objective(name="Risk", direction="minimize", aggregation="quadratic")]
        return Problem(name="t", approach="proportional", objectives=objs,
                       options=[Option(name=n) for n in ["A", "B"]])

    def test_heuristic_frontier_provenance_no_overlay(self, solved_problem):
        # A plain NSGA frontier with no exact run: labeled heuristic, nothing to certify.
        viz = get_tradeoffs(solved_problem)["viz_data"]
        assert viz["provenance"]["kind"] == "heuristic"
        assert viz["provenance"]["solver"] == solved_problem.run.solver
        assert viz["provenance"]["exact_certified"] is False
        assert "exact_overlay" not in viz

    def test_exact_view_all_certified_no_overlay(self, solved_problem):
        # source="exact" renders the exact frontier itself — every point certified, no overlay.
        p = solved_problem
        p.exact_run = self._run([(9.0, 2.0), (7.0, 1.0)], ["Revenue", "Effort"],
                                solver="highs", exact=True)
        viz = get_tradeoffs(p, source="exact")["viz_data"]
        assert viz["provenance"]["kind"] == "exact"
        assert viz["provenance"]["solver"] == "highs"
        assert viz["provenance"]["exact_certified"] is True
        assert "exact_overlay" not in viz

    def test_heuristic_with_exact_overlay(self):
        # Heuristic frontier + an exact overlay that dominates two heuristic points and sharpens
        # the Risk corner past the heuristic range. Geometry mirrors test_certify.
        p = self._mean_variance_problem()
        names = ["Return", "Risk"]
        # Mutually non-dominated heuristic frontier (Return↑, Risk↓).
        p.run = self._run([(8.0, 3.0), (12.0, 5.0), (10.0, 4.0)], names, solver="nsga-ii")
        p.exact_run = self._run([(10.0, 2.0)], names, solver="highs", exact=True)
        viz = get_tradeoffs(p)["viz_data"]

        assert viz["provenance"]["kind"] == "heuristic"
        ov = viz["exact_overlay"]
        assert ov["solver"] == "highs"
        assert ov["exact_run_id"] == p.exact_run.run_id
        assert ov["exact_certified"] is True
        assert [pt["solution_id"] for pt in ov["points"]] == [0]
        assert ov["points"][0]["values"] == {"Return": 10.0, "Risk": 2.0}
        # (10,2) dominates the heuristic (8,3) [id 0] and (10,4) [id 2, equal Return, lower Risk]
        # but not the (12,5) trade-off [id 1].
        assert set(ov["dominated_ids"]) == {0, 2}
        # The exact Risk (2.0) sits below the heuristic Risk min (3.0) → the axis widens to it,
        # so the sharpened corner renders in-frame rather than clipped at the plot edge.
        risk = next(o for o in viz["objectives"] if o["name"] == "Risk")
        assert risk["min"] == 2.0
        ret = next(o for o in viz["objectives"] if o["name"] == "Return")
        assert ret["max"] == 12.0  # exact Return (10) within the heuristic range → unchanged

    def test_dominated_ids_match_certify_dominance(self):
        # Same inputs → same verdict: the scatter's dominated_ids and certify's dominance
        # count must agree on identical (run, exact_run) pairs. A live test read the two at
        # different times (the exact overlay was re-solved in between) and saw 9 vs 29 —
        # legitimate then, but only because the semantics are identical per overlay. Pin that.
        from engine.explorer import certify_against_exact
        p = self._mean_variance_problem()
        names = ["Return", "Risk"]
        p.run = self._run([(8.0, 3.0), (12.0, 5.0), (10.0, 4.0), (6.0, 2.5)], names,
                          solver="nsga-ii")
        p.exact_run = self._run([(10.0, 2.0), (12.0, 4.5)], names, solver="highs", exact=True)
        ov = get_tradeoffs(p)["viz_data"]["exact_overlay"]
        cert = certify_against_exact(p, p.run, p.exact_run)
        assert len(set(ov["dominated_ids"])) == cert["dominance_audit"]["nsga_dominated_by_exact"]
        # And both sides name the overlay they measured against, so a stale read is detectable.
        assert ov["exact_run_id"] == cert["exact_run_id"] == p.exact_run.run_id

    def test_scenario_frontier_does_not_leak_overlay(self, solved_problem):
        # A scenario frontier is NSGA-only; the base-case exact overlay must not attach to it
        # (mirrors `_frontier_provenance`'s scenario guard).
        p = solved_problem
        p.exact_run = self._run([(10.0, 2.0)], ["Revenue", "Effort"], solver="highs", exact=True)
        p.scenario_run = ScenarioRun(scenario_runs={
            "s1": self._run([(8.0, 5.0), (6.0, 3.0), (9.0, 7.0)], ["Revenue", "Effort"])
        })
        viz = get_tradeoffs(p, scenario="s1")["viz_data"]
        assert viz["provenance"]["kind"] == "heuristic"
        assert "exact_overlay" not in viz


# ─── Composition / mining (knowledge-discovery pillar) ───


def _csol(sid, opts, objs, alloc=None):
    s = Solution(solution_id=sid, selected_options=opts, objective_values=objs, allocations=alloc)
    s.content_signature = _content_signature(opts, alloc)
    return s


def _problem_with_run(sols, options, objectives, approach="binary"):
    p = Problem(approach=approach, objectives=objectives, options=[Option(name=o) for o in options])
    p.run = Run(solutions=sols)
    return p


class TestOptionSelectionStats:
    def test_binary_counts_and_pct(self):
        sols = [
            _csol(0, ["A", "B"], {"X": 1}),
            _csol(1, ["A", "C"], {"X": 2}),
            _csol(2, ["A", "B"], {"X": 3}),
        ]
        stats = option_selection_stats(sols, "binary")
        by = {e["option"]: e for e in stats}
        assert by["A"]["selection_count"] == 3
        assert by["A"]["selection_pct"] == 1.0
        assert by["B"]["selection_count"] == 2
        assert by["C"]["selection_count"] == 1
        assert stats[0]["option"] == "A"  # sorted desc by pct
        assert "mean_weight" not in by["A"]  # binary → no weight

    def test_proportional_weights(self):
        sols = [
            _csol(0, ["A", "B"], {"X": 1}, alloc={"A": 60, "B": 40}),
            _csol(1, ["A", "B"], {"X": 2}, alloc={"A": 20, "B": 80}),
        ]
        by = {e["option"]: e for e in option_selection_stats(sols, "proportional")}
        assert by["A"]["selection_count"] == 2
        assert by["A"]["mean_weight"] == 40.0
        assert by["A"]["mean_weight_if_included"] == 40.0
        assert by["B"]["mean_weight"] == 60.0


class TestAnalyzeComposition:
    def _objs(self):
        return [Objective(name="X", direction="maximize"), Objective(name="Y", direction="maximize")]

    def test_design_principles_always_never(self):
        sols = [
            _csol(0, ["A", "B", "C"], {"X": 3, "Y": 1}),
            _csol(1, ["A", "B", "C"], {"X": 2, "Y": 2}),
            _csol(2, ["A", "E"], {"X": 1, "Y": 3}),
        ]
        p = _problem_with_run(sols, ["A", "B", "C", "D", "E"], self._objs())
        r = analyze_composition(p)
        types = {(pp["type"], tuple(pp["options"])) for pp in r["design_principles"]}
        assert ("always", ("A",)) in types
        assert ("never", ("D",)) in types
        assert r["scope"]["n_solutions"] == 3
        assert r["feedback_rules"]["available"] is False
        assert isinstance(r["co_occurrence"], list)

    def test_decision_space_clusters(self):
        sols = [
            _csol(0, ["A", "B"], {"X": 1, "Y": 9}),
            _csol(1, ["A", "B"], {"X": 2, "Y": 8}),
            _csol(2, ["A", "B"], {"X": 1, "Y": 9}),
            _csol(3, ["C", "D"], {"X": 9, "Y": 1}),
            _csol(4, ["C", "D"], {"X": 8, "Y": 2}),
            _csol(5, ["C", "D"], {"X": 9, "Y": 1}),
        ]
        p = _problem_with_run(sols, ["A", "B", "C", "D"], self._objs())
        clusters = analyze_composition(p)["clusters"]
        assert len(clusters) >= 2
        defining = [set(c["defining_options"]) for c in clusters]
        assert any({"A", "B"} & d for d in defining)
        assert any({"C", "D"} & d for d in defining)

    def test_curated_subset_via_ids(self):
        sols = [
            _csol(0, ["A", "B"], {"X": 1, "Y": 9}),
            _csol(1, ["C", "D"], {"X": 9, "Y": 1}),
        ]
        p = _problem_with_run(sols, ["A", "B", "C", "D"], self._objs())
        r = analyze_composition(p, solution_ids=[0])
        assert r["scope"]["set"] == "curated"
        assert r["scope"]["n_solutions"] == 1

    def test_feedback_rules_separate_liked_disliked(self):
        # Liked all include A (not B); disliked all include B (not A) — a clean separator.
        sols = [
            _csol(0, ["A", "C"], {"X": 5, "Y": 5}),
            _csol(1, ["A", "D"], {"X": 6, "Y": 4}),
            _csol(2, ["B", "C"], {"X": 4, "Y": 6}),
            _csol(3, ["B", "D"], {"X": 3, "Y": 7}),
        ]
        p = _problem_with_run(sols, ["A", "B", "C", "D"], self._objs())
        p.feedback = [
            Feedback(content_signature=sols[0].content_signature, rating=5),
            Feedback(content_signature=sols[1].content_signature, rating=4),
            Feedback(content_signature=sols[2].content_signature, rating=1),
            Feedback(content_signature=sols[3].content_signature, rating=2),
        ]
        fr = analyze_composition(p)["feedback_rules"]
        assert fr["available"] is True
        assert any("A" in rule["condition"] for rule in fr["rules"])
        assert fr["n_liked"] == 2 and fr["n_disliked"] == 2

    def _rated(self, ratings: dict[int, int]):
        """A 6-solution frontier where the liked side is exactly the A-including ones,
        rated per the {index: rating} map."""
        sols = [
            _csol(0, ["A", "C"], {"X": 5, "Y": 5}),
            _csol(1, ["A", "D"], {"X": 6, "Y": 4}),
            _csol(2, ["A", "E"], {"X": 7, "Y": 3}),
            _csol(3, ["B", "C"], {"X": 4, "Y": 6}),
            _csol(4, ["B", "D"], {"X": 3, "Y": 7}),
            _csol(5, ["B", "E"], {"X": 2, "Y": 8}),
        ]
        p = _problem_with_run(sols, ["A", "B", "C", "D", "E"], self._objs())
        p.feedback = [Feedback(content_signature=sols[i].content_signature, rating=r)
                      for i, r in ratings.items()]
        return p

    def test_thin_evidence_is_named_on_the_rules_block(self):
        """Observed twice: separation 1.0 / coverage 1.0 learned from ONE liked solution,
        presented as a clean rule (one was later falsified by certify). The counts and the
        hypothesis-not-constraint note make the evidence size readable."""
        r = analyze_composition(self._rated({0: 5, 3: 1, 4: 2}))
        fr = r["feedback_rules"]
        assert fr["available"] is True and fr["rules"]
        assert fr["n_liked"] == 1 and fr["n_disliked"] == 2
        assert fr["rules"][0]["separation"] == 1.0     # by arithmetic, off one solution
        assert "Thin evidence: 1 liked, 2 disliked" in fr["note"]
        assert "hypothesis to confirm" in fr["note"]
        assert "region_bound" in fr["note"]
        assert "thin evidence" in r["visualization"]
        assert "(1 liked / 2 disliked)" in r["visualization"]

    def test_thick_evidence_carries_counts_without_the_note(self):
        r = analyze_composition(self._rated({0: 5, 1: 5, 2: 4, 3: 1, 4: 2, 5: 1}))
        fr = r["feedback_rules"]
        assert fr["n_liked"] == 3 and fr["n_disliked"] == 3
        assert "Thin evidence" not in fr["note"]
        assert "candidate latent constraints" in fr["note"]
        assert "thin evidence" not in r["visualization"]

    def test_counts_present_when_one_side_is_missing(self):
        fr = analyze_composition(self._rated({0: 5, 1: 4}))["feedback_rules"]
        assert fr["available"] is False
        assert fr["n_liked"] == 2 and fr["n_disliked"] == 0

    def test_off_frontier_ratings_are_named_not_vanished(self):
        """Observed live: dislikes linked to plans the read frontier wasn't carrying read
        as n_disliked: 0 with nothing saying where they went. They can't feed rule
        induction, but the block names them (`n_rated_off_frontier` + note), and the
        advice points only at levers composition actually has (`source`)."""
        p = self._rated({0: 5, 1: 4})
        p.feedback.append(Feedback(content_signature="feedbeefcafe", rating=1))
        p.feedback.append(Feedback(content_signature="deadbeefcafe", rating=2))
        fr = analyze_composition(p)["feedback_rules"]
        assert fr["available"] is False
        assert fr["n_liked"] == 2 and fr["n_disliked"] == 0
        assert fr["n_rated_off_frontier"] == 2
        assert "2 rated solution(s) sit on a frontier" in fr["note"]
        assert "scenario" not in fr["note"]  # composition takes source, not scenario

    def test_on_frontier_ratings_carry_no_off_frontier_field(self):
        fr = analyze_composition(self._rated({0: 5, 3: 1}))["feedback_rules"]
        assert "n_rated_off_frontier" not in fr


def test_tradeoffs_has_option_selection(solved_problem):
    r = get_tradeoffs(solved_problem)
    assert "option_selection" in r
    assert all("selection_pct" in e for e in r["option_selection"])


def test_solution_has_option_context(solved_problem):
    sols = get_solutions(solved_problem)["solutions"]
    sol = get_solution(solved_problem, sols[0]["solution_id"])
    assert "option_context" in sol
    assert all(0.0 <= v <= 1.0 for v in sol["option_context"].values())


class TestShadowPriceTrend:
    """Size-aware trend: frontier-length row dumps get downsampled, never shipped whole
    (the live rationing test returned 182 rows); -0.0 is folded at the display layer."""

    @staticmethod
    def _sols(n):
        from engine.models import ShadowPrice, SolutionSensitivity
        return [Solution(solution_id=i, selected_options=["A"],
                         objective_values={"Return": float(i)},
                         sensitivity=SolutionSensitivity(shadow_prices=[
                             ShadowPrice(name="Return", role="linear_floor",
                                         shadow_price=float(i))]))
                for i in range(n)]

    def test_small_frontier_passes_whole(self):
        from engine.explorer import _shadow_price_trend
        trend, elided = _shadow_price_trend(self._sols(5))
        assert len(trend) == 5 and elided is None

    def test_portfolio_scale_downsampled_endpoints_kept(self):
        from engine.explorer import _TREND_MAX_POINTS, _shadow_price_trend
        trend, elided = _shadow_price_trend(self._sols(182))
        assert len(trend) == _TREND_MAX_POINTS
        assert trend[0]["solution_id"] == 0 and trend[-1]["solution_id"] == 181
        assert elided["total_points"] == 182 and elided["shown"] == _TREND_MAX_POINTS
        assert "downsampled" in elided["note"]

    def test_negative_zero_folded(self):
        import math
        from engine.models import ShadowPrice, SolutionSensitivity
        from engine.explorer import _shadow_price_trend
        s = Solution(solution_id=0, selected_options=["A"], objective_values={"Return": 1.0},
                     sensitivity=SolutionSensitivity(shadow_prices=[
                         ShadowPrice(name="Return", role="linear_floor", shadow_price=-0.0)]))
        trend, _ = _shadow_price_trend([s])
        assert math.copysign(1.0, trend[0]["shadow_price"]) == 1.0


class TestClustersSummary:
    """Cluster sizes without a partition-quality signal read alike: a 104/1/1/1 split and a
    meaningful 155/19 both come back as "4 families" / "2 families" and nothing else."""

    def test_lopsided_split_is_named_as_one_strategy(self):
        from engine.explorer import _clusters_summary
        s = _clusters_summary([{"size": 104}, {"size": 1}, {"size": 1}, {"size": 1}])
        assert s["n_families"] == 4 and s["largest_family_share"] == 0.97
        assert "corner variants" in s["note"] and "not a menu" in s["note"]

    def test_real_split_carries_the_share_without_a_note(self):
        from engine.explorer import _clusters_summary
        s = _clusters_summary([{"size": 155}, {"size": 19}])
        assert s["n_families"] == 2 and s["largest_family_share"] == 0.89
        assert "note" not in s

    def test_no_clusters_no_summary(self):
        from engine.explorer import _clusters_summary
        assert _clusters_summary([]) is None

    def test_composition_payload_carries_the_summary(self):
        objs = [Objective(name="X", direction="maximize"), Objective(name="Y", direction="maximize")]
        # One strategy (A+B) with corner variants, plus a single outlier.
        sols = [_csol(i, ["A", "B"], {"X": 1 + i * 0.1, "Y": 9 - i * 0.1}) for i in range(10)]
        sols.append(_csol(10, ["C", "D"], {"X": 9, "Y": 1}))
        p = _problem_with_run(sols, ["A", "B", "C", "D"], objs)
        r = analyze_composition(p)
        s = r["clusters_summary"]
        assert s["n_families"] == len(r["clusters"])
        assert s["largest_family_share"] >= 0.9 and "corner variants" in s["note"]
        assert "largest holds" in r["visualization"]


class TestCompareRunsCoverageElision:
    """compare_runs' coverage table is options × runs — 360 rows on the 180-option
    portfolio. It elides the way solve's own option_coverage does, and ranks by the
    quantity the action is about: how much each option MOVED between the runs."""

    def _two_runs(self, n_options=70):
        names = [f"opt{i:03d}" for i in range(n_options)]
        objs = [Objective(name="X", direction="maximize")]
        # Run A holds every option once; run B drops the last 5 — those are the movers.
        movers, steady = names[-5:], names[:-5]
        a = Run(solutions=[_csol(0, names, {"X": 1.0})])
        b = Run(solutions=[_csol(0, steady, {"X": 2.0})])
        p = Problem(objectives=objs, options=[Option(name=o) for o in names])
        p.runs = [a]
        p.run = b
        return p, a, b, movers, steady

    def test_head_keeps_the_options_that_moved(self):
        from engine.explorer import compare_runs
        from engine.metrics import _MAX_COVERAGE_RETURNED
        p, a, b, movers, steady = self._two_runs()
        out = compare_runs(p, [a.run_id, b.run_id])
        kept = set(out["option_coverage"][a.run_id])
        assert len(kept) == _MAX_COVERAGE_RETURNED
        assert set(movers) <= kept, "an option that changed must never be elided"
        assert out["option_coverage"][b.run_id].keys() == kept, "same head on both runs"

    def test_elision_says_absence_is_not_zero(self):
        from engine.explorer import compare_runs
        from engine.metrics import _MAX_COVERAGE_RETURNED
        p, a, b, movers, steady = self._two_runs()
        elided = compare_runs(p, [a.run_id, b.run_id])["option_coverage_elided"]
        assert elided["shown"] == _MAX_COVERAGE_RETURNED and elided["total_options"] == 70
        assert elided["tail_max_delta"] == 0, "everything that moved is in the head"
        assert "NOT a coverage of zero" in elided["note"]
        assert "holds the same count in all compared runs" in elided["note"]
        assert "spread in selection count" in elided["ranked_by"]
        # The pointer must be one this action can deliver: composition resolves
        # run/exact/scenario, never a historical run_id — so it is scoped to the current
        # frontier rather than promising per-option rates for an archived run.
        assert "CURRENT frontier" in elided["note"]

    def test_ranking_and_prose_generalize_past_a_pair(self):
        """`run_ids` takes 2+, so the spread is max−min over EVERY compared run. The
        ranking already generalized; the prose must not hardcode two runs either."""
        from engine.explorer import _elide_coverage_diff
        names = [f"opt{i:03d}" for i in range(70)]
        # Three runs: one option swings only between the first and the third.
        a = {o: 5 for o in names}
        b = {o: 5 for o in names}
        c = {o: (20 if o == "opt069" else 5) for o in names}
        kept, elided = _elide_coverage_diff({"a": a, "b": b, "c": c}, names)
        assert "opt069" in kept["a"], "a swing visible only in a third run must survive"
        assert elided["tail_max_delta"] == 0
        assert "both runs" not in elided["note"]
        assert "all compared runs" in elided["note"]

    def test_note_states_the_tail_when_the_tail_moved(self):
        """A nonzero tail must report how far an elided option could have moved."""
        from engine.explorer import _elide_coverage_diff
        names = [f"opt{i:03d}" for i in range(70)]
        a = {o: 10 for o in names}
        b = {o: 10 - min(i, 3) for i, o in enumerate(reversed(names))}
        _, elided = _elide_coverage_diff({"a": a, "b": b}, names)
        assert elided["tail_max_delta"] == 3
        assert "at most 3 plan(s)" in elided["note"]

    def test_small_problem_passes_through_whole(self):
        from engine.explorer import compare_runs
        p, a, b, _, _ = self._two_runs(n_options=8)
        out = compare_runs(p, [a.run_id, b.run_id])
        assert "option_coverage_elided" not in out
        assert len(out["option_coverage"][a.run_id]) == 8
