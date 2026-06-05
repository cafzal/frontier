"""Tests for the explorer — tradeoffs, compare, get_solutions, get_solution."""

import pytest

from engine.explorer import (
    _frontier_provenance,
    compare_solutions,
    get_solution,
    get_solutions,
    get_tradeoffs,
    marginal_analysis,
)
from engine.models import (
    CardinalityConstraint,
    Objective,
    ObjectiveBoundConstraint,
    Option,
    Problem,
    Run,
    Score,
    Solution,
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
