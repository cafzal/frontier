"""Edge-case tests for optimizer internals — quality, hypervolume, _evaluate, scaling."""

import numpy as np
import pytest

from engine.models import (
    BoundOperator,
    CardinalityConstraint,
    ForceExcludeConstraint,
    ForceIncludeConstraint,
    Objective,
    ObjectiveBoundConstraint,
    Option,
    Problem,
    QualityIndicators,
    Score,
    Solution,
)
from engine.optimizer import (
    _approx_hypervolume,
    _compute_quality,
    _FrontierProblem,
    analyze_infeasibility,
    optimize,
    validate,
)


def _make_problem(**overrides):
    defaults = dict(
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
    defaults.update(overrides)
    return Problem(**defaults)


# ─── _FrontierProblem._evaluate ───


class TestFrontierProblemEvaluate:
    def _make_pymoo_problem(self, **kwargs):
        defaults = dict(
            n_options=5,
            score_matrix=np.array([
                [8, 5], [6, 3], [9, 7], [4, 2], [7, 4],
            ], dtype=float),
            objectives=[
                Objective(name="Revenue", direction="maximize"),
                Objective(name="Effort", direction="minimize"),
            ],
            forced_in=set(),
            forced_out=set(),
            cardinality_min=1,
            cardinality_max=5,
            obj_bounds=[],
        )
        defaults.update(kwargs)
        return _FrontierProblem(**defaults)

    def test_objective_negation_for_maximize(self):
        """Maximize objectives should have F values negated."""
        prob = self._make_pymoo_problem()
        X = np.array([[1, 0, 0, 0, 0]], dtype=float)  # Select only A
        out = {}
        prob._evaluate(X, out)
        # Revenue is maximize → negated: -8; Effort is minimize → kept: 5
        assert out["F"][0, 0] == -8.0
        assert out["F"][0, 1] == 5.0

    def test_cardinality_constraint_feasible(self):
        """Solution within cardinality bounds → G ≤ 0."""
        prob = self._make_pymoo_problem(cardinality_min=2, cardinality_max=3)
        X = np.array([[1, 1, 0, 0, 0]], dtype=float)  # 2 selected
        out = {}
        prob._evaluate(X, out)
        G = out["G"]
        assert G[0, 0] <= 0  # min - count: 2 - 2 = 0
        assert G[0, 1] <= 0  # count - max: 2 - 3 = -1

    def test_cardinality_constraint_infeasible_below_min(self):
        """Too few selected → G[0] > 0."""
        prob = self._make_pymoo_problem(cardinality_min=3, cardinality_max=5)
        X = np.array([[1, 0, 0, 0, 0]], dtype=float)  # 1 selected, min is 3
        out = {}
        prob._evaluate(X, out)
        assert out["G"][0, 0] > 0  # 3 - 1 = 2

    def test_force_include_feasible(self):
        """Forced-in option is selected → constraint satisfied."""
        prob = self._make_pymoo_problem(forced_in={0})
        X = np.array([[1, 1, 0, 0, 0]], dtype=float)
        out = {}
        prob._evaluate(X, out)
        # Force-include constraint: 1 - X[0] ≤ 0
        fi_idx = 2  # After 2 cardinality constraints
        assert out["G"][0, fi_idx] <= 0

    def test_force_include_infeasible(self):
        """Forced-in option not selected → constraint violated."""
        prob = self._make_pymoo_problem(forced_in={0})
        X = np.array([[0, 1, 0, 0, 0]], dtype=float)
        out = {}
        prob._evaluate(X, out)
        fi_idx = 2
        assert out["G"][0, fi_idx] > 0  # 1 - 0 = 1

    def test_force_exclude_feasible(self):
        """Forced-out option not selected → constraint satisfied."""
        prob = self._make_pymoo_problem(forced_out={4})
        X = np.array([[1, 1, 0, 0, 0]], dtype=float)
        out = {}
        prob._evaluate(X, out)
        fe_idx = 2  # After 2 cardinality constraints
        assert out["G"][0, fe_idx] <= 0

    def test_force_exclude_infeasible(self):
        """Forced-out option selected → constraint violated."""
        prob = self._make_pymoo_problem(forced_out={4})
        X = np.array([[1, 1, 0, 0, 1]], dtype=float)
        out = {}
        prob._evaluate(X, out)
        fe_idx = 2
        assert out["G"][0, fe_idx] > 0

    def test_objective_bound_max_feasible(self):
        """Objective value within max bound → G ≤ 0."""
        prob = self._make_pymoo_problem(
            obj_bounds=[(1, BoundOperator.max, 10.0)]  # Effort ≤ 10
        )
        X = np.array([[1, 1, 0, 0, 0]], dtype=float)  # Effort = 5+3 = 8
        out = {}
        prob._evaluate(X, out)
        ob_idx = 2  # After 2 cardinality constraints
        assert out["G"][0, ob_idx] <= 0  # 8 - 10 = -2

    def test_objective_bound_max_infeasible(self):
        """Objective value exceeds max bound → G > 0."""
        prob = self._make_pymoo_problem(
            obj_bounds=[(1, BoundOperator.max, 5.0)]  # Effort ≤ 5
        )
        X = np.array([[1, 0, 1, 0, 0]], dtype=float)  # Effort = 5+7 = 12
        out = {}
        prob._evaluate(X, out)
        ob_idx = 2
        assert out["G"][0, ob_idx] > 0  # 12 - 5 = 7

    def test_objective_bound_min_feasible(self):
        """Objective value above min bound → G ≤ 0."""
        prob = self._make_pymoo_problem(
            obj_bounds=[(0, BoundOperator.min, 10.0)]  # Revenue ≥ 10
        )
        X = np.array([[1, 1, 0, 0, 0]], dtype=float)  # Revenue = 8+6 = 14
        out = {}
        prob._evaluate(X, out)
        ob_idx = 2
        assert out["G"][0, ob_idx] <= 0  # 10 - 14 = -4

    def test_objective_bound_min_infeasible(self):
        """Objective value below min bound → G > 0."""
        prob = self._make_pymoo_problem(
            obj_bounds=[(0, BoundOperator.min, 20.0)]  # Revenue ≥ 20
        )
        X = np.array([[1, 1, 0, 0, 0]], dtype=float)  # Revenue = 14
        out = {}
        prob._evaluate(X, out)
        ob_idx = 2
        assert out["G"][0, ob_idx] > 0  # 20 - 14 = 6

    def test_batch_evaluation(self):
        """Multiple solutions evaluated at once."""
        prob = self._make_pymoo_problem()
        X = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ], dtype=float)
        out = {}
        prob._evaluate(X, out)
        assert out["F"].shape == (3, 2)
        assert out["G"].shape[0] == 3


# ─── _compute_quality ───


class FakeResult:
    def __init__(self, F):
        self.F = F


class TestComputeQuality:
    def test_none_result(self):
        qi = _compute_quality(FakeResult(F=None))
        assert qi.hypervolume_normalized is None
        assert qi.spacing_cv is None

    def test_single_solution(self):
        qi = _compute_quality(FakeResult(F=np.array([[1.0, 2.0]])))
        assert qi.hypervolume_normalized is None
        assert qi.spacing_cv is None

    def test_two_solutions_no_spacing(self):
        """2 solutions → hypervolume computed, spacing_cv is None (needs ≥3)."""
        qi = _compute_quality(FakeResult(F=np.array([[1.0, 5.0], [3.0, 2.0]])))
        assert qi.hypervolume_normalized is not None
        assert qi.spacing_cv is None

    def test_three_solutions_full_quality(self):
        qi = _compute_quality(FakeResult(F=np.array([
            [1.0, 5.0], [3.0, 2.0], [2.0, 3.5],
        ])))
        assert qi.hypervolume_normalized is not None
        assert qi.spacing_cv is not None

    def test_constant_dimension_zero_spread(self):
        """All solutions same value on one objective → hypervolume is None."""
        qi = _compute_quality(FakeResult(F=np.array([
            [5.0, 1.0], [5.0, 2.0], [5.0, 3.0],
        ])))
        assert qi.hypervolume_normalized is None
        assert qi.spacing_cv is not None


# ─── _approx_hypervolume ───


class TestApproxHypervolume:
    def test_deterministic_with_seed(self):
        F = np.array([[0.3, 0.5], [0.5, 0.3]])
        ref = np.array([1.1, 1.1])
        hv1 = _approx_hypervolume(F, ref)
        hv2 = _approx_hypervolume(F, ref)
        assert hv1 == hv2

    def test_single_point_positive_hv(self):
        F = np.array([[0.5, 0.5]])
        ref = np.array([1.1, 1.1])
        hv = _approx_hypervolume(F, ref)
        assert hv > 0

    def test_dominated_space_grows_with_more_points(self):
        """More non-dominated points should give higher hypervolume."""
        ref = np.array([1.1, 1.1])
        F1 = np.array([[0.5, 0.5]])
        F2 = np.array([[0.5, 0.5], [0.3, 0.7]])
        hv1 = _approx_hypervolume(F1, ref)
        hv2 = _approx_hypervolume(F2, ref)
        assert hv2 >= hv1


# ─── Optimization with minimize-first ───


class TestOptimizeMinimizeFirst:
    def test_solutions_sorted_ascending_for_minimize_first(self):
        """When first objective is minimize, solutions should be ascending."""
        p = _make_problem(
            objectives=[
                Objective(name="Effort", direction="minimize", unit="weeks"),
                Objective(name="Revenue", direction="maximize", unit="$"),
            ],
        )
        run = optimize(p)
        if len(run.solutions) >= 2:
            for i in range(len(run.solutions) - 1):
                assert (
                    run.solutions[i].objective_values["Effort"]
                    <= run.solutions[i + 1].objective_values["Effort"]
                )


# ─── analyze_infeasibility edge cases ───


class TestAnalyzeInfeasibilityEdge:
    def test_loose_constraints_no_binding(self):
        """Very loose constraints → no individual constraint is binding."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=1, max=5),
        ])
        result = analyze_infeasibility(p)
        # All combos feasible, no individual constraint is binding
        # Should fall through to "jointly infeasible" message
        assert "suggestions" in result

    def test_min_bound_on_objective(self):
        """Tight min bound identified as binding."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ObjectiveBoundConstraint(
                objective="Revenue", operator="min", value=100
            ),  # impossibly high
        ])
        result = analyze_infeasibility(p)
        assert any("Revenue" in s for s in result["suggestions"])

    def test_suggestion_for_cardinality(self):
        """Tight cardinality + excludes = jointly infeasible, should say so."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=5, max=5),
            ForceExcludeConstraint(option="A"),
            ForceExcludeConstraint(option="B"),
        ])
        result = analyze_infeasibility(p)
        # These are jointly infeasible — no single relaxation helps
        assert any("jointly" in s.lower() or "multiple" in s.lower()
                    for s in result["suggestions"])


# ─── Validation combinatorial edge cases ───


class TestValidationCombinatorial:
    def test_multiple_issues_at_once(self):
        """Problem with multiple simultaneous issues reports all of them."""
        p = Problem(
            objectives=[Objective(name="Rev", direction="maximize")],
            options=[Option(name="A")],
            scores=[],
            constraints=[
                CardinalityConstraint(min=10, max=3),
                ForceIncludeConstraint(option="Nonexistent"),
            ],
        )
        vr = validate(p)
        assert vr.ready is False
        assert len(vr.issues) >= 3  # too few objs, too few options, card min>max, unknown opt

    def test_warning_does_not_block_ready(self):
        """Warnings alone shouldn't prevent ready=True."""
        p = _make_problem()
        vr = validate(p)
        # All valid → no warnings or errors → ready
        assert vr.ready is True
        assert all(i.severity != "error" for i in vr.issues)


class TestProblemSizeBounds:
    """Upper bounds are a generous anti-abuse backstop, not a usability limit — they
    only reject a runaway/garbage payload, never a real (even very large) problem.
    Patch the module ceilings low so a normal _make_problem (5 options / 2 obj / 10
    scores) trips them; defaults (100k options) never flag a real problem."""

    def test_rejects_too_many_options(self, monkeypatch):
        import engine.optimizer as opt
        monkeypatch.setattr(opt, "_MAX_OPTIONS", 3)
        vr = validate(_make_problem())
        assert vr.ready is False
        assert any("Too many options" in i.message for i in vr.issues)

    def test_rejects_too_many_objectives(self, monkeypatch):
        import engine.optimizer as opt
        monkeypatch.setattr(opt, "_MAX_OBJECTIVES", 1)
        vr = validate(_make_problem())
        assert any("Too many objectives" in i.message for i in vr.issues)

    def test_rejects_too_many_scores(self, monkeypatch):
        import engine.optimizer as opt
        monkeypatch.setattr(opt, "_MAX_SCORES", 3)
        vr = validate(_make_problem())
        assert any("Too many scores" in i.message for i in vr.issues)

    def test_normal_problem_has_no_size_error(self):
        vr = validate(_make_problem())
        assert not any("Too many" in i.message for i in vr.issues)
