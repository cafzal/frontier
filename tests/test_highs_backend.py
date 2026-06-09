"""Tests for the HiGHS exact-solver backend (solvers/highs_backend.py).

Skipped wholesale when ``highspy`` is not installed, so the default-install test run is
unaffected. The backend is CPU/cross-platform, so unlike the cuOpt backend these run
anywhere CI does.
"""

import numpy as np
import pytest

pytest.importorskip("highspy")

from engine import explorer
from engine.models import (
    CardinalityConstraint,
    ForceIncludeConstraint,
    GroupLimitConstraint,
    InteractionMatrix,
    MaxAllocationConstraint,
    Objective,
    OptimizeMode,
    Option,
    Problem,
    Score,
)
from engine.optimizer import optimize
from solvers import exact_solver_fits
from solvers.highs_backend import _optimize_highs


# ─── Fixtures ───

def _binary_problem(**overrides):
    """Small 3-objective binary selection with a spread of constraint types."""
    names = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(0)
    scores = []
    table = {
        "NPV": [9, 7, 8, 4, 6, 5],
        "Cost": [5, 3, 6, 2, 4, 3],
        "Fit": [8, 6, 5, 9, 7, 4],
    }
    for opt_i, name in enumerate(names):
        for obj, col in table.items():
            scores.append(Score(option=name, objective=obj, value=col[opt_i]))
    defaults = dict(
        approach="binary",
        objectives=[
            Objective(name="NPV", direction="maximize"),
            Objective(name="Cost", direction="minimize"),
            Objective(name="Fit", direction="maximize"),
        ],
        options=[Option(name=n) for n in names],
        scores=scores,
        constraints=[CardinalityConstraint(min=2, max=4)],
    )
    defaults.update(overrides)
    return Problem(**defaults)


def _qp_problem(**overrides):
    """Small proportional mean-variance portfolio: linear return + quadratic risk."""
    names = ["A", "B", "C", "D"]
    returns = {"A": 10, "B": 12, "C": 8, "D": 15}
    cov = {
        "A": {"A": 0.04, "B": 0.01, "C": 0.00, "D": 0.00},
        "B": {"A": 0.01, "B": 0.09, "C": 0.01, "D": 0.00},
        "C": {"A": 0.00, "B": 0.01, "C": 0.16, "D": 0.02},
        "D": {"A": 0.00, "B": 0.00, "C": 0.02, "D": 0.25},
    }
    scores = []
    for n in names:
        scores.append(Score(option=n, objective="Return", value=returns[n]))
        scores.append(Score(option=n, objective="Risk", value=cov[n][n] * 100))  # placeholder; quad uses the matrix
    defaults = dict(
        approach="proportional",
        objectives=[
            Objective(name="Return", direction="maximize", aggregation="avg"),
            Objective(name="Risk", direction="minimize", aggregation="quadratic"),
        ],
        options=[Option(name=n) for n in names],
        scores=scores,
        interaction_matrices=[InteractionMatrix(objective="Risk", entries=cov)],
        constraints=[MaxAllocationConstraint(max=50)],
    )
    defaults.update(overrides)
    return Problem(**defaults)


def _nondominated_ok(solutions, objs):
    """No returned solution is strictly dominated by another (a valid Pareto set)."""
    F = np.array([[(-s.objective_values[o.name] if o.direction.value == "maximize"
                    else s.objective_values[o.name]) for o in objs] for s in solutions])
    for i, a in enumerate(F):
        for j, b in enumerate(F):
            if i != j and np.all(b <= a) and np.any(b < a):
                return False
    return True


# ─── Shape gate (shared exact_solver_fits) + routing ───

class TestGate:
    def test_binary_fits(self):
        assert exact_solver_fits(_binary_problem())[0] is True

    def test_quadratic_portfolio_fits(self):
        assert exact_solver_fits(_qp_problem())[0] is True

    def test_cardinality_qp_fits(self):
        # which-K-of-N + quadratic is mixed-integer-quadratic; the EA support-search picks the
        # K assets and HiGHS solves the continuous QP on them, so this IS in scope.
        p = _qp_problem(constraints=[CardinalityConstraint(min=1, max=2)])
        assert exact_solver_fits(p)[0] is True

    def test_group_limited_qp_fits(self):
        p = _qp_problem(constraints=[GroupLimitConstraint(options=["A", "B"], max=1)])
        assert exact_solver_fits(p)[0] is True

    def test_proportional_linear_fits_as_lp(self):
        # Two purely linear objectives (no quadratic) → the exact multi-objective LP path now owns
        # this shape (it previously declined to NSGA).
        p = _qp_problem(objectives=[Objective(name="Return", direction="maximize", aggregation="avg"),
                                    Objective(name="Risk", direction="minimize", aggregation="sum")],
                        interaction_matrices=[])
        assert exact_solver_fits(p)[0] is True

    def test_proportional_minmax_does_not_fit(self):
        # min/max aggregation is nonlinear → outside both the QP and LP exact shapes; NSGA owns it.
        p = _qp_problem(objectives=[Objective(name="Return", direction="maximize", aggregation="avg"),
                                    Objective(name="Risk", direction="minimize", aggregation="min")],
                        interaction_matrices=[])
        assert exact_solver_fits(p)[0] is False

    def test_ill_fitting_request_raises(self):
        # Requesting highs on a shape it can't solve (min/max aggregation is nonlinear) raises
        # rather than silently degrading.
        p = _qp_problem(objectives=[Objective(name="Return", direction="maximize", aggregation="avg"),
                                    Objective(name="Risk", direction="minimize", aggregation="min")],
                        interaction_matrices=[])
        with pytest.raises(ValueError, match="does not fit"):
            optimize(p, mode=OptimizeMode.fast, solver="highs")

    def test_binary_nonsum_aggregation_does_not_fit(self):
        # The binary MILP is linear (Σ coef·x) → only additive (sum) objectives. avg is fractional
        # over a variable-size selection, min/max are nonlinear; declining keeps the certificate
        # from comparing the NSGA-evaluated aggregation against an exact run that silently summed.
        for agg in ("avg", "min", "max"):
            p = _binary_problem(objectives=[
                Objective(name="NPV", direction="maximize"),
                Objective(name="Cost", direction="minimize"),
                Objective(name="Fit", direction="maximize", aggregation=agg)])
            fits, reason = exact_solver_fits(p)
            assert fits is False, agg
            assert "sum" in reason and "Fit" in reason
        # optimize() refuses rather than silently optimizing the sum (the bug this guards).
        p = _binary_problem(objectives=[
            Objective(name="NPV", direction="maximize"),
            Objective(name="Cost", direction="minimize"),
            Objective(name="Fit", direction="maximize", aggregation="avg")])
        with pytest.raises(ValueError, match="does not fit"):
            optimize(p, mode=OptimizeMode.fast, solver="highs")

    def test_proportional_minmax_aggregation_does_not_fit(self):
        # The mean-variance QP handles linear (sum/avg) + the quadratic risk term; min/max are
        # nonlinear and out of scope.
        for agg in ("min", "max"):
            p = _qp_problem(objectives=[
                Objective(name="Return", direction="maximize", aggregation=agg),
                Objective(name="Risk", direction="minimize", aggregation="quadratic")])
            fits, reason = exact_solver_fits(p)
            assert fits is False, agg
            assert agg in reason and "Return" in reason


# ─── Binary MILP path ───

class TestBinaryMILP:
    def test_produces_valid_frontier(self):
        p = _binary_problem()
        run = _optimize_highs(p, mode=OptimizeMode.fast)
        assert len(run.solutions) > 0
        assert _nondominated_ok(run.solutions, p.objectives)
        # Provenance is stamped by the backend, so a direct call is labelled correctly.
        assert run.solver == "highs"

    def test_respects_cardinality_and_force_include(self):
        p = _binary_problem(constraints=[CardinalityConstraint(min=2, max=3),
                                         ForceIncludeConstraint(option="A")])
        run = _optimize_highs(p, mode=OptimizeMode.fast)
        assert run.solutions
        for s in run.solutions:
            assert 2 <= len(s.selected_options) <= 3
            assert "A" in s.selected_options

    def test_deterministic(self):
        p = _binary_problem()
        a = _optimize_highs(p, mode=OptimizeMode.fast)
        b = _optimize_highs(p, mode=OptimizeMode.fast)
        assert [s.objective_values for s in a.solutions] == [s.objective_values for s in b.solutions]

    def test_routes_through_optimize(self):
        run = optimize(_binary_problem(), mode=OptimizeMode.fast, solver="highs")
        assert len(run.solutions) > 0
        assert run.solver == "highs"


# ─── Convex QP path ───

class TestProportionalQP:
    def test_produces_valid_frontier(self):
        p = _qp_problem()
        run = _optimize_highs(p, mode=OptimizeMode.fast)
        assert len(run.solutions) > 0
        assert _nondominated_ok(run.solutions, p.objectives)
        assert run.solver == "highs"  # backend stamps its own provenance

    def test_allocations_budget_and_box(self):
        p = _qp_problem(constraints=[MaxAllocationConstraint(max=50)])
        run = _optimize_highs(p, mode=OptimizeMode.fast)
        assert run.solutions
        for s in run.solutions:
            assert sum(s.allocations.values()) == 100
            assert max(s.allocations.values()) <= 50

    def test_cardinality_via_ea_support_search(self):
        # The EA picks which ≤K assets are eligible; HiGHS solves the continuous QP on that
        # support. This is the mixed-integer-quadratic case HiGHS can't solve in one shot.
        p = _qp_problem(constraints=[CardinalityConstraint(min=1, max=2)])
        run = _optimize_highs(p, mode=OptimizeMode.fast)
        assert run.solutions
        for s in run.solutions:
            assert 1 <= len(s.selected_options) <= 2

    def test_deterministic(self):
        p = _qp_problem()
        a = _optimize_highs(p, mode=OptimizeMode.fast)
        b = _optimize_highs(p, mode=OptimizeMode.fast)
        assert [s.objective_values for s in a.solutions] == [s.objective_values for s in b.solutions]


# ─── Sensitivity (exact-solver duals) ───

class TestSensitivity:
    def test_qp_solutions_carry_solver_exact_duals(self):
        p = _qp_problem()
        run = _optimize_highs(p, mode=OptimizeMode.fast)
        assert run.solutions
        for s in run.solutions:
            assert s.sensitivity is not None, "QP solution missing sensitivity"
            assert s.sensitivity.source == "solver_exact"
            assert len(s.sensitivity.reduced_costs) == len(p.options)  # one per option
            roles = {sp.role for sp in s.sensitivity.shadow_prices}
            assert "budget" in roles and "return_floor" in roles

    def test_return_shadow_price_sign_and_variation(self):
        # The return-floor shadow price = marginal risk per unit of required return: ≥ 0
        # everywhere (more return costs more variance), and varies along the frontier (convex →
        # not all equal). This is the exactness the frontier regression only approximates.
        run = _optimize_highs(_qp_problem(), mode=OptimizeMode.fast)
        sp = [next(x.shadow_price for x in s.sensitivity.shadow_prices if x.role == "return_floor")
              for s in run.solutions]
        assert min(sp) >= -1e-6                  # economically non-negative
        assert max(sp) - min(sp) > 1e-4          # genuinely varies (diminishing returns)

    def test_near_miss_reduced_cost_for_excluded_option(self):
        # An option the optimizer leaves at 0 carries a non-zero reduced cost (a near-miss);
        # held interior options are ~0. So at least one solution shows a near-miss.
        run = _optimize_highs(_qp_problem(), mode=OptimizeMode.fast)
        found = any(rc.allocation == 0 and abs(rc.reduced_cost) > 1e-6
                    for s in run.solutions for rc in s.sensitivity.reduced_costs)
        assert found, "expected at least one excluded-option near-miss on the frontier"

    def test_milp_solutions_have_no_exact_duals(self):
        # Duals/reduced costs are LP/QP-only — integer solutions must not carry them.
        run = _optimize_highs(_binary_problem(), mode=OptimizeMode.fast)
        assert run.solutions
        assert all(s.sensitivity is None for s in run.solutions)

    def test_explore_sensitivity_exact_path(self):
        p = _qp_problem()
        p.exact_run = _optimize_highs(p, mode=OptimizeMode.fast)
        res = explorer.sensitivity_analysis(p)
        assert res["source"] == "solver_exact"
        assert res["where_to_invest"]                       # budget + return levers
        assert "near_misses" in res and "frontier_shadow_price_trend" in res
        assert all("shadow_price" in t for t in res["frontier_shadow_price_trend"])

    def test_explore_sensitivity_falls_back_for_milp(self):
        p = _binary_problem()
        p.exact_run = _optimize_highs(p, mode=OptimizeMode.fast)
        res = explorer.sensitivity_analysis(p)
        assert res["source"] == "frontier_inferred"
        assert "binding_analysis" in res

    def test_cardinality_qp_keeps_offsupport_out_of_near_misses(self):
        # With a cardinality cap the EA pins off-support assets to ub=0 — their reduced cost is
        # about the cap, not a near-miss. They must be flagged ineligible and never surface in
        # the explore near_misses list.
        p = _qp_problem(constraints=[CardinalityConstraint(min=1, max=2)])
        run = _optimize_highs(p, mode=OptimizeMode.fast)
        assert any(not rc.eligible for s in run.solutions for rc in s.sensitivity.reduced_costs), \
            "expected some off-support (ineligible) assets under a cardinality cap"
        p.exact_run = run
        res = explorer.sensitivity_analysis(p)
        if res["source"] == "solver_exact":
            ref = next(s for s in run.solutions if s.solution_id == res["reference_solution"]["solution_id"])
            ineligible = {rc.option for rc in ref.sensitivity.reduced_costs if not rc.eligible}
            near_options = {nm["option"] for nm in res["near_misses"]}
            assert near_options.isdisjoint(ineligible)

    def test_lp_shape_has_exact_path_with_duals(self):
        # Per-type matrix (updated): solver-exact duals now cover the LP path too. A purely linear
        # proportional problem (>=2 objectives, no quadratic risk term) is an exact multi-objective
        # LP — explore sensitivity returns solver_exact shadow prices + reduced costs, not the
        # frontier-inferred fallback. (Closes the gap the old version of this test documented.)
        names = ["A", "B", "C", "D"]
        ret, yld = {"A": 10, "B": 12, "C": 8, "D": 15}, {"A": 3, "B": 2, "C": 5, "D": 1}
        sc = []
        for n in names:
            sc += [Score(option=n, objective="Return", value=ret[n]),
                   Score(option=n, objective="Yield", value=yld[n])]
        p = Problem(approach="proportional",
                    objectives=[Objective(name="Return", direction="maximize", aggregation="avg"),
                                Objective(name="Yield", direction="maximize", aggregation="avg")],
                    options=[Option(name=n) for n in names], scores=sc,
                    constraints=[MaxAllocationConstraint(max=50)])
        assert exact_solver_fits(p)[0] is True          # linear-continuous now has an exact LP path
        run = _optimize_highs(p, mode=OptimizeMode.fast)
        assert run.solver == "highs"
        assert len(run.solutions) >= 2 and _nondominated_ok(run.solutions, p.objectives)
        for s in run.solutions:
            assert abs(sum(s.allocations.values()) - 100) <= 1   # fully invested
            assert all(v <= 50 for v in s.allocations.values())  # max-allocation cap

        withsens = [s for s in run.solutions if s.sensitivity]
        assert withsens, "LP frontier points should carry solver-exact sensitivity"
        sens = withsens[0].sensitivity
        assert sens.source == "solver_exact"
        roles = {sp.role for sp in sens.shadow_prices}
        assert "budget" in roles                                  # the Σw=1 dual is always present
        assert "linear_floor" in roles                            # the epsilon-constrained objective
        assert len(sens.reduced_costs) == len(names)              # one reduced cost per option

        p.exact_run = run
        res = explorer.sensitivity_analysis(p)
        assert res["source"] == "solver_exact"
        assert "where_to_invest" in res and "near_misses" in res
        # Regression: the diminishing-returns trend must populate on the LP path. _shadow_price_trend
        # matched only the QP "return_floor" role, leaving the LP "linear_floor" trend empty.
        trend = res["frontier_shadow_price_trend"]
        assert trend, "LP path must populate frontier_shadow_price_trend (the swept linear_floor objective)"
        assert all("shadow_price" in t and t["lever"] == "Yield" for t in trend)
