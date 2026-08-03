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
    AllocationBoundConstraint,
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

    def test_exact_proportional_honors_membership_constraints(self):
        # force_exclude/include fold into the variable box (0 cap / 1% activity floor); the
        # genuinely combinatorial trio declines rather than certifying the wrong region.
        from engine.models import (CardinalityConstraint, DependencyConstraint,
                                   ExclusionPairConstraint, ForceExcludeConstraint,
                                   ForceIncludeConstraint)

        base = dict(objectives=[Objective(name="Return", direction="maximize", aggregation="avg"),
                                Objective(name="Risk", direction="minimize", aggregation="sum")],
                    interaction_matrices=[])
        p = _qp_problem(constraints=[MaxAllocationConstraint(max=60),
                                     ForceExcludeConstraint(option="D"),
                                     ForceIncludeConstraint(option="C")], **base)
        run = optimize(p, solver="highs", seed=3)
        assert len(run.solutions) > 0
        for s in run.solutions:
            assert s.allocations.get("D", 0) == 0
            assert s.allocations.get("C", 0) >= 1
        for bad in (ExclusionPairConstraint(option_a="A", option_b="B"),
                    DependencyConstraint(if_option="A", then_option="B"),
                    CardinalityConstraint(min=2, max=3)):
            fits, why = exact_solver_fits(_qp_problem(constraints=[bad], **base))
            assert fits is False and "combinatorial" in why, (bad.type, why)
        # cardinality min=1 is the engine default — always satisfiable, stays in scope.
        assert exact_solver_fits(_qp_problem(constraints=[CardinalityConstraint(min=1, max=3)],
                                             **base))[0] is True

    def test_quadratic_bound_cap_filtered_floor_declined(self):
        # A MAX cap on the quadratic minimand rides the exact path via post-filtering (the
        # inner solve minimizes it, so violation proves the targets infeasible); a MIN floor
        # on it is non-convex and declines.
        from engine.models import ObjectiveBoundConstraint

        capped = _qp_problem(constraints=[
            MaxAllocationConstraint(max=50),
            ObjectiveBoundConstraint(objective="Risk", operator="max", value=0.30)])
        assert exact_solver_fits(capped)[0] is True
        run = optimize(capped, solver="highs", seed=5)
        assert len(run.solutions) > 0
        for s in run.solutions:   # every certified point honors the model's own risk cap
            assert s.objective_values["Risk"] <= 0.30 + 1e-6
        floored = _qp_problem(constraints=[
            ObjectiveBoundConstraint(objective="Risk", operator="min", value=0.05)])
        fits, why = exact_solver_fits(floored)
        assert fits is False and "non-convex" in why

    def test_group_floor_declined_on_proportional_but_fits_on_binary(self):
        # A floor counts *active* options — combinatorial on the continuous QP/LP path,
        # a plain MILP row on the binary path.
        p = _qp_problem(constraints=[GroupLimitConstraint(options=["A", "B"], min=1, max=2)])
        fits, why = exact_solver_fits(p)
        assert fits is False and "floor" in why.lower() or "minimum" in why.lower()
        b = _binary_problem(constraints=[GroupLimitConstraint(options=["A", "B"], min=1, max=2)])
        assert exact_solver_fits(b)[0] is True

    def test_qp_floor_off_support_reports_infeasible_instead_of_raising(self):
        """A floored asset dropped from ``support`` makes the weight box empty (lb > ub = 0).
        highspy raises on such a box rather than reporting infeasible, so the entry-point
        guard must short-circuit to (zeros, False) — the EA then discards that support
        choice. Pre-guard behavior was an exception mid-solve (QP + allocation floors)."""
        from solvers.highs_backend import _solve_qp_highs, _solve_qp_highs_sensitivity

        cov = np.eye(3)
        mu = np.array([1.0, 2.0, 3.0])
        min_w = np.array([0.0, 0.10, 0.0])            # floor on asset 1
        w, ok = _solve_qp_highs(cov, mu, None, True, 0.5, support=[0, 2], min_weight=min_w)
        assert ok is False and np.allclose(w, 0)
        w, ok, raw = _solve_qp_highs_sensitivity(cov, mu, None, True, 0.5,
                                                 support=[0, 2], min_weight=min_w)
        assert ok is False and raw is None

    def test_exact_lp_respects_allocation_bounds_and_prices_the_floor(self):
        """E2 on the exact LP path: per-option floors/caps become variable bounds, every
        overlay point honors them, and the dual read still returns per-option reduced costs
        (the floor's price surfaces there)."""
        from engine.models import AllocationBoundConstraint

        names = ["A", "B", "C", "D"]
        rev = {"A": 14, "B": 11, "C": 8, "D": 5}
        stab = {"A": 3, "B": 6, "C": 8, "D": 9}
        scores = []
        for n_ in names:
            scores.append(Score(option=n_, objective="Revenue", value=rev[n_]))
            scores.append(Score(option=n_, objective="Stability", value=stab[n_]))
        p = Problem(
            name="lp-bounds", approach="proportional",
            objectives=[Objective(name="Revenue", direction="maximize", aggregation="sum"),
                        Objective(name="Stability", direction="maximize", aggregation="sum")],
            options=[Option(name=n_) for n_ in names], scores=scores,
            constraints=[MaxAllocationConstraint(max=60),
                         AllocationBoundConstraint(option="D", min=15, max=100),
                         AllocationBoundConstraint(option="A", min=0, max=35)],
        )
        assert exact_solver_fits(p)[0] is True
        run = optimize(p, solver="highs", seed=11)
        assert len(run.solutions) > 0
        priced = False
        for s in run.solutions:
            assert s.allocations.get("D", 0) >= 15    # the floor holds on every exact point
            assert s.allocations.get("A", 0) <= 35    # the per-option cap holds
            if s.sensitivity is not None and s.sensitivity.reduced_costs:
                priced = True
        assert priced, "exact LP overlay carries no dual read"
        # D/E/F score low on NPV; a floor on {D, E, F} must still pull one into every plan.
        p = _binary_problem(constraints=[
            CardinalityConstraint(min=2, max=4),
            GroupLimitConstraint(options=["D", "E", "F"], min=1, max=2),
        ])
        run = optimize(p, solver="highs", seed=7)
        assert len(run.solutions) > 0
        for s in run.solutions:
            in_group = len({"D", "E", "F"} & set(s.selected_options))
            assert 1 <= in_group <= 2

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

    def test_duplicate_cardinality_rows_bind_the_exact_frontier(self):
        """End to end: the exact frontier obeys the SAME merged range the NSGA pass
        searched. Encoding one row here (the last, max 6) returns plans the heuristic
        frontier could never contain — and `explore certify` compares those two frontiers,
        so its optimality gap and never-dominates invariant would span two feasible sets."""
        from engine.optimizer import merged_cardinality

        p = _binary_problem(constraints=[CardinalityConstraint(min=1, max=2),
                                         CardinalityConstraint(min=1, max=6)])
        assert merged_cardinality(p.constraints) == (1, 2)
        exact = _optimize_highs(p, mode=OptimizeMode.fast)
        assert exact.solutions
        assert all(1 <= len(s.selected_options) <= 2 for s in exact.solutions)
        # The heuristic half agrees, which is what makes the two frontiers comparable.
        nsga = optimize(p, mode=OptimizeMode.fast, seed=3)
        assert all(1 <= len(s.selected_options) <= 2 for s in nsga.solutions)

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
        # provenance + the named ε-constraint primary (QP primary = the quadratic objective)
        assert res["frontier_source"]["kind"] == "exact"
        quad = next(o.name for o in p.objectives
                    if getattr(o.aggregation, "value", o.aggregation) == "quadratic")
        assert res["optimized_objective"] == quad
        assert f"'{quad}'" in res["where_to_invest"][0]["interpretation"]

    def test_explore_sensitivity_falls_back_for_milp(self):
        p = _binary_problem()
        p.exact_run = _optimize_highs(p, mode=OptimizeMode.fast)
        res = explorer.sensitivity_analysis(p)
        assert res["source"] == "frontier_inferred"
        assert "binding_analysis" in res
        assert "frontier_source" in res    # provenance echoed on the fallback too

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


# ─── Audit-pass regressions: support caps, direction-aware sweeps, dual wording ───

class TestSupportRespectsCountCaps:
    def test_group_caps_plus_global_cardinality_never_violated(self):
        """Group caps whose sum exceeds the global cardinality cap: every exact point
        must respect BOTH (the old support decode unioned per-group tops and could
        activate more options than the global cap allows)."""
        names = [f"o{i}" for i in range(6)]
        scores = []
        for i, n in enumerate(names):
            scores.append(Score(option=n, objective="Ret", value=float(3 + (i * 2) % 5)))
            scores.append(Score(option=n, objective="Risk", value=30.0))
        cov = {a: {b: (0.3 if a == b else 0.02) for b in names} for a in names}
        p = Problem(
            name="caps", approach="proportional",
            objectives=[Objective(name="Ret", direction="maximize", aggregation="sum"),
                        Objective(name="Risk", direction="minimize", aggregation="quadratic")],
            options=[Option(name=n) for n in names], scores=scores,
            interaction_matrices=[InteractionMatrix(objective="Risk", entries=cov)],
            constraints=[GroupLimitConstraint(options=["o0", "o1", "o2"], max=2),
                         GroupLimitConstraint(options=["o3", "o4", "o5"], max=2),
                         CardinalityConstraint(min=1, max=3)],
        )
        run = optimize(p, mode="fast", seed=1, solver="highs")
        assert run.solutions
        for s in run.solutions:
            active = [o for o, v in (s.allocations or {}).items() if v > 0]
            assert len(active) <= 3, active
            assert sum(1 for o in active if o in ("o0", "o1", "o2")) <= 2, active
            assert sum(1 for o in active if o in ("o3", "o4", "o5")) <= 2, active

    def test_qp_sweep_covers_a_minimize_linear_objective(self):
        """Direction-aware epsilon bounds: a mean-variance model whose linear objective
        MINIMIZES (cost) must still sweep the full tradeoff — the old max(coef) upper
        bound left the ceiling forever slack and the frontier collapsed to ~2 points."""
        names = [f"o{i}" for i in range(6)]
        cost = [2.0, 7.9, 3.5, 5.0, 6.2, 4.1]
        scores = []
        for i, n in enumerate(names):
            scores.append(Score(option=n, objective="Cost", value=cost[i]))
            scores.append(Score(option=n, objective="Risk", value=30.0))
        cov = {a: {b: (0.3 if a == b else 0.02) for b in names} for a in names}
        p = Problem(
            name="qpmin", approach="proportional",
            objectives=[Objective(name="Risk", direction="minimize", aggregation="quadratic"),
                        Objective(name="Cost", direction="minimize", aggregation="sum")],
            options=[Option(name=n) for n in names], scores=scores,
            interaction_matrices=[InteractionMatrix(objective="Risk", entries=cov)],
        )
        run = optimize(p, mode="fast", seed=1, solver="highs")
        costs = sorted(s.objective_values["Cost"] for s in run.solutions)
        assert len(run.solutions) >= 10
        # The sweep reaches the cheap corner (all-in on the 2.0 option) and trades away from it.
        assert costs[0] <= 2.2
        assert costs[-1] - costs[0] >= 1.0

    def test_qp_sweep_spans_past_the_min_variance_point_on_multi_linear(self):
        """Full-range epsilon box on a multi-linear QP: the frontier holds points where one
        linear objective is WORSE than its value at the min-variance portfolio, traded for
        another. An at-mv loose end (the old bound) made that whole region unreachable —
        the exact overlay covered only the mv-side slice (supplier_selection: exact Cost
        spanned 10.42–10.84 while NSGA spanned 8.64–11.75)."""
        names = [f"o{i}" for i in range(6)]
        # Cost and Quality are in tension by construction: the dear options are the good ones.
        cost = [2.0, 3.0, 4.0, 6.0, 8.0, 9.0]
        quality = [1.0, 2.0, 3.0, 6.0, 8.5, 9.5]
        scores = []
        for i, n in enumerate(names):
            scores.append(Score(option=n, objective="Cost", value=cost[i]))
            scores.append(Score(option=n, objective="Quality", value=quality[i]))
            scores.append(Score(option=n, objective="Risk", value=30.0))
        # Identical variances: the min-variance portfolio spreads evenly, so its Cost sits
        # mid-range and the high-Quality corner costs strictly more than at-mv.
        cov = {a: {b: (0.3 if a == b else 0.02) for b in names} for a in names}
        p = Problem(
            name="qpspan", approach="proportional",
            objectives=[Objective(name="Risk", direction="minimize", aggregation="quadratic"),
                        Objective(name="Cost", direction="minimize", aggregation="sum"),
                        Objective(name="Quality", direction="maximize", aggregation="sum")],
            options=[Option(name=n) for n in names], scores=scores,
            interaction_matrices=[InteractionMatrix(objective="Risk", entries=cov)],
            constraints=[MaxAllocationConstraint(max=30)],
        )
        run = optimize(p, mode="fast", seed=1, solver="highs")
        assert len(run.solutions) >= 10
        # The even mv spread costs ~5.33; the best-Quality fill under the 30% cap costs
        # 0.3*(9+8+6)+0.1*4 = 7.3. The overlay must reach Quality corners costlier than
        # at-mv — the region the mv-anchored loose end cut off.
        costs = [s.objective_values["Cost"] for s in run.solutions]
        qualities = [s.objective_values["Quality"] for s in run.solutions]
        at_mv_cost = sum(cost) / len(cost)  # even spread ≈ the identical-variance mv mix
        assert max(costs) > at_mv_cost + 0.5, (max(costs), at_mv_cost)
        # And the Quality corner itself is sampled near its box-feasible extreme
        # (0.3*9.5 + 0.3*8.5 + 0.3*6 + 0.1*3 = 7.5).
        assert max(qualities) >= 7.0, max(qualities)


class TestShapeGateQuadCount:
    def test_two_quadratics_declined_in_words(self):
        names = ["A", "B", "C"]
        cov = {a: {b: (0.2 if a == b else 0.01) for b in names} for a in names}
        scores = []
        for n in names:
            scores.append(Score(option=n, objective="RiskA", value=20.0))
            scores.append(Score(option=n, objective="RiskB", value=20.0))
        p = Problem(
            name="twoquad", approach="proportional",
            objectives=[Objective(name="RiskA", direction="minimize", aggregation="quadratic"),
                        Objective(name="RiskB", direction="minimize", aggregation="quadratic")],
            options=[Option(name=n) for n in names], scores=scores,
            interaction_matrices=[InteractionMatrix(objective="RiskA", entries=cov),
                                  InteractionMatrix(objective="RiskB", entries=cov)],
        )
        fits, msg = exact_solver_fits(p)
        assert fits is False
        assert "single variance term" in msg


class TestLpDualDirection:
    def test_floor_price_on_a_maximize_primary_is_a_cost(self):
        """Ground truth for the dual convention: on a Revenue-max LP with a binding
        allocation floor, the reported floor price must equal how much Revenue is LOST
        per unit the floor rises (cost sense), and the interpretation must say 'worsens'
        — the old wording claimed the primary shifts UP by the price."""
        names = ["x", "y", "z"]
        rev = {"x": 3.0, "y": 1.0, "z": 2.0}
        scores = []
        for n in names:
            scores.append(Score(option=n, objective="Rev", value=rev[n]))
            scores.append(Score(option=n, objective="Frag", value={"x": 2.0, "y": 1.0, "z": 3.0}[n]))
        def _mk(floor):
            return Problem(
                name="lpdir", approach="proportional",
                objectives=[Objective(name="Rev", direction="maximize", aggregation="sum"),
                            Objective(name="Frag", direction="minimize", aggregation="sum")],
                options=[Option(name=n) for n in names], scores=scores,
                constraints=[AllocationBoundConstraint(option="y", min=floor, max=100)],
            )
        p = _mk(20)
        p.run = optimize(p, mode="fast", seed=1)
        p.exact_run = optimize(p, mode="fast", seed=1, solver="highs")
        # Revenue-max corner: x carries everything above y's floor. Raising the floor by
        # one point moves 1% from x (3.0/pt) to y (1.0/pt): Revenue drops by 2.0/pt... on
        # the per-1% scale used by sum aggregation over integer-percent allocations.
        best = max(p.exact_run.solutions, key=lambda s: s.objective_values["Rev"])
        sens = best.sensitivity
        assert sens is not None
        rc = {r.option: r for r in sens.reduced_costs}
        y = rc["y"]
        assert y.allocation <= 21
        # Cost sense: the floor's price is positive (it costs Revenue), magnitude = the
        # marginal Rev given up per allocation point moved from the best option to y.
        assert y.reduced_cost > 0
        # Ground-truth re-solve: raise the floor one percentage point (0.01 weight
        # units — duals are per unit weight); the Rev-max corner drops by about the
        # reported price scaled to that move.
        p2 = _mk(21)
        p2.exact_run = optimize(p2, mode="fast", seed=1, solver="highs")
        best2 = max(p2.exact_run.solutions, key=lambda s: s.objective_values["Rev"])
        drop = best.objective_values["Rev"] - best2.objective_values["Rev"]
        expected = y.reduced_cost / 100.0
        assert abs(drop - expected) <= 0.5 * expected

    def test_shadow_interpretation_uses_cost_framing(self):
        from engine.explorer import _shadow_interpretation
        from engine.models import ShadowPrice

        sp = ShadowPrice(name="StrategicValue", role="linear_floor", shadow_price=0.21)
        text = _shadow_interpretation(sp, "Revenue")
        assert "worsens 'Revenue' by ~0.21" in text
        assert "shifts" not in text

    def test_floored_option_always_in_support_under_caps(self):
        """An allocation floor force-activates its option, so every decoded support —
        the EA's and the deterministic anchors' — must admit it first: a support without
        it is infeasible at the solver, and before this guard the per-objective anchor
        corners could all come back infeasible and silently vanish."""
        names = [f"o{i}" for i in range(6)]
        # o5 scores WORST on both objectives so priority ranking would drop it last.
        rev = [9.0, 8.0, 7.0, 6.0, 5.0, 1.0]
        frag = [2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
        scores = []
        for i, n in enumerate(names):
            scores.append(Score(option=n, objective="Rev", value=rev[i]))
            scores.append(Score(option=n, objective="Frag", value=frag[i]))
        p = Problem(
            name="floorcap", approach="proportional",
            objectives=[Objective(name="Rev", direction="maximize", aggregation="sum"),
                        Objective(name="Frag", direction="minimize", aggregation="sum")],
            options=[Option(name=n) for n in names], scores=scores,
            constraints=[GroupLimitConstraint(options=["o0", "o1", "o2"], max=2),
                         CardinalityConstraint(min=1, max=3),
                         AllocationBoundConstraint(option="o5", min=10, max=100)],
        )
        run = optimize(p, mode="fast", seed=1, solver="highs")
        assert run.solutions
        for s in run.solutions:
            assert (s.allocations or {}).get("o5", 0) >= 10, s.allocations


class TestEpsPrescreen:
    """The epsilon pre-screen (`_feasible_eps`): provably unattainable targets are rejected
    with no solver call, and targets in the razor band just inside a support's attainable
    extreme are BACKED OFF to the band edge — the band is the geometry where the Linux
    HiGHS active-set QP cycles without terminating (the CI hang this guards against), and
    backing off (loosening) can only enlarge the feasible region, so joint feasibility
    with the other targets is preserved."""

    def test_unattainable_target_rejected_without_solver(self):
        from solvers._scalarization import _feasible_eps
        coef = np.array([3.0, 5.0, 7.0, 4.0, 6.0, 3.0])
        # Floor of 7 is unattainable on a support whose best coefficient is 6.
        assert _feasible_eps([7.0], [coef], [True], None, None, [1, 4, 5]) is None
        # Ceiling of 3 is unattainable when the cheapest in-support coefficient is 4.
        assert _feasible_eps([3.0], [coef], [False], None, None, [1, 3, 4]) is None
        # A support whose caps can't cover the budget is infeasible outright.
        assert _feasible_eps([1.0], [coef], [True], 0.15, None, [0, 1]) is None

    def test_razor_band_target_backs_off_to_the_band_edge(self):
        from solvers._scalarization import _feasible_eps
        coef = np.array([3.0, 5.0, 7.0, 4.0, 6.0, 3.0])
        # The observed Linux cycle: floor 6.9992 on a support with tied coefficients
        # ([3, 7, 3]) — inside the band, so it backs off to 7.0 − 1e-3·7 = 6.993.
        out = _feasible_eps([6.9992], [coef], [True], None, None, [0, 2, 5])
        assert out == [pytest.approx(6.993)]
        # An interior target passes through unchanged; a target exactly AT the extreme
        # stays — the corner remains fully sampled (anchors pin there).
        assert _feasible_eps([5.5], [coef], [True], None, None, [0, 2, 5]) == [pytest.approx(5.5)]
        assert _feasible_eps([7.0], [coef], [True], None, None, [0, 2, 5]) == [pytest.approx(7.0)]
        # Minimize direction backs off away from the attainable min the same way.
        out = _feasible_eps([3.0004], [coef], [False], None, None, [0, 2, 5])
        assert out == [pytest.approx(3.003)]

    def test_backoff_preserves_joint_feasibility(self):
        """Snapping UP would break this: A=0.9995 (in band of extreme 1.0) snapped to 1.0
        plus B's floor 0.0004 makes Σw=1 infeasible; backing A off keeps both floors
        jointly satisfiable, so the previously-feasible scalarization survives."""
        from solvers._scalarization import _feasible_eps
        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        out = _feasible_eps([0.9995, 0.0004], [a, b], [True, True], None, None, None)
        assert out is not None
        eps_a, eps_b = out
        assert eps_a == pytest.approx(0.999)     # backed off, not snapped to 1.0
        assert eps_b == pytest.approx(0.0004)
        assert eps_a + eps_b <= 1.0              # jointly satisfiable on the simplex

    def test_multi_minimize_sweep_stays_jointly_feasible(self):
        """Several epsilon-constrained linear objectives at once (the supplier_selection
        shape): the epsilon box's loose corner must be feasible for every portfolio, so the
        sweep produces a healthy frontier — not a boundary sliver that only survives on
        favorable floating-point behavior."""
        names = [f"s{i}" for i in range(8)]
        cost = [2.0, 7.0, 3.5, 5.0, 6.0, 4.0, 5.5, 3.0]
        lead = [9.0, 2.0, 7.0, 4.0, 3.0, 6.0, 5.0, 8.0]
        scores = []
        for i, n in enumerate(names):
            scores.append(Score(option=n, objective="Cost", value=cost[i]))
            scores.append(Score(option=n, objective="Lead", value=lead[i]))
            scores.append(Score(option=n, objective="Risk", value=30.0))
        cov = {a: {b: (0.3 if a == b else 0.02) for b in names} for a in names}
        p = Problem(
            name="multimin", approach="proportional",
            objectives=[Objective(name="Cost", direction="minimize", aggregation="sum"),
                        Objective(name="Lead", direction="minimize", aggregation="sum"),
                        Objective(name="Risk", direction="minimize", aggregation="quadratic")],
            options=[Option(name=n) for n in names], scores=scores,
            interaction_matrices=[InteractionMatrix(objective="Risk", entries=cov)],
        )
        run = optimize(p, mode="fast", seed=1, solver="highs")
        assert len(run.solutions) >= 10
        # Both tightening directions are actually swept: the cheap corner and the
        # short-lead corner are far apart, and each is reached.
        costs = [s.objective_values["Cost"] for s in run.solutions]
        leads = [s.objective_values["Lead"] for s in run.solutions]
        assert min(costs) <= 2.5
        assert min(leads) <= 2.5
