"""Tests for agent-facing solver selection — the wiring that lets the agent opt into an
exact backend per run alongside the default evolutionary search.

Covers the shared selection surface (``solvers.available_solvers`` /
``solvers.exact_solver_fits``), ``optimize(solver=...)`` routing + run stamping, and the
``solve`` tool's guard/echo behavior. The exact backends themselves are tested in
``test_highs_backend``; here we care about *selection*, not the inner solve, so the
HiGHS-dependent cases are skipped when ``highspy`` isn't installed.
"""

import importlib.util
import tempfile

import pytest

import mcp_server.server as srv
from engine.models import (
    CardinalityConstraint,
    InteractionMatrix,
    Objective,
    OptimizeMode,
    Option,
    Problem,
    Score,
)
from engine.optimizer import optimize
from engine.store import Store
from solvers import available_solvers, exact_solver_fits, is_exact_solver

_HAS_HIGHS = importlib.util.find_spec("highspy") is not None


@pytest.fixture(autouse=True)
def tmp_store(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Store(tmpdir)
        monkeypatch.setattr(srv, "store", s)
        srv._injected_skills.clear()
        yield s


# ─── Fixtures ───

def _binary_problem(**overrides):
    names = ["A", "B", "C", "D", "E"]
    table = {"NPV": [9, 7, 8, 4, 6], "Cost": [5, 3, 6, 2, 4]}
    scores = [Score(option=n, objective=o, value=col[i])
              for o, col in table.items() for i, n in enumerate(names)]
    defaults = dict(
        approach="binary",
        objectives=[Objective(name="NPV", direction="maximize"),
                    Objective(name="Cost", direction="minimize")],
        options=[Option(name=n) for n in names],
        scores=scores,
        constraints=[CardinalityConstraint(min=1, max=3)],
    )
    defaults.update(overrides)
    return Problem(**defaults)


def _four_objective_problem():
    names = ["A", "B", "C", "D", "E"]
    objs = ["W", "X", "Y", "Z"]
    scores = [Score(option=n, objective=o, value=(i + j + 1))
              for j, o in enumerate(objs) for i, n in enumerate(names)]
    return Problem(
        approach="binary",
        objectives=[Objective(name=o, direction="maximize") for o in objs],
        options=[Option(name=n) for n in names],
        scores=scores,
        constraints=[CardinalityConstraint(min=1, max=3)],
    )


def _proportional_no_quad():
    """Proportional but sum-aggregated — NOT a mean-variance shape, so exact backends decline."""
    names = ["A", "B", "C"]
    scores = [Score(option=n, objective=o, value=v)
              for o in ("Return", "Cost")
              for n, v in zip(names, [10, 12, 8])]
    return Problem(
        approach="proportional",
        objectives=[Objective(name="Return", direction="maximize", aggregation="avg"),
                    Objective(name="Cost", direction="minimize", aggregation="sum")],
        options=[Option(name=n) for n in names],
        scores=scores,
    )


def _make_solvable_problem_via_tool() -> str:
    """Create a ready-to-solve binary problem through the server tool; return problem_id."""
    pid = srv.model(action="create")["problem_id"]
    srv.model(action="update", problem_id=pid,
              objectives=[{"name": "Rev", "direction": "maximize"},
                          {"name": "Eff", "direction": "minimize"}],
              options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
              scores=[{"option": o, "objective": ob, "value": v}
                      for o, ob, v in [("A", "Rev", 8), ("A", "Eff", 5),
                                       ("B", "Rev", 6), ("B", "Eff", 3),
                                       ("C", "Rev", 9), ("C", "Eff", 7)]],
              constraints=[{"type": "cardinality", "min": 1, "max": 2}])
    return pid


# ─── Shared selection surface ───

class TestSelectionSurface:
    def test_nsga_always_available(self):
        assert available_solvers()["nsga"] is True

    def test_keys_present(self):
        avail = available_solvers()
        assert set(avail) == {"nsga", "highs", "cuopt"}

    def test_is_exact_solver_classifies(self):
        # The single classifier the explorer/server/optimizer all share.
        assert is_exact_solver("highs") and is_exact_solver("cuopt")
        assert not is_exact_solver("nsga-ii") and not is_exact_solver("nsga-iii")
        assert not is_exact_solver(None) and not is_exact_solver("")

    def test_binary_fits_exact(self):
        fits, reason = exact_solver_fits(_binary_problem())
        assert fits is True and reason == ""

    def test_proportional_without_quadratic_does_not_fit(self):
        fits, reason = exact_solver_fits(_proportional_no_quad())
        assert fits is False
        assert "quadratic" in reason

    def test_maximize_quadratic_does_not_fit(self):
        # The exact QP is a min-variance solver; a maximize-direction quadratic objective
        # (e.g. channel_budget's "Reach") is non-convex maximization and must be declined,
        # not silently minimized into a degenerate frontier.
        names = ["A", "B", "C"]
        cov = {a: {b: (0.1 if a == b else 0.01) for b in names} for a in names}
        scores = [Score(option=n, objective=o, value=v)
                  for o in ("Reach", "Cost")
                  for n, v in zip(names, [10, 12, 8])]
        p = Problem(
            approach="proportional",
            objectives=[Objective(name="Reach", direction="maximize", aggregation="quadratic"),
                        Objective(name="Cost", direction="minimize", aggregation="sum")],
            options=[Option(name=n) for n in names],
            scores=scores,
            interaction_matrices=[InteractionMatrix(objective="Reach", entries=cov)],
        )
        fits, reason = exact_solver_fits(p)
        assert fits is False
        assert "minimize" in reason


# ─── optimize() routing + run stamping ───

class TestOptimizeStamping:
    def test_default_stamps_nsga_ii(self):
        run = optimize(_binary_problem(), mode=OptimizeMode.fast, seed=1)
        assert run.solver == "nsga-ii"
        assert run.exact is False

    def test_four_objectives_stamps_nsga_iii(self):
        run = optimize(_four_objective_problem(), mode=OptimizeMode.fast, seed=1)
        assert run.solver == "nsga-iii"

    def test_ill_fitting_exact_request_raises(self):
        # Requesting an exact solver on a shape it can't solve raises rather than silently
        # degrading to NSGA (no-silent-degradation). The server tool catches this earlier and
        # returns a clean error dict; the library layer fails loud.
        with pytest.raises(ValueError, match="does not fit"):
            optimize(_proportional_no_quad(), mode=OptimizeMode.fast, seed=1, solver="highs")

    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            optimize(_binary_problem(), mode=OptimizeMode.fast, seed=1, solver="bogus")

    @pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
    def test_explicit_highs_stamps_highs(self):
        run = optimize(_binary_problem(), mode=OptimizeMode.fast, seed=1, solver="highs")
        assert run.solver == "highs"


# ─── solve tool: validate surfaces solvers, run guards + echoes ───

class TestSolveToolSelection:
    def test_validate_reports_solvers(self):
        pid = _make_solvable_problem_via_tool()
        res = srv.solve(action="validate", problem_id=pid)
        assert "solvers" in res
        assert res["solvers"]["default"] == "nsga"
        assert "nsga" in res["solvers"]["available"]
        assert res["solvers"]["exact_fits_shape"] is True  # binary problem

    def test_run_default_echoes_solver_used(self):
        pid = _make_solvable_problem_via_tool()
        res = srv.solve(action="run", problem_id=pid)
        assert res["solver_used"] == "nsga-ii"
        assert res["exact"] is False

    def test_run_unknown_solver_errors(self):
        pid = _make_solvable_problem_via_tool()
        res = srv.solve(action="run", problem_id=pid, solver="bogus")
        assert "error" in res
        assert "bogus" in res["error"]

    def test_run_unavailable_solver_errors(self, monkeypatch):
        pid = _make_solvable_problem_via_tool()
        # Simulate highs not installed regardless of the test environment. _resolve_solver
        # does `from solvers import available_solvers` at call time, so patching the module
        # attribute takes effect.
        monkeypatch.setattr("solvers.available_solvers",
                            lambda: {"nsga": True, "highs": False, "cuopt": False})
        res = srv.solve(action="run", problem_id=pid, solver="highs")
        assert "error" in res
        assert "not available" in res["error"]

    @pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
    def test_run_ill_fitting_shape_errors_cleanly(self):
        # An exact solver requested for a non-fitting shape returns a clean error dict from the
        # tool (guarded by _resolve_solver) — optimize()'s raise is never reached, so the tool
        # never surfaces an exception. Availability is checked first, so this needs highs present.
        pid = srv.model(action="create")["problem_id"]
        srv.model(action="update", problem_id=pid, approach="proportional",
                  objectives=[{"name": "Rev", "direction": "maximize", "aggregation": "avg"},
                              {"name": "Cost", "direction": "minimize", "aggregation": "sum"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
                  scores=[{"option": o, "objective": ob, "value": v}
                          for o, ob, v in [("A", "Rev", 8), ("A", "Cost", 5),
                                           ("B", "Rev", 6), ("B", "Cost", 3),
                                           ("C", "Rev", 9), ("C", "Cost", 7)]])
        res = srv.solve(action="run", problem_id=pid, solver="highs")
        assert "error" in res and "doesn't fit" in res["error"]

    @pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
    def test_run_highs_echoes_highs(self):
        pid = _make_solvable_problem_via_tool()
        res = srv.solve(action="run", problem_id=pid, solver="highs")
        assert res["solver_used"] == "highs"

    @pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
    def test_exact_run_is_overlay_not_replacement(self):
        # The exact solver populates exact_run and leaves the exploratory `run` (NSGA) intact,
        # so the problem holds both frontiers at once.
        pid = _make_solvable_problem_via_tool()
        srv.solve(action="run", problem_id=pid, seed=42)                 # NSGA → run
        srv.solve(action="run", problem_id=pid, seed=42, solver="highs") # highs → exact_run
        p = srv.store.load(pid)
        assert p.run is not None and p.run.solver.startswith("nsga")
        assert p.exact_run is not None and p.exact_run.solver == "highs"
        # both are reachable via model get sections
        assert srv.model(action="get", problem_id=pid, section="exact_run")["exact_run"]["solver"] == "highs"

    @pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
    def test_exact_only_solve_is_explorable(self):
        # Solving ONLY with an exact solver (no NSGA run) stores the frontier in
        # exact_run with p.run=None; explore must fall back to it rather than report
        # "no run", and status must register the problem as solved.
        pid = _make_solvable_problem_via_tool()
        res = srv.solve(action="run", problem_id=pid, seed=42, solver="highs")
        assert res["solutions_found"] > 0
        p = srv.store.load(pid)
        assert p.run is None and p.exact_run is not None

        assert "error" not in srv.explore(action="tradeoffs", problem_id=pid)
        assert "error" not in srv.explore(action="solutions", problem_id=pid)
        assert srv.model(action="get", problem_id=pid, section="summary")["has_exact_run"] is True

    @pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
    def test_source_exact_targets_overlay_when_both_present(self):
        # With both an exploratory run and an exact overlay, explore defaults to the
        # NSGA run but source="exact" retargets the analytics at the exact-solver frontier.
        pid = _make_solvable_problem_via_tool()
        srv.solve(action="run", problem_id=pid, seed=42)                  # NSGA → run
        srv.solve(action="run", problem_id=pid, seed=42, solver="highs")  # highs → exact_run
        p = srv.store.load(pid)
        assert p.run is not None and p.exact_run is not None

        default = srv.explore(action="solutions", problem_id=pid)
        exact = srv.explore(action="solutions", problem_id=pid, source="exact")
        assert "error" not in default and "error" not in exact
        assert exact["run_id"] == p.exact_run.run_id
        assert default["run_id"] == p.run.run_id
        # analytics run against the exact frontier
        assert "error" not in srv.explore(action="tradeoffs", problem_id=pid, source="exact")
        # invalid source is rejected
        assert "Unknown source" in srv.explore(action="solutions", problem_id=pid, source="bogus")["error"]

    @pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
    def test_exact_solve_drops_stale_exploratory_run(self):
        # NSGA run, then edit the problem (results go stale), then exact solve. The
        # stale exploratory run predates the edit, so it must be dropped rather than
        # silently marked fresh when results_stale is cleared.
        pid = _make_solvable_problem_via_tool()
        srv.solve(action="run", problem_id=pid, seed=42)  # NSGA → run
        srv.model(action="update", problem_id=pid,
                  constraints=[{"type": "cardinality", "min": 1, "max": 3}])  # edit → results_stale
        assert srv.store.load(pid).results_stale is True
        srv.solve(action="run", problem_id=pid, seed=42, solver="highs")  # exact overlay
        p = srv.store.load(pid)
        assert p.results_stale is False
        assert p.exact_run is not None and p.exact_run.solver == "highs"
        assert p.run is None  # stale NSGA run dropped, not falsely vouched for
