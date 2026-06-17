"""Progressive certify (`optimizer.certify_curated`): exact-solve only an existing run's frontier
points, the lean explore-then-certify overlay. Locks the faithfulness properties that let it stand in
for a full exact pass: a proper filtered exact Run, solver duals on the continuous path, idempotence on
an already-exact frontier, and coverage of every supported shape (binary MILP, proportional QP/LP)."""
import importlib.util as _ilu

import numpy as np
import pytest

from engine.optimizer import certify_curated, optimize
from engine.problem_io import examples_dir, read_bundle

_HAS_HIGHS = _ilu.find_spec("highspy") is not None


def _load(name):
    return read_bundle(examples_dir() / name)


def _ranges(run, objs):
    M = np.array([[s.objective_values[o] for o in objs] for s in run.solutions])
    return M.min(0), M.max(0)


@pytest.mark.parametrize("name", ["investment_portfolio", "budget_allocation"])
def test_certify_curated_is_a_filtered_exact_run(name):
    p = _load(name)
    nsga = optimize(p, seed=42)
    cert = certify_curated(p, nsga, solver="highs")

    assert cert.solver == "highs" and cert.exact is False          # stamped, not a heuristic run
    assert len(cert.solutions) > 0
    # Internally non-dominated (a proper Pareto frontier, like any exact Run).
    objs = [o.name for o in p.objectives]
    dirs = np.array([1.0 if o.direction.value == "minimize" else -1.0 for o in p.objectives])
    M = np.array([[s.objective_values[o] for o in objs] for s in cert.solutions]) * dirs
    for i in range(len(M)):
        dominated = np.all(M <= M[i] + 1e-9, axis=1) & np.any(M < M[i] - 1e-9, axis=1)
        assert not dominated.any(), "certified frontier contains a dominated point"
    # Continuous (QP/LP) points carry solver-exact duals — parity with the full exact pass.
    assert cert.solutions[0].sensitivity is not None


@pytest.mark.parametrize("name", ["investment_portfolio", "budget_allocation"])
def test_certify_curated_idempotent_on_exact_frontier(name):
    """Re-certifying an already-exact frontier reproduces it (each exact point is min-variance / optimal
    for its own epsilon targets, so it is a fixed point) — up to whole-percent allocation rounding."""
    p = _load(name)
    exact = optimize(p, seed=42, solver="highs")
    recert = certify_curated(p, exact, solver="highs")
    objs = [o.name for o in p.objectives]
    lo_e, hi_e = _ranges(exact, objs)
    lo_r, hi_r = _ranges(recert, objs)
    span = np.maximum(hi_e - lo_e, 1e-9)
    assert np.all(np.abs(lo_r - lo_e) / span < 0.05)               # same objective envelope, within rounding
    assert np.all(np.abs(hi_r - hi_e) / span < 0.05)


@pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
def test_certify_curated_binary_milp():
    """Binary selection is supported too: re-solve each NSGA point's scalarization as a 0/1 MILP."""
    from engine.problem_io import from_portable
    prob = from_portable(
        {"name": "x", "domain": "d", "context": "c", "approach": "binary",
         "objectives": [{"name": "Value", "direction": "maximize", "aggregation": "sum"},
                        {"name": "Cost", "direction": "minimize", "aggregation": "sum"}],
         "constraints": [{"type": "cardinality", "min": 2, "max": 4}]},
        {"options": [{"name": n} for n in "ABCDEF"],
         "scores": [{"option": o, "objective": ob, "value": float(v)}
                    for o, vv in zip("ABCDEF", [(9, 5), (7, 3), (8, 6), (5, 2), (6, 4), (4, 1)])
                    for ob, v in zip(["Value", "Cost"], vv)]})
    nsga = optimize(prob, seed=42)
    cert = certify_curated(prob, nsga, solver="highs")
    assert cert.solver == "highs" and len(cert.solutions) > 0
    # every certified pick respects the cardinality window and is a real 0/1 selection
    for s in cert.solutions:
        assert 2 <= len(s.selected_options) <= 4


def test_certify_curated_needs_exact_solver():
    p = _load("budget_allocation")
    nsga = optimize(p, seed=42)
    with pytest.raises(ValueError, match="exact solver"):
        certify_curated(p, nsga, solver="nsga")


# --- MCP surface: solve(solver=…, scope=…) — curated is the default exact overlay --------------
import tempfile

from mcp_server import server as srv
from engine.store import Store


@pytest.fixture()
def tmp_store(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(srv, "store", Store(d))
        yield


def _proportional_pid():
    pid = srv.model(action="create")["problem_id"]
    srv.model(action="update", problem_id=pid, approach="proportional",
              objectives=[{"name": "ROI", "direction": "maximize"},
                          {"name": "Reach", "direction": "maximize"}],
              options=[{"name": n} for n in ["A", "B", "C", "D", "E"]],
              scores=[{"option": o, "objective": ob, "value": v}
                      for o, ob, v in [("A", "ROI", 8), ("A", "Reach", 3), ("B", "ROI", 6), ("B", "Reach", 7),
                                       ("C", "ROI", 9), ("C", "Reach", 2), ("D", "ROI", 4), ("D", "Reach", 8),
                                       ("E", "ROI", 7), ("E", "Reach", 5)]],
              constraints=[{"type": "max_allocation", "max": 40}])
    return pid


@pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
def test_solve_scope_defaults_to_curated_after_a_run(tmp_store):
    pid = _proportional_pid()
    srv.solve(action="run", problem_id=pid, seed=42)                          # NSGA explore first
    res = srv.solve(action="run", problem_id=pid, seed=42, solver="highs")    # default exact overlay
    assert res["overlay_scope"] == "curated" and res["solver_used"] == "highs"
    p = srv.store.load(pid)
    assert p.run.solver.startswith("nsga") and p.exact_run.solver == "highs"  # overlay, run intact


@pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
def test_solve_scope_full_runs_full_pass(tmp_store):
    pid = _proportional_pid()
    srv.solve(action="run", problem_id=pid, seed=42)
    res = srv.solve(action="run", problem_id=pid, seed=42, solver="highs", scope="full")
    assert res["overlay_scope"] == "full" and res["solver_used"] == "highs"


@pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
def test_solve_scope_curated_falls_back_to_full_without_a_run(tmp_store):
    pid = _proportional_pid()
    res = srv.solve(action="run", problem_id=pid, seed=42, solver="highs")    # no prior NSGA run
    assert res["overlay_scope"] == "full"                                     # nothing to certify → full pass
