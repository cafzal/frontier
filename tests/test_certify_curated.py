"""Progressive certify (`optimizer.certify_curated`): exact-solve only an existing run's frontier
points, the lean explore-then-certify overlay. Locks the faithfulness properties that let it stand in
for a full exact pass: a proper filtered exact Run, solver duals on the continuous path, idempotence on
an already-exact frontier, and the proportional-only scope."""
import numpy as np
import pytest

from engine.optimizer import certify_curated, optimize
from engine.problem_io import examples_dir, read_bundle


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


def test_certify_curated_rejects_binary():
    p = _load("capital_project_selection_120")                     # binary MILP → full exact pass only
    nsga = optimize(p, seed=42)
    with pytest.raises(ValueError, match="proportional"):
        certify_curated(p, nsga, solver="highs")


def test_certify_curated_needs_exact_solver():
    p = _load("budget_allocation")
    nsga = optimize(p, seed=42)
    with pytest.raises(ValueError, match="exact solver"):
        certify_curated(p, nsga, solver="nsga")
