"""Pluggable solver backends for Frontier.

Each backend mirrors the engine's existing solve contract (same inputs, returns a
``Run`` of ``Solution``s in the identical shape) so it can be swapped in behind a
gate without any downstream changes. See ``cuopt_backend`` for the cuOpt QP spike
and ``highs_backend`` for the co-equal CPU exact backend.

This module is the **selection surface** the agent reaches through (via the ``solve``
tool's ``solver`` argument) and that ``engine.optimizer.optimize`` reaches through to
route a run. It owns two pure, dependency-light checks so both callers agree:

  * ``available_solvers()`` — which backends can actually run here (the exact ones
    depend on an optional import being present).
  * ``exact_solver_fits(problem)`` — whether a problem's *shape* is one the exact
    backends solve (binary selection, or a quadratic mean-variance portfolio).

Probing availability with ``importlib.util.find_spec`` keeps this importable with no
solver installed; the backends themselves import their solver lazily inside each
inner solve, so nothing here forces a GPU/`highspy` dependency on the default path.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.models import Problem

# The default engine (``nsga``) is pymoo's NSGA-II/III and is always present. The exact
# backends are keyed by the import that gates their availability.
EXACT_SOLVERS = ("highs", "cuopt")
_SOLVER_PROBE = {
    "nsga": None,      # default evolutionary engine — always available
    "highs": "highspy",
    "cuopt": "cuopt",
}


def available_solvers() -> dict[str, bool]:
    """Map each known solver key to whether it can run in this environment.

    ``nsga`` is always True (the default engine). An exact backend is available iff
    its optional dependency is importable — note this is a *presence* check, not a
    *runnable* one: cuOpt also needs a GPU at solve time, which only surfaces then.
    """
    return {
        key: (probe is None or importlib.util.find_spec(probe) is not None)
        for key, probe in _SOLVER_PROBE.items()
    }


def exact_solver_fits(problem: "Problem") -> tuple[bool, str]:
    """Whether an exact backend supports this problem's shape. Returns ``(fits, reason)``.

    Identical scope for both exact backends (they differ only in the inner solve):

      * **binary** select-a-subset → exact MILP.
      * **proportional + quadratic** (mean-variance) with an interaction matrix →
        exact convex QP, the EA picking the support under cardinality/group caps.

    Anything else stays with the default NSGA engine. The ``reason`` explains a
    rejection so the tool can tell the agent *why* an exact run was declined.
    """
    from engine import optimizer as _opt
    from engine.models import Aggregation, Approach, Direction

    if problem.approach == Approach.binary:
        return True, ""
    if problem.approach != Approach.proportional:
        return False, (
            "exact backends support binary selection or proportional mean-variance "
            f"portfolios; approach is '{getattr(problem.approach, 'value', problem.approach)}'"
        )
    quad = [o for o in problem.objectives if o.aggregation == Aggregation.quadratic]
    if not quad:
        return False, (
            "proportional problems need a quadratic-aggregated objective (mean-variance "
            "risk) for the exact QP path; this one has none"
        )
    # The exact QP minimizes wᵀQw — it is strictly a min-variance/risk solver. A maximize
    # quadratic objective is non-convex maximization, outside that shape; the QP would
    # silently minimize it and return a degenerate frontier. Require minimize so the gate
    # only accepts a genuine mean-variance formulation (matches the skill's "risk objective").
    if any(o.direction != Direction.minimize for o in quad):
        return False, (
            "the exact QP is a min-variance solver, so the quadratic objective must be "
            "minimize (risk/variance); a maximize quadratic is non-convex and out of scope"
        )
    if len(_opt._build_interaction_matrices(problem)) == 0:
        return False, (
            "the quadratic objective needs an interaction (covariance) matrix for the "
            "exact QP path"
        )
    return True, ""


def is_exact_solver(solver: str | None) -> bool:
    """Whether a solver name denotes an exact backend (vs the default heuristic NSGA).

    The one place that answers "did/should an exact backend produce this?" — pass a
    ``Run.solver`` or a requested backend name. Tolerates ``None``/empty (an unstamped or
    legacy run counts as heuristic). Centralizes the ``EXACT_SOLVERS`` membership test so
    every caller classifies runs the same way.
    """
    return (solver or "") in EXACT_SOLVERS
