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
    backends solve (binary selection, a quadratic mean-variance portfolio, or a
    purely linear proportional allocation).

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
      * **proportional + purely linear** (no quadratic), ≥2 objectives → exact
        multi-objective LP — a linear objective optimized, the rest ε-constrained.

    Anything else stays with the default NSGA engine. The ``reason`` explains a
    rejection so the tool can tell the agent *why* an exact run was declined.
    Composition contract: reasons are complete sentences (most carry a trailing
    period) — a caller embedding one mid-message strips the trailing period.
    """
    from engine import optimizer as _opt
    from engine.models import Aggregation, Approach, Direction

    if problem.approach == Approach.binary:
        # The binary inner solve is a linear MILP (minimize Σ coef·x), so it represents only
        # additive (sum) objectives. `avg` is fractional over a variable-size selection;
        # `min`/`max`/`quadratic` are nonlinear — none linearize into the MILP, and silently
        # optimizing their sum would mis-certify (NSGA evaluates the true aggregation, the
        # exact run the sum, so the certificate compares different objective spaces). Decline
        # with a redefine hint instead.
        nonsum = [o for o in problem.objectives if o.aggregation != Aggregation.sum]
        if nonsum:
            present = sorted({o.aggregation.value for o in nonsum})
            # Explain only the aggregations actually present — a decline that lectures
            # about shapes the model doesn't use reads as a wrong diagnosis.
            because = {"avg": "avg is fractional over a variable-size pick",
                       "min": "min is nonlinear", "max": "max is nonlinear",
                       "quadratic": "quadratic is nonlinear"}
            names = ", ".join(o.name for o in nonsum)
            verb = "use" if len(nonsum) > 1 else "uses"
            return False, (
                f"the exact MILP optimizes additive (sum) objectives, and on this binary "
                f"selection {names} {verb} {', '.join(present)} aggregation — "
                f"{'; '.join(because[a] for a in present)} — outside the exact scope. "
                "Redefine these as sum to certify, or keep the aggregation and explore "
                "with the NSGA heuristic."
            )
        return True, ""
    if problem.approach != Approach.proportional:
        return False, (
            "exact backends support binary selection or proportional mean-variance "
            f"portfolios; approach is '{getattr(problem.approach, 'value', problem.approach)}'"
        )
    # Membership logic beyond a box is combinatorial on a continuous shape: an exclusion
    # pair (never both active), a dependency (one active forces another), and a cardinality
    # floor above 1 (at least K active) are all semicontinuous — the convex QP/LP can't
    # express them, and NSGA enforces them exactly. (force_include / force_exclude ARE in
    # scope — they fold into the variable box as a 1% floor / 0 cap.)
    combinatorial = sorted({c.type for c in (problem.constraints or [])
                            if c.type in ("exclusion_pair", "dependency")
                            or (c.type == "cardinality" and int(c.min or 0) > 1)})
    if combinatorial:
        return False, (
            f"{', '.join(combinatorial)} constraints on a proportional allocation are "
            "combinatorial (they gate which options are ACTIVE), outside the exact QP/LP "
            "scope — explore with the NSGA heuristic, which enforces them."
        )
    # Group floors on a proportional shape are a count of *active* options — combinatorial
    # (semicontinuous), outside the convex QP/LP scope. Caps stay in scope: the EA's support
    # decode enforces them. The binary MILP handles floors natively, so this gate is
    # proportional-only.
    floored = [c for c in (problem.constraints or [])
               if c.type == "group_limit" and int(getattr(c, "min", 0) or 0) > 0]
    if floored:
        return False, (
            "group_limit minimums on a proportional allocation count *active* options — "
            "combinatorial, outside the exact QP/LP scope. Drop the floors to certify, or "
            "explore with the NSGA heuristic (which enforces them)."
        )
    # The proportional inner solve is a convex QP: linear (sum/avg) objectives plus one
    # quadratic variance term. `min`/`max` are nonlinear and out of scope.
    nonlinear = [o for o in problem.objectives if o.aggregation in (Aggregation.min, Aggregation.max)]
    if nonlinear:
        aggs = ", ".join(sorted({o.aggregation.value for o in nonlinear}))
        names = ", ".join(o.name for o in nonlinear)
        return False, (
            f"the exact QP optimizes linear (sum/avg) and quadratic objectives; {names} use "
            f"{aggs} aggregation (nonlinear), out of exact scope. Redefine as sum/avg, or explore "
            "with the NSGA heuristic."
        )
    quad = [o for o in problem.objectives if o.aggregation == Aggregation.quadratic]
    # The QP shape has exactly one quadratic minimand swept against linear objectives —
    # a second quadratic can be neither the minimand nor an epsilon row, so decline it in
    # words rather than mis-solve. (The all-quadratic single-objective case is unreachable:
    # engine validation requires ≥2 objectives before any solve.)
    if len(quad) > 1:
        names = ", ".join(o.name for o in quad)
        return False, (
            f"the exact QP minimizes a single variance term; {names} are all quadratic. "
            "Keep one quadratic risk objective (fold the rest into its interaction matrix, or "
            "redefine them as sum/avg), or explore with the NSGA heuristic."
        )
    # A model objective_bound on the quadratic objective can't be a linear row
    # (``_model_bound_rows`` skips it). A MAX cap on the quadratic minimand is still exactly
    # servable: each inner solve MINIMIZES the quadratic, so any returned point above the cap
    # proves those epsilon-targets infeasible and the QP paths filter it out post-solve. A MIN
    # floor on the quadratic is non-convex — decline that direction.
    quad_names = {o.name for o in quad}
    quad_floors = sorted({c.objective for c in (problem.constraints or [])
                          if c.type == "objective_bound" and c.objective in quad_names
                          and getattr(c.operator, "value", c.operator) == "min"})
    if quad_floors:
        return False, (
            f"an objective_bound floor on the quadratic objective ({', '.join(quad_floors)}) is "
            "non-convex — the exact QP can cap it (max) but not floor it. Drop the floor to "
            "certify, or explore with the NSGA heuristic (which enforces it)."
        )
    if not quad:
        # No quadratic term → purely linear proportional allocation is an exact multi-objective LP
        # (one linear objective optimized, the rest epsilon-constrained), carrying shadow prices +
        # reduced costs just like the QP path. Needs >=2 objectives for a frontier; a single linear
        # objective is a trivial one-shot allocation, left to NSGA for now.
        if len(problem.objectives) < 2:
            return False, (
                "a purely linear proportional problem needs >=2 objectives for the exact LP "
                "frontier; with one linear objective the allocation is a trivial single solve — "
                "use the NSGA heuristic, or add a mean-variance risk objective for the QP path"
            )
        return True, ""
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


# --- Scale bands (advisory) --------------------------------------------------
# The shape gate above answers "which engine CAN solve this"; the scale band
# answers "what posture fits this size". Thresholds are measured regime
# boundaries on the real optimize() path (binary, 3 objectives, fast mode,
# seed 42, max_solutions=1000, time_limit=300s, post binary-hash-dedup fix,
# 2026-07): n=300 → 1.5s · n=1,000 → 12s · n=3,000 → 117s converged ·
# n=10,000 → 311s (cap hit, best-so-far usable). Re-measure after engine
# changes; Run.telemetry (solve telemetry stage 1) is the production record
# these get recalibrated against.
_NSGA_BACKGROUND_N = 1_000  # 12s measured — past the ~10s inline window, expect a background job
_NSGA_ROUTING_N = 10_000    # 311s measured, wall-clock cap hit — wants the scale posture, not just patience


def scale_band(problem: "Problem") -> dict:
    """Advisory scale signal for ``solve validate``'s ``solvers`` block.

    Routes toward the lane that fits the size, never warns users off big
    problems: the ``note`` names the measured boundary the problem sits near
    and the posture that fits (background polling, a ``time_limit``,
    curated-scope certification). Advisory only — validate's ready status is
    untouched and nothing blocks. Demo-scale problems get band ``interactive``
    and no note: zero noise until scale actually changes the right move.
    """
    n = len(problem.options)
    block: dict = {"n_options": n, "band": "interactive"}
    if n >= _NSGA_ROUTING_N:
        block["band"] = "needs_routing"
        block["note"] = (
            f"At {n:,} options a solve runs to its wall-clock budget and returns a usable "
            "best-so-far frontier (measured: ~5 min at 10,000) — set a time_limit, keep "
            "mode='fast', poll via solve status, and certify with scope='curated' "
            "(targeted points) rather than a full exact pass."
        )
    elif n >= _NSGA_BACKGROUND_N:
        block["band"] = "background"
        block["note"] = (
            f"At {n:,} options expect a background solve (measured: ~12s at 1,000, "
            "~2 min at 3,000) — keep mode='fast' while iterating, poll via solve "
            "status, and prefer scope='curated' for the exact overlay."
        )
    return block


def problem_features(problem: "Problem") -> dict:
    """Feature snapshot of a problem at solve time — ``Run.problem_snapshot``.

    The feature-space sibling of ``constraints_snapshot``: deterministic facts about
    the problem *as solved*, recorded so solve telemetry can later be read against
    problem scale/shape (which engine ran on what, and how long it took). Lives
    beside ``exact_solver_fits`` on purpose — the snapshot's ``exact_fits`` and the
    gate share one implementation, so features and gate can never drift.
    """
    from engine.models import Aggregation

    fits, reason = exact_solver_fits(problem)
    agg_mix: dict[str, int] = {}
    for o in problem.objectives:
        key = getattr(o.aggregation, "value", str(o.aggregation))
        agg_mix[key] = agg_mix.get(key, 0) + 1
    constraint_counts: dict[str, int] = {}
    for c in problem.constraints or []:
        constraint_counts[c.type] = constraint_counts.get(c.type, 0) + 1
    return {
        "n_options": len(problem.options),
        "n_objectives": len(problem.objectives),
        "n_scores": len(problem.scores),
        "approach": getattr(problem.approach, "value", str(problem.approach)),
        "aggregation_mix": agg_mix,
        "constraint_type_counts": constraint_counts,
        # Same inclusion rule as optimizer._build_interaction_matrices (a quadratic
        # objective with a matching matrix), WITHOUT building the dense n×n arrays —
        # this runs on every solve, and the build is O(n²) memory at scale.
        "interaction_matrix": bool(
            {o.name for o in problem.objectives if o.aggregation == Aggregation.quadratic}
            & {m.objective for m in problem.interaction_matrices}
        ),
        "scenarios": len(problem.scenario_config.scenarios) if problem.scenario_config else 0,
        "exact_fits": fits,
        "exact_fits_reason": reason,
    }


def is_exact_solver(solver: str | None) -> bool:
    """Whether a solver name denotes an exact backend (vs the default heuristic NSGA).

    The one place that answers "did/should an exact backend produce this?" — pass a
    ``Run.solver`` or a requested backend name. Tolerates ``None``/empty (an unstamped or
    legacy run counts as heuristic). Centralizes the ``EXACT_SOLVERS`` membership test so
    every caller classifies runs the same way.
    """
    return (solver or "") in EXACT_SOLVERS


def run_is_certified(run, approach) -> bool:
    """Whether every point on ``run`` carries an optimality certificate.

    The one place that answers "is this overlay certified?" (the ``exact_certified``
    surfaces): the continuous proportional path (LP/QP scalarization) is exact by
    construction, so any exact-backend proportional run qualifies; a binary MILP run
    qualifies only at a zero gap (``exact=True`` — the default accepts 0.1%-gap
    incumbents); heuristic NSGA runs never do. ``approach`` is ``problem.approach``
    (enum or plain string).
    """
    if not is_exact_solver(getattr(run, "solver", None)):
        return False
    if bool(getattr(run, "exact", False)):
        return True
    return getattr(approach, "value", approach) == "proportional"
