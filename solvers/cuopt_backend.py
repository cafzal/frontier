"""cuOpt inner-solve backend for Frontier's exact NSGA-scalarization engine.

The NSGA outer loop, genome decoding, seeding, and result marshaling live in
``solvers._scalarization`` and are shared with the HiGHS backend; this module supplies only
the **cuOpt-specific inner solves** (a GPU convex QP and a GPU MILP) plus a thin delegation.
cuOpt and HiGHS differ *only* in these inner solves.

This is **additive, reversible**: the engine routes here only when ``optimize(solver="cuopt")``
is requested for a shape ``solvers.exact_solver_fits`` accepts; cuOpt is imported lazily
*inside* each inner solve, so the module loads cleanly on a machine with no GPU (an actual
solve needs ``cuopt-cu12`` + a GPU, which is why the prototype runs in Colab); and the
engine's Run/Solution contract is preserved by the shared engine, so explorer / metrics /
store need no changes.

EA ↔ cuOpt decomposition: the EA individual encodes a scalarization parameter (the
epsilon-constraint return target, plus an asset-selection priority under a cardinality/group
cap), and cuOpt maps it to the optimal weights::

    EA individual → r_target → cuOpt: min wᵀΣw s.t. μᵀw ≥ r_target, Σw=1, w≥0 → w* → (risk, return)

On a convex QP this makes the EA a frontier *walker* (a plain epsilon sweep matches it); it
earns its keep once non-convex structure (cardinality / group caps) makes r_target→w*
multimodal, where the EA picks the support and cuOpt solves the continuous QP on it exactly.
"""

from __future__ import annotations

import gc

import numpy as np

from engine.models import Aggregation, Approach, OptimizeMode, Problem, Run
from solvers._scalarization import (
    _build_raw_sensitivity,
    _qp_weights_ok,
    certify_curated_frontier,
    optimize_lp,
    optimize_milp,
    optimize_qp,
)

# Re-exported from the shared engine so notebooks/panels that reach into this module by its
# historical name (e.g. ``cuopt_backend._nearest_psd``) keep resolving after the extraction.
from solvers._scalarization import (  # noqa: F401  (back-compat re-exports)
    _build_milp_data,
    _group_limits,
    _nearest_psd,
    _resolve_linear_objectives,
    _resolve_objective_roles,
)

# Deliberately small EA budget (smallest pop/gen): each cuOpt call is an expensive GPU solve,
# so the total inner-solve count is kept bounded on Colab.
_SPIKE_POP = 30
_SPIKE_GEN = 15

# Binary-MILP path. Each cuOpt MILP solve gets a wall-clock cap and a mild relative gap.
# Without them, branch-and-bound finds the integer incumbent in <0.1s but then spends
# *minutes* certifying optimality across a ~0.05% gap; on a small Colab box that runaway
# exhausts host RAM (kernel restart) and, multiplied across the EA's many inner solves, never
# returns. A 0.1% gap on integer scores is sub-unit, so the returned incumbent is the exact
# optimum — only the (irrelevant) proof is skipped.
_MILP_TIME_LIMIT = 8.0   # seconds, hard cap per MILP solve
_MILP_REL_GAP = 1e-3     # stop B&B once the incumbent is within 0.1% of the bound
# Absolute MILP gap (default off). Set < the score granularity in code to make the
# bounded-mode incumbent provably optimal cheaply; `optimize(..., exact=True)` certifies.
# (The MILP EA pop/gen budget auto-scales in _scalarization._milp_budget — shared with HiGHS.)
_MILP_ABS_GAP = 0.0

# Matrix (CSR `data_model`) QP path. The term-by-term high-level `Problem` API builds the
# quadratic objective as an O(n²) Python expression — one `+` per covariance entry — which is
# the real ceiling on a dense-covariance QP, *not* cuOpt itself. cuOpt's low-level
# `data_model.DataModel` takes the objective Q and constraint matrix A directly as CSR arrays,
# so the build is a single vectorised numpy→scipy conversion (O(nnz)) and the GPU solve is
# unchanged. `_solve_qp_cuopt_matrix` is that path; it passes the SAME (PSD-projected)
# covariance the verified term-by-term path packs into `set_quadratic_objective_matrix`
# (cuOpt symmetrises Q+Qᵀ internally either way), so the two are equivalent *by construction*
# and differ only in build cost. The GPU A/B confirmed equivalence (matrix vs term-by-term:
# 0.00e+00 weight max-diff on a dense n=60 QP, identical optimum out to n=1500 in the comparison
# notebook's dense-QP panel), so this is now the default cuOpt QP build — `optimize(solver="cuopt")`
# proportional runs go through the scalable O(nnz) construction. (MILP still builds term-by-term.)
_USE_MATRIX_QP = True


def _solve_qp_cuopt(
    cov: np.ndarray,
    mu: np.ndarray,
    target_return: float | None,
    return_maximize: bool,
    max_weight: float | None,
    support: "np.ndarray | list[int] | None" = None,
    extra_linears: "list[tuple[np.ndarray, float, bool]] | None" = None,
) -> tuple[np.ndarray, bool]:
    """One epsilon-constraint inner solve via cuOpt. Mirrors NVIDIA's
    ``QP_portfolio_optimization.ipynb`` fixture::

        minimize   wᵀΣw
        subject to Σw = 1,  0 ≤ w ≤ max_weight
                   μᵀw ≥ target_return     (epsilon-constraint, injected per individual)

    Two optional extensions: ``support`` restricts holdable assets to an index set by pinning
    excluded assets' upper bound to 0 (cardinality — the EA picks which K, cuOpt solves the
    exact QP on that support, so it stays in the continuous-QP beta with no MIQP);
    ``extra_linears`` adds linear epsilon-constraints ``(coef, target, maximize)`` for >2
    objectives (e.g. a yield floor). cuOpt is imported here so the module loads without a GPU.
    Returns ``(weights as fractions summing to 1, optimal_flag)``.
    """
    from cuopt.linear_programming.problem import Problem as CuProblem, MINIMIZE

    n = len(mu)
    prob = CuProblem("frontier_portfolio_qp")
    ub = float(max_weight) if max_weight is not None else 1.0
    # Cardinality support: excluded assets get a 0 upper bound, so the QP is solved exactly
    # over the chosen subset without leaving the continuous beta.
    if support is None:
        ubs = [ub] * n
    else:
        supp = {int(i) for i in support}
        ubs = [ub if i in supp else 0.0 for i in range(n)]
    w = [prob.addVariable(lb=0.0, ub=ubs[i], name=f"w_{i}") for i in range(n)]

    # Quadratic objective wᵀΣw, built term-by-term exactly as the fixture does.
    quad = None
    for i in range(n):
        for j in range(n):
            c = float(cov[i, j])
            if abs(c) > 1e-12:
                term = c * w[i] * w[j]
                quad = term if quad is None else quad + term
    prob.setObjective(quad, sense=MINIMIZE)

    # Fully invested.
    prob.addConstraint(sum(w) == 1, name="fully_invested")

    # Epsilon-constraint on the primary (return) objective. Direction-aware: ≥ for a maximize
    # objective. Named so the dual (shadow price) is readable.
    if target_return is not None:
        ret_expr = sum(float(mu[i]) * w[i] for i in range(n))
        if return_maximize:
            prob.addConstraint(ret_expr >= float(target_return), name="return_target")
        else:
            prob.addConstraint(ret_expr <= float(target_return), name="return_target")

    # Extra linear epsilon-constraints (e.g. a yield floor) — one per objective beyond
    # risk+return, so the same QP serves 3+ objectives.
    for k, (coef, tgt, maximize) in enumerate(extra_linears or []):
        expr = sum(float(coef[i]) * w[i] for i in range(n))
        if maximize:
            prob.addConstraint(expr >= float(tgt), name=f"linear_{k}")
        else:
            prob.addConstraint(expr <= float(tgt), name=f"linear_{k}")

    prob.solve()
    # ``prob.Status`` is an ``LPTerminationStatus`` IntEnum. Accept BOTH certified Optimal
    # (==1) and PrimalFeasible (==7): cuOpt's PDLP (the first-order solver behind the QP beta)
    # frequently terminates PrimalFeasible rather than certified-Optimal on these convex QPs,
    # and NVIDIA's own portfolio reference treats both as solved.
    ok = getattr(prob.Status, "name", "") in ("Optimal", "PrimalFeasible")
    weights = np.array([w[i].Value for i in range(n)], dtype=float) if ok else np.zeros(n)
    # Status says "solved" but the returned point can still be degenerate (non-finite or off
    # the Σw=1 / box) — gate on the weights themselves, else one bad solve blows up the frontier.
    if ok and not _qp_weights_ok(weights, ub):
        weights, ok = np.zeros(n), False
    # Free the cuOpt problem before the next inner solve (the EA issues many).
    del prob
    gc.collect()
    return weights, ok


def _qp_to_csr(cov, mu, target_return, return_maximize, max_weight,
               support=None, extra_linears=None):
    """Marshal one epsilon-constraint mean-variance QP into the CSR arrays cuOpt's low-level
    ``data_model.DataModel`` consumes — the vectorised matrix build that replaces the
    term-by-term ``Problem`` API. **Pure** (no cuOpt import), so it unit-tests on CPU by
    reconstructing the dense problem and comparing to the intended QP.

    Objective: minimize wᵀ(cov)w. ``Q`` is the FULL covariance as CSR — bit-for-bit the matrix
    the verified term-by-term path packs into ``set_quadratic_objective_matrix`` (cuOpt
    symmetrises Q+Qᵀ internally), so the solve is identical; only the build differs (O(nnz) one
    vectorised conversion vs O(n²) Python ``+``). Sub-1e-12 entries are dropped to match that
    path's ``abs(c) > 1e-12`` skip exactly. Constraint rows, in order: fully-invested Σw=1
    ('E'), the return epsilon-constraint μ·w ≥/≤ target ('G' for a maximize return, else 'L'),
    then one row per extra linear floor. Box: 0 ≤ w ≤ ``max_weight``, with off-``support`` assets
    pinned to ub=0 (cardinality, stays in the continuous beta — no MIQP).

    Returns a dict of numpy arrays ready for the DataModel setters.
    """
    from scipy.sparse import csr_matrix

    n = len(mu)
    ub = float(max_weight) if max_weight is not None else 1.0
    var_ub = np.full(n, ub, dtype=np.float64)
    if support is not None:
        supp = {int(i) for i in support}
        var_ub[[i for i in range(n) if i not in supp]] = 0.0

    # Quadratic objective Q = full covariance (threshold to match the term-wise skip), as CSR.
    Q = csr_matrix(np.where(np.abs(cov) > 1e-12, cov, 0.0).astype(np.float64))

    # Constraint matrix A (few dense rows): fully-invested, return eps, extra linear floors.
    rows = [np.ones(n, dtype=np.float64)]
    b = [1.0]
    row_types = ["E"]
    if target_return is not None:
        rows.append(np.asarray(mu, dtype=np.float64))
        b.append(float(target_return))
        row_types.append("G" if return_maximize else "L")
    for coef, tgt, maximize in (extra_linears or []):
        rows.append(np.asarray(coef, dtype=np.float64))
        b.append(float(tgt))
        row_types.append("G" if maximize else "L")
    A = csr_matrix(np.vstack(rows))

    return {
        "Q_data": Q.data.astype(np.float64),
        "Q_indices": Q.indices.astype(np.int32),
        "Q_offsets": Q.indptr.astype(np.int32),
        "A_data": A.data.astype(np.float64),
        "A_indices": A.indices.astype(np.int32),
        "A_offsets": A.indptr.astype(np.int32),
        "b": np.asarray(b, dtype=np.float64),
        "row_types": np.array(row_types, dtype="S1"),   # matches the high-level path's dtype
        "c": np.zeros(n, dtype=np.float64),              # objective is purely quadratic
        "var_lb": np.zeros(n, dtype=np.float64),
        "var_ub": var_ub,
    }


def _solve_qp_cuopt_matrix(cov, mu, target_return, return_maximize, max_weight,
                           support=None, extra_linears=None):
    """Matrix-API twin of ``_solve_qp_cuopt`` — same contract, built via cuOpt's low-level
    ``data_model.DataModel`` (CSR objective + constraints) and solved with ``Solve``, instead of
    the term-by-term ``Problem`` API. The build is O(nnz) — one vectorised CSR conversion — so a
    dense covariance no longer pays the O(n²) Python-expression cost; the GPU solve is identical
    to the verified path. Returns ``(weights as fractions summing to 1, optimal_flag)``.

    Marshaling (``_qp_to_csr``) is pure and CPU-tested; only the DataModel/Solve calls below
    need a GPU, so this loads without one (like every solve in this module).
    """
    from cuopt.linear_programming import DataModel, Solve

    n = len(mu)
    ub = float(max_weight) if max_weight is not None else 1.0
    a = _qp_to_csr(cov, mu, target_return, return_maximize, max_weight, support, extra_linears)

    dm = DataModel()
    dm.set_csr_constraint_matrix(a["A_data"], a["A_indices"], a["A_offsets"])
    dm.set_constraint_bounds(a["b"])
    dm.set_row_types(a["row_types"])
    dm.set_objective_coefficients(a["c"])
    dm.set_quadratic_objective_matrix(a["Q_data"], a["Q_indices"], a["Q_offsets"])
    dm.set_variable_lower_bounds(a["var_lb"])
    dm.set_variable_upper_bounds(a["var_ub"])
    sol = Solve(dm)

    # Same accept-both gate as the term-by-term path: PDLP (the QP beta's first-order solver)
    # frequently terminates PrimalFeasible rather than certified-Optimal on these convex QPs.
    # get_termination_status() is the same enum Problem.Status exposes (Problem.solve sets it
    # from exactly this call), so the names match across both paths.
    ok = getattr(sol.get_termination_status(), "name", "") in ("Optimal", "PrimalFeasible")
    weights = np.asarray(sol.get_primal_solution(), dtype=float) if ok else np.zeros(n)
    if ok and not _qp_weights_ok(weights, ub):
        weights, ok = np.zeros(n), False
    del dm, sol
    gc.collect()
    return weights, ok


# ─── Sensitivity (duals) — HiGHS-parity dual extraction ───
# The cuOpt twin of HiGHS ``_extract_qp_sensitivity`` / ``_solve_qp_highs_sensitivity``: read the
# solver-exact duals off a solved QP (or LP) into the SAME role-tagged payload via the shared engine
# marshaller ``_build_raw_sensitivity`` — so HiGHS/cuOpt × QP/LP are byte-identical and
# ``_build_solution_sensitivity`` consumes them all the same way. The QP has two read paths mirroring
# the two plain solves: the default matrix/``DataModel`` path (``get_dual_solution`` /
# ``get_reduced_cost`` off the ``Solve`` result) and the term-by-term ``Problem`` path (``DualValue`` /
# ``ReducedCost``); the LP path reads the matrix ``Solve`` result the same way. Constraint-row order is
# the dual reader's contract, shared with ``_qp_to_csr`` / ``_lp_to_csr``: budget Σw=1 at row 0, then
# (QP only) the return floor, then one row per extra linear floor — so a row dual maps back to the
# lever it prices. ``ranging`` is always None: cuOpt has no ranging API (tracked in NVIDIA/cuopt#1395).
#
# QP and LP duals are supported in cuOpt (LP under its native LP problem-category). The marshalling is
# pure (attribute/array reads), so it is CPU-tested here with stand-ins; the dual *numbers* are
# confirmed on a GPU run by cross-checking the HiGHS oracle (same problem → matching shadow prices +
# reduced costs), the standard parity check.


def _extract_cuopt_qp_sensitivity_matrix(sol, n, has_return, n_extra, support):
    """Matrix/``DataModel`` dual read (the **default** QP path): ``get_dual_solution()`` returns one
    row dual per constraint in ``_qp_to_csr`` order, ``get_reduced_cost()`` one per variable. A cuOpt
    QP is solved under the LP problem-category (there is no separate QP category), so both accessors
    are populated — they only raise for MILP. Tolerant of a None/empty return so a degenerate solve
    degrades to a None payload upstream rather than crashing the frontier marshalling."""
    dual = sol.get_dual_solution()
    rc = sol.get_reduced_cost()
    row_dual = [float(x) for x in dual] if dual is not None else []
    col_dual = [float(x) for x in rc] if rc is not None else []
    return _build_raw_sensitivity(row_dual, col_dual, n, has_return, n_extra, support)


def _solve_qp_cuopt_matrix_sensitivity(cov, mu, target_return, return_maximize, max_weight,
                                       support=None, extra_linears=None):
    """Dual-returning twin of ``_solve_qp_cuopt_matrix`` (the default QP path): returns
    ``(weights, ok, raw_sensitivity)``. Same CSR build + ``Solve`` as the plain matrix path, then reads
    the duals off the ``Solve`` result; ``raw_sensitivity`` is None when the solve is rejected. Called
    only on the final-frontier re-solve, never in the EA hot loop, so it adds at most one dual read per
    surviving point — no extra solves. The cuOpt twin of HiGHS ``_solve_qp_highs_sensitivity``."""
    from cuopt.linear_programming import DataModel, Solve

    n = len(mu)
    ub = float(max_weight) if max_weight is not None else 1.0
    a = _qp_to_csr(cov, mu, target_return, return_maximize, max_weight, support, extra_linears)

    dm = DataModel()
    dm.set_csr_constraint_matrix(a["A_data"], a["A_indices"], a["A_offsets"])
    dm.set_constraint_bounds(a["b"])
    dm.set_row_types(a["row_types"])
    dm.set_objective_coefficients(a["c"])
    dm.set_quadratic_objective_matrix(a["Q_data"], a["Q_indices"], a["Q_offsets"])
    dm.set_variable_lower_bounds(a["var_lb"])
    dm.set_variable_upper_bounds(a["var_ub"])
    sol = Solve(dm)

    ok = getattr(sol.get_termination_status(), "name", "") in ("Optimal", "PrimalFeasible")
    weights = np.asarray(sol.get_primal_solution(), dtype=float) if ok else np.zeros(n)
    if ok and not _qp_weights_ok(weights, ub):
        weights, ok = np.zeros(n), False
    raw = (_extract_cuopt_qp_sensitivity_matrix(sol, n, target_return is not None,
                                                len(extra_linears or []), support)
           if ok else None)
    del dm, sol
    gc.collect()
    return weights, ok, raw


def _extract_cuopt_qp_sensitivity(prob, w, has_return, n_extra, support):
    """Term-by-term ``Problem``-API dual read (used when ``_USE_MATRIX_QP`` is off): constraints come
    back from ``prob.getConstraints()`` in add order (budget, return floor, extra floors), so
    ``DualValue`` maps to the lever it prices; ``w[i].ReducedCost`` gives the per-asset reduced costs.
    Feeds the same shared marshaller as the matrix path, so the payload is identical."""
    cons = list(prob.getConstraints())
    row_dual = [float(getattr(c, "DualValue", 0.0)) for c in cons]
    col_dual = [float(getattr(w[i], "ReducedCost", 0.0)) for i in range(len(w))]
    return _build_raw_sensitivity(row_dual, col_dual, len(w), has_return, n_extra, support)


def _solve_qp_cuopt_sensitivity(cov, mu, target_return, return_maximize, max_weight,
                                support=None, extra_linears=None):
    """Dual-returning twin of ``_solve_qp_cuopt`` (term-by-term path): ``(weights, ok,
    raw_sensitivity)``. Mirrors ``_solve_qp_cuopt`` line-for-line, then reads the modeling-API duals;
    kept separate so the GPU-verified plain path stays untouched."""
    from cuopt.linear_programming.problem import MINIMIZE, Problem as CuProblem

    n = len(mu)
    prob = CuProblem("frontier_portfolio_qp")
    ub = float(max_weight) if max_weight is not None else 1.0
    if support is None:
        ubs = [ub] * n
    else:
        supp = {int(i) for i in support}
        ubs = [ub if i in supp else 0.0 for i in range(n)]
    w = [prob.addVariable(lb=0.0, ub=ubs[i], name=f"w_{i}") for i in range(n)]

    quad = None
    for i in range(n):
        for j in range(n):
            c = float(cov[i, j])
            if abs(c) > 1e-12:
                term = c * w[i] * w[j]
                quad = term if quad is None else quad + term
    prob.setObjective(quad, sense=MINIMIZE)

    prob.addConstraint(sum(w) == 1, name="fully_invested")               # row 0: budget
    has_return = target_return is not None
    if has_return:                                                       # row 1: return floor
        ret_expr = sum(float(mu[i]) * w[i] for i in range(n))
        prob.addConstraint(ret_expr >= float(target_return) if return_maximize
                           else ret_expr <= float(target_return), name="return_target")
    n_extra = len(extra_linears or [])
    for k, (coef, tgt, maximize) in enumerate(extra_linears or []):      # rows 2..: extra floors
        expr = sum(float(coef[i]) * w[i] for i in range(n))
        prob.addConstraint(expr >= float(tgt) if maximize else expr <= float(tgt), name=f"linear_{k}")

    prob.solve()
    ok = getattr(prob.Status, "name", "") in ("Optimal", "PrimalFeasible")
    weights = np.array([w[i].Value for i in range(n)], dtype=float) if ok else np.zeros(n)
    if ok and not _qp_weights_ok(weights, ub):
        weights, ok = np.zeros(n), False
    raw = _extract_cuopt_qp_sensitivity(prob, w, has_return, n_extra, support) if ok else None
    del prob
    gc.collect()
    return weights, ok, raw


# ─── LP inner solves (proportional + purely linear objectives) ───
# The pure-linear twin of the matrix QP path: no quadratic Q, the objective is the linear primary
# coefficient vector, the rest of the objectives are epsilon-constraint rows. LP is cuOpt's native
# problem-category (dual simplex / PDLP), so duals + reduced costs are first-class here.


def _lp_to_csr(primary_coef, primary_maximize, eps_list, max_weight, support=None):
    """Marshal one proportional LP into the CSR arrays cuOpt's ``DataModel`` consumes — the linear
    twin of ``_qp_to_csr`` (no quadratic Q; the objective is the linear ``c``). Pure (no cuOpt
    import), so it unit-tests on CPU. Objective ``c = ±primary_coef`` (negated to maximize, since
    cuOpt minimizes). Rows, in order: budget Σw=1 ('E'), then one row per ``eps_list`` floor ('G' for
    a maximize objective, else 'L'). Box 0 ≤ w ≤ ``max_weight``, off-``support`` pinned to ub=0."""
    from scipy.sparse import csr_matrix

    n = len(primary_coef)
    ub = float(max_weight) if max_weight is not None else 1.0
    var_ub = np.full(n, ub, dtype=np.float64)
    if support is not None:
        supp = {int(i) for i in support}
        var_ub[[i for i in range(n) if i not in supp]] = 0.0

    rows = [np.ones(n, dtype=np.float64)]
    b = [1.0]
    row_types = ["E"]
    for coef, tgt, maximize in eps_list:
        rows.append(np.asarray(coef, dtype=np.float64))
        b.append(float(tgt))
        row_types.append("G" if maximize else "L")
    A = csr_matrix(np.vstack(rows))
    sign = -1.0 if primary_maximize else 1.0                             # cuOpt minimizes

    return {
        "A_data": A.data.astype(np.float64),
        "A_indices": A.indices.astype(np.int32),
        "A_offsets": A.indptr.astype(np.int32),
        "b": np.asarray(b, dtype=np.float64),
        "row_types": np.array(row_types, dtype="S1"),
        "c": (sign * np.asarray(primary_coef, dtype=np.float64)).astype(np.float64),
        "var_lb": np.zeros(n, dtype=np.float64),
        "var_ub": var_ub,
    }


def _solve_lp_cuopt(primary_coef, primary_maximize, eps_list, max_weight, support=None):
    """One proportional LP via cuOpt's low-level ``DataModel`` — the linear inner-solve contract
    shared with HiGHS (optimize a linear objective over Σw=1, box, support, with each non-primary
    objective epsilon-constrained). Returns ``(weights as fractions, ok)``. Marshaling (``_lp_to_csr``)
    is pure/CPU-tested; only the ``Solve`` below needs a GPU."""
    from cuopt.linear_programming import DataModel, Solve

    n = len(primary_coef)
    ub = float(max_weight) if max_weight is not None else 1.0
    a = _lp_to_csr(primary_coef, primary_maximize, eps_list, max_weight, support)

    dm = DataModel()
    dm.set_csr_constraint_matrix(a["A_data"], a["A_indices"], a["A_offsets"])
    dm.set_constraint_bounds(a["b"])
    dm.set_row_types(a["row_types"])
    dm.set_objective_coefficients(a["c"])
    dm.set_variable_lower_bounds(a["var_lb"])
    dm.set_variable_upper_bounds(a["var_ub"])
    sol = Solve(dm)

    ok = getattr(sol.get_termination_status(), "name", "") in ("Optimal", "PrimalFeasible")
    weights = np.asarray(sol.get_primal_solution(), dtype=float) if ok else np.zeros(n)
    if ok and not _qp_weights_ok(weights, ub):
        weights, ok = np.zeros(n), False
    del dm, sol
    gc.collect()
    return weights, ok


def _extract_cuopt_lp_sensitivity_matrix(sol, n, n_extra, support):
    """LP dual read off the cuOpt ``Solve`` result: ``get_dual_solution()`` → budget + objective-floor
    shadow prices (``_lp_to_csr`` row order), ``get_reduced_cost()`` → per-variable reduced costs. LP
    is cuOpt's native problem-category, so both accessors are populated. ``has_return=False`` — the
    primary objective is the LP objective, not an epsilon-constraint, so it carries no shadow price."""
    dual = sol.get_dual_solution()
    rc = sol.get_reduced_cost()
    row_dual = [float(x) for x in dual] if dual is not None else []
    col_dual = [float(x) for x in rc] if rc is not None else []
    return _build_raw_sensitivity(row_dual, col_dual, n, has_return=False, n_extra=n_extra,
                                  support=support)


def _solve_lp_cuopt_sensitivity(primary_coef, primary_maximize, eps_list, max_weight, support=None):
    """Dual-returning twin of ``_solve_lp_cuopt``: ``(weights, ok, raw_sensitivity)``. Same CSR build +
    ``Solve``, then reads the duals off the result; None when the solve is rejected. The cuOpt LP
    counterpart of ``_solve_lp_highs_sensitivity``."""
    from cuopt.linear_programming import DataModel, Solve

    n = len(primary_coef)
    ub = float(max_weight) if max_weight is not None else 1.0
    a = _lp_to_csr(primary_coef, primary_maximize, eps_list, max_weight, support)

    dm = DataModel()
    dm.set_csr_constraint_matrix(a["A_data"], a["A_indices"], a["A_offsets"])
    dm.set_constraint_bounds(a["b"])
    dm.set_row_types(a["row_types"])
    dm.set_objective_coefficients(a["c"])
    dm.set_variable_lower_bounds(a["var_lb"])
    dm.set_variable_upper_bounds(a["var_ub"])
    sol = Solve(dm)

    ok = getattr(sol.get_termination_status(), "name", "") in ("Optimal", "PrimalFeasible")
    weights = np.asarray(sol.get_primal_solution(), dtype=float) if ok else np.zeros(n)
    if ok and not _qp_weights_ok(weights, ub):
        weights, ok = np.zeros(n), False
    raw = _extract_cuopt_lp_sensitivity_matrix(sol, n, len(eps_list), support) if ok else None
    del dm, sol
    gc.collect()
    return weights, ok, raw


def _parallel_solve(fn, arg_tuples, max_workers=None):
    """Run independent inner solves concurrently. The EA's scalarizations are independent, so a
    **thread** pool lets their solves overlap: cuOpt releases the GIL during the C++/GPU
    ``Solve`` (and HiGHS during its C++ solve), so threads — not processes — suffice and the
    GPU/CPU overlaps the work. ``max_workers`` in ``(None, 0, 1)`` runs the **sequential
    baseline** (the "CPU wins the loop" regime); ``>1`` is the **parallel throughput** test.
    Each item in ``arg_tuples`` is the positional-arg tuple for one ``fn`` call; results
    preserve input order.

    Solver-agnostic on purpose — the same harness times cuOpt (GPU overlap) and HiGHS (CPU
    cores), so the sequential-vs-parallel flip is *measured*, not assumed. This is the DIY
    ``concurrent.futures`` pattern cuOpt's deprecated ``BatchSolve`` now points users to.
    """
    if not max_workers or max_workers <= 1:
        return [fn(*args) for args in arg_tuples]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
        return list(ex.map(lambda args: fn(*args), arg_tuples))


def _solve_milp_cuopt(min_coef, eps_list, mc, n, exact=False):
    """One binary MILP via cuOpt: minimize ``min_coef·x`` over x∈{0,1}ⁿ subject to epsilon
    constraints ``(coef, op, rhs)`` (op 'ge'/'le') and the combinatorial constraints in
    ``mc``. Returns ``(0/1 selection array, ok)``. ``exact=True`` certifies (gap→0, accept
    only ``Optimal``); default bounds for speed.
    """
    from cuopt.linear_programming import SolverSettings
    from cuopt.linear_programming.problem import INTEGER, MINIMIZE, Problem as CuProblem
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_MIP_ABSOLUTE_GAP,
        CUOPT_MIP_DETERMINISM_MODE,
        CUOPT_MIP_RELATIVE_GAP,
        CUOPT_TIME_LIMIT,
    )

    prob = CuProblem("frontier_milp")
    x = [prob.addVariable(lb=0.0, ub=1.0, vtype=INTEGER, name=f"x{i}") for i in range(n)]
    prob.setObjective(sum(float(min_coef[i]) * x[i] for i in range(n)), sense=MINIMIZE)
    for coef, op, rhs in eps_list:
        expr = sum(float(coef[i]) * x[i] for i in range(n))
        prob.addConstraint((expr >= float(rhs)) if op == "ge" else (expr <= float(rhs)), name="eps")
    if mc["card"] is not None:
        lo, hi = mc["card"]
        prob.addConstraint(sum(x) >= lo, name="card_lo")
        prob.addConstraint(sum(x) <= hi, name="card_hi")
    for coef, op, val in mc["bounds"]:
        expr = sum(float(coef[i]) * x[i] for i in range(n))
        prob.addConstraint((expr <= val) if op == "max" else (expr >= val), name="bound")
    for i in mc["force_in"]:
        prob.addConstraint(x[i] >= 1, name="fi")
    for i in mc["force_out"]:
        prob.addConstraint(x[i] <= 0, name="fo")
    for a, b in mc["deps"]:
        prob.addConstraint(x[a] - x[b] <= 0, name="dep")   # if a then b
    for a, b in mc["excl"]:
        prob.addConstraint(x[a] + x[b] <= 1, name="excl")
    for grp, gmax in mc["groups"]:
        prob.addConstraint(sum(x[i] for i in grp) <= gmax, name="grp")
    # Bounded (default, trades the optimality proof for speed) vs exact (gap→0, accept only
    # Optimal — _MILP_TIME_LIMIT still applies as a safety deadline). _MILP_ABS_GAP (default
    # off) sets an absolute gap on the bounded path; below the score granularity it certifies.
    settings = SolverSettings()
    if _MILP_TIME_LIMIT is not None:
        settings.set_parameter(CUOPT_TIME_LIMIT, _MILP_TIME_LIMIT)
    settings.set_parameter(CUOPT_MIP_RELATIVE_GAP, 0.0 if exact else _MILP_REL_GAP)
    if not exact and _MILP_ABS_GAP > 0:
        settings.set_parameter(CUOPT_MIP_ABSOLUTE_GAP, _MILP_ABS_GAP)
    # Deterministic mode (CUOPT_MODE_DETERMINISTIC = 1; cuOpt's default is opportunistic = 0).
    # Two reasons, one config: (1) reproducibility — an opportunistic (non-deterministic) solver
    # can return a different optimum run-to-run, so the cuOpt frontier wouldn't reproduce the way
    # the NSGA/seed paths do; (2) it makes cuOpt run the LP root relaxation *sequentially* instead
    # of concurrently. The EA's epsilon sweep routinely proposes infeasible corners (expected —
    # the engine scores them dominated); cuOpt's opportunistic *concurrent* root solve aborts the
    # process (std::terminate, an uncatchable kernel crash) on an infeasible relaxation, while the
    # deterministic sequential path returns Infeasible as a status the engine reads as ok=False,
    # exactly like HiGHS. So this is a configuration fix, not a workaround. (MIP-only knob; the
    # continuous QP/PDLP path returns an infeasibility status rather than aborting.)
    settings.set_parameter(CUOPT_MIP_DETERMINISM_MODE, 1)
    prob.solve(settings)
    # MILP uses ``MILPTerminationStatus`` (NoTermination / Optimal / FeasibleFound) — a
    # DIFFERENT enum from the QP's ``LPTerminationStatus``. Bounded accepts Optimal AND
    # FeasibleFound (a time/gap stop = a proven-feasible incumbent); exact only Optimal.
    ok_statuses = ("Optimal",) if exact else ("Optimal", "FeasibleFound")
    ok = getattr(prob.Status, "name", "") in ok_statuses
    sel = np.array([round(x[i].Value) for i in range(n)], dtype=float) if ok else np.zeros(n)
    # Release the cuOpt problem's device/host buffers before the next inner solve.
    del prob
    gc.collect()
    return sel, ok


def _optimize_cuopt(
    problem: Problem,
    mode: OptimizeMode,
    max_solutions: int | None = None,
    seed: int = 42,
    exact: bool = False,
    time_limit: float | None = None,
) -> Run:
    """Delegate to the shared NSGA-scalarization engine with the cuOpt inner solves: binary → exact
    MILP, proportional + quadratic → exact convex QP, proportional + purely linear → exact LP, each
    per scalarization. Returns a ``Run`` in the engine's exact shape (identical to the NSGA paths).
    ``exact`` certifies each MILP solve; ``time_limit`` (s) bounds the scalarization sweep."""
    if problem.approach == Approach.binary:
        run = optimize_milp(problem, mode, inner_milp=_solve_milp_cuopt,
                            max_solutions=max_solutions, seed=seed, exact=exact,
                            time_limit=time_limit)
    elif any(o.aggregation == Aggregation.quadratic for o in problem.objectives):
        # Mean-variance QP. Scalable matrix build when enabled, else the GPU-verified term-by-term
        # path; each has a dual-returning twin, so solver-exact sensitivity (shadow prices + reduced
        # costs) rides the final-frontier marshaling re-solve — HiGHS parity, on by default.
        if _USE_MATRIX_QP:
            inner_qp, inner_qp_sensitivity = _solve_qp_cuopt_matrix, _solve_qp_cuopt_matrix_sensitivity
        else:
            inner_qp, inner_qp_sensitivity = _solve_qp_cuopt, _solve_qp_cuopt_sensitivity
        run = optimize_qp(problem, mode, inner_qp=inner_qp,
                          inner_qp_sensitivity=inner_qp_sensitivity,
                          pop=_SPIKE_POP, gen=_SPIKE_GEN,
                          max_solutions=max_solutions, seed=seed, time_limit=time_limit)
    else:
        # Purely linear proportional allocation → exact LP (dual simplex / PDLP), with shadow prices +
        # reduced costs from the same shared marshaller.
        run = optimize_lp(problem, mode, inner_lp=_solve_lp_cuopt,
                          inner_lp_sensitivity=_solve_lp_cuopt_sensitivity,
                          pop=_SPIKE_POP, gen=_SPIKE_GEN,
                          max_solutions=max_solutions, seed=seed, time_limit=time_limit)
    # Provenance lives with the producer: stamp here so a direct call is labelled correctly,
    # not only when routed through optimize(). exact is a no-op on the always-exact QP path.
    run.solver, run.exact = "cuopt", exact
    return run


def _certify_curated_cuopt(problem: Problem, source_run: Run, *, exact: bool = False,
                           mode=None, max_solutions=None) -> Run:
    """Progressive certify with cuOpt (GPU) inner solves: exact-solve only ``source_run``'s frontier
    points — the ~|frontier|-solve twin of a full ``_optimize_cuopt`` exact pass, on any supported
    shape (binary MILP, proportional QP/LP)."""
    if problem.approach == Approach.binary:
        run = certify_curated_frontier(problem, source_run, inner_milp=_solve_milp_cuopt,
                                       exact=exact, mode=mode, max_solutions=max_solutions)
        run.solver, run.exact = "cuopt", exact
        return run
    if any(o.aggregation == Aggregation.quadratic for o in problem.objectives):
        inner, inner_sens = ((_solve_qp_cuopt_matrix, _solve_qp_cuopt_matrix_sensitivity) if _USE_MATRIX_QP
                             else (_solve_qp_cuopt, _solve_qp_cuopt_sensitivity))
    else:
        inner, inner_sens = _solve_lp_cuopt, _solve_lp_cuopt_sensitivity
    run = certify_curated_frontier(problem, source_run, inner=inner, inner_sensitivity=inner_sens,
                                   mode=mode, max_solutions=max_solutions)
    run.solver, run.exact = "cuopt", False
    return run
