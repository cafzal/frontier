"""HiGHS inner-solve backend for Frontier's exact NSGA-scalarization engine — the co-equal
CPU exact backend alongside the cuOpt GPU backend.

The NSGA outer loop, genome decoding, seeding, and result marshaling live in
``solvers._scalarization`` and are shared with the cuOpt backend; this module supplies only
the **HiGHS-specific inner solves** (a convex QP and a MILP) plus a thin delegation. So HiGHS
*complements* NSGA exactly as cuOpt does — NSGA stays the outer evolutionary search, and
HiGHS makes each evaluated scalarization optimal rather than heuristic. The two backends
differ *only* in this inner solve: cuOpt on a GPU, HiGHS on the CPU.

  * **Binary (select-a-subset)** → ``highspy`` MILP per scalarization. Every Frontier
    combinatorial constraint (cardinality, force in/out, dependency, exclusion, group limit,
    objective bound) is linear-integer, so HiGHS branch-and-bound handles them exactly.
  * **Proportional + quadratic (mean-variance)** → ``highspy`` convex QP per scalarization,
    minimizing wᵀΣw over the support the EA selects. HiGHS can't solve MIQP directly, but it
    never has to: the EA picks *which* assets are eligible (cardinality / group caps), and
    HiGHS solves the *continuous* convex QP on that support exactly.

Additive + reversible, like the cuOpt backend: the engine routes here only when
``optimize(solver="highs")`` is requested for a shape ``solvers.exact_solver_fits`` accepts;
``highspy`` is imported lazily so the module loads whether or not it is installed; and the
Run/Solution shape (preserved by the shared engine) means explorer / metrics / store need
zero changes. Unlike cuOpt the solver is a plain ``pip install highspy`` (CPU, cross-platform,
no special index, no GPU) — so this exact path runs on the same machine as the engine, which
also makes the shared NSGA engine testable locally for the first time.
"""

from __future__ import annotations

import numpy as np

from engine.models import Approach, OptimizeMode, Problem, Run
from solvers._scalarization import _qp_weights_ok, optimize_milp, optimize_qp

# Per-scalarization MILP controls. The default ``mip_rel_gap`` is far below unit score
# granularity, so the incumbent is the true optimum while skipping the (irrelevant) proof;
# ``exact=True`` drops the gap to 0. ``time_limit`` is only a safety deadline.
_MILP_REL_GAP = 1e-4
_MILP_TIME_LIMIT = 30.0

# QP EA budget (pop, gen) by mode. The cheap continuous QP can afford a dense search; the
# returned frontier has at most ``pop`` points, so the population drives coverage. (The MILP
# path's budget auto-scales with problem size in _scalarization._milp_budget, shared with cuOpt.)
_QP_BUDGET = {"fast": (60, 20), "thorough": (100, 30)}


def _hessian_from_cov(cov: np.ndarray):
    """Pack 2·cov as a HiGHS Hessian in CSC lower-triangular form. HiGHS minimizes
    ``c·x + ½ xᵀQx``, so Q = 2·cov makes the quadratic term exactly wᵀ(cov)w — the portfolio
    variance whose √ is Frontier's risk objective."""
    import highspy

    n = cov.shape[0]
    Q = 2.0 * cov
    start, index, value = [0], [], []
    for j in range(n):
        for i in range(j, n):  # lower triangle, column-major
            if abs(Q[i, j]) > 1e-15:
                index.append(i)
                value.append(float(Q[i, j]))
        start.append(len(index))
    H = highspy.HighsHessian()
    H.dim_ = n
    H.format_ = highspy.HessianFormat.kTriangular
    H.start_ = np.array(start)
    H.index_ = np.array(index)
    H.value_ = np.array(value, dtype=np.double)
    return H


def _build_qp_model(cov, mu, target_return, return_maximize, max_weight,
                    support=None, extra_linears=None):
    """Build (but don't solve) the min-variance QP shared by the plain and dual-returning
    inner solves. Returns ``(h, w, has_return)``. **Constraint (row) order is the dual
    reader's contract**: budget Σw=1 at idx 0, the return floor at idx 1 (iff
    ``target_return`` is set), then one row per ``extra_linears`` entry — so a row dual can
    be mapped back to the lever it prices."""
    import highspy

    n = len(mu)
    ub = float(max_weight) if max_weight is not None else 1.0
    if support is None:
        ubs = [ub] * n
    else:
        supp = {int(i) for i in support}
        ubs = [ub if i in supp else 0.0 for i in range(n)]

    h = highspy.Highs()
    h.silent()
    w = [h.addVariable(lb=0.0, ub=ubs[i]) for i in range(n)]
    h.addConstr(sum(w) == 1)                                          # row 0: budget
    has_return = target_return is not None
    if has_return:
        ret = sum(float(mu[i]) * w[i] for i in range(n))
        h.addConstr(ret >= float(target_return) if return_maximize   # row 1: return floor
                    else ret <= float(target_return))
    for coef, tgt, maximize in (extra_linears or []):                # rows 2..: extra floors
        expr = sum(float(coef[i]) * w[i] for i in range(n))
        h.addConstr(expr >= float(tgt) if maximize else expr <= float(tgt))
    h.passHessian(_hessian_from_cov(cov))
    return h, w, has_return


def _qp_primal(h, w, max_weight):
    """Read primal weights + feasibility from a solved QP. Gates on the returned weights,
    not just status (parity with the cuOpt inner solve): a degenerate 'solved' point whose
    weights are non-finite or violate Σw=1 / the box is treated as infeasible."""
    n = len(w)
    ub = float(max_weight) if max_weight is not None else 1.0
    ok = h.modelStatusToString(h.getModelStatus()) == "Optimal"
    weights = np.array(h.vals(w), dtype=float) if ok else np.zeros(n)
    if ok and not _qp_weights_ok(weights, ub):
        weights, ok = np.zeros(n), False
    return weights, ok


def _solve_qp_highs(cov, mu, target_return, return_maximize, max_weight,
                    support=None, extra_linears=None):
    """One min-variance QP via HiGHS — the convex inner solve contract shared with cuOpt::

        minimize   wᵀ(cov)w
        subject to Σw = 1,  0 ≤ w ≤ max_weight  (0 for assets off ``support``)
                   mu·w ≥/≤ target_return,  and each (coef, target, maximize) in extra_linears

    ``support`` (the EA's chosen eligible assets) is enforced by pinning excluded assets'
    upper bound to 0, so HiGHS solves the exact continuous QP on that subset — no MIQP.
    Returns ``(weights as fractions, ok)``. Convex and exact.
    """
    h, w, _ = _build_qp_model(cov, mu, target_return, return_maximize, max_weight,
                              support, extra_linears)
    h.minimize()
    return _qp_primal(h, w, max_weight)


def _extract_qp_sensitivity(h, w, has_return, n_extra, support):
    """Read HiGHS duals into a role-tagged raw payload the engine maps to option names.

    Row order matches ``_build_qp_model`` (budget, return floor, extra floors), so
    ``row_dual`` becomes role-tagged shadow prices and ``col_dual`` the per-variable reduced
    costs. ``eligible`` flags which variables the EA left *in the support*: an off-support
    asset is pinned to ub=0 by the cardinality/group search, so its reduced cost is about that
    cap — not a near-miss — and the engine filters it out of the near-miss list. ``ranging`` is
    left None: HiGHS ``getRanging()`` is classically LP-only and its QP output is unverified (L4).
    """
    sol = h.getSolution()
    row_dual = [float(x) for x in sol.row_dual]
    col_dual = [float(x) for x in sol.col_dual]
    n = len(w)
    if support is None:
        eligible = [True] * n
    else:
        ss = {int(i) for i in support}
        eligible = [i in ss for i in range(n)]

    def _row(i):  # defensive: HiGHS returns one dual per constraint, in add order
        return row_dual[i] if 0 <= i < len(row_dual) else 0.0

    shadow_prices = [{"role": "budget", "linear_index": None, "value": _row(0)}]
    idx = 1
    if has_return:
        shadow_prices.append({"role": "return_floor", "linear_index": 0, "value": _row(idx)})
        idx += 1
    for t in range(n_extra):
        shadow_prices.append({"role": "linear_floor", "linear_index": t + 1, "value": _row(idx)})
        idx += 1

    return {"shadow_prices": shadow_prices, "reduced_costs": col_dual,
            "eligible": eligible, "ranging": None}


def _solve_qp_highs_sensitivity(cov, mu, target_return, return_maximize, max_weight,
                                support=None, extra_linears=None):
    """Same convex inner solve as ``_solve_qp_highs``, plus the exact duals:
    ``(weights, ok, raw_sensitivity)``. ``raw_sensitivity`` carries role-tagged constraint
    shadow prices (row duals) and per-variable reduced costs (col duals); None when the
    solve is rejected. Called only on the final-frontier re-solve, never in the EA hot loop,
    so it adds at most one dual read per surviving point (no extra solves)."""
    n_extra = len(extra_linears or [])
    h, w, has_return = _build_qp_model(cov, mu, target_return, return_maximize, max_weight,
                                       support, extra_linears)
    h.minimize()
    weights, ok = _qp_primal(h, w, max_weight)
    if not ok:
        return weights, ok, None
    return weights, ok, _extract_qp_sensitivity(h, w, has_return, n_extra, support)


def _solve_milp_highs(min_coef, eps_list, mc, n, exact=False):
    """One binary MILP via HiGHS: minimize ``min_coef·x`` over x∈{0,1}ⁿ subject to the epsilon
    constraints ``(coef, op, rhs)`` (op 'ge'/'le') and the combinatorial constraints in
    ``mc``. Returns ``(0/1 selection array, ok)``. Exact; ``exact=True`` certifies with a zero
    gap, otherwise a sub-granularity gap is allowed for speed.
    """
    import highspy

    h = highspy.Highs()
    h.silent()
    h.setOptionValue("mip_rel_gap", 0.0 if exact else _MILP_REL_GAP)
    h.setOptionValue("time_limit", _MILP_TIME_LIMIT)

    x = h.addBinaries(n)
    for coef, op, rhs in eps_list:
        expr = (x * [float(c) for c in coef]).sum()
        h.addConstr(expr >= float(rhs) if op == "ge" else expr <= float(rhs))
    if mc["card"] is not None:
        lo, hi = mc["card"]
        h.addConstr(x.sum() >= lo)
        h.addConstr(x.sum() <= hi)
    for coef, op, val in mc["bounds"]:
        expr = (x * [float(c) for c in coef]).sum()
        h.addConstr(expr <= val if op == "max" else expr >= val)
    for i in mc["force_in"]:
        h.addConstr(x[i] >= 1)
    for i in mc["force_out"]:
        h.addConstr(x[i] <= 0)
    for a, b in mc["deps"]:
        h.addConstr(x[a] - x[b] <= 0)   # if a then b
    for a, b in mc["excl"]:
        h.addConstr(x[a] + x[b] <= 1)
    for grp, gmax in mc["groups"]:
        gc = np.zeros(n)
        gc[grp] = 1.0
        h.addConstr((x * list(gc)).sum() <= gmax)

    h.minimize((x * [float(c) for c in min_coef]).sum())

    status = h.modelStatusToString(h.getModelStatus())
    ok = status == "Optimal"
    if not ok and not exact and status != "Infeasible":
        # Bounded mode: a time/gap stop with a proven-feasible incumbent still counts.
        ok = h.solutionStatusToString(h.getInfo().primal_solution_status) == "Feasible"
    sel = np.array([round(v) for v in h.vals(x)], dtype=float) if ok else np.zeros(n)
    return sel, ok


def _optimize_highs(
    problem: Problem,
    mode: OptimizeMode,
    max_solutions: int | None = None,
    seed: int = 42,
    exact: bool = False,
    time_limit: float | None = None,
) -> Run:
    """Delegate to the shared NSGA-scalarization engine with the HiGHS inner solves: binary →
    exact MILP per scalarization, proportional → exact convex QP per scalarization. Returns a
    ``Run`` in the engine's exact shape (identical to the NSGA paths). ``exact`` certifies each
    MILP solve; ``time_limit`` (s) bounds the scalarization sweep (best-so-far on a cap)."""
    if problem.approach == Approach.binary:
        run = optimize_milp(problem, mode, inner_milp=_solve_milp_highs,
                            max_solutions=max_solutions, seed=seed, exact=exact,
                            time_limit=time_limit)
    else:
        m = getattr(mode, "value", "fast")
        pop, gen = _QP_BUDGET.get(m, _QP_BUDGET["fast"])
        # inner_qp_sensitivity attaches solver-exact duals (shadow prices + reduced costs) to
        # each final-frontier point — the dual read rides the marshaling re-solve, no extra cost.
        run = optimize_qp(problem, mode, inner_qp=_solve_qp_highs,
                          inner_qp_sensitivity=_solve_qp_highs_sensitivity, pop=pop, gen=gen,
                          max_solutions=max_solutions, seed=seed, time_limit=time_limit)
    # Provenance lives with the producer: stamp here so a direct call is labelled correctly,
    # not only when routed through optimize(). exact is a no-op on the always-exact QP path.
    run.solver, run.exact = "highs", exact
    return run
