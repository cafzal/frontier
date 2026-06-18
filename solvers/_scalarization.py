"""Shared NSGA-driven epsilon-constraint scalarization engine for Frontier's exact backends.

A multi-objective problem is decomposed by an EA (pymoo NSGA-II) that evolves
epsilon-constraint scalarization targets — and, under a cardinality/group cap, an N-length
asset *selection-priority* vector — while an injected **inner solver** solves each scalarized
single-objective problem to optimality. The Pareto frontier is assembled from those exact
inner optima. The EA complements NSGA's search: it explores the scalarization space, the
inner solver makes each evaluated point optimal rather than heuristic.

cuOpt (GPU) and HiGHS (CPU) are the two backends, and they differ **only** in the injected
inner solve — so the NSGA loop, genome decoding, seeding, and result marshaling live here,
once. Each backend supplies inner solves matching these contracts:

  * QP  ``inner_qp(cov, mu, target_return, return_maximize, max_weight, support,
                   extra_linears) -> (weights_frac, ok)``
        min wᵀ(cov)w  s.t. Σw=1, 0≤w≤max_weight (0 for assets off ``support``),
        mu·w ≥/≤ target_return, and each ``(coef, target, maximize)`` in ``extra_linears``.
  * MILP ``inner_milp(min_coef, eps_list, mc, n, exact) -> (selection_01, ok)``
        min min_coef·x  over x∈{0,1}ⁿ  s.t. each ``(coef, op, rhs)`` in ``eps_list`` (op
        'ge'/'le') and the combinatorial constraints in ``mc``.

For a convex QP the EA is a frontier *walker* — an epsilon sweep matches it — so it earns
its keep on the non-convex structure (cardinality / group caps) where the support choice is
combinatorial. The inner solver solves the *continuous* QP on each chosen support exactly,
so no MIQP is needed even when the overall shape is mixed-integer-quadratic.
"""

from __future__ import annotations

import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from engine import optimizer as _opt
from engine.models import (
    Aggregation,
    OptimizeMode,
    Problem,
    ReducedCost,
    Run,
    ShadowPrice,
    Solution,
    SolutionSensitivity,
    _content_signature,
)

# Cost assigned to scalarizations the inner solver reports infeasible, so they are dominated
# and constraint-flagged in the EA.
_INFEASIBLE_PENALTY = 1e9

# MILP EA budget, auto-scaled with the number of options ``n``. Each evaluation is an exact
# integer solve (~seconds on a GPU, ~tens of ms on the CPU), so the inner-solve count
# (≈ pop × gen) is the real budget — an order of magnitude below the NSGA path's cheap-eval
# budget. A *fixed* small budget under-covers at scale: on the 120-project capital MILP,
# pop16×gen8 reaches only HV 0.64 (below NSGA's 0.68), while pop50×gen15 reaches 0.87 (above
# it). So ramp pop/gen with ``n`` — small problems (≤30 vars, which wash anyway) stay cheap;
# large combinatorial ones get the budget to actually fill the frontier. Set _MILP_POP /
# _MILP_GEN to pin either value (the notebook tuning knob). Shared by both backends.
_MILP_POP: int | None = None
_MILP_GEN: int | None = None


def _milp_budget(n: int) -> tuple[int, int]:
    """(pop, gen) for the MILP EA, scaled with the option count ``n``. Anchors: ``n ≤ 30 →
    (16, 8)`` (the wash regime), ``n ≥ 120 → (50, 15)`` (the verified covering budget),
    linearly interpolated between and held flat past 120 so the inner-solve count can't run
    away. ``_MILP_POP`` / ``_MILP_GEN``, when set, pin either value."""
    frac = min(1.0, max(0.0, (n - 30) / 90.0))   # 0 at n≤30, 1 at n≥120
    pop = _MILP_POP if _MILP_POP is not None else int(round(16 + frac * 34))   # 16 → 50
    gen = _MILP_GEN if _MILP_GEN is not None else int(round(8 + frac * 7))     # 8 → 15
    return pop, gen


# --------------------------------------------------------------------------- #
# Problem readers (pure)
# --------------------------------------------------------------------------- #
def _resolve_objective_roles(problem: Problem) -> tuple[int, int]:
    """Identify the quadratic 'risk' objective (variance is the QP's minimand) and the linear
    'return' objective the epsilon-constraint sweeps. Heuristic, not keyword matching: risk =
    the objective whose aggregation is quadratic; return = the first non-quadratic objective.
    Returns ``(risk_idx, return_idx)``.
    """
    risk_idx = next(j for j, o in enumerate(problem.objectives)
                    if o.aggregation == Aggregation.quadratic)
    return_idx = next(j for j, o in enumerate(problem.objectives)
                      if o.aggregation != Aggregation.quadratic)
    return risk_idx, return_idx


def _resolve_linear_objectives(problem: Problem) -> list[int]:
    """All non-quadratic objective indices, in declaration order — the linear objectives the
    epsilon-constraint walks. The first is the 'return' role; any beyond it become extra
    linear floors on the same min-variance QP."""
    return [j for j, o in enumerate(problem.objectives)
            if o.aggregation != Aggregation.quadratic]


def _cardinality_k(problem: Problem) -> int | None:
    """Cardinality cap K ('hold at most K of N'), read from the engine's own parsed
    ``cardinality_max`` so it matches the rest of the stack. None when there is no real cap
    (max == number of options), so the backend then behaves exactly as the convex path."""
    n = len(problem.options)
    k = _opt._parse_constraints(problem).get("cardinality_max", n)
    return int(k) if (k is not None and k < n) else None


def _group_limits(problem: Problem) -> list[tuple[list[int], int]]:
    """Per-group caps from ``group_limit`` constraints → ``[(option-index-list, max), …]``,
    driving group-aware support selection (e.g. ≤3 active per region)."""
    ix = {o.name: i for i, o in enumerate(problem.options)}
    return [([ix[o] for o in c.options], int(c.max))
            for c in (problem.constraints or []) if getattr(c, "type", "") == "group_limit"]


def _build_milp_data(problem: Problem):
    """Extract MILP ingredients from a binary problem: score matrix, per-objective
    minimize-convention directions, and the combinatorial constraints as index structures."""
    n = len(problem.options)
    names = [o.name for o in problem.options]
    ix = {nm: i for i, nm in enumerate(names)}
    S = _opt._build_score_matrix(problem)
    dirs = np.array([1.0 if o.direction.value == "minimize" else -1.0 for o in problem.objectives])
    ocol = {o.name: j for j, o in enumerate(problem.objectives)}
    mc = {"card": None, "bounds": [], "force_in": [], "force_out": [], "deps": [], "excl": [], "groups": []}
    for c in (problem.constraints or []):
        t = c.type
        if t == "cardinality":
            mc["card"] = (int(c.min), int(c.max))
        elif t == "objective_bound":
            mc["bounds"].append((S[:, ocol[c.objective]].copy(), c.operator, float(c.value)))
        elif t == "force_include":
            mc["force_in"].append(ix[c.option])
        elif t == "force_exclude":
            mc["force_out"].append(ix[c.option])
        elif t == "dependency":
            mc["deps"].append((ix[c.if_option], ix[c.then_option]))
        elif t == "exclusion_pair":
            mc["excl"].append((ix[c.option_a], ix[c.option_b]))
        elif t == "group_limit":
            mc["groups"].append(([ix[o] for o in c.options], int(c.max)))
    return n, names, S, dirs, mc


# --------------------------------------------------------------------------- #
# Numeric helpers (pure)
# --------------------------------------------------------------------------- #
def _round_weights_to_pct(w_frac: np.ndarray, n_options: int, max_allocation: int | None) -> np.ndarray:
    """Round continuous fractional weights → integer percentages summing to 100, matching the
    NSGA proportional path's exact representation."""
    x = np.asarray(w_frac, dtype=float) * 100.0
    x = np.where(np.isfinite(x), x, 0.0)  # NaN/inf → 0 before the int cast
    raw = np.maximum(np.round(x), 0).astype(int)
    if max_allocation is not None:
        raw = np.minimum(raw, max_allocation)
    diff = 100 - int(raw.sum())
    if diff != 0:
        cap = max_allocation or 100
        if diff > 0:
            for _ in range(abs(diff)):
                headroom = cap - raw
                headroom[raw == 0] = 0  # don't create new holdings here
                if headroom.max() <= 0:
                    break
                raw[np.argmax(headroom)] += 1
        else:
            for _ in range(abs(diff)):
                if raw.max() <= 0:
                    break
                raw[np.argmax(raw)] -= 1
    return raw


def _nearest_psd(M: np.ndarray, rel_floor: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by clipping eigenvalues up to a small
    positive floor, so a convex QP solver accepts the covariance. Covariance estimates built
    from correlations × volatilities are frequently indefinite; only the inner solve sees
    this projection — reported risk uses the ORIGINAL matrix via the engine's aggregator."""
    A = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    floor = rel_floor * max(1.0, float(eigvals.max()))
    eigvals = np.maximum(eigvals, floor)
    A_psd = (eigvecs * eigvals) @ eigvecs.T
    return 0.5 * (A_psd + A_psd.T)  # re-symmetrize away float asymmetry


def _qp_weights_ok(weights: "np.ndarray | None", ub: float, tol: float = 1e-3) -> bool:
    """Feasibility gate on a QP solver's *returned* weights — not just its status. First-order
    QP solvers can terminate 'solved' on a degenerate point whose weights are non-finite or
    violate Σw=1 / the [0, ub] box; the downstream aggregation then explodes one point and
    blows out the frontier. Reject those here so the scalarization is treated as infeasible."""
    if weights is None or not np.all(np.isfinite(weights)):
        return False
    if abs(float(weights.sum()) - 1.0) > tol:
        return False
    return bool(weights.min() >= -1e-6 and weights.max() <= ub + 1e-4)


def _nondominated(solutions: list[Solution], obj_list: list) -> list[Solution]:
    """Keep only the non-dominated front. Marshaled convex-walker points are already
    non-dominated, but integer rounding after the per-individual re-solve can make a few
    cardinality portfolios dominate one another — so re-filter in objective space (minimize
    convention) before pruning, which only down-samples and assumes a clean front."""
    if len(solutions) <= 1:
        return solutions
    F = np.array([
        [(-s.objective_values[o.name] if o.direction.value == "maximize"
          else s.objective_values[o.name]) for o in obj_list]
        for s in solutions
    ])
    keep = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return [solutions[i] for i in keep]


def _seed_cardinality_population(linear_coefs, cov, eps_bounds, k, pop, seed) -> np.ndarray:
    """Domain-informed initial population for the cardinality EA. The support search is
    combinatorial, so at a small pop/gen the EA can miss the high-return corner. Seed with
    sensible supports — lowest-volatility K, highest-return K, highest return/vol K — each
    across a span of return targets, so the inner solver reaches those corners exactly from
    generation 0 and the EA refines around them. Every seeded support is still solved exactly.

    Genome row = ``[eps per linear objective] + [priority vector]``; a support is selected by
    giving its assets the top-K priorities (≥0.7 vs ≤0.3 elsewhere)."""
    rng = np.random.default_rng(seed)
    n_assets = cov.shape[0]
    vols = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ret = np.asarray(linear_coefs[0], float)
    sharpe = ret / np.where(vols > 0, vols, 1e-9)
    supports = [np.argsort(vols)[:k], np.argsort(ret)[-k:], np.argsort(sharpe)[-k:]]
    lo0, hi0 = eps_bounds[0]
    n_frac = max(2, pop // (len(supports) * 2))
    rows: list[list[float]] = []
    for supp in supports:
        for frac in np.linspace(0.0, 1.0, n_frac):
            eps = [lo0 + frac * (hi0 - lo0)] + [b[0] for b in eps_bounds[1:]]
            pri = rng.uniform(0.0, 0.3, n_assets)
            pri[np.asarray(supp)] = rng.uniform(0.7, 1.0, len(supp))
            rows.append(eps + list(pri))
    while len(rows) < pop:                       # fill remainder with random genomes
        eps = [rng.uniform(b[0], b[1]) for b in eps_bounds]
        rows.append(eps + list(rng.uniform(0.0, 1.0, n_assets)))
    return np.array(rows[:pop], dtype=float)


# --------------------------------------------------------------------------- #
# QP genome (proportional mean-variance)
# --------------------------------------------------------------------------- #
def _solve_individual(cov, linear_coefs, linear_maximize, max_weight, eps, support, inner_qp):
    """Inner QP for one EA individual: minimize variance subject to every linear objective k
    meeting its epsilon target ``eps[k]``, restricted to ``support``. Shared by the genome's
    ``_evaluate`` and the marshaling re-solve so the two can never disagree on what an
    individual decodes to."""
    extra = [(linear_coefs[t], float(eps[t]), linear_maximize[t])
             for t in range(1, len(linear_coefs))]
    return inner_qp(cov, linear_coefs[0], float(eps[0]), linear_maximize[0],
                    max_weight, support, extra)


def _solve_individual_sensitivity(cov, linear_coefs, linear_maximize, max_weight, eps,
                                  support, inner_qp_sensitivity):
    """Dual-returning sibling of ``_solve_individual`` — same scalarization, but the inner
    solve also returns the exact duals. Returns ``(weights_frac, ok, raw_sensitivity)``."""
    extra = [(linear_coefs[t], float(eps[t]), linear_maximize[t])
             for t in range(1, len(linear_coefs))]
    return inner_qp_sensitivity(cov, linear_coefs[0], float(eps[0]), linear_maximize[0],
                                max_weight, support, extra)


def _solve_individual_lp(linear_coefs, linear_maximize, max_weight, eps, support, inner_lp):
    """Inner LP for one EA individual (proportional allocation, purely linear objectives): optimize
    the **primary** linear objective (``linear_coefs[0]``) subject to every NON-primary objective
    meeting its epsilon target ``eps[t]``, restricted to ``support``. The continuous-allocation
    analogue of ``_solve_individual`` — but one linear objective is *optimized* rather than the
    quadratic, so the genome carries one fewer target (the primary isn't epsilon-constrained)."""
    eps_list = [(linear_coefs[t + 1], float(eps[t]), linear_maximize[t + 1])
                for t in range(len(eps))]
    return inner_lp(linear_coefs[0], linear_maximize[0], eps_list, max_weight, support)


def _solve_individual_lp_sensitivity(linear_coefs, linear_maximize, max_weight, eps,
                                     support, inner_lp_sensitivity):
    """Dual-returning sibling of ``_solve_individual_lp``: ``(weights_frac, ok, raw_sensitivity)``."""
    eps_list = [(linear_coefs[t + 1], float(eps[t]), linear_maximize[t + 1])
                for t in range(len(eps))]
    return inner_lp_sensitivity(linear_coefs[0], linear_maximize[0], eps_list, max_weight, support)


def _decode_support(pri, cardinality_k, groups):
    """Decode an asset-selection priority vector → the eligible asset indices (or ``None`` when
    unconstrained). Group-aware: keep the top-``max`` priorities per group and, if a global
    cardinality cap is also set, the top-K of the remainder. Shared by the QP and LP genomes so the
    support search behaves identically on both exact paths."""
    if cardinality_k is None and not groups:
        return None
    pri = np.asarray(pri, dtype=float)
    n = len(pri)
    if groups:
        support, grouped = set(), set()
        for grp, gmax in groups:
            grouped.update(grp)
            support.update(int(i) for i in sorted(grp, key=lambda i: pri[i])[-gmax:])
        ungrouped = [i for i in range(n) if i not in grouped]
        if cardinality_k is not None:
            rem = max(0, cardinality_k - len(support))
            support.update(int(i) for i in sorted(ungrouped, key=lambda i: pri[i])[-rem:])
        else:
            support.update(ungrouped)
        return np.array(sorted(support))
    return np.argsort(pri)[-cardinality_k:]


def _build_raw_sensitivity(row_dual, col_dual, n, has_return, n_extra, support):
    """Shared dual marshaller for every exact path (HiGHS / cuOpt × QP / LP): role-tag the constraint
    duals by add-order — budget Σw=1 (row 0), then the return floor (QP only) and the linear-objective
    floors — and pair the per-variable reduced costs with an ``eligible`` flag, producing the dict
    shape ``_build_solution_sensitivity`` consumes. ``has_return`` is True on the QP path (its primary
    linear objective is epsilon-constrained as a return floor) and False on the LP path (its primary
    linear objective is *optimized*, so it carries no shadow price). An off-``support`` asset is pinned
    to ub=0 by the cardinality/group search → its reduced cost prices that cap, not a near-miss, so it
    is flagged ineligible. ``ranging`` is None (no solver exposes it on these paths)."""
    def _row(i):
        return float(row_dual[i]) if 0 <= i < len(row_dual) else 0.0

    shadow_prices = [{"role": "budget", "linear_index": None, "value": _row(0)}]
    idx = 1
    if has_return:
        shadow_prices.append({"role": "return_floor", "linear_index": 0, "value": _row(idx)})
        idx += 1
    for t in range(n_extra):
        shadow_prices.append({"role": "linear_floor", "linear_index": t + 1, "value": _row(idx)})
        idx += 1

    if support is None:
        eligible = [True] * n
    else:
        ss = {int(i) for i in support}
        eligible = [i in ss for i in range(n)]

    return {"shadow_prices": shadow_prices, "reduced_costs": [float(x) for x in col_dual],
            "eligible": eligible, "ranging": None}


def _build_solution_sensitivity(raw, opt_names, alloc_row, linear_idxs, obj_list):
    """Map a backend's raw dual payload onto option/objective names → a
    ``SolutionSensitivity``. Backend-agnostic: ``raw['shadow_prices']`` is a list of
    ``{role, linear_index, value}`` (``linear_index`` indexes ``linear_idxs``; None =
    budget), and ``raw['reduced_costs']`` is one value per option index."""
    shadow = []
    for sp in raw["shadow_prices"]:
        li = sp.get("linear_index")
        name = "budget" if li is None else obj_list[linear_idxs[li]].name
        shadow.append(ShadowPrice(name=name, role=sp["role"],
                                  shadow_price=round(float(sp["value"]), 6)))
    rcs = raw.get("reduced_costs") or []
    elig = raw.get("eligible") or [True] * len(opt_names)
    reduced = [ReducedCost(option=opt_names[i], allocation=int(alloc_row[i]),
                           reduced_cost=round(float(rcs[i]), 6),
                           eligible=bool(elig[i]) if i < len(elig) else True)
               for i in range(len(opt_names)) if i < len(rcs)]
    return SolutionSensitivity(source="solver_exact", shadow_prices=shadow,
                               reduced_costs=reduced, ranging=raw.get("ranging"))


class _QpFrontierProblem(PymooProblem):
    """Epsilon-constraint genome for proportional mean-variance problems.

    Decision variables: one epsilon target per linear objective, plus — when a cardinality
    cap K or group caps apply — an N-length real *priority* vector whose top entries select
    the eligible assets. The plain 2-objective walker is the ``len(linear_coefs)==1`` /
    ``cardinality_k is None`` / no-groups special case. The inner solver solves each QP
    exactly; objective values come back through the engine's own aggregator (apples-to-apples
    with NSGA)."""

    def __init__(self, prop_problem, cov, linear_coefs, linear_maximize, max_weight,
                 eps_bounds, cardinality_k=None, groups=None, *, inner_qp):
        self.prop = prop_problem
        self.cov = cov
        self.linear_coefs = linear_coefs
        self.linear_maximize = linear_maximize
        self.max_weight = max_weight
        self.cardinality_k = cardinality_k
        self.groups = groups or []
        self.inner_qp = inner_qp
        n_assets = cov.shape[0]
        xl = [b[0] for b in eps_bounds]
        xu = [b[1] for b in eps_bounds]
        if cardinality_k is not None or self.groups:    # + per-asset selection priorities
            xl = xl + [0.0] * n_assets
            xu = xu + [1.0] * n_assets
        super().__init__(n_var=len(xl), n_obj=prop_problem.n_obj,
                         n_ieq_constr=1, xl=np.array(xl), xu=np.array(xu))

    def _support_from_row(self, x_row: np.ndarray) -> "np.ndarray | None":
        """Decode the selection-priority tail of one genome row → eligible asset indices (None when
        unconstrained). Delegates to the shared ``_decode_support`` so QP and LP search identically."""
        return _decode_support(np.asarray(x_row[len(self.linear_coefs):], dtype=float),
                               self.cardinality_k, self.groups)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        n_pop = X.shape[0]
        n_assets = self.cov.shape[0]
        n_lin = len(self.linear_coefs)
        objectives = self.prop.objectives

        W_pct = np.zeros((n_pop, n_assets))
        feasible = np.ones(n_pop, dtype=bool)
        for k in range(n_pop):
            w_frac, ok = _solve_individual(
                self.cov, self.linear_coefs, self.linear_maximize,
                self.max_weight, X[k, :n_lin], self._support_from_row(X[k]), self.inner_qp,
            )
            if ok:
                W_pct[k] = w_frac * 100.0
            else:
                feasible[k] = False

        F = np.zeros((n_pop, len(objectives)))
        for j, obj in enumerate(objectives):
            natural = self.prop._aggregate_objective(W_pct, j)
            F[:, j] = -natural if obj.direction.value == "maximize" else natural

        F[~feasible, :] = _INFEASIBLE_PENALTY
        out["F"] = F
        out["G"] = np.where(feasible, -1.0, 1.0).reshape(-1, 1)


def optimize_qp(problem, mode, *, inner_qp, inner_qp_sensitivity=None, pop, gen,
                max_solutions=None, seed=42, time_limit=None) -> Run:
    """Proportional mean-variance solve: the EA walks an epsilon-constraint target per linear
    objective (and, under a cardinality/group cap, the asset-selection priorities) while
    ``inner_qp`` solves each inner min-variance QP exactly. Returns a ``Run`` in the engine's
    exact shape, so explorer / metrics / store need no changes.

    ``time_limit`` (s) bounds the EA scalarization sweep — its dominant pop×gen inner-solve
    cost — stopping at the generation budget or the wall-clock cap, whichever fires first
    (best-effort: the cheap eps-bound and marshaling solves around the sweep are not counted).
    A capped run returns a sparser best-so-far frontier; each returned point is still optimal
    for its scalarization."""
    n_options = len(problem.options)
    opt_names = [o.name for o in problem.options]
    obj_list = problem.objectives
    score_matrix = _opt._build_score_matrix(problem)
    cp = _opt._parse_constraints(problem)
    im = _opt._build_interaction_matrices(problem)

    risk_idx, _ = _resolve_objective_roles(problem)
    linear_idxs = _resolve_linear_objectives(problem)
    linear_coefs = [score_matrix[:, j] for j in linear_idxs]
    linear_maximize = [obj_list[j].direction.value == "maximize" for j in linear_idxs]

    cov = _nearest_psd(im[risk_idx])
    max_weight = (cp["max_allocation"] / 100.0) if cp.get("max_allocation") else None
    cardinality_k = _cardinality_k(problem)
    groups = _group_limits(problem)

    prop = _opt._ProportionalProblem(
        n_options=n_options, score_matrix=score_matrix, objectives=obj_list,
        interaction_matrices=im, **cp,
    )

    # Epsilon range per linear objective = [value at the global min-variance portfolio, best
    # single-asset value]. Lower bound from one unconstrained inner solve; upper bound max(coef).
    w_mv, ok_mv = inner_qp(cov, linear_coefs[0], None, linear_maximize[0], max_weight, None, [])
    eps_bounds: list[tuple[float, float]] = []
    for coef in linear_coefs:
        lo = float(coef @ w_mv) if ok_mv else float(coef.min())
        hi = float(coef.max())
        if hi <= lo:
            hi = lo + 1e-6  # degenerate guard (all values equal)
        eps_bounds.append((lo, hi))

    pymoo_problem = _QpFrontierProblem(
        prop, cov, linear_coefs, linear_maximize, max_weight, eps_bounds,
        cardinality_k, groups, inner_qp=inner_qp,
    )
    if cardinality_k is not None:
        seed_X = _seed_cardinality_population(linear_coefs, cov, eps_bounds, cardinality_k, pop, seed)
        algorithm = NSGA2(pop_size=pop, sampling=seed_X)
    else:
        algorithm = NSGA2(pop_size=pop)
    _t0 = time.monotonic()
    result = pymoo_minimize(pymoo_problem, algorithm, _opt._build_termination(gen, time_limit),
                            seed=seed, verbose=False)
    _elapsed = time.monotonic() - _t0

    n_lin = len(linear_coefs)
    solutions: list[Solution] = []
    seen: set[str] = set()
    if result.X is not None and len(np.atleast_2d(result.X)) > 0:
        for row in np.atleast_2d(result.X):
            support = pymoo_problem._support_from_row(row)
            if inner_qp_sensitivity is not None:
                w_frac, ok, raw_sens = _solve_individual_sensitivity(
                    cov, linear_coefs, linear_maximize, max_weight,
                    row[:n_lin], support, inner_qp_sensitivity,
                )
            else:
                w_frac, ok = _solve_individual(
                    cov, linear_coefs, linear_maximize, max_weight,
                    row[:n_lin], support, inner_qp,
                )
                raw_sens = None
            if not ok:
                continue
            raw = _round_weights_to_pct(w_frac, n_options, cp.get("max_allocation"))
            selected = [opt_names[i] for i in range(n_options) if raw[i] > 0]
            alloc_map = {opt_names[i]: int(raw[i]) for i in range(n_options)}
            sig = _content_signature(selected, alloc_map)
            if sig in seen:
                continue
            seen.add(sig)

            W_pct = raw.astype(float).reshape(1, -1)
            obj_values = {
                obj.name: round(float(prop._aggregate_objective(W_pct, j)[0]), 4)
                for j, obj in enumerate(obj_list)
            }
            sensitivity = (_build_solution_sensitivity(raw_sens, opt_names, raw, linear_idxs, obj_list)
                           if raw_sens else None)
            solutions.append(Solution(
                solution_id=len(solutions), selected_options=selected,
                objective_values=obj_values, allocations=alloc_map,
                sensitivity=sensitivity,
            ))

    solutions = _nondominated(solutions, obj_list)
    max_n = max_solutions or _opt.MAX_PARETO_SOLUTIONS
    solutions, total_found = _opt._prune_pareto(solutions, obj_list, max_n=max_n)
    solutions = _opt._sort_and_reindex(solutions, obj_list)
    return Run(solutions=solutions, total_pareto_found=total_found,
               quality=_opt._compute_quality(result, seed=seed), mode=mode, seed_used=seed,
               time_limit=time_limit, time_limited=_opt._was_time_limited(time_limit, _elapsed))


# --------------------------------------------------------------------------- #
# LP genome (proportional allocation, purely linear objectives)
# --------------------------------------------------------------------------- #
class _LpFrontierProblem(PymooProblem):
    """EA genome for proportional allocation with purely linear objectives: one epsilon target per
    NON-primary objective (plus, under a cardinality/group cap, the asset-selection priorities). The
    inner solver optimizes the primary linear objective exactly subject to those targets + budget /
    box / support; NSGA-II explores the epsilon space. The continuous-allocation twin of
    ``_MilpFrontierProblem`` — an LP inner solve instead of a 0/1 MILP — and the proportional
    aggregator scores the resulting weights apples-to-apples with the NSGA paths."""

    def __init__(self, prop, linear_coefs, linear_maximize, max_weight, eps_bounds,
                 cardinality_k=None, groups=None, *, inner_lp):
        self.prop = prop
        self.linear_coefs = linear_coefs
        self.linear_maximize = linear_maximize
        self.max_weight = max_weight
        self.cardinality_k = cardinality_k
        self.groups = groups or []
        self.inner_lp = inner_lp
        self.n_eps = len(linear_coefs) - 1            # one epsilon target per non-primary objective
        n_assets = len(linear_coefs[0])
        xl = [b[0] for b in eps_bounds]
        xu = [b[1] for b in eps_bounds]
        if cardinality_k is not None or self.groups:
            xl = xl + [0.0] * n_assets
            xu = xu + [1.0] * n_assets
        super().__init__(n_var=len(xl), n_obj=prop.n_obj, n_ieq_constr=1,
                         xl=np.array(xl), xu=np.array(xu))

    def _support_from_row(self, x_row):
        return _decode_support(np.asarray(x_row[self.n_eps:], dtype=float),
                               self.cardinality_k, self.groups)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        n_pop = X.shape[0]
        n_assets = len(self.linear_coefs[0])
        objectives = self.prop.objectives
        W_pct = np.zeros((n_pop, n_assets))
        feasible = np.ones(n_pop, dtype=bool)
        for k in range(n_pop):
            w_frac, ok = _solve_individual_lp(
                self.linear_coefs, self.linear_maximize, self.max_weight,
                X[k, :self.n_eps], self._support_from_row(X[k]), self.inner_lp,
            )
            if ok:
                W_pct[k] = w_frac * 100.0
            else:
                feasible[k] = False
        F = np.zeros((n_pop, len(objectives)))
        for j, obj in enumerate(objectives):
            natural = self.prop._aggregate_objective(W_pct, j)
            F[:, j] = -natural if obj.direction.value == "maximize" else natural
        F[~feasible, :] = _INFEASIBLE_PENALTY
        out["F"] = F
        out["G"] = np.where(feasible, -1.0, 1.0).reshape(-1, 1)


def optimize_lp(problem, mode, *, inner_lp, inner_lp_sensitivity=None, pop, gen,
                max_solutions=None, seed=42, time_limit=None) -> Run:
    """Proportional allocation with purely linear objectives — an exact multi-objective LP. The EA
    walks an epsilon-constraint target on each non-primary objective while ``inner_lp`` optimizes the
    primary linear objective exactly per individual (continuous weights, Σw=1). The pure-linear
    sibling of ``optimize_qp`` (no covariance / quadratic term); solver-exact shadow prices + reduced
    costs ride the final-frontier re-solve via ``inner_lp_sensitivity``. Returns a Run in the engine's
    exact shape, so explorer / metrics / store need no changes."""
    n_options = len(problem.options)
    opt_names = [o.name for o in problem.options]
    obj_list = problem.objectives
    score_matrix = _opt._build_score_matrix(problem)
    cp = _opt._parse_constraints(problem)

    linear_idxs = _resolve_linear_objectives(problem)   # all objectives (no quadratic on this path)
    linear_coefs = [score_matrix[:, j] for j in linear_idxs]
    linear_maximize = [obj_list[j].direction.value == "maximize" for j in linear_idxs]

    max_weight = (cp["max_allocation"] / 100.0) if cp.get("max_allocation") else None
    cardinality_k = _cardinality_k(problem)
    groups = _group_limits(problem)

    prop = _opt._ProportionalProblem(
        n_options=n_options, score_matrix=score_matrix, objectives=obj_list,
        interaction_matrices=_opt._build_interaction_matrices(problem), **cp,
    )

    # Epsilon range per NON-primary objective = [min, max] over the budget/box-feasible region, from
    # two LP anchor solves that optimize that objective alone (the primary objective ignored).
    eps_bounds: list[tuple[float, float]] = []
    for t in range(1, len(linear_coefs)):
        coef, maximize = linear_coefs[t], linear_maximize[t]
        w_hi, ok_hi = inner_lp(coef, maximize, [], max_weight, None)        # best for this objective
        w_lo, ok_lo = inner_lp(coef, not maximize, [], max_weight, None)    # worst for this objective
        vals = [float(coef @ w) for w, ok in ((w_hi, ok_hi), (w_lo, ok_lo)) if ok]
        lo, hi = (min(vals), max(vals)) if vals else (float(coef.min()), float(coef.max()))
        eps_bounds.append((lo, hi if hi > lo else lo + 1e-6))

    pymoo_problem = _LpFrontierProblem(
        prop, linear_coefs, linear_maximize, max_weight, eps_bounds,
        cardinality_k, groups, inner_lp=inner_lp,
    )
    _t0 = time.monotonic()
    result = pymoo_minimize(pymoo_problem, NSGA2(pop_size=pop),
                            _opt._build_termination(gen, time_limit), seed=seed, verbose=False)
    _elapsed = time.monotonic() - _t0

    n_eps = pymoo_problem.n_eps
    solutions: list[Solution] = []
    seen: set[str] = set()
    if result.X is not None and len(np.atleast_2d(result.X)) > 0:
        for row in np.atleast_2d(result.X):
            support = pymoo_problem._support_from_row(row)
            if inner_lp_sensitivity is not None:
                w_frac, ok, raw_sens = _solve_individual_lp_sensitivity(
                    linear_coefs, linear_maximize, max_weight, row[:n_eps], support,
                    inner_lp_sensitivity)
            else:
                w_frac, ok = _solve_individual_lp(
                    linear_coefs, linear_maximize, max_weight, row[:n_eps], support, inner_lp)
                raw_sens = None
            if not ok:
                continue
            raw = _round_weights_to_pct(w_frac, n_options, cp.get("max_allocation"))
            selected = [opt_names[i] for i in range(n_options) if raw[i] > 0]
            alloc_map = {opt_names[i]: int(raw[i]) for i in range(n_options)}
            sig = _content_signature(selected, alloc_map)
            if sig in seen:
                continue
            seen.add(sig)
            W_pct = raw.astype(float).reshape(1, -1)
            obj_values = {obj.name: round(float(prop._aggregate_objective(W_pct, j)[0]), 4)
                          for j, obj in enumerate(obj_list)}
            sensitivity = (_build_solution_sensitivity(raw_sens, opt_names, raw, linear_idxs, obj_list)
                           if raw_sens else None)
            solutions.append(Solution(
                solution_id=len(solutions), selected_options=selected,
                objective_values=obj_values, allocations=alloc_map, sensitivity=sensitivity))

    solutions = _nondominated(solutions, obj_list)
    max_n = max_solutions or _opt.MAX_PARETO_SOLUTIONS
    solutions, total_found = _opt._prune_pareto(solutions, obj_list, max_n=max_n)
    solutions = _opt._sort_and_reindex(solutions, obj_list)
    return Run(solutions=solutions, total_pareto_found=total_found,
               quality=_opt._compute_quality(result, seed=seed), mode=mode, seed_used=seed,
               time_limit=time_limit, time_limited=_opt._was_time_limited(time_limit, _elapsed))


# --------------------------------------------------------------------------- #
# Progressive certify (exact-solve only an existing run's frontier points)
# --------------------------------------------------------------------------- #
def certify_curated_frontier(problem, source_run, *, inner=None, inner_sensitivity=None,
                             inner_milp=None, exact=False, mode=None, max_solutions=None) -> Run:
    """Exact-solve **only the points of an existing run** (typically the exploratory NSGA frontier),
    re-solving each for its own scalarization (support + epsilon targets), and assemble them into an
    exact ``Run`` — non-dominated-filtered, pruned, and re-indexed exactly like a full scalarization.

    This is the lean *explore-then-certify* overlay: where ``optimize_{qp,lp,milp}`` run the inner solver
    on every one of ~pop×gen NSGA *evaluations* (the inner solve **is** NSGA's fitness), this runs it once
    per *curated frontier point* (~dozens). The heuristic explores; the exact solver certifies only what
    it kept. Covers all three shapes (inner shape-matched by the caller): binary 0/1 **MILP** via
    ``inner_milp``, proportional **QP**/**LP** via ``inner`` (+ ``inner_sensitivity`` for solver duals).
    Each returned point is optimal for its scalarization, so the overlay can only confirm or sharpen the
    source frontier — the same auditor guarantee as the full exact pass, at a fraction of the cost."""
    n_options = len(problem.options)
    opt_names = [o.name for o in problem.options]
    opt_index = {n: i for i, n in enumerate(opt_names)}
    obj_list = problem.objectives

    solutions: list[Solution] = []
    seen: set = set()

    if getattr(problem.approach, "value", problem.approach) == "binary":
        # Binary MILP: re-solve each source point's scalarization — minimize the primary objective
        # subject to the source point's non-primary values as epsilon targets + the combinatorial
        # constraints. Mirrors optimize_milp's inner solve, once per curated point instead of pop×gen.
        n, names, S, dirs, mc = _build_milp_data(problem)
        primary, nonprimary = 0, [j for j in range(len(obj_list)) if j != 0]
        for src in source_run.solutions:
            eps_list = [(S[:, j], "ge" if dirs[j] < 0 else "le",
                         float(src.objective_values.get(obj_list[j].name, 0.0))) for j in nonprimary]
            sel, ok = inner_milp(dirs[primary] * S[:, primary], eps_list, mc, n, exact)
            if not ok:
                continue
            selected = [names[i] for i in range(n) if sel[i] > 0.5]
            key = tuple(sorted(selected))
            if key in seen:
                continue
            seen.add(key)
            obj_values = {o.name: round(float(S[:, j] @ sel), 4) for j, o in enumerate(obj_list)}
            solutions.append(Solution(solution_id=len(solutions), selected_options=selected,
                                      objective_values=obj_values))
    else:
        score_matrix = _opt._build_score_matrix(problem)
        cp = _opt._parse_constraints(problem)
        im = _opt._build_interaction_matrices(problem)
        linear_idxs = _resolve_linear_objectives(problem)
        linear_coefs = [score_matrix[:, j] for j in linear_idxs]
        linear_maximize = [obj_list[j].direction.value == "maximize" for j in linear_idxs]
        max_weight = (cp["max_allocation"] / 100.0) if cp.get("max_allocation") else None
        prop = _opt._ProportionalProblem(n_options=n_options, score_matrix=score_matrix,
                                         objectives=obj_list, interaction_matrices=im, **cp)
        is_qp = bool(im)
        cov = _nearest_psd(im[_resolve_objective_roles(problem)[0]]) if is_qp else None
        for src in source_run.solutions:
            support = [opt_index[n] for n in src.selected_options if n in opt_index]
            # Each source point's reported objective values become its epsilon targets: QP epsilon-
            # constrains every linear objective (variance minimized); LP optimizes the primary and
            # epsilon-constrains the rest, so it carries one fewer target.
            idxs = linear_idxs if is_qp else linear_idxs[1:]
            eps = np.array([src.objective_values.get(obj_list[j].name, 0.0) for j in idxs], dtype=float)
            if is_qp and inner_sensitivity is not None:
                w_frac, ok, raw_sens = _solve_individual_sensitivity(
                    cov, linear_coefs, linear_maximize, max_weight, eps, support, inner_sensitivity)
            elif is_qp:
                w_frac, ok = _solve_individual(cov, linear_coefs, linear_maximize, max_weight, eps, support, inner)
                raw_sens = None
            elif inner_sensitivity is not None:
                w_frac, ok, raw_sens = _solve_individual_lp_sensitivity(
                    linear_coefs, linear_maximize, max_weight, eps, support, inner_sensitivity)
            else:
                w_frac, ok = _solve_individual_lp(linear_coefs, linear_maximize, max_weight, eps, support, inner)
                raw_sens = None
            if not ok:
                continue
            raw = _round_weights_to_pct(w_frac, n_options, cp.get("max_allocation"))
            selected = [opt_names[i] for i in range(n_options) if raw[i] > 0]
            alloc_map = {opt_names[i]: int(raw[i]) for i in range(n_options)}
            sig = _content_signature(selected, alloc_map)
            if sig in seen:
                continue
            seen.add(sig)
            W_pct = raw.astype(float).reshape(1, -1)
            obj_values = {obj.name: round(float(prop._aggregate_objective(W_pct, j)[0]), 4)
                          for j, obj in enumerate(obj_list)}
            sensitivity = (_build_solution_sensitivity(raw_sens, opt_names, raw, linear_idxs, obj_list)
                           if raw_sens else None)
            solutions.append(Solution(solution_id=len(solutions), selected_options=selected,
                                      objective_values=obj_values, allocations=alloc_map, sensitivity=sensitivity))

    solutions = _nondominated(solutions, obj_list)
    max_n = max_solutions or _opt.MAX_PARETO_SOLUTIONS
    solutions, total_found = _opt._prune_pareto(solutions, obj_list, max_n=max_n)
    solutions = _opt._sort_and_reindex(solutions, obj_list)
    return Run(solutions=solutions, total_pareto_found=total_found,
               quality=source_run.quality, mode=mode or source_run.mode, seed_used=source_run.seed_used)


# --------------------------------------------------------------------------- #
# MILP genome (binary selection)
# --------------------------------------------------------------------------- #
class _MilpFrontierProblem(PymooProblem):
    """EA genome for binary selection: one epsilon target per non-primary objective. The
    inner solver minimizes the primary objective exactly subject to those targets + the
    combinatorial constraints; the EA (NSGA-II) explores the epsilon space."""

    def __init__(self, S, dirs, primary, nonprimary, mc, n, objectives, eps_bounds,
                 exact=False, *, inner_milp):
        self.S, self.dirs, self.primary, self.nonprimary = S, dirs, primary, nonprimary
        self.mc, self.n, self.objectives, self.exact = mc, n, objectives, exact
        self.inner_milp = inner_milp
        super().__init__(n_var=len(nonprimary), n_obj=len(objectives), n_ieq_constr=1,
                         xl=np.array([b[0] for b in eps_bounds]),
                         xu=np.array([b[1] for b in eps_bounds]))

    def _solve_row(self, row):
        eps_list = [(self.S[:, j], "ge" if self.dirs[j] < 0 else "le", float(row[k]))
                    for k, j in enumerate(self.nonprimary)]
        return self.inner_milp(self.dirs[self.primary] * self.S[:, self.primary], eps_list,
                               self.mc, self.n, self.exact)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(X)
        F = np.zeros((X.shape[0], len(self.objectives)))
        feas = np.ones(X.shape[0], dtype=bool)
        for r in range(X.shape[0]):
            sel, ok = self._solve_row(X[r])
            if ok:
                for j, o in enumerate(self.objectives):
                    v = float(self.S[:, j] @ sel)
                    F[r, j] = -v if o.direction.value == "maximize" else v
            else:
                feas[r] = False
        F[~feas, :] = _INFEASIBLE_PENALTY
        out["F"] = F
        out["G"] = np.where(feas, -1.0, 1.0).reshape(-1, 1)


def audit_milp(problem, negated_disjuncts, *, inner_audit):
    """Backend-agnostic witness / feasibility audit over a binary problem's feasible region.

    The feasible region is the problem's own hard constraints (``_build_milp_data``'s ``mc``).
    Each entry of ``negated_disjuncts`` is a set of extra linear rows ``(coef, op, rhs)`` encoding
    ONE way to violate the audited property — a *disjunction*, because some properties negate to an
    OR (cardinality → count≤min-1 OR count≥max+1). The property holds iff EVERY disjunct is
    infeasible against the feasible region; the first feasible disjunct is a counterexample witness.

    ``inner_audit(eps_list, mc, n) -> (status, sel)`` is the injected feasibility solve (HiGHS
    today; a cuOpt sibling can drop in later — same split as ``optimize_milp``'s ``inner_milp``).
    Returns ``{feasible, witness_options, statuses}``; the engine maps that — with the property /
    probe context — to a verdict."""
    n, names, _, _, mc = _build_milp_data(problem)   # score matrix / directions unused for feasibility
    statuses: list[str] = []
    for rows in negated_disjuncts:
        status, sel = inner_audit(rows, mc, n)
        statuses.append(status)
        if status == "Optimal":   # a feasible witness violating the property
            witness = [names[i] for i in range(n) if sel[i] > 0.5]
            return {"feasible": True, "witness_options": witness, "statuses": statuses}
    return {"feasible": False, "witness_options": None, "statuses": statuses}


def optimize_milp(problem, mode, *, inner_milp, max_solutions=None,
                  seed=42, exact=False, time_limit=None) -> Run:
    """Binary selection solve: the EA evolves epsilon targets on the non-primary objectives
    while ``inner_milp`` solves the scalarized 0/1 MILP (minimizing objective 0) exactly per
    individual. The EA budget auto-scales with the option count (``_milp_budget``). Same
    Run/Solution shape as the NSGA binary path. ``exact`` certifies each inner solve.

    ``time_limit`` (s) bounds the EA sweep (generation budget or wall clock, whichever first);
    a capped run returns a sparser best-so-far frontier, each point still exact for its
    scalarization. Best-effort — the eps-bound and marshaling solves around the sweep are not
    counted."""
    n, names, S, dirs, mc = _build_milp_data(problem)
    pop, gen = _milp_budget(n)
    objs = problem.objectives
    primary = 0
    nonprimary = [j for j in range(len(objs)) if j != primary]

    # epsilon range per non-primary objective = [min, max] over the feasible set.
    eps_bounds = []
    for j in nonprimary:
        smin, ok1 = inner_milp(S[:, j], [], mc, n, exact)
        smax, ok2 = inner_milp(-S[:, j], [], mc, n, exact)
        lo = float(S[:, j] @ smin) if ok1 else float(S[:, j].min())
        hi = float(S[:, j] @ smax) if ok2 else float(S[:, j].sum())
        eps_bounds.append((lo, hi if hi > lo else lo + 1e-6))

    pp = _MilpFrontierProblem(S, dirs, primary, nonprimary, mc, n, objs, eps_bounds,
                              exact=exact, inner_milp=inner_milp)
    _t0 = time.monotonic()
    result = pymoo_minimize(pp, NSGA2(pop_size=pop), _opt._build_termination(gen, time_limit),
                            seed=seed, verbose=False)
    _elapsed = time.monotonic() - _t0

    solutions: list[Solution] = []
    seen: set = set()
    if result.X is not None and len(np.atleast_2d(result.X)) > 0:
        for row in np.atleast_2d(result.X):
            sel, ok = pp._solve_row(row)
            if not ok:
                continue
            selected = [names[i] for i in range(n) if sel[i] > 0.5]
            key = tuple(sorted(selected))
            if key in seen:
                continue
            seen.add(key)
            obj_values = {o.name: round(float(S[:, j] @ sel), 4) for j, o in enumerate(objs)}
            solutions.append(Solution(solution_id=len(solutions), selected_options=selected,
                                      objective_values=obj_values))

    solutions = _nondominated(solutions, objs)
    max_n = max_solutions or _opt.MAX_PARETO_SOLUTIONS
    solutions, total = _opt._prune_pareto(solutions, objs, max_n=max_n)
    solutions = _opt._sort_and_reindex(solutions, objs)
    return Run(solutions=solutions, total_pareto_found=total,
               quality=_opt._compute_quality(result, seed=seed), mode=mode, seed_used=seed,
               time_limit=time_limit, time_limited=_opt._was_time_limited(time_limit, _elapsed))
