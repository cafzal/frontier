"""Optimizer: validate problem and run NSGA-II/NSGA-III via pymoo."""

from __future__ import annotations

import contextlib
import hashlib
import math
import threading
import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem as PymooProblem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.pm import PM
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import BinaryRandomSampling, FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination.collection import TerminationCollection
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.max_time import TimeBasedTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions


# --- Deterministic RNG for pymoo's unseeded fallbacks ---
#
# pymoo 0.6.1.6 seeds the algorithm's own ``random_state`` and threads it into
# sampling/crossover/mutation, but NOT into the survival and selection operators —
# those fall back to ``@default_random_state``'s ``np.random.default_rng(None)``
# (OS entropy). Any run that hits a random branch there (notably NSGA-III niche
# tie-breaking, which fires every generation once the last front overflows
# ``pop_size``) is then non-reproducible even at a fixed seed.
#
# We can't change pymoo from here, so we install one process-wide shim over
# ``np.random.default_rng``: a call with an explicit seed is untouched (pymoo's
# own ``default_rng(self.seed)`` still works), but a ``None``/no-arg call inside a
# seeded solve draws a deterministic child seed from a thread-local
# ``SeedSequence``. The state is thread-local and only set inside
# ``_seeded_rng_fallback``, so it is transparent outside our solve and safe under
# the scenario ThreadPoolExecutor (each worker carries its own SeedSequence).
_ORIG_DEFAULT_RNG = np.random.default_rng
_rng_tls = threading.local()


def _patched_default_rng(seed=None, *args, **kwargs):
    if seed is None:
        ss = getattr(_rng_tls, "seed_sequence", None)
        if ss is not None:
            return _ORIG_DEFAULT_RNG(ss.spawn(1)[0])
    return _ORIG_DEFAULT_RNG(seed, *args, **kwargs)


np.random.default_rng = _patched_default_rng


@contextlib.contextmanager
def _seeded_rng_fallback(seed: int):
    """Within this block, pymoo's unseeded ``default_rng(None)`` fallbacks become
    deterministic (seeded off ``seed``), so a fixed seed reproduces the whole run."""
    prev = getattr(_rng_tls, "seed_sequence", None)
    _rng_tls.seed_sequence = np.random.SeedSequence(seed)
    try:
        yield
    finally:
        _rng_tls.seed_sequence = prev


class _SimplexSampling(Sampling):
    """Sample initial populations on the simplex (allocations sum to 100).

    Optionally prepends ``seed_population`` (an (n_seed, n_var) array) to the
    random sample; remaining slots are filled by uniform simplex sampling.
    Seeds + random collectively produce ``n_samples`` individuals.
    """

    def __init__(self, seed_population: np.ndarray | None = None):
        super().__init__()
        self.seed_population = seed_population

    def _do(self, problem, n_samples, **kwargs):
        # pymoo (0.6.1.6+) passes its seeded Generator as ``random_state`` — draw from it,
        # not global ``np.random``, so a fixed seed reproduces the initial population.
        rs = kwargs.get("random_state") or np.random.default_rng()
        n_var = problem.n_var
        seeds = self.seed_population
        n_seed = 0
        if seeds is not None and len(seeds) > 0:
            # Clip seeds to n_samples and to correct width
            seeds = np.asarray(seeds, dtype=float)
            if seeds.shape[1] != n_var:
                seeds = None  # shape mismatch → drop seeds silently
            else:
                seeds = seeds[:n_samples]
                n_seed = len(seeds)
        n_random = n_samples - n_seed
        raw = rs.dirichlet(np.ones(n_var), size=n_random) * 100.0
        X = np.floor(raw).astype(float)
        remainders = raw - X
        for i in range(n_random):
            diff = int(100 - X[i].sum())
            if diff > 0:
                indices = np.argsort(remainders[i])[::-1][:diff]
                X[i, indices] += 1.0
        if n_seed > 0:
            X = np.concatenate([seeds, X], axis=0)
        return X


class _SeededBinarySampling(Sampling):
    """Binary-mode initial population with optional seed individuals prepended.

    Returns a bool array — pymoo's BitflipMutation expects bool for ``~X``.
    """

    def __init__(self, seed_population: np.ndarray | None = None):
        super().__init__()
        self.seed_population = seed_population

    def _do(self, problem, n_samples, **kwargs):
        # Use pymoo's seeded ``random_state`` Generator (not global ``np.random``) so a fixed
        # seed reproduces the initial population.
        rs = kwargs.get("random_state") or np.random.default_rng()
        n_var = problem.n_var
        seeds = self.seed_population
        n_seed = 0
        if seeds is not None and len(seeds) > 0:
            seeds = np.asarray(seeds) > 0.5  # coerce to bool
            if seeds.shape[1] != n_var:
                seeds = None
                n_seed = 0
            else:
                seeds = seeds[:n_samples]
                n_seed = len(seeds)
        n_random = n_samples - n_seed
        X = rs.random((n_random, n_var)) < 0.5  # bool
        if n_seed > 0:
            X = np.concatenate([seeds, X], axis=0)
        return X


class _SimplexRepair(Repair):
    """After crossover/mutation, project solutions back onto the simplex (sum=100, non-negative).

    When the pymoo problem has a max_allocation attribute, iteratively clamp
    and redistribute so no variable exceeds the cap while preserving sum=100.
    """

    def _do(self, problem, X, **kwargs):
        # Clamp to non-negative
        X = np.maximum(X, 0.0)
        # Normalize each row to sum to 100
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-9)  # avoid div by zero
        X = X / row_sums * 100.0

        # Enforce max_allocation: clamp and redistribute excess
        cap = getattr(problem, "max_allocation", None)
        if cap is not None:
            for _ in range(5):  # iterate until stable (typically 1-2 passes)
                excess = np.maximum(X - cap, 0.0)
                total_excess = excess.sum(axis=1, keepdims=True)
                if np.all(total_excess < 0.01):
                    break
                X = np.minimum(X, cap)
                # Redistribute excess proportionally to under-cap variables
                under_cap = X < cap - 0.01
                under_cap_sum = np.where(under_cap, X, 0.0).sum(axis=1, keepdims=True)
                under_cap_sum = np.maximum(under_cap_sum, 1e-9)
                redistribution = np.where(under_cap, X / under_cap_sum * total_excess, 0.0)
                X = X + redistribution

        return X

from .models import (
    Aggregation,
    Approach,
    BoundOperator,
    InteractionMatrix,
    OptimizeMode,
    Problem,
    QualityIndicators,
    Run,
    Solution,
    ValidationIssue,
    ValidationResult,
    _content_signature,
)


def validate(problem: Problem) -> ValidationResult:
    """Check if a problem is ready to optimize."""
    issues: list[ValidationIssue] = []
    missing_scores: list[dict[str, str]] = []

    # Duplicate objective names
    obj_name_list = [o.name for o in problem.objectives]
    obj_name_dupes = {n for n in obj_name_list if obj_name_list.count(n) > 1}
    if obj_name_dupes:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Duplicate objective names: {obj_name_dupes}.",
        ))

    # Duplicate option names
    opt_name_list = [o.name for o in problem.options]
    opt_name_dupes = {n for n in opt_name_list if opt_name_list.count(n) > 1}
    if opt_name_dupes:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Duplicate option names: {opt_name_dupes}.",
        ))

    # Need >= 2 objectives
    if len(problem.objectives) < 2:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Need at least 2 objectives, have {len(problem.objectives)}.",
        ))

    # Need >= 3 options
    if len(problem.options) < 3:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Need at least 3 options, have {len(problem.options)}.",
        ))

    # Score matrix completeness
    obj_names = {o.name for o in problem.objectives}
    opt_names = {o.name for o in problem.options}
    scored = {(s.option, s.objective) for s in problem.scores}

    for opt in opt_names:
        for obj in obj_names:
            if (opt, obj) not in scored:
                missing_scores.append({"option": opt, "objective": obj})

    if missing_scores:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Score matrix incomplete: {len(missing_scores)} missing scores.",
        ))

    # Constraint feasibility checks
    for c in problem.constraints:
        if c.type == "force_include" and c.option not in opt_names:
            issues.append(ValidationIssue(
                severity="error",
                message=f"force_include references unknown option '{c.option}'.",
            ))
        elif c.type == "force_exclude" and c.option not in opt_names:
            issues.append(ValidationIssue(
                severity="error",
                message=f"force_exclude references unknown option '{c.option}'.",
            ))
        elif c.type == "objective_bound" and c.objective not in obj_names:
            issues.append(ValidationIssue(
                severity="error",
                message=f"objective_bound references unknown objective '{c.objective}'.",
            ))
        elif c.type == "exclusion_pair":
            for opt in (c.option_a, c.option_b):
                if opt not in opt_names:
                    issues.append(ValidationIssue(
                        severity="error",
                        message=f"exclusion_pair references unknown option '{opt}'.",
                    ))
            if c.option_a == c.option_b:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"exclusion_pair has same option for both: '{c.option_a}'.",
                ))
        elif c.type == "dependency":
            for opt in (c.if_option, c.then_option):
                if opt not in opt_names:
                    issues.append(ValidationIssue(
                        severity="error",
                        message=f"dependency references unknown option '{opt}'.",
                    ))
        elif c.type == "group_limit":
            for opt in c.options:
                if opt not in opt_names:
                    issues.append(ValidationIssue(
                        severity="error",
                        message=f"group_limit references unknown option '{opt}'.",
                    ))
            if c.max < 0:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"group_limit max ({c.max}) must be non-negative.",
                ))
        elif c.type == "max_allocation":
            if not (1 <= c.max <= 100):
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"max_allocation must be between 1 and 100, got {c.max}.",
                ))
            if problem.approach != Approach.proportional:
                issues.append(ValidationIssue(
                    severity="warning",
                    message="max_allocation constraint only applies to proportional mode; ignored in binary mode.",
                ))

    # Check force_include + force_exclude conflict
    forced_in = {c.option for c in problem.constraints if c.type == "force_include"}
    forced_out = {c.option for c in problem.constraints if c.type == "force_exclude"}
    conflict = forced_in & forced_out
    if conflict:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Options both force_include and force_exclude: {conflict}.",
        ))

    # Check cardinality feasibility
    available = len(opt_names - forced_out)
    for c in problem.constraints:
        if c.type == "cardinality":
            if c.min > available:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"Cardinality min ({c.min}) exceeds available options ({available}).",
                ))
            if c.min > c.max:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"Cardinality min ({c.min}) > max ({c.max}).",
                ))
            if len(forced_in) > c.max:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"force_include count ({len(forced_in)}) exceeds cardinality max ({c.max}).",
                ))

    # ── 1.9 Pre-solve constraint conflict detection ──
    issues.extend(_check_constraint_conflicts(problem, forced_in, forced_out, available))

    # Validate interaction matrices for quadratic objectives
    im_map = {m.objective: m for m in problem.interaction_matrices}
    for obj in problem.objectives:
        if obj.aggregation == Aggregation.quadratic:
            if obj.name not in im_map:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"Objective '{obj.name}' uses quadratic aggregation but has no interaction matrix.",
                ))
            else:
                im = im_map[obj.name]
                # Check all options are covered
                im_opts = set(im.entries.keys())
                missing = opt_names - im_opts
                if missing:
                    issues.append(ValidationIssue(
                        severity="error",
                        message=f"Interaction matrix for '{obj.name}' missing options: {missing}.",
                    ))
    for im in problem.interaction_matrices:
        if im.objective not in obj_names:
            issues.append(ValidationIssue(
                severity="warning",
                message=f"Interaction matrix references unknown objective '{im.objective}'.",
            ))
        for sg in im.scale_groups:
            unknown = [opt for opt in sg.options if opt not in opt_names]
            if unknown:
                issues.append(ValidationIssue(
                    severity="warning",
                    message=f"Interaction matrix for '{im.objective}' scale_group references unknown options: {unknown}.",
                ))

    # Validate scenario interaction_matrix_overrides references
    if problem.scenario_config is not None:
        for sc in problem.scenario_config.scenarios:
            for im in sc.interaction_matrix_overrides:
                if im.objective not in obj_names:
                    issues.append(ValidationIssue(
                        severity="warning",
                        message=f"Scenario '{sc.name}' interaction_matrix_override references unknown objective '{im.objective}'.",
                    ))
                # Check entries row/column keys
                entry_opts = set(im.entries.keys())
                for row_opts in im.entries.values():
                    entry_opts.update(row_opts.keys())
                unknown_entries = entry_opts - opt_names
                if unknown_entries:
                    issues.append(ValidationIssue(
                        severity="warning",
                        message=f"Scenario '{sc.name}' interaction_matrix_override for '{im.objective}' entries reference unknown options: {sorted(unknown_entries)}.",
                    ))
                # Check scale_groups option names
                for sg in im.scale_groups:
                    unknown = [opt for opt in sg.options if opt not in opt_names]
                    if unknown:
                        issues.append(ValidationIssue(
                            severity="warning",
                            message=f"Scenario '{sc.name}' interaction_matrix_override for '{im.objective}' scale_group references unknown options: {unknown}.",
                        ))

    ready = all(i.severity != "error" for i in issues)
    return ValidationResult(ready=ready, issues=issues, missing_scores=missing_scores)


def _check_constraint_conflicts(
    problem: Problem, forced_in: set[str], forced_out: set[str], available: int,
) -> list[ValidationIssue]:
    """1.9 — Detect pre-solve conflicts among hard constraints.

    Pairwise/arithmetic infeasibilities the solver would only surface after a full
    run. Caller has already validated that referenced options/objectives exist; this
    layer only checks consistency among the constraint set itself.

    Issues caught:
      A. GroupLimit vs force_include — |force_in ∩ group| > group.max
      B. ExclusionPair vs force_include — both members force-included
      C. Dependency cycles — A→B→…→A in the dependency graph
      D. Dependency vs force_exclude — if-option forced in but then-option forced out
      E. MaxAllocation arithmetic (proportional) — cap × available_options < 100
    """
    issues: list[ValidationIssue] = []

    # A. GroupLimit vs force_include
    for c in problem.constraints:
        if c.type == "group_limit":
            forced_in_group = forced_in.intersection(c.options)
            if len(forced_in_group) > c.max:
                issues.append(ValidationIssue(
                    severity="error",
                    message=(
                        f"group_limit max ({c.max}) is below the number of force_included options "
                        f"in the group ({len(forced_in_group)}): {sorted(forced_in_group)}."
                    ),
                ))

    # B. ExclusionPair vs force_include
    for c in problem.constraints:
        if c.type == "exclusion_pair":
            if c.option_a in forced_in and c.option_b in forced_in:
                issues.append(ValidationIssue(
                    severity="error",
                    message=(
                        f"exclusion_pair ('{c.option_a}', '{c.option_b}') conflicts with "
                        "force_include on both options."
                    ),
                ))

    # C + D. Dependency graph: cycles and force_exclude conflicts
    dep_edges = [
        (c.if_option, c.then_option)
        for c in problem.constraints if c.type == "dependency"
    ]
    if dep_edges:
        # D. Conflict with force_exclude (only when if-option is actually selectable)
        for if_opt, then_opt in dep_edges:
            if if_opt in forced_in and then_opt in forced_out:
                issues.append(ValidationIssue(
                    severity="error",
                    message=(
                        f"dependency '{if_opt}' → '{then_opt}' is infeasible: '{if_opt}' is "
                        f"force_included but '{then_opt}' is force_excluded."
                    ),
                ))

        # C. Cycle detection (DFS on adjacency)
        adj: dict[str, list[str]] = {}
        for if_opt, then_opt in dep_edges:
            adj.setdefault(if_opt, []).append(then_opt)

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in adj}
        reported_cycles: set[tuple[str, ...]] = set()

        def _dfs(node: str, stack: list[str]) -> None:
            color[node] = GRAY
            stack.append(node)
            for nxt in adj.get(node, []):
                if color.get(nxt, WHITE) == GRAY:
                    # Cycle found — extract the cycle portion of the stack
                    cycle = stack[stack.index(nxt):] + [nxt]
                    # Canonicalize so different traversal entry points dedupe to one report
                    key = tuple(sorted(cycle[:-1]))
                    if key not in reported_cycles:
                        reported_cycles.add(key)
                        issues.append(ValidationIssue(
                            severity="error",
                            message=f"dependency cycle detected: {' → '.join(cycle)}.",
                        ))
                elif color.get(nxt, WHITE) == WHITE:
                    _dfs(nxt, stack)
            stack.pop()
            color[node] = BLACK

        for node in list(adj):
            if color[node] == WHITE:
                _dfs(node, [])

    # E. MaxAllocation arithmetic (proportional mode only)
    if problem.approach == Approach.proportional:
        for c in problem.constraints:
            if c.type == "max_allocation":
                if c.max * available < 100:
                    issues.append(ValidationIssue(
                        severity="error",
                        message=(
                            f"max_allocation cap ({c.max}%) × available options ({available}) = "
                            f"{c.max * available}% < 100% required. Allocation cannot sum to 100."
                        ),
                    ))

    return issues


def _parse_constraints(problem: Problem) -> dict:
    """Extract constraint parameters from problem. Shared by binary and proportional."""
    opt_names = [o.name for o in problem.options]
    obj_list = problem.objectives
    opt_index = {name: i for i, name in enumerate(opt_names)}

    forced_in_idx = set()
    forced_out_idx = set()
    cardinality_min = 1
    cardinality_max = len(opt_names)
    obj_bounds: list[tuple[int, BoundOperator, float]] = []
    exclusion_pairs: list[tuple[int, int]] = []
    dependencies: list[tuple[int, int]] = []  # (if_idx, then_idx)
    group_limits: list[tuple[list[int], int]] = []  # (group_indices, max)
    max_allocation: int | None = None  # max allocation % per option (proportional only)

    for c in problem.constraints:
        if c.type == "force_include":
            forced_in_idx.add(opt_index[c.option])
        elif c.type == "force_exclude":
            forced_out_idx.add(opt_index[c.option])
        elif c.type == "cardinality":
            cardinality_min = c.min
            cardinality_max = c.max
        elif c.type == "objective_bound":
            obj_idx = next(j for j, o in enumerate(obj_list) if o.name == c.objective)
            obj_bounds.append((obj_idx, c.operator, c.value))
        elif c.type == "exclusion_pair":
            exclusion_pairs.append((opt_index[c.option_a], opt_index[c.option_b]))
        elif c.type == "dependency":
            dependencies.append((opt_index[c.if_option], opt_index[c.then_option]))
        elif c.type == "group_limit":
            group_indices = [opt_index[o] for o in c.options]
            group_limits.append((group_indices, c.max))
        elif c.type == "max_allocation":
            max_allocation = c.max

    return {
        "forced_in": forced_in_idx,
        "forced_out": forced_out_idx,
        "cardinality_min": cardinality_min,
        "cardinality_max": cardinality_max,
        "obj_bounds": obj_bounds,
        "exclusion_pairs": exclusion_pairs,
        "dependencies": dependencies,
        "group_limits": group_limits,
        "max_allocation": max_allocation,
    }


def _build_score_matrix(problem: Problem) -> np.ndarray:
    """Build score_matrix[opt_idx][obj_idx] from problem scores."""
    opt_names = [o.name for o in problem.options]
    obj_list = problem.objectives
    score_map = {(s.option, s.objective): s.value for s in problem.scores}
    matrix = np.zeros((len(opt_names), len(obj_list)))
    for i, opt in enumerate(opt_names):
        for j, obj in enumerate(obj_list):
            matrix[i, j] = score_map[(opt, obj.name)]
    return matrix


def _build_interaction_matrices(problem: Problem) -> dict[int, np.ndarray]:
    """Build numpy interaction matrices for quadratic objectives.

    Returns dict mapping objective index → (n_options, n_options) numpy array.
    Only includes objectives with quadratic aggregation that have a matching matrix.
    """
    opt_names = [o.name for o in problem.options]
    opt_index = {name: i for i, name in enumerate(opt_names)}
    n = len(opt_names)
    matrix_map = {m.objective: m for m in problem.interaction_matrices}

    result = {}
    for j, obj in enumerate(problem.objectives):
        if obj.aggregation == Aggregation.quadratic and obj.name in matrix_map:
            im = matrix_map[obj.name]
            M = np.zeros((n, n))
            for a, row in im.entries.items():
                if a in opt_index:
                    for b, val in row.items():
                        if b in opt_index:
                            M[opt_index[a], opt_index[b]] = val
            result[j] = M
    return result


def _compute_extreme_seeds(problem: Problem, score_matrix: np.ndarray, cp: dict) -> np.ndarray:
    """Build one seed individual per objective, biased toward that objective's extreme.

    Greedy heuristic: rank options by their score for objective j (descending if
    maximize, ascending if minimize), then select a set of top-K options that
    respects constraints (forced_in/out, cardinality, group_limits). Build a
    portfolio from that set:
      - binary: 1 for selected, 0 otherwise.
      - proportional: evenly distribute 100% across selected, capped at max_allocation.

    Seeds are near-extreme starting points — NSGA polishes them. Works for any
    aggregation (sum/avg/quadratic) because NSGA will refine; the seed just needs
    to be closer to the corner than random. Single-objective fitness is never
    computed here; this is a heuristic, not a sub-solver.

    Returns an (n_seeds, n_options) float array. n_seeds ≤ len(objectives).
    """
    opt_names = [o.name for o in problem.options]
    n_options = len(opt_names)
    obj_list = problem.objectives
    is_binary = problem.approach == Approach.binary

    forced_in = cp.get("forced_in", set())
    forced_out = cp.get("forced_out", set())
    card_min = cp.get("cardinality_min", 1)
    card_max = cp.get("cardinality_max", n_options)
    group_limits = cp.get("group_limits", [])
    max_allocation = cp.get("max_allocation")  # may be None

    # For proportional: target K holdings that spend the full budget at cap.
    # If max_allocation=30 and sum=100, we need at least ceil(100/30)=4 holdings.
    # For binary, K is bounded only by cardinality.
    if is_binary:
        target_k = card_max
    else:
        if max_allocation:
            min_k_for_budget = int(np.ceil(100.0 / max_allocation))
            target_k = max(card_min, min_k_for_budget)
            target_k = min(target_k, card_max)
        else:
            target_k = max(card_min, min(6, card_max))  # small default

    seeds = []
    for j, obj in enumerate(obj_list):
        order = np.argsort(-score_matrix[:, j]) if obj.direction.value == "maximize" else np.argsort(score_matrix[:, j])

        selected = list(forced_in)
        group_counts = [0 for _ in group_limits]
        # Pre-count forced-ins against group caps
        for g_idx, (g_indices, _g_max) in enumerate(group_limits):
            group_counts[g_idx] = sum(1 for idx in selected if idx in g_indices)

        for idx in order:
            if len(selected) >= target_k:
                break
            if idx in forced_out or idx in selected:
                continue
            # Check group caps
            ok = True
            for g_idx, (g_indices, g_max) in enumerate(group_limits):
                if idx in g_indices and group_counts[g_idx] >= g_max:
                    ok = False
                    break
            if not ok:
                continue
            selected.append(int(idx))
            for g_idx, (g_indices, _g_max) in enumerate(group_limits):
                if idx in g_indices:
                    group_counts[g_idx] += 1

        # If we couldn't reach cardinality_min due to tight constraints, skip this seed
        if len(selected) < card_min:
            continue

        if is_binary:
            vec = np.zeros(n_options, dtype=float)
            vec[selected] = 1.0
        else:
            vec = np.zeros(n_options, dtype=float)
            k = len(selected)
            cap = max_allocation if max_allocation is not None else 100
            # Start with equal allocation, capped
            base_alloc = min(cap, 100 // k)
            for idx in selected:
                vec[idx] = base_alloc
            # Distribute remaining units to highest-ranked options up to cap
            used = int(vec.sum())
            remaining = 100 - used
            for idx in order:
                if remaining <= 0:
                    break
                if idx in selected and vec[idx] < cap:
                    delta = min(cap - int(vec[idx]), remaining)
                    vec[idx] += delta
                    remaining -= delta

        seeds.append(vec)

    if not seeds:
        return np.empty((0, n_options))
    return np.array(seeds, dtype=float)


def _tune_parameters(problem: Problem, mode: OptimizeMode, seed_population: np.ndarray | None = None) -> tuple:
    """Return (algorithm, n_gen) adapted to problem size and mode.

    Scales directly from the two independent dimensions that matter:
      1. Solution space size — driven by n_options and approach type.
         Binary: 2^n values. Proportional: ~100^n (continuous, much larger).
         We use log2 of the space to set pop/gen multipliers.
      2. Objective count — independently drives algorithm selection and
         population needs (more objectives need more population for coverage).

    Mode controls the budget: fast for exploration, thorough for convergence.
    """
    n = len(problem.options)
    n_obj = len(problem.objectives)
    is_binary = problem.approach == Approach.binary
    is_fast = mode == OptimizeMode.fast
    use_nsga3 = n_obj >= 4

    # --- Solution space drives pop_size and n_gen ---
    # Binary space: 2^n. Proportional space: effectively continuous (~100^n).
    # Use log2 of space as a scaling factor, capped to keep things sane.
    # Binary: log2(2^n) = n. Proportional: log2(100^n) ≈ 6.6n.
    if is_binary:
        space_factor = n
    else:
        space_factor = 6.6 * n

    # Population: scales with sqrt of space_factor (diminishing returns).
    # More objectives need more population to cover the Pareto surface.
    obj_multiplier = 1.0 + 0.3 * (n_obj - 2)  # 1.0 at 2 obj, 2.8 at 8 obj
    mode_pop = 1.0 if is_fast else 1.8
    pop_size = int(math.sqrt(space_factor) * 10 * obj_multiplier * mode_pop)

    # Generations: scales with space_factor more aggressively than pop.
    # Thorough mode gets substantially more generations for convergence.
    mode_gen = 1.0 if is_fast else 2.5
    gen_base = max(8, int(space_factor * 0.5))
    n_gen = int(gen_base * obj_multiplier * mode_gen)

    # Floors and caps
    pop_size = max(50, min(pop_size, 2000))
    n_gen = max(50, min(n_gen, 3000))

    # --- Algorithm selection (objective count only) ---
    if use_nsga3:
        n_partitions = max(3, 12 - n_obj)
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        pop_size = len(ref_dirs)  # pymoo requires pop_size == len(ref_dirs)
        AlgClass = NSGA3
        alg_kwargs: dict = {"ref_dirs": ref_dirs, "pop_size": pop_size}
    else:
        AlgClass = NSGA2
        alg_kwargs = {"pop_size": pop_size}

    # --- Operator adaptation (approach type only) ---
    if is_binary:
        # Mutation prob: ~1 bit flip per individual on average, bounded
        mut_prob = max(0.01, min(0.25, 1.0 / n))
        sampling = _SeededBinarySampling(seed_population) if seed_population is not None else BinaryRandomSampling()
        alg_kwargs.update(
            sampling=sampling,
            crossover=TwoPointCrossover(prob=0.9),
            mutation=BitflipMutation(prob=mut_prob),
            eliminate_duplicates=True,
        )
    else:
        # Larger search spaces benefit from higher eta (more exploitation)
        eta_scale = min(1.0, math.log2(max(n, 3)) / math.log2(200))
        sbx_eta = int(10 + eta_scale * 20)   # 10-30
        pm_eta = int(15 + eta_scale * 25)     # 15-40
        alg_kwargs.update(
            sampling=_SimplexSampling(seed_population),
            crossover=SBX(prob=0.9, eta=sbx_eta),
            mutation=PM(eta=pm_eta),
            repair=_SimplexRepair(),
            eliminate_duplicates=True,
        )

    return AlgClass(**alg_kwargs), n_gen


def _build_termination(n_gen: int, time_limit: float | None):
    """Termination for pymoo: the generation cap, optionally OR'd with a wall-clock cap.

    ``TerminationCollection`` stops at whichever sub-criterion fires first (its progress is
    the max over members), so a ``time_limit`` bounds a long ``thorough`` run without
    truncating a fast one that converges by generation count first. None → the plain
    ``("n_gen", n_gen)`` tuple, i.e. the prior behavior unchanged.
    """
    if time_limit and time_limit > 0:
        return TerminationCollection(
            MaximumGenerationTermination(n_gen),
            TimeBasedTermination(float(time_limit)),
        )
    return ("n_gen", n_gen)


def _was_time_limited(time_limit: float | None, elapsed: float) -> bool:
    """True only when the wall-clock cap actually cut the search short (best-so-far frontier).

    pymoo's ``TimeBasedTermination`` fires at the first generation boundary where elapsed
    reaches the cap, so a *time*-terminated run measures ``elapsed >= time_limit`` while a run
    that *converged* by generation count first stops below the cap. Keying on that threshold
    (not a fractional margin) avoids mislabeling a run that merely happened to finish near its
    cap. ``time_limit <= 0`` is uncapped (``_build_termination`` ignores it), so never limited."""
    return bool(time_limit) and time_limit > 0 and elapsed >= time_limit


def optimize(
    problem: Problem,
    mode: OptimizeMode | None = None,
    max_solutions: int | None = None,
    seed: int | None = None,
    exact: bool = False,
    solver: str | None = None,
    time_limit: float | None = None,
) -> Run:
    """Validate and run multi-objective optimization. Returns a Run with solutions.

    seed: deterministic RNG seed for reproducibility, recorded on Run.seed_used.
    Same problem + same seed reproduce the exact frontier (in- and cross-process).
    When None, a fresh 32-bit seed is drawn and recorded so the run can be
    reproduced later even when a seed wasn't requested up front.

    solver: which engine to run. None/"nsga" (default) is the pymoo NSGA-II/III
    evolutionary search that fits any problem shape. "highs" / "cuopt" select an
    exact inner-solve backend (each frontier point solved to optimality) — opt-in,
    so callers should pre-check ``solvers.available_solvers`` /
    ``solvers.exact_solver_fits``. An exact backend requested for a shape it can't
    solve raises ``ValueError`` rather than silently degrading to NSGA (the same
    no-silent-degradation contract the solve tool enforces); an unknown solver name
    also raises.

    exact: exact backends only — certify each MILP scalarization (gap→0, accept only a
    proven-Optimal status) instead of the default gap/time-bounded solve. Slower; lets a
    run trade speed for certified optimality. No-op on the NSGA paths.

    time_limit: optional wall-clock cap in seconds. On the NSGA path it is OR'd with the
    generation budget (stop at whichever fires first); on an exact path it bounds the
    scalarization sweep. When the cap is hit the returned frontier is best-so-far and the
    Run is flagged ``time_limited`` (each exact point is still optimal for its scalarization
    — only coverage is partial). None = uncapped (the prior behavior).

    The chosen engine is recorded on ``Run.solver`` (and ``Run.exact``) so results stay
    traceable to how they were produced.
    """
    if mode is None:
        mode = OptimizeMode.fast
    vr = validate(problem)
    if not vr.ready:
        raise ValueError(f"Problem not ready: {[i.message for i in vr.issues]}")

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))

    # Opt-in exact-solver backends, selected by the ``solver`` arg (the single selection
    # mechanism). Cheap string gate first so the default path never imports a backend; each
    # backend imports its solver lazily (safe whether or not cuOpt/highspy is installed) and
    # stamps its own provenance on the Run. An explicitly requested exact solver that doesn't
    # fit the problem shape raises rather than silently degrading to NSGA — the same
    # no-silent-degradation contract the solve tool enforces.
    from solvers import exact_solver_fits, is_exact_solver

    backend = (solver or "").strip().lower()
    if backend in ("", "nsga", "auto", "default"):
        backend = ""
    elif is_exact_solver(backend):
        fits, reason = exact_solver_fits(problem)
        if not fits:
            raise ValueError(f"Exact solver '{backend}' does not fit this problem: {reason}")
        if backend == "cuopt":
            from solvers.cuopt_backend import _optimize_cuopt
            return _optimize_cuopt(problem, mode, max_solutions=max_solutions, seed=seed,
                                   exact=exact, time_limit=time_limit)
        from solvers.highs_backend import _optimize_highs
        return _optimize_highs(problem, mode, max_solutions=max_solutions, seed=seed,
                               exact=exact, time_limit=time_limit)
    else:
        raise ValueError(f"Unknown solver '{solver}'. Use 'nsga' (default), 'highs', or 'cuopt'.")

    if problem.approach == Approach.proportional:
        run = _optimize_proportional(problem, mode, max_solutions=max_solutions, seed=seed,
                                     time_limit=time_limit)
    else:
        run = _optimize_binary(problem, mode, max_solutions=max_solutions, seed=seed,
                               time_limit=time_limit)
    run.solver = "nsga-iii" if len(problem.objectives) >= 4 else "nsga-ii"
    return run


def _apply_matrix_override(base: "InteractionMatrix | None", override: "InteractionMatrix") -> "InteractionMatrix":
    """Apply a scenario override to a base interaction matrix.

    Modes:
    - "replace" (default): discard the base, use override.entries directly.
    - "upsert": start from base.entries (empty if no base), merge override.entries,
      enforcing symmetry (writing (a, b) also writes (b, a)).

    scale_groups (applied after replace/upsert, composable):
    - For each group, multiply off-diagonal entries where both endpoints are
      in the group by ``factor``. Diagonal entries are untouched.
    """
    import copy

    if override.mode == "upsert":
        if base is None:
            merged = {}
        else:
            merged = copy.deepcopy(base.entries)
        # Apply sparse cells with symmetry
        for row_key, cells in override.entries.items():
            for col_key, value in cells.items():
                merged.setdefault(row_key, {})[col_key] = value
                merged.setdefault(col_key, {})[row_key] = value
    else:
        # replace: override.entries is authoritative
        merged = copy.deepcopy(override.entries)

    # Apply scale_groups
    for group in override.scale_groups:
        options_in_group = set(group.options)
        for a in options_in_group:
            if a not in merged:
                continue
            for b, value in list(merged[a].items()):
                if b == a:
                    continue  # skip diagonal
                if b in options_in_group:
                    merged[a][b] = value * group.factor

    return InteractionMatrix(
        objective=override.objective,
        entries=merged,
        mode="replace",  # merged result is authoritative
        scale_groups=[],
    )


def build_scenario_problem(problem: Problem, scenario: "Scenario") -> Problem:
    """Apply a scenario's overrides to a deep copy of ``problem`` and return a
    base-case problem (``scenario_config`` cleared) ready to optimize or evaluate.

    Overrides, in order: bulk score_adjustments (multiply/add), explicit
    score_overrides (take precedence), constraint_overrides (full replacement),
    interaction_matrix overrides (replace/upsert/scale_groups). Extracted from
    ``optimize_scenarios`` so the same overridden scores can be reused to score a
    specific slate under the scenario (see ``evaluate_slate``)."""
    sp = problem.model_copy(deep=True)

    if scenario.score_adjustments:
        adj_map = {a.objective: a for a in scenario.score_adjustments}
        for score in sp.scores:
            if score.objective in adj_map:
                adj = adj_map[score.objective]
                if adj.multiply is not None:
                    score.value *= adj.multiply
                if adj.add is not None:
                    score.value += adj.add

    if scenario.score_overrides:
        override_map = {(s.option, s.objective): s.value for s in scenario.score_overrides}
        for score in sp.scores:
            key = (score.option, score.objective)
            if key in override_map:
                score.value = override_map[key]

    if scenario.constraint_overrides:
        sp.constraints = list(scenario.constraint_overrides)

    if scenario.interaction_matrix_overrides:
        base_by_obj = {m.objective: m for m in sp.interaction_matrices}
        for override in scenario.interaction_matrix_overrides:
            base = base_by_obj.get(override.objective)
            base_by_obj[override.objective] = _apply_matrix_override(base, override)
        sp.interaction_matrices = list(base_by_obj.values())

    sp.scenario_config = None
    return sp


def score_slate(
    problem: Problem,
    selected_options: list[str],
    allocations: dict[str, int] | None = None,
    scenario: "Scenario | None" = None,
) -> dict:
    """Objective vector **and** feasibility for one specific slate under the base case
    or a named scenario, in a single problem build.

    Reuses the optimizer's own aggregation (sum/avg/min/max/quadratic, incl. quadratic
    interaction matrices) and constraint encoding, so values and feasibility match what
    ``solve`` would report. ``scenario`` is a ``Scenario`` object (None = base case).
    For proportional problems pass ``allocations``; for binary, ``selected_options``.
    Returns ``{"values": {obj: v}, "feasible": bool}``."""
    p = build_scenario_problem(problem, scenario) if scenario is not None else problem
    n = len(p.options)
    idx = {o.name: i for i, o in enumerate(p.options)}
    score_matrix = _build_score_matrix(p)
    im = _build_interaction_matrices(p)
    cp = _parse_constraints(p)

    x = np.zeros((1, n))
    if p.approach == Approach.proportional:
        for name, pct in (allocations or {}).items():
            if name in idx:
                x[0, idx[name]] = pct
        prob = _ProportionalProblem(
            n_options=n, score_matrix=score_matrix, objectives=p.objectives,
            interaction_matrices=im, **cp,
        )
    else:
        for name in selected_options:
            if name in idx:
                x[0, idx[name]] = 1.0
        prob = _FrontierProblem(
            n_options=n, score_matrix=score_matrix, objectives=p.objectives,
            interaction_matrices=im, **cp,
        )

    values = {
        obj.name: round(float(prob._aggregate_objective(x, j)[0]), 4)
        for j, obj in enumerate(p.objectives)
    }
    out: dict = {}
    prob._evaluate(x.copy(), out)
    g = out.get("G")
    feasible = bool(g is None or np.asarray(g).size == 0 or np.all(np.asarray(g) <= 1e-6))
    return {"values": values, "feasible": feasible}


def evaluate_slate(
    problem: Problem,
    selected_options: list[str],
    allocations: dict[str, int] | None = None,
    scenario: "Scenario | None" = None,
) -> dict[str, float]:
    """Objective vector for one slate under the base case or a scenario.
    Thin wrapper over ``score_slate`` (which also returns feasibility)."""
    return score_slate(problem, selected_options, allocations, scenario)["values"]


def optimize_scenarios(
    problem: Problem,
    mode: OptimizeMode | None = None,
    max_solutions: int | None = None,
    seed: int | None = None,
    exact: bool = False,
    solver: str | None = None,
    time_limit: float | None = None,
) -> dict[str, Run]:
    """Run optimization independently per scenario. Returns {scenario_name: Run}.

    Each scenario is solved as a fully independent optimization — no probability
    weighting required. Probabilities are optional and only used downstream in
    explorer for expected-value analysis.

    Scenarios run in parallel to stay within MCP tool-call timeouts.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not problem.scenario_config or not problem.scenario_config.enabled:
        raise ValueError("Scenario config not enabled.")
    if not problem.scenario_config.scenarios:
        raise ValueError("No scenarios defined.")

    def _build_and_solve(scenario: "Scenario") -> tuple[str, Run]:
        scenario_problem = build_scenario_problem(problem, scenario)

        # Derive a distinct, deterministic per-scenario seed so they don't
        # collide on identical initializations while remaining reproducible.
        # Uses hashlib (not built-in hash) because Python's hash(str) is
        # PYTHONHASHSEED-randomized per process, breaking reproducibility.
        scenario_seed = None
        if seed is not None:
            name_hash = int(hashlib.sha256(scenario.name.encode()).hexdigest()[:8], 16)
            scenario_seed = (seed + name_hash) % (2**31 - 1)
        return scenario.name, optimize(
            scenario_problem, mode=mode, max_solutions=max_solutions, seed=scenario_seed,
            exact=exact, solver=solver, time_limit=time_limit,
        )

    with ThreadPoolExecutor(max_workers=len(problem.scenario_config.scenarios)) as pool:
        futures = [pool.submit(_build_and_solve, s) for s in problem.scenario_config.scenarios]
        results = dict(f.result() for f in futures)

    return results


def _union_with_elites(
    result,
    pymoo_problem,
    seed_population: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Protect extreme-point seeds from evolutionary dilution.

    After NSGA finishes, evaluate the precomputed corner seeds (constraint-compliant
    by construction) and union them with result.X/result.F. Infeasible seeds are
    dropped. The combined set is filtered by non-dominated sorting so only
    Pareto-optimal points survive.

    Effect: generic NSGA operators don't preserve every seed across ~300 generations
    (they get outcompeted by mid-frontier individuals with better crowding distance).
    Protecting the seeds post-hoc keeps corner sharpness without perturbing the
    main evolutionary search or mid-frontier diversity.

    Works across all modes (binary / proportional), any n_obj, any problem size.
    """
    if result.X is None or result.F is None or len(result.F) == 0:
        return result.X, result.F
    if seed_population is None or len(seed_population) == 0:
        return result.X, result.F

    # Evaluate seeds through the pymoo problem (gives F and G in pymoo's minimize
    # convention, so comparisons with result.F are apples-to-apples).
    seeds = np.asarray(seed_population, dtype=float)
    # pymoo's Problem.evaluate returns a dict when return_values_of is specified.
    out = pymoo_problem.evaluate(seeds, return_values_of=["F", "G"], return_as_dictionary=True)
    seed_F = out.get("F")
    seed_G = out.get("G")

    if seed_F is None or len(seed_F) == 0:
        return result.X, result.F

    # Drop infeasible seeds (any G > 0 means a constraint is violated).
    if seed_G is not None and seed_G.size > 0:
        feasible_mask = np.all(seed_G <= 1e-6, axis=1)
        seeds = seeds[feasible_mask]
        seed_F = seed_F[feasible_mask]
    if len(seeds) == 0:
        return result.X, result.F

    # Union and take non-dominated front.
    union_X = np.vstack([result.X, seeds])
    union_F = np.vstack([result.F, seed_F])
    nds = NonDominatedSorting()
    fronts = nds.do(union_F, only_non_dominated_front=True)
    pareto_X, pareto_F = union_X[fronts], union_F[fronts]

    # Deduplicate: elite preservation can bring in seeds identical to existing
    # result points, producing duplicate solutions after non-dominated sorting.
    # Keep the first occurrence of each unique X row.
    _, unique_idx = np.unique(
        np.round(pareto_X, 6),  # tolerate small float noise
        axis=0,
        return_index=True,
    )
    unique_idx.sort()  # preserve original order
    return pareto_X[unique_idx], pareto_F[unique_idx]


def _optimize_binary(
    problem: Problem,
    mode: OptimizeMode,
    max_solutions: int | None = None,
    seed: int = 42,
    time_limit: float | None = None,
) -> Run:
    """Binary mode: pick K of N options."""
    n_options = len(problem.options)
    opt_names = [o.name for o in problem.options]
    obj_list = problem.objectives
    score_matrix = _build_score_matrix(problem)
    cp = _parse_constraints(problem)
    im = _build_interaction_matrices(problem)

    pymoo_problem = _FrontierProblem(
        n_options=n_options, score_matrix=score_matrix, objectives=obj_list,
        interaction_matrices=im, **cp,
    )

    seeds = _compute_extreme_seeds(problem, score_matrix, cp)
    algorithm, n_gen = _tune_parameters(problem, mode, seed_population=seeds if len(seeds) > 0 else None)

    with _seeded_rng_fallback(seed):  # make pymoo's unseeded survival/selection RNG deterministic
        t0 = time.monotonic()
        result = pymoo_minimize(
            pymoo_problem, algorithm, _build_termination(n_gen, time_limit), seed=seed, verbose=False,
        )
        elapsed = time.monotonic() - t0

    # Elite preservation: union extreme-point seeds with result, keep Pareto front.
    union_X, union_F = _union_with_elites(result, pymoo_problem, seeds if len(seeds) > 0 else None)

    solutions: list[Solution] = []
    if union_F is not None and len(union_F) > 0:
        for idx, (x, f) in enumerate(zip(union_X, union_F)):
            selected = [opt_names[i] for i in range(n_options) if x[i] > 0.5]
            obj_values = {}
            for j, obj in enumerate(obj_list):
                val = -f[j] if obj.direction.value == "maximize" else f[j]
                obj_values[obj.name] = round(float(val), 4)
            solutions.append(Solution(
                solution_id=idx, selected_options=selected, objective_values=obj_values,
            ))

    max_n = max_solutions or MAX_PARETO_SOLUTIONS
    solutions, total_found = _prune_pareto(solutions, obj_list, max_n=max_n)
    solutions = _sort_and_reindex(solutions, obj_list)
    return Run(
        solutions=solutions,
        total_pareto_found=total_found,
        quality=_compute_quality(result, seed=seed),
        mode=mode,
        seed_used=seed,
        time_limit=time_limit,
        time_limited=_was_time_limited(time_limit, elapsed),
    )


def _optimize_proportional(
    problem: Problem,
    mode: OptimizeMode,
    max_solutions: int | None = None,
    seed: int = 42,
    time_limit: float | None = None,
) -> Run:
    """Proportional mode: allocate integer percentages (0-100, sum to 100)."""
    n_options = len(problem.options)
    opt_names = [o.name for o in problem.options]
    obj_list = problem.objectives
    score_matrix = _build_score_matrix(problem)
    cp = _parse_constraints(problem)
    im = _build_interaction_matrices(problem)

    pymoo_problem = _ProportionalProblem(
        n_options=n_options, score_matrix=score_matrix, objectives=obj_list,
        interaction_matrices=im, **cp,
    )

    seeds = _compute_extreme_seeds(problem, score_matrix, cp)
    algorithm, n_gen = _tune_parameters(problem, mode, seed_population=seeds if len(seeds) > 0 else None)

    with _seeded_rng_fallback(seed):  # make pymoo's unseeded survival/selection RNG deterministic
        t0 = time.monotonic()
        result = pymoo_minimize(
            pymoo_problem, algorithm, _build_termination(n_gen, time_limit), seed=seed, verbose=False,
        )
        elapsed = time.monotonic() - t0

    # Elite preservation: union extreme-point seeds with result, keep Pareto front.
    union_X, union_F = _union_with_elites(result, pymoo_problem, seeds if len(seeds) > 0 else None)

    solutions: list[Solution] = []
    if union_F is not None and len(union_F) > 0:
        for idx, (x, f) in enumerate(zip(union_X, union_F)):
            # Round to integers and normalize to sum to 100
            raw = np.maximum(np.round(x), 0).astype(int)
            # Clamp to max_allocation before redistributing
            if cp.get("max_allocation") is not None:
                raw = np.minimum(raw, cp["max_allocation"])
            # Redistribute rounding error across eligible variables
            diff = 100 - raw.sum()
            if diff != 0:
                cap = cp.get("max_allocation") or 100
                if diff > 0:
                    # Need to add: distribute to variables with most headroom
                    for _ in range(abs(diff)):
                        headroom = cap - raw
                        headroom[raw == 0] = 0  # don't create new holdings here
                        if headroom.max() <= 0:
                            break
                        raw[np.argmax(headroom)] += 1
                else:
                    # Need to remove: take from largest allocations
                    for _ in range(abs(diff)):
                        if raw.max() <= 0:
                            break
                        raw[np.argmax(raw)] -= 1

            allocs = {opt_names[i]: int(raw[i]) for i in range(n_options)}
            selected = [opt_names[i] for i in range(n_options) if raw[i] > 0]

            obj_values = {}
            for j, obj in enumerate(obj_list):
                val = -f[j] if obj.direction.value == "maximize" else f[j]
                obj_values[obj.name] = round(float(val), 4)

            solutions.append(Solution(
                solution_id=idx,
                selected_options=selected,
                objective_values=obj_values,
                allocations=allocs,
            ))

    max_n = max_solutions or MAX_PARETO_SOLUTIONS
    solutions, total_found = _prune_pareto(solutions, obj_list, max_n=max_n)
    solutions = _sort_and_reindex(solutions, obj_list)
    return Run(
        solutions=solutions,
        total_pareto_found=total_found,
        quality=_compute_quality(result, seed=seed),
        mode=mode,
        seed_used=seed,
        time_limit=time_limit,
        time_limited=_was_time_limited(time_limit, elapsed),
    )


MAX_PARETO_SOLUTIONS = 1000  # Safety valve; pymoo pop_size bounds Pareto size in practice.


def _prune_pareto(solutions: list[Solution], obj_list: list, max_n: int = MAX_PARETO_SOLUTIONS) -> tuple[list[Solution], int]:
    """Prune a Pareto front to max_n solutions, preserving extremes + even spacing.

    Returns (pruned_solutions, total_found_before_pruning).
    """
    total = len(solutions)
    if total <= max_n:
        return solutions, total

    obj_names = [o.name for o in obj_list]

    # Always keep extreme solutions (best per objective)
    keep_indices: set[int] = set()
    for obj in obj_list:
        name = obj.name
        if obj.direction.value == "maximize":
            best_idx = max(range(total), key=lambda i: solutions[i].objective_values.get(name, 0))
        else:
            best_idx = min(range(total), key=lambda i: solutions[i].objective_values.get(name, 0))
        keep_indices.add(best_idx)

    # Fill remaining slots with farthest-point sampling in normalized objective space
    remaining = max_n - len(keep_indices)
    if remaining > 0 and len(obj_names) > 0:
        matrix = np.array([
            [s.objective_values.get(name, 0) for name in obj_names]
            for s in solutions
        ])
        # Normalize to [0, 1]
        col_min = matrix.min(axis=0)
        col_max = matrix.max(axis=0)
        spread = col_max - col_min
        spread[spread == 0] = 1.0
        norm = (matrix - col_min) / spread

        # Greedy farthest-point sampling from the kept set
        available = set(range(total)) - keep_indices
        kept_list = list(keep_indices)

        for _ in range(remaining):
            if not available:
                break
            # For each available point, find min distance to any kept point
            best_candidate, best_min_dist = -1, -1.0
            kept_points = norm[kept_list]
            for idx in available:
                dists = np.linalg.norm(kept_points - norm[idx], axis=1)
                min_dist = dists.min()
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = idx
            if best_candidate >= 0:
                keep_indices.add(best_candidate)
                kept_list.append(best_candidate)
                available.discard(best_candidate)

    pruned = [solutions[i] for i in sorted(keep_indices)]
    return pruned, total


def _sort_and_reindex(solutions: list[Solution], obj_list: list) -> list[Solution]:
    """Sort solutions by first objective, reindex, and stamp content signatures."""
    if solutions and obj_list:
        first_obj = obj_list[0].name
        reverse = obj_list[0].direction.value == "maximize"
        solutions.sort(key=lambda s: s.objective_values.get(first_obj, 0), reverse=reverse)
        for i, s in enumerate(solutions, start=1):
            s.solution_id = i
            s.content_signature = _content_signature(s.selected_options, s.allocations)
    return solutions


class _FrontierProblem(PymooProblem):
    """pymoo problem wrapper for binary portfolio selection."""

    def __init__(
        self,
        n_options: int,
        score_matrix: np.ndarray,
        objectives: list,
        forced_in: set[int],
        forced_out: set[int],
        cardinality_min: int,
        cardinality_max: int,
        obj_bounds: list[tuple[int, BoundOperator, float]],
        exclusion_pairs: list[tuple[int, int]] | None = None,
        dependencies: list[tuple[int, int]] | None = None,
        group_limits: list[tuple[list[int], int]] | None = None,
        max_allocation: int | None = None,  # ignored for binary mode
        interaction_matrices: dict[int, np.ndarray] | None = None,
    ):
        exclusion_pairs = exclusion_pairs or []
        dependencies = dependencies or []
        group_limits = group_limits or []

        n_ieq = 0
        n_ieq += 2  # cardinality min and max
        n_ieq += len(forced_in)
        n_ieq += len(forced_out)
        n_ieq += len(obj_bounds)
        n_ieq += len(exclusion_pairs)  # x[a] + x[b] <= 1
        n_ieq += len(dependencies)     # x[a] - x[b] <= 0
        n_ieq += len(group_limits)     # sum(group) <= max

        super().__init__(
            n_var=n_options,
            n_obj=len(objectives),
            n_ieq_constr=n_ieq,
            xl=0,
            xu=1,
        )
        self.score_matrix = score_matrix
        self.objectives = objectives
        self.forced_in = forced_in
        self.forced_out = forced_out
        self.cardinality_min = cardinality_min
        self.cardinality_max = cardinality_max
        self.obj_bounds = obj_bounds
        self.exclusion_pairs = exclusion_pairs
        self.dependencies = dependencies
        self.group_limits = group_limits
        self.interaction_matrices = interaction_matrices or {}

    def _aggregate_objective(self, X_bin: np.ndarray, j: int) -> np.ndarray:
        """Compute objective j values for all population members using the correct aggregation."""
        agg = self.objectives[j].aggregation
        n_pop = X_bin.shape[0]

        if agg == Aggregation.sum:
            return X_bin @ self.score_matrix[:, j]
        elif agg == Aggregation.avg:
            sums = X_bin @ self.score_matrix[:, j]
            counts = np.maximum(X_bin.sum(axis=1), 1.0)  # avoid div by zero
            return sums / counts
        elif agg == Aggregation.min:
            vals = np.full(n_pop, np.inf)
            for i in range(n_pop):
                selected = X_bin[i] > 0.5
                if selected.any():
                    vals[i] = self.score_matrix[selected, j].min()
                else:
                    vals[i] = 0.0
            return vals
        elif agg == Aggregation.max:
            vals = np.full(n_pop, -np.inf)
            for i in range(n_pop):
                selected = X_bin[i] > 0.5
                if selected.any():
                    vals[i] = self.score_matrix[selected, j].max()
                else:
                    vals[i] = 0.0
            return vals
        elif agg == Aggregation.quadratic:
            M = self.interaction_matrices[j]
            vals = np.zeros(n_pop)
            for i in range(n_pop):
                selected = X_bin[i] > 0.5
                n_sel = selected.sum()
                if n_sel > 0:
                    w = np.zeros(X_bin.shape[1])
                    w[selected] = 1.0 / n_sel  # equal weight
                    vals[i] = np.sqrt(max(w @ M @ w, 0.0))
            return vals
        else:
            return X_bin @ self.score_matrix[:, j]  # fallback to sum

    def _evaluate(self, X, out, *args, **kwargs):
        # Round to binary
        X_bin = (X > 0.5).astype(float)
        n_pop = X_bin.shape[0]
        n_obj = len(self.objectives)

        # Objectives: compute per-objective using aggregation method
        F = np.zeros((n_pop, n_obj))
        for j in range(n_obj):
            F[:, j] = self._aggregate_objective(X_bin, j)

        # pymoo minimizes — negate maximize objectives
        for j, obj in enumerate(self.objectives):
            if obj.direction.value == "maximize":
                F[:, j] = -F[:, j]

        out["F"] = F

        # Constraints: g(x) <= 0 means feasible
        G = []

        # Cardinality: min <= sum(x) <= max
        counts = X_bin.sum(axis=1)
        G.append(self.cardinality_min - counts)  # min - count <= 0
        G.append(counts - self.cardinality_max)  # count - max <= 0

        # Force include: x[i] >= 1 → 1 - x[i] <= 0
        for i in self.forced_in:
            G.append(1.0 - X_bin[:, i])

        # Force exclude: x[i] <= 0 → x[i] <= 0
        for i in self.forced_out:
            G.append(X_bin[:, i])

        # Objective bounds — use same aggregation as the objective
        for obj_idx, operator, value in self.obj_bounds:
            obj_vals = self._aggregate_objective(X_bin, obj_idx)
            if operator == BoundOperator.max:
                G.append(obj_vals - value)
            else:
                G.append(value - obj_vals)

        # Exclusion pairs: x[a] + x[b] <= 1
        for a, b in self.exclusion_pairs:
            G.append(X_bin[:, a] + X_bin[:, b] - 1.0)

        # Dependencies: if x[a] then x[b] → x[a] - x[b] <= 0
        for if_idx, then_idx in self.dependencies:
            G.append(X_bin[:, if_idx] - X_bin[:, then_idx])

        # Group limits: sum(x[i] for i in group) <= max
        for group_indices, max_count in self.group_limits:
            group_sum = sum(X_bin[:, i] for i in group_indices)
            G.append(group_sum - max_count)

        out["G"] = np.column_stack(G) if G else np.zeros((n_pop, 0))


class _ProportionalProblem(PymooProblem):
    """pymoo problem wrapper for proportional allocation (0-100 percentages)."""

    def __init__(
        self,
        n_options: int,
        score_matrix: np.ndarray,
        objectives: list,
        forced_in: set[int],
        forced_out: set[int],
        cardinality_min: int,
        cardinality_max: int,
        obj_bounds: list[tuple[int, BoundOperator, float]],
        exclusion_pairs: list[tuple[int, int]] | None = None,
        dependencies: list[tuple[int, int]] | None = None,
        group_limits: list[tuple[list[int], int]] | None = None,
        max_allocation: int | None = None,
        interaction_matrices: dict[int, np.ndarray] | None = None,
    ):
        exclusion_pairs = exclusion_pairs or []
        dependencies = dependencies or []
        group_limits = group_limits or []

        n_ieq = 2 + 2 + len(forced_in) + len(forced_out) + len(obj_bounds)
        n_ieq += len(exclusion_pairs) + len(dependencies) + len(group_limits)
        if max_allocation is not None:
            n_ieq += n_options  # one constraint per option

        xu = max_allocation if max_allocation is not None else 100

        super().__init__(
            n_var=n_options,
            n_obj=len(objectives),
            n_ieq_constr=n_ieq,
            xl=0,
            xu=xu,
        )
        self.score_matrix = score_matrix
        self.objectives = objectives
        self.forced_in = forced_in
        self.forced_out = forced_out
        self.cardinality_min = cardinality_min
        self.cardinality_max = cardinality_max
        self.obj_bounds = obj_bounds
        self.exclusion_pairs = exclusion_pairs
        self.dependencies = dependencies
        self.group_limits = group_limits
        self.max_allocation = max_allocation
        self.interaction_matrices = interaction_matrices or {}

    def _aggregate_objective(self, X: np.ndarray, j: int) -> np.ndarray:
        """Compute objective j for proportional allocations."""
        agg = self.objectives[j].aggregation
        n_pop = X.shape[0]
        # Fractions: allocation / 100
        fracs = X / 100.0

        if agg == Aggregation.sum:
            # Weighted sum: sum(frac_k * score_k)
            return fracs @ self.score_matrix[:, j]
        elif agg == Aggregation.avg:
            # Weighted average: sum(frac_k * score_k) / sum(frac_k) where frac_k > 0
            weighted_sums = fracs @ self.score_matrix[:, j]
            total_fracs = np.maximum(fracs.sum(axis=1), 0.01)  # avoid div by zero
            return weighted_sums / total_fracs
        elif agg == Aggregation.min:
            vals = np.full(n_pop, np.inf)
            for i in range(n_pop):
                selected = X[i] > 0.5
                if selected.any():
                    vals[i] = self.score_matrix[selected, j].min()
                else:
                    vals[i] = 0.0
            return vals
        elif agg == Aggregation.max:
            vals = np.full(n_pop, -np.inf)
            for i in range(n_pop):
                selected = X[i] > 0.5
                if selected.any():
                    vals[i] = self.score_matrix[selected, j].max()
                else:
                    vals[i] = 0.0
            return vals
        elif agg == Aggregation.quadratic:
            M = self.interaction_matrices[j]
            # sqrt(w^T M w) — vectorized via einsum
            Mw = fracs @ M  # (n_pop, n_var)
            raw = np.einsum('ij,ij->i', Mw, fracs)
            return np.sqrt(np.maximum(raw, 0.0))  # clamp for numerical safety
        else:
            return fracs @ self.score_matrix[:, j]

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        n_obj = len(self.objectives)

        # Objectives
        F = np.zeros((n_pop, n_obj))
        for j in range(n_obj):
            F[:, j] = self._aggregate_objective(X, j)

        for j, obj in enumerate(self.objectives):
            if obj.direction.value == "maximize":
                F[:, j] = -F[:, j]

        out["F"] = F

        # Constraints
        G = []

        # Sum to 100: |sum - 100| <= 0 via two inequalities
        sums = X.sum(axis=1)
        G.append(sums - 100.0)   # sum <= 100
        G.append(100.0 - sums)   # sum >= 100

        # Cardinality: count of non-zero allocations
        allocated_count = (X > 0.5).sum(axis=1).astype(float)
        G.append(self.cardinality_min - allocated_count)
        G.append(allocated_count - self.cardinality_max)

        # Force include: allocation must be > 0 (use threshold 0.5)
        for i in self.forced_in:
            G.append(0.5 - X[:, i])

        # Force exclude: allocation must be 0 (use threshold 0.5)
        for i in self.forced_out:
            G.append(X[:, i] - 0.5)

        # Objective bounds
        for obj_idx, operator, value in self.obj_bounds:
            obj_vals = self._aggregate_objective(X, obj_idx)
            if operator == BoundOperator.max:
                G.append(obj_vals - value)
            else:
                G.append(value - obj_vals)

        # Exclusion pairs: can't allocate to both (both > 0.5)
        allocated = (X > 0.5).astype(float)
        for a, b in self.exclusion_pairs:
            G.append(allocated[:, a] + allocated[:, b] - 1.0)

        # Dependencies: if allocated to a, must allocate to b
        for if_idx, then_idx in self.dependencies:
            G.append(allocated[:, if_idx] - allocated[:, then_idx])

        # Group limits: count of allocated in group <= max
        for group_indices, max_count in self.group_limits:
            group_sum = sum(allocated[:, i] for i in group_indices)
            G.append(group_sum - max_count)

        # Max position: each allocation <= max_allocation
        if self.max_allocation is not None:
            for i in range(X.shape[1]):
                G.append(X[:, i] - self.max_allocation)

        out["G"] = np.column_stack(G) if G else np.zeros((n_pop, 0))


def analyze_infeasibility(problem: Problem) -> dict:
    """Diagnose which constraints are most likely causing infeasibility.

    Tests each constraint by relaxing it individually and checking if
    feasible solutions exist via brute-force enumeration of small subsets.
    """
    from itertools import combinations

    opt_names = [o.name for o in problem.options]
    obj_list = problem.objectives
    n_options = len(opt_names)

    # Build score lookup
    score_map: dict[tuple[str, str], float] = {}
    for s in problem.scores:
        score_map[(s.option, s.objective)] = s.value

    # Build aggregation map from objectives
    agg_map = {o.name: o.aggregation for o in problem.objectives}

    # Parse constraints
    forced_in = {c.option for c in problem.constraints if c.type == "force_include"}
    forced_out = {c.option for c in problem.constraints if c.type == "force_exclude"}
    exclusion_pairs = [(c.option_a, c.option_b) for c in problem.constraints if c.type == "exclusion_pair"]
    dependencies = [(c.if_option, c.then_option) for c in problem.constraints if c.type == "dependency"]
    group_limits = [(c.options, c.max) for c in problem.constraints if c.type == "group_limit"]
    card_min, card_max = 1, n_options
    obj_bounds = []
    for c in problem.constraints:
        if c.type == "cardinality":
            card_min, card_max = c.min, c.max
        elif c.type == "objective_bound":
            obj_bounds.append((c.objective, c.operator, c.value))

    # Build interaction matrix lookup for quadratic objectives
    im_map = {m.objective: m for m in problem.interaction_matrices}

    def _aggregate_for_portfolio(selected: set[str], obj_name: str) -> float:
        """Compute aggregated objective value for a portfolio, respecting aggregation mode."""
        scores = [score_map.get((o, obj_name), 0) for o in selected]
        if not scores:
            return 0.0
        agg = agg_map.get(obj_name, Aggregation.sum)
        if agg == Aggregation.sum:
            return sum(scores)
        elif agg == Aggregation.avg:
            return sum(scores) / len(scores)
        elif agg == Aggregation.min:
            return min(scores)
        elif agg == Aggregation.max:
            return max(scores)
        elif agg == Aggregation.quadratic:
            if obj_name not in im_map:
                return sum(scores) / len(scores)  # fallback
            im = im_map[obj_name]
            sel = list(selected)
            n = len(sel)
            w = 1.0 / n
            total = 0.0
            for a in sel:
                for b in sel:
                    total += w * w * im.entries.get(a, {}).get(b, 0.0)
            return total ** 0.5 if total > 0 else 0.0
        return sum(scores)

    def portfolio_feasible(selected: set[str], skip_constraint=None) -> bool:
        """Check if a portfolio satisfies all constraints except skip_constraint."""
        # Cardinality
        if skip_constraint != "cardinality":
            if not (card_min <= len(selected) <= card_max):
                return False
        # Force include
        for opt in forced_in:
            if skip_constraint == f"force_include:{opt}":
                continue
            if opt not in selected:
                return False
        # Force exclude
        for opt in forced_out:
            if skip_constraint == f"force_exclude:{opt}":
                continue
            if opt in selected:
                return False
        # Objective bounds
        for obj_name, operator, value in obj_bounds:
            if skip_constraint == f"objective_bound:{obj_name}:{operator}:{value}":
                continue
            agg_val = _aggregate_for_portfolio(selected, obj_name)
            if operator.value == "max" and agg_val > value:
                return False
            if operator.value == "min" and agg_val < value:
                return False
        # Exclusion pairs
        for opt_a, opt_b in exclusion_pairs:
            if skip_constraint == f"exclusion_pair:{opt_a}:{opt_b}":
                continue
            if opt_a in selected and opt_b in selected:
                return False
        # Dependencies
        for if_opt, then_opt in dependencies:
            if skip_constraint == f"dependency:{if_opt}:{then_opt}":
                continue
            if if_opt in selected and then_opt not in selected:
                return False
        # Group limits
        for group_opts, max_count in group_limits:
            if skip_constraint == f"group_limit:{','.join(group_opts)}:{max_count}":
                continue
            if sum(1 for o in group_opts if o in selected) > max_count:
                return False
        return True

    # Available options (respecting force_exclude for enumeration)
    available = [o for o in opt_names if o not in forced_out]

    # Try to find any feasible portfolio (brute force for small problems)
    # Limit enumeration to avoid combinatorial explosion
    max_combos = 50000

    def has_feasible(skip_constraint=None) -> bool:
        pool = available if skip_constraint is None or not skip_constraint.startswith("force_exclude:") else opt_names
        for k in range(max(1, card_min), min(len(pool), card_max) + 1):
            count = 0
            for combo in combinations(pool, k):
                if portfolio_feasible(set(combo), skip_constraint):
                    return True
                count += 1
                if count >= max_combos:
                    return False  # can't tell, assume not
        return False

    # Build constraint labels for testing
    constraint_labels = []
    for c in problem.constraints:
        if c.type == "cardinality":
            constraint_labels.append(("cardinality", c.model_dump()))
        elif c.type == "force_include":
            constraint_labels.append((f"force_include:{c.option}", c.model_dump()))
        elif c.type == "force_exclude":
            constraint_labels.append((f"force_exclude:{c.option}", c.model_dump()))
        elif c.type == "objective_bound":
            constraint_labels.append(
                (f"objective_bound:{c.objective}:{c.operator}:{c.value}", c.model_dump())
            )
        elif c.type == "exclusion_pair":
            constraint_labels.append(
                (f"exclusion_pair:{c.option_a}:{c.option_b}", c.model_dump())
            )
        elif c.type == "dependency":
            constraint_labels.append(
                (f"dependency:{c.if_option}:{c.then_option}", c.model_dump())
            )
        elif c.type == "group_limit":
            constraint_labels.append(
                (f"group_limit:{','.join(c.options)}:{c.max}", c.model_dump())
            )
        elif c.type == "max_allocation":
            # max_allocation is proportional-only; skip in binary feasibility analysis
            pass

    # Test each constraint: does removing it make the problem feasible?
    binding = []
    suggestions = []
    for label, constraint_dict in constraint_labels:
        if has_feasible(skip_constraint=label):
            binding.append(constraint_dict)
            if label == "cardinality":
                suggestions.append(f"Relaxing cardinality ({card_min}-{card_max}) may help.")
            elif label.startswith("force_include:"):
                opt = label.split(":", 1)[1]
                suggestions.append(f"Removing force_include on '{opt}' may help.")
            elif label.startswith("force_exclude:"):
                opt = label.split(":", 1)[1]
                suggestions.append(f"Removing force_exclude on '{opt}' may help.")
            elif label.startswith("objective_bound:"):
                parts = label.split(":")
                suggestions.append(f"Relaxing {parts[2]} bound on '{parts[1]}' ({parts[3]}) may help.")
            elif label.startswith("exclusion_pair:"):
                parts = label.split(":")
                suggestions.append(f"Removing exclusion between '{parts[1]}' and '{parts[2]}' may help.")
            elif label.startswith("dependency:"):
                parts = label.split(":")
                suggestions.append(f"Removing dependency '{parts[1]}' → '{parts[2]}' may help.")
            elif label.startswith("group_limit:"):
                suggestions.append(f"Relaxing group limit may help.")

    if not binding:
        binding = [c.model_dump() for c in problem.constraints]
        suggestions = ["Constraints may be jointly infeasible. Try relaxing multiple constraints."]

    return {"binding_constraints": binding, "suggestions": suggestions}


def _compute_quality(result, seed: int = 42) -> QualityIndicators:
    """Compute quality indicators from pymoo result."""
    if result.F is None or len(result.F) < 2:
        return QualityIndicators()

    F = result.F

    # Normalized hypervolume (approximate): fraction of bounding box dominated
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    spread = f_max - f_min
    if np.all(spread > 0):
        F_norm = (F - f_min) / spread
        # Simple hypervolume approximation using reference point at (1.1, ..., 1.1)
        ref = np.ones(F.shape[1]) * 1.1
        hv = _approx_hypervolume(F_norm, ref, seed=seed)
        hv_max = np.prod(ref)
        hv_normalized = round(float(hv / hv_max), 4) if hv_max > 0 else None
    else:
        hv_normalized = None

    # Spacing CV: coefficient of variation of nearest-neighbor distances
    if len(F) >= 3:
        from scipy.spatial.distance import cdist
        dists = cdist(F, F)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)
        mean_d = nn_dists.mean()
        spacing_cv = round(float(nn_dists.std() / mean_d), 4) if mean_d > 0 else None
    else:
        spacing_cv = None

    return QualityIndicators(
        hypervolume_normalized=hv_normalized,
        spacing_cv=spacing_cv,
    )


def _approx_hypervolume(F: np.ndarray, ref: np.ndarray, seed: int = 42) -> float:
    """Monte Carlo hypervolume approximation for small dimensions."""
    n_samples = 10000
    n_obj = F.shape[1]
    rng = np.random.default_rng(seed)

    # Sample random points in the bounding box [0, ref]
    samples = rng.uniform(0, ref, size=(n_samples, n_obj))

    # A sample is dominated if any Pareto point dominates it
    dominated = np.zeros(n_samples, dtype=bool)
    for f in F:
        dominated |= np.all(samples >= f, axis=1)

    vol_box = np.prod(ref)
    return float(dominated.sum() / n_samples * vol_box)
