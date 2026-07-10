"""Optimizer: validate problem and run NSGA-II/NSGA-III via pymoo."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.duplicate import DuplicateElimination
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


class _BinaryHashDuplicateElimination(DuplicateElimination):
    """Exact duplicate elimination for 0/1 genomes via a byte-packed hash set.

    pymoo's ``DefaultDuplicateElimination`` computes O(pop² × n_var) float distance
    matrices every generation — at industrial scale (n_var in the thousands) that is
    nearly all of solve wall-clock. For binary variables exact equality IS the
    default's duplicate semantics (bit vectors within its 1e-16 epsilon are
    identical), so a hash set over ``np.packbits`` rows gives the same result in
    O(pop × n_var). Keep-first within a population (the default's upper-triangle
    rule) and mark-against-``other`` only, so fronts are bit-identical to the
    default's at a fixed seed. Binary path only — continuous genomes need the
    distance-based default.
    """

    def _keys(self, pop) -> list[bytes]:
        # Operators may hand back bool or 0.0/1.0 float genomes; both pack the same.
        X = np.asarray(self.func(pop))
        return [row.tobytes() for row in np.packbits(X.astype(bool), axis=1)]

    def _do(self, pop, other, is_duplicate):
        seen = set() if other is None else set(self._keys(other))
        for i, key in enumerate(self._keys(pop)):
            if key in seen:
                is_duplicate[i] = True
            elif other is None:
                seen.add(key)
        return is_duplicate


class _SimplexRepair(Repair):
    """After crossover/mutation, project solutions back onto the simplex (sum=100, non-negative).

    When the pymoo problem carries a max_allocation cap and/or per-option allocation bounds
    (``alloc_lo``/``alloc_hi`` percent vectors), iteratively clamp to the box and redistribute
    so the row still sums to 100. If the iteration doesn't converge (tight boxes), fall back
    to the deterministic feasible point lo + slack ∝ (hi − lo), which satisfies the box and
    the budget whenever the bounds are jointly feasible (Σlo ≤ 100 ≤ Σhi — validate enforces).
    """

    def _do(self, problem, X, **kwargs):
        # Clamp to non-negative
        X = np.maximum(X, 0.0)
        # Normalize each row to sum to 100
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-9)  # avoid div by zero
        X = X / row_sums * 100.0

        lo = getattr(problem, "alloc_lo", None)
        hi = getattr(problem, "alloc_hi", None)
        cap = getattr(problem, "max_allocation", None)
        if lo is None or hi is None or not getattr(problem, "allocation_bounds", None):
            # Global-cap-only path (the pre-allocation_bound behavior, unchanged).
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

        # Box-constrained path: clamp to [lo, hi] and redistribute the budget residual onto
        # options with slack, both directions (floors pull mass in, caps push it out).
        for _ in range(8):
            X = np.clip(X, lo, hi)
            resid = 100.0 - X.sum(axis=1, keepdims=True)     # >0: need to add; <0: remove
            if np.all(np.abs(resid) < 0.01):
                return X
            up_slack = np.maximum(hi - X, 0.0)
            dn_slack = np.maximum(X - lo, 0.0)
            up_total = np.maximum(up_slack.sum(axis=1, keepdims=True), 1e-9)
            dn_total = np.maximum(dn_slack.sum(axis=1, keepdims=True), 1e-9)
            add = np.where(resid > 0, up_slack / up_total * resid, 0.0)
            sub = np.where(resid < 0, dn_slack / dn_total * (-resid), 0.0)
            X = X + add - sub
        # Deterministic feasible fallback for rows that didn't converge.
        bad = np.abs(100.0 - np.clip(X, lo, hi).sum(axis=1)) >= 0.01
        if bad.any():
            span = np.maximum((hi - lo).sum(), 1e-9)
            fallback = lo + (hi - lo) * ((100.0 - lo.sum()) / span)
            X[bad] = fallback
        return np.clip(X, lo, hi)

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


# Upper bounds are an anti-abuse / accidental-blowup backstop, NOT a usability limit.
# Real problems (large asset universes, big project portfolios) can run to many
# thousands of options — far past the bundled examples (≤120) — and a large problem
# taking a long time is fine (that's what time_limit + background jobs are for). So
# these defaults sit absurdly high: they only catch a runaway/garbage payload (e.g. a
# millions-of-options memory/storage bomb), never a real decision. Raise via env if a
# genuinely larger deployment ever needs it.
_MAX_OPTIONS = int(os.environ.get("FRONTIER_MAX_OPTIONS", "100000"))
_MAX_OBJECTIVES = int(os.environ.get("FRONTIER_MAX_OBJECTIVES", "50"))
_MAX_SCORES = int(os.environ.get("FRONTIER_MAX_SCORES", "10000000"))


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

    # Upper bounds — refuse pathologically large problems before the solver allocates
    # population/genome arrays that scale with these counts. Generous backstops, not a
    # usability limit (raise via FRONTIER_MAX_OPTIONS / _OBJECTIVES / _SCORES if needed).
    if len(problem.options) > _MAX_OPTIONS:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Too many options ({len(problem.options)}); max is {_MAX_OPTIONS} "
                    f"(raise FRONTIER_MAX_OPTIONS for a larger deployment).",
        ))
    if len(problem.objectives) > _MAX_OBJECTIVES:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Too many objectives ({len(problem.objectives)}); max is {_MAX_OBJECTIVES} "
                    f"(raise FRONTIER_MAX_OBJECTIVES for a larger deployment).",
        ))
    if len(problem.scores) > _MAX_SCORES:
        issues.append(ValidationIssue(
            severity="error",
            message=f"Too many scores ({len(problem.scores)}); max is {_MAX_SCORES} "
                    f"(raise FRONTIER_MAX_SCORES for a larger deployment).",
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
            if c.min < 0 or c.min > c.max:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"group_limit min ({c.min}) must be between 0 and max ({c.max}).",
                ))
            # min vs the group's selectable size is checked in _check_constraint_conflicts
            # (it accounts for force_excludes), so it isn't duplicated here.
        elif c.type == "allocation_bound":
            if c.option not in opt_names:
                issues.append(ValidationIssue(
                    severity="error",
                    message=f"allocation_bound references unknown option '{c.option}'.",
                ))
            if not (0 <= c.min <= c.max <= 100):
                issues.append(ValidationIssue(
                    severity="error",
                    message=(f"allocation_bound for '{c.option}' needs 0 <= min <= max <= 100, "
                             f"got min={c.min}, max={c.max}."),
                ))
            if problem.approach != Approach.proportional:
                issues.append(ValidationIssue(
                    severity="warning",
                    message=("allocation_bound constraint only applies to proportional mode; "
                             "ignored in binary mode."),
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
      A. GroupLimit vs force_include — |force_in ∩ group| > group.max; and floor arithmetic —
         min > selectable members after force_excludes, or disjoint floors summing past the
         cardinality max
      B. ExclusionPair vs force_include — both members force-included
      C. Dependency cycles — A→B→…→A in the dependency graph
      D. Dependency vs force_exclude — if-option forced in but then-option forced out
      E. MaxAllocation arithmetic (proportional) — cap × available_options < 100
    """
    issues: list[ValidationIssue] = []

    # A. GroupLimit vs force_include — and floor arithmetic (min vs force_exclude / cardinality)
    group_floors: list = []
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
            if c.min > 0:
                group_floors.append(c)
                selectable = [o for o in c.options if o not in forced_out]
                if c.min > len(selectable):
                    issues.append(ValidationIssue(
                        severity="error",
                        message=(
                            f"group_limit min ({c.min}) exceeds the group's selectable members "
                            f"({len(selectable)} after force_excludes) — the floor can never be met."
                        ),
                    ))
    # Floors vs the cardinality cap: sound only over pairwise-disjoint floored groups
    # (overlapping groups would double-count), so overlapping floors are left to the solver.
    if group_floors:
        card_max = next((c.max for c in problem.constraints if c.type == "cardinality"), None)
        all_disjoint = all(
            not set(a.options) & set(b.options)
            for i, a in enumerate(group_floors) for b in group_floors[i + 1:]
        )
        if card_max is not None and all_disjoint:
            floor_sum = sum(c.min for c in group_floors)
            if floor_sum > card_max:
                issues.append(ValidationIssue(
                    severity="error",
                    message=(
                        f"group_limit floors sum to {floor_sum}, above the cardinality max "
                        f"({card_max}) — the floors cannot all be met."
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

    # E. MaxAllocation / allocation_bound arithmetic (proportional mode only)
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
        ab = [c for c in problem.constraints if c.type == "allocation_bound"]
        if ab:
            floor_sum = sum(c.min for c in ab)
            if floor_sum > 100:
                issues.append(ValidationIssue(
                    severity="error",
                    message=(f"allocation_bound floors sum to {floor_sum}% > 100% — "
                             "the floors cannot all be met."),
                ))
            global_cap_for_floors = next(
                (c.max for c in problem.constraints if c.type == "max_allocation"), 100)
            for c in ab:
                eff_cap = min(global_cap_for_floors, c.max)
                if c.min > eff_cap:
                    issues.append(ValidationIssue(
                        severity="error",
                        message=(f"allocation_bound floor ({c.min}%) on '{c.option}' exceeds its "
                                 f"effective cap ({eff_cap}% = min of the global max_allocation "
                                 "and its own max) — the bound box is empty."),
                    ))
            for c in ab:
                if c.min > 0 and c.option in forced_out:
                    issues.append(ValidationIssue(
                        severity="error",
                        message=(f"allocation_bound floor ({c.min}%) on '{c.option}' conflicts "
                                 "with force_exclude on the same option."),
                    ))
            global_cap = next((c.max for c in problem.constraints if c.type == "max_allocation"),
                              None)
            bounded = {c.option: c for c in ab}
            cap_sum = sum(
                min(global_cap or 100, bounded[o].max) if o in bounded else (global_cap or 100)
                for o in (opt.name for opt in problem.options) if o not in forced_out
            )
            if cap_sum < 100:
                issues.append(ValidationIssue(
                    severity="error",
                    message=(f"effective per-option caps sum to {cap_sum}% < 100% — "
                             "allocation cannot sum to 100."),
                ))

    return issues


def _parse_constraints(problem: Problem, search_floor: bool = False) -> dict:
    """Extract constraint parameters from problem. Shared by binary and proportional.

    By default the encoding is the model as written — a model with no cardinality
    floor legally permits the empty plan, so feasibility consumers (audit witnesses,
    regret re-evaluation, the exact backends) get honest semantics without opting in.
    The EA entry points pass ``search_floor=True`` to inject their never-propose-empty
    SEARCH default; that preference belongs to the search, not the model."""
    opt_names = [o.name for o in problem.options]
    obj_list = problem.objectives
    opt_index = {name: i for i, name in enumerate(opt_names)}

    forced_in_idx = set()
    forced_out_idx = set()
    cardinality_min = 1 if search_floor else 0
    cardinality_max = len(opt_names)
    obj_bounds: list[tuple[int, BoundOperator, float]] = []
    exclusion_pairs: list[tuple[int, int]] = []
    dependencies: list[tuple[int, int]] = []  # (if_idx, then_idx)
    group_limits: list[tuple[list[int], int, int]] = []  # (group_indices, min, max)
    max_allocation: int | None = None  # max allocation % per option (proportional only)
    allocation_bounds: dict[int, tuple[int, int]] = {}  # idx -> (min%, max%) (proportional only)

    def _opt_idx(name: str, c) -> int:
        # A stored constraint can outlive an option rename/removal (validation flags the
        # mismatch but persists the model) — decline in words instead of a raw KeyError
        # deep inside audit/sensitivity/solve consumers.
        if name not in opt_index:
            raise ValueError(f"constraint {c.type} references unknown option '{name}' — "
                             "update or remove that constraint (model update), then retry.")
        return opt_index[name]

    for c in problem.constraints:
        if c.type == "force_include":
            forced_in_idx.add(_opt_idx(c.option, c))
        elif c.type == "force_exclude":
            forced_out_idx.add(_opt_idx(c.option, c))
        elif c.type == "cardinality":
            cardinality_min = c.min
            cardinality_max = c.max
        elif c.type == "objective_bound":
            obj_idx = next((j for j, o in enumerate(obj_list) if o.name == c.objective), None)
            if obj_idx is None:
                raise ValueError(f"objective_bound references unknown objective "
                                 f"'{c.objective}' — update or remove that constraint "
                                 "(model update), then retry.")
            obj_bounds.append((obj_idx, c.operator, c.value))
        elif c.type == "exclusion_pair":
            exclusion_pairs.append((_opt_idx(c.option_a, c), _opt_idx(c.option_b, c)))
        elif c.type == "dependency":
            dependencies.append((_opt_idx(c.if_option, c), _opt_idx(c.then_option, c)))
        elif c.type == "group_limit":
            group_indices = [_opt_idx(o, c) for o in c.options]
            group_limits.append((group_indices, int(c.min), int(c.max)))
        elif c.type == "max_allocation":
            max_allocation = c.max
        elif c.type == "allocation_bound":
            allocation_bounds[_opt_idx(c.option, c)] = (int(c.min), int(c.max))

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
        "allocation_bounds": allocation_bounds,
    }


def _round_weights_to_pct(w_frac: np.ndarray, n_options: int, max_allocation: int | None,
                          bounds_pct: "tuple[np.ndarray, np.ndarray] | None" = None) -> np.ndarray:
    """Round continuous fractional weights → integer percentages summing to 100, matching the
    NSGA proportional path's exact representation. ``bounds_pct`` = per-option integer
    ``(lo, hi)`` vectors (allocation_bound constraints); rounding must not break a floor."""
    x = np.asarray(w_frac, dtype=float) * 100.0
    x = np.where(np.isfinite(x), x, 0.0)  # NaN/inf → 0 before the int cast
    raw = np.maximum(np.round(x), 0).astype(int)
    if max_allocation is not None:
        raw = np.minimum(raw, max_allocation)
    lo = hi = None
    if bounds_pct is not None:
        lo, hi = (np.asarray(v, dtype=int) for v in bounds_pct)
        raw = np.clip(raw, lo, hi)
    diff = 100 - int(raw.sum())
    if diff != 0:
        cap = np.full(n_options, max_allocation or 100, dtype=int)
        if hi is not None:
            cap = np.minimum(cap, hi)
        floor = lo if lo is not None else np.zeros(n_options, dtype=int)
        if diff > 0:
            for _ in range(abs(diff)):
                headroom = cap - raw
                headroom[(raw == 0) & (floor == 0)] = 0  # don't create new holdings here
                if headroom.max() <= 0:
                    break
                raw[np.argmax(headroom)] += 1
        else:
            for _ in range(abs(diff)):
                slack = raw - floor
                if slack.max() <= 0:
                    break
                raw[np.argmax(slack)] -= 1
    return raw

def _build_score_matrix(problem: Problem) -> np.ndarray:
    """Build score_matrix[opt_idx][obj_idx] from problem scores.

    Guards completeness at the root: every analysis path funnels through this build
    (solve, score_slate/regret, audit, viz), so a missing cell fails HERE with the cell
    named instead of a raw KeyError somewhere downstream — e.g. an option added after a
    solve and not yet scored used to crash `explore scenario_results` outright."""
    opt_names = [o.name for o in problem.options]
    obj_list = problem.objectives
    score_map = {(s.option, s.objective): s.value for s in problem.scores}
    missing = [(opt, obj.name) for opt in opt_names for obj in obj_list
               if (opt, obj.name) not in score_map]
    if missing:
        preview = ", ".join(f"{o}×{b}" for o, b in missing[:3])
        raise ValueError(
            f"this analysis needs a complete score matrix — {len(missing)} option×objective "
            f"cell(s) unscored (e.g. {preview}); finish SCORE (model update) first.")
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


def _feasibility_witness_seed(problem: Problem, seeds: np.ndarray,
                              score_matrix: np.ndarray) -> np.ndarray | None:
    """One exact feasibility witness as a seed row — the audit probe seeding the explorer.

    When objective_bound constraints couple the region (a budget cap plus a target floor), the
    greedy per-objective corner seeds can all be infeasible — they respect forced/cardinality/
    group constraints but not bounds — and the EA can then terminate with an empty feasible
    front on a *provably feasible* problem (seed-dependent). One MILP feasibility witness
    plants the population inside the band; NSGA expands the frontier from there.

    Skipped when any existing corner seed already satisfies every objective bound (the common
    case — the band only needs planting when the greedy corners all miss it), and served by
    the engine-layer ``audit_milp`` probe directly, not ``audit()`` (whose presentation path
    would run a full conflict diagnosis on an infeasible model, only for this helper to
    discard it). Binary problems with HiGHS present only; silently absent otherwise (note:
    this makes the seeded population — and thus the same-seed frontier — depend on whether
    ``highspy`` is installed)."""
    if problem.approach != Approach.binary:
        return None
    bounds = [(c.objective, getattr(c.operator, "value", c.operator), float(c.value))
              for c in (problem.constraints or []) if c.type == "objective_bound"]
    if not bounds:
        return None
    ocol = {o.name: j for j, o in enumerate(problem.objectives)}
    if len(seeds) > 0:
        vals = np.asarray(seeds, dtype=float) @ score_matrix   # (n_seeds, n_obj) sums
        for row in vals:
            if all((row[ocol[nm]] <= v + 1e-9) if op == "max" else (row[ocol[nm]] >= v - 1e-9)
                   for nm, op, v in bounds if nm in ocol):
                return None   # a corner seed is already inside the band
    from solvers import available_solvers
    if not available_solvers().get("highs"):
        return None
    # Same bound-encodability condition as audit's gate: bounds enter as linear sum rows.
    if any(o.name in {nm for nm, _, _ in bounds} and o.aggregation != Aggregation.sum
           for o in problem.objectives):
        return None
    from solvers._scalarization import audit_milp
    from solvers.highs_backend import _audit_milp_highs

    raw = audit_milp(problem, [[]], inner_audit=_audit_milp_highs)
    if not raw["feasible"]:
        return None
    selected = set(raw["witness_options"])
    return np.array([[1.0 if o.name in selected else 0.0 for o in problem.options]])


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
        for g_idx, (g_indices, _g_min, _g_max) in enumerate(group_limits):
            group_counts[g_idx] = sum(1 for idx in selected if idx in g_indices)

        # Pre-fill floored groups to their minimum (best-ranked members first) before the
        # greedy fill, so seeds start inside the floor region — the forced_in treatment,
        # extended to group floors.
        for g_idx, (g_indices, g_min, _g_max) in enumerate(group_limits):
            for idx in order:
                if group_counts[g_idx] >= g_min or len(selected) >= card_max:
                    break
                if idx not in g_indices or idx in forced_out or idx in selected:
                    continue
                if any(idx in gi and group_counts[gj] >= gm
                       for gj, (gi, _gn, gm) in enumerate(group_limits) if gj != g_idx):
                    continue
                selected.append(int(idx))
                for gj, (gi, _gn, _gm) in enumerate(group_limits):
                    if idx in gi:
                        group_counts[gj] += 1

        for idx in order:
            if len(selected) >= target_k:
                break
            if idx in forced_out or idx in selected:
                continue
            # Check group caps
            ok = True
            for g_idx, (g_indices, _g_min, g_max) in enumerate(group_limits):
                if idx in g_indices and group_counts[g_idx] >= g_max:
                    ok = False
                    break
            if not ok:
                continue
            selected.append(int(idx))
            for g_idx, (g_indices, _g_min, _g_max) in enumerate(group_limits):
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
            eliminate_duplicates=_BinaryHashDuplicateElimination(),
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


def _nsga_telemetry(result, n_gen: int, elapsed: float) -> dict:
    """``Run.telemetry`` for the NSGA paths: recorded facts about how the solve ran.

    Same record-what-actually-ran convention as ``seed_used``/``solver`` — pure
    instrumentation, no judgment. ``evals_or_solves`` is pymoo's evaluation count
    (captures early termination); ``n_gen_run`` vs ``n_gen_budget`` shows how much
    of the generation budget a time cap left on the table. Attribute access is
    deliberately direct: a pymoo API drift must fail loudly here, not record
    silent zeros into the data the routing thresholds get recalibrated from.
    (Whether a cap cut the run short is ``Run.time_limited`` — not duplicated here.)
    """
    alg = result.algorithm
    return {
        "duration_s": round(elapsed, 3),
        "engine_detail": {
            "pop_size": int(alg.pop_size),
            "n_gen_budget": int(n_gen),
            # pymoo's n_iter counts the initialization pass as iteration 1.
            "n_gen_run": max(0, int(alg.n_iter) - 1),
        },
        "evals_or_solves": int(alg.evaluator.n_eval),
    }


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
            run = _optimize_cuopt(problem, mode, max_solutions=max_solutions, seed=seed,
                                  exact=exact, time_limit=time_limit)
        else:
            from solvers.highs_backend import _optimize_highs
            run = _optimize_highs(problem, mode, max_solutions=max_solutions, seed=seed,
                                  exact=exact, time_limit=time_limit)
    else:
        raise ValueError(f"Unknown solver '{solver}'. Use 'nsga' (default), 'highs', or 'cuopt'.")

    if not backend:
        if problem.approach == Approach.proportional:
            run = _optimize_proportional(problem, mode, max_solutions=max_solutions, seed=seed,
                                         time_limit=time_limit)
        else:
            run = _optimize_binary(problem, mode, max_solutions=max_solutions, seed=seed,
                                   time_limit=time_limit)
        run.solver = "nsga-iii" if len(problem.objectives) >= 4 else "nsga-ii"
    # Provenance for every caller (the MCP layer re-stamps under its staleness guard):
    # snapshot the constraints the run was solved under, so compare_runs never reports a
    # phantom criteria diff against a direct-call run (e.g. a re-baked example bundle).
    run.constraints_snapshot = [c.model_dump() for c in problem.constraints]
    from solvers import problem_features

    run.problem_snapshot = problem_features(problem)
    return run


def certify_curated(
    problem: Problem,
    source_run: Run,
    *,
    solver: str,
    exact: bool = False,
    mode: OptimizeMode | None = None,
    max_solutions: int | None = None,
) -> Run:
    """Progressive certify: produce the exact overlay by exact-solving **only** ``source_run``'s
    frontier points, rather than the full ~pop×gen exact scalarization ``optimize(solver=…)`` runs.

    The lean *explore-then-certify* path — the heuristic (NSGA) explores, the exact solver certifies
    only what it kept — on any supported shape (binary MILP, proportional QP/LP). ``exact`` certifies
    each MILP point to a zero gap (no-op on the always-exact QP/LP). Dispatches to the requested exact
    backend; returns a ``Run`` in the engine's exact shape, stamped with its solver.
    """
    from solvers import exact_solver_fits, is_exact_solver

    if not is_exact_solver(solver):
        raise ValueError(f"certify_curated needs an exact solver ('highs'/'cuopt'), got '{solver}'.")
    fits, reason = exact_solver_fits(problem)
    if not fits:
        raise ValueError(f"Exact solver '{solver}' does not fit this problem: {reason}")
    if solver == "cuopt":
        from solvers.cuopt_backend import _certify_curated_cuopt
        run = _certify_curated_cuopt(problem, source_run, exact=exact, mode=mode, max_solutions=max_solutions)
    else:
        from solvers.highs_backend import _certify_curated_highs
        run = _certify_curated_highs(problem, source_run, exact=exact, mode=mode, max_solutions=max_solutions)
    from solvers import problem_features

    run.problem_snapshot = problem_features(problem)
    return run


def fill_gaps(
    problem: Problem,
    nsga_run: Run,
    exact_run: Run,
    *,
    solver: str,
    exact: bool = False,
    mode: OptimizeMode | None = None,
    max_solutions: int | None = None,
) -> tuple[Run, dict]:
    """Targeted completion of an exact overlay: exact-solve **only the gap-region witnesses**
    (NSGA points the overlay neither dominates nor matches — ``explore certify``'s
    ``completeness`` block) and merge the results into the existing overlay. Returns
    ``(merged overlay Run, fill report)``; inputs are never mutated (merged solutions are
    copies), so a rejected persist can't scramble the caller's overlay. Mechanism is a
    ``certify_curated`` reuse with the witnesses as the source points; design rationale
    lives in architecture.md ("Three ways to produce the `exact_run` overlay").

    Contract with the overlay being extended:

    - **Same backend.** The fill must run the overlay's own solver — a mixed-backend merge
      would stamp every point with one engine's provenance. A mismatch raises.
    - **Same (or stronger) certification.** A zero-gap (``exact=True``) overlay implies
      zero-gap fills: ``exact`` is inherited from the base overlay when not requested, so a
      fill can never silently demote the overlay's certification flag.
    - **The work-list is noise-gated on the certificate's own criterion**
      (``explorer.gap_witness_solutions``), so a fill on a "complete" overlay is a no-op
      returning the overlay unchanged (confirm semantics — idempotent).
    - **Discard-and-report.** The per-gap solve budget is the inner solve's existing safety
      deadline; a witness whose re-solve fails or times out stays unfilled and is REPORTED
      (``unfilled`` + ``unfilled_witness_ids``), never absorbed as an uncertified incumbent —
      every merged point comes from the same inner solves as any exact point, so run-level
      provenance (``Run.solver``/``exact``) stays sound.
    - **The merge never shrinks the overlay to a caller cap**: ``max_solutions`` bounds the
      fill pass itself; the merged union is capped only by the engine-wide
      ``MAX_PARETO_SOLUTIONS`` safety valve (reported as ``pruned`` if ever hit).
    """
    from engine.explorer import gap_witness_solutions

    base_solver = getattr(exact_run, "solver", None)
    if base_solver != solver:
        raise ValueError(
            f"fill_gaps extends the existing '{base_solver}' overlay — run it with "
            f"solver='{base_solver}', or rebuild the overlay with '{solver}' "
            "(scope='curated' or 'full') and fill that."
        )
    exact = bool(exact or getattr(exact_run, "exact", False))  # never demote the overlay's certification

    obj_list = problem.objectives
    witnesses = gap_witness_solutions(problem, nsga_run, exact_run)
    if not witnesses:
        return exact_run, {
            "gap_witnesses": 0, "filled": 0, "unfilled": 0,
            "note": "overlay already complete — nothing to fill (no-op; confirm semantics)",
        }

    source = Run(solutions=witnesses, mode=mode or nsga_run.mode,
                 seed_used=nsga_run.seed_used, solver=nsga_run.solver)
    fill_run = certify_curated(problem, source, solver=solver, exact=exact,
                               mode=mode, max_solutions=max_solutions)

    # Merge fill points into the overlay: dedupe by content, non-dominated filter, prune,
    # re-index — the same postprocessing every scalarization Run gets. Solutions are COPIED
    # (re-indexing restamps solution_id) so the caller's exact_run stays untouched.
    from engine.models import _content_signature
    from solvers._scalarization import _nondominated

    union, seen = [], set()
    for s in list(exact_run.solutions) + list(fill_run.solutions):
        sig = _content_signature(s.selected_options, s.allocations)
        if sig in seen:
            continue
        seen.add(sig)
        union.append(s.model_copy(deep=True))
    merged = _nondominated(union, obj_list)
    merged, total = _prune_pareto(merged, obj_list, max_n=MAX_PARETO_SOLUTIONS)
    merged = _sort_and_reindex(merged, obj_list)
    # The merged overlay inherits the base overlay's identity and provenance (solver, quality,
    # time_limit/time_limited — a capped best-so-far sweep stays flagged partial after a fill)
    # under a fresh run identity; certification only survives if base AND fill carried it.
    merged_run = exact_run.model_copy(update={
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc),
        "solutions": merged,
        "total_pareto_found": total,
        "mode": mode or exact_run.mode,
        "exact": bool(exact_run.exact and fill_run.exact),
    })
    # The fill's own instrumentation describes the solves that just ran (fields exist once
    # solve telemetry lands; guarded so this stays correct on either side of that change).
    for field in ("telemetry", "problem_snapshot"):
        if hasattr(fill_run, field) and hasattr(merged_run, field):
            setattr(merged_run, field, getattr(fill_run, field))

    still_open = gap_witness_solutions(problem, nsga_run, merged_run)
    unfilled_ids = [s.solution_id for s in still_open]
    pruned = max(0, total - len(merged))
    report = {
        "gap_witnesses": len(witnesses),
        "filled": max(0, len(witnesses) - len(still_open)),
        "unfilled": len(still_open),
        **({"unfilled_witness_ids": unfilled_ids[:10]} if still_open else {}),
        **({"pruned": pruned} if pruned else {}),
        "note": ("every detected gap closed — re-run explore certify for the updated certificate"
                 if not still_open else
                 f"{len(still_open)} witness(es) stayed unfilled ("
                 + ("the merged overlay hit the engine's size cap"
                    if pruned else "inner solve declined or hit its per-gap budget")
                 + ") — discarded and reported rather than merged as uncertified "
                 "incumbents; the NSGA points remain the best known there"),
    }
    return merged_run, report


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
    # Copy the MODEL only — run history, curation, and feedback are dead weight for a
    # scenario evaluation, and deep-copying them turns a per-slate scoring call into
    # seconds (regret alone scores every base solution under every scenario, so a full
    # deep copy here made scenario_results minutes-slow on run-heavy problems).
    sp = problem.model_copy(update={
        "run": None, "runs": [], "exact_run": None, "scenario_run": None,
        "curated_solutions": [], "feedback": [],
    }, deep=False).model_copy(deep=True)

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


def make_slate_scorer(problem: Problem, scenario: "Scenario | None" = None):
    """Batch form of ``score_slate``: apply the scenario overrides and build the scoring
    context (score matrix, interaction matrices, constraint encoding) ONCE, and return a
    ``score(selected_options, allocations=None) -> {"values", "feasible"}`` callable for
    many slates. Regret alone scores every base solution under every scenario, so the
    per-call rebuild — O(options×objectives) plus O(n²) per quadratic matrix — is the
    dominant cost once the problem copy is lean; hoist it per scenario instead."""
    p = build_scenario_problem(problem, scenario) if scenario is not None else problem
    n = len(p.options)
    idx = {o.name: i for i, o in enumerate(p.options)}
    score_matrix = _build_score_matrix(p)
    im = _build_interaction_matrices(p)
    cp = _parse_constraints(p)
    cls = _ProportionalProblem if p.approach == Approach.proportional else _FrontierProblem
    prob = cls(n_options=n, score_matrix=score_matrix, objectives=p.objectives,
               interaction_matrices=im, **cp)
    proportional = p.approach == Approach.proportional
    objectives = p.objectives

    def score(selected_options: list[str], allocations: dict[str, int] | None = None) -> dict:
        x = np.zeros((1, n))
        if proportional:
            for name, pct in (allocations or {}).items():
                if name in idx:
                    x[0, idx[name]] = pct
        else:
            for name in selected_options:
                if name in idx:
                    x[0, idx[name]] = 1.0
        values = {
            obj.name: round(float(prob._aggregate_objective(x, j)[0]), 4)
            for j, obj in enumerate(objectives)
        }
        out: dict = {}
        prob._evaluate(x.copy(), out)
        g = out.get("G")
        feasible = bool(g is None or np.asarray(g).size == 0 or np.all(np.asarray(g) <= 1e-6))
        return {"values": values, "feasible": feasible}

    return score


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
    Returns ``{"values": {obj: v}, "feasible": bool}``. Scoring many slates under the
    same scenario? Use ``make_slate_scorer`` and pay the context build once."""
    return make_slate_scorer(problem, scenario)(selected_options, allocations)


def evaluate_slate(
    problem: Problem,
    selected_options: list[str],
    allocations: dict[str, int] | None = None,
    scenario: "Scenario | None" = None,
) -> dict[str, float]:
    """Objective vector for one slate under the base case or a scenario.
    Thin wrapper over ``score_slate`` (which also returns feasibility)."""
    return score_slate(problem, selected_options, allocations, scenario)["values"]


def _negate_property(problem: Problem, prop) -> list[list[tuple]]:
    """Encode the *negation* of an audited property as binary MILP rows — the witness condition.
    INFEASIBLE on the negation (intersected with the problem's own constraints) proves the property
    holds across the whole feasible space; a feasible point is a concrete counterexample.

    Returns a *disjunction* (list of row-sets), each row ``(coef, op, rhs)`` with op 'ge'/'le'.
    ``prop is None`` (feasibility probe) → one empty row-set ("is any plan feasible at all?"). Most
    properties negate to a single row-set; cardinality negates to two (below-min OR above-max).
    The property is expressed in Frontier's existing ``Constraint`` vocabulary — no new grammar."""
    n = len(problem.options)
    ix = {o.name: i for i, o in enumerate(problem.options)}

    def idx(name: str) -> int:
        if name not in ix:
            raise ValueError(f"audit property references unknown option '{name}'.")
        return ix[name]

    def unit(name: str) -> list[float]:
        v = [0.0] * n
        v[idx(name)] = 1.0
        return v

    if prop is None:
        return [[]]
    t = prop.type
    if t == "force_include":       # holds: in every feasible plan → witness: a plan without it
        return [[(unit(prop.option), "le", 0.0)]]
    if t == "force_exclude":       # holds: in no feasible plan → witness: a plan with it
        return [[(unit(prop.option), "ge", 1.0)]]
    if t == "exclusion_pair":      # holds: never both → witness: a plan with both
        coef = [0.0] * n
        coef[idx(prop.option_a)] = 1.0
        coef[idx(prop.option_b)] = 1.0
        return [[(coef, "ge", 2.0)]]
    if t == "dependency":          # holds: if_option ⇒ then_option → witness: if ∧ ¬then
        return [[(unit(prop.if_option), "ge", 1.0), (unit(prop.then_option), "le", 0.0)]]
    if t == "group_limit":   # holds: min ≤ group count ≤ max → witness: ≤min-1 OR ≥max+1
        coef = [0.0] * n
        for o in prop.options:
            coef[idx(o)] = 1.0
        disjuncts = []
        if int(prop.min) > 0:
            disjuncts.append([(coef, "le", prop.min - 1)])
        disjuncts.append([(coef, "ge", prop.max + 1)])
        return disjuncts
    if t == "cardinality":         # holds: min ≤ count ≤ max → witness: count≤min-1 OR count≥max+1
        ones = [1.0] * n
        disjuncts = []
        if prop.min > 0:
            disjuncts.append([(ones, "le", prop.min - 1)])
        disjuncts.append([(ones, "ge", prop.max + 1)])
        return disjuncts
    if t == "objective_bound":
        ocol = {o.name: j for j, o in enumerate(problem.objectives)}
        if prop.objective not in ocol:
            raise ValueError(f"audit property references unknown objective '{prop.objective}'.")
        obj = problem.objectives[ocol[prop.objective]]
        if obj.aggregation != Aggregation.sum:
            # The witness row J = col·x equals the objective ONLY under sum aggregation. The
            # audit gate covers bounds in the MODEL's constraints; bound PROPERTIES arrive here
            # directly, so this guard is load-bearing — not redundant with the gate.
            raise ValueError(
                f"audit objective_bound needs a sum-aggregated objective; '{prop.objective}' uses "
                f"'{obj.aggregation.value}'.")
        col = _build_score_matrix(problem)[:, ocol[prop.objective]].astype(float).tolist()
        # FP strict-violation band: J is a sum of (discrete) per-option scores, so a small relative
        # epsilon cleanly separates "J > value" from "J = value" without floating-point ties.
        tol = max(1e-6, 1e-6 * abs(prop.value))
        if prop.operator == "max":     # holds: J ≤ value → witness: J ≥ value + tol
            return [[(col, "ge", prop.value + tol)]]
        return [[(col, "le", prop.value - tol)]]   # operator 'min': holds J ≥ value
    raise ValueError(
        f"audit doesn't support property type '{t}'. Auditable: force_include, force_exclude, "
        "exclusion_pair, dependency, group_limit, cardinality, objective_bound "
        "(max_allocation is proportional-only)."
    )


def diagnose_conflicts(problem: Problem) -> dict:
    """Minimal conflict set for an infeasible binary problem — deletion filtering over the
    user's named constraints.

    Rebuilds the feasibility MILP dropping one constraint at a time: a constraint whose removal
    leaves the rest still infeasible is unnecessary for the contradiction (dropped permanently);
    one whose removal restores feasibility is a conflict member (kept). Converges to ONE minimal
    infeasible subset: relaxing any single member clears THIS conflict, while other independent
    conflicts may remain — the members are leads, not proof, and nothing is auto-relaxed.
    O(#constraints) feasibility solves, cheap on binary-selection shapes.

    highspy's row-level IIS (``getIis``) is available but names solver rows; the deletion filter
    keeps the explanation in the user's constraint vocabulary. Every filter solve must reach a
    solver-proven verdict — a stopped solve yields an inconclusive diagnosis, never a guessed
    conflict (the same never-overclaim bar as the audit verdicts)."""
    from solvers._scalarization import audit_milp
    from solvers.highs_backend import _audit_milp_highs

    def _feasible_without(trial) -> bool | None:
        sub = problem.model_copy(update={"constraints": list(trial)})
        raw = audit_milp(sub, [[]], inner_audit=_audit_milp_highs)
        if raw["feasible"]:
            return True
        return False if all(s == "Infeasible" for s in raw["statuses"]) else None

    members = list(problem.constraints or [])
    # Precondition guard: the filter's logic is only sound over a PROVEN-infeasible set — on a
    # satisfiable one it would keep every constraint and present the whole model as a "conflict".
    f0 = _feasible_without(members)
    if f0 is True:
        return {"satisfiable": True,
                "note": "the constraint set is satisfiable — there is no contradiction to diagnose. "
                        "An empty frontier here comes from the search, not the model: the solver "
                        "proposes only non-empty plans, so if the empty plan is the sole feasible "
                        "one, add a cardinality or objective floor; otherwise raise the solve budget."}
    inconclusive = {"inconclusive": True,
                    "reason": "a deletion-filter solve stopped without a verdict — no conflict set "
                              "is claimed (a guessed conflict would be worse than none)."}
    if f0 is None:
        return inconclusive
    i = 0
    while i < len(members):
        trial = members[:i] + members[i + 1:]
        f = _feasible_without(trial)
        if f is None:
            return inconclusive
        if f is False:
            members = trial   # still contradictory without it → not part of this conflict
        else:
            i += 1            # removal restores feasibility → conflict member
    return {
        "constraints": [c.model_dump(mode="json") for c in members],
        "minimal": True,
        "diagnosis": "these constraints cannot all hold together; relaxing any ONE of them clears "
                     "this conflict",
        "caveat": "leads, not proof — other independent conflicts may remain; relax one member and "
                  "re-run `explore audit` to confirm feasibility (nothing is relaxed for you)",
    }


def constraints_fingerprint(constraints) -> str:
    """Order-insensitive content hash of a constraint set — the region identity behind
    `explore audit`'s ``feasible_region`` pin, echoed by ``model get`` so a stored verdict
    can be checked against the current model. Order-insensitive on purpose: a client that
    re-sends the same constraints reordered describes the same region, and counts-by-type
    can't catch a bound tweak (cardinality 5→6) that this hash does."""
    canon = sorted(json.dumps(c.model_dump(mode="json"), sort_keys=True)
                   for c in (constraints or []))
    return hashlib.md5("\n".join(canon).encode()).hexdigest()[:12]


def audit(problem: Problem, prop=None, solver: str = "highs") -> dict:
    """Witness / feasibility audit over a binary problem's whole feasible region — the auditor
    sibling of ``certify`` (which audits optimality; this audits feasibility / property claims).

    With ``prop=None`` it *probes feasibility* — "is any plan feasible under the current
    constraints?". With ``prop`` (a ``Constraint``) it *proves a guarantee* — does the property hold
    for EVERY feasible plan (the negation is infeasible across the whole region, not a sample), or is
    there a concrete counterexample witness? With a *list* of constraints it proves the conjunction —
    a compound guarantee holds iff every conjunct holds, so each gets its own feasibility solve and
    the payload carries a per-property breakdown. Binary selection only in v1 (the exact MILP
    feasibility path).

    Returns a structured verdict dict (``verdict`` ∈ feasible / no_feasible_plan / violated / holds /
    holds_vacuously / inconclusive). **Raises ``ValueError`` when the shape or backend can't serve an
    audit** (non-binary, an objective_bound on a non-sum objective, or HiGHS not installed) — the
    same hard-decline convention as ``solve``'s exact gate and ``certify``'s precondition, never a
    silent or structured degradation."""
    from solvers import available_solvers
    from solvers._scalarization import audit_milp

    if solver != "highs":
        raise ValueError(f"audit v1 runs on the HiGHS backend; solver='{solver}' is not supported.")
    if problem.approach != Approach.binary:
        raise ValueError(
            "audit v1 supports binary selection problems; a proportional/continuous feasibility "
            "audit is not yet implemented — explore the allocation frontier with the EA instead.")
    # Audit is a *feasibility* solve over the constraint region: objectives enter the encoding only
    # through objective_bound rows (linear in the score matrix — exact under sum aggregation only).
    # Gate on exactly that rather than the full exact-solve shape: an avg/min/max/quadratic
    # objective that no bound touches doesn't shape the region, so e.g. a binary problem with a
    # quadratic interaction objective still audits. ``_negate_property`` re-guards bound
    # *properties* the same way.
    bound_objs = {c.objective for c in (problem.constraints or []) if c.type == "objective_bound"}
    by_name = {o.name: o for o in problem.objectives}
    nonsum = sorted(n for n in bound_objs
                    if n in by_name and by_name[n].aggregation != Aggregation.sum)
    if nonsum:
        raise ValueError(
            "audit encodes objective_bound constraints as linear (sum) rows; "
            f"{', '.join(nonsum)} use a non-sum aggregation, so the bounded region can't be "
            "encoded exactly. Drop or re-aggregate those bounds to audit.")
    if not available_solvers().get("highs"):
        raise ValueError("audit needs the HiGHS exact backend — `pip install highspy`.")
    # Score completeness is enforced at the root by _build_score_matrix (a worded decline,
    # not a KeyError), which the MILP encoding below funnels through.

    from solvers.highs_backend import _audit_milp_highs

    # Normalize: None (probe), one property, or a conjunction (list — holds iff every conjunct holds).
    if isinstance(prop, (list, tuple)):
        if not prop:
            raise ValueError(
                "audit property list is empty — pass one or more constraint-shaped properties, "
                "or omit the property to probe feasibility.")
        props = list(prop)
    else:
        props = None if prop is None else [prop]

    # Pin the audited region so a `holds` is self-certifying — the guarantee is conditional on
    # exactly these constraints, the traceable-claims convention the explore analytics follow.
    # Pinned as counts-by-type + a content fingerprint rather than the full constraint echo:
    # a 62-constraint region dumped verbatim (twice per violated→fix→holds arc) dominated the
    # payload while saying nothing the fingerprint doesn't certify. `model get` echoes the
    # same fingerprint, so the pin is checkable against the current model, not just legible.
    cons = problem.constraints or []
    from collections import Counter
    base = {
        "audit_kind": "feasibility_probe" if props is None else "property_audit",
        "solver": "highs",
        "feasible_region": {
            "approach": problem.approach.value,
            "n_options": len(problem.options),
            "n_constraints": len(cons),
            "constraints_by_type": dict(sorted(Counter(c.type for c in cons).items())),
            "constraints_fingerprint": constraints_fingerprint(cons),
            "note": ("the verdict is conditional on exactly this constraint set — "
                     "`model get` echoes the same constraints_fingerprint, so compare the "
                     "two before re-asserting a guarantee"),
        },
    }

    def _witness(names: list[str]) -> dict:
        sl = score_slate(problem, names)
        return {"selected_options": names,
                "objective_values": sl["values"], "feasible": sl["feasible"]}

    def _stopped(statuses) -> list[str]:
        # Any status besides Optimal / Infeasible (time limit, iteration limit, solver error, …)
        # proves nothing in either direction; keep the raw statuses so the stop cause is traceable.
        return sorted({s for s in statuses if s != "Infeasible"})

    if props is None:
        raw = audit_milp(problem, [[]], inner_audit=_audit_milp_highs)
        if raw["feasible"]:
            return {**base, "verdict": "feasible", "witness": _witness(raw["witness_options"])}
        if all(s == "Infeasible" for s in raw["statuses"]):
            return {**base, "verdict": "no_feasible_plan", "witness": None,
                    "conflicts": diagnose_conflicts(problem)}
        stopped = _stopped(raw["statuses"])
        return {**base, "verdict": "inconclusive", "witness": None, "solver_status": stopped,
                "reason": f"the feasibility solve stopped without a verdict (HiGHS: {', '.join(stopped)}) — "
                          "INCONCLUSIVE, not evidence either way; simplify the model and re-probe."}

    # Property audit — every conjunct gets its own feasibility solve of its negation. All conjuncts
    # are evaluated even after a violation, so the breakdown names each guarantee's own verdict.
    per_prop: list[dict] = []
    witness = None
    violated_prop: dict | None = None
    stopped_all: set[str] = set()
    for p in props:
        disjuncts = _negate_property(problem, p)   # raises ValueError on unknown option/objective
        raw = audit_milp(problem, disjuncts, inner_audit=_audit_milp_highs)
        if raw["feasible"]:
            verdict = "violated"
            if witness is None:
                witness = _witness(raw["witness_options"])
                violated_prop = p.model_dump(mode="json")
        elif all(s == "Infeasible" for s in raw["statuses"]):
            verdict = "holds"
        else:
            verdict = "inconclusive"
            stopped_all.update(_stopped(raw["statuses"]))
        per_prop.append({"property": p.model_dump(mode="json"), "verdict": verdict})
    conj = {"properties": per_prop} if len(props) > 1 else {}

    stopped = sorted(stopped_all)
    status_note = ({"solver_status": stopped} if stopped else {})
    if witness is not None:
        # A violation is a definitive verdict, but a sibling conjunct whose solve stopped
        # still surfaces its stop cause — its breakdown entry says `inconclusive`, and the
        # caller needs the raw status to tell solver-stop from model complexity.
        out = {**base, "verdict": "violated", "witness": witness, **conj, **status_note}
        if len(props) > 1:
            out["violated_property"] = violated_prop
        return out
    if any(r["verdict"] == "inconclusive" for r in per_prop):
        return {**base, "verdict": "inconclusive", "witness": None, **conj,
                "solver_status": stopped,
                "reason": f"the feasibility solve stopped without a verdict (HiGHS: {', '.join(stopped)}) — "
                          "INCONCLUSIVE, not evidence either way; narrow the property or simplify "
                          "the model."}
    # Every conjunct's negation is proven infeasible — but confirm the feasible region isn't itself
    # empty, else "holds" is only vacuously true (the classic "num_points==0 ≠ PASS" trap). "Empty
    # region" is itself a claim needing a solver-proven Infeasible; a probe that stopped early
    # (time limit / error) is not evidence of emptiness.
    nonempty = audit_milp(problem, [[]], inner_audit=_audit_milp_highs)
    if not nonempty["feasible"]:
        if all(s == "Infeasible" for s in nonempty["statuses"]):
            return {**base, "verdict": "holds_vacuously", "witness": None, **conj,
                    "note": "the feasible region is empty — the property holds only vacuously; "
                            "probe feasibility first (audit with no property).",
                    "conflicts": diagnose_conflicts(problem)}
        return {**base, "verdict": "holds", "witness": None, **conj,
                "note": "the negation is proven infeasible, so the property holds for every "
                        "feasible plan — but the non-emptiness probe stopped without a verdict "
                        f"(HiGHS: {', '.join(sorted(set(nonempty['statuses'])))}), so whether any "
                        "feasible plan exists is unconfirmed; probe feasibility (audit with no "
                        "property) to check."}
    return {**base, "verdict": "holds", "witness": None, **conj}


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
    cp = _parse_constraints(problem, search_floor=True)
    im = _build_interaction_matrices(problem)

    pymoo_problem = _FrontierProblem(
        n_options=n_options, score_matrix=score_matrix, objectives=obj_list,
        interaction_matrices=im, **cp,
    )

    seeds = _compute_extreme_seeds(problem, score_matrix, cp)
    witness = _feasibility_witness_seed(problem, seeds, score_matrix)
    if witness is not None:
        seeds = np.vstack([seeds, witness]) if len(seeds) > 0 else witness
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
    limited = _was_time_limited(time_limit, elapsed)
    return Run(
        solutions=solutions,
        total_pareto_found=total_found,
        quality=_compute_quality(result, seed=seed),
        mode=mode,
        seed_used=seed,
        time_limit=time_limit,
        time_limited=limited,
        telemetry=_nsga_telemetry(result, n_gen, elapsed),
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
    cp = _parse_constraints(problem, search_floor=True)
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

    ab = cp.get("allocation_bounds") or {}
    bounds_pct = None
    if ab:
        lo = np.zeros(n_options, dtype=int)
        hi = np.full(n_options, int(cp.get("max_allocation") or 100), dtype=int)
        for i, (l, h) in ab.items():
            lo[i], hi[i] = int(l), min(int(hi[i]), int(h))
        bounds_pct = (lo, hi)
    solutions: list[Solution] = []
    if union_F is not None and len(union_F) > 0:
        for idx, (x, f) in enumerate(zip(union_X, union_F)):
            # Round to integer percents summing to 100 via the shared helper (also used by
            # the exact overlay), so per-option allocation bounds survive the rounding.
            raw = _round_weights_to_pct(np.asarray(x, dtype=float) / 100.0, n_options,
                                        cp.get("max_allocation"), bounds_pct)

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
    limited = _was_time_limited(time_limit, elapsed)
    return Run(
        solutions=solutions,
        total_pareto_found=total_found,
        quality=_compute_quality(result, seed=seed),
        mode=mode,
        seed_used=seed,
        time_limit=time_limit,
        time_limited=limited,
        telemetry=_nsga_telemetry(result, n_gen, elapsed),
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
        group_limits: list[tuple[list[int], int, int]] | None = None,  # (indices, min, max)
        max_allocation: int | None = None,  # ignored for binary mode
        allocation_bounds: dict | None = None,  # ignored for binary mode (proportional-only)
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
        n_ieq += sum(1 for _, g_min, _ in group_limits if g_min > 0)  # sum(group) >= min

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

        # Group limits: min <= sum(x[i] for i in group) <= max
        for group_indices, min_count, max_count in self.group_limits:
            group_sum = sum(X_bin[:, i] for i in group_indices)
            G.append(group_sum - max_count)
            if min_count > 0:
                G.append(min_count - group_sum)

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
        group_limits: list[tuple[list[int], int, int]] | None = None,  # (indices, min, max)
        max_allocation: int | None = None,
        allocation_bounds: dict | None = None,  # idx -> (min%, max%)
        interaction_matrices: dict[int, np.ndarray] | None = None,
    ):
        exclusion_pairs = exclusion_pairs or []
        dependencies = dependencies or []
        group_limits = group_limits or []

        n_ieq = 2 + 2 + len(forced_in) + len(forced_out) + len(obj_bounds)
        n_ieq += len(exclusion_pairs) + len(dependencies) + len(group_limits)
        n_ieq += sum(1 for _, g_min, _ in group_limits if g_min > 0)  # active-count >= min
        if max_allocation is not None:
            n_ieq += n_options  # one constraint per option
        n_ieq += 2 * len(allocation_bounds or {})   # per-option lo/hi rows

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
        self.allocation_bounds = allocation_bounds or {}
        # Box vectors in percent for the repair operator (0 / effective cap where unbounded).
        eff_cap = float(max_allocation) if max_allocation is not None else 100.0
        self.alloc_lo = np.zeros(n_options)
        self.alloc_hi = np.full(n_options, eff_cap)
        for i, (lo, hi) in self.allocation_bounds.items():
            self.alloc_lo[i] = float(lo)
            self.alloc_hi[i] = min(eff_cap, float(hi))
        self._bound_idx = (np.fromiter(self.allocation_bounds, dtype=int)
                           if self.allocation_bounds else None)
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

        # Group limits: min <= count of allocated in group <= max
        for group_indices, min_count, max_count in self.group_limits:
            group_sum = sum(allocated[:, i] for i in group_indices)
            G.append(group_sum - max_count)
            if min_count > 0:
                G.append(min_count - group_sum)

        # Per-option allocation bounds: lo <= x[i] <= hi (allocation_bound constraints)
        if self._bound_idx is not None:
            Xb = X[:, self._bound_idx]
            G.append(self.alloc_lo[self._bound_idx] - Xb)
            G.append(Xb - self.alloc_hi[self._bound_idx])

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
    group_limits = [(c.options, int(c.min), c.max) for c in problem.constraints if c.type == "group_limit"]
    # Floor of 1 mirrors the EA's search_floor: this diagnoses why the SEARCH came
    # back empty, so it reasons over the plans the search actually proposes.
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
        for group_opts, min_count, max_count in group_limits:
            if skip_constraint == f"group_limit:{','.join(group_opts)}:{min_count}:{max_count}":
                continue
            in_group = sum(1 for o in group_opts if o in selected)
            if in_group > max_count or in_group < min_count:
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
                (f"group_limit:{','.join(c.options)}:{int(c.min)}:{c.max}", c.model_dump())
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

    result = {"binding_constraints": binding, "suggestions": suggestions}
    if problem.approach == Approach.proportional:
        # The enumeration above reasons over option SUBSETS (unweighted score sums, allocation
        # constraints invisible) — a selection-level screen. Say so, and point at the layer
        # that does the allocation arithmetic exactly, so a proportional user reads the
        # binding list as candidates rather than a verdict.
        result["scope_note"] = (
            "selection-level diagnosis: constraints are tested over option subsets with "
            "unweighted score sums, so allocation-level rules (per-option floors/caps, "
            "weighted objective bounds) are screened approximately here — `solve validate` "
            "checks the allocation arithmetic exactly (floor sums, floor-vs-cap conflicts).")
    # Exact upgrade where the shape supports it: a solver-proven minimal conflict set beats the
    # one-at-a-time heuristic above (which can only say "may help", and enumerates with a cutoff).
    from solvers import available_solvers, exact_solver_fits
    if (problem.approach == Approach.binary and exact_solver_fits(problem)[0]
            and available_solvers().get("highs")):
        result["conflicts"] = diagnose_conflicts(problem)
    return result


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
