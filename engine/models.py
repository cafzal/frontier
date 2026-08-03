"""Pydantic data models for Frontier."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Reject non-finite floats (inf / nan) on user-supplied numeric fields. A NaN score
# silently passes validation, serializes to JSON `null` on save, then raises an
# uncaught ValidationError on every later load — permanently bricking the record.
# allow_inf_nan=False makes pydantic reject it at input time instead. Applied only to
# user-input fields; engine-computed outputs (Solution/Run/quality) keep plain float.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


# --- Enums ---


class Direction(str, Enum):
    maximize = "maximize"
    minimize = "minimize"


class Aggregation(str, Enum):
    sum = "sum"
    avg = "avg"
    min = "min"
    max = "max"
    quadratic = "quadratic"


class Approach(str, Enum):
    binary = "binary"
    proportional = "proportional"


class OptimizeMode(str, Enum):
    fast = "fast"
    thorough = "thorough"


class BoundOperator(str, Enum):
    min = "min"
    max = "max"


# --- Core models ---


class Objective(BaseModel):
    name: str
    direction: Direction
    unit: str = ""
    aggregation: Aggregation = Aggregation.sum


class Option(BaseModel):
    name: str
    description: str = ""


class Score(BaseModel):
    option: str
    objective: str
    value: FiniteFloat


class InteractionScaleGroup(BaseModel):
    """Scale interactions among a group of options by a factor.

    Used in scenario overrides to express regime shifts (e.g. "equity-equity
    correlations rise 50% in recession") without re-uploading a full matrix.
    """
    options: list[str]
    factor: FiniteFloat


class InteractionMatrix(BaseModel):
    """Pairwise interaction matrix for quadratic aggregation.

    entries[option_a][option_b] = interaction value (must be symmetric).
    For portfolio volatility, this is the covariance matrix.

    Modes (apply identically to the base matrix and to scenario overrides):
    - "replace": full matrix replacement (default).
    - "upsert": sparse cell upsert — merge ``entries`` into the existing matrix,
      preserving cells not mentioned. Symmetry auto-enforced (writing (a,b)
      also writes (b,a)). This is how a matrix larger than one tool call gets
      built: send it across several upserts.

    ``scale_groups`` (optional): applied AFTER replace/upsert. For each group,
    multiply off-diagonal entries where both endpoints are in the group by
    ``factor``. Composable with either mode. Set factor=0 to zero correlations
    within a group; negative factors flip the sign.
    """
    objective: str
    entries: dict[str, dict[str, FiniteFloat]] = {}
    mode: Literal["replace", "upsert"] = "replace"
    scale_groups: list[InteractionScaleGroup] = []


# --- Constraints ---


class _Motivated(BaseModel):
    """Provenance shared by every constraint type: why this rule exists.

    A constraint shapes the feasible region silently — `binding_analysis` can price a cap
    exactly and still not say what it is protecting, and the curated handoff renders finalists
    whose option space was cut by rules that reach review anonymous. The field is the user's own
    words, carried and echoed, never reasoned over. Same name and role as `Scenario.motivated_by`
    (one term per concept), and the natural moment to fill it is the one `problem_framing` →
    *Post-Solve Constraint Discovery* already teaches: a rejected-but-valid solution reveals a
    latent constraint, and the rejection IS the motive.

    Optional and defaulted, so existing bundles, saved problems, and tool calls parse unchanged.
    """
    motivated_by: str = ""


class CardinalityConstraint(_Motivated):
    type: Literal["cardinality"] = "cardinality"
    min: int
    max: int


class ForceIncludeConstraint(_Motivated):
    type: Literal["force_include"] = "force_include"
    option: str


class ForceExcludeConstraint(_Motivated):
    type: Literal["force_exclude"] = "force_exclude"
    option: str


class ObjectiveBoundConstraint(_Motivated):
    type: Literal["objective_bound"] = "objective_bound"
    objective: str
    operator: BoundOperator
    value: FiniteFloat


class ExclusionPairConstraint(_Motivated):
    type: Literal["exclusion_pair"] = "exclusion_pair"
    option_a: str
    option_b: str


class DependencyConstraint(_Motivated):
    type: Literal["dependency"] = "dependency"
    if_option: str
    then_option: str


class GroupLimitConstraint(_Motivated):
    type: Literal["group_limit"] = "group_limit"
    options: list[str]
    min: int = 0  # minimum selected/active options from the group (0 = no floor)
    max: int


class MaxAllocationConstraint(_Motivated):
    type: Literal["max_allocation"] = "max_allocation"
    max: int  # maximum allocation percentage for any single option (1-100)


class AllocationBoundConstraint(_Motivated):
    """Per-option allocation floor/cap in percent (proportional only) — contractual minimums,
    service floors, per-channel caps. The effective cap is min(global max_allocation, max);
    a floor > 0 force-activates the option."""
    type: Literal["allocation_bound"] = "allocation_bound"
    option: str
    min: int = 0
    max: int = 100


Constraint = (
    CardinalityConstraint
    | ForceIncludeConstraint
    | ForceExcludeConstraint
    | ObjectiveBoundConstraint
    | ExclusionPairConstraint
    | DependencyConstraint
    | GroupLimitConstraint
    | MaxAllocationConstraint
    | AllocationBoundConstraint
)


# --- Reference Points ---


class ReferencePoint(BaseModel):
    type: Literal["baseline", "aspirational"]
    name: str = ""
    objective_values: dict[str, FiniteFloat] = {}  # partial OK
    selected_options: list[str] = []  # baseline only — the current portfolio
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Scenarios ---


class ScoreAdjustment(BaseModel):
    """Adjust scores for an objective across all options in a scenario."""
    objective: str
    multiply: FiniteFloat | None = None  # e.g. 0.8 = reduce by 20%
    add: FiniteFloat | None = None  # e.g. -5 = subtract 5 from all scores


class Scenario(BaseModel):
    # A typoed or unknown field must error, not silently drop — scenario dicts arrive over
    # the wire, and a silently-ignored "motivatedby" is indistinguishable from working.
    model_config = ConfigDict(extra="forbid")

    name: str
    probability: FiniteFloat | None = None  # optional; only needed for expected-value weighting
    description: str = ""
    # Provenance of the scenario when seeded from an analysis lever (e.g. a sensitivity
    # suggestion's `motivated_by`) — echoed by scenario_results so the reading cites its motive.
    motivated_by: str = ""
    # The base constraint set this scenario was authored against (optimizer.constraints_fingerprint),
    # stamped server-side on every scenario_config write and on load. `constraint_overrides` replace
    # the base rules WHOLESALE, so a later base-constraint edit does not flow through — comparing
    # this stamp to the current fingerprint says whether the overrides predate the current rules
    # (annotation, never a solve input: it is stripped from the solve fingerprints).
    base_constraints_fingerprint: str = ""
    score_overrides: list[Score] = []  # only changed scores; base matrix fills rest
    score_adjustments: list[ScoreAdjustment] = []  # bulk adjustments by objective
    constraint_overrides: list[Constraint] = []  # replaces base constraints when non-empty
    interaction_matrix_overrides: list[InteractionMatrix] = []  # upserts per objective; base matrices fill rest


class ScenarioConfig(BaseModel):
    enabled: bool = False
    scenarios: list[Scenario] = []


class ScenarioRun(BaseModel):
    """Results from per-scenario optimization."""
    scenario_runs: dict[str, Run] = {}  # scenario name → Run
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Scenario solve-inputs as solved = the base fields PLUS scenario_config (see
    # scenario_solve_fingerprint) — edits compare against it, so scenario reads can say
    # "these results predate the current model" instead of serving them as findings.
    solve_fingerprint: str | None = None
    # The BASE half of the same stamp (see solve_fingerprint at module scope), so a read that
    # joins base solutions to scenario frontiers — regret — can tell whether the two sides
    # describe the same base model. Comparable to Run.solve_fingerprint; the composite above
    # is not.
    base_fingerprint: str | None = None


# --- Run / Solution ---


def _content_signature(selected_options: list[str], allocations: dict[str, int] | None = None) -> str:
    """Stable hash of solution composition. Survives re-indexing across runs."""
    if allocations:
        content = str(sorted((k, v) for k, v in allocations.items() if v > 0))
    else:
        content = str(sorted(selected_options))
    return hashlib.md5(content.encode()).hexdigest()[:12]


# --- Solution sensitivity (exact-solver duals) ---


class ShadowPrice(BaseModel):
    """Dual of a binding constraint, in the solver's COST sense: the price the
    constraint charges the optimized objective per unit of tightening (positive =
    tighter hurts, whichever direction that objective optimizes). Decision read:
    *where to invest* — the lever with the largest price buys the most when
    renegotiated. Exact for the continuous LP/QP path only; integer/MILP solutions
    carry no exact duals."""
    name: str            # "budget", or the swept objective's name (e.g. "Return")
    role: str            # "budget" | "return_floor" | "linear_floor"
    shadow_price: float


class ReducedCost(BaseModel):
    """Reduced cost of a decision variable — for an option left at zero, how far its
    objective coefficient must improve before it would enter the optimal solution.
    Decision read: a *near-miss* — the smallest-magnitude unheld option is closest to
    making the cut. Non-zero only for options the optimizer left at a bound."""
    option: str
    allocation: int      # this solution's allocation % (0 = unheld)
    reduced_cost: float
    eligible: bool = True  # False = pinned out by a cardinality/group cap (structurally excluded, not a near-miss)


class SolutionSensitivity(BaseModel):
    """Post-optimal sensitivity for one solution, from the exact solver's duals.

    ``source`` separates solver-exact duals (continuous LP/QP) from frontier-inferred
    regression estimates; only solver-exact duals are attached here."""
    source: str = "solver_exact"        # vs "frontier_inferred"
    shadow_prices: list[ShadowPrice] = []
    reduced_costs: list[ReducedCost] = []
    ranging: dict | None = None         # best-effort RHS/objective ranging (LP); None if unavailable


class Solution(BaseModel):
    solution_id: int
    selected_options: list[str]
    objective_values: dict[str, float]
    allocations: dict[str, int] | None = None  # proportional mode: option → percentage (0-100)
    content_signature: str = ""  # stable hash, computed post-init
    sensitivity: SolutionSensitivity | None = None  # exact-solver duals (LP/QP path); None for MILP/heuristic


class QualityIndicators(BaseModel):
    hypervolume_normalized: float | None = None
    spacing_cv: float | None = None


class Run(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: OptimizeMode = OptimizeMode.fast
    solutions: list[Solution] = []
    total_pareto_found: int = 0  # Pre-pruning count; equals len(solutions) when no pruning happened
    quality: QualityIndicators = QualityIndicators()
    constraints_snapshot: list[dict] = []
    seed_used: int | None = None  # Deterministic RNG seed; echoed for reproducibility
    solver: str = "nsga-ii"  # engine that produced this run: nsga-ii / nsga-iii / highs / cuopt
    exact: bool = False  # MILP zero-gap certification was requested (no-op on the always-exact QP and on NSGA)
    time_limit: float | None = None  # wall-clock cap (s) requested for this solve; None = uncapped
    time_limited: bool = False  # True when the cap was hit, so this frontier is best-so-far, not fully converged
    solve_fingerprint: str | None = None  # hash of the base solve-input fields as solved (see solve_fingerprint) — edits compare against it, so a round-trip edit lands back at results_stale=False
    telemetry: dict | None = None  # how this solve ran: {duration_s, engine_detail, evals_or_solves} — machine-local facts, stripped from portable bundles (cap-hit lives on time_limited)
    problem_snapshot: dict | None = None  # problem features as solved (solvers.problem_features) — pairs with telemetry so routing advice can be calibrated from real workload


# --- Feedback ---


class Feedback(BaseModel):
    content_signature: str | None = None  # stable link — survives re-runs
    solution_id: int | None = None  # ephemeral index — convenience only
    rating: int | None = None  # 1-5
    notes: str = ""
    stage: str = ""  # "exploration", "decision", "post-refinement"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Curated Solutions ---


class CuratedSolution(BaseModel):
    content_signature: str
    custom_name: str = ""
    selected_options: list[str] = []
    allocations: dict[str, int] | None = None
    objective_values: dict[str, float] = {}
    curated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_run_id: str = ""
    notes: str = ""
    feedback: list[Feedback] = []  # preference context — accumulated across runs


# --- Problem ---


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Problem(BaseModel):
    problem_id: str = Field(default_factory=_new_uuid)
    name: str = ""
    domain: str = ""
    context: str = ""
    approach: Approach = Approach.binary
    objectives: list[Objective] = []
    options: list[Option] = []
    scores: list[Score] = []
    constraints: list[Constraint] = []
    interaction_matrices: list[InteractionMatrix] = []
    reference_points: list[ReferencePoint] = []
    scenario_config: ScenarioConfig | None = None
    scenario_run: ScenarioRun | None = None
    run: Run | None = None
    runs: list[Run] = []
    exact_run: Run | None = None  # exact-solver overlay, alongside the exploratory `run`
    results_stale: bool = False
    curated_solutions: list[CuratedSolution] = []
    feedback: list[Feedback] = []
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


# --- Solve fingerprints (staleness) ---
#
# One definition of "what determines a solve's result", used by every side that asks whether a
# stored frontier still describes the current model: the solve workers' mid-solve stale guard,
# `model update`'s results_stale, and the scenario reads' stale marker. Two scopes, because the
# base frontier and the scenario frontiers have different inputs — the base solve cannot see
# scenario_config, so a scenarios-only edit must not read as staleness on the base run, and a
# scenario solve depends on both halves. The scenario stamp carries its base half separately
# (ScenarioRun.base_fingerprint) so a read joining the two sides can compare them.

_SOLVE_INPUT_FIELDS = {
    "approach", "objectives", "options", "scores", "constraints", "interaction_matrices",
}
_SCENARIO_SOLVE_INPUT_FIELDS = _SOLVE_INPUT_FIELDS | {"scenario_config"}

# Annotations: recorded ON solve inputs but never determining a solve's result — provenance
# (why a rule exists) and the scenario's authored-against constraints stamp. Annotating a rule
# after a solve must not read as editing the model. The sibling rule at
# `explorer._constraint_key` keeps `motivated_by` out of run-diff identity too.
_ANNOTATION_KEYS = {"motivated_by", "base_constraints_fingerprint"}


def _strip_annotations(node):
    """Drop annotation keys from a dumped payload (see _ANNOTATION_KEYS)."""
    if isinstance(node, dict):
        return {k: _strip_annotations(v) for k, v in node.items() if k not in _ANNOTATION_KEYS}
    if isinstance(node, list):
        return [_strip_annotations(v) for v in node]
    return node


def _fingerprint(problem: "Problem", fields: set) -> str:
    payload = _strip_annotations(problem.model_dump(mode="json", include=fields))
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def solve_fingerprint(problem: "Problem") -> str:
    """Fingerprint of the inputs a BASE solve reads (Run.solve_fingerprint)."""
    return _fingerprint(problem, _SOLVE_INPUT_FIELDS)


def scenario_solve_fingerprint(problem: "Problem") -> str:
    """Fingerprint of the inputs a PER-SCENARIO solve reads — the base inputs plus
    scenario_config (ScenarioRun.solve_fingerprint)."""
    return _fingerprint(problem, _SCENARIO_SOLVE_INPUT_FIELDS)


def scenario_results_stale(problem: "Problem") -> bool:
    """True when the stored scenario runs were solved against an earlier model.

    False when they match — and also when the stamp is missing (a pre-stamp run is
    unknowable, and guessing staleness would cry wolf on every legacy problem).
    """
    sr = problem.scenario_run
    if sr is None or not sr.scenario_runs or not sr.solve_fingerprint:
        return False
    return sr.solve_fingerprint != scenario_solve_fingerprint(problem)


def scenario_base_mismatch(problem: "Problem") -> bool:
    """True when the base run and the scenario runs were solved against DIFFERENT base
    models — the join of the two (regret re-scores base plans against scenario frontiers)
    would mix two models. Unknowable when either side is unstamped: False."""
    sr, run = problem.scenario_run, problem.run
    if sr is None or run is None or not sr.scenario_runs or not run.solutions:
        return False
    if not sr.base_fingerprint or not run.solve_fingerprint:
        return False
    return sr.base_fingerprint != run.solve_fingerprint


def stale_scenario_overrides(problem: "Problem", constraints_fingerprint: str) -> list[str]:
    """Scenarios whose `constraint_overrides` were authored against a different base
    constraint set than the one passed in — they replace the base rules wholesale, so a
    later base edit did NOT flow through and the overrides may need restating. Scenarios
    without overrides, or without an authored-against stamp, are never named."""
    sc = problem.scenario_config
    if not sc or not sc.scenarios:
        return []
    return [s.name for s in sc.scenarios
            if s.constraint_overrides and s.base_constraints_fingerprint
            and s.base_constraints_fingerprint != constraints_fingerprint]


# --- Validation result ---

# Message prefix of optimizer.validate's aggregated score-matrix issue. Shared so
# metrics.readiness can classify the issue structurally instead of mirroring the
# literal — reword the message here and both sides move together.
SCORE_MATRIX_MSG = "Score matrix incomplete"


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    message: str


class ValidationResult(BaseModel):
    ready: bool
    issues: list[ValidationIssue] = []
    missing_scores: list[dict[str, str]] = []
