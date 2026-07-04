"""Pydantic data models for Frontier."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

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


class ConstraintType(str, Enum):
    cardinality = "cardinality"
    force_include = "force_include"
    force_exclude = "force_exclude"
    objective_bound = "objective_bound"
    exclusion_pair = "exclusion_pair"
    dependency = "dependency"
    group_limit = "group_limit"
    max_allocation = "max_allocation"


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

    Modes (meaningful primarily in scenario overrides; base matrix uses default):
    - "replace": full matrix replacement (default). Base case always uses this.
    - "upsert": sparse cell upsert — merge ``entries`` into the base matrix,
      preserving cells not mentioned. Symmetry auto-enforced (writing (a,b)
      also writes (b,a)).

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


class CardinalityConstraint(BaseModel):
    type: Literal["cardinality"] = "cardinality"
    min: int
    max: int


class ForceIncludeConstraint(BaseModel):
    type: Literal["force_include"] = "force_include"
    option: str


class ForceExcludeConstraint(BaseModel):
    type: Literal["force_exclude"] = "force_exclude"
    option: str


class ObjectiveBoundConstraint(BaseModel):
    type: Literal["objective_bound"] = "objective_bound"
    objective: str
    operator: BoundOperator
    value: FiniteFloat


class ExclusionPairConstraint(BaseModel):
    type: Literal["exclusion_pair"] = "exclusion_pair"
    option_a: str
    option_b: str


class DependencyConstraint(BaseModel):
    type: Literal["dependency"] = "dependency"
    if_option: str
    then_option: str


class GroupLimitConstraint(BaseModel):
    type: Literal["group_limit"] = "group_limit"
    options: list[str]
    max: int


class MaxAllocationConstraint(BaseModel):
    type: Literal["max_allocation"] = "max_allocation"
    max: int  # maximum allocation percentage for any single option (1-100)


Constraint = (
    CardinalityConstraint
    | ForceIncludeConstraint
    | ForceExcludeConstraint
    | ObjectiveBoundConstraint
    | ExclusionPairConstraint
    | DependencyConstraint
    | GroupLimitConstraint
    | MaxAllocationConstraint
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
    name: str
    probability: FiniteFloat | None = None  # optional; only needed for expected-value weighting
    description: str = ""
    # Provenance of the scenario when seeded from an analysis lever (e.g. a sensitivity
    # suggestion's `motivated_by`) — echoed by scenario_results so the reading cites its motive.
    motivated_by: str = ""
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
    """Dual of a binding constraint — the marginal change in the optimized objective
    per unit the constraint is relaxed. Decision read: *where to invest* — the lever
    with the largest shadow price buys the most improvement. Exact for the continuous
    LP/QP path only; integer/MILP solutions carry no exact duals."""
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


# --- Validation result ---


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    message: str


class ValidationResult(BaseModel):
    ready: bool
    issues: list[ValidationIssue] = []
    missing_scores: list[dict[str, str]] = []
