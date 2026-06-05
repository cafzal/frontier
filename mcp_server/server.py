"""Frontier MCP server: 4 tools (model, solve, explore, get_skill) + 4 skill resources.

Skill auto-injection: tool responses include the relevant skill content for
the *next* workflow phase, so agents always receive guidance at the right time
without needing to manually call get_skill().

Injection map:
  MCP connect (server instructions) → condensed problem_framing + constraint schemas
  model/create response             → data_collection skill
  model/update (objectives/options)  → data_collection skill (if not already injected)
  model/update (scores hit 100%)     → optimization_strategy skill
  solve/run response                → solution_interpreter skill (always, on every solve)
  solve/validate (ready=true)       → optimization_strategy skill (if not already injected)
"""

from __future__ import annotations

import functools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from engine import explorer, metrics, optimizer, problem_io
from engine.models import (
    Approach,
    Constraint,
    CardinalityConstraint,
    DependencyConstraint,
    ExclusionPairConstraint,
    Feedback,
    ForceExcludeConstraint,
    _content_signature,
    ForceIncludeConstraint,
    GroupLimitConstraint,
    InteractionMatrix,
    MaxAllocationConstraint,
    Objective,
    ObjectiveBoundConstraint,
    OptimizeMode,
    Option,
    Problem,
    ReferencePoint,
    Scenario,
    ScoreAdjustment,
    ScenarioConfig,
    ScenarioRun,
    Score,
)
from engine.store import Store

mcp = FastMCP(
    "Frontier",
    instructions=(
        "Multi-objective portfolio optimization engine.\n\n"
        "WORKFLOW: model/create → model/update (objectives, options, scores, constraints) → solve/run → explore. "
        "For a FINAL/decision run on a supported shape (binary selection, or mean-variance QP), opt into an exact "
        "solver — solve(solver=\"highs\"|\"cuopt\") — then `explore certify` to audit NSGA vs exact (dominance, the "
        "NSGA-never-dominates invariant, corner sharpening), then — on continuous/QP — `explore sensitivity` for "
        "solver-exact duals, then decide.\n\n"
        "Domain skills — problem_framing, data_collection, optimization_strategy, solution_interpreter — "
        "are the canonical guides. data_collection / optimization_strategy / solution_interpreter "
        "auto-inject into tool responses at phase transitions. "
        "problem_framing is NOT auto-injected — read it via get_skill('problem_framing') "
        "BEFORE the first model/create whenever the framing isn't obvious. Re-read via get_skill() to refresh.\n\n"
        "## Pre-create framing checklist (read before model/create)\n"
        "Picking the right approach + aggregation upfront avoids a re-solve. Ask:\n"
        "1. Approach — does the user want to *select* options or *allocate* across them?\n"
        "   - Selection ('which K of N?') → `approach=\"binary\"`.\n"
        "   - Allocation ('how much of each?', distributing a fixed pool of capital/time/headcount) → `approach=\"proportional\"`.\n"
        "   - Portfolios usually mean allocation. If the user says 'portfolio' or 'diversified', default to proportional unless they explicitly say 'pick K'.\n"
        "2. Aggregation — for each objective, how do per-option scores combine into a portfolio score?\n"
        "   - `sum` (default): additive quantities (revenue, cost, effort).\n"
        "   - `avg`: when portfolio quality is the weighted/equal average (return %, yield %, quality scores). Most rate-like objectives want avg, not sum.\n"
        "   - `min` / `max`: when the weakest/strongest member determines the portfolio value.\n"
        "   - `quadratic`: when value depends on pairwise interactions (portfolio variance/risk via covariance, synergy, redundancy). Requires `interaction_matrices` entries.\n"
        "3. Risk / volatility objectives — if the user mentions risk, volatility, variance, or diversification, the objective almost certainly needs `aggregation=\"quadratic\"` with a covariance (or interaction) matrix. Bare `sum` over per-option vol is wrong for portfolios.\n"
        "If any of these are ambiguous, call get_skill('problem_framing') for the full guide before create.\n\n"
        "## Scores Schema\n"
        "Flat list of triples: [{\"option\": \"<name>\", \"objective\": \"<name>\", \"value\": <float>}, ...]\n"
        "Upsert by (option, objective). Every option × objective combination must be scored before solve.\n\n"
        "## Objective Schema\n"
        "{\"name\": \"<name>\", \"direction\": \"maximize\"|\"minimize\", \"aggregation\": \"sum\"|\"avg\"|\"min\"|\"max\"|\"quadratic\"}\n\n"
        "## Key Rules\n"
        "- Scores merge (upsert); objectives / options / constraints use full replacement.\n"
        "- Never say 'best' — every Pareto solution is optimal at its tradeoff.\n"
        "- Use plain text. Emojis and decorative symbols add visual noise without conveying tradeoff information and undermine the analytical tone of decision presentation.\n"
        "- Do not narrate progress between tool calls. Tool calls and results are already visible to the user — extra prose like 'Now I'll batch the scores' or 'Let me fix that field' is noise. Speak only in the final response after the work is complete.\n"
        "- Constraint schemas: see the `model` tool docstring for the full set."
    ),
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("PORT", "8000")),
)
store = Store()

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# Soft cap on `explore` actions that can return unbounded per-solution detail.
# Above this, the response truncates and points the caller at full_result_path
# on disk to avoid blowing the MCP token budget.
EXPLORE_DETAIL_CAP = 50

# Single inject prompt used by both solve/run and solve/run_scenarios — covers
# both the regular-frontier presentation framing and the scenario-specific
# next-step pointer, so the throttle's single shared flag never silently drops
# guidance depending on which solve mode fired first.
_SOLUTION_INTERPRETER_PROMPT = (
    "Optimization complete. Use this guide to present results — never say 'best', "
    "start with extremes and balanced, quantify tradeoffs. "
    "For scenario runs, also surface cross-scenario robustness via "
    "`explore scenario_results` and present results per scenario."
)


# ─── Skill auto-injection helpers ───


# Per-problem tracking of which skills have been injected, to avoid redundancy.
_injected_skills: dict[str, set[str]] = {}


def _mark_injected(problem_id: str, skill_name: str) -> None:
    _injected_skills.setdefault(problem_id, set()).add(skill_name)


def _reset_injection(problem_id: str, skill_name: str) -> None:
    if problem_id in _injected_skills:
        _injected_skills[problem_id].discard(skill_name)


def _reset_all_injections(problem_id: str) -> None:
    _injected_skills.pop(problem_id, None)


def _was_injected(problem_id: str, skill_name: str) -> bool:
    return skill_name in _injected_skills.get(problem_id, set())


@functools.lru_cache(maxsize=4)
def _load_skill(skill_name: str) -> str:
    """Load skill content from disk, cached across calls."""
    dirname = _SKILL_MAP.get(skill_name)
    if not dirname:
        return ""
    path = SKILLS_DIR / dirname / "SKILL.md"
    return path.read_text() if path.exists() else ""


def _inject_skill(result: dict, skill_name: str, reason: str) -> dict:
    """Append skill content to a tool response under a standard key."""
    content = _load_skill(skill_name)
    if content:
        result["_skill_guidance"] = {
            "skill": skill_name,
            "reason": reason,
            "content": content,
        }
    return result


# ─── Skills as resources ───


@mcp.resource("frontier://skills/problem_framing")
def skill_problem_framing() -> str:
    return (SKILLS_DIR / "problem-framing" / "SKILL.md").read_text()


@mcp.resource("frontier://skills/data_collection")
def skill_data_collection() -> str:
    return (SKILLS_DIR / "data-collection" / "SKILL.md").read_text()


@mcp.resource("frontier://skills/optimization_strategy")
def skill_optimization_strategy() -> str:
    return (SKILLS_DIR / "optimization-strategy" / "SKILL.md").read_text()


@mcp.resource("frontier://skills/solution_interpreter")
def skill_solution_interpreter() -> str:
    return (SKILLS_DIR / "solution-interpreter" / "SKILL.md").read_text()


# ─── Skill delivery tool (works with all MCP clients) ───

_SKILL_MAP = {
    "problem_framing": "problem-framing",
    "data_collection": "data-collection",
    "optimization_strategy": "optimization-strategy",
    "solution_interpreter": "solution-interpreter",
}


@mcp.tool()
def get_skill(skill_name: str) -> str:
    """Retrieve workflow guidance for a specific stage.

    NOTE: Skills are auto-injected into tool responses at phase transitions.
    Use this tool only if you need to re-read a skill or access it outside
    the normal workflow sequence.

    Available skills:
    - problem_framing — structuring objectives, options, constraints, approach
    - data_collection — scoring best practices, anchoring, completeness
    - optimization_strategy — mode selection, constraint strategy, iteration
    - solution_interpreter — presenting tradeoffs, eliciting preferences, curation

    Returns the full skill guide as markdown.
    """
    dirname = _SKILL_MAP.get(skill_name)
    if not dirname:
        return f"Unknown skill: {skill_name}. Available: {list(_SKILL_MAP.keys())}"
    path = SKILLS_DIR / dirname / "SKILL.md"
    if not path.exists():
        return f"Skill file not found: {skill_name}"
    return path.read_text()


# ─── Tool 1: model ───


@mcp.tool()
def model(
    action: str,
    problem_id: str | None = None,
    name: str | None = None,
    domain: str | None = None,
    context: str | None = None,
    objectives: list[dict] | None = None,
    options: list[dict | str] | None = None,
    scores: list[dict] | None = None,
    constraints: list[dict] | None = None,
    approach: str | None = None,
    reference_points: list[dict] | None = None,
    scenario_config: dict | None = None,
    interaction_matrices: list[dict] | None = None,
    section: str | None = None,
    source: str | None = None,
    save_as: str | None = None,
) -> dict:
    """Build and modify the optimization problem.

    Skill auto-injection: when an action triggers a skill, the response carries
    `_skill_guidance: {skill, reason, content}` — `content` is the full skill body
    inline, so no separate `get_skill()` call is needed once injected this session.
    Repeat calls suppress the injection (the skill is already in your context);
    structural changes re-arm it so the next phase delivers fresh guidance. For
    skills not auto-injected (e.g., `problem_framing` before `create`), fetch via
    `get_skill(<name>)`.

    Actions:
      create  — Start a new problem. Params: name?, domain?, context?, approach?,
                objectives?, options? (all optional; add via update later).
      update  — Modify problem. Params: problem_id (required), plus any of:
                name, domain, context, objectives, options, scores, constraints,
                approach ("binary" or "proportional"),
                reference_points (list of {type, name?, objective_values, selected_options?}),
                interaction_matrices (list of {objective, entries} for quadratic aggregation).
                Scores use merge semantics; everything else is full replacement.
                Side effects: structural edits (objectives/options/constraints/approach/matrices/scenarios)
                mark the latest run stale and reset the optimization_strategy and solution_interpreter
                injection flags so the next solve re-injects them.
      get     — Problem state. Params: problem_id, section? (optional).
                Without section: full dump (can exceed token cap for large models).
                With section: targeted slice. Valid sections:
                  summary (counts + status flags — always small),
                  objectives, options, scores, constraints, matrices,
                  scenarios, run, runs, curated, references.
                Prefer section="summary" for a lightweight overview, then drill down.
      list    — All problems. No params.
      delete  — Remove problem. Params: problem_id
                Side effects: deletes the problem and its run history; resets all skill-injection
                flags for the problem id.
      load    — Load a problem by name from the bundled examples or the user's saved
                library, in the portable examples format (problem.json + scores.json
                [+ solutions.json]). Params: source (the name, e.g. "facility_location").
                Resolves saved/ first, then examples/ — a saved problem shadows a bundled
                example of the same name. Registers a fresh problem in the store and returns
                its problem_id. Scenarios and prior results (when present) are restored.
                Omit source to list the available names.
      save    — Save a problem to the user's library (saved/<name>/) in the portable format,
                always including the current solutions when the problem has been solved.
                Params: problem_id, save_as? (the name; defaults to a slug of the problem name).
                The bundle round-trips back via `load`.

    Key guidance:
    - Objectives: 2-4 is the sweet spot. If >4, consolidate.
    - Ask "does quantity matter?" to choose binary vs proportional approach.
    - Classify carefully: "budget" might be an objective OR a constraint.
    - First formulation is a hypothesis — rough scores > no scores, fewer objectives > many.

    Schema reference: see the `problem_framing` skill (read it via
    `get_skill('problem_framing')` or before calling `model/create`) for
    constraint, interaction-matrix, and scenario JSON schemas.
    """
    params = {
        k: v for k, v in {
            "problem_id": problem_id, "name": name, "domain": domain,
            "context": context, "objectives": objectives, "options": options,
            "scores": scores, "constraints": constraints, "approach": approach,
            "reference_points": reference_points, "scenario_config": scenario_config,
            "interaction_matrices": interaction_matrices, "section": section,
            "source": source, "save_as": save_as,
        }.items() if v is not None
    }
    match action:
        case "create":
            return _model_create(params)
        case "update":
            return _model_update(params)
        case "get":
            return _model_get(params)
        case "list":
            return _model_list()
        case "delete":
            return _model_delete(params)
        case "save":
            return _model_save(params)
        case "load":
            return _model_load(params)
        case _:
            return {"error": f"Unknown action: {action}. Use create/update/get/list/delete/save/load."}


def _model_create(params: dict) -> dict:
    kwargs = dict(
        name=params.get("name", ""),
        domain=params.get("domain", ""),
        context=params.get("context", ""),
    )
    if "approach" in params:
        kwargs["approach"] = Approach(params["approach"])
    if "objectives" in params:
        kwargs["objectives"] = [
            Objective(**o) if isinstance(o, dict) else o for o in params["objectives"]
        ]
    if "options" in params:
        kwargs["options"] = [
            Option(name=o) if isinstance(o, str) else
            Option(**o) if isinstance(o, dict) else o
            for o in params["options"]
        ]
    p = Problem(**kwargs)
    store.save(p)
    result = {
        "problem_id": p.problem_id,
        "name": p.name,
        "domain": p.domain,
        "created_at": p.created_at.isoformat(),
        "objectives": len(p.objectives),
        "options": len(p.options),
    }
    # Next step is scoring — inject data_collection guidance
    _inject_skill(result, "data_collection",
        "Problem created. Use this guide when entering scores — "
        "it covers anchoring, batch efficiency, and completeness.")
    _mark_injected(p.problem_id, "data_collection")
    return result


def _model_update(params: dict) -> dict:
    pid = params.get("problem_id")
    if not pid:
        return {"error": "problem_id is required."}

    try:
        p = store.load(pid)
    except FileNotFoundError:
        return {"error": f"Problem {pid} not found."}

    structural_change = False

    # Metadata updates
    if "name" in params:
        p.name = params["name"]
    if "domain" in params:
        p.domain = params["domain"]
    if "context" in params:
        p.context = params["context"]

    # Objectives — full replacement
    if "objectives" in params:
        old_names = {o.name for o in p.objectives}
        p.objectives = [Objective(**o) if isinstance(o, dict) else o for o in params["objectives"]]
        new_names = {o.name for o in p.objectives}
        # Drop scores and constraints referencing removed objectives
        removed = old_names - new_names
        if removed:
            p.scores = [s for s in p.scores if s.objective not in removed]
            p.constraints = [
                c for c in p.constraints
                if not (c.type == "objective_bound" and c.objective in removed)
            ]
            p.interaction_matrices = [
                m for m in p.interaction_matrices if m.objective not in removed
            ]
        structural_change = True

    # Options — full replacement
    if "options" in params:
        old_names = {o.name for o in p.options}
        p.options = [
            Option(name=o) if isinstance(o, str) else
            Option(**o) if isinstance(o, dict) else o
            for o in params["options"]
        ]
        new_names = {o.name for o in p.options}
        # Drop scores and constraints referencing removed options
        removed = old_names - new_names
        if removed:
            p.scores = [s for s in p.scores if s.option not in removed]

            def _constraint_references_removed(c, removed_opts):
                if c.type in ("force_include", "force_exclude") and c.option in removed_opts:
                    return True
                if c.type == "exclusion_pair" and (c.option_a in removed_opts or c.option_b in removed_opts):
                    return True
                if c.type == "dependency" and (c.if_option in removed_opts or c.then_option in removed_opts):
                    return True
                if c.type == "group_limit" and any(o in removed_opts for o in c.options):
                    return True
                return False

            p.constraints = [c for c in p.constraints if not _constraint_references_removed(c, removed)]
        structural_change = True

    # Scores — merge semantics (upsert by option+objective)
    if "scores" in params:
        new_scores = [Score(**s) if isinstance(s, dict) else s for s in params["scores"]]
        valid_opts = {o.name for o in p.options}
        valid_objs = {o.name for o in p.objectives}
        invalid = [
            s for s in new_scores
            if s.option not in valid_opts or s.objective not in valid_objs
        ]
        if invalid:
            bad_refs = [
                f"{s.option}/{s.objective}" for s in invalid
            ]
            return {"error": f"Scores reference unknown options/objectives: {bad_refs}"}
        score_map = {(s.option, s.objective): s for s in p.scores}
        for s in new_scores:
            score_map[(s.option, s.objective)] = s
        p.scores = list(score_map.values())
        structural_change = True

    # Constraints — full replacement
    if "constraints" in params:
        p.constraints = [_parse_constraint(c) for c in params["constraints"]]
        structural_change = True

    # Approach
    if "approach" in params:
        p.approach = Approach(params["approach"])
        structural_change = True

    # Scenario config
    if "scenario_config" in params:
        sc = params["scenario_config"]
        if isinstance(sc, dict):
            # Parse nested scenarios with score overrides and adjustments
            scenarios = []
            for s in sc.get("scenarios", []):
                overrides = [Score(**o) if isinstance(o, dict) else o for o in s.get("score_overrides", [])]
                adjustments = [
                    ScoreAdjustment(**a) if isinstance(a, dict) else a
                    for a in s.get("score_adjustments", [])
                ]
                constraint_ov = [
                    _parse_constraint(c) for c in s.get("constraint_overrides", [])
                ]
                matrix_ov = [
                    InteractionMatrix(**m) if isinstance(m, dict) else m
                    for m in s.get("interaction_matrix_overrides", [])
                ]
                scenarios.append(Scenario(
                    name=s["name"],
                    probability=s.get("probability"),  # optional
                    description=s.get("description", ""),
                    score_overrides=overrides,
                    score_adjustments=adjustments,
                    constraint_overrides=constraint_ov,
                    interaction_matrix_overrides=matrix_ov,
                ))
            p.scenario_config = ScenarioConfig(enabled=sc.get("enabled", True), scenarios=scenarios)
        structural_change = True

    # Interaction matrices — upsert by objective name
    if "interaction_matrices" in params:
        im_map = {m.objective: m for m in p.interaction_matrices}
        for im_dict in params["interaction_matrices"]:
            im = InteractionMatrix(**im_dict) if isinstance(im_dict, dict) else im_dict
            im_map[im.objective] = im
        p.interaction_matrices = list(im_map.values())
        structural_change = True

    # Reference points — full replacement (interpretive, no structural impact)
    if "reference_points" in params:
        p.reference_points = [
            ReferencePoint(**rp) if isinstance(rp, dict) else rp
            for rp in params["reference_points"]
        ]

    # Mark results stale on structural change (preserve run for comparison)
    if structural_change:
        p.results_stale = True

    p.updated_at = datetime.now(timezone.utc)
    store.save(p)

    total_possible = len(p.objectives) * len(p.options)
    result = {
        "problem_id": p.problem_id,
        "updated_at": p.updated_at.isoformat(),
        "status": {
            "objectives": len(p.objectives),
            "options": len(p.options),
            "scores_complete": round(len(p.scores) / total_possible, 2) if total_possible > 0 else 0.0,
            "has_run": p.run is not None,
            "has_exact_run": p.exact_run is not None,
            "results_stale": p.results_stale,
            "total_runs": len(p.runs) + (1 if p.run else 0),
        },
    }

    # Include metrics on structural changes so the LLM can coach the user
    if structural_change:
        result["metrics"] = {
            "framing": metrics.framing_metrics(p),
            "data": metrics.data_metrics(p),
        }

        # Surface validation issues that are NOT about incomplete scoring
        # (incomplete scores are expected during setup). Catches broken
        # constraints, unknown option references, etc. — fail-fast.
        try:
            vr = optimizer.validate(p)
            non_noise = [
                json.loads(i.model_dump_json())
                for i in vr.issues
                if "Score matrix incomplete" not in i.message
            ]
            if non_noise:
                result["validation_issues"] = non_noise
        except Exception:
            # Never block a model update on validation errors — they're advisory
            pass

    # ─── Skill auto-injection based on what changed ───
    added_objectives = "objectives" in params
    added_options = "options" in params
    scores_complete = result["status"]["scores_complete"]

    # If structure was (re)defined, inject data_collection for scoring guidance
    if (added_objectives or added_options) and not _was_injected(pid, "data_collection"):
        _inject_skill(result, "data_collection",
            "Objectives/options defined. Use this guide for scoring — "
            "anchor best/worst first, batch by objective, push for 100% completeness.")
        _mark_injected(pid, "data_collection")

    # If scores just reached 100%, inject optimization_strategy for solve guidance
    elif scores_complete == 1.0 and not _was_injected(pid, "optimization_strategy"):
        _inject_skill(result, "optimization_strategy",
            "Score matrix is 100% complete. Use this guide before running solve — "
            "it covers mode selection, constraint strategy, and iteration expectations.")
        _mark_injected(pid, "optimization_strategy")

    # Reset optimization_strategy injection only when the problem's *shape* changes
    # (objectives or options) — those invalidate the methodology guidance.
    # Refinements (constraints, interaction_matrices, scenarios, reference_points,
    # approach flip, score updates) don't warrant re-firing the skill.
    # Skip reset when scores are in the same call — that means the inject block above
    # already fired fresh guidance for the current shape.
    problem_shape_change = ("objectives" in params or "options" in params) and "scores" not in params
    if problem_shape_change:
        _reset_injection(pid, "optimization_strategy")

    # Any structural edit invalidates prior solve results — re-arm
    # solution_interpreter so the next solve re-injects fresh guidance.
    if structural_change:
        _reset_injection(pid, "solution_interpreter")

    return result


def _model_get(params: dict) -> dict:
    pid = params.get("problem_id")
    if not pid:
        return {"error": "problem_id is required."}
    try:
        p = store.load(pid)
    except FileNotFoundError:
        return {"error": f"Problem {pid} not found."}

    section = params.get("section")
    if section:
        return _model_get_section(p, section)

    return json.loads(p.model_dump_json())


def _fmt_num(v) -> str:
    """Trim a trailing .0 so integer-valued floats read as integers (140.0 → 140)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() else str(f)


def _format_constraint(c, units: dict | None = None) -> str:
    """Human-readable summary carrying the constraint's actual content — option
    names, bounds, group members — so the formulation card reads as a precise
    problem, not a list of constraint types. ``units`` maps objective→unit for bounds."""
    d = c.model_dump(mode="json") if hasattr(c, "model_dump") else dict(c)
    t = d.get("type")
    units = units or {}
    if t == "max_allocation":
        return f"≤{_fmt_num(d.get('max'))}% per option"
    if t == "min_allocation":
        return f"≥{_fmt_num(d.get('min'))}% per option"
    if t == "objective_bound":
        op = {"max": "≤", "min": "≥"}.get(d.get("operator"), str(d.get("operator")))
        obj = d.get("objective")
        unit = units.get(obj, "")
        return f"{obj} {op} {_fmt_num(d.get('value'))}{(' ' + unit) if unit else ''}"
    if t == "cardinality":
        lo, hi = d.get("min"), d.get("max")
        return f"select exactly {lo}" if lo == hi else f"select {lo}–{hi}"
    if t == "force_include":
        return f"must include {d.get('option')}"
    if t == "force_exclude":
        return f"exclude {d.get('option')}"
    if t == "exclusion_pair":
        return f"{d.get('option_a')} ⊕ {d.get('option_b')}"
    if t == "dependency":
        return f"{d.get('if_option')} requires {d.get('then_option')}"
    if t == "group_limit":
        opts = d.get("options", [])
        members = ", ".join(opts) if 0 < len(opts) <= 5 else f"{len(opts)} options"
        return f"≤{_fmt_num(d.get('max'))} of {{{members}}}"
    return t or "constraint"


def _viz_data_formulation(p: Problem) -> dict:
    """Structured formulation card for the web UI: typed objectives, constraints, scenarios."""
    total = len(p.objectives) * len(p.options)
    units = {o.name: o.unit for o in p.objectives}
    return {
        "type": "formulation",
        "name": p.name,
        "domain": p.domain,
        "approach": p.approach.value,
        "options_count": len(p.options),
        "scores_complete": round(len(p.scores) / total, 2) if total else 0.0,
        "objectives": [
            {
                "name": o.name,
                "direction": o.direction.value,
                "aggregation": o.aggregation.value,
                "unit": o.unit,
            }
            for o in p.objectives
        ],
        "constraints": [_format_constraint(c, units) for c in p.constraints],
        "scenarios": [s.name for s in p.scenario_config.scenarios] if p.scenario_config else [],
    }


def _render_formulation(p: Problem) -> str:
    """Formulation as readable text — the ASCII/MD equivalent of the web UI's
    formulation card, for chat / coding-agent surfaces."""
    lines = [f"─── Formulation: {p.name} ({p.domain} · {p.approach.value}) ───", "", "Objectives:"]
    for o in p.objectives:
        arrow = "max ↑" if o.direction.value == "maximize" else "min ↓"
        unit = f" {o.unit}" if o.unit else ""
        lines.append(f"  • {o.name}: {arrow}, {o.aggregation.value}{unit}")
    if p.constraints:
        units = {o.name: o.unit for o in p.objectives}
        lines.append("")
        lines.append("Constraints:")
        for c in p.constraints:
            lines.append(f"  • {_format_constraint(c, units)}")
    scens = [s.name for s in p.scenario_config.scenarios] if p.scenario_config else []
    if scens:
        lines.append("")
        lines.append("Scenarios: " + ", ".join(scens))
    lines.append("")
    total = len(p.objectives) * len(p.options)
    pct = f", scores {round(100 * len(p.scores) / total)}% complete" if total and len(p.scores) < total else ""
    lines.append(f"{len(p.options)} options{pct}")
    return "\n".join(lines)


def _model_get_section(p: Problem, section: str) -> dict:
    """Return a targeted slice of the Problem for progressive inspection.

    Sections: summary, objectives, options, scores, constraints, matrices,
    scenarios, run, runs, curated, references.
    """
    header = {"problem_id": p.problem_id, "section": section}
    match section:
        case "summary":
            return {
                **header,
                "name": p.name,
                "domain": p.domain,
                "approach": p.approach.value,
                "objectives_count": len(p.objectives),
                "options_count": len(p.options),
                "scores_count": len(p.scores),
                "constraints_count": len(p.constraints),
                "interaction_matrices_count": len(p.interaction_matrices),
                "scenarios_count": len(p.scenario_config.scenarios) if p.scenario_config else 0,
                "has_run": p.run is not None,
                "has_exact_run": p.exact_run is not None,
                "has_scenario_run": p.scenario_run is not None,
                "results_stale": p.results_stale,
                "curated_count": len(p.curated_solutions),
                "visualization": _render_formulation(p),
                "viz_data": _viz_data_formulation(p),
            }
        case "objectives":
            return {**header, "objectives": [o.model_dump(mode="json") for o in p.objectives]}
        case "options":
            return {**header, "options": [o.model_dump(mode="json") for o in p.options]}
        case "scores":
            return {**header, "scores": [s.model_dump(mode="json") for s in p.scores]}
        case "constraints":
            return {**header, "constraints": [c.model_dump(mode="json") for c in p.constraints]}
        case "matrices":
            return {**header, "interaction_matrices": [m.model_dump(mode="json") for m in p.interaction_matrices]}
        case "scenarios":
            return {
                **header,
                "scenario_config": p.scenario_config.model_dump(mode="json") if p.scenario_config else None,
            }
        case "run":
            return {**header, "run": p.run.model_dump(mode="json") if p.run else None}
        case "runs":
            return {**header, "runs": [r.model_dump(mode="json") for r in p.runs]}
        case "exact_run":
            return {**header, "exact_run": p.exact_run.model_dump(mode="json") if p.exact_run else None}
        case "curated":
            return {**header, "curated": [c.model_dump(mode="json") for c in p.curated_solutions]}
        case "references":
            return {**header, "reference_points": [r.model_dump(mode="json") for r in p.reference_points]}
        case _:
            return {"error": f"Unknown section: {section}. Valid: summary, objectives, options, scores, constraints, matrices, scenarios, run, runs, exact_run, curated, references."}


def _model_list() -> dict:
    # Wrap in a dict so FastMCP serializes to a TextContent block — bare list
    # returns aren't serialized by FastMCP 1.25.0, leaving the Anthropic MCP
    # connector with no usable tool_result content.
    return {"problems": store.list()}


def _model_delete(params: dict) -> dict:
    pid = params.get("problem_id")
    if not pid:
        return {"error": "problem_id is required."}
    try:
        store.delete(pid)
    except FileNotFoundError:
        return {"error": f"Problem {pid} not found."}
    _reset_all_injections(pid)
    return {"deleted": pid}


def _model_save(params: dict) -> dict:
    pid = params.get("problem_id")
    if not pid:
        return {"error": "problem_id is required."}
    try:
        p = store.load(pid)
    except FileNotFoundError:
        return {"error": f"Problem {pid} not found."}
    name = problem_io.slugify(params.get("save_as") or p.name)
    info = problem_io.save_problem(p, name)
    includes_solutions = "solutions" in info["files"]
    return {
        "saved": True,
        "problem_id": pid,
        "name": info["name"],
        "dir": info["dir"],
        "files": list(info["files"].keys()),
        "includes_solutions": includes_solutions,
        "note": (
            "Saved in the portable examples format. Reload anytime with "
            f'model load source="{info["name"]}".'
        ),
    }


def _model_load(params: dict) -> dict:
    source = params.get("source")
    if not source:
        return {
            "error": "source is required — the example or saved-problem name to load.",
            "available": problem_io.list_available(),
        }
    try:
        p = problem_io.load_problem(source)
    except FileNotFoundError as e:
        return {"error": str(e), "available": problem_io.list_available()}
    except ValueError as e:
        return {"error": str(e)}

    store.save(p)
    _reset_all_injections(p.problem_id)

    total = len(p.objectives) * len(p.options)
    result = {
        "problem_id": p.problem_id,
        "name": p.name,
        "domain": p.domain,
        "loaded_from": source,
        "approach": p.approach.value,
        "status": {
            "objectives": len(p.objectives),
            "options": len(p.options),
            "scores_complete": round(len(p.scores) / total, 2) if total else 0.0,
            "constraints": len(p.constraints),
            "scenarios": len(p.scenario_config.scenarios) if p.scenario_config else 0,
            "interaction_matrices": len(p.interaction_matrices),
            "has_run": p.run is not None,
            "has_exact_run": p.exact_run is not None,
            "has_scenario_run": p.scenario_run is not None,
            "results_stale": p.results_stale,
            "curated": len(p.curated_solutions),
        },
    }

    # Next-phase guidance: restored, fresh results mean explore; otherwise solve.
    has_results = (p.run is not None or p.exact_run is not None
                   or p.scenario_run is not None) and not p.results_stale
    if has_results:
        _inject_skill(result, "solution_interpreter",
            "Loaded a solved problem with results. Use this guide to present the "
            "frontier — never say 'best', start with extremes and the balanced solution.")
        _mark_injected(p.problem_id, "solution_interpreter")
    else:
        _inject_skill(result, "optimization_strategy",
            "Problem loaded and ready. Use this guide before solving — it covers "
            "mode selection, constraint strategy, and iteration expectations.")
        _mark_injected(p.problem_id, "optimization_strategy")
    return result


def _parse_constraint(c: dict | Constraint) -> Constraint:
    if not isinstance(c, dict):
        return c
    ctype = c.get("type")
    match ctype:
        case "cardinality":
            return CardinalityConstraint(**c)
        case "force_include":
            return ForceIncludeConstraint(**c)
        case "force_exclude":
            return ForceExcludeConstraint(**c)
        case "objective_bound":
            return ObjectiveBoundConstraint(**c)
        case "exclusion_pair":
            return ExclusionPairConstraint(**c)
        case "dependency":
            return DependencyConstraint(**c)
        case "group_limit":
            return GroupLimitConstraint(**c)
        case "max_allocation":
            return MaxAllocationConstraint(**c)
        case _:
            raise ValueError(f"Unknown constraint type: {ctype}")


# ─── Tool 2: solve ───


@mcp.tool()
def solve(
    action: str,
    problem_id: str,
    mode: str | None = None,
    max_solutions: int | None = None,
    seed: int | None = None,
    solver: str | None = None,
    exact: bool = False,
) -> dict:
    """Validate and run the optimizer.

    Skills auto-inject on phase completions: `optimization_strategy` when scores
    reach 100% or validation passes, `solution_interpreter` on the first solve
    per problem. See the `model` docstring for `_skill_guidance` shape and the
    once-per-problem throttle.

    Actions:
      validate       — Check if problem is ready. Returns issues, missing scores, and the
                       available solvers (`solvers` block) for this environment.
      run            — Validate, then optimize. Returns Pareto frontier + solution_interpreter skill.
                       Side effects: archives the previous run; clears results_stale; persists full
                       results to full_result_path on disk.
      run_scenarios  — Run optimization independently per scenario. Requires scenario_config.

    Args:
      mode: "fast" (default) for quick exploration iterations, "thorough" for
            final convergence when the problem is refined. Adapts population size,
            generations, and operator parameters to problem complexity. Also selects
            NSGA-III for 4+ objectives.
      max_solutions: Cap the number of Pareto solutions returned (default 100).
            Useful for proportional mode which can produce large frontiers.
      seed: Optional RNG seed for reproducibility, echoed as `seed_used` on every
            run. Same problem state + same seed → the same frontier (in- and
            cross-process); omit it and a fresh seed is drawn and recorded. Vary
            seeds to check frontier stability; fix the seed to reproduce or share a
            result. For `run_scenarios`, the parent seed is propagated to each
            scenario (per-scenario `seed_used` derived via SHA-256 of scenario name
            + parent seed), so pinning the parent reproduces the whole run.
      solver: Which engine to run. Omit (or "nsga") for the default NSGA-II/III
            evolutionary search — it fits any problem shape and is the right choice for
            exploration and most runs. "highs" (CPU) or "cuopt" (GPU) select an OPTIONAL
            exact backend that solves each frontier point to optimality for its scalarization
            (optimal to a 0.1% gap, not heuristic; add `exact=true` to certify a zero MILP
            gap) — reach for it on a final/decision run over a supported shape
            (binary selection, or a quadratic mean-variance portfolio), or to certify an
            EA frontier. If the requested solver isn't installed or the shape doesn't fit,
            this returns a clear error listing the available solvers — check `solve
            validate`'s `solvers` block first. The engine that ran is echoed as
            `solver_used`. See the optimization_strategy skill ("Exact Solvers").
      exact: Exact backends only — certify each MILP inner solve to a zero optimality gap
            instead of the default speed-oriented bounded solve. Slower; use when stakeholders
            need certified optimality. No-op on the always-exact QP and on the default NSGA path.

    Key guidance:
    - Expect iteration: first run is exploration, not the answer.
    - Use "fast" while refining scores/constraints, "thorough" when finalized.
    - Default NSGA explores; opt into an exact `solver` for provable optimality on a final run.
    - If zero solutions: check cardinality constraints and objective bounds for conflicts.
    - After re-run: check explore/curated to see which curated solutions survived.
    - 5-10 solutions is healthy; very few = constraints too tight.
    """
    try:
        p = store.load(problem_id)
    except FileNotFoundError:
        return {"error": f"Problem {problem_id} not found."}

    opt_mode = None
    if mode is not None:
        try:
            opt_mode = OptimizeMode(mode)
        except ValueError:
            return {"error": f"Invalid mode '{mode}'. Use 'fast' or 'thorough'."}

    match action:
        case "validate":
            vr = optimizer.validate(p)
            result = json.loads(vr.model_dump_json())
            result["solvers"] = _solver_availability(p)
            # If ready to solve, inject optimization_strategy
            if vr.ready and not _was_injected(problem_id, "optimization_strategy"):
                _inject_skill(result, "optimization_strategy",
                    "Problem validates as ready. Review this guide before running solve.")
                _mark_injected(problem_id, "optimization_strategy")
            return result
        case "run":
            return _solve_run(p, opt_mode, max_solutions=max_solutions, seed=seed,
                              solver=solver, exact=exact)
        case "run_scenarios":
            return _solve_run_scenarios(p, opt_mode, max_solutions=max_solutions, seed=seed,
                                        solver=solver, exact=exact)
        case _:
            return {"error": f"Unknown action: {action}. Use validate/run/run_scenarios."}


def _solver_availability(p: Problem) -> dict:
    """Which solver engines this environment can run, and whether the optional exact
    backends fit this problem's shape. Surfaced on solve/validate so the agent knows
    before requesting an exact `solver` — keeps the choice informed, not trial-and-error."""
    from solvers import EXACT_SOLVERS, available_solvers, exact_solver_fits

    avail = available_solvers()
    fits, reason = exact_solver_fits(p)
    return {
        "default": "nsga",
        "available": [k for k, ok in avail.items() if ok],
        "exact_available": [k for k in EXACT_SOLVERS if avail.get(k)],
        "exact_fits_shape": fits,
        "exact_shape_note": reason or "shape supported (binary selection / mean-variance portfolio)",
    }


def _resolve_solver(p: Problem, solver: str | None) -> tuple[str | None, dict | None]:
    """Validate an agent-requested solver before running. Returns ``(solver, None)`` to run,
    or ``(None, error_dict)`` to abort with actionable guidance. Default/NSGA → ``(None, None)``
    so the optimizer takes its normal path. Guards both availability (is the backend
    installed) and shape fit (does it solve this problem), so the exact path never silently
    degrades to NSGA — provable optionality means the agent knows which engine actually ran."""
    from solvers import available_solvers, exact_solver_fits

    if solver is None:
        return None, None
    key = solver.strip().lower()
    if key in ("", "nsga", "auto", "default"):
        return None, None
    avail = available_solvers()
    if key not in avail:
        return None, {"error": f"Unknown solver '{solver}'. Options: "
                      f"{', '.join(sorted(avail))} (default: nsga)."}
    if not avail[key]:
        installed = ", ".join(k for k, ok in avail.items() if ok)
        return None, {"error": f"Solver '{key}' is not available here — its dependency isn't "
                      f"installed. Available: {installed}. Run with the default solver, or install "
                      "the backend (highs: `pip install highspy`; cuopt: NVIDIA GPU + cuopt-cu12)."}
    fits, reason = exact_solver_fits(p)
    if not fits:
        return None, {"error": f"Solver '{key}' doesn't fit this problem: {reason}. "
                      "Use the default solver (nsga) for this shape."}
    return key, None


def _solve_run(p: Problem, mode: OptimizeMode | None = None, max_solutions: int | None = None,
               seed: int | None = None, solver: str | None = None, exact: bool = False) -> dict:
    vr = optimizer.validate(p)
    if not vr.ready:
        vr_data = json.loads(vr.model_dump_json())
        return {"ready": False, "issues": vr_data["issues"], "missing_scores": vr_data["missing_scores"]}

    solver, solver_err = _resolve_solver(p, solver)
    if solver_err:
        return solver_err

    try:
        run = optimizer.optimize(p, mode=mode, max_solutions=max_solutions, seed=seed,
                                 solver=solver, exact=exact)
    except Exception as e:
        return {"feasible": False, "error": str(e)}

    if not run.solutions:
        analysis = optimizer.analyze_infeasibility(p)
        return {
            "feasible": False,
            **analysis,
        }

    # Snapshot constraints on the run
    run.constraints_snapshot = [c.model_dump() for c in p.constraints]

    # An exact solve is an *overlay* — store it in exact_run and leave the
    # exploratory `run` (NSGA) intact, so a problem can hold both frontiers at once
    # (explore-with-EA, then certify). A default/NSGA run replaces `run` and archives
    # the prior one for comparison.
    #
    # A solve only refreshes the frontier it produces. If results were stale (the
    # problem was edited since the last solve), the *other* stored frontier predates
    # that edit and is no longer valid — drop it, so clearing results_stale below
    # doesn't vouch for a frontier that no longer matches the problem.
    from solvers import EXACT_SOLVERS

    was_stale = p.results_stale
    if run.solver in EXACT_SOLVERS:
        p.exact_run = run
        if was_stale:
            p.run = None
    else:
        if p.run is not None:
            p.runs.append(p.run)
        p.run = run
        if was_stale:
            p.exact_run = None
    p.results_stale = False
    store.save(p)

    # Persist the full run result to disk so agents can consume it without
    # pulling the whole payload into context. Path is returned in the response.
    frontier_complete = run.total_pareto_found <= len(run.solutions)
    frontier_quality = metrics.frontier_quality(run.solutions, p.objectives, run.quality.spacing_cv)

    full_payload = {
        "run_id": run.run_id,
        "problem_id": p.problem_id,
        "solutions_found": len(run.solutions),
        "total_pareto_found": run.total_pareto_found,
        "frontier_complete": frontier_complete,
        "frontier_quality": frontier_quality,
        "solutions": [json.loads(s.model_dump_json()) for s in run.solutions],
        "quality": json.loads(run.quality.model_dump_json()),
        "constraints_snapshot": run.constraints_snapshot,
        "mode": run.mode.value if run.mode else None,
        "seed_used": run.seed_used,
        "solver_used": run.solver,
        "exact": run.exact,
    }
    full_result_path = store.write_run_result(p.problem_id, run.run_id, full_payload)

    result = {
        "run_id": run.run_id,
        "solutions_found": len(run.solutions),
        "total_pareto_found": run.total_pareto_found,
        "frontier_complete": frontier_complete,
        "frontier_quality": frontier_quality,
        "seed_used": run.seed_used,
        "solver_used": run.solver,
        "exact": run.exact,
        "objective_ranges": _objective_ranges(run.solutions, p.objectives),
        "preview": _solve_preview(run.solutions, p.objectives),
        "quality": json.loads(run.quality.model_dump_json()),
        "metrics": {
            "solve": metrics.solve_metrics(p),
            "diagnostics": metrics.diagnostics(p),
        },
        "full_result_path": str(full_result_path),
        "next_steps": (
            ("This is an exact overlay, stored alongside the exploratory NSGA frontier — run "
             "`explore certify` (no params) to audit it: dominance audit, the NSGA-never-dominates "
             "invariant, and per-objective corner sharpening. Then use "
             if run.solver in EXACT_SOLVERS else "Use ")
            + "`explore tradeoffs` for the frontier overview, `explore solution <id>` for a "
            "single solution's detail, and `explore curate` to pin strategies. For bulk export "
            "or assembling artifacts that need every solution with allocations, read the file at "
            "`full_result_path` directly — it contains all solutions without token overhead. "
            "(`explore solutions` is available with optional detail=true but `full_result_path` "
            "is the preferred path for anything beyond a few dozen solutions.)"
        ),
    }

    # Inject solution_interpreter once per problem (until structural change resets it).
    if not _was_injected(p.problem_id, "solution_interpreter"):
        _inject_skill(result, "solution_interpreter", _SOLUTION_INTERPRETER_PROMPT)
        _mark_injected(p.problem_id, "solution_interpreter")
    # Reset optimization_strategy so it can re-fire on the next model change cycle
    _reset_injection(p.problem_id, "optimization_strategy")

    return result


def _objective_ranges(solutions: list, objectives: list) -> dict:
    """min/max per objective across the Pareto set."""
    if not solutions:
        return {}
    ranges = {}
    for obj in objectives:
        vals = [s.objective_values.get(obj.name) for s in solutions if obj.name in s.objective_values]
        if vals:
            ranges[obj.name] = {"min": round(min(vals), 4), "max": round(max(vals), 4)}
    return ranges


def _solve_preview(solutions: list, objectives: list) -> dict:
    """Compact preview: extremes per objective + balanced solution, by id + objective_values only.

    Full solution detail (allocations, selected_options) retrievable via `explore solution <id>`.
    """
    if not solutions:
        return {}
    extremes = {}
    for obj in objectives:
        name = obj.name
        reverse = obj.direction.value == "maximize"
        best = max(solutions, key=lambda s: s.objective_values.get(name, 0)) if reverse \
            else min(solutions, key=lambda s: s.objective_values.get(name, 0))
        extremes[f"extreme_{name}"] = {
            "solution_id": best.solution_id,
            "objective_values": best.objective_values,
        }

    preview: dict = {"extremes": extremes}
    try:
        from engine.explorer import _find_balanced
        balanced = _find_balanced(solutions, objectives)
        if balanced is not None:
            preview["balanced"] = {
                "solution_id": balanced.solution_id,
                "objective_values": balanced.objective_values,
            }
    except Exception:
        pass
    return preview


def _solve_run_scenarios(p: Problem, mode: OptimizeMode | None = None, max_solutions: int | None = None,
                         seed: int | None = None, solver: str | None = None, exact: bool = False) -> dict:
    vr = optimizer.validate(p)
    if not vr.ready:
        vr_data = json.loads(vr.model_dump_json())
        return {"ready": False, "issues": vr_data["issues"]}

    solver, solver_err = _resolve_solver(p, solver)
    if solver_err:
        return solver_err

    try:
        scenario_results = optimizer.optimize_scenarios(p, mode=mode, max_solutions=max_solutions,
                                                        seed=seed, solver=solver, exact=exact)
    except ValueError as e:
        return {"error": str(e)}

    p.scenario_run = ScenarioRun(scenario_runs=scenario_results)
    p.results_stale = False
    store.save(p)

    summary = {}
    result_paths: dict[str, str] = {}
    for name, run in scenario_results.items():
        frontier_complete = run.total_pareto_found <= len(run.solutions)
        frontier_quality = metrics.frontier_quality(run.solutions, p.objectives, run.quality.spacing_cv)

        full_payload = {
            "run_id": run.run_id,
            "problem_id": p.problem_id,
            "scenario": name,
            "solutions_found": len(run.solutions),
            "total_pareto_found": run.total_pareto_found,
            "frontier_complete": frontier_complete,
            "frontier_quality": frontier_quality,
            "solutions": [json.loads(s.model_dump_json()) for s in run.solutions],
            "quality": json.loads(run.quality.model_dump_json()),
            "mode": run.mode.value if run.mode else None,
            "seed_used": run.seed_used,
            "solver_used": run.solver,
            "exact": run.exact,
        }
        path = store.write_run_result(p.problem_id, run.run_id, full_payload, scenario=name)
        result_paths[name] = str(path)

        summary[name] = {
            "solutions_found": len(run.solutions),
            "total_pareto_found": run.total_pareto_found,
            "frontier_complete": frontier_complete,
            "frontier_quality": frontier_quality,
            "quality": json.loads(run.quality.model_dump_json()),
            "seed_used": run.seed_used,
            "solver_used": run.solver,
            "exact": run.exact,
            "full_result_path": str(path),
        }

    result = {
        "scenarios_optimized": len(scenario_results),
        "results": summary,
        "full_result_paths": result_paths,
        "next_steps": (
            "Per-scenario full solution sets are on disk at the paths in `full_result_paths`. "
            "For interactive navigation use `explore` with scenario=<name> (tradeoffs, solution, "
            "curate, marginal_analysis, scenario_results). For assembling artifacts that need "
            "every solution per scenario, read the files at `full_result_paths` directly — this "
            "is the preferred path for bulk export (no token overhead)."
        ),
    }

    # Inject solution_interpreter once per problem (until structural change resets it).
    if not _was_injected(p.problem_id, "solution_interpreter"):
        _inject_skill(result, "solution_interpreter", _SOLUTION_INTERPRETER_PROMPT)
        _mark_injected(p.problem_id, "solution_interpreter")
    # Reset optimization_strategy so it can re-fire on the next model change cycle
    _reset_injection(p.problem_id, "optimization_strategy")

    return result


def _format_explore(result: dict) -> dict:
    """Compact explore output: visualization first, redundant bulk trimmed.

    Keeps the dict return type (tests and consumers index into it) but trims
    fields already covered by the visualization to prevent MCP truncation.
    """
    # Compact extreme_solutions — keep id + value, drop option lists
    # (agent can use `explore solution <id>` for full detail)
    if "extreme_solutions" in result:
        result["extreme_solutions"] = {
            name: {"solution_id": ext["solution_id"], "value": ext["value"]}
            for name, ext in result["extreme_solutions"].items()
        }

    # Compact scenario_specific_options — group by scenario (less verbose)
    if "scenario_specific_options" in result:
        by_scenario: dict[str, list[str]] = {}
        for opt, scenarios in result["scenario_specific_options"].items():
            for s in scenarios:
                by_scenario.setdefault(s, []).append(opt)
        result["scenario_specific"] = by_scenario
        del result["scenario_specific_options"]

    return result


# ─── Tool 3: explore ───


@mcp.tool()
def explore(
    action: str,
    problem_id: str,
    solution_ids: list[int] | None = None,
    solution_id: int | None = None,
    rating: int | None = None,
    notes: str | None = None,
    stage: str | None = None,
    run_ids: list[str] | None = None,
    custom_name: str | None = None,
    content_signature: str | None = None,
    signatures: list[str] | None = None,
    scenario: str | None = None,
    source: str | None = None,
    detail: bool = False,
    cvar_alpha: float | None = None,
    format: str | None = None,
) -> dict:
    """Navigate results after solving.

    `solution_interpreter` auto-injects on the first solve per problem (see the
    `model` docstring for `_skill_guidance` shape and the once-per-problem throttle).
    Structural `model/update` edits re-arm so the next solve re-injects.

    full_result_path: large-result actions (solutions detail=true, marginal_analysis detail=true)
    write a full JSON dump to disk; the path is in solve/run's response — reference it instead
    of re-requesting.
    content_signature: stable hash of a solution's selected options (and allocations for
    proportional problems); survives re-solves, so prefer it over solution_id when referencing
    curated solutions or recording feedback across runs.

    Actions:
      tradeoffs  — Frontier overview.
                   Returns: objective_ranges, key_tradeoffs, extreme_solutions
                   (per-objective best/worst IDs, keyed `extreme_<objective>`),
                   balanced_solution (ideal-point closest), inflection_point_candidates
                   (solutions where marginal tradeoff cost jumps sharply — diminishing
                   returns boundaries), frontier_shape, binding_analysis.
                   Optional: scenario.
      compare    — Side-by-side comparison.
                   Requires: solution_ids (list of 2+ ints). Optional: scenario.
      solutions  — Pareto frontier listing.
                   Default: compact — solution_id + objective_values + content_signature per solution.
                   Pass detail=true for full dump including selected_options and allocations.
                   Optional: detail, scenario.
      solution   — Single solution detail.
                   Requires: solution_id (int). Optional: scenario.
      feedback   — Record user feedback. Feedback links to content_signature (stable across runs)
                   and attaches to curated solutions.
                   Requires: solution_id OR content_signature.
                   Optional: rating (1-5), notes, stage.
      compare_runs — Compare run history.
                   Requires: run_ids (list of 2+ run ID strings).
      certify    — Audit the exploratory NSGA frontier against the exact overlay: how many
                   NSGA points the exact frontier dominates (heuristic slack), the invariant that
                   NSGA dominates no exact point, and per-objective corner sharpening (strongest at
                   the convex risk/variance corner). The explore→certify workflow made measurable.
                   No params needed — audits `run` (NSGA) against `exact_run` (the exact
                   overlay), so the flow is: solve(), then solve(solver="highs"|"cuopt"), then
                   certify. Optional run_ids (exactly 2 — one NSGA, one exact, order-free) overrides
                   with explicit historical runs.
      scenario_results — Per-scenario robustness analysis with frequency-weighted option importance.
                   Returns option_robustness (sorted by importance = avg_frequency × avg_weight),
                   with tiers: core (>50% freq in all scenarios), common (>25%), marginal (<25%).
                   Also: scenario-specific options, expected values (ideal-point, not achievable),
                   scenario_risk per objective (expected / worst_case / best_case / cvar_<alpha>%).
                   Optional: cvar_alpha (float in (0,1), default 0.2) to set the CVaR tail fraction.
      curate     — Add solution to curated set.
                   Requires: solution_id. Optional: custom_name, notes, scenario.
      uncurate   — Remove from curated set.
                   Requires: content_signature.
      rename_curated — Update name.
                   Requires: content_signature, custom_name.
      curated    — List curated solutions. No params.
                   Returns: total_curated, curated_solutions[] each with
                   content_signature, custom_name, selected_options, allocations,
                   objective_values, source_run_id, notes, feedback[], feedback_count,
                   avg_rating, and `in_current_frontier` (survival flag — false means
                   the latest re-solve no longer produces this solution).
      export_curated — Raw formatted export of curated solutions for handoff.
                   Optional: format ("markdown" default pipe table, or "csv").
      compare_curated — Compare curated solutions.
                   Default: compact — shared/differentiating option sets + objective values per solution.
                   Pass detail=true for full selected_options and allocations per solution.
                   Requires: signatures (list of 2+ content_signature strings). Optional: detail.
      marginal_analysis — Marginal rate analysis: cost-per-unit between adjacent solutions, inflection point detection.
                   Default: summary per pair (inflection, stats, top-5 steepest, truncated viz).
                   Pass detail=true for full rates array + untruncated visualization.
                   Optional: detail, scenario.
      sensitivity — Exact-solver explainability: shadow prices + reduced costs.
                   where_to_invest (constraint shadow prices, ranked — marginal objective change
                   per unit relaxed) and near_misses (unheld options by reduced cost — closest to
                   entering), for a reference solution (default balanced; solution_id overrides),
                   plus frontier_shadow_price_trend (diminishing-returns curve). Needs an exact run
                   (solve(solver="highs"|"cuopt")); falls back to the frontier-inferred binding
                   analysis otherwise. Tagged source=solver_exact|frontier_inferred. Continuous/QP
                   only — integer/MILP solutions have no exact duals.
                   Optional: solution_id, scenario, source.

    Scenario param (optional):
      Pass scenario="<name>" to inspect a specific scenario's results instead of the base case.
      Works with: tradeoffs, compare, solutions, solution, curate, marginal_analysis.
      Use scenario_results (no scenario param) for cross-scenario robustness analysis.

    Source param (optional):
      Selects which base-case frontier to analyze: "run" (default) is the exploratory
      NSGA frontier; "exact" is the exact-solver overlay (exact_run). Use
      source="exact" to run tradeoffs/marginal_analysis/etc. over the exact-solver
      frontier (each point optimal to a 0.1% gap) when a problem holds both frontiers
      (explore-with-NSGA, then certify). Works
      with: tradeoffs, compare, solutions, solution, curate, marginal_analysis. When the
      problem has only an exact_run (no NSGA run), the default already falls back to it.
      Ignored when scenario is set (scenario runs are NSGA-only).

    Key guidance:
    - Never say "best" — every Pareto solution is optimal at its tradeoff.
    - Present extremes first, then balanced + inflection points, then ask what resonates.
    - Quantify tradeoffs: "Solution A gains 20% revenue but costs 15% more risk."
    - Elicit preferences via marginal tradeoff questions, not abstract weights.
    - Curated solutions persist across re-runs via content_signature — check `in_current_frontier`.
    - Once 3+ curated: that IS the decision set, present curated first.
    - With scenarios: use option_robustness tiers (core/common/marginal) to identify safe bets.
    """
    try:
        p = store.load(problem_id)
    except FileNotFoundError:
        return {"error": f"Problem {problem_id} not found."}

    match action:
        case "tradeoffs":
            try:
                return _format_explore(explorer.get_tradeoffs(p, scenario=scenario, source=source))
            except ValueError as e:
                return {"error": str(e)}
        case "sensitivity":
            try:
                result = explorer.sensitivity_analysis(
                    p, solution_id=solution_id, scenario=scenario, source=source)
            except ValueError as e:
                return {"error": str(e)}
            # Close the loop back to the decision (duals are local to the reference solution).
            if isinstance(result, dict) and "error" not in result:
                result.setdefault(
                    "next_steps",
                    "Duals are local to this reference solution — route a large shadow price back to "
                    "framing (is that floor/cap negotiable?), or `explore curate` the chosen "
                    "allocation to pin it.")
            return result
        case "compare":
            if not solution_ids or len(solution_ids) < 2:
                return {"error": "solution_ids must contain at least 2 IDs for compare."}
            try:
                return _format_explore(explorer.compare_solutions(p, solution_ids, scenario=scenario, source=source))
            except ValueError as e:
                return {"error": str(e)}
        case "solutions":
            try:
                result = explorer.get_solutions(p, scenario=scenario, detail=detail, source=source)
            except ValueError as e:
                return {"error": str(e)}
            # Soft-cap detail=true payloads — point at full_result_path on disk.
            # `total_solutions` is already on the result, so we don't duplicate it.
            if detail and len(result.get("solutions", [])) > EXPLORE_DETAIL_CAP:
                result["solutions"] = result["solutions"][:EXPLORE_DETAIL_CAP]
                result["truncated"] = True
                result["shown"] = EXPLORE_DETAIL_CAP
                result["full_result_path"] = str(
                    store.run_result_path(p.problem_id, result["run_id"], scenario=scenario)
                )
            return result
        case "solution":
            if solution_id is None:
                return {"error": "solution_id required for solution action."}
            try:
                return explorer.get_solution(p, solution_id, scenario=scenario, source=source)
            except ValueError as e:
                return {"error": str(e)}
        case "feedback":
            return _explore_feedback(p, solution_id, content_signature, rating, notes, stage)
        case "compare_runs":
            if not run_ids or len(run_ids) < 2:
                return {"error": "run_ids must contain at least 2 run IDs for compare_runs."}
            try:
                return explorer.compare_runs(p, run_ids)
            except ValueError as e:
                return {"error": str(e)}
        case "certify":
            from solvers import EXACT_SOLVERS
            # Default: audit the exploratory NSGA run (`p.run`) against the exact overlay
            # (`p.exact_run`) — the "explore with the EA, then certify" flow the exact_run
            # overlay is built for. `run_ids` (exactly 2) overrides with explicit historical runs.
            if run_ids:
                if len(run_ids) != 2:
                    return {"error": "certify run_ids must be exactly 2 (one NSGA, one exact); "
                                     "omit run_ids to certify the current run against the exact overlay."}
                pool = {r.run_id: r for r in p.runs}
                if p.run:
                    pool[p.run.run_id] = p.run
                if p.exact_run:
                    pool[p.exact_run.run_id] = p.exact_run
                missing = [rid for rid in run_ids if rid not in pool]
                if missing:
                    return {"error": f"Run(s) not found: {missing}. Available: {list(pool)}"}
                chosen = [pool[rid] for rid in run_ids]
                exact = [r for r in chosen if (r.solver or "") in EXACT_SOLVERS]
                nsga = [r for r in chosen if (r.solver or "") not in EXACT_SOLVERS]
                if len(exact) != 1 or len(nsga) != 1:
                    return {"error": "certify needs one NSGA run and one exact-solver run; got "
                                     f"solvers {[r.solver for r in chosen]}."}
                nsga_run, exact_run = nsga[0], exact[0]
            else:
                nsga_run, exact_run = p.run, p.exact_run
                if nsga_run is None or exact_run is None:
                    return {"error": "certify needs both an exploratory NSGA run and an exact overlay. "
                                     "Run solve(), then solve(solver='highs'|'cuopt'), then explore certify."}
            try:
                result = explorer.certify_against_exact(p, nsga_run, exact_run)
            except ValueError as e:
                return {"error": str(e)}
            # Surface the read-side guidance ('Reading the Certificate') at the moment the
            # certificate lands, mirroring the solve-response injection.
            if not _was_injected(p.problem_id, "solution_interpreter"):
                _inject_skill(result, "solution_interpreter", _SOLUTION_INTERPRETER_PROMPT)
                _mark_injected(p.problem_id, "solution_interpreter")
            return result
        case "scenario_results":
            try:
                return _format_explore(explorer.get_scenario_results(p, cvar_alpha=cvar_alpha))
            except ValueError as e:
                return {"error": str(e)}
        case "scenario_frontiers":
            try:
                return explorer.get_scenario_frontiers(p)
            except ValueError as e:
                return {"error": str(e)}
        case "curate":
            if solution_id is None:
                return {"error": "solution_id required for curate action."}
            try:
                result = explorer.curate_solution(p, solution_id, custom_name or "", notes or "", scenario=scenario, source=source)
                store.save(p)
                return result
            except ValueError as e:
                return {"error": str(e)}
        case "uncurate":
            if not content_signature:
                return {"error": "content_signature required for uncurate action."}
            try:
                result = explorer.uncurate_solution(p, content_signature)
                store.save(p)
                return result
            except ValueError as e:
                return {"error": str(e)}
        case "rename_curated":
            if not content_signature or not custom_name:
                return {"error": "content_signature and custom_name required for rename_curated."}
            try:
                result = explorer.rename_curated(p, content_signature, custom_name)
                store.save(p)
                return result
            except ValueError as e:
                return {"error": str(e)}
        case "curated":
            return explorer.list_curated(p)
        case "export_curated":
            try:
                return explorer.export_curated(p, format=format or "markdown")
            except ValueError as e:
                return {"error": str(e)}
        case "compare_curated":
            if not signatures or len(signatures) < 2:
                return {"error": "signatures must contain at least 2 content_signature strings."}
            try:
                return _format_explore(explorer.compare_curated(p, signatures, detail=detail))
            except ValueError as e:
                return {"error": str(e)}
        case "marginal_analysis":
            try:
                return explorer.marginal_analysis(p, scenario=scenario, detail=detail, source=source)
            except ValueError as e:
                return {"error": str(e)}
        case _:
            return {"error": f"Unknown action: {action}."}


def _explore_feedback(
    p: Problem,
    solution_id: int | None,
    content_signature: str | None,
    rating: int | None,
    notes: str | None,
    stage: str | None,
) -> dict:
    """Record user feedback on solutions or the overall process.

    Feedback is linked to a content_signature (stable across runs). If only
    solution_id is provided, the signature is computed from the current run.
    Feedback is also appended to any curated solution with the same signature.
    """
    if rating is not None and not (1 <= rating <= 5):
        return {"error": "rating must be 1-5."}

    # Resolve content_signature from solution_id if not provided directly
    sig = content_signature
    frontier = p.run or p.exact_run  # exact-only solves carry the frontier in exact_run
    if sig is None and solution_id is not None and frontier:
        for s in frontier.solutions:
            if s.solution_id == solution_id:
                sig = s.content_signature or _content_signature(
                    s.selected_options, s.allocations
                )
                break

    fb = Feedback(
        content_signature=sig,
        solution_id=solution_id,
        rating=rating,
        notes=notes or "",
        stage=stage or "",
    )

    # Store on problem-level feedback list
    p.feedback.append(fb)

    # Also attach to matching curated solution (preference travels with curation)
    if sig:
        for cs in p.curated_solutions:
            if cs.content_signature == sig:
                cs.feedback.append(fb)
                break

    p.updated_at = datetime.now(timezone.utc)
    store.save(p)

    return {
        "recorded": True,
        "content_signature": sig,
        "feedback_count": len(p.feedback),
        "outcome": metrics.outcome_metrics(p),
    }


# ─── Entry point ───


def _bearer_gate(app, token: str, gated_prefixes: tuple[str, ...]):
    """ASGI wrapper that requires `Authorization: Bearer <token>` on MCP paths.

    Only HTTP requests to the transport endpoints are gated, leaving health
    checks and other paths open. When FRONTIER_MCP_TOKEN is unset the engine
    runs ungated (local dev / single-user self-host); set it to lock a publicly
    hosted engine. Per-user scoping (owner_id) is a later step — this is a
    single shared secret shared by the web app and direct MCP-client users.
    """
    import hmac

    expected = f"Bearer {token}"

    async def gated(scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").startswith(gated_prefixes):
            headers = dict(scope.get("headers") or [])
            provided = headers.get(b"authorization", b"").decode()
            if not hmac.compare_digest(provided, expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await app(scope, receive, send)

    return gated


def main():
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    token = os.environ.get("FRONTIER_MCP_TOKEN")

    # Gate the HTTP transports when a shared token is configured; stdio is
    # local-trust and never gated.
    if token and transport in ("sse", "streamable-http"):
        import uvicorn

        app = mcp.sse_app() if transport == "sse" else mcp.streamable_http_app()
        app = _bearer_gate(app, token, ("/sse", "/messages", "/mcp"))
        uvicorn.run(
            app,
            host=os.environ.get("MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8000")),
        )
    else:
        mcp.run(transport=transport)


if __name__ == "__main__":
    main()
