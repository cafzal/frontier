"""Skill delivery + guidance state: the judgment layer's plumbing.

Owns the skills directory, the name→dir map, core/section loading (get_skill's
resolver), per-problem injection state (the once-per-phase throttle), and the
decision-action guidance pointers. Split from server.py (F2); the server imports
these names, and tests reach them via the server namespace — the mutable state
(_injected_skills) is shared by object identity, so mutation-based test resets
keep working.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

_SKILL_MAP = {
    "problem_framing": "problem_framing",
    "data_collection": "data_collection",
    "optimization_strategy": "optimization_strategy",
    "solution_interpreter": "solution_interpreter",
}

_HEADING_RE = re.compile(r"^(#{2,4})\s+(.*\S)\s*$")


def _skill_files(dirname: str) -> list[Path]:
    """A skill's core (SKILL.md) plus its on-demand reference files."""
    base = SKILLS_DIR / dirname
    files = [base / "SKILL.md"]
    refdir = base / "references"
    if refdir.is_dir():
        files += sorted(refdir.glob("*.md"))
    return [f for f in files if f.exists()]


def _extract_section(text: str, section: str) -> str | None:
    """Return one markdown section (heading through next same-or-higher heading)."""
    want = section.strip().lower()
    out: list[str] = []
    level: int | None = None
    for ln in text.splitlines(keepends=True):
        m = _HEADING_RE.match(ln)
        if m:
            if level is not None and len(m.group(1)) <= level:
                break
            if level is None and m.group(2).strip().lower() == want:
                level = len(m.group(1))
        if level is not None:
            out.append(ln)
    return "".join(out) if out else None


def _section_titles(dirname: str) -> list[str]:
    titles: list[str] = []
    for f in _skill_files(dirname):
        titles += [m.group(2).strip() for m in map(_HEADING_RE.match, f.read_text().splitlines()) if m]
    return titles


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


# Navigation/recording actions (solutions, curated, feedback) intentionally carry no
# pointer — they list or record rather than present a decision read, so there is no
# interpretation guidance to cite.
_DECISION_GUIDANCE: dict[str, tuple[str, str]] = {
    "tradeoffs": ("solution_interpreter", "Presentation Order: Extremes → Balanced → Inflection → Risk → Preference"),
    "compare": ("solution_interpreter", "Differentiating Options"),
    "compare_runs": ("solution_interpreter", "Run Diff Interpretation"),
    "scenario_results": ("solution_interpreter", "Scenario Results Presentation"),
    "scenario_frontiers": ("solution_interpreter", "Scenario Results Presentation"),
    "composition": ("solution_interpreter", "Mining the Solution Set"),
    "marginal_analysis": ("solution_interpreter", "Marginal Analysis Interpretation"),
    "curate": ("solution_interpreter", "Solution Curation"),
    "certify": ("solution_interpreter", "Reading the Certificate (explore certify)"),
    "audit": ("solution_interpreter", "Reading the Audit (explore audit)"),
    "sensitivity": ("solution_interpreter", "Exact Sensitivity — Shadow Prices & Reduced Costs (solver duals)"),
}


def _attach_guidance_pointer(result: dict, action: str) -> dict:
    """Point a decision action's response at the skill section that governs reading it.

    No-op for non-decision actions and for error results (nothing to present). The note's
    conditional phrasing holds whether or not the full skill is also in context: if it is,
    the agent reads on; if it scrolled out, the agent re-fetches — so this composes cleanly
    with the once-per-problem full-skill injection rather than contradicting it."""
    if not isinstance(result, dict) or "error" in result:
        return result
    entry = _DECISION_GUIDANCE.get(action)
    if not entry:
        return result
    skill, section = entry
    # Sensitivity falls back to the frontier-inferred binding analysis when a problem has no
    # exact duals — point at that section instead so the cited guidance matches the output.
    if action == "sensitivity" and result.get("source") == "frontier_inferred":
        section = "Binding Analysis"
    result["guidance_pointer"] = {
        "skill": skill,
        "section": section,
        "note": (f"Present this with the {skill} skill → '{section}'. If that section isn't in "
                 f"recent context, fetch exactly it with get_skill('{skill}', section='{section}') "
                 "before presenting — don't go from memory."),
    }
    return result
