"""Resolver-integrity regression gate for the skill product surface.

Skill content is the product surface: a guidance pointer or a `get_skill(section=…)` that
names a heading which doesn't resolve is a silent broken link, and the highest-frequency way
to introduce one is moving/renaming a section (e.g. during distillation). These checks are the
deterministic half of the regression gate (see .claude/plans/eval-baseline.md): they don't
prove behavior, they prove every referenced section still resolves and no heading collides.
"""
import pathlib
import re

from mcp_server import guidance

SKILLS = list(guidance._SKILL_MAP)
_REPO = pathlib.Path(guidance.__file__).resolve().parent.parent


def _headings(skill):
    out = []
    for f in guidance._skill_files(guidance._SKILL_MAP[skill]):
        out += [m.group(2).strip() for m in map(guidance._HEADING_RE.match, f.read_text().splitlines()) if m]
    return out


def _resolves(skill, section):
    txt = "".join(f.read_text() for f in guidance._skill_files(guidance._SKILL_MAP[skill]))
    return guidance._extract_section(txt, section) is not None


def test_no_heading_collisions_within_a_skill():
    # The resolver returns the FIRST heading matching the text; a duplicate (case-insensitive)
    # across core + references makes get_skill(section=…) ambiguous.
    collisions = {}
    for skill in SKILLS:
        seen = {}
        for h in _headings(skill):
            seen[h.lower()] = seen.get(h.lower(), 0) + 1
        dupes = {h: n for h, n in seen.items() if n > 1}
        if dupes:
            collisions[skill] = dupes
    assert not collisions, f"heading collisions (ambiguous get_skill): {collisions}"


def test_decision_guidance_pointer_sections_resolve():
    # Every explore-action pointer must name a section that resolves.
    for action, (skill, section) in guidance._DECISION_GUIDANCE.items():
        assert _resolves(skill, section), f"_DECISION_GUIDANCE[{action!r}] -> {skill}:{section!r} unresolved"


def test_get_skill_section_literals_resolve():
    # Every get_skill('skill', section='X') literal in skills prose + server/guidance code
    # must resolve (handles single + pipe-listed sections; skips <placeholder>/{template}).
    call = re.compile(r"get_skill\(\s*['\"](\w+)['\"]\s*,\s*section=([^)]+)\)")
    lit = re.compile(r"['\"]([^'\"]+)['\"]")
    checked = 0
    files = list((_REPO / "skills").rglob("*.md")) + list((_REPO / "mcp_server").glob("*.py"))
    for f in files:
        for skill, rest in call.findall(f.read_text()):
            if skill not in guidance._SKILL_MAP:
                continue
            for sec in lit.findall(rest):
                if "<" in sec or "{" in sec:
                    continue
                assert _resolves(skill, sec), f"{f.name}: get_skill({skill!r}, section={sec!r}) unresolved"
                checked += 1
    assert checked >= 3, f"expected to find real get_skill literals, found {checked}"


def test_prose_and_code_referenced_sections_resolve():
    # Sections referenced by name in prose or code that the literal scan can't catch
    # (italic "→ *X*", server tool descriptions, the signal-keyed solve pointers, the
    # validate framing pointer, the sensitivity fallback). Renaming any of these without
    # updating the reference is a silent break — this is the curated cross-ref contract.
    referenced = [
        ("optimization_strategy", "Exact Solvers"),          # SKILL.md:13, problem_framing:201, server.py:995
        ("optimization_strategy", "Exact Solvers — Depth"),  # the deep reference
        ("solution_interpreter", "Frontier Quality and Completeness Signals"),  # solve quality pointer
        ("solution_interpreter", "Diagnostic Patterns"),     # solve diagnostics pointer
        ("solution_interpreter", "Binding Analysis"),        # sensitivity frontier-inferred fallback
        ("solution_interpreter", "Reference Point Narration"),               # B-P2: model/update reference_points
        ("solution_interpreter", "Denoting Certification — Prose & Tables"), # B-P2: exact-overlay solve
        ("solution_interpreter", "Stakeholder Writeup & the Why-Triplet"),   # B-P2: curated export
        ("problem_framing", "Formalization Checkpoint"),     # validate framing pointer
        ("problem_framing", "Preference Timing"),            # cited from optimization_strategy + solution_interpreter prose
        ("problem_framing", "Constraint schemas"),           # schema-extraction targets
        ("problem_framing", "Interaction matrix schema"),
        ("problem_framing", "Interaction matrix override schema"),
        ("problem_framing", "Scenario schema"),
    ]
    unresolved = [f"{s}:{sec!r}" for s, sec in referenced if not _resolves(s, sec)]
    assert not unresolved, f"referenced sections that don't resolve: {unresolved}"
