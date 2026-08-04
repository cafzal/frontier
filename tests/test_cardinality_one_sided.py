"""One-sided cardinality constraints, end to end.

The natural user statement is often one-sided — "fund at least 20" or "hold at most 24" —
so the schema accepts `min` alone, `max` alone, or both; an absent bound leaves that side
unbounded. These tests pin the whole path: schema, the merged resolver, validation, the
NSGA search encoding, the exact MILP builder, the audit negation, and the server's
constraint formatting — so no layer quietly assumes both bounds are present.
"""

import pytest

from engine.models import (
    CardinalityConstraint,
    ForceIncludeConstraint,
    Objective,
    Option,
    Problem,
    Score,
)
from engine.optimizer import (
    _negate_property,
    _parse_constraints,
    analyze_infeasibility,
    merged_cardinality,
    optimize,
    validate,
)


def _make_problem(**overrides):
    """Five-option binary selection, no default cardinality row."""
    names = ["A", "B", "C", "D", "E"]
    revenue = {"A": 8, "B": 6, "C": 9, "D": 4, "E": 7}
    effort = {"A": 5, "B": 3, "C": 7, "D": 2, "E": 4}
    defaults = dict(
        objectives=[
            Objective(name="Revenue", direction="maximize"),
            Objective(name="Effort", direction="minimize"),
        ],
        options=[Option(name=n) for n in names],
        scores=[Score(option=n, objective=o, value=v)
                for o, table in (("Revenue", revenue), ("Effort", effort))
                for n, v in table.items()],
        constraints=[],
    )
    defaults.update(overrides)
    return Problem(**defaults)


# ─── Schema ───


class TestSchema:
    def test_min_only_is_legal(self):
        c = CardinalityConstraint(min=2)
        assert c.min == 2
        assert c.max is None

    def test_max_only_is_legal(self):
        c = CardinalityConstraint(max=3)
        assert c.min is None
        assert c.max == 3

    def test_two_sided_still_works(self):
        c = CardinalityConstraint(min=2, max=3)
        assert (c.min, c.max) == (2, 3)

    def test_neither_bound_is_rejected(self):
        # A row with no bound states no rule — refuse it loudly at the schema.
        with pytest.raises(ValueError):
            CardinalityConstraint()

    def test_roundtrips_through_dump_and_validate(self):
        for c in (CardinalityConstraint(min=2), CardinalityConstraint(max=3)):
            again = CardinalityConstraint.model_validate(c.model_dump())
            assert (again.min, again.max) == (c.min, c.max)


# ─── Resolver + validation ───


class TestResolution:
    def test_merged_min_only(self):
        assert merged_cardinality([CardinalityConstraint(min=2)]) == (2, None)

    def test_merged_max_only(self):
        assert merged_cardinality([CardinalityConstraint(max=3)]) == (None, 3)

    def test_one_sided_rows_intersect(self):
        # A min-only row beside a max-only row combine into the stated box.
        rows = [CardinalityConstraint(min=2), CardinalityConstraint(max=4)]
        assert merged_cardinality(rows) == (2, 4)

    def test_parse_resolves_absent_sides_to_the_defaults(self):
        cp = _parse_constraints(_make_problem(constraints=[CardinalityConstraint(min=2)]))
        assert (cp["cardinality_min"], cp["cardinality_max"]) == (2, 5)
        cp = _parse_constraints(_make_problem(constraints=[CardinalityConstraint(max=3)]))
        assert (cp["cardinality_min"], cp["cardinality_max"]) == (0, 3)

    def test_max_only_keeps_the_search_floor(self):
        # A max-only row says nothing about the minimum, so the EA's never-propose-empty
        # search default stays in force — exactly as it does with no cardinality row.
        cp = _parse_constraints(_make_problem(constraints=[CardinalityConstraint(max=3)]),
                                search_floor=True)
        assert (cp["cardinality_min"], cp["cardinality_max"]) == (1, 3)

    def test_validate_accepts_one_sided(self):
        for c in (CardinalityConstraint(min=2), CardinalityConstraint(max=3)):
            vr = validate(_make_problem(constraints=[c]))
            assert vr.ready is True, [i.message for i in vr.issues]

    def test_validate_still_catches_min_over_available(self):
        vr = validate(_make_problem(constraints=[CardinalityConstraint(min=9)]))
        assert any("exceeds available options" in i.message and i.severity == "error"
                   for i in vr.issues)

    def test_validate_still_catches_forced_over_max_only(self):
        vr = validate(_make_problem(constraints=[
            CardinalityConstraint(max=2),
            *(ForceIncludeConstraint(option=n) for n in "ABC")]))
        assert any("exceeds cardinality max (2)" in i.message and i.severity == "error"
                   for i in vr.issues)


# ─── Solve (NSGA) ───


class TestSolve:
    def test_solve_respects_max_only(self):
        p = _make_problem(constraints=[CardinalityConstraint(max=2)])
        run = optimize(p, mode="fast", seed=42)
        assert run.solutions
        assert all(len(s.selected_options) <= 2 for s in run.solutions)

    def test_solve_respects_min_only(self):
        p = _make_problem(constraints=[CardinalityConstraint(min=4)])
        run = optimize(p, mode="fast", seed=42)
        assert run.solutions
        assert all(len(s.selected_options) >= 4 for s in run.solutions)

    def test_infeasibility_diagnosis_handles_one_sided(self):
        p = _make_problem(constraints=[
            CardinalityConstraint(max=1),
            *(ForceIncludeConstraint(option=n) for n in "AB")])
        d = analyze_infeasibility(p)
        assert d["binding_constraints"]


# ─── Exact path (MILP builder + HiGHS overlay) ───


class TestExactPath:
    def test_milp_data_carries_one_sided_range(self):
        from solvers._scalarization import _build_milp_data

        assert _build_milp_data(
            _make_problem(constraints=[CardinalityConstraint(min=2)]))[-1]["card"] == (2, None)
        assert _build_milp_data(
            _make_problem(constraints=[CardinalityConstraint(max=3)]))[-1]["card"] == (None, 3)

    def test_highs_respects_max_only(self):
        pytest.importorskip("highspy")
        p = _make_problem(constraints=[CardinalityConstraint(max=2)])
        run = optimize(p, mode="fast", seed=7, solver="highs")
        assert run.solutions
        assert all(len(s.selected_options) <= 2 for s in run.solutions)

    def test_highs_respects_min_only(self):
        pytest.importorskip("highspy")
        p = _make_problem(constraints=[CardinalityConstraint(min=4)])
        run = optimize(p, mode="fast", seed=7, solver="highs")
        assert run.solutions
        assert all(len(s.selected_options) >= 4 for s in run.solutions)


# ─── Audit negation ───


class TestAuditNegation:
    def test_min_only_property_negates_to_one_disjunct(self):
        p = _make_problem()
        disjuncts = _negate_property(p, CardinalityConstraint(min=2))
        assert len(disjuncts) == 1
        [(coef, op, rhs)] = disjuncts[0]
        assert (op, rhs) == ("le", 1)

    def test_max_only_property_negates_to_one_disjunct(self):
        p = _make_problem()
        disjuncts = _negate_property(p, CardinalityConstraint(max=3))
        assert len(disjuncts) == 1
        [(coef, op, rhs)] = disjuncts[0]
        assert (op, rhs) == ("ge", 4)

    def test_vacuous_property_is_declined(self):
        # min=0 with no max binds nothing; a "holds" on it would certify a tautology.
        p = _make_problem()
        with pytest.raises(ValueError, match="vacuous"):
            _negate_property(p, CardinalityConstraint(min=0))


# ─── Server formatting + binding analytics ───


class TestServerSurface:
    def test_format_constraint_one_sided(self):
        from mcp_server.server import _format_constraint

        assert _format_constraint(CardinalityConstraint(min=20)) == "select ≥20"
        assert _format_constraint(CardinalityConstraint(max=24)) == "select ≤24"
        assert _format_constraint(CardinalityConstraint(min=2, max=3)) == "select 2–3"

    def test_merge_note_survives_one_sided_rows(self):
        from mcp_server.server import _attach_constraint_merge_note

        p = _make_problem(constraints=[CardinalityConstraint(min=2),
                                       CardinalityConstraint(min=3)])
        result: dict = {}
        _attach_constraint_merge_note(result, p)
        assert "select ≥3" in result["constraints_merged_note"]

    def test_binding_checks_handle_one_sided(self):
        from engine.explorer import _binding_cardinality
        from engine.metrics import _check_binding_cardinality

        p = _make_problem(constraints=[CardinalityConstraint(min=4)])
        run = optimize(p, mode="fast", seed=42)
        results: list = []
        _check_binding_cardinality(p.constraints[0], run.solutions, results)  # no TypeError
        _binding_cardinality(p.constraints[0], run.solutions, p.objectives)  # no TypeError
