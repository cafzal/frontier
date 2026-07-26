"""Post-solve constraint verification (`optimizer.verify_run`) + constraint provenance.

The engine enforces hard constraints during the search; this re-reads the plans the search
returned. The gate that matters is the drift guard: `_constraint_row_labels` mirrors the G-row
order of the two `_evaluate` blocks by hand, so a G row added without a label must fail loudly
here rather than silently misattribute a violation to the wrong constraint.
"""

import pytest

from engine.models import (
    AllocationBoundConstraint,
    Approach,
    CardinalityConstraint,
    DependencyConstraint,
    Direction,
    ExclusionPairConstraint,
    ForceExcludeConstraint,
    ForceIncludeConstraint,
    GroupLimitConstraint,
    MaxAllocationConstraint,
    Objective,
    ObjectiveBoundConstraint,
    Option,
    Problem,
    Run,
    Score,
    Solution,
)
from engine.optimizer import (
    _constraint_row_labels,
    _parse_constraints,
    make_slate_scorer,
    verify_run,
)

OPTS = ["A", "B", "C", "D"]


def _problem(approach=Approach.binary, constraints=None):
    return Problem(
        name="t",
        approach=approach,
        objectives=[Objective(name="Value", direction=Direction.maximize),
                    Objective(name="Cost", direction=Direction.minimize)],
        options=[Option(name=n) for n in OPTS],
        scores=[Score(option=n, objective=o, value=float(i + 1))
                for i, n in enumerate(OPTS) for o in ("Value", "Cost")],
        constraints=constraints or [],
    )


def _run(*solutions):
    return Run(solutions=list(solutions))


def _sol(sid, selected, allocations=None):
    return Solution(solution_id=sid, selected_options=selected,
                    objective_values={"Value": 0.0, "Cost": 0.0}, allocations=allocations)


# --- The drift gate -------------------------------------------------------------------

def _every_constraint_type(approach):
    """One constraint of every type the approach supports — the widest G the encoder builds."""
    common = [
        CardinalityConstraint(min=1, max=3),
        ForceIncludeConstraint(option="A"),
        ForceExcludeConstraint(option="D"),
        ObjectiveBoundConstraint(objective="Cost", operator="max", value=99.0),
        ExclusionPairConstraint(option_a="B", option_b="C"),
        DependencyConstraint(if_option="B", then_option="A"),
        GroupLimitConstraint(options=["A", "B"], min=1, max=2),
    ]
    if approach == Approach.proportional:
        common += [MaxAllocationConstraint(max=60),
                   AllocationBoundConstraint(option="A", min=10, max=50)]
    return common


@pytest.mark.parametrize("approach", [Approach.binary, Approach.proportional])
def test_labels_cover_every_g_row(approach):
    """Label count == pymoo's declared row count, with every constraint type present.

    This is the anti-drift gate: edit an `_evaluate` G block without updating
    `_constraint_row_labels` and this fails.
    """
    p = _problem(approach, _every_constraint_type(approach))
    cp = _parse_constraints(p)
    labels = _constraint_row_labels(p, cp)
    scorer = make_slate_scorer(p)
    # constraint_labels is None precisely when the mirror disagreed with n_ieq_constr.
    assert scorer.constraint_labels is not None, (
        f"row-label mirror out of sync for {approach.value}: built {len(labels)} labels")
    assert len(scorer.constraint_labels) == len(labels)


@pytest.mark.parametrize("approach", [Approach.binary, Approach.proportional])
def test_labels_degrade_rather_than_misattribute(approach, monkeypatch):
    """A short label list disables labelling instead of shifting rows onto wrong constraints."""
    import engine.optimizer as opt
    real = opt._constraint_row_labels
    monkeypatch.setattr(opt, "_constraint_row_labels", lambda p, cp: real(p, cp)[:-1])
    p = _problem(approach, _every_constraint_type(approach))
    assert make_slate_scorer(p).constraint_labels is None


# --- Detection per constraint type ----------------------------------------------------

@pytest.mark.parametrize("constraint,bad,label_fragment", [
    (CardinalityConstraint(min=1, max=2), ["A", "B", "C"], "cardinality"),
    (ForceIncludeConstraint(option="A"), ["B"], "force_include"),
    (ForceExcludeConstraint(option="D"), ["A", "D"], "force_exclude"),
    (ExclusionPairConstraint(option_a="B", option_b="C"), ["B", "C"], "exclusion_pair"),
    (DependencyConstraint(if_option="B", then_option="A"), ["B"], "dependency"),
    (GroupLimitConstraint(options=["A", "B"], max=1), ["A", "B"], "group"),
    (GroupLimitConstraint(options=["A", "B"], min=1, max=2), ["C"], "group"),
    (ObjectiveBoundConstraint(objective="Cost", operator="max", value=1.5), ["C", "D"],
     "Cost"),
])
def test_binary_violation_is_named(constraint, bad, label_fragment):
    p = _problem(Approach.binary, [constraint])
    r = verify_run(p, _run(_sol(7, bad)))
    assert r["status"] == "violations_found", r
    assert r["plans_checked"] == 1
    hit = [v for v in r["violations"] if label_fragment in v["constraint"]]
    assert hit, r["violations"]
    assert hit[0]["solution_id"] == 7
    assert hit[0]["severity"] == "violation"
    assert hit[0]["margin"] > 0


def test_rows_map_to_the_right_constraint_instance():
    """Order guard, not just count. `forced_in`/`forced_out` are sets and the allocation-bound
    rows are vectorized, so several same-type rows sit adjacent in G — a mirror that drifted by
    one would still have the right length and name the wrong option."""
    p = _problem(Approach.binary, [ForceIncludeConstraint(option=n) for n in ("A", "B", "C")])
    # Only 'B' is missing, so exactly one force_include row may fire, and it must name B.
    r = verify_run(p, _run(_sol(0, ["A", "C"])))
    named = [v["constraint"] for v in r["violations"] if v["constraint_type"] == "force_include"]
    assert named == ["force_include 'B'"], r["violations"]


def test_allocation_bound_rows_name_the_right_option_and_side():
    p = _problem(Approach.proportional, [
        AllocationBoundConstraint(option="A", min=10, max=50),
        AllocationBoundConstraint(option="B", min=30, max=90),
    ])
    # A sits above its cap; B sits below its floor — one row each, distinct options and sides.
    r = verify_run(p, _run(_sol(0, ["A", "B"], {"A": 80, "B": 20})))
    named = sorted(v["constraint"] for v in r["violations"]
                   if v["constraint_type"] == "allocation_bound")
    assert named == ["'A' allocation ≤ 50%", "'B' allocation ≥ 30%"], r["violations"]


def test_clean_binary_run_verifies():
    p = _problem(Approach.binary, [CardinalityConstraint(min=1, max=3),
                                   ForceIncludeConstraint(option="A")])
    r = verify_run(p, _run(_sol(0, ["A", "B"]), _sol(1, ["A", "C"])))
    assert r == {"status": "verified", "plans_checked": 2, "violations": []}


# --- Severity: the proportional rounding split ----------------------------------------

def test_proportional_objective_bound_is_rounding_not_violation():
    """Objective values are recorded from the continuous x while allocations are recorded
    rounded, so a re-derived objective can sit just outside a bound the solve honored."""
    p = _problem(Approach.proportional,
                 [ObjectiveBoundConstraint(objective="Cost", operator="max", value=1.5)])
    r = verify_run(p, _run(_sol(3, ["C", "D"], {"C": 50, "D": 50})))
    assert r["status"] == "verified", "a rounding row must not fail the check"
    assert [v["severity"] for v in r["violations"]] == ["rounding"]
    assert "rounding" in r["note"]


def test_proportional_non_bound_breach_is_a_real_violation():
    """Everything except objective_bound is judged on the allocations themselves — both the
    writer and the checker read the same field, so a breach there is a property of the plan."""
    p = _problem(Approach.proportional, [MaxAllocationConstraint(max=40)])
    r = verify_run(p, _run(_sol(2, ["A", "B"], {"A": 90, "B": 10})))
    assert r["status"] == "violations_found"
    assert all(v["severity"] == "violation" for v in r["violations"])


def test_violation_list_is_capped_with_a_total():
    from engine.optimizer import _VERIFY_MAX_ROWS
    p = _problem(Approach.binary, [ForceIncludeConstraint(option="A")])
    run = _run(*[_sol(i, ["B"]) for i in range(_VERIFY_MAX_ROWS + 5)])
    r = verify_run(p, run)
    assert len(r["violations"]) == _VERIFY_MAX_ROWS
    assert r["violations_total"] == _VERIFY_MAX_ROWS + 5


def test_scenario_is_verified_against_its_own_overrides():
    """A scenario replaces the whole base constraint set; checking against the base rules
    would judge the plan by a model the scenario never used."""
    from engine.models import Scenario
    p = _problem(Approach.binary, [CardinalityConstraint(min=1, max=1)])
    sc = Scenario(name="loose", constraint_overrides=[CardinalityConstraint(min=1, max=3)])
    run = _run(_sol(0, ["A", "B", "C"]))
    assert verify_run(p, run)["status"] == "violations_found"
    assert verify_run(p, run, scenario=sc)["status"] == "verified"


# --- Every bundled example must be clean ----------------------------------------------

def test_bundled_examples_verify():
    """The calibration regression: this check must be silent on every shipped example, or it
    is miscalibrated. Rounding rows are permitted (and expected on proportional shapes);
    a `violation` on a bundled bundle is a defect in the engine or in this checker."""
    import pathlib

    from engine import problem_io

    root = pathlib.Path(__file__).resolve().parent.parent / "examples"
    names = sorted(d.name for d in root.iterdir()
                   if d.is_dir() and (d / "problem.json").exists())
    assert names, "no bundled examples found"
    for name in names:
        p = problem_io.load_problem(name)
        if not p.run:
            continue
        r = verify_run(p, p.run)
        breaches = [v for v in r["violations"] if v["severity"] == "violation"]
        assert not breaches, f"{name}: {breaches}"
        assert r["status"] == "verified", name


# --- Provenance -----------------------------------------------------------------------

def test_motivated_by_defaults_empty_and_round_trips():
    c = CardinalityConstraint(min=1, max=3)
    assert c.motivated_by == ""
    assert "motivated_by" in c.model_dump()
    withwhy = CardinalityConstraint(min=1, max=3, motivated_by="board caps active bets at 3")
    assert CardinalityConstraint(**withwhy.model_dump()).motivated_by == withwhy.motivated_by


def test_legacy_constraint_payload_still_parses():
    """A bundle written before the field exists must load unchanged."""
    p = Problem(**{
        "name": "legacy",
        "objectives": [{"name": "Value", "direction": "maximize"}],
        "options": [{"name": "A"}],
        "constraints": [{"type": "cardinality", "min": 1, "max": 1}],
    })
    assert p.constraints[0].motivated_by == ""


def test_binding_analysis_echoes_motivated_by():
    from engine.explorer import _binding_analysis
    why = "regulator caps single-name exposure"
    p = _problem(Approach.binary, [
        ObjectiveBoundConstraint(objective="Cost", operator="max", value=6.0,
                                 motivated_by=why)])
    sols = [Solution(solution_id=i, selected_options=OPTS[: (i % 3) + 1],
                     objective_values={"Value": float(i), "Cost": 6.0 - i * 0.05})
            for i in range(8)]
    entries = _binding_analysis(p, sols)
    assert entries, "expected the bound to read as binding"
    assert all(e.get("motivated_by") == why for e in entries)


def test_binding_analysis_omits_motive_when_absent():
    from engine.explorer import _binding_analysis
    p = _problem(Approach.binary,
                 [ObjectiveBoundConstraint(objective="Cost", operator="max", value=6.0)])
    sols = [Solution(solution_id=i, selected_options=OPTS[: (i % 3) + 1],
                     objective_values={"Value": float(i), "Cost": 6.0 - i * 0.05})
            for i in range(8)]
    assert all("motivated_by" not in e for e in _binding_analysis(p, sols))
