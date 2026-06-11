"""Tests for Frontier engine models — round-trip serialization."""

import math

import pytest
from pydantic import ValidationError

from engine.models import (
    CardinalityConstraint,
    ForceExcludeConstraint,
    ForceIncludeConstraint,
    Objective,
    ObjectiveBoundConstraint,
    Option,
    Problem,
    Run,
    Scenario,
    Score,
    ScoreAdjustment,
    Solution,
)


def test_problem_defaults():
    p = Problem()
    assert p.problem_id
    assert p.name == ""
    assert p.objectives == []
    assert p.run is None


def test_problem_round_trip():
    p = Problem(
        name="Test",
        objectives=[Objective(name="Rev", direction="maximize", unit="$")],
        options=[Option(name="A"), Option(name="B"), Option(name="C")],
        scores=[Score(option="A", objective="Rev", value=5.0)],
        constraints=[CardinalityConstraint(min=1, max=2)],
    )
    data = p.model_dump_json()
    p2 = Problem.model_validate_json(data)
    assert p2.name == "Test"
    assert len(p2.objectives) == 1
    assert len(p2.options) == 3
    assert len(p2.scores) == 1
    assert p2.constraints[0].type == "cardinality"
    assert p2.constraints[0].min == 1


def test_constraint_types_serialize():
    constraints = [
        CardinalityConstraint(min=2, max=5),
        ForceIncludeConstraint(option="SSO"),
        ForceExcludeConstraint(option="Mobile"),
        ObjectiveBoundConstraint(objective="Effort", operator="max", value=40),
    ]
    p = Problem(constraints=constraints)
    data = p.model_dump_json()
    p2 = Problem.model_validate_json(data)
    assert len(p2.constraints) == 4
    assert p2.constraints[0].type == "cardinality"
    assert p2.constraints[1].type == "force_include"
    assert p2.constraints[2].type == "force_exclude"
    assert p2.constraints[3].type == "objective_bound"


def test_run_with_solutions():
    run = Run(solutions=[
        Solution(solution_id=0, selected_options=["A", "B"], objective_values={"Rev": 10.0}),
        Solution(solution_id=1, selected_options=["A", "C"], objective_values={"Rev": 8.0}),
    ])
    data = run.model_dump_json()
    run2 = Run.model_validate_json(data)
    assert len(run2.solutions) == 2
    assert run2.solutions[0].selected_options == ["A", "B"]


# ─── Non-finite (inf/nan) rejection on user-input numeric fields ───
#
# A NaN score used to pass validation, serialize to JSON null on save, then raise an
# uncaught ValidationError on every later load — permanently bricking the record. These
# guard that non-finite values are rejected at the model boundary instead.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_score_rejects_non_finite(bad):
    with pytest.raises(ValidationError):
        Score(option="A", objective="X", value=bad)


def test_score_accepts_large_finite():
    # The cap is on finiteness, not magnitude — a huge but finite score is fine.
    assert Score(option="A", objective="X", value=1e18).value == 1e18


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_objective_bound_rejects_non_finite(bad):
    with pytest.raises(ValidationError):
        ObjectiveBoundConstraint(objective="X", operator="max", value=bad)


def test_score_adjustment_rejects_non_finite():
    with pytest.raises(ValidationError):
        ScoreAdjustment(objective="X", multiply=float("inf"))
    with pytest.raises(ValidationError):
        ScoreAdjustment(objective="X", add=float("nan"))


def test_scenario_probability_rejects_non_finite():
    with pytest.raises(ValidationError):
        Scenario(name="s", probability=float("nan"))


def test_non_finite_never_round_trips_through_problem_json():
    # The brick path: build → save (JSON) → load. With finiteness enforced, the bad
    # value can't be constructed in the first place, so a Problem holding it can't exist.
    with pytest.raises(ValidationError):
        Problem(
            objectives=[Objective(name="X", direction="maximize")],
            options=[Option(name="A")],
            scores=[Score(option="A", objective="X", value=math.nan)],
        )
