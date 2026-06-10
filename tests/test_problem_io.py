"""Tests for portable problem bundles (save/load in the examples format)."""

import json

import pytest

from engine import problem_io
from engine.models import (
    CardinalityConstraint,
    CuratedSolution,
    InteractionMatrix,
    ObjectiveBoundConstraint,
    Objective,
    Option,
    Problem,
    ReferencePoint,
    Run,
    Scenario,
    ScenarioConfig,
    Score,
    Solution,
)


def _rich_problem() -> Problem:
    """A problem exercising every portable section: scenarios, matrices,
    reference points, constraints, and a solved run with a curated pick."""
    return Problem(
        name="Rich Problem",
        domain="testing",
        context="A problem touching every portable field.",
        approach="proportional",
        objectives=[
            Objective(name="Return", direction="maximize", unit="%", aggregation="avg"),
            Objective(name="Risk", direction="minimize", unit="%", aggregation="quadratic"),
        ],
        options=[Option(name="A"), Option(name="B"), Option(name="C")],
        scores=[
            Score(option=o, objective=obj, value=v)
            for o, vals in {"A": (8.0, 3.0), "B": (5.0, 2.0), "C": (9.0, 5.0)}.items()
            for obj, v in zip(("Return", "Risk"), vals)
        ],
        constraints=[
            CardinalityConstraint(min=1, max=3),
            ObjectiveBoundConstraint(objective="Return", operator="min", value=4.0),
        ],
        interaction_matrices=[
            InteractionMatrix(
                objective="Risk",
                entries={
                    "A": {"A": 1.0, "B": 0.2, "C": 0.1},
                    "B": {"A": 0.2, "B": 1.0, "C": 0.3},
                    "C": {"A": 0.1, "B": 0.3, "C": 1.0},
                },
            )
        ],
        reference_points=[
            ReferencePoint(type="baseline", name="Current", objective_values={"Return": 6.0}),
        ],
        scenario_config=ScenarioConfig(
            enabled=True,
            scenarios=[
                Scenario(
                    name="downturn",
                    description="Risk floor tightens.",
                    constraint_overrides=[ObjectiveBoundConstraint(objective="Return", operator="min", value=5.0)],
                ),
            ],
        ),
    )


# ─── pure dict round-trip ───


def test_round_trip_preserves_definition():
    p = _rich_problem()
    problem, scores, _ = problem_io.to_portable(p)
    p2 = problem_io.from_portable(problem, scores)

    assert [o.model_dump() for o in p2.objectives] == [o.model_dump() for o in p.objectives]
    assert [o.model_dump() for o in p2.options] == [o.model_dump() for o in p.options]
    assert {(s.option, s.objective): s.value for s in p2.scores} == {
        (s.option, s.objective): s.value for s in p.scores
    }
    assert [c.model_dump() for c in p2.constraints] == [c.model_dump() for c in p.constraints]
    assert [m.model_dump() for m in p2.interaction_matrices] == [
        m.model_dump() for m in p.interaction_matrices
    ]
    assert [r.model_dump() for r in p2.reference_points] == [
        r.model_dump() for r in p.reference_points
    ]


def test_scenarios_survive_round_trip():
    """Regression: scenarios live under scenario_config, but the portable format
    stores them as a top-level `scenarios` list. A naive Problem(**problem_json)
    drops them; from_portable must bridge them back — and active (enabled)."""
    p = _rich_problem()
    problem, scores, _ = problem_io.to_portable(p)

    # Emitted at top level, NOT as scenario_config.
    assert "scenarios" in problem and "scenario_config" not in problem
    assert len(problem["scenarios"]) == 1

    p2 = problem_io.from_portable(problem, scores)
    assert p2.scenario_config is not None
    assert p2.scenario_config.enabled is True  # loaded scenarios are active
    assert [s.name for s in p2.scenario_config.scenarios] == ["downturn"]
    assert p2.scenario_config.scenarios[0].constraint_overrides[0].value == 5.0


def test_from_portable_accepts_explicit_scenario_config():
    """Robustness: a bundle may carry an explicit scenario_config instead of the
    examples' top-level scenarios list."""
    problem = {
        "name": "X",
        "scenario_config": {"enabled": True, "scenarios": [{"name": "s1"}]},
    }
    p = problem_io.from_portable(problem, {})
    assert [s.name for s in p.scenario_config.scenarios] == ["s1"]


def test_empty_optional_sections_omitted():
    p = Problem(
        name="Bare",
        objectives=[Objective(name="R", direction="maximize")],
        options=[Option(name="A")],
        scores=[Score(option="A", objective="R", value=1.0)],
    )
    problem, scores, solutions = problem_io.to_portable(p)
    assert "scenarios" not in problem
    assert "reference_points" not in problem
    assert "interaction_matrices" not in scores
    assert solutions is None  # unsolved → no solutions file


def test_fresh_problem_id_minted_by_default():
    p = _rich_problem()
    problem, scores, _ = problem_io.to_portable(p)
    assert problem_io.from_portable(problem, scores).problem_id != p.problem_id
    assert problem_io.from_portable(problem, scores, problem_id="keep-me").problem_id == "keep-me"


# ─── solutions ───


def test_solutions_included_when_solved():
    p = _rich_problem()
    p.run = Run(
        solutions=[Solution(solution_id=0, selected_options=["A", "B"], objective_values={"Return": 6.5, "Risk": 2.5})],
        seed_used=7,
    )
    p.curated_solutions = [
        CuratedSolution(content_signature="abc123", custom_name="balanced", objective_values={"Return": 6.5, "Risk": 2.5}),
    ]
    problem, scores, solutions = problem_io.to_portable(p)
    assert solutions is not None
    assert solutions["run"]["seed_used"] == 7
    assert len(solutions["curated_solutions"]) == 1

    p2 = problem_io.from_portable(problem, scores, solutions)
    assert p2.run is not None
    assert p2.run.solutions[0].selected_options == ["A", "B"]
    assert p2.run.seed_used == 7
    assert p2.curated_solutions[0].custom_name == "balanced"


def test_exact_run_overlay_round_trips_alongside_run():
    """A single bundle carries both the exploratory `run` and the certified `exact_run`."""
    p = _rich_problem()
    p.run = Run(solver="nsga-ii",
                solutions=[Solution(solution_id=0, selected_options=["A", "B"], objective_values={"Return": 6.5, "Risk": 2.5})])
    p.exact_run = Run(solver="highs", exact=False,
                      solutions=[Solution(solution_id=0, selected_options=["A", "C"], objective_values={"Return": 7.0, "Risk": 2.4})])
    problem, scores, solutions = problem_io.to_portable(p)
    assert "run" in solutions and "exact_run" in solutions
    assert solutions["exact_run"]["solver"] == "highs"

    p2 = problem_io.from_portable(problem, scores, solutions)
    assert p2.run.solver == "nsga-ii"
    assert p2.exact_run is not None
    assert p2.exact_run.solver == "highs"
    assert p2.exact_run.solutions[0].selected_options == ["A", "C"]


def test_exact_run_alone_is_a_meaningful_bundle():
    """An exact_run with no exploratory run still produces a solutions bundle."""
    p = _rich_problem()
    p.exact_run = Run(solver="highs",
                      solutions=[Solution(solution_id=0, selected_options=["A"], objective_values={"Return": 7.0, "Risk": 2.4})])
    _, _, solutions = problem_io.to_portable(p)
    assert solutions is not None and "exact_run" in solutions


# ─── file bundles + name resolution ───


@pytest.fixture
def io_dirs(tmp_path, monkeypatch):
    saved = tmp_path / "saved"
    examples = tmp_path / "examples"
    saved.mkdir()
    examples.mkdir()
    monkeypatch.setenv("FRONTIER_SAVED_DIR", str(saved))
    monkeypatch.setenv("FRONTIER_EXAMPLES_DIR", str(examples))
    return saved, examples


def test_save_then_load_by_name(io_dirs):
    saved, _ = io_dirs
    p = _rich_problem()
    info = problem_io.save_problem(p, "my_problem")
    assert (saved / "my_problem" / "problem.json").exists()
    assert (saved / "my_problem" / "scores.json").exists()

    loaded = problem_io.load_problem("my_problem")
    assert loaded.name == "Rich Problem"
    assert [s.name for s in loaded.scenario_config.scenarios] == ["downturn"]


def test_saved_shadows_examples_on_name_collision(io_dirs):
    saved, examples = io_dirs
    # Same name in both bases; saved/ should win.
    (examples / "dup").mkdir()
    (examples / "dup" / "problem.json").write_text(json.dumps({"name": "FromExamples"}))
    problem_io.save_problem(Problem(name="FromSaved"), "dup")

    assert problem_io.resolve_source("dup") == saved / "dup"
    assert problem_io.load_problem("dup").name == "FromSaved"


def test_list_available(io_dirs):
    _, examples = io_dirs
    (examples / "ex1").mkdir()
    (examples / "ex1" / "problem.json").write_text("{}")
    problem_io.save_problem(Problem(name="S"), "saved1")

    avail = problem_io.list_available()
    assert avail["saved"] == ["saved1"]
    assert avail["examples"] == ["ex1"]


def test_load_unknown_name_raises(io_dirs):
    with pytest.raises(FileNotFoundError):
        problem_io.load_problem("does_not_exist")


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", "with space", ""])
def test_unsafe_names_rejected(io_dirs, bad):
    with pytest.raises(ValueError):
        problem_io.save_problem(Problem(name="x"), bad)
    with pytest.raises(ValueError):
        problem_io.load_problem(bad)


def test_slugify():
    assert problem_io.slugify("My Test Portfolio") == "My_Test_Portfolio"
    assert problem_io.slugify("a/b\\c..d") == "a_b_c_d"
    assert problem_io.slugify("../../etc") == "etc"  # traversal neutralized
    assert problem_io.slugify("!!!") == "problem"  # never empty


# ─── guard the real bundled examples ───


def test_bundled_examples_load_faithfully():
    """Every bundled example loads, and any with scenarios keep them (active)."""
    avail = problem_io.list_available()
    assert avail["examples"], "expected bundled examples to be present"
    for name in avail["examples"]:
        p = problem_io.load_problem(name)
        assert p.objectives and p.options and p.scores
        if p.scenario_config and p.scenario_config.scenarios:
            assert p.scenario_config.enabled is True


def test_bundled_examples_baked_runs_snapshot_constraints():
    """Baked run/exact_run must snapshot the problem's constraints — an empty snapshot
    makes the first compare_runs against a fresh solve report a phantom criteria diff
    (post-streamlining user test, finding P4). Scenario runs carry no snapshot by
    engine behavior, so they are exempt."""
    avail = problem_io.list_available()
    for name in avail["examples"]:
        p = problem_io.load_problem(name)
        for label, run in (("run", p.run), ("exact_run", p.exact_run)):
            if run is None:
                continue
            assert len(run.constraints_snapshot) == len(p.constraints), (
                f"{name}/{label}: constraints_snapshot has {len(run.constraints_snapshot)} "
                f"entries but the problem has {len(p.constraints)} constraints"
            )
