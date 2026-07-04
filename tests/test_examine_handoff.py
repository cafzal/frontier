"""Tests for the EXAMINE handoff contracts: sensitivity → suggested scenarios ("duals rank,
scenarios quantify") and scenario-sweep discipline (varies / held-fixed restatement,
motivated_by provenance echo). Response-contract work over existing tools — no new solves.
"""
from engine.explorer import get_scenario_results, sensitivity_analysis
from engine.models import (
    CardinalityConstraint,
    Objective,
    ObjectiveBoundConstraint,
    Option,
    Problem,
    Run,
    Scenario,
    ScenarioConfig,
    ScenarioRun,
    Score,
    ScoreAdjustment,
    ShadowPrice,
    Solution,
    SolutionSensitivity,
)


def _problem(approach="binary", constraints=None, scenario_config=None):
    names = ["A", "B", "C", "D"]
    scores = []
    for k, o in enumerate(names):
        scores.append(Score(option=o, objective="Value", value=5 + k))
        scores.append(Score(option=o, objective="Cost", value=1 + k))
    return Problem(
        name="handoff-t", approach=approach,
        objectives=[Objective(name="Value", direction="maximize"),
                    Objective(name="Cost", direction="minimize")],
        options=[Option(name=o) for o in names],
        scores=scores, constraints=constraints or [],
        scenario_config=scenario_config,
    )


def _run(points, sensitivities=None):
    sols = []
    for i, (v, c) in enumerate(points):
        sols.append(Solution(
            solution_id=i, selected_options=["A"],
            objective_values={"Value": float(v), "Cost": float(c)},
            sensitivity=(sensitivities[i] if sensitivities else None),
        ))
    return Run(solutions=sols)


# ─── C1: sensitivity ends with suggested scenarios seeded from its top levers ───

def test_binding_constraint_yields_matching_scenario_suggestion():
    # Golden path: a bound binding on most of the frontier → the suggestion references it.
    p = _problem(constraints=[ObjectiveBoundConstraint(objective="Cost", operator="max", value=10)])
    p.run = _run([(5, 9.6), (7, 9.8), (9, 10.0), (3, 5.0)])
    out = sensitivity_analysis(p)
    assert out["source"] == "frontier_inferred"
    s = out["suggested_scenarios"][0]
    assert s["motivated_by"].startswith("binding_constraint:Cost")
    assert "Cost" in s["vary"]
    assert "re-solve" in s["why"]                        # quantify, not just the local rate
    assert "scenario_results" in s["how"]


def test_exact_duals_yield_score_stress_suggestion():
    sens = [SolutionSensitivity(shadow_prices=[
        ShadowPrice(name="Value", role="linear_floor", shadow_price=2.0 + i)]) for i in range(3)]
    p = _problem(approach="proportional")
    p.run = _run([(1, 1), (2, 3), (3, 6)], sensitivities=sens)
    out = sensitivity_analysis(p)
    assert out["source"] == "solver_exact"
    s = out["suggested_scenarios"][0]
    assert s["motivated_by"] == "shadow_price:Value"
    assert s["rate"] is not None
    assert "hold every other anchor fixed" in s["vary"]  # sweep discipline in the seed itself
    assert "motivated_by" in s["how"]                    # provenance instruction for the agent


def test_no_binding_levers_no_suggestions():
    # Nothing binding → no suggested_scenarios key at all (no invented levers).
    p = _problem()
    p.run = _run([(5, 2), (7, 4), (9, 6)])
    out = sensitivity_analysis(p)
    assert "suggested_scenarios" not in out


# ─── C2: scenario results restate varies / held-fixed and cite their motive ───

def test_scenario_results_restate_varies_held_fixed_and_motive():
    cfg = ScenarioConfig(enabled=True, scenarios=[
        Scenario(name="value-stress", motivated_by="shadow_price:Value",
                 score_adjustments=[ScoreAdjustment(objective="Value", multiply=0.9)]),
        Scenario(name="relaxed", constraint_overrides=[CardinalityConstraint(min=1, max=4)]),
    ])
    p = _problem(scenario_config=cfg)
    p.scenario_run = ScenarioRun(scenario_runs={
        "value-stress": _run([(4, 2), (6, 4)]),
        "relaxed": _run([(5, 2), (8, 5)]),
    })
    per = get_scenario_results(p)["per_scenario"]

    stress = per["value-stress"]
    assert stress["motivated_by"] == "shadow_price:Value"     # cites the seeding marginal
    assert any("'Value' ×0.9" in v for v in stress["varies"])
    assert "inherits the base model" in stress["held_fixed"]

    relaxed = per["relaxed"]
    assert "motivated_by" not in relaxed                      # unseeded scenario stays unclaimed
    # Replace-all override semantics are stated, not assumed.
    assert any("REPLACED" in v for v in relaxed["varies"])
