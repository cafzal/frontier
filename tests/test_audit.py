"""Tests for the witness / feasibility audit (engine.optimizer.audit, explorer.audit_property,
and the `explore audit` MCP action).

Skipped wholesale without ``highspy`` — the audit is a HiGHS feasibility solve, so like the rest of
the exact-backend suite these run anywhere CI does (CPU, cross-platform) but not on a bare install.

The audit reuses the exact MILP constraint encoding (``_add_milp_constraints``), so a feasible region
here is provably the region ``solve`` optimizes over. The verdict turns on the raw solver status —
Infeasible (the property holds across the *whole* feasible space) must never collapse into a time
limit (inconclusive); these tests pin that distinction and the negation encodings.
"""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("highspy")

import solvers.highs_backend as hb
from engine import explorer
from engine.models import (
    CardinalityConstraint,
    DependencyConstraint,
    ExclusionPairConstraint,
    ForceExcludeConstraint,
    ForceIncludeConstraint,
    GroupLimitConstraint,
    ObjectiveBoundConstraint,
    Objective,
    Option,
    Problem,
    Score,
)
from engine.optimizer import audit, diagnose_conflicts

_HAS_MCP_CLIENT = importlib.util.find_spec("mcp.client.stdio") is not None
REPO_ROOT = Path(__file__).resolve().parent.parent


def _problem(approach="binary", constraints=None, aggregation="sum"):
    """4-option binary selection, Value (max) vs Cost (min)."""
    val = {"A": 9, "B": 7, "C": 4, "D": 2}
    cost = {"A": 8, "B": 5, "C": 3, "D": 1}
    scores = []
    for o in ["A", "B", "C", "D"]:
        scores.append(Score(option=o, objective="Value", value=val[o]))
        scores.append(Score(option=o, objective="Cost", value=cost[o]))
    return Problem(
        name="audit-t", approach=approach,
        objectives=[Objective(name="Value", direction="maximize", aggregation=aggregation),
                    Objective(name="Cost", direction="minimize", aggregation="sum")],
        options=[Option(name=o) for o in ["A", "B", "C", "D"]],
        scores=scores, constraints=constraints or [],
    )


# ─── Feasibility probe (no property) ───

def test_probe_satisfiable_returns_feasible_witness():
    r = audit(_problem(constraints=[CardinalityConstraint(min=1, max=2)]))
    assert r["verdict"] == "feasible"
    assert r["audit_kind"] == "feasibility_probe"
    assert r["witness"] is not None and r["witness"]["feasible"] is True
    # Region echo pins what the verdict is conditional on; raw solver fields are not surfaced.
    assert r["feasible_region"]["n_options"] == 4
    assert any(c["type"] == "cardinality" for c in r["feasible_region"]["constraints"])
    assert "statuses" not in r and "mode" not in r


def test_probe_overconstrained_returns_no_feasible_plan():
    # force_include(A) AND force_exclude(A) → empty feasible region.
    r = audit(_problem(constraints=[ForceIncludeConstraint(option="A"),
                                    ForceExcludeConstraint(option="A")]))
    assert r["verdict"] == "no_feasible_plan"
    assert r["witness"] is None


# ─── Property audit: holds ───

def test_property_holds_across_feasible_space():
    # Region pins exactly 2 selected; "1 ≤ count ≤ 4" therefore holds for every feasible plan.
    r = audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
              CardinalityConstraint(min=1, max=4))
    assert r["verdict"] == "holds"
    assert r["witness"] is None
    assert r["audit_kind"] == "property_audit"


def test_objective_bound_holds_when_cap_never_reached():
    # Region pins exactly 1 selected; the dearest single option costs 8, so "Cost ≤ 20" holds.
    r = audit(_problem(constraints=[CardinalityConstraint(min=1, max=1)]),
              ObjectiveBoundConstraint(objective="Cost", operator="max", value=20))
    assert r["verdict"] == "holds"


# ─── Property audit: violated, with a real counterexample ───

def test_force_include_property_violated_with_witness():
    # A is not forced, so a feasible plan without A exists → the property is not guaranteed.
    r = audit(_problem(constraints=[CardinalityConstraint(min=1, max=2)]),
              ForceIncludeConstraint(option="A"))
    assert r["verdict"] == "violated"
    w = r["witness"]
    assert w is not None and "A" not in w["selected_options"]   # the witness genuinely lacks A


def test_force_exclude_property_violated_with_witness():
    # A is selectable, so a feasible plan with A exists → "A never appears" isn't guaranteed.
    r = audit(_problem(constraints=[CardinalityConstraint(min=1, max=2)]),
              ForceExcludeConstraint(option="A"))
    assert r["verdict"] == "violated"
    w = r["witness"]
    assert w is not None and "A" in w["selected_options"]       # the witness genuinely includes A


def test_objective_bound_violated_witness_breaches_cap():
    # Region pins exactly 1; option A costs 8 > 5, so "Cost ≤ 5" is violated and A is a witness.
    r = audit(_problem(constraints=[CardinalityConstraint(min=1, max=1)]),
              ObjectiveBoundConstraint(objective="Cost", operator="max", value=5))
    assert r["verdict"] == "violated"
    assert r["witness"]["objective_values"]["Cost"] > 5          # the witness really breaches it


def test_group_limit_property_violated():
    # Region pins exactly 2; the plan {A,B} takes 2 from the group → "≤1 from {A,B}" is violated.
    r = audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
              GroupLimitConstraint(options=["A", "B"], max=1))
    assert r["verdict"] == "violated"
    assert set(r["witness"]["selected_options"]) >= {"A", "B"}


def test_group_floor_property_negates_as_disjunction():
    # E1: a floored group property (min ≤ count ≤ max) negates below-min OR above-max.
    # Model floors {A,B} at 1 → "≥1 from {A,B}" holds across the whole space...
    region = [CardinalityConstraint(min=1, max=2),
              GroupLimitConstraint(options=["A", "B"], min=1, max=2)]
    r = audit(_problem(constraints=region), GroupLimitConstraint(options=["A", "B"], min=1, max=4))
    assert r["verdict"] == "holds"
    # ...but "≥2 from {A,B}" does not — a witness takes just one of them.
    r2 = audit(_problem(constraints=region), GroupLimitConstraint(options=["A", "B"], min=2, max=4))
    assert r2["verdict"] == "violated"
    assert len({"A", "B"} & set(r2["witness"]["selected_options"])) == 1


def test_dependency_property_violated():
    # force A, exclude B → a feasible plan has A without B, breaking "A ⇒ B".
    r = audit(_problem(constraints=[ForceIncludeConstraint(option="A"),
                                    ForceExcludeConstraint(option="B")]),
              DependencyConstraint(if_option="A", then_option="B"))
    assert r["verdict"] == "violated"
    assert "A" in r["witness"]["selected_options"] and "B" not in r["witness"]["selected_options"]


def test_exclusion_pair_property_violated():
    r = audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
              ExclusionPairConstraint(option_a="A", option_b="B"))
    assert r["verdict"] == "violated"
    assert set(r["witness"]["selected_options"]) >= {"A", "B"}


# ─── Edge cases & gating ───

def test_property_holds_vacuously_on_empty_region():
    r = audit(_problem(constraints=[ForceIncludeConstraint(option="A"),
                                    ForceExcludeConstraint(option="A")]),
              ForceIncludeConstraint(option="B"))
    assert r["verdict"] == "holds_vacuously"
    assert "note" in r


def test_proportional_shape_declined_raises():
    # An unfit shape is a hard decline (a tool error), not a structured "unsupported" verdict —
    # the same convention as solve's exact gate and certify's precondition.
    with pytest.raises(ValueError, match="binary"):
        audit(_problem(approach="proportional"))


def test_nonsum_aggregation_gates_on_bound_reference_only():
    # Audit is a feasibility solve: objectives shape the region only via objective_bound rows.
    # A non-sum objective NO bound touches is fine (e.g. a quadratic interaction objective on a
    # binary problem still audits)...
    r = audit(_problem(aggregation="min",
                       constraints=[CardinalityConstraint(min=1, max=2)]))
    assert r["verdict"] == "feasible"
    # ...but a bound ON the non-sum objective can't be encoded exactly → hard decline.
    with pytest.raises(ValueError, match="non-sum"):
        audit(_problem(aggregation="min",
                       constraints=[ObjectiveBoundConstraint(objective="Value", operator="min",
                                                             value=5)]))


# ─── Conjunction (compound guarantees) ───

def test_conjunction_holds_when_every_conjunct_holds():
    # Region pins exactly 2 selected; both conjuncts hold across the whole space.
    r = audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
              [CardinalityConstraint(min=1, max=4),
               ObjectiveBoundConstraint(objective="Cost", operator="max", value=20)])
    assert r["verdict"] == "holds"
    assert [p["verdict"] for p in r["properties"]] == ["holds", "holds"]


def test_conjunction_violated_names_the_failing_conjunct():
    # "1≤count≤4" holds, but A isn't forced — the compound guarantee fails on the second conjunct,
    # the witness genuinely violates it, and the breakdown still reports the first as holding.
    r = audit(_problem(constraints=[CardinalityConstraint(min=1, max=2)]),
              [CardinalityConstraint(min=1, max=4), ForceIncludeConstraint(option="A")])
    assert r["verdict"] == "violated"
    assert r["violated_property"]["type"] == "force_include"
    assert "A" not in r["witness"]["selected_options"]
    assert [p["verdict"] for p in r["properties"]] == ["holds", "violated"]


def test_conjunction_single_item_list_matches_single_property_payload():
    # A one-element list behaves as the single property: same verdict, no breakdown keys.
    single = audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
                   CardinalityConstraint(min=1, max=4))
    listed = audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
                   [CardinalityConstraint(min=1, max=4)])
    assert listed == single
    assert "properties" not in listed


def test_conjunction_empty_list_rejected():
    with pytest.raises(ValueError, match="empty"):
        audit(_problem(), [])


def test_unknown_option_raises():
    with pytest.raises(ValueError, match="unknown option"):
        audit(_problem(), ForceIncludeConstraint(option="ZZ"))


def test_incomplete_scores_declined_clearly():
    # Auditing before SCORE used to surface a raw KeyError ('A', 'Value'); the gate now
    # declines in words, matching the shape/backend gates.
    p = _problem()
    p.scores = [s for s in p.scores if s.option != "A"]
    with pytest.raises(ValueError, match="complete score matrix"):
        audit(p)


# ─── Verdict hardening: every non-proof status maps to inconclusive, never to holds ───
#
# `holds` requires a solver-proven Infeasible on the negation; anything else the solver can return
# (time limit, iteration limit, error statuses) must land on `inconclusive` with the raw status and
# a reason attached, and never surface a witness. Real-path check first (HiGHS at a zero time limit
# reliably returns 'Time limit reached' with no incumbent — verified empirically), then injected
# statuses for paths a real solve can't force deterministically.

def test_time_limit_real_solver_probe_maps_to_inconclusive(monkeypatch):
    # A stopped probe is not evidence of infeasibility. (Presolve fully resolves this tiny
    # model's *property* negations even at a zero limit — correctly — so the probe is the one
    # path a real zero-limit solve stalls deterministically; property-audit stops are covered
    # below via injected statuses.)
    monkeypatch.setattr(hb, "_MILP_TIME_LIMIT", 0.0)
    r = audit(_problem(constraints=[CardinalityConstraint(min=1, max=2)]))
    assert r["verdict"] == "inconclusive"
    assert r["witness"] is None                       # witnesses are verdict-gated
    assert "Time limit reached" in r["solver_status"]
    assert "Time limit reached" in r["reason"] and "INCONCLUSIVE" in r["reason"]


def test_time_limit_property_audit_maps_to_inconclusive(monkeypatch):
    _inject(monkeypatch, lambda eps, mc, n: ("Time limit reached", np.zeros(n)))
    r = audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
              CardinalityConstraint(min=1, max=4))
    assert r["verdict"] == "inconclusive"
    assert r["witness"] is None
    assert r["solver_status"] == ["Time limit reached"]
    assert "INCONCLUSIVE" in r["reason"]


def _inject(monkeypatch, inner):
    """Route audit()'s inner feasibility solve through a fake, for statuses a real solve
    can't force deterministically. audit() imports `_audit_milp_highs` at call time, so
    patching the backend module attribute reaches it."""
    monkeypatch.setattr(hb, "_audit_milp_highs", inner)


def test_error_status_maps_to_inconclusive_never_holds(monkeypatch):
    # A solver error on the negation is not a proof of infeasibility.
    _inject(monkeypatch, lambda eps, mc, n: ("Solve error", np.zeros(n)))
    r = audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
              CardinalityConstraint(min=1, max=4))
    assert r["verdict"] == "inconclusive"
    assert r["solver_status"] == ["Solve error"]
    assert r["witness"] is None


def test_mixed_statuses_map_to_inconclusive(monkeypatch):
    # Cardinality negates to two disjuncts; one proven Infeasible + one stopped ≠ holds.
    calls = iter(["Infeasible", "Time limit reached"])
    _inject(monkeypatch, lambda eps, mc, n: (next(calls), np.zeros(n)))
    r = audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
              CardinalityConstraint(min=2, max=3))
    assert r["verdict"] == "inconclusive"
    assert r["solver_status"] == ["Time limit reached"]


def test_vacuity_probe_timeout_does_not_claim_empty_region(monkeypatch):
    # Negation proven infeasible, but the non-emptiness re-probe stops early: the guarantee
    # stands (holds), while "the region is empty" (holds_vacuously) would be an overclaim.
    _inject(monkeypatch, lambda eps, mc, n:
            (("Time limit reached", np.zeros(n)) if not eps else ("Infeasible", np.zeros(n))))
    r = audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
              CardinalityConstraint(min=1, max=4))
    assert r["verdict"] == "holds"
    assert "unconfirmed" in r["note"] and "Time limit reached" in r["note"]


def test_solver_exception_propagates_as_error_not_verdict(monkeypatch):
    # A crash inside the solve is a hard tool error (the same convention as unfit shapes),
    # never converted into any verdict.
    def boom(eps, mc, n):
        raise RuntimeError("solver crashed")
    _inject(monkeypatch, boom)
    with pytest.raises(RuntimeError, match="solver crashed"):
        audit(_problem(constraints=[CardinalityConstraint(min=2, max=2)]),
              CardinalityConstraint(min=1, max=4))


# ─── Infeasibility diagnosis: minimal conflict sets on the user's named constraints ───
#
# When no plan is feasible, the audit attaches `conflicts`: a minimal set of the user's own
# constraints that cannot all hold together (deletion filtering). Members are leads, never
# auto-relaxed; one conflict at a time — clearing it re-probes for the next.

def _ckeys(conflicts):
    return {(c["type"], c.get("option"), c.get("objective"), c.get("value"))
            for c in conflicts["constraints"]}


def test_planted_budget_vs_forced_include_conflict():
    # A costs 8, "Cost ≤ 5" — together contradictory; the cardinality distractor is satisfiable.
    r = audit(_problem(constraints=[
        ObjectiveBoundConstraint(objective="Cost", operator="max", value=5),
        ForceIncludeConstraint(option="A"),
        CardinalityConstraint(min=1, max=3),
    ]))
    assert r["verdict"] == "no_feasible_plan"
    assert _ckeys(r["conflicts"]) == {
        ("objective_bound", None, "Cost", 5.0),
        ("force_include", "A", None, None),
    }
    assert r["conflicts"]["minimal"] is True
    assert "leads" in r["conflicts"]["caveat"]


def test_relaxing_a_conflict_member_restores_feasibility():
    r = audit(_problem(constraints=[ForceIncludeConstraint(option="A"),
                                    ObjectiveBoundConstraint(objective="Cost", operator="max", value=5)]))
    assert r["verdict"] == "no_feasible_plan"
    # Repair loop: drop one named member (the budget), re-probe — feasible again.
    r2 = audit(_problem(constraints=[ForceIncludeConstraint(option="A")]))
    assert r2["verdict"] == "feasible"


def test_two_independent_conflicts_surface_one_at_a_time():
    # Two non-interacting contradictions: A forced in AND out; "select exactly 3" vs "≤ 2 from
    # everything". No cross-subset is minimally infeasible, so the diagnosis must be exactly one
    # of the planted pairs — never a blend.
    conflict_1 = [ForceIncludeConstraint(option="A"), ForceExcludeConstraint(option="A")]
    conflict_2 = [CardinalityConstraint(min=3, max=3),
                  GroupLimitConstraint(options=["A", "B", "C", "D"], max=2)]
    r = audit(_problem(constraints=conflict_1 + conflict_2))
    assert r["verdict"] == "no_feasible_plan"
    found = _ckeys(r["conflicts"])
    planted = [_ckeys({"constraints": [c.model_dump(mode="json") for c in cs]})
               for cs in (conflict_1, conflict_2)]
    assert found in planted
    # Clear the reported conflict; the audit now reports the other one.
    cleared_first = found == planted[0]
    r2 = audit(_problem(constraints=conflict_2 if cleared_first else conflict_1))
    assert r2["verdict"] == "no_feasible_plan"
    assert _ckeys(r2["conflicts"]) == (planted[1] if cleared_first else planted[0])


def test_single_constraint_conflict_is_reported_alone():
    # min > n options: contradictory on its own.
    r = audit(_problem(constraints=[CardinalityConstraint(min=5, max=5)]))
    assert r["verdict"] == "no_feasible_plan"
    assert [c["type"] for c in r["conflicts"]["constraints"]] == ["cardinality"]


def test_vacuous_property_audit_carries_conflicts():
    r = audit(_problem(constraints=[ForceIncludeConstraint(option="A"),
                                    ForceExcludeConstraint(option="A")]),
              ForceIncludeConstraint(option="B"))
    assert r["verdict"] == "holds_vacuously"
    assert {c["type"] for c in r["conflicts"]["constraints"]} == {"force_include", "force_exclude"}


def test_diagnosis_on_satisfiable_set_claims_no_conflict():
    # Precondition guard: on a satisfiable set the filter must not present the whole model
    # as a "conflict" — it reports satisfiable instead.
    d = diagnose_conflicts(_problem(constraints=[CardinalityConstraint(min=1, max=2)]))
    assert d["satisfiable"] is True
    assert "constraints" not in d


def test_solve_path_infeasibility_analysis_carries_exact_conflicts():
    # The empty-frontier solve path (analyze_infeasibility) attaches the same solver-proven
    # conflict set on shapes the exact backend fits.
    from engine.optimizer import analyze_infeasibility
    out = analyze_infeasibility(_problem(constraints=[
        ForceIncludeConstraint(option="A"),
        ObjectiveBoundConstraint(objective="Cost", operator="max", value=5)]))
    assert _ckeys(out["conflicts"]) == {
        ("force_include", "A", None, None),
        ("objective_bound", None, "Cost", 5.0),
    }


def test_stopped_filter_solve_yields_inconclusive_diagnosis(monkeypatch):
    # A deletion-filter solve that stops without a verdict must never produce a guessed conflict.
    _inject(monkeypatch, lambda eps, mc, n: ("Time limit reached", np.zeros(n)))
    d = diagnose_conflicts(_problem(constraints=[ForceIncludeConstraint(option="A"),
                                                 ForceExcludeConstraint(option="A")]))
    assert d["inconclusive"] is True
    assert "constraints" not in d


# ─── Explorer payload (the MCP shape) ───

def test_explorer_payload_frames_verdict_and_echoes_property():
    prop = {"type": "force_include", "option": "A"}
    out = explorer.audit_property(_problem(constraints=[CardinalityConstraint(min=1, max=2)]), prop)
    assert out["verdict"] == "violated"
    assert out["audited"] == prop                       # entity-native echo of what was audited
    assert out["recommendation"] and out["next_steps"]  # pre-built framing for presentation


def test_explorer_payload_rejects_malformed_property():
    with pytest.raises(ValueError, match="valid constraint"):
        explorer.audit_property(_problem(), {"type": "not_a_constraint"})


def test_explorer_payload_parses_conjunction_and_echoes_it():
    props = [{"type": "cardinality", "min": 1, "max": 4},
             {"type": "force_include", "option": "A"}]
    out = explorer.audit_property(_problem(constraints=[CardinalityConstraint(min=1, max=2)]), props)
    assert out["verdict"] == "violated"
    assert out["audited"] == props
    assert len(out["properties"]) == 2
    with pytest.raises(ValueError, match="empty"):
        explorer.audit_property(_problem(), [])
    with pytest.raises(ValueError, match="valid constraint"):
        explorer.audit_property(_problem(), [{"type": "force_include", "option": "A"}, {"type": "nope"}])


def test_explorer_payload_defaults_floor_only_group_property():
    # A floor-only group guarantee ("at least 1 from {A,B}") omits `max`; the parser
    # defaults it to the vacuous group-size cap instead of rejecting the property.
    region = [CardinalityConstraint(min=1, max=2),
              GroupLimitConstraint(options=["A", "B"], min=1, max=2)]
    out = explorer.audit_property(
        _problem(constraints=region), {"type": "group_limit", "options": ["A", "B"], "min": 1})
    assert out["verdict"] == "holds"


def test_explorer_payload_error_names_the_intended_constraint_fields():
    # Discriminated validation: the error points at the group_limit fields, not an
    # arbitrary union member ("Input should be 'cardinality'").
    with pytest.raises(ValueError, match="group_limit"):
        explorer.audit_property(_problem(), {"type": "group_limit", "min": 1})


def test_explorer_payload_raises_on_unsupported_shape():
    # An unfit shape propagates the engine's ValueError, so the MCP layer returns a tool error
    # (consistent with solve), not a structured "unsupported" verdict.
    with pytest.raises(ValueError):
        explorer.audit_property(_problem(approach="proportional"), None)


# ─── Wire-level: explore audit over real stdio ───

_BOOTSTRAP = (
    "import sys;"
    "from engine.store import Store;"
    "import mcp_server.server as srv;"
    "srv.store = Store(sys.argv[1]);"
    "srv.main()"
)


def _payload(result):
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    return getattr(result, "structuredContent", None)


async def _drive_audit() -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    with tempfile.TemporaryDirectory() as data_dir:
        params = StdioServerParameters(
            command=sys.executable, args=["-c", _BOOTSTRAP, data_dir],
            cwd=str(REPO_ROOT), env=os.environ.copy(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                async def call(tool, **args):
                    return _payload(await session.call_tool(tool, args))

                pid = (await call("model", action="create"))["problem_id"]
                await call(
                    "model", action="update", problem_id=pid,
                    objectives=[{"name": "Value", "direction": "maximize"},
                                {"name": "Cost", "direction": "minimize"}],
                    options=[{"name": n} for n in ("A", "B", "C", "D")],
                    scores=[{"option": o, "objective": ob, "value": v}
                            for o, ob, v in [("A", "Value", 9), ("A", "Cost", 8),
                                             ("B", "Value", 7), ("B", "Cost", 5),
                                             ("C", "Value", 4), ("C", "Cost", 3),
                                             ("D", "Value", 2), ("D", "Cost", 1)]],
                    constraints=[{"type": "cardinality", "min": 1, "max": 2}],
                )
                # No prior solve — audit reads the model directly.
                probe = await call("explore", action="audit", problem_id=pid)
                violated = await call("explore", action="audit", problem_id=pid,
                                      audit_property={"type": "force_include", "option": "A"})
                bad = await call("explore", action="audit", problem_id=pid,
                                 audit_property={"type": "not_a_constraint"})
                return {"probe": probe, "violated": violated, "bad": bad}


@pytest.mark.skipif(not _HAS_MCP_CLIENT, reason="mcp stdio client not available")
def test_explore_audit_over_stdio():
    out = asyncio.run(asyncio.wait_for(_drive_audit(), timeout=120))

    # Feasibility probe crosses the wire and runs with no prior solve.
    assert out["probe"]["verdict"] == "feasible", out["probe"]

    # The audit_property param is delivered and drives a property audit end-to-end.
    v = out["violated"]
    assert v["verdict"] == "violated", v
    assert v["witness"] and "A" not in v["witness"]["selected_options"]
    # The read-side guidance pointer is attached for the agent.
    assert v.get("guidance_pointer", {}).get("section") == "Reading the Audit (explore audit)"

    # A malformed property is validated on the engine side — proof the param really reached it
    # (a dropped param would default to None and silently return a feasibility probe instead).
    assert out["bad"].get("error") and "valid constraint" in out["bad"]["error"]
