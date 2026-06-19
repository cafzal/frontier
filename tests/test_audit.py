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

import pytest

pytest.importorskip("highspy")

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
from engine.optimizer import audit

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


def test_nonsum_aggregation_declined_raises():
    # The exact MILP only encodes additive objectives; a min-aggregated one is out of scope.
    with pytest.raises(ValueError):
        audit(_problem(aggregation="min"))


def test_unknown_option_raises():
    with pytest.raises(ValueError, match="unknown option"):
        audit(_problem(), ForceIncludeConstraint(option="ZZ"))


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
