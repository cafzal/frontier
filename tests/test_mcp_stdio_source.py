"""Live-transport regression: ``explore``'s ``source`` kwarg must survive the real MCP
stdio round trip.

``test_solver_selection.test_source_exact_targets_overlay_when_both_present`` exercises the
same behavior *in-process* (calling ``srv.explore(..., source="exact")`` directly), so it
cannot catch a dispatch/wiring regression where ``source`` is dropped on the wire — e.g. the
param being removed from the tool signature (FastMCP would then schema-strip it) or the
threading into ``explorer`` being lost. This test spawns the server as a subprocess and drives
it through the actual stdio transport, the same path the live ``mcp__frontier__explore`` tool
uses.

The decisive, dependency-free assertion is ``source="bogus"`` raising "Unknown source": that
error originates in ``engine.explorer._require_run``, so it can only surface if ``source``
actually reached the engine. The highs-gated case additionally proves ``source="exact"``
retargets the exact overlay end-to-end — the handoff the PR #13 ``explore certify`` next_steps
points the agent at.
"""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_HAS_HIGHS = importlib.util.find_spec("highspy") is not None
_HAS_MCP_CLIENT = importlib.util.find_spec("mcp.client.stdio") is not None
REPO_ROOT = Path(__file__).resolve().parent.parent

# Boot the server with an isolated store so the test never writes to the real data dir.
# The tools resolve the module-global ``store`` at call time, so reassigning it before
# ``main()`` redirects every persistence path — the same indirection the in-process tests
# monkeypatch, applied here in the subprocess.
_BOOTSTRAP = (
    "import sys;"
    "from engine.store import Store;"
    "import mcp_server.server as srv;"
    "srv.store = Store(sys.argv[1]);"
    "srv.main()"
)


def _payload(result):
    """Extract a Frontier tool's dict return from a CallToolResult.

    These tools return a dict, which FastMCP serializes to a TextContent JSON block — always
    present and unambiguous, so parse that. (structuredContent is only attached when a tool
    declares an output schema, which these bare ``-> dict`` tools do not.)
    """
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    structured = getattr(result, "structuredContent", None)
    return structured if isinstance(structured, dict) else None


async def _drive_repro() -> dict:
    """Build a problem, solve it, and probe ``explore source=...`` over real stdio."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    with tempfile.TemporaryDirectory() as data_dir:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", _BOOTSTRAP, data_dir],
            cwd=str(REPO_ROOT),
            # Inherit the full parent env — env=None strips the child to a ~6-var allowlist,
            # dropping VIRTUAL_ENV / PYTHONPATH / CONDA_PREFIX, so it could fail to import deps.
            env=os.environ.copy(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                async def call(tool, **args):
                    return _payload(await session.call_tool(tool, args))

                pid = (await call("model", action="create"))["problem_id"]
                await call(
                    "model", action="update", problem_id=pid,
                    objectives=[{"name": "Rev", "direction": "maximize"},
                                {"name": "Eff", "direction": "minimize"}],
                    options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
                    scores=[{"option": o, "objective": ob, "value": v}
                            for o, ob, v in [("A", "Rev", 8), ("A", "Eff", 5),
                                             ("B", "Rev", 6), ("B", "Eff", 3),
                                             ("C", "Rev", 9), ("C", "Eff", 7)]],
                    constraints=[{"type": "cardinality", "min": 1, "max": 2}],
                )

                nsga = await call("solve", action="run", problem_id=pid, seed=42)
                assert "run_id" in nsga, f"NSGA solve did not return a run: {nsga}"
                highs = None
                if _HAS_HIGHS:
                    # Exact overlay alongside the NSGA run, so the problem holds both frontiers
                    # before we probe — mirrors the solve()->solve(solver=...)->explore repro.
                    highs = await call("solve", action="run", problem_id=pid, seed=42, solver="highs")
                    assert "run_id" in highs, f"HiGHS solve did not return a run: {highs}"

                default = await call("explore", action="solutions", problem_id=pid)
                bogus = await call("explore", action="solutions", problem_id=pid, source="bogus")

                out = {
                    "nsga_run_id": nsga["run_id"],
                    "default_run_id": default.get("run_id"),
                    "default_frontier_source": default.get("frontier_source"),
                    "bogus_error": bogus.get("error"),
                }
                if _HAS_HIGHS:
                    exact = await call("explore", action="solutions", problem_id=pid, source="exact")
                    out["exact_run_id"] = highs["run_id"]
                    out["exact_source_run_id"] = exact.get("run_id")
                    out["exact_source_error"] = exact.get("error")
                    out["exact_frontier_source"] = exact.get("frontier_source")
                return out


async def _drive_background() -> dict:
    """Force a background solve over real stdio (wait_seconds=0) and poll it to completion —
    proves a long solve hands back a job handle and finishes off the turn, the timeout fix."""
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
                    objectives=[{"name": "Rev", "direction": "maximize"},
                                {"name": "Eff", "direction": "minimize"}],
                    options=[{"name": n} for n in ("A", "B", "C", "D")],
                    scores=[{"option": o, "objective": ob, "value": v}
                            for o, ob, v in [("A", "Rev", 8), ("A", "Eff", 5), ("B", "Rev", 6),
                                             ("B", "Eff", 3), ("C", "Rev", 9), ("C", "Eff", 7),
                                             ("D", "Rev", 4), ("D", "Eff", 2)]],
                )

                handle = await call("solve", action="run", problem_id=pid, wait_seconds=0, seed=42)
                final = handle
                if handle.get("status") == "running":
                    for _ in range(400):
                        final = await call("solve", action="status", job_id=handle["job_id"])
                        if final.get("status") != "running":
                            break
                        await asyncio.sleep(0.05)
                return {"handle": handle, "final": final}


@pytest.mark.skipif(not _HAS_MCP_CLIENT, reason="mcp stdio client not available")
def test_background_solve_polls_to_completion_over_stdio():
    out = asyncio.run(asyncio.wait_for(_drive_background(), timeout=120))
    # wait_seconds=0 hands back a running job handle instead of blocking the turn.
    assert out["handle"].get("status") == "running", f"expected a job handle, got {out['handle']}"
    assert out["handle"].get("job_id")
    # Polling `solve status` returns the finished frontier — the solve ran off the turn.
    final = out["final"]
    assert final.get("status") == "complete", f"solve did not complete on poll: {final}"
    assert final.get("solutions_found", 0) >= 1 and "run_id" in final


@pytest.mark.skipif(not _HAS_MCP_CLIENT, reason="mcp stdio client not available")
def test_source_survives_stdio_transport():
    # Hard timeout so a wedged stdio handshake fails loudly instead of hanging CI (a healthy
    # run is ~1-2s; the ceiling only guards a stuck subprocess). asyncio.wait_for cancels the
    # coroutine, and stdio_client's __aexit__ then terminates the server subprocess.
    out = asyncio.run(asyncio.wait_for(_drive_repro(), timeout=120))

    # Dependency-free proof that ``source`` crossed the transport: an unknown value can only be
    # rejected by the engine if the kwarg was actually delivered. If ``source`` were dropped on
    # the wire it would default to None and this call would silently return the NSGA frontier.
    assert out["bogus_error"], "explore source='bogus' returned no error — source was dropped on the wire"
    assert "Unknown source" in out["bogus_error"]
    # The default still targets the exploratory NSGA run.
    assert out["default_run_id"] == out["nsga_run_id"]

    # Provenance label: the default frontier is unambiguously tagged heuristic, so a dropped or
    # omitted source can't pass it off as exact.
    default_prov = out["default_frontier_source"]
    assert default_prov, "explore result is missing the frontier_source provenance label"
    assert default_prov["kind"] == "heuristic"
    assert default_prov["run_id"] == out["nsga_run_id"]
    assert default_prov["solver"].startswith("nsga")

    if _HAS_HIGHS:
        # source="exact" retargets the analytics at the exact overlay end-to-end (run_id flips
        # to exact_run, not the NSGA run) — the originally-reported failure mode.
        assert out["exact_source_error"] is None
        assert out["exact_source_run_id"] == out["exact_run_id"]
        assert out["exact_source_run_id"] != out["nsga_run_id"]

        # With both frontiers present, the heuristic default advertises the exact overlay...
        assert default_prov.get("exact_overlay_available") is True
        # ...and the exact result is tagged exact (no "go find the overlay" hint).
        exact_prov = out["exact_frontier_source"]
        assert exact_prov and exact_prov["kind"] == "exact"
        assert exact_prov["solver"] == "highs"
        assert exact_prov["run_id"] == out["exact_run_id"]
        assert "exact_overlay_available" not in exact_prov
