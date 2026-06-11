"""Background-solve job mechanism: dispatch, status polling, persistence, the stale-guard,
the time_limit knob, and the preserved synchronous fast path.

These exercise the internal handler functions directly (like test_server.py), which is enough
because the wire path is independently covered by test_mcp_stdio_source.py. The product point:
a long solve no longer blocks the MCP turn — `solve run` hands back a job handle the agent
polls via `solve status`, while fast solves still return inline.
"""

import tempfile
import threading
import time
from datetime import datetime, timezone

import pytest

from engine.models import Objective, Option, Problem, Score
from engine.optimizer import _build_termination, _was_time_limited
from engine.store import Store

import mcp_server.server as srv


@pytest.fixture(autouse=True)
def tmp_store(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Store(tmpdir)
        monkeypatch.setattr(srv, "store", s)
        srv._injected_skills.clear()
        with srv._solve_jobs_lock:
            srv._solve_jobs.clear()
        yield s


def _ready(approach="binary", n_opt=5, n_obj=2, name="t") -> str:
    """Persist a solve-ready problem (≥2 objectives, ≥3 options, complete score matrix). Built
    directly like test_certify does, since `model create` doesn't apply a full score matrix."""
    objs = [Objective(name=f"obj{j}", direction="maximize" if j % 2 == 0 else "minimize",
                      aggregation="sum") for j in range(n_obj)]
    opts = [Option(name=f"o{i}") for i in range(n_opt)]
    scores = [Score(option=f"o{i}", objective=f"obj{j}", value=float((i * 7 + j * 5) % 13 + 1))
              for i in range(n_opt) for j in range(n_obj)]
    p = Problem(name=name, approach=approach, objectives=objs, options=opts, scores=scores)
    srv.store.save(p)
    return p.problem_id


def _poll(job_id, timeout=60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = srv.solve(action="status", job_id=job_id)
        if s.get("status") != "running":
            return s
        time.sleep(0.02)
    raise AssertionError(f"solve job {job_id} did not finish within {timeout}s")


def _fake_job(job_id, *, finished):
    """A registry entry with no real worker, for exercising the in-flight cap."""
    now = datetime.now(timezone.utc)
    return srv.SolveJob(
        job_id=job_id, problem_id="p", action="run", label="x",
        started_at=now, done=threading.Event(),
        finished_at=now if finished else None,
    )


class TestInflightCap:
    """A flood of solve calls must not queue unbounded work (each item pins a Problem
    snapshot). Past the cap, dispatch rejects with a clear error instead of submitting."""

    def test_rejects_when_inflight_cap_reached(self, monkeypatch):
        import mcp_server.jobs as jobs
        monkeypatch.setattr(jobs, "_SOLVE_MAX_INFLIGHT", 2)
        with srv._solve_jobs_lock:
            srv._solve_jobs.clear()
            for i in range(2):  # two unfinished jobs fill the cap
                srv._solve_jobs[f"j{i}"] = _fake_job(f"j{i}", finished=False)
        r = jobs._solve_dispatch(Problem(name="x"), "run", lambda: {"run_id": "r"},
                                 label="t", wait_seconds=0)
        assert r.get("status") == "error"
        assert "Too many solves" in r["error"]

    def test_finished_jobs_do_not_count_toward_cap(self, monkeypatch):
        import mcp_server.jobs as jobs
        monkeypatch.setattr(jobs, "_SOLVE_MAX_INFLIGHT", 1)
        with srv._solve_jobs_lock:
            srv._solve_jobs.clear()
            srv._solve_jobs["done"] = _fake_job("done", finished=True)  # not in flight
        # A finished job doesn't occupy a slot, so a real solve still dispatches.
        r = srv.solve(action="run", problem_id=_ready(), wait_seconds=0, seed=42)
        assert r.get("status") in ("running", "complete")


class TestBackgroundDispatch:
    def test_wait_zero_returns_running_handle(self):
        pid = _ready()
        r = srv.solve(action="run", problem_id=pid, wait_seconds=0, seed=42)
        assert r["status"] == "running"
        assert r["job_id"] and r["problem_id"] == pid and r["action"] == "run"
        assert "poll_with" in r and "label" in r and "elapsed_s" in r
        # No frontier yet — the agent must poll, not report results.
        assert "solutions_found" not in r

    def test_status_polls_to_complete(self):
        pid = _ready()
        r = srv.solve(action="run", problem_id=pid, wait_seconds=0, seed=42)
        done = _poll(r["job_id"])
        assert done["status"] == "complete"
        assert done["job_id"] == r["job_id"]
        assert done["solutions_found"] >= 1 and "run_id" in done
        # solution_interpreter is injected at delivery, not in the worker.
        assert done["_skill_guidance"]["skill"] == "solution_interpreter"

    def test_default_wait_returns_inline_complete(self):
        """A tiny solve finishes within the inline budget, so the call is synchronous as
        before — this is what keeps the ~117 existing direct-call tests green."""
        pid = _ready()
        r = srv.solve(action="run", problem_id=pid, seed=7)
        assert r["status"] == "complete" and "run_id" in r and r["solutions_found"] >= 1
        assert r["job_id"]  # still traceable to a job

    def test_persistence_survives_without_polling(self):
        """The worker persists before signalling done, so the run is on disk (and explorable)
        the moment status reports complete — even if the agent never polled mid-run."""
        pid = _ready()
        r = srv.solve(action="run", problem_id=pid, wait_seconds=0, seed=42)
        done = _poll(r["job_id"])
        stored = srv.store.load(pid)
        assert stored.run is not None and stored.run.run_id == done["run_id"]
        assert "error" not in srv.explore(action="tradeoffs", problem_id=pid)

    def test_skill_injected_once_across_solves(self):
        pid = _ready()
        first = _poll(srv.solve(action="run", problem_id=pid, wait_seconds=0, seed=1)["job_id"])
        second = _poll(srv.solve(action="run", problem_id=pid, wait_seconds=0, seed=2)["job_id"])
        assert "_skill_guidance" in first
        assert "_skill_guidance" not in second  # once-per-problem throttle preserved

    def test_deliver_labels_outcome_not_just_complete(self):
        """A finished job's `status` reflects the OUTCOME — an infeasible/error solve is not
        labeled the success-implying 'complete', and neither injects solution_interpreter."""
        def _job(result):
            return srv.SolveJob(job_id="j", problem_id="p", action="run", label="x",
                                started_at=datetime.now(timezone.utc), done=threading.Event(),
                                status="complete", result=result)
        infeasible = srv._deliver("p", _job({"feasible": False, "binding_constraints": []}))
        assert infeasible["status"] == "infeasible" and "_skill_guidance" not in infeasible
        errored = srv._deliver("p", _job({"feasible": False, "error": "boom"}))
        assert errored["status"] == "error" and "_skill_guidance" not in errored
        ok = srv._deliver("p", _job({"run_id": "r", "solutions_found": 3}))
        assert ok["status"] == "complete" and "_skill_guidance" in ok

    def test_repolling_finished_job_does_not_rearm_injection(self):
        """Delivery side effects (skill inject/reset) fire once per job, so re-reading a finished
        job's result can't suppress guidance the user re-armed via a later model edit."""
        pid = _ready()
        jid = srv.solve(action="run", problem_id=pid, wait_seconds=0, seed=1)["job_id"]
        _poll(jid)
        srv._mark_injected(pid, "optimization_strategy")  # a model cycle re-arms it
        srv.solve(action="status", job_id=jid)            # re-poll the completed job
        assert srv._was_injected(pid, "optimization_strategy")  # still armed — not reset again


class TestStatusAction:
    def test_unknown_job_id_errors_cleanly(self):
        assert "error" in srv.solve(action="status", job_id="does-not-exist")

    def test_status_requires_job_id(self):
        assert "error" in srv.solve(action="status")

    def test_status_needs_no_problem_id(self):
        """`status` is dispatched before any problem load, so a held job_id is enough."""
        pid = _ready()
        r = srv.solve(action="run", problem_id=pid, wait_seconds=0, seed=3)
        done = _poll(r["job_id"])
        assert done["status"] == "complete"


class TestStaleGuard:
    def test_body_reports_stale_without_clobbering(self):
        """If the problem's solve-inputs change while a background solve runs, the worker
        reports stale and writes nothing — never overwriting the concurrent edit."""
        pid = _ready()
        p = srv.store.load(pid)
        runs_before = len(p.runs)
        res = srv._solve_run_body(p, "stale-fingerprint", mode=None, seed=1)
        assert res["status"] == "stale" and res["stale"] is True
        after = srv.store.load(pid)
        assert after.run is None and len(after.runs) == runs_before

    def test_fingerprint_changes_with_structural_edit(self):
        pid = _ready()
        fp1 = srv._solve_fingerprint(srv.store.load(pid))
        srv.model(action="update", problem_id=pid, scores=[{"option": "o0", "objective": "obj0", "value": 99.0}])
        fp2 = srv._solve_fingerprint(srv.store.load(pid))
        assert fp1 != fp2

    def test_fingerprint_stable_under_nonstructural_edit(self):
        """Curating a solution must NOT mark an in-flight solve stale — only solve-inputs do."""
        pid = _ready()
        _poll(srv.solve(action="run", problem_id=pid, wait_seconds=0, seed=5)["job_id"])
        fp1 = srv._solve_fingerprint(srv.store.load(pid))
        srv.explore(action="curate", problem_id=pid, solution_id=0, custom_name="pick")
        fp2 = srv._solve_fingerprint(srv.store.load(pid))
        assert fp1 == fp2


class TestConcurrency:
    def test_two_problems_solve_without_store_corruption(self):
        pid1, pid2 = _ready(name="a"), _ready(name="b")
        r1 = srv.solve(action="run", problem_id=pid1, wait_seconds=0, seed=1)
        r2 = srv.solve(action="run", problem_id=pid2, wait_seconds=0, seed=2)
        assert r1["status"] == "running" and r2["status"] == "running"
        s1, s2 = _poll(r1["job_id"]), _poll(r2["job_id"])
        assert s1["status"] == "complete" and s2["status"] == "complete"
        p1, p2 = srv.store.load(pid1), srv.store.load(pid2)
        assert p1.run.run_id == s1["run_id"] and p2.run.run_id == s2["run_id"]
        assert p1.run.run_id != p2.run.run_id

    def test_concurrent_same_problem_solves_dont_lose_a_run(self):
        """Two background solves on the SAME problem (the run-then-overlay flow) serialize their
        persist under the store-write lock, so neither read-modify-write is lost: the second run
        becomes current and the first is archived, rather than silently clobbered."""
        pid = _ready()
        r1 = srv.solve(action="run", problem_id=pid, wait_seconds=0, seed=1)
        r2 = srv.solve(action="run", problem_id=pid, wait_seconds=0, seed=2)
        s1, s2 = _poll(r1["job_id"]), _poll(r2["job_id"])
        assert s1["status"] == "complete" and s2["status"] == "complete"
        p = srv.store.load(pid)
        run_ids = {p.run.run_id} | {r.run_id for r in p.runs}
        assert {s1["run_id"], s2["run_id"]} <= run_ids  # both runs survived the concurrent persist


class TestTimeLimit:
    def test_uncapped_run_is_not_time_limited(self):
        pid = _ready()
        r = srv.solve(action="run", problem_id=pid, seed=1, time_limit=5.0)
        assert r["time_limit"] == 5.0 and r["time_limited"] is False

    def test_small_cap_truncates_and_flags(self):
        """A large proportional run capped well below its natural runtime returns a best-so-far
        frontier flagged time_limited — bounded, not failed."""
        pid = _ready(approach="proportional", n_opt=30, n_obj=2)
        r = srv.solve(action="run", problem_id=pid, mode="thorough", time_limit=0.2,
                      wait_seconds=30, seed=1)
        assert r["status"] == "complete"
        assert r["time_limited"] is True
        assert r["solutions_found"] >= 1  # still returns the population's non-dominated front

    def test_build_termination_helper(self):
        # Uncapped → the plain generation tuple (prior behavior, unchanged).
        assert _build_termination(50, None) == ("n_gen", 50)
        assert _build_termination(50, 0) == ("n_gen", 50)
        # Capped → a TerminationCollection that ORs the gen and time criteria.
        term = _build_termination(50, 1.5)
        assert term.__class__.__name__ == "TerminationCollection"

    def test_was_time_limited_helper(self):
        assert _was_time_limited(None, 100.0) is False
        assert _was_time_limited(0, 100.0) is False      # 0 = uncapped
        assert _was_time_limited(-1.0, 100.0) is False   # negative = uncapped, never limited
        assert _was_time_limited(2.0, 1.99) is False     # converged just under the cap → NOT limited
        assert _was_time_limited(2.0, 2.0) is True       # hit the cap
        assert _was_time_limited(2.0, 2.5) is True        # ran past the cap (gen-boundary overshoot)
