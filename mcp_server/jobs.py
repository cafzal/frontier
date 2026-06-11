"""Background solve jobs: dispatch, registry, staleness fingerprint, delivery.

Split from server.py (F2). The dispatcher takes the heavy solve+persist as a
``work_fn`` closure, so this module needs no store access of its own; delivery
injects solution_interpreter via the guidance module (same once-per-problem
throttle as before). Mutable registry/locks are shared with the server
namespace by object identity, so tests that clear them keep working.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

from engine.models import OptimizeMode, Problem

from mcp_server.guidance import (
    _SOLUTION_INTERPRETER_PROMPT,
    _inject_skill,
    _mark_injected,
    _was_injected,
)

# ─── Long-running solve jobs ───
#
# FastMCP runs *sync* tool functions directly on the asyncio event loop, so a 150 s
# optimize() blocks the whole server (no pings/keepalive, no other calls) AND blows the
# client's per-call timeout — the turn dies before the solve returns. So `solve run`
# dispatches the heavy optimize+persist to a worker thread, waits inline only a short
# budget, and otherwise hands back a {status:"running", job_id} handle the agent polls via
# `solve status`. Fast solves still finish within the wait and return inline (unchanged).
# The worker persists the run, so results survive even if polling is interrupted.
# Determinism holds: the optimizer's seeded-RNG shim is thread-local, and optimize_scenarios
# already runs solves off-thread the same way.

_SOLVE_WAIT_SECONDS = float(os.environ.get("FRONTIER_SOLVE_WAIT_SECONDS", "10"))  # inline budget
_SOLVE_WAIT_CAP = float(os.environ.get("FRONTIER_SOLVE_WAIT_CAP", "60"))          # max caller-set wait
_SOLVE_WORKERS = int(os.environ.get("FRONTIER_SOLVE_WORKERS", "4"))
_SOLVE_JOB_TTL = float(os.environ.get("FRONTIER_SOLVE_JOB_TTL", "3600"))          # keep finished jobs (s)
# Cap concurrent + queued solves so a flood of `solve run` calls can't grow the executor
# queue (each item pins a Problem snapshot) without bound. The UI rate-limits its callers,
# but direct MCP clients don't, so this is their backstop. Generous for normal flows
# (NSGA + an exact overlay, per-scenario runs); tune via env.
_SOLVE_MAX_INFLIGHT = int(os.environ.get("FRONTIER_SOLVE_MAX_INFLIGHT", "24"))


@dataclass
class SolveJob:
    job_id: str
    problem_id: str
    action: str            # "run" | "run_scenarios"
    label: str             # human phase label for progress narration, e.g. "solve: highs thorough"
    started_at: datetime
    done: threading.Event
    status: str = "running"      # running | complete | error
    result: dict | None = None   # delivered payload (success / infeasible / stale)
    error: str | None = None     # unexpected worker crash only (optimize errors are caught in-body)
    finished_at: datetime | None = None
    delivered: bool = False      # once-guard for delivery side effects (skill injection)


_solve_jobs: dict[str, SolveJob] = {}
_solve_jobs_lock = threading.Lock()
_solve_pool = ThreadPoolExecutor(max_workers=_SOLVE_WORKERS, thread_name_prefix="frontier-solve")

# Serializes the read-modify-write a solve worker does to attach its run to the stored problem.
# Concurrent same-problem background solves are a normal flow (run NSGA, then run an exact
# overlay) — without this their load→modify→save could interleave and lose one run. Held only
# for the brief persist, on the worker thread; foreground edits made *during* a solve are caught
# separately by the fingerprint stale-guard.
_store_write_lock = threading.Lock()

# Problem fields that determine a solve's result. If any change while a background solve runs,
# the computed frontier no longer matches the stored problem, so the worker reports `stale`
# rather than overwriting the concurrent edit (results/curation/etc. are deliberately excluded).
_SOLVE_INPUT_FIELDS = {
    "approach", "objectives", "options", "scores", "constraints",
    "interaction_matrices", "scenario_config",
}


def _solve_fingerprint(p: Problem) -> str:
    payload = p.model_dump(mode="json", include=_SOLVE_INPUT_FIELDS)
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _solve_label(action: str, mode: OptimizeMode | None, solver: str | None,
                 exact: bool, time_limit: float | None) -> str:
    parts = [f"{solver or 'nsga'} {(mode.value if mode else 'fast')}"]
    if exact and solver:
        parts.append("exact-certified")
    if time_limit:
        parts.append(f"≤{time_limit:g}s")
    base = "scenarios" if action == "run_scenarios" else "solve"
    return f"{base}: {' '.join(parts)}"


def _resolve_wait(wait_seconds: float | None) -> float:
    if wait_seconds is None:
        return _SOLVE_WAIT_SECONDS
    return max(0.0, min(float(wait_seconds), _SOLVE_WAIT_CAP))


def _reap_jobs() -> None:
    """Evict finished jobs past their TTL so the registry can't grow unbounded. Called (under
    the lock) when a new job is registered — cheap, and recent results stay pollable."""
    now = datetime.now(timezone.utc)
    expired = [jid for jid, j in _solve_jobs.items()
               if j.finished_at and (now - j.finished_at).total_seconds() > _SOLVE_JOB_TTL]
    for jid in expired:
        _solve_jobs.pop(jid, None)


def _running_handle(job: SolveJob) -> dict:
    elapsed = (datetime.now(timezone.utc) - job.started_at).total_seconds()
    return {
        "status": "running",
        "job_id": job.job_id,
        "problem_id": job.problem_id,
        "action": job.action,
        "label": job.label,
        "started_at": job.started_at.isoformat(),
        "elapsed_s": round(elapsed, 1),
        "poll_with": f"solve(action='status', job_id='{job.job_id}')",
        "note": (
            f"Solve is running in the background ({job.label}) — expected for thorough, exact, "
            "or large problems. Poll `solve status` with this job_id until status is 'complete'; "
            "the run is persisted, so `explore` works once it finishes. Between polls, tell the "
            "user it's optimizing and how long it's taken so far — don't claim results yet."
        ),
    }


def _deliver(problem_id: str, job: SolveJob) -> dict:
    """Render a finished job for return. Adds job_id + a status that reflects the OUTCOME
    (complete / infeasible / error / stale), and on a successful solve injects
    solution_interpreter once per problem, at delivery time (inline-complete or status),
    always on a request thread (never the worker), so the injection state stays
    single-threaded. Delivery does NOT re-arm optimization_strategy — per the model
    docstring only an objectives/options shape change re-arms it, so the post-solve
    tweak→re-solve loop doesn't re-pay the full core. The injection side effect fires
    once per job (`delivered`), so re-polling a finished job is a pure read and can't
    suppress guidance for a later model cycle."""
    if job.status == "error":  # unexpected worker crash
        return {"status": "error", "job_id": job.job_id, "feasible": False, "error": job.error}
    result = dict(job.result or {})
    result["job_id"] = job.job_id
    succeeded = "run_id" in result or "scenarios_optimized" in result
    if "status" not in result:  # stale already sets its own; classify the rest by outcome
        result["status"] = ("complete" if succeeded
                            else "error" if result.get("error")
                            else "infeasible")
    if succeeded and not job.delivered:  # a real frontier was produced, first delivery only
        if not _was_injected(problem_id, "solution_interpreter"):
            _inject_skill(result, "solution_interpreter", _SOLUTION_INTERPRETER_PROMPT)
            _mark_injected(problem_id, "solution_interpreter")
    job.delivered = True
    return result


def _solve_dispatch(p: Problem, action: str, work_fn, *, label: str,
                    wait_seconds: float | None) -> dict:
    """Run ``work_fn`` (the heavy solve+persist) in a worker thread, wait inline up to the
    resolved budget, and return the full result if it finished — else a running handle to poll."""
    job = SolveJob(
        job_id=uuid.uuid4().hex,
        problem_id=p.problem_id,
        action=action,
        label=label,
        started_at=datetime.now(timezone.utc),
        done=threading.Event(),
    )
    with _solve_jobs_lock:
        _reap_jobs()
        inflight = sum(1 for j in _solve_jobs.values() if j.finished_at is None)
        if inflight >= _SOLVE_MAX_INFLIGHT:
            return {
                "status": "error",
                "error": (
                    f"Too many solves in progress ({inflight} of {_SOLVE_MAX_INFLIGHT} "
                    "in flight). Wait for running solves to finish (poll their job_ids) "
                    "or retry shortly."
                ),
            }
        _solve_jobs[job.job_id] = job

    def _runner() -> None:
        try:
            job.result = work_fn()
            job.status = "complete"
        except Exception as e:  # optimize errors are caught in-body; this is an unexpected crash
            job.status = "error"
            job.error = str(e)
        finally:
            job.finished_at = datetime.now(timezone.utc)
            job.done.set()

    _solve_pool.submit(_runner)
    if job.done.wait(timeout=_resolve_wait(wait_seconds)):
        return _deliver(p.problem_id, job)
    return _running_handle(job)


def _solve_status(job_id: str | None) -> dict:
    if not job_id:
        return {"error": "status requires a job_id (from a prior solve run/run_scenarios response)."}
    with _solve_jobs_lock:
        job = _solve_jobs.get(job_id)
    if job is None:
        return {"error": f"No solve job '{job_id}'. It may have expired (TTL) or the id is wrong."}
    if not job.done.is_set():
        return _running_handle(job)
    return _deliver(job.problem_id, job)
