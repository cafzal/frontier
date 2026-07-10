"""Front gap detection (certify's completeness block) + targeted fill (scope="fill_gaps").

The inverse audit: certify's coverage measures what exact reclaims over NSGA; completeness
measures what the heuristic holds that the exact sweep never sampled, localized as witness
regions. The fill re-solves only those witnesses (certify_curated reuse) and merges — a gap
whose solve fails is discarded and reported, never merged as an uncertified incumbent.
"""

import importlib.util
import tempfile

import pytest

import engine.optimizer as optimizer
import mcp_server.server as srv
from engine.explorer import certify_against_exact, gap_witness_solutions
from engine.models import (
    CardinalityConstraint,
    Objective,
    Option,
    Problem,
    Run,
    Score,
    Solution,
)
from engine.optimizer import OptimizeMode, fill_gaps, optimize
from engine.store import Store

_HAS_HIGHS = importlib.util.find_spec("highspy") is not None


@pytest.fixture(autouse=True)
def tmp_store(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Store(tmpdir)
        monkeypatch.setattr(srv, "store", s)
        srv._injected_skills.clear()
        yield s


def _problem(n: int = 3, n_obj: int = 2) -> Problem:
    names = [f"P{i}" for i in range(n)]
    objs = [("value", "maximize"), ("cost", "minimize"), ("risk", "minimize")][:n_obj]
    return Problem(
        approach="binary",
        objectives=[Objective(name=o, direction=d) for o, d in objs],
        options=[Option(name=nm) for nm in names],
        scores=[Score(option=nm, objective=o, value=float((i * 5 + j * 3) % 7 + 1))
                for j, (o, _) in enumerate(objs) for i, nm in enumerate(names)],
    )


def _run(points: list[dict], selections: list[list[str]]) -> Run:
    return Run(solutions=[
        Solution(solution_id=i, selected_options=sel, objective_values=vals)
        for i, (vals, sel) in enumerate(zip(points, selections))
    ])


class TestDetection:
    def test_witness_flags_under_covered(self):
        p = _problem()
        # Exact overlay holds only the extremes; NSGA also holds a middle point neither covers.
        exact = _run([{"value": 10, "cost": 8}, {"value": 4, "cost": 2}],
                     [["P0", "P1"], ["P2"]])
        nsga = _run([{"value": 10, "cost": 8}, {"value": 7, "cost": 4}, {"value": 4, "cost": 2}],
                    [["P0", "P1"], ["P1"], ["P2"]])
        cert = certify_against_exact(p, nsga, exact)
        comp = cert["completeness"]
        assert comp["verdict"] == "under_covered"
        assert comp["heuristic_reclaims"] > comp["noise_floor"]
        region = comp["gap_regions"][0]
        assert region["witness_solution_ids"] == [1]  # self-certifying: names the middle point
        assert region["bounding_box"]["value"] == [7, 7]
        assert "fill_gaps" in cert["recommendation"]
        assert cert["coverage"]["mirror"] == "completeness"  # cross-referenced both ways
        assert comp["mirror"] == "coverage"

    def test_complete_overlay_is_quiet(self):
        p = _problem()
        exact = _run([{"value": 10, "cost": 8}, {"value": 7, "cost": 4}, {"value": 4, "cost": 2}],
                     [["P0", "P1"], ["P1"], ["P2"]])
        nsga = _run([{"value": 10, "cost": 8}, {"value": 7, "cost": 4}],
                    [["P0", "P1"], ["P1"]])
        cert = certify_against_exact(p, nsga, exact)
        assert cert["completeness"]["verdict"] == "complete"
        assert "gap_regions" not in cert["completeness"]
        assert "fill_gaps" not in cert["recommendation"]

    def test_duplicate_of_exact_point_is_not_a_witness(self):
        p = _problem()
        exact = _run([{"value": 10, "cost": 8}], [["P0", "P1"]])
        nsga = _run([{"value": 10, "cost": 8}], [["P0", "P1"]])
        assert gap_witness_solutions(p, nsga, exact) == []

    def test_three_objective_witnesses(self):
        # Boxes from nondominated witnesses — no ordered-front assumption at 3 objectives.
        p = _problem(n=4, n_obj=3)
        exact = _run([{"value": 10, "cost": 8, "risk": 5}, {"value": 4, "cost": 2, "risk": 6}],
                     [["P0", "P1"], ["P2"]])
        nsga = _run([{"value": 7, "cost": 4, "risk": 9}], [["P1"]])  # covered by neither
        witnesses = gap_witness_solutions(p, nsga, exact)
        assert [w.solution_id for w in witnesses] == [0]
        cert = certify_against_exact(p, nsga, exact)
        box = cert["completeness"]["gap_regions"][0]["bounding_box"]
        assert set(box) == {"value", "cost", "risk"}


class TestDetectionEdges:
    def test_empty_runs_do_not_crash(self):
        # Direct engine use can hand in empty runs; the guard must fire before any
        # broadcast against the sign vector.
        p = _problem()
        empty, full = _run([], []), _run([{"value": 10, "cost": 8}], [["P0", "P1"]])
        assert gap_witness_solutions(p, empty, full) == []
        assert [w.solution_id for w in gap_witness_solutions(p, full, empty)] == [0]

    def test_below_noise_witness_means_complete_and_noop_fill(self, monkeypatch):
        # ONE criterion decides verdict and work-list: a witness whose reclaim sits under
        # the noise floor reads "complete" AND yields an empty fill work-list — the fill
        # can never mutate an overlay the certificate called complete.
        p = _problem()
        exact = _run([{"value": 10, "cost": 8}, {"value": 4, "cost": 2}],
                     [["P0", "P1"], ["P2"]])
        exact.solver = "highs"
        nsga = _run([{"value": 10.0001, "cost": 8}, {"value": 4, "cost": 2}],
                    [["P0", "P1"], ["P2"]])  # epsilon-better on one axis: witness, negligible HV
        cert = certify_against_exact(p, nsga, exact)
        assert cert["completeness"]["verdict"] == "complete"
        assert gap_witness_solutions(p, nsga, exact) == []
        monkeypatch.setattr(optimizer, "certify_curated",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no solve on a complete overlay")))
        merged, report = fill_gaps(p, nsga, exact, solver="highs")
        assert merged.run_id == exact.run_id and report["gap_witnesses"] == 0

    def test_relative_tolerance_on_large_magnitude_objectives(self):
        # Solver-noise-scale differences on million-scale values must not mint witnesses.
        p = _problem()
        exact = _run([{"value": 1_000_000.0, "cost": 8}, {"value": 400_000.0, "cost": 2}],
                     [["P0", "P1"], ["P2"]])
        nsga = _run([{"value": 1_000_000.1, "cost": 8}], [["P0", "P1"]])  # ~1e-7 relative
        assert gap_witness_solutions(p, nsga, exact) == []


class TestFillEngine:
    def test_unfilled_gaps_discarded_and_reported(self, monkeypatch):
        # An inner solve that produces nothing must surface in the report, and the merge
        # must fall back to the base overlay — never an uncertified incumbent.
        p = _problem()
        exact = _run([{"value": 10, "cost": 8}, {"value": 4, "cost": 2}],
                     [["P0", "P1"], ["P2"]])
        exact.solver, exact.exact = "highs", True
        nsga = _run([{"value": 10, "cost": 8}, {"value": 7, "cost": 4}, {"value": 4, "cost": 2}],
                    [["P0", "P1"], ["P1"], ["P2"]])
        empty = Run(solutions=[])
        empty.solver = "highs"
        monkeypatch.setattr(optimizer, "certify_curated", lambda *a, **k: empty)
        merged, report = fill_gaps(p, nsga, exact, solver="highs")
        assert report["gap_witnesses"] == 1
        assert report["unfilled"] == 1 and report["filled"] == 0
        assert report["unfilled_witness_ids"] == [1]
        assert "discarded and reported" in report["note"]
        assert len(merged.solutions) == len(exact.solutions)  # base overlay intact

    def test_mismatched_backend_raises(self):
        p = _problem()
        exact = _run([{"value": 10, "cost": 8}], [["P0", "P1"]])
        exact.solver = "cuopt"
        nsga = _run([{"value": 7, "cost": 4}], [["P1"]])
        with pytest.raises(ValueError, match="cuopt"):
            fill_gaps(p, nsga, exact, solver="highs")

    def test_inputs_not_mutated_and_provenance_inherited(self, monkeypatch):
        # The merge must copy solutions (re-indexing restamps solution_id) and inherit the
        # base overlay's partial-sweep flags; certification is never silently demoted.
        p = _problem()
        exact = _run([{"value": 10, "cost": 8}, {"value": 4, "cost": 2}],
                     [["P0", "P1"], ["P2"]])
        exact.solver, exact.exact = "highs", True
        exact.time_limit, exact.time_limited = 60.0, True
        base_ids = [s.solution_id for s in exact.solutions]
        nsga = _run([{"value": 7, "cost": 4}], [["P1"]])
        fill = _run([{"value": 7, "cost": 4}], [["P1"]])
        fill.solver, fill.exact = "highs", True
        seen_exact = {}
        monkeypatch.setattr(optimizer, "certify_curated",
                            lambda *a, **k: seen_exact.setdefault("exact", k.get("exact")) and fill or fill)
        merged, report = fill_gaps(p, nsga, exact, solver="highs")  # exact NOT passed
        assert seen_exact["exact"] is True          # inherited from the certified overlay
        assert merged.exact is True and merged.solver == "highs"
        assert (merged.time_limit, merged.time_limited) == (60.0, True)
        assert merged.run_id != exact.run_id
        assert [s.solution_id for s in exact.solutions] == base_ids  # caller's overlay untouched
        assert report["filled"] == 1

    def test_user_max_solutions_never_shrinks_the_overlay(self, monkeypatch):
        p = _problem()
        exact = _run([{"value": 10, "cost": 8}, {"value": 7, "cost": 5}, {"value": 4, "cost": 2}],
                     [["P0", "P1"], ["P0"], ["P2"]])
        exact.solver = "highs"
        nsga = _run([{"value": 9, "cost": 6}], [["P1"]])
        fill = _run([{"value": 9, "cost": 6}], [["P1"]])
        fill.solver = "highs"
        monkeypatch.setattr(optimizer, "certify_curated", lambda *a, **k: fill)
        merged, _ = fill_gaps(p, nsga, exact, solver="highs", max_solutions=2)
        assert len(merged.solutions) == 4  # base 3 + fill 1; the caller cap bounds the fill pass only

    def test_noop_on_complete_overlay(self):
        p = _problem()
        exact = _run([{"value": 10, "cost": 8}, {"value": 7, "cost": 4}],
                     [["P0", "P1"], ["P1"]])
        exact.solver = "highs"
        nsga = _run([{"value": 10, "cost": 8}], [["P0", "P1"]])
        merged, report = fill_gaps(p, nsga, exact, solver="highs")
        assert merged.run_id == exact.run_id  # the overlay itself, unchanged (confirm semantics)
        assert report == {"gap_witnesses": 0, "filled": 0, "unfilled": 0,
                          "note": "overlay already complete — nothing to fill (no-op; confirm semantics)"}


@pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
class TestFillEndToEnd:
    def _solved_problem(self) -> Problem:
        p = _problem(n=12)
        p.constraints = [CardinalityConstraint(min=1, max=6)]
        p.run = optimize(p, mode=OptimizeMode.fast, seed=11)
        return p

    def test_fill_closes_thinned_overlay_and_invariant_holds(self):
        p = self._solved_problem()
        full_overlay = optimize(p, mode=OptimizeMode.fast, seed=11, solver="highs")
        assert len(full_overlay.solutions) >= 3
        # Thin the overlay to its two most extreme points — a sparse sweep in miniature.
        thinned = Run(solutions=[full_overlay.solutions[0], full_overlay.solutions[-1]])
        thinned.solver, thinned.exact = full_overlay.solver, full_overlay.exact
        witnesses = gap_witness_solutions(p, p.run, thinned)
        assert witnesses, "thinned overlay should leave NSGA witnesses"

        merged, report = fill_gaps(p, p.run, thinned, solver="highs")
        assert report["filled"] > 0
        assert len(merged.solutions) > len(thinned.solutions)
        assert merged.solver == "highs"
        # Certify invariant still holds after the merge: NSGA dominates no merged point
        # beyond the bounded-incumbent artifact — and every witness region shrank.
        cert = certify_against_exact(p, p.run, merged)
        after = gap_witness_solutions(p, p.run, merged)
        assert len(after) < len(witnesses)
        if report["unfilled"] == 0:
            assert cert["completeness"]["verdict"] == "complete"

    def test_refill_is_idempotent(self):
        p = self._solved_problem()
        overlay = optimize(p, mode=OptimizeMode.fast, seed=11, solver="highs")
        merged, first = fill_gaps(p, p.run, overlay, solver="highs")
        if first["unfilled"] == 0:
            again, report = fill_gaps(p, p.run, merged, solver="highs")
            assert report["gap_witnesses"] == 0
            assert again.run_id == merged.run_id


@pytest.mark.skipif(not _HAS_HIGHS, reason="highspy not installed")
class TestToolSurface:
    def _tool_problem(self) -> str:
        pid = srv.model(action="create")["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": nm} for nm in "ABCDEF"],
                  scores=[{"option": nm, "objective": ob, "value": float((i * 5 + j * 3) % 7 + 1)}
                          for j, ob in enumerate(["Rev", "Eff"]) for i, nm in enumerate("ABCDEF")],
                  constraints=[{"type": "cardinality", "min": 1, "max": 4}])
        return pid

    def test_fill_gaps_requires_exact_overlay(self):
        pid = self._tool_problem()
        srv.solve(action="run", problem_id=pid, seed=5)
        res = srv.solve(action="run", problem_id=pid, solver="highs", scope="fill_gaps")
        assert "error" in res and "scope='curated' or 'full'" in res["error"]

    def test_fill_gaps_requires_exact_solver(self):
        pid = self._tool_problem()
        res = srv.solve(action="run", problem_id=pid, scope="fill_gaps")
        assert "error" in res and "exact" in res["error"]

    def test_unknown_scope_errors(self):
        pid = self._tool_problem()
        res = srv.solve(action="run", problem_id=pid, solver="highs", scope="gaps")
        assert "error" in res and "Unknown scope" in res["error"]

    def test_scope_spelling_is_normalized(self):
        # 'Full' must run the FULL pass — the raw-string comparison once sent it to curated.
        pid = self._tool_problem()
        srv.solve(action="run", problem_id=pid, seed=5)
        res = srv.solve(action="run", problem_id=pid, solver="highs", scope="Full")
        assert res.get("overlay_scope") == "full"

    def test_fill_requires_fresh_results(self):
        pid = self._tool_problem()
        srv.solve(action="run", problem_id=pid, seed=5)
        srv.solve(action="run", problem_id=pid, solver="highs")
        srv.model(action="update", problem_id=pid,
                  scores=[{"option": "A", "objective": "Rev", "value": 99.0}])  # stales results
        res = srv.solve(action="run", problem_id=pid, solver="highs", scope="fill_gaps")
        assert "error" in res and "stale" in res["error"]

    def test_fill_requires_matching_backend(self):
        pid = self._tool_problem()
        srv.solve(action="run", problem_id=pid, seed=5)
        srv.solve(action="run", problem_id=pid, solver="highs")
        p = srv.store.load(pid)
        p.exact_run.solver = "cuopt"  # simulate an overlay from the other backend
        srv.store.save(p)
        res = srv.solve(action="run", problem_id=pid, solver="highs", scope="fill_gaps")
        assert "error" in res and "cuopt" in res["error"]

    def test_fill_after_certify_returns_report(self):
        pid = self._tool_problem()
        srv.solve(action="run", problem_id=pid, seed=5)
        srv.solve(action="run", problem_id=pid, solver="highs")  # scope defaults to curated
        res = srv.solve(action="run", problem_id=pid, solver="highs", scope="fill_gaps")
        assert res.get("overlay_scope") == "fill_gaps"
        assert "fill" in res
        assert res["fill"]["gap_witnesses"] == res["fill"]["filled"] + res["fill"]["unfilled"]
        assert res["solver_used"] == "highs"
