"""Scenario-side staleness: the fingerprint treatment the base runs already had.

Three live captures motivated these, one per test class below — in each, a base-model edit
left the stored scenario side frozen and nothing said so:

1. `explore scenario_results` reported a 0/33 scenario "wipeout" that was an artifact of
   `constraint_overrides` frozen at the pre-edit rules (they replace the base set wholesale).
2. `solve run_scenarios` happily re-ran the frozen pre-edit overrides — `status: complete`,
   an excluded option still selected, no warning.
3. `explore scenario_results` served PRE-edit per-scenario ranges beside a freshly re-solved
   base run, with `regret.per_solution` ranking the NEW base run's solution_ids against the
   OLD scenario frontiers. Confidently wrong, no signal anywhere.

The base-run convention is the reference: runs carry the fingerprint of the inputs they were
solved on, comparison is round-trip-safe, and a cleared flag never vouches for a frontier it
can't see. Reads that stay internally coherent are served WITH a marker; the one read that
joins the two sides incoherently (regret) declines instead.
"""

import tempfile

import pytest

from engine import explorer, models
from engine.models import Objective, Option, Problem, Score
from engine.optimizer import constraints_fingerprint
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


_BASE_CONSTRAINTS = [{"type": "cardinality", "min": 1, "max": 3}]


def _scenario_problem(n_opt=5) -> str:
    """A solve-ready binary problem with one scenario that carries constraint_overrides."""
    objs = [Objective(name="value", direction="maximize", aggregation="sum"),
            Objective(name="cost", direction="minimize", aggregation="sum")]
    opts = [Option(name=f"o{i}") for i in range(n_opt)]
    scores = [Score(option=f"o{i}", objective=obj, value=float((i * 7 + k * 5) % 13 + 1))
              for i in range(n_opt) for k, obj in enumerate(("value", "cost"))]
    p = Problem(name="scen", objectives=objs, options=opts, scores=scores)
    srv.store.save(p)
    srv.model(action="update", problem_id=p.problem_id, constraints=_BASE_CONSTRAINTS)
    srv.model(action="update", problem_id=p.problem_id, scenario_config={
        "enabled": True,
        "scenarios": [
            {"name": "tight", "constraint_overrides": [{"type": "cardinality", "min": 1, "max": 2}]},
            {"name": "loose"},
        ],
    })
    return p.problem_id


def _solve_both(pid: str) -> None:
    assert srv.solve(action="run", problem_id=pid, seed=7, wait_seconds=60)["status"] != "running"
    r = srv.solve(action="run_scenarios", problem_id=pid, seed=7, wait_seconds=60)
    assert r.get("scenarios_optimized") == 2, r
    return r


def _edit_base_constraints(pid: str) -> None:
    """A base-constraint edit — the edit scenario overrides do NOT inherit."""
    srv.model(action="update", problem_id=pid,
              constraints=[{"type": "cardinality", "min": 1, "max": 4}])


class TestFrozenOverridesReadAsFindings:
    """Receipt 1 — a scenario whose overrides predate the base edit must say so on the read,
    not just restate the wholesale-replacement convention in `varies`."""

    def test_scenario_results_carries_the_stale_marker(self):
        pid = _scenario_problem()
        _solve_both(pid)
        p = srv.store.load(pid)
        assert not explorer.get_scenario_results(p).get("scenario_results_stale")

        _edit_base_constraints(pid)
        res = explorer.get_scenario_results(srv.store.load(pid))
        assert res["scenario_results_stale"] is True
        assert "solve run_scenarios" in res["scenario_stale_note"]
        # The marker leads the payload — a reader meets it before the numbers.
        assert next(iter(res)) == "scenario_results_stale"
        # ...and rides into viz_data, so the chart surface can't render the ranges unmarked.
        assert res["viz_data"]["scenario_results_stale"] is True
        assert res["viz_data"]["scenario_stale_note"] == res["scenario_stale_note"]

    def test_frozen_overrides_are_named_per_scenario(self):
        """Even after the scenario runs are refreshed, the OVERRIDES are still frozen at the
        pre-edit rules — the scenario carrying them is named, the one without is not."""
        pid = _scenario_problem()
        _solve_both(pid)
        _edit_base_constraints(pid)
        _solve_both(pid)  # scenario runs now current; overrides still pre-edit

        p = srv.store.load(pid)
        assert not models.scenario_results_stale(p), "re-run scenarios clear the run marker"
        res = explorer.get_scenario_results(p)
        assert res["per_scenario"]["tight"]["constraint_overrides_stale"] is True
        assert "replace the base rules wholesale" in \
            res["per_scenario"]["tight"]["constraint_overrides_note"]
        assert "constraint_overrides_stale" not in res["per_scenario"]["loose"]

    def test_frontiers_and_per_scenario_tradeoffs_carry_it_too(self):
        pid = _scenario_problem()
        _solve_both(pid)
        _edit_base_constraints(pid)
        p = srv.store.load(pid)
        assert explorer.get_scenario_frontiers(p)["scenario_results_stale"] is True
        # Per-scenario detail reads inherit it through frontier_source (one place).
        prov = explorer.get_tradeoffs(p, scenario="tight")["frontier_source"]
        assert prov["scenario_results_stale"] is True
        # The base frontier is a different frontier — its provenance says nothing of it.
        assert "scenario_results_stale" not in explorer.get_tradeoffs(p)["frontier_source"]


class TestRunScenariosWarnsAtSolveTime:
    """Receipt 2 — re-running scenarios with overrides authored against different base rules
    must name them in the response rather than reporting a clean `complete`."""

    def test_response_names_scenarios_with_pre_edit_overrides(self):
        pid = _scenario_problem()
        _solve_both(pid)
        _edit_base_constraints(pid)
        res = _solve_both(pid)

        stale = res["constraint_overrides_stale"]
        assert stale["scenarios"] == ["tight"]           # `loose` carries no overrides
        assert "REPLACE the base rules wholesale" in stale["note"]
        assert "solve run_scenarios" in stale["note"]

    def test_no_warning_when_overrides_match_the_current_rules(self):
        pid = _scenario_problem()
        res = _solve_both(pid)
        assert "constraint_overrides_stale" not in res

    def test_restating_the_overrides_clears_the_warning(self):
        pid = _scenario_problem()
        _solve_both(pid)
        _edit_base_constraints(pid)
        # The remedy the note names: restate the overrides against the current base rules.
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "tight",
                 "constraint_overrides": [{"type": "cardinality", "min": 1, "max": 3}]},
                {"name": "loose"},
            ],
        })
        assert "constraint_overrides_stale" not in _solve_both(pid)

    def test_warning_is_a_fingerprint_comparison_not_a_diff(self):
        """Authored-against is a stamp, so re-stating the SAME overrides against the same
        base rules is silent, and only the base rules moving makes it speak."""
        pid = _scenario_problem()
        p = srv.store.load(pid)
        fp = constraints_fingerprint(p.constraints)
        assert models.stale_scenario_overrides(p, fp) == []
        assert models.stale_scenario_overrides(p, "some-other-rule-set") == ["tight"]


class TestRegretJoinRefusesAcrossRuns:
    """Receipt 3 — the read that MIXES runs. Every other scenario read is internally
    coherent and gets served with a marker; regret re-scores the current base solutions
    against stored scenario frontiers, so a base/scenario fingerprint disagreement makes
    the join itself meaningless. It declines."""

    def test_regret_withheld_when_base_and_scenario_runs_disagree(self):
        pid = _scenario_problem()
        _solve_both(pid)
        assert explorer.scenario_regret(srv.store.load(pid))["available"] is True

        # The capture's shape: edit the model, re-solve the BASE only.
        srv.model(action="update", problem_id=pid,
                  scores=[{"option": "o0", "objective": "value", "value": 99.0}])
        srv.solve(action="run", problem_id=pid, seed=7, wait_seconds=60)

        p = srv.store.load(pid)
        assert models.scenario_base_mismatch(p) is True
        regret = explorer.scenario_regret(p)
        assert regret["available"] is False
        assert regret["reason"] == "base_scenario_mismatch"
        assert "solve run_scenarios" in regret["note"]

        # And the surrounding read still serves — coherent parts flagged, join withheld.
        res = explorer.get_scenario_results(p)
        assert res["scenario_results_stale"] is True
        assert res["regret"]["available"] is False
        assert "minimax_choice" not in res["regret"]

    def test_coherent_pair_still_serves_regret_when_both_sides_are_stale(self):
        """Both sides solved against the SAME earlier model is a coherent join — served,
        with the staleness marked. Refusal is for mixing, not for age."""
        pid = _scenario_problem()
        _solve_both(pid)
        srv.model(action="update", problem_id=pid,
                  scores=[{"option": "o0", "objective": "value", "value": 99.0}])

        p = srv.store.load(pid)
        assert models.scenario_base_mismatch(p) is False
        res = explorer.get_scenario_results(p)
        assert res["scenario_results_stale"] is True
        assert res["regret"]["available"] is True

    def test_unstamped_runs_are_unknowable_not_mismatched(self):
        """A pre-stamp run can't be proven stale — guessing would cry wolf on every legacy
        problem, so the join serves (same rule as the read marker)."""
        pid = _scenario_problem()
        _solve_both(pid)
        p = srv.store.load(pid)
        srv.model(action="update", problem_id=pid,
                  scores=[{"option": "o0", "objective": "value", "value": 99.0}])
        srv.solve(action="run", problem_id=pid, seed=7, wait_seconds=60)
        p = srv.store.load(pid)
        assert models.scenario_base_mismatch(p) is True          # stamped: provably mixed
        p.scenario_run.base_fingerprint = None
        assert models.scenario_base_mismatch(p) is False          # unstamped: unknowable
        p.scenario_run.solve_fingerprint = None
        assert models.scenario_results_stale(p) is False


class TestRoundTripAndScopedFingerprints:
    """The base-run convention, held on the scenario side: an edit-then-revert lands back at
    NOT stale, and each frontier is compared only against inputs it actually reads."""

    def test_edit_then_revert_is_not_stale(self):
        pid = _scenario_problem()
        _solve_both(pid)
        _edit_base_constraints(pid)
        p = srv.store.load(pid)
        assert models.scenario_results_stale(p) is True
        assert explorer.get_scenario_results(p)["scenario_results_stale"] is True

        srv.model(action="update", problem_id=pid, constraints=_BASE_CONSTRAINTS)
        p = srv.store.load(pid)
        assert models.scenario_results_stale(p) is False, (
            "restoring the solved constraint set must clear the scenario marker too")
        assert "scenario_results_stale" not in explorer.get_scenario_results(p)
        assert p.results_stale is False
        # The revert also restores the authored-against rules, so the override warning goes.
        assert models.stale_scenario_overrides(p, constraints_fingerprint(p.constraints)) == []

    def test_scenario_only_edit_does_not_flag_the_base_frontier(self):
        """The inverse over-staleness: a scenario_config-only edit cannot affect a base
        frontier, so it must not mark one stale."""
        pid = _scenario_problem()
        assert srv.solve(action="run", problem_id=pid, seed=7,
                         wait_seconds=60)["status"] != "running"
        assert srv.store.load(pid).results_stale is False

        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [{"name": "tight",
                           "constraint_overrides": [{"type": "cardinality", "min": 1, "max": 2}]},
                          {"name": "loose"},
                          {"name": "extra"}],
        })
        assert srv.store.load(pid).results_stale is False, (
            "adding a scenario cannot change the base frontier — it must not read as stale")

    def test_scenario_only_edit_does_flag_the_scenario_frontier(self):
        """...and the frontier it CAN affect still flags, per-run and on the blanket flag."""
        pid = _scenario_problem()
        _solve_both(pid)
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [{"name": "tight",
                           "constraint_overrides": [{"type": "cardinality", "min": 1, "max": 2}]},
                          {"name": "loose"},
                          {"name": "extra"}],
        })
        p = srv.store.load(pid)
        assert models.scenario_results_stale(p) is True
        assert p.results_stale is True
        assert srv.model(action="get", problem_id=pid,
                         section="summary")["scenario_results_stale"] is True

    def test_run_scenarios_does_not_vouch_for_a_stale_base_frontier(self):
        """Refreshing the scenario set clears staleness for the scenario set only — the
        base frontier still predates the edit, so the blanket flag stays honest."""
        pid = _scenario_problem()
        _solve_both(pid)
        srv.model(action="update", problem_id=pid,
                  scores=[{"option": "o0", "objective": "value", "value": 42.0}])
        _solve_both(pid)  # re-runs the base too — clean baseline
        srv.model(action="update", problem_id=pid,
                  scores=[{"option": "o1", "objective": "value", "value": 43.0}])
        srv.solve(action="run_scenarios", problem_id=pid, seed=7, wait_seconds=60)

        p = srv.store.load(pid)
        assert models.scenario_results_stale(p) is False   # the scenario set is fresh
        assert p.results_stale is True, "the base run still predates the edit"

    def test_a_loaded_bundle_starts_stamped(self):
        """Bundled examples are the demo path these receipts came from — their runs arrive
        unstamped, so load stamps them; otherwise a post-load edit stays undetectable."""
        loaded = srv.model(action="load", source="capacity_planning")
        p = srv.store.load(loaded["problem_id"])
        assert p.scenario_run and p.scenario_run.solve_fingerprint
        assert models.scenario_results_stale(p) is False

        srv.model(action="update", problem_id=p.problem_id,
                  scores=[{"option": p.options[0].name, "objective": p.objectives[0].name,
                           "value": 1.0}])
        assert models.scenario_results_stale(srv.store.load(p.problem_id)) is True
