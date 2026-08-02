"""Tests for the metrics module — automated checkpoints and diagnostics."""

import pytest

from engine.metrics import (
    compute_metrics,
    data_metrics,
    diagnostics,
    framing_metrics,
    frontier_quality,
    outcome_metrics,
    readiness,
    solve_metrics,
)
from engine.models import (
    CardinalityConstraint,
    Feedback,
    ObjectiveBoundConstraint,
    Objective,
    Option,
    Problem,
    QualityIndicators,
    Run,
    Score,
    Solution,
)
from engine.optimizer import optimize


@pytest.fixture
def empty_problem():
    return Problem()


@pytest.fixture
def framed_problem():
    return Problem(
        objectives=[
            Objective(name="Revenue", direction="maximize", unit="$"),
            Objective(name="Effort", direction="minimize", unit="days"),
        ],
        options=[Option(name=n) for n in ["A", "B", "C", "D", "E"]],
        constraints=[CardinalityConstraint(min=2, max=3)],
    )


@pytest.fixture
def scored_problem(framed_problem):
    p = framed_problem
    p.scores = [
        Score(option="A", objective="Revenue", value=8),
        Score(option="A", objective="Effort", value=5),
        Score(option="B", objective="Revenue", value=6),
        Score(option="B", objective="Effort", value=3),
        Score(option="C", objective="Revenue", value=9),
        Score(option="C", objective="Effort", value=7),
        Score(option="D", objective="Revenue", value=4),
        Score(option="D", objective="Effort", value=2),
        Score(option="E", objective="Revenue", value=7),
        Score(option="E", objective="Effort", value=4),
    ]
    return p


@pytest.fixture
def solved_problem(scored_problem):
    p = scored_problem
    p.run = optimize(p)
    return p


class TestFramingMetrics:
    def test_empty_problem(self, empty_problem):
        m = framing_metrics(empty_problem)
        assert m["objective_count"] == 0
        assert m["option_count"] == 0
        assert m["constraint_count"] == 0
        assert m["directions_set"] is True  # vacuously true

    def test_framed_problem(self, framed_problem):
        m = framing_metrics(framed_problem)
        assert m["objective_count"] == 2
        assert m["option_count"] == 5
        assert m["constraint_count"] == 1
        assert m["directions_set"] is True


class TestDataMetrics:
    def test_no_scores(self, framed_problem):
        m = data_metrics(framed_problem)
        assert m["score_completeness"] == 0.0
        assert len(m["missing_scores"]) == 10  # 2 obj * 5 opt

    def test_complete_scores(self, scored_problem):
        m = data_metrics(scored_problem)
        assert m["score_completeness"] == 1.0
        assert len(m["missing_scores"]) == 0

    def test_partial_scores(self, framed_problem):
        p = framed_problem
        p.scores = [
            Score(option="A", objective="Revenue", value=8),
            Score(option="A", objective="Effort", value=5),
        ]
        m = data_metrics(p)
        assert m["score_completeness"] == 0.2  # 2 of 10

    def test_variance_computed(self, scored_problem):
        m = data_metrics(scored_problem)
        assert "Revenue" in m["score_variance_by_objective"]
        assert m["score_variance_by_objective"]["Revenue"] > 0

    def test_dominated_options(self):
        """Option D is dominated by B (lower revenue AND higher effort)."""
        p = Problem(
            objectives=[
                Objective(name="Revenue", direction="maximize"),
                Objective(name="Effort", direction="minimize"),
            ],
            options=[Option(name="B"), Option(name="D")],
            scores=[
                Score(option="B", objective="Revenue", value=6),
                Score(option="B", objective="Effort", value=3),
                # D is worse on both: lower revenue, same effort
                Score(option="D", objective="Revenue", value=4),
                Score(option="D", objective="Effort", value=3),
            ],
        )
        m = data_metrics(p)
        assert "D" in m["dominated_options"]

    @staticmethod
    def _twin_problem(scores_by_option):
        """2-objective problem from {option: (revenue, effort)} — the twin-detection rig."""
        return Problem(
            objectives=[Objective(name="Revenue", direction="maximize"),
                        Objective(name="Effort", direction="minimize")],
            options=[Option(name=n) for n in scores_by_option],
            scores=[Score(option=n, objective=o, value=v)
                    for n, (r, e) in scores_by_option.items()
                    for o, v in (("Revenue", r), ("Effort", e))],
        )

    def test_exact_twins_flagged_not_dominated(self):
        """The catch strict dominance can't make: an exact tie dominates nothing, yet A
        and B are the same bet entered twice."""
        m = data_metrics(self._twin_problem(
            {"A": (6, 3), "B": (6, 3), "C": (10, 9), "D": (1, 1)}))
        assert {"options": ["A", "B"], "max_gap_fraction": 0.0} in m["near_duplicate_options"]
        assert "A" not in m["dominated_options"] and "B" not in m["dominated_options"]

    def test_near_twins_within_tolerance_flagged(self):
        """Within 2% of each objective's spread on every objective → twins, closest first."""
        m = data_metrics(self._twin_problem(
            {"A": (6.0, 3.0), "B": (6.1, 3.05), "C": (16.0, 9.0), "D": (1.0, 1.0)}))
        pairs = [d["options"] for d in m["near_duplicate_options"]]
        assert ["A", "B"] in pairs
        gap = next(d["max_gap_fraction"] for d in m["near_duplicate_options"]
                   if d["options"] == ["A", "B"])
        assert 0.0 < gap <= 0.02

    def test_distinct_options_not_flagged(self):
        m = data_metrics(self._twin_problem(
            {"A": (6, 3), "B": (8, 5), "C": (10, 9), "D": (1, 1)}))
        assert m["near_duplicate_options"] == []

    def test_dominated_by_a_hair_twin_rides_both_lists(self):
        """A twin edged out on every objective is dominated AND a near-duplicate — both
        flags stand; the twin read ('pick either') is the truthful framing."""
        m = data_metrics(self._twin_problem(
            {"A": (6.0, 3.0), "B": (6.05, 2.95), "C": (16.0, 9.0), "D": (1.0, 1.0)}))
        assert "A" in m["dominated_options"]
        assert ["A", "B"] in [d["options"] for d in m["near_duplicate_options"]]

    def test_flat_axis_distinguishes_nothing(self):
        """A zero-spread objective ties every option; twins are decided on the live axes."""
        m = data_metrics(self._twin_problem(
            {"A": (6.0, 3.0), "B": (6.0, 3.0), "C": (16.0, 3.0)}))
        assert ["A", "B"] in [d["options"] for d in m["near_duplicate_options"]]
        assert ["A", "C"] not in [d["options"] for d in m["near_duplicate_options"]]

    def test_near_duplicates_capped_with_total(self):
        """Twelve identical options → 66 pairs, capped at 10 with the total echoed —
        the data_metrics cap convention."""
        names = [f"O{i}" for i in range(12)]
        m = data_metrics(self._twin_problem(
            {**{n: (5.0, 5.0) for n in names}, "far": (50.0, 50.0)}))
        assert len(m["near_duplicate_options"]) == 10
        assert m["near_duplicate_options_total"] == 66

    def test_incomplete_scores_excluded_from_twin_check(self):
        """Only complete-scored options can be compared on every objective."""
        p = self._twin_problem({"A": (6, 3), "B": (6, 3), "C": (16, 9)})
        p.scores = [s for s in p.scores if not (s.option == "B" and s.objective == "Effort")]
        assert data_metrics(p)["near_duplicate_options"] == []


class TestSolveMetrics:
    def test_no_run(self, scored_problem):
        m = solve_metrics(scored_problem)
        assert m["solve_success"] is False
        assert m["solution_count"] == 0

    def test_with_run(self, solved_problem):
        m = solve_metrics(solved_problem)
        assert m["solve_success"] is True
        assert m["solution_count"] >= 1
        assert isinstance(m["objective_variation"], dict)
        assert "Revenue" in m["objective_variation"]
        assert isinstance(m["option_coverage"], dict)

    def test_run_param_describes_that_run_not_problem_run(self, solved_problem):
        """An exact overlay's response metrics must describe the run being returned —
        passing `run=` overrides the default problem.run (the NSGA-echo regression)."""
        overlay = Run(solutions=[Solution(
            solution_id=0, selected_options=["A"],
            objective_values={"Revenue": 8.0, "Effort": 2.0},
        )], quality=QualityIndicators(hypervolume_normalized=0.5, spacing_cv=0.1))
        m = solve_metrics(solved_problem, run=overlay)
        assert m["solution_count"] == 1
        assert m["option_coverage"]["A"] == 1
        # The default path still describes problem.run — the two calls agree only
        # through their run argument, never implicitly.
        assert solve_metrics(solved_problem)["solution_count"] == len(solved_problem.run.solutions)
        d = diagnostics(solved_problem, run=Run(solutions=[]))
        assert any(x["pattern"] == "zero_solutions" for x in d)


class TestOutcomeMetrics:
    def test_no_feedback(self, solved_problem):
        m = outcome_metrics(solved_problem)
        assert m["user_selected_solution"] is None
        assert m["user_rating"] is None
        assert m["feedback_count"] == 0

    def test_with_feedback(self, solved_problem):
        solved_problem.feedback.append(
            Feedback(solution_id=0, rating=4, notes="Good", stage="decision")
        )
        m = outcome_metrics(solved_problem)
        assert m["user_selected_solution"] == 0
        assert m["user_rating"] == 4
        assert m["feedback_count"] == 1


class TestDiagnostics:
    def test_no_run(self, scored_problem):
        diags = diagnostics(scored_problem)
        assert len(diags) == 0  # no run, no diagnostics

    def test_zero_solutions(self):
        """Run exists but has no solutions → zero_solutions diagnostic."""
        p = Problem(run=Run(solutions=[]))
        diags = diagnostics(p)
        assert any(d["pattern"] == "zero_solutions" for d in diags)

    def test_option_never_selected(self):
        """An option that's never in any solution gets flagged."""
        p = Problem(
            objectives=[
                Objective(name="X", direction="maximize"),
            ],
            options=[Option(name="A"), Option(name="B"), Option(name="C")],
            run=Run(solutions=[
                Solution(solution_id=0, selected_options=["A"], objective_values={"X": 10}),
                Solution(solution_id=1, selected_options=["A", "B"], objective_values={"X": 15}),
            ]),
        )
        diags = diagnostics(p)
        patterns = [d["pattern"] for d in diags]
        assert "option_never_selected" in patterns
        # C is never selected
        flagged_opts = [d["option"] for d in diags if d["pattern"] == "option_never_selected"]
        assert "C" in flagged_opts

    def test_zero_mean_axis_with_real_spread_not_clustered(self):
        """A mean-straddling-zero objective (net P&L −50/+50) has genuine spread; the
        zero-safe scale must keep it from reading as rel_range 0 → false clustering."""
        p = Problem(
            objectives=[
                Objective(name="NetPnL", direction="maximize"),
                Objective(name="Cost", direction="minimize"),
            ],
            options=[Option(name="A"), Option(name="B")],
            run=Run(solutions=[
                Solution(solution_id=0, selected_options=["A"], objective_values={"NetPnL": -50.0, "Cost": 100.0}),
                Solution(solution_id=1, selected_options=["B"], objective_values={"NetPnL": 50.0, "Cost": 101.0}),
            ]),
        )
        diags = diagnostics(p)
        assert not any(d["pattern"] == "clustered_solutions" for d in diags)
        assert not any(d.get("objective") == "NetPnL" and d["pattern"] == "low_variation_objective" for d in diags)

    def test_two_near_identical_solutions_cluster(self):
        """A 2-point frontier within 5% on every objective is clustered — it used to
        fall between this check (which wanted ≥3) and the <1% all-flat POOR gate."""
        p = Problem(
            objectives=[
                Objective(name="Cost", direction="minimize"),
                Objective(name="Quality", direction="maximize"),
            ],
            options=[Option(name="A"), Option(name="B")],
            run=Run(solutions=[
                Solution(solution_id=0, selected_options=["A"], objective_values={"Cost": 100.0, "Quality": 50.0}),
                Solution(solution_id=1, selected_options=["B"], objective_values={"Cost": 103.0, "Quality": 51.5}),
            ]),
        )
        diags = diagnostics(p)
        assert any(d["pattern"] == "clustered_solutions" for d in diags)

    def test_low_variation_objective(self):
        """Objective with <10% variation gets flagged."""
        p = Problem(
            objectives=[
                Objective(name="Cost", direction="minimize"),
                Objective(name="Quality", direction="maximize"),
            ],
            options=[Option(name="A"), Option(name="B")],
            run=Run(solutions=[
                Solution(solution_id=0, selected_options=["A"],
                         objective_values={"Cost": 100, "Quality": 50}),
                Solution(solution_id=1, selected_options=["B"],
                         objective_values={"Cost": 102, "Quality": 90}),
            ]),
        )
        diags = diagnostics(p)
        patterns = [d["pattern"] for d in diags]
        # Cost has ~2% variation (100 to 102), should be flagged
        assert "low_variation_objective" in patterns

    def test_binding_constraint(self):
        """Constraint at boundary gets flagged."""
        p = Problem(
            objectives=[
                Objective(name="Cost", direction="minimize"),
                Objective(name="Quality", direction="maximize"),
            ],
            options=[Option(name="A"), Option(name="B")],
            constraints=[
                ObjectiveBoundConstraint(
                    objective="Cost", operator="max", value=100,
                ),
            ],
            run=Run(solutions=[
                Solution(solution_id=0, selected_options=["A"],
                         objective_values={"Cost": 98, "Quality": 50}),
                Solution(solution_id=1, selected_options=["B"],
                         objective_values={"Cost": 99, "Quality": 70}),
            ]),
        )
        diags = diagnostics(p)
        patterns = [d["pattern"] for d in diags]
        assert "binding_constraint" in patterns


class TestComputeMetrics:
    def test_returns_all_sections(self, solved_problem):
        m = compute_metrics(solved_problem)
        assert "framing" in m
        assert "data" in m
        assert "solve" in m
        assert "outcome" in m
        assert "diagnostics" in m


# ─── frontier_quality (1.11) ───


class TestFrontierQuality:
    objectives = [
        Objective(name="Revenue", direction="maximize"),
        Objective(name="Effort", direction="minimize"),
    ]

    def test_empty_frontier_is_poor(self):
        q = frontier_quality([], self.objectives)
        assert q["status"] == "POOR"
        assert q["gates"]["frontier_returned"] is False
        assert q["gates"]["non_trivial"] is False
        assert q["gates"]["diverse"] is False
        assert q["issues"]

    def test_single_solution_is_poor(self):
        sols = [Solution(solution_id=1, selected_options=["A"], objective_values={"Revenue": 8, "Effort": 5})]
        q = frontier_quality(sols, self.objectives)
        assert q["status"] == "POOR"
        assert q["gates"]["non_trivial"] is False
        assert any("1 solution" in i.lower() or "degenerate" in i.lower() for i in q["issues"])

    def test_all_objectives_flat_is_poor(self):
        # Same objective values across solutions → no real frontier
        sols = [
            Solution(solution_id=i, selected_options=[f"O{i}"], objective_values={"Revenue": 10.0, "Effort": 5.0})
            for i in range(3)
        ]
        q = frontier_quality(sols, self.objectives)
        assert q["status"] == "POOR"
        assert q["gates"]["non_trivial"] is False
        assert any("flat" in i.lower() or "collapsed" in i.lower() for i in q["issues"])

    def test_diverse_frontier_is_good(self):
        sols = [
            Solution(solution_id=1, selected_options=["A", "B"], objective_values={"Revenue": 10, "Effort": 8}),
            Solution(solution_id=2, selected_options=["B", "C"], objective_values={"Revenue": 14, "Effort": 12}),
            Solution(solution_id=3, selected_options=["A", "C"], objective_values={"Revenue": 18, "Effort": 17}),
        ]
        q = frontier_quality(sols, self.objectives, spacing_cv=0.3)
        assert q["status"] == "GOOD"
        assert q["gates"]["diverse"] is True
        assert q["issues"] == []

    def test_high_spacing_cv_warning(self):
        sols = [
            Solution(solution_id=1, selected_options=["A"], objective_values={"Revenue": 10, "Effort": 8}),
            Solution(solution_id=2, selected_options=["B"], objective_values={"Revenue": 14, "Effort": 12}),
            Solution(solution_id=3, selected_options=["C"], objective_values={"Revenue": 18, "Effort": 17}),
        ]
        q = frontier_quality(sols, self.objectives, spacing_cv=2.0)
        assert q["status"] == "WARNING"
        assert q["gates"]["diverse"] is False
        assert any("spacing" in i.lower() for i in q["issues"])

    def test_proportional_concentration_warning(self):
        # One solution allocates 90% to a single option → single-winner WARNING
        sols = [
            Solution(solution_id=1, selected_options=["A", "B"],
                     objective_values={"Revenue": 10, "Effort": 8},
                     allocations={"A": 60, "B": 40}),
            Solution(solution_id=2, selected_options=["A", "B"],
                     objective_values={"Revenue": 14, "Effort": 12},
                     allocations={"A": 90, "B": 10}),
            Solution(solution_id=3, selected_options=["A", "B"],
                     objective_values={"Revenue": 18, "Effort": 17},
                     allocations={"A": 50, "B": 50}),
        ]
        q = frontier_quality(sols, self.objectives, spacing_cv=0.3)
        assert q["status"] == "WARNING"
        assert q["gates"]["diverse"] is False
        assert any("90%" in i and "A" in i for i in q["issues"])

    def test_proportional_well_diversified_is_good(self):
        sols = [
            Solution(solution_id=1, selected_options=["A", "B", "C"],
                     objective_values={"Revenue": 10, "Effort": 8},
                     allocations={"A": 40, "B": 30, "C": 30}),
            Solution(solution_id=2, selected_options=["A", "B", "C"],
                     objective_values={"Revenue": 14, "Effort": 12},
                     allocations={"A": 30, "B": 40, "C": 30}),
            Solution(solution_id=3, selected_options=["A", "B", "C"],
                     objective_values={"Revenue": 18, "Effort": 17},
                     allocations={"A": 30, "B": 30, "C": 40}),
        ]
        q = frontier_quality(sols, self.objectives, spacing_cv=0.3)
        assert q["status"] == "GOOD"
        assert q["gates"]["diverse"] is True

    def test_objective_value_exactly_zero_flat(self):
        # Edge case: all objective values are zero — must be detected as flat (no zero-division)
        sols = [
            Solution(solution_id=i, selected_options=[f"O{i}"], objective_values={"Revenue": 0.0, "Effort": 0.0})
            for i in range(3)
        ]
        q = frontier_quality(sols, self.objectives)
        assert q["status"] == "POOR"
        assert q["gates"]["non_trivial"] is False

    def test_one_flat_objective_is_warning(self):
        # One axis dead (<1% variation) while the other varies: the plans differ, but
        # 'Effort' is out of the tradeoff — must downgrade, not present as clean.
        sols = [
            Solution(solution_id=i, selected_options=[f"O{i}"],
                     objective_values={"Revenue": 10.0 + 4 * i, "Effort": 100.0 + 0.1 * i})
            for i in range(4)
        ]
        q = frontier_quality(sols, self.objectives, spacing_cv=0.3)
        assert q["status"] == "WARNING"
        assert q["gates"]["non_trivial"] is True
        assert q["gates"]["diverse"] is False
        assert any("Effort" in i and "flat" in i.lower() for i in q["issues"])

    def test_three_objectives_one_flat_names_only_the_dead_axis(self):
        objs = [*self.objectives, Objective(name="Risk", direction="minimize")]
        sols = [
            Solution(solution_id=i, selected_options=[f"O{i}"],
                     objective_values={"Revenue": 10.0 + 4 * i, "Effort": 8.0 + 3 * i, "Risk": 50.0})
            for i in range(4)
        ]
        q = frontier_quality(sols, objs, spacing_cv=0.3)
        assert q["status"] == "WARNING"
        flat_issues = [i for i in q["issues"] if "flat" in i.lower()]
        assert len(flat_issues) == 1
        assert "Risk" in flat_issues[0]
        assert "Revenue" not in flat_issues[0] and "Effort" not in flat_issues[0]

    def test_conflicting_pair_stays_good(self):
        # Regression guard for the diverse gate: a genuine tradeoff (Revenue up, Effort up
        # = worse) with both axes varying must remain GOOD.
        sols = [
            Solution(solution_id=i, selected_options=[f"O{i}"],
                     objective_values={"Revenue": 10.0 + 4 * i, "Effort": 8.0 + 3 * i})
            for i in range(5)
        ]
        q = frontier_quality(sols, self.objectives, spacing_cv=0.3)
        assert q["status"] == "GOOD"
        assert q["issues"] == []


# ─── readiness (validate verdict) ───


class TestReadiness:
    """The readiness classifier over optimizer.validate — which kind of work remains."""

    def _validate(self, p):
        from engine.optimizer import validate
        return validate(p)

    def _scored(self, missing: list[tuple[str, str]] = ()):
        objs = [Objective(name="Rev", direction="maximize"), Objective(name="Eff", direction="minimize")]
        opts = [Option(name=n) for n in ("A", "B", "C")]
        scores = [Score(option=o.name, objective=ob.name, value=1.0)
                  for o in opts for ob in objs if (o.name, ob.name) not in missing]
        return Problem(objectives=objs, options=opts, scores=scores)

    def test_ready(self):
        p = self._scored()
        r = readiness(p, self._validate(p))
        assert r["verdict"] == "ready"
        assert r["next_steps"] == "solve run"
        assert "score_gaps" not in r

    def test_framing_gap_thin_structure(self):
        p = Problem(objectives=[Objective(name="Rev", direction="maximize")],
                    options=[Option(name="A"), Option(name="B"), Option(name="C")])
        r = readiness(p, self._validate(p))
        assert r["verdict"] == "framing_gap"
        assert "problem_framing" in r["next_steps"]

    def test_data_gap_with_per_objective_rollup(self):
        # 'Eff' entirely unscored, 'Rev' one hole — the rollup must distinguish the two.
        missing = [("A", "Eff"), ("B", "Eff"), ("C", "Eff"), ("A", "Rev")]
        p = self._scored(missing=missing)
        r = readiness(p, self._validate(p))
        assert r["verdict"] == "data_gap"
        gaps = {g["objective"]: g for g in r["score_gaps"]}
        assert gaps["Eff"]["missing"] == 3 and gaps["Eff"]["of"] == 3 and gaps["Eff"].get("unscored") is True
        assert gaps["Rev"]["missing"] == 1 and "unscored" not in gaps["Rev"]
        assert "Eff" in r["summary"]  # the fully-dead axis is named, not buried in a count
        assert "data_collection" in r["next_steps"]

    def test_framing_gap_wins_over_data_gap(self):
        # One objective AND missing scores: scores can't settle an unsettled structure.
        p = Problem(objectives=[Objective(name="Rev", direction="maximize")],
                    options=[Option(name="A"), Option(name="B"), Option(name="C")])
        r = readiness(p, self._validate(p))
        assert r["verdict"] == "framing_gap"
        assert r["score_gaps"]  # the data gap still travels alongside

    def test_score_issue_carries_shared_prefix(self):
        """readiness classifies the score-matrix issue by the SCORE_MATRIX_MSG prefix
        both modules import from models — guard against a hand-written replacement
        message bypassing the shared constant."""
        p = self._scored(missing=[("A", "Rev")])
        vr = self._validate(p)
        from engine.models import SCORE_MATRIX_MSG
        assert any(i.message.startswith(SCORE_MATRIX_MSG) for i in vr.issues)

    def test_duplicate_option_names_keep_rollup_denominator_honest(self):
        # Duplicate names are their own framing error; the per-objective rollup must
        # count unique options so a fully-unscored axis still reads unscored.
        objs = [Objective(name="Rev", direction="maximize"), Objective(name="Eff", direction="minimize")]
        opts = [Option(name="A"), Option(name="A"), Option(name="B"), Option(name="C")]
        p = Problem(objectives=objs, options=opts,
                    scores=[Score(option=o, objective="Rev", value=1.0) for o in ("A", "B", "C")])
        r = readiness(p, self._validate(p))
        assert r["verdict"] == "framing_gap"  # the duplicate-name error
        eff = next(g for g in r["score_gaps"] if g["objective"] == "Eff")
        assert eff["of"] == 3 and eff["unscored"] is True


class TestDominatedOptionsCap:
    def test_long_dominated_list_ships_head_plus_total(self):
        """The list is echoed on every structural model update — at portfolio scale
        (205 entries on capital-300) it's capped like missing_scores, with the full
        count preserved."""
        from engine.metrics import _MAX_DOMINATED_RETURNED

        objs = [Objective(name="Revenue", direction="maximize"),
                Objective(name="Effort", direction="minimize")]
        options = [Option(name="A")] + [Option(name=f"B{i}") for i in range(25)]
        scores = [Score(option="A", objective="Revenue", value=10),
                  Score(option="A", objective="Effort", value=1)]
        for i in range(25):
            scores += [Score(option=f"B{i}", objective="Revenue", value=1),
                       Score(option=f"B{i}", objective="Effort", value=9)]
        m = data_metrics(Problem(objectives=objs, options=options, scores=scores))
        assert len(m["dominated_options"]) == _MAX_DOMINATED_RETURNED
        assert m["dominated_options_total"] == 25


class TestSolveResponseElision:
    """Portfolio-scale solve responses ship ranked heads, not per-option floods
    (capital-300 returned ~170 option_never_selected rows and a 300-entry coverage dict)."""

    @staticmethod
    def _big_solved_problem(n_opt=80, n_selected=5):
        objs = [Objective(name="Revenue", direction="maximize"),
                Objective(name="Effort", direction="minimize")]
        opts = [Option(name=f"o{i:03d}") for i in range(n_opt)]
        scores = [Score(option=o.name, objective=ob.name, value=float(i % 9 + 1))
                  for i, o in enumerate(opts) for ob in objs]
        sols = [Solution(solution_id=k, selected_options=[f"o{i:03d}" for i in range(n_selected)],
                         objective_values={"Revenue": 10.0 + k, "Effort": 5.0 - 0.1 * k})
                for k in range(3)]
        p = Problem(objectives=objs, options=opts, scores=scores)
        p.run = Run(solutions=sols)
        return p

    def test_option_coverage_ships_ranked_head_plus_summary(self):
        from engine.metrics import _MAX_COVERAGE_RETURNED, solve_metrics
        m = solve_metrics(self._big_solved_problem())
        assert len(m["option_coverage"]) == _MAX_COVERAGE_RETURNED
        elided = m["option_coverage_elided"]
        assert elided["total_options"] == 80 and elided["shown"] == _MAX_COVERAGE_RETURNED
        # Absence below the head is elision, not zero — the summary must bound it.
        assert elided["tail_max_count"] == 0  # this fixture's tail is all zeros
        assert "NOT zero" in elided["note"]
        # Ranked head: every selected option outranks the zero tail and is present.
        for i in range(5):
            assert m["option_coverage"][f"o{i:03d}"] == 3
        assert "explore composition" in elided["note"]

    def test_small_coverage_passes_whole(self):
        from engine.metrics import solve_metrics
        m = solve_metrics(self._big_solved_problem(n_opt=10))
        assert len(m["option_coverage"]) == 10
        assert "option_coverage_elided" not in m

    def test_never_selected_collapses_to_summary_at_scale(self):
        from engine.metrics import _MAX_NEVER_SELECTED_LISTED, diagnostics
        diags = diagnostics(self._big_solved_problem())
        rows = [d for d in diags if d["pattern"] == "option_never_selected"]
        assert len(rows) == 1
        row = rows[0]
        assert row["count"] == 75 and len(row["options"]) == _MAX_NEVER_SELECTED_LISTED
        assert "explore composition" in row["note"]

    def test_never_selected_stays_per_option_when_few(self):
        from engine.metrics import diagnostics
        diags = diagnostics(self._big_solved_problem(n_opt=8, n_selected=5))
        rows = [d for d in diags if d["pattern"] == "option_never_selected"]
        assert {r["option"] for r in rows} == {"o005", "o006", "o007"}
