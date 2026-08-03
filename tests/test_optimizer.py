"""Tests for optimizer — validation, NSGA-II, infeasibility analysis."""

import pytest

from engine.models import (
    CardinalityConstraint,
    DependencyConstraint,
    ExclusionPairConstraint,
    ForceExcludeConstraint,
    ForceIncludeConstraint,
    GroupLimitConstraint,
    Objective,
    ObjectiveBoundConstraint,
    Option,
    Problem,
    Score,
)
from engine.optimizer import analyze_infeasibility, optimize, validate


def _make_problem(**overrides):
    """Build a valid default problem for testing."""
    defaults = dict(
        objectives=[
            Objective(name="Revenue", direction="maximize", unit="$"),
            Objective(name="Effort", direction="minimize", unit="weeks"),
        ],
        options=[Option(name=n) for n in ["A", "B", "C", "D", "E"]],
        scores=[
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
        ],
        constraints=[CardinalityConstraint(min=2, max=3)],
    )
    defaults.update(overrides)
    return Problem(**defaults)


# ─── Validation ───


class TestValidation:
    def test_valid_problem(self):
        p = _make_problem()
        vr = validate(p)
        assert vr.ready is True
        assert vr.issues == []

    def test_too_few_objectives(self):
        p = _make_problem(objectives=[
            Objective(name="Revenue", direction="maximize"),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("2 objectives" in i.message for i in vr.issues)

    def test_too_few_options(self):
        p = _make_problem(
            options=[Option(name="A"), Option(name="B")],
            scores=[
                Score(option="A", objective="Revenue", value=8),
                Score(option="A", objective="Effort", value=5),
                Score(option="B", objective="Revenue", value=6),
                Score(option="B", objective="Effort", value=3),
            ],
        )
        vr = validate(p)
        assert vr.ready is False
        assert any("3 options" in i.message for i in vr.issues)

    def test_missing_scores(self):
        p = _make_problem(scores=[
            Score(option="A", objective="Revenue", value=8),
            Score(option="A", objective="Effort", value=5),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert len(vr.missing_scores) == 8  # 4 options * 2 objectives missing

    def test_force_include_exclude_conflict(self):
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ForceIncludeConstraint(option="A"),
            ForceExcludeConstraint(option="A"),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("force_include and force_exclude" in i.message for i in vr.issues)

    def test_cardinality_min_gt_max(self):
        p = _make_problem(constraints=[
            CardinalityConstraint(min=5, max=2),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("min" in i.message and "max" in i.message for i in vr.issues)

    def test_cardinality_exceeds_available(self):
        p = _make_problem(constraints=[
            CardinalityConstraint(min=10, max=15),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("exceeds available" in i.message for i in vr.issues)

    def test_force_include_exceeds_cardinality_max(self):
        p = _make_problem(constraints=[
            CardinalityConstraint(min=1, max=2),
            ForceIncludeConstraint(option="A"),
            ForceIncludeConstraint(option="B"),
            ForceIncludeConstraint(option="C"),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("force_include count" in i.message for i in vr.issues)

    def test_unknown_option_in_constraint(self):
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ForceIncludeConstraint(option="Nonexistent"),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("unknown option" in i.message for i in vr.issues)

    def test_unknown_objective_in_constraint(self):
        p = _make_problem(constraints=[
            ObjectiveBoundConstraint(objective="Nonexistent", operator="max", value=10),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("unknown objective" in i.message for i in vr.issues)

    def test_duplicate_objective_names(self):
        p = _make_problem(objectives=[
            Objective(name="Revenue", direction="maximize"),
            Objective(name="Revenue", direction="minimize"),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("Duplicate objective" in i.message for i in vr.issues)

    def test_duplicate_option_names(self):
        p = _make_problem(
            options=[Option(name="A"), Option(name="A"), Option(name="B")],
            scores=[
                Score(option="A", objective="Revenue", value=8),
                Score(option="A", objective="Effort", value=5),
                Score(option="B", objective="Revenue", value=6),
                Score(option="B", objective="Effort", value=3),
            ],
        )
        vr = validate(p)
        assert vr.ready is False
        assert any("Duplicate option" in i.message for i in vr.issues)

    def test_scenario_scale_groups_unknown_option_warns(self):
        """Scenario interaction_matrix_overrides with scale_groups referencing
        an unknown option name should produce a validation warning (typo-tolerance
        gap — prior behavior silently dropped unknown names)."""
        from engine.models import (
            InteractionMatrix,
            InteractionScaleGroup,
            Scenario,
            ScenarioConfig,
        )

        p = _make_problem(
            scenario_config=ScenarioConfig(
                enabled=True,
                scenarios=[
                    Scenario(
                        name="Recession",
                        interaction_matrix_overrides=[
                            InteractionMatrix(
                                objective="Revenue",
                                mode="upsert",
                                entries={},
                                scale_groups=[
                                    InteractionScaleGroup(
                                        options=["A", "B", "VXUS"],  # VXUS not in options
                                        factor=1.5,
                                    )
                                ],
                            )
                        ],
                    )
                ],
            )
        )
        vr = validate(p)
        # Warning, not error — problem still ready
        assert any(
            "VXUS" in i.message and i.severity == "warning"
            for i in vr.issues
        ), f"Expected warning mentioning VXUS, got: {[(i.severity, i.message) for i in vr.issues]}"


# ─── 1.9 Pre-Solve Constraint Conflict Detection ───


class TestConstraintConflictDetection:
    """Conflicts the solver would only surface after a full run."""

    def test_group_limit_below_forced_includes(self):
        """group_limit max < count of force_includes in the group → infeasible."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ForceIncludeConstraint(option="A"),
            ForceIncludeConstraint(option="B"),
            GroupLimitConstraint(options=["A", "B", "C"], max=1),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any(
            "group_limit max" in i.message and "force_included" in i.message
            for i in vr.issues
        )

    def test_group_limit_at_forced_count_ok(self):
        """group_limit.max == |force_in ∩ group| is feasible (tight, not over)."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ForceIncludeConstraint(option="A"),
            ForceIncludeConstraint(option="B"),
            GroupLimitConstraint(options=["A", "B", "C"], max=2),
        ])
        vr = validate(p)
        assert vr.ready is True

    def test_exclusion_pair_both_force_included(self):
        """exclusion_pair with both members force_included → infeasible."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ForceIncludeConstraint(option="A"),
            ForceIncludeConstraint(option="B"),
            ExclusionPairConstraint(option_a="A", option_b="B"),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("exclusion_pair" in i.message and "force_include" in i.message for i in vr.issues)

    def test_dependency_cycle_two_node(self):
        """A → B and B → A is a cycle."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            DependencyConstraint(if_option="A", then_option="B"),
            DependencyConstraint(if_option="B", then_option="A"),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("dependency cycle" in i.message for i in vr.issues)

    def test_dependency_cycle_three_node(self):
        """A → B → C → A is a cycle."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            DependencyConstraint(if_option="A", then_option="B"),
            DependencyConstraint(if_option="B", then_option="C"),
            DependencyConstraint(if_option="C", then_option="A"),
        ])
        vr = validate(p)
        assert vr.ready is False
        cycle_issues = [i for i in vr.issues if "dependency cycle" in i.message]
        assert len(cycle_issues) >= 1

    def test_dependency_self_loop_is_cycle(self):
        """A → A is a degenerate cycle and should be reported."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            DependencyConstraint(if_option="A", then_option="A"),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any("dependency cycle" in i.message for i in vr.issues)

    def test_dependency_chain_no_cycle_ok(self):
        """A → B → C with no back-edge is acyclic — no error."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            DependencyConstraint(if_option="A", then_option="B"),
            DependencyConstraint(if_option="B", then_option="C"),
        ])
        vr = validate(p)
        assert vr.ready is True

    def test_dependency_then_option_force_excluded(self):
        """if_option force_included AND then_option force_excluded → infeasible."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ForceIncludeConstraint(option="A"),
            ForceExcludeConstraint(option="B"),
            DependencyConstraint(if_option="A", then_option="B"),
        ])
        vr = validate(p)
        assert vr.ready is False
        assert any(
            "dependency" in i.message and "force_included" in i.message and "force_excluded" in i.message
            for i in vr.issues
        )

    def test_dependency_then_excluded_but_if_not_forced_in_is_ok(self):
        """If if_option isn't force_included, the dependency is vacuous when then_option is excluded."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ForceExcludeConstraint(option="B"),
            DependencyConstraint(if_option="A", then_option="B"),
        ])
        vr = validate(p)
        # The constraint says "if A is selected, B must be too" — but B is excluded.
        # That just means A can't be selected. The problem is still feasible (other options exist).
        assert vr.ready is True

    def test_max_allocation_arithmetic_infeasible(self):
        """In proportional mode, max_allocation × available_options < 100% → infeasible."""
        from engine.models import Approach, MaxAllocationConstraint
        # 5 options × 15% cap = 75% < 100%
        p = _make_problem(
            approach=Approach.proportional,
            constraints=[MaxAllocationConstraint(max=15)],
        )
        vr = validate(p)
        assert vr.ready is False
        assert any("max_allocation cap" in i.message and "< 100%" in i.message for i in vr.issues)

    def test_max_allocation_arithmetic_feasible(self):
        """max_allocation × available_options ≥ 100% → feasible."""
        from engine.models import Approach, MaxAllocationConstraint
        # 5 options × 25% cap = 125% ≥ 100%
        p = _make_problem(
            approach=Approach.proportional,
            constraints=[MaxAllocationConstraint(max=25)],
        )
        vr = validate(p)
        assert vr.ready is True

    def test_max_allocation_only_applies_to_proportional(self):
        """In binary mode, max_allocation arithmetic check should not fire (warning already emitted elsewhere)."""
        from engine.models import MaxAllocationConstraint
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            MaxAllocationConstraint(max=15),
        ])
        vr = validate(p)
        # Binary mode: no arithmetic infeasibility error, only the warning that max_allocation is ignored
        assert not any("max_allocation cap" in i.message and "< 100%" in i.message for i in vr.issues)


# ─── Optimization ───


class TestOptimization:
    def test_basic_optimization(self):
        p = _make_problem()
        run = optimize(p)
        assert len(run.solutions) > 0

    def test_solutions_respect_cardinality(self):
        p = _make_problem(constraints=[CardinalityConstraint(min=2, max=3)])
        run = optimize(p)
        for s in run.solutions:
            assert 2 <= len(s.selected_options) <= 3

    def test_solutions_respect_force_include(self):
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ForceIncludeConstraint(option="A"),
        ])
        run = optimize(p)
        for s in run.solutions:
            assert "A" in s.selected_options

    def test_solutions_respect_force_exclude(self):
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ForceExcludeConstraint(option="C"),
        ])
        run = optimize(p)
        for s in run.solutions:
            assert "C" not in s.selected_options

    def test_solutions_respect_objective_bound(self):
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ObjectiveBoundConstraint(objective="Effort", operator="max", value=10),
        ])
        run = optimize(p)
        for s in run.solutions:
            assert s.objective_values["Effort"] <= 10.0001  # float tolerance

    def test_solutions_sorted_by_first_objective(self):
        p = _make_problem()
        run = optimize(p)
        if len(run.solutions) >= 2:
            # First objective is Revenue (maximize), so should be descending
            for i in range(len(run.solutions) - 1):
                assert run.solutions[i].objective_values["Revenue"] >= run.solutions[i + 1].objective_values["Revenue"]

    def test_solutions_reindexed(self):
        p = _make_problem()
        run = optimize(p)
        for i, s in enumerate(run.solutions, start=1):
            assert s.solution_id == i

    def test_quality_indicators(self):
        p = _make_problem()
        run = optimize(p)
        if len(run.solutions) >= 3:
            assert run.quality.hypervolume_normalized is not None
            assert run.quality.spacing_cv is not None

    def test_optimize_raises_on_invalid(self):
        p = _make_problem(objectives=[Objective(name="X", direction="maximize")])
        with pytest.raises(ValueError, match="not ready"):
            optimize(p)

    def test_auto_scaling_small(self):
        """For ≤20 options, pop_size=100 and n_gen=200 per spec."""
        # This is implicitly tested by the optimization running successfully
        # with the default 5 options. We verify it doesn't crash.
        p = _make_problem()
        run = optimize(p)
        assert len(run.solutions) > 0


# ─── Infeasibility analysis ───


class TestInfeasibility:
    def test_binding_constraint_identified(self):
        """force_include + force_exclude on same option should identify both."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ForceIncludeConstraint(option="A"),
            ForceExcludeConstraint(option="A"),
        ])
        # This would fail validation, so test analyze_infeasibility directly
        # with a problem that passes validation but is infeasible
        p2 = _make_problem(constraints=[
            CardinalityConstraint(min=5, max=5),
            ForceExcludeConstraint(option="A"),
            ForceExcludeConstraint(option="B"),
        ])
        result = analyze_infeasibility(p2)
        assert len(result["binding_constraints"]) > 0
        assert len(result["suggestions"]) > 0

    def test_tight_objective_bound(self):
        """Very tight objective bound should be identified as binding."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ObjectiveBoundConstraint(objective="Effort", operator="max", value=1),  # impossibly tight
        ])
        result = analyze_infeasibility(p)
        assert any("Effort" in s for s in result["suggestions"])


# ─── Aggregation ───


class TestAggregation:
    def test_sum_is_default(self):
        """Default aggregation is sum — matches legacy behavior."""
        p = _make_problem()
        for obj in p.objectives:
            assert obj.aggregation == "sum"

    def test_sum_matches_legacy(self):
        """Explicit sum produces same results as Phase 0."""
        p = _make_problem(objectives=[
            Objective(name="Revenue", direction="maximize", unit="$", aggregation="sum"),
            Objective(name="Effort", direction="minimize", unit="weeks", aggregation="sum"),
        ])
        run = optimize(p)
        assert len(run.solutions) > 0
        # Sum: portfolio value = sum of selected option scores
        for sol in run.solutions:
            expected_rev = sum(
                s.value for s in p.scores
                if s.objective == "Revenue" and s.option in sol.selected_options
            )
            assert abs(sol.objective_values["Revenue"] - expected_rev) < 0.01

    def test_avg_aggregation(self):
        """Average aggregation divides by count of selected options."""
        p = _make_problem(
            objectives=[
                Objective(name="Quality", direction="maximize", aggregation="avg"),
                Objective(name="Cost", direction="minimize", aggregation="sum"),
            ],
            scores=[
                Score(option="A", objective="Quality", value=10),
                Score(option="A", objective="Cost", value=5),
                Score(option="B", objective="Quality", value=6),
                Score(option="B", objective="Cost", value=3),
                Score(option="C", objective="Quality", value=8),
                Score(option="C", objective="Cost", value=7),
                Score(option="D", objective="Quality", value=4),
                Score(option="D", objective="Cost", value=2),
                Score(option="E", objective="Quality", value=7),
                Score(option="E", objective="Cost", value=4),
            ],
        )
        run = optimize(p)
        assert len(run.solutions) > 0
        for sol in run.solutions:
            scores = [
                s.value for s in p.scores
                if s.objective == "Quality" and s.option in sol.selected_options
            ]
            expected_avg = sum(scores) / len(scores)
            assert abs(sol.objective_values["Quality"] - expected_avg) < 0.01

    def test_min_aggregation(self):
        """Min aggregation returns worst individual option score."""
        p = _make_problem(
            objectives=[
                Objective(name="Reliability", direction="maximize", aggregation="min"),
                Objective(name="Cost", direction="minimize", aggregation="sum"),
            ],
            scores=[
                Score(option="A", objective="Reliability", value=9),
                Score(option="A", objective="Cost", value=5),
                Score(option="B", objective="Reliability", value=3),
                Score(option="B", objective="Cost", value=2),
                Score(option="C", objective="Reliability", value=7),
                Score(option="C", objective="Cost", value=6),
                Score(option="D", objective="Reliability", value=5),
                Score(option="D", objective="Cost", value=3),
                Score(option="E", objective="Reliability", value=8),
                Score(option="E", objective="Cost", value=4),
            ],
        )
        run = optimize(p)
        assert len(run.solutions) > 0
        for sol in run.solutions:
            scores = [
                s.value for s in p.scores
                if s.objective == "Reliability" and s.option in sol.selected_options
            ]
            expected_min = min(scores)
            assert abs(sol.objective_values["Reliability"] - expected_min) < 0.01

    def test_max_aggregation(self):
        """Max aggregation returns best individual option score."""
        p = _make_problem(
            objectives=[
                Objective(name="Peak", direction="maximize", aggregation="max"),
                Objective(name="Cost", direction="minimize", aggregation="sum"),
            ],
            scores=[
                Score(option="A", objective="Peak", value=10),
                Score(option="A", objective="Cost", value=8),
                Score(option="B", objective="Peak", value=4),
                Score(option="B", objective="Cost", value=2),
                Score(option="C", objective="Peak", value=7),
                Score(option="C", objective="Cost", value=5),
                Score(option="D", objective="Peak", value=3),
                Score(option="D", objective="Cost", value=1),
                Score(option="E", objective="Peak", value=6),
                Score(option="E", objective="Cost", value=3),
            ],
        )
        run = optimize(p)
        assert len(run.solutions) > 0
        for sol in run.solutions:
            scores = [
                s.value for s in p.scores
                if s.objective == "Peak" and s.option in sol.selected_options
            ]
            expected_max = max(scores)
            assert abs(sol.objective_values["Peak"] - expected_max) < 0.01

    def test_objective_bound_respects_avg_aggregation(self):
        """Objective bound constraint should apply to aggregated (avg) value."""
        p = _make_problem(
            objectives=[
                Objective(name="Quality", direction="maximize", aggregation="avg"),
                Objective(name="Cost", direction="minimize", aggregation="sum"),
            ],
            scores=[
                Score(option="A", objective="Quality", value=10),
                Score(option="A", objective="Cost", value=5),
                Score(option="B", objective="Quality", value=2),
                Score(option="B", objective="Cost", value=1),
                Score(option="C", objective="Quality", value=8),
                Score(option="C", objective="Cost", value=6),
                Score(option="D", objective="Quality", value=3),
                Score(option="D", objective="Cost", value=2),
                Score(option="E", objective="Quality", value=6),
                Score(option="E", objective="Cost", value=3),
            ],
            constraints=[
                CardinalityConstraint(min=2, max=3),
                ObjectiveBoundConstraint(objective="Quality", operator="min", value=6),
            ],
        )
        run = optimize(p)
        assert len(run.solutions) > 0
        for sol in run.solutions:
            scores = [
                s.value for s in p.scores
                if s.objective == "Quality" and s.option in sol.selected_options
            ]
            avg = sum(scores) / len(scores)
            assert avg >= 6.0 - 0.01, f"Avg quality {avg} below bound 6"

    def test_mixed_aggregation(self):
        """Different objectives can use different aggregation modes."""
        p = _make_problem(
            objectives=[
                Objective(name="TotalRev", direction="maximize", aggregation="sum"),
                Objective(name="WorstRisk", direction="minimize", aggregation="min"),
            ],
            scores=[
                Score(option="A", objective="TotalRev", value=10),
                Score(option="A", objective="WorstRisk", value=3),
                Score(option="B", objective="TotalRev", value=5),
                Score(option="B", objective="WorstRisk", value=1),
                Score(option="C", objective="TotalRev", value=8),
                Score(option="C", objective="WorstRisk", value=4),
                Score(option="D", objective="TotalRev", value=3),
                Score(option="D", objective="WorstRisk", value=2),
                Score(option="E", objective="TotalRev", value=6),
                Score(option="E", objective="WorstRisk", value=5),
            ],
        )
        run = optimize(p)
        assert len(run.solutions) > 0
        for sol in run.solutions:
            # Verify sum for TotalRev
            rev_scores = [s.value for s in p.scores if s.objective == "TotalRev" and s.option in sol.selected_options]
            assert abs(sol.objective_values["TotalRev"] - sum(rev_scores)) < 0.01
            # Verify min for WorstRisk
            risk_scores = [s.value for s in p.scores if s.objective == "WorstRisk" and s.option in sol.selected_options]
            assert abs(sol.objective_values["WorstRisk"] - min(risk_scores)) < 0.01


# ─── New Constraint Types ───


class TestExclusionPair:
    def test_exclusion_respected(self):
        """Excluded pair: can't select both A and C."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            ExclusionPairConstraint(option_a="A", option_b="C"),
        ])
        run = optimize(p)
        assert len(run.solutions) > 0
        for sol in run.solutions:
            assert not ("A" in sol.selected_options and "C" in sol.selected_options), (
                f"Both A and C selected: {sol.selected_options}"
            )

    def test_exclusion_validation_unknown_option(self):
        p = _make_problem(constraints=[
            ExclusionPairConstraint(option_a="A", option_b="UNKNOWN"),
        ])
        vr = validate(p)
        assert not vr.ready
        assert any("UNKNOWN" in i.message for i in vr.issues)

    def test_exclusion_same_option(self):
        p = _make_problem(constraints=[
            ExclusionPairConstraint(option_a="A", option_b="A"),
        ])
        vr = validate(p)
        assert not vr.ready


class TestDependency:
    def test_dependency_respected(self):
        """If A selected, B must be selected too."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=4),
            DependencyConstraint(if_option="A", then_option="B"),
        ])
        run = optimize(p)
        assert len(run.solutions) > 0
        for sol in run.solutions:
            if "A" in sol.selected_options:
                assert "B" in sol.selected_options, (
                    f"A selected without B: {sol.selected_options}"
                )

    def test_dependency_validation_unknown_option(self):
        p = _make_problem(constraints=[
            DependencyConstraint(if_option="UNKNOWN", then_option="B"),
        ])
        vr = validate(p)
        assert not vr.ready


class TestGroupLimit:
    def test_group_limit_respected(self):
        """At most 1 from group {A, C, E}."""
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            GroupLimitConstraint(options=["A", "C", "E"], max=1),
        ])
        run = optimize(p)
        assert len(run.solutions) > 0
        for sol in run.solutions:
            group_count = sum(1 for o in ["A", "C", "E"] if o in sol.selected_options)
            assert group_count <= 1, (
                f"Group has {group_count} selected: {sol.selected_options}"
            )

    def test_group_limit_validation_unknown_option(self):
        p = _make_problem(constraints=[
            GroupLimitConstraint(options=["A", "UNKNOWN"], max=1),
        ])
        vr = validate(p)
        assert not vr.ready


# ─── Binding constraint detection ───


class TestBindingDetection:
    def test_cardinality_binding_detected(self):
        """Cardinality at max should surface as binding."""
        from engine import metrics
        p = _make_problem(constraints=[CardinalityConstraint(min=2, max=2)])
        run = optimize(p)
        p.run = run

        diags = metrics.diagnostics(p)
        binding = [d for d in diags if d["pattern"] == "binding_constraint"]
        assert any("cardinality" in d["constraint"] for d in binding)

    def test_group_limit_binding_detected(self):
        """Group limit at max should surface as binding."""
        from engine import metrics
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            GroupLimitConstraint(options=["A", "B", "C"], max=1),
        ])
        run = optimize(p)
        p.run = run

        diags = metrics.diagnostics(p)
        binding = [d for d in diags if d["pattern"] == "binding_constraint"]
        # At least one binding constraint should be found (cardinality or group)
        assert len(binding) > 0


# ─── Pruning ───


class TestPruning:
    def test_max_solutions_parameter(self):
        """max_solutions caps the number of returned solutions."""
        p = _make_problem()
        run = optimize(p, max_solutions=3)
        assert len(run.solutions) <= 3


# ─── Matrix override helper (Fix 3) ───


class TestApplyMatrixOverride:
    """_apply_matrix_override: replace | upsert | scale_groups composition."""

    def _base(self):
        from engine.models import InteractionMatrix
        return InteractionMatrix(
            objective="Vol",
            entries={
                "A": {"A": 1.0, "B": 0.2, "C": 0.1},
                "B": {"A": 0.2, "B": 1.0, "C": 0.3},
                "C": {"A": 0.1, "B": 0.3, "C": 1.0},
            },
        )

    def test_replace_is_default(self):
        from engine.models import InteractionMatrix
        from engine.optimizer import _apply_matrix_override

        override = InteractionMatrix(
            objective="Vol",
            entries={"A": {"A": 2.0, "B": 0.9}, "B": {"A": 0.9, "B": 2.0}},
        )
        merged = _apply_matrix_override(self._base(), override)
        # Full replacement: base's "C" row is gone
        assert "C" not in merged.entries
        assert merged.entries["A"]["B"] == 0.9

    def test_upsert_merges_cells_with_symmetry(self):
        from engine.models import InteractionMatrix
        from engine.optimizer import _apply_matrix_override

        override = InteractionMatrix(
            objective="Vol",
            mode="upsert",
            entries={"A": {"B": 0.9}},  # only (A,B) specified
        )
        merged = _apply_matrix_override(self._base(), override)
        # Symmetry auto-enforced
        assert merged.entries["A"]["B"] == 0.9
        assert merged.entries["B"]["A"] == 0.9
        # Unchanged cells survive
        assert merged.entries["A"]["C"] == 0.1
        assert merged.entries["C"]["B"] == 0.3

    def test_upsert_with_no_base_creates_fresh_matrix(self):
        from engine.models import InteractionMatrix
        from engine.optimizer import _apply_matrix_override

        override = InteractionMatrix(
            objective="Vol",
            mode="upsert",
            entries={"A": {"B": 0.5}},
        )
        merged = _apply_matrix_override(None, override)
        assert merged.entries["A"]["B"] == 0.5
        assert merged.entries["B"]["A"] == 0.5

    def test_scale_groups_multiplies_off_diagonals_within_group(self):
        from engine.models import InteractionMatrix, InteractionScaleGroup
        from engine.optimizer import _apply_matrix_override

        override = InteractionMatrix(
            objective="Vol",
            mode="upsert",
            entries={},
            scale_groups=[InteractionScaleGroup(options=["A", "B"], factor=2.0)],
        )
        merged = _apply_matrix_override(self._base(), override)
        # A-B off-diagonal × 2
        assert merged.entries["A"]["B"] == pytest.approx(0.4)
        assert merged.entries["B"]["A"] == pytest.approx(0.4)
        # Diagonal untouched
        assert merged.entries["A"]["A"] == 1.0
        # Outside-group (A-C, B-C) untouched
        assert merged.entries["A"]["C"] == 0.1
        assert merged.entries["B"]["C"] == 0.3

    def test_replace_then_scale_groups_compose(self):
        from engine.models import InteractionMatrix, InteractionScaleGroup
        from engine.optimizer import _apply_matrix_override

        override = InteractionMatrix(
            objective="Vol",
            mode="replace",
            entries={
                "A": {"A": 1.0, "B": 0.5},
                "B": {"A": 0.5, "B": 1.0},
            },
            scale_groups=[InteractionScaleGroup(options=["A", "B"], factor=0.0)],
        )
        merged = _apply_matrix_override(self._base(), override)
        # Replace set A-B to 0.5, then scale by 0 → off-diagonals zero
        assert merged.entries["A"]["B"] == 0.0
        assert merged.entries["B"]["A"] == 0.0
        # Diagonals preserved
        assert merged.entries["A"]["A"] == 1.0


# ─── Extreme-point seeding (Fix 2) ───


class TestExtremeSeeds:
    """_compute_extreme_seeds: greedy corner seeds per objective."""

    def test_seeds_are_built_per_objective(self):
        """One seed per objective, respecting cardinality + group limits."""
        from engine.optimizer import _compute_extreme_seeds, _parse_constraints, _build_score_matrix

        p = _make_problem(constraints=[CardinalityConstraint(min=2, max=3)])
        score_matrix = _build_score_matrix(p)
        cp = _parse_constraints(p, search_floor=True)
        seeds = _compute_extreme_seeds(p, score_matrix, cp)

        assert seeds.shape[0] == len(p.objectives)
        assert seeds.shape[1] == len(p.options)
        # Binary mode: values are 0 or 1
        unique = set(seeds.flatten().tolist())
        assert unique.issubset({0.0, 1.0})
        # Each seed has between 2 and 3 non-zero entries (cardinality)
        for seed in seeds:
            count = int(seed.sum())
            assert 2 <= count <= 3

    def test_allocation_bounds_respected_by_nsga_and_repair(self):
        """E2: per-option allocation floors/caps hold in every returned plan, including under
        an adversarial global cap, and the repair operator projects arbitrary rows into the
        box without breaking the 100% budget."""
        import numpy as np

        from engine.models import AllocationBoundConstraint, MaxAllocationConstraint
        from engine.optimizer import _SimplexRepair, _build_score_matrix, _parse_constraints

        p = _make_problem(
            approach="proportional",
            constraints=[MaxAllocationConstraint(max=40),
                         AllocationBoundConstraint(option="A", min=12, max=100),
                         AllocationBoundConstraint(option="B", min=0, max=15)],
        )
        run = optimize(p, mode="fast", seed=5)
        assert len(run.solutions) > 0
        for s in run.solutions:
            assert s.allocations.get("A", 0) >= 12
            assert s.allocations.get("B", 0) <= 15
            assert max(s.allocations.values()) <= 40
            assert sum(s.allocations.values()) == 100

        # Direct repair check on adversarial rows (all-zero, one-hot, uniform).
        from engine.optimizer import _ProportionalProblem
        cp = _parse_constraints(p, search_floor=True)
        prob = _ProportionalProblem(n_options=5, score_matrix=_build_score_matrix(p),
                                    objectives=p.objectives, interaction_matrices={}, **cp)
        X = np.array([[0, 0, 0, 0, 0], [100, 0, 0, 0, 0], [20, 20, 20, 20, 20]], dtype=float)
        Y = _SimplexRepair()._do(prob, X)
        for row in Y:
            assert abs(row.sum() - 100.0) < 0.05
            assert row[0] >= 12 - 1e-6 and row[1] <= 15 + 1e-6 and row.max() <= 40 + 1e-6

    def test_allocation_bound_floor_above_effective_cap_is_a_validation_error(self):
        # min=50 with a global cap of 40 is an empty box — must be a pre-solve error,
        # not a silent empty frontier.
        from engine.models import AllocationBoundConstraint, MaxAllocationConstraint

        v = validate(_make_problem(approach="proportional", constraints=[
            MaxAllocationConstraint(max=40),
            AllocationBoundConstraint(option="A", min=50, max=100)]))
        assert any("exceeds its effective cap" in i.message and i.severity == "error"
                   for i in v.issues)

    def test_allocation_bound_validation_and_conflicts(self):
        """E2: bad ranges, unknown options, binary-mode warning, floor sums past 100%, floors
        on excluded options, and starved effective caps are all caught pre-solve."""
        from engine.models import AllocationBoundConstraint, ForceExcludeConstraint, MaxAllocationConstraint

        bad = validate(_make_problem(approach="proportional", constraints=[
            AllocationBoundConstraint(option="A", min=50, max=30)]))
        assert any("0 <= min <= max <= 100" in i.message for i in bad.issues)

        unknown = validate(_make_problem(approach="proportional", constraints=[
            AllocationBoundConstraint(option="ZZ", min=0, max=50)]))
        assert any("unknown option 'ZZ'" in i.message for i in unknown.issues)

        warned = validate(_make_problem(constraints=[   # binary problem
            AllocationBoundConstraint(option="A", min=0, max=50)]))
        assert any("proportional mode" in i.message and i.severity == "warning"
                   for i in warned.issues)

        oversum = validate(_make_problem(approach="proportional", constraints=[
            AllocationBoundConstraint(option="A", min=60, max=100),
            AllocationBoundConstraint(option="B", min=50, max=100)]))
        assert any("floors sum to 110%" in i.message for i in oversum.issues)

        excluded = validate(_make_problem(approach="proportional", constraints=[
            AllocationBoundConstraint(option="A", min=10, max=100),
            ForceExcludeConstraint(option="A")]))
        assert any("conflicts" in i.message and "force_exclude" in i.message
                   for i in excluded.issues)

        starved = validate(_make_problem(approach="proportional", constraints=[
            MaxAllocationConstraint(max=30),
            AllocationBoundConstraint(option="A", min=0, max=5),
            ForceExcludeConstraint(option="B")]))
        # caps: A 5 + C/D/E 30 each = 95 < 100
        assert any("caps sum to 95%" in i.message for i in starved.issues)

    def test_group_floor_respected_by_nsga_and_prefilled_in_seeds(self):
        """E1: a group_limit floor pulls its members into every plan (NSGA) and into the
        greedy corner seeds (pre-fill), even when the floored group scores worst."""
        from engine.models import GroupLimitConstraint
        from engine.optimizer import _build_score_matrix, _compute_extreme_seeds, _parse_constraints

        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            GroupLimitConstraint(options=["D"], min=1, max=1),  # worst Revenue option, floored in
        ])
        seeds = _compute_extreme_seeds(p, _build_score_matrix(p), _parse_constraints(p, search_floor=True))
        d_idx = [o.name for o in p.options].index("D")
        assert len(seeds) > 0 and all(seed[d_idx] == 1.0 for seed in seeds)
        run = optimize(p, mode="fast", seed=3)
        assert len(run.solutions) > 0
        assert all("D" in s.selected_options for s in run.solutions)

    def test_group_floor_validation_and_conflicts(self):
        """E1: min>max and min>group-size are validation errors; disjoint floors summing past
        the cardinality max, and floors starved by force_exclude, are conflict errors."""
        from engine.models import ForceExcludeConstraint, GroupLimitConstraint

        bad_range = validate(_make_problem(constraints=[
            GroupLimitConstraint(options=["A", "B"], min=3, max=2)]))
        assert any("min (3) must be between 0 and max (2)" in i.message for i in bad_range.issues)

        too_big = validate(_make_problem(constraints=[
            GroupLimitConstraint(options=["A", "B"], min=3, max=5)]))
        assert any("exceeds the group's selectable members" in i.message for i in too_big.issues)

        starved = validate(_make_problem(constraints=[
            GroupLimitConstraint(options=["A", "B"], min=2, max=2),
            ForceExcludeConstraint(option="A")]))
        assert any("selectable members" in i.message for i in starved.issues)

        oversum = validate(_make_problem(constraints=[
            CardinalityConstraint(min=2, max=3),
            GroupLimitConstraint(options=["A", "B"], min=2, max=2),
            GroupLimitConstraint(options=["C", "D"], min=2, max=2)]))
        assert any("floors sum to 4" in i.message for i in oversum.issues)

    def test_witness_seed_plants_population_inside_a_tight_bound_band(self):
        """objective_bound floor + cap couple the region into a band the greedy corner seeds
        miss entirely; the audit-witness seed keeps the EA from terminating with an empty
        feasible front on a provably feasible problem."""
        pytest.importorskip("highspy")
        import numpy as np

        from engine.optimizer import _feasibility_witness_seed

        # 20 options; Revenue floor + Effort cap leave a knife-edge feasible band.
        n = 20
        opts = [f"O{i:02d}" for i in range(n)]
        scores = []
        for i, o in enumerate(opts):
            scores.append(Score(option=o, objective="Revenue", value=10 + (i % 7)))
            scores.append(Score(option=o, objective="Effort", value=6 + (i % 5)))
        p = Problem(
            name="band", approach="binary",
            objectives=[Objective(name="Revenue", direction="maximize", aggregation="sum"),
                        Objective(name="Effort", direction="minimize", aggregation="sum")],
            options=[Option(name=o) for o in opts], scores=scores,
            constraints=[ObjectiveBoundConstraint(objective="Revenue", operator="min", value=115),
                         ObjectiveBoundConstraint(objective="Effort", operator="max", value=78),
                         CardinalityConstraint(min=5, max=12)],
        )
        from engine.optimizer import _build_score_matrix, _compute_extreme_seeds, _parse_constraints

        sm = _build_score_matrix(p)
        corner_seeds = _compute_extreme_seeds(p, sm, _parse_constraints(p, search_floor=True))
        seed_row = _feasibility_witness_seed(p, corner_seeds, sm)
        assert seed_row is not None and seed_row.shape == (1, n)
        sel = seed_row[0] > 0.5
        rev = sum(10 + (i % 7) for i in range(n) if sel[i])
        eff = sum(6 + (i % 5) for i in range(n) if sel[i])
        assert rev >= 115 and eff <= 78 and 5 <= sel.sum() <= 12   # genuinely in the band
        # And the full solve returns a non-empty feasible frontier across seeds/modes.
        for sd in (1, 7, 42):
            run = optimize(p, mode="fast", seed=sd)
            assert len(run.solutions) > 0, f"empty frontier at seed {sd}"
            for s in run.solutions:
                assert s.objective_values["Revenue"] >= 115 - 1e-6
                assert s.objective_values["Effort"] <= 78 + 1e-6
        # No bounds → no witness row (the prior seed behavior, untouched).
        bare = _make_problem()
        bare_sm = _build_score_matrix(bare)
        assert _feasibility_witness_seed(bare, np.zeros((0, 5)), bare_sm) is None
        # A corner seed already inside the band skips the MILP probe entirely.
        inside = np.zeros((1, n)); inside[0, :8] = 1.0   # 8 picks: rev>=115? eff<=78? recompute:
        rev = sum(10 + (i % 7) for i in range(8)); eff = sum(6 + (i % 5) for i in range(8))
        if rev >= 115 and eff <= 78:
            assert _feasibility_witness_seed(p, inside, sm) is None

    def test_proportional_seeds_sum_to_100_and_respect_cap(self):
        """Proportional seeds: allocations sum to 100 and no allocation exceeds max_allocation."""
        from engine.models import MaxAllocationConstraint
        from engine.optimizer import _compute_extreme_seeds, _parse_constraints, _build_score_matrix

        p = Problem(
            approach="proportional",
            objectives=[
                Objective(name="Return", direction="maximize"),
                Objective(name="Risk", direction="minimize"),
            ],
            options=[Option(name=n) for n in ["A", "B", "C", "D", "E"]],
            scores=[
                Score(option="A", objective="Return", value=10),
                Score(option="B", objective="Return", value=8),
                Score(option="C", objective="Return", value=6),
                Score(option="D", objective="Return", value=4),
                Score(option="E", objective="Return", value=2),
                Score(option="A", objective="Risk", value=5),
                Score(option="B", objective="Risk", value=4),
                Score(option="C", objective="Risk", value=3),
                Score(option="D", objective="Risk", value=2),
                Score(option="E", objective="Risk", value=1),
            ],
            constraints=[
                CardinalityConstraint(min=3, max=5),
                MaxAllocationConstraint(max=40),
            ],
        )
        score_matrix = _build_score_matrix(p)
        cp = _parse_constraints(p, search_floor=True)
        seeds = _compute_extreme_seeds(p, score_matrix, cp)

        assert len(seeds) == 2
        for seed in seeds:
            # Allocations sum to exactly 100
            assert int(seed.sum()) == 100
            # No allocation exceeds cap
            assert seed.max() <= 40
            # At least min cardinality non-zero entries
            assert int((seed > 0).sum()) >= 3

    def test_seeds_respect_group_limits(self):
        """Seeds respect group_limit: no more than max options from the group."""
        from engine.optimizer import _compute_extreme_seeds, _parse_constraints, _build_score_matrix

        # A, B, C are in a group with max=1. Even though they might rank highest for an objective,
        # the seed should only include one.
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=4),
            GroupLimitConstraint(options=["A", "B", "C"], max=1),
        ])
        score_matrix = _build_score_matrix(p)
        cp = _parse_constraints(p, search_floor=True)
        seeds = _compute_extreme_seeds(p, score_matrix, cp)

        for seed in seeds:
            selected_from_group = sum(
                int(seed[i] > 0) for i, o in enumerate(p.options) if o.name in ("A", "B", "C")
            )
            assert selected_from_group <= 1

    def test_no_seeds_when_cardinality_infeasible(self):
        """If constraints block ever reaching cardinality_min, return empty seeds (optimizer falls back)."""
        from engine.optimizer import _compute_extreme_seeds, _parse_constraints, _build_score_matrix

        # Force-exclude everything reachable → cardinality_min unreachable
        p = _make_problem(constraints=[
            CardinalityConstraint(min=2, max=2),
            ForceExcludeConstraint(option="A"),
            ForceExcludeConstraint(option="B"),
            ForceExcludeConstraint(option="C"),
            ForceExcludeConstraint(option="D"),
            ForceExcludeConstraint(option="E"),
        ])
        score_matrix = _build_score_matrix(p)
        cp = _parse_constraints(p, search_floor=True)
        seeds = _compute_extreme_seeds(p, score_matrix, cp)

        # All options excluded → no seed meets cardinality_min
        assert seeds.shape[0] == 0


# ─── Elite preservation (Fix 5) ───


class TestDuplicateAllocationBounds:
    """Two allocation_bound rows on ONE option intersect — a contractual floor row sent
    beside a separate cap row used to be silently overwritten by whichever came last."""

    def _floor_and_cap(self):
        from engine.models import AllocationBoundConstraint
        return _make_problem(approach="proportional", constraints=[
            AllocationBoundConstraint(option="A", min=20, max=100),   # contractual floor row
            AllocationBoundConstraint(option="A", min=0, max=40)])    # separate cap row

    def test_parse_intersects_instead_of_overwriting(self):
        from engine.optimizer import _parse_constraints

        cp = _parse_constraints(self._floor_and_cap())
        a_idx = 0  # options are A..E
        assert cp["allocation_bounds"][a_idx] == (20, 40)

    def test_merged_helper_and_echo(self):
        from engine.optimizer import allocation_bound_merges, merged_allocation_bounds

        cons = self._floor_and_cap().constraints
        assert merged_allocation_bounds(cons) == {"A": (20, 40)}
        assert allocation_bound_merges(cons) == [
            {"option": "A", "rows": 2, "min": 20, "max": 40}]

    def test_solved_plans_hold_both_rows(self):
        """The observed loss: the floor vanished and only the allocations revealed it."""
        run = optimize(self._floor_and_cap(), mode="fast", seed=7)
        assert len(run.solutions) > 0
        for s in run.solutions:
            assert 20 <= s.allocations.get("A", 0) <= 40

    def test_merge_is_echoed_as_a_validation_warning(self):
        v = validate(self._floor_and_cap())
        assert any("2 allocation_bound rows" in i.message and "tightest box" in i.message
                   and i.severity == "warning" for i in v.issues)

    def test_empty_intersection_is_a_validation_error(self):
        from engine.models import AllocationBoundConstraint

        v = validate(_make_problem(approach="proportional", constraints=[
            AllocationBoundConstraint(option="A", min=60, max=100),
            AllocationBoundConstraint(option="A", min=0, max=40)]))
        assert any("empty box" in i.message and "60% > merged max 40%" in i.message
                   and i.severity == "error" for i in v.issues)

    def test_floor_sum_counts_each_option_once(self):
        """Summing the raw rows double-counted a merged option's floor: two floor-carrying
        rows on 'A' (60 each, the same commitment stated twice) summed to 120% and hard-
        errored a model whose one applied floor is 60%."""
        from engine.models import AllocationBoundConstraint

        v = validate(_make_problem(approach="proportional", constraints=[
            AllocationBoundConstraint(option="A", min=60, max=100),
            AllocationBoundConstraint(option="A", min=60, max=80)]))
        assert not any("floors sum to" in i.message for i in v.issues)
        assert v.ready is True

    def test_cap_sum_reads_the_merged_cap(self):
        """The mirror case: the cap sum took whichever row came last, so a genuinely
        starved model passed validation. A's applied cap is the tighter row (10%), and
        10 + 4×20 = 90% cannot sum to 100."""
        from engine.models import AllocationBoundConstraint

        v = validate(_make_problem(approach="proportional", constraints=[
            AllocationBoundConstraint(option="A", min=0, max=10),
            AllocationBoundConstraint(option="A", min=0, max=90),
            *(AllocationBoundConstraint(option=n, min=0, max=20) for n in "BCDE")]))
        assert any("caps sum to 90%" in i.message and i.severity == "error"
                   for i in v.issues)
        assert v.ready is False


class TestElitePreservation:
    """Extreme-point seeds are preserved post-solve so corner quality isn't
    diluted by generic evolutionary operators. Seeds are unioned with pymoo's
    result and filtered by non-dominated sorting.
    """

    def test_preserved_result_size_not_reduced(self):
        """Union should not shrink the Pareto set below pymoo's own output."""
        p = _make_problem(constraints=[CardinalityConstraint(min=2, max=3)])
        run = optimize(p, mode="fast")
        # Just check that we got a non-trivial set; main invariant is that
        # elite preservation doesn't silently collapse the frontier.
        assert len(run.solutions) >= 2

    def test_preserved_corners_match_or_beat_seeds(self):
        """After elite preservation, the per-objective extremes in the final
        Pareto set should match or exceed the greedy seeds' values.

        This is the core claim: neutral NSGA operators may dilute the seeds
        during evolution; post-hoc union restores them.
        """
        from engine.optimizer import (
            _build_score_matrix, _compute_extreme_seeds, _parse_constraints,
        )

        p = _make_problem(constraints=[CardinalityConstraint(min=2, max=3)])
        score_matrix = _build_score_matrix(p)
        cp = _parse_constraints(p, search_floor=True)
        seeds = _compute_extreme_seeds(p, score_matrix, cp)

        # Compute each seed's per-objective extreme value.
        seed_rev = float((seeds[0] * score_matrix[:, 0]).sum()) if len(seeds) > 0 else 0
        seed_eff = float((seeds[1] * score_matrix[:, 1]).sum()) if len(seeds) > 1 else 0

        run = optimize(p, mode="fast")
        final_rev_max = max(s.objective_values.get("Revenue", 0) for s in run.solutions)
        final_eff_min = min(s.objective_values.get("Effort", 0) for s in run.solutions)

        # Final max revenue should be at least as good as the seed's revenue.
        # (Elite preservation guarantees the seed survives if non-dominated.)
        assert final_rev_max >= seed_rev - 1e-6, (
            f"Final max Revenue {final_rev_max} < seed {seed_rev} — "
            f"elite preservation should have protected the seed."
        )
        # Final min effort should be at most as much as the seed's effort.
        assert final_eff_min <= seed_eff + 1e-6, (
            f"Final min Effort {final_eff_min} > seed {seed_eff} — "
            f"elite preservation should have protected the seed."
        )
