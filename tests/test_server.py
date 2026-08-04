"""Tests for MCP server tool handlers — model, solve, explore."""

import tempfile

import pytest

from engine.store import Store

# We test the internal handler functions directly, not via MCP protocol.
# This validates the logic without needing an MCP client.
import mcp_server.server as srv
import mcp_server.guidance as guidance


@pytest.fixture(autouse=True)
def tmp_store(monkeypatch):
    """Use a temp directory for all store operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Store(tmpdir)
        monkeypatch.setattr(srv, "store", s)
        # Clear injection tracking between tests
        srv._injected_skills.clear()
        yield s


# ─── model tool ───


class TestModelCreate:
    def test_create_returns_problem_id(self):
        result = srv.model(action="create", name="Test", domain="product")
        assert "problem_id" in result
        assert result["name"] == "Test"
        assert result["domain"] == "product"

    def test_create_defaults(self):
        result = srv.model(action="create")
        assert "problem_id" in result
        assert result["name"] == ""

    def test_create_accepts_constraints(self):
        """Constraints are problem structure — they land at create, not silently vanish."""
        result = srv.model(
            action="create",
            options=[{"name": "A"}, {"name": "B"}],
            constraints=[{"type": "cardinality", "min": 1, "max": 1}],
        )
        assert result["constraints"] == 1
        got = srv.model(action="get", problem_id=result["problem_id"], section="constraints")
        assert got["constraints"][0]["type"] == "cardinality"

    def test_create_rejects_content_params_it_does_not_apply(self):
        """Scores (and scenarios etc.) at create error with a pointer to update —
        silently dropping them would solve a different problem than described."""
        result = srv.model(
            action="create",
            options=[{"name": "A"}],
            scores=[{"option": "A", "objective": "Rev", "value": 1}],
        )
        assert "error" in result
        assert "update" in result["error"]
        result = srv.model(action="create", scenario_config={"scenarios": []})
        assert "error" in result

    def test_create_rejects_source_with_load_pointer(self):
        """source is the loader param — create must point at action='load' rather than
        silently making an empty problem (a real agent-confusion trap)."""
        result = srv.model(action="create", source="investment_portfolio")
        assert "error" in result
        assert "load" in result["error"]
        assert "source" in result["error"]
        # It must NOT have created an (empty) problem.
        assert "problem_id" not in result


class TestInteractionMatrixIncrementalWrite:
    """A matrix too large for one tool call must be buildable across several.

    Regression for a silent ceiling: the update path swapped the whole matrix
    object in, so mode="upsert" was ignored on the base matrix and the only way
    to supply one was a single payload. A dense matrix over dozens of options
    exceeds what a model can emit in one call, so those problems could not be
    framed through the tools at all — the partial call is dropped and every
    retry hits the same wall.
    """

    def _setup(self):
        pid = srv.model(action="create", name="Matrix")["problem_id"]
        srv.model(action="update", problem_id=pid, objectives=[
            {"name": "Risk", "direction": "minimize", "aggregation": "quadratic"},
        ], options=[{"name": n} for n in ("A", "B", "C")])
        return pid

    def _matrix(self, pid):
        # Read the stored problem directly: `model get` returns a summary view
        # rather than the raw matrix payload.
        ims = srv.store.load(pid).interaction_matrices
        return ims[0].entries if ims else {}

    def test_upsert_accumulates_across_calls(self):
        pid = self._setup()
        srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "mode": "upsert", "entries": {"A": {"B": 0.5}}}])
        srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "mode": "upsert", "entries": {"A": {"C": 0.25}}}])
        entries = self._matrix(pid)
        # Both chunks survive, and symmetry is enforced on each.
        assert entries["A"]["B"] == 0.5 and entries["B"]["A"] == 0.5
        assert entries["A"]["C"] == 0.25 and entries["C"]["A"] == 0.25

    def test_replace_still_discards_prior_cells(self):
        pid = self._setup()
        srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "mode": "upsert", "entries": {"A": {"B": 0.5}}}])
        srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "entries": {"A": {"C": 0.25}}}])  # default: replace
        entries = self._matrix(pid)
        assert "B" not in entries.get("A", {})
        assert entries["A"]["C"] == 0.25

    def test_cell_echo_rises_across_upsert_chunks(self):
        """Every touched objective echoes its post-merge cell count — the response is
        self-certifying, so a chunked build can watch the matrix grow instead of trusting."""
        pid = self._setup()
        r1 = srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "mode": "upsert", "entries": {"A": {"B": 0.5}}}])
        r2 = srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "mode": "upsert", "entries": {"A": {"C": 0.25}}}])
        assert r1["interaction_matrix_cells"]["Risk"]["cells"] == 2   # A-B mirrored
        assert r2["interaction_matrix_cells"]["Risk"]["cells"] == 4   # + A-C mirrored
        assert "note" not in r2["interaction_matrix_cells"]["Risk"]

    def test_cell_echo_names_a_wipe_when_mode_is_omitted(self):
        """The silent-loss case from the live capture: a chunked build that omits
        mode="upsert" on a later chunk replaces the objective's map, and only a later
        validation error revealed it. A falling cell count is the wipe's signature —
        the response must say so and name the cause (the matrix twin of constraints_note)."""
        pid = self._setup()
        srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "mode": "upsert",
             "entries": {"A": {"B": 0.5, "C": 0.3}, "B": {"C": 0.2}}}])
        r = srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "entries": {"A": {"C": 0.25}}}])  # default: replace
        echo = r["interaction_matrix_cells"]["Risk"]
        assert echo["cells"] == 1 and "cell count fell" in echo["note"]
        assert 'mode="upsert"' in echo["note"]

    def test_scale_groups_apply_to_the_base_matrix(self):
        pid = self._setup()
        srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "mode": "upsert", "entries": {"A": {"B": 0.4}}}])
        srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "mode": "upsert", "entries": {},
             "scale_groups": [{"options": ["A", "B"], "factor": 1.5}]}])
        assert self._matrix(pid)["A"]["B"] == pytest.approx(0.6)


class TestModelUpdate:
    def test_update_metadata(self):
        created = srv.model(action="create", name="Old")
        pid = created["problem_id"]
        result = srv.model(action="update", problem_id=pid, name="New")
        assert result["problem_id"] == pid

    def test_update_objectives(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        result = srv.model(action="update", problem_id=pid, objectives=[
            {"name": "Revenue", "direction": "maximize", "unit": "$"},
            {"name": "Effort", "direction": "minimize", "unit": "weeks"},
        ])
        assert result["status"]["objectives"] == 2

    def test_constraint_replacement_echoes_count_and_notes_shrink(self):
        """status echoes the constraint count; a shrinking replacement is named,
        a growing one is silent."""
        pid = srv.model(
            action="create",
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            objectives=[{"name": "Value", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
            constraints=[{"type": "cardinality", "min": 1, "max": 2},
                         {"type": "force_include", "option": "A"}],
        )["problem_id"]

        grown = srv.model(action="update", problem_id=pid, constraints=[
            {"type": "cardinality", "min": 1, "max": 2},
            {"type": "force_include", "option": "A"},
            {"type": "force_exclude", "option": "C"},
        ])
        assert grown["status"]["constraints"] == 3
        assert "constraints_note" not in grown

        shrunk = srv.model(action="update", problem_id=pid, constraints=[
            {"type": "force_include", "option": "A"},
        ])
        assert shrunk["status"]["constraints"] == 1
        assert "replaced the whole set" in shrunk["constraints_note"]
        assert "(was 3)" in shrunk["constraints_note"]
        # The dropped rules are named by key — no verification `get` needed.
        assert "cardinality:1:2" in shrunk["constraints_note"]
        assert "force_exclude:C" in shrunk["constraints_note"]

    def test_duplicate_allocation_bounds_are_merged_and_echoed(self):
        """The silent-loss case from the live capture: an option carrying a contractual
        floor row AND a separate cap row kept only the last one. They now intersect, and
        the response says which box was applied instead of leaving it to the allocations."""
        pid = srv.model(
            action="create", approach="proportional",
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            objectives=[{"name": "Value", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
        )["problem_id"]
        r = srv.model(action="update", problem_id=pid, constraints=[
            {"type": "allocation_bound", "option": "A", "min": 20, "max": 100},
            {"type": "allocation_bound", "option": "A", "min": 0, "max": 40},
        ])
        assert r["status"]["constraints"] == 2
        note = r["constraints_merged_note"]
        assert "'A' (2 allocation_bound rows) → min 20%, max 40%" in note
        assert any("2 allocation_bound rows" in i["message"] and i["severity"] == "warning"
                   for i in r["validation_issues"])
        # The formulation reads the APPLIED box back, once, on both surfaces — the raw
        # rows ("A allocation 20–100%" and "A allocation 0–40%") describe neither.
        card = srv.model(action="get", problem_id=pid, section="summary")
        assert card["viz_data"]["constraints"] == ["A allocation 20–40% (2 rows merged)"]
        assert "A allocation 20–40% (2 rows merged)" in card["visualization"]
        assert "20–100%" not in card["visualization"]

        # A single-row model says nothing — the note and the merge caption are for the
        # merged case only.
        clean = srv.model(action="update", problem_id=pid, constraints=[
            {"type": "allocation_bound", "option": "A", "min": 20, "max": 40}])
        assert "constraints_merged_note" not in clean
        clean_card = srv.model(action="get", problem_id=pid, section="summary")
        assert clean_card["viz_data"]["constraints"] == ["A allocation 20–40%"]

    def test_whole_plan_constraint_merge_is_echoed(self):
        """Several rows of a whole-plan type resolve to one applied value, which is not
        any single row the caller sent — the SAME note names it."""
        pid = srv.model(
            action="create", approach="proportional",
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}],
            objectives=[{"name": "Value", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
        )["problem_id"]
        r = srv.model(action="update", problem_id=pid, constraints=[
            {"type": "max_allocation", "max": 50},
            {"type": "max_allocation", "max": 40},
            {"type": "cardinality", "min": 1, "max": 3},
            {"type": "cardinality", "min": 2, "max": 4},
        ])
        note = r["constraints_merged_note"]
        assert "2 max_allocation rows → ≤40% per option" in note
        assert "2 cardinality rows → select 2–3" in note
        assert any("apply as the tightest cap" in i["message"] and i["severity"] == "warning"
                   for i in r["validation_issues"])

        # One row per type says nothing.
        clean = srv.model(action="update", problem_id=pid, constraints=[
            {"type": "max_allocation", "max": 40}])
        assert "constraints_merged_note" not in clean

    def test_both_merge_kinds_share_one_note(self):
        """One builder writes `constraints_merged_note`, so a model carrying both kinds
        reports both. Two independent writers would have dropped one silently — the exact
        failure class these notes exist to prevent."""
        pid = srv.model(
            action="create", approach="proportional",
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}],
            objectives=[{"name": "Value", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
        )["problem_id"]
        note = srv.model(action="update", problem_id=pid, constraints=[
            {"type": "allocation_bound", "option": "A", "min": 20, "max": 100},
            {"type": "allocation_bound", "option": "A", "min": 0, "max": 40},
            {"type": "max_allocation", "max": 50},
            {"type": "max_allocation", "max": 45},
            {"type": "cardinality", "min": 1, "max": 3},
            {"type": "cardinality", "min": 2, "max": 4},
        ])["constraints_merged_note"]
        assert "'A' (2 allocation_bound rows) → min 20%, max 40%" in note
        assert "2 max_allocation rows → ≤45% per option" in note
        assert "2 cardinality rows → select 2–3" in note

        # The formulation reads every collapsed rule back as the rule applied — six rows
        # in, three applied lines out, each captioned with the rows behind it.
        card = srv.model(action="get", problem_id=pid, section="summary")
        assert card["viz_data"]["constraints"] == [
            "A allocation 20–40% (2 rows merged)",
            "≤45% per option (2 rows merged)",
            "select 2–3 (2 rows merged)",
        ]
        for raw in ("≤50% per option", "select 1–3", "select 2–4"):
            assert raw not in card["visualization"]

    def test_empty_intersections_are_left_to_the_validation_error(self):
        """The note reports resolutions; "select 4–2" or "min 60%, max 40%" captioned as a
        tightest combination describes a rule the model applies. The errors beside the note
        state those cases correctly, so the note stays quiet on them."""
        pid = srv.model(
            action="create", approach="proportional",
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            objectives=[{"name": "Value", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
        )["problem_id"]
        r = srv.model(action="update", problem_id=pid, constraints=[
            {"type": "cardinality", "min": 4, "max": 5},
            {"type": "cardinality", "min": 1, "max": 2},
            {"type": "allocation_bound", "option": "A", "min": 60, "max": 100},
            {"type": "allocation_bound", "option": "A", "min": 0, "max": 40},
        ])
        assert "constraints_merged_note" not in r
        msgs = [i["message"] for i in r["validation_issues"] if i["severity"] == "error"]
        assert any("empty range" in m and "min 4 > merged max 2" in m for m in msgs)
        assert any("empty box" in m and "60% > merged max 40%" in m for m in msgs)

    def test_create_checks_the_constraints_it_was_given(self):
        """The create-time echo gap: constraints passed at create went unchecked until the
        next update, so a one-shot framing call learned what its model applies at solve."""
        created = srv.model(
            action="create", approach="proportional",
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            objectives=[{"name": "Value", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
            constraints=[{"type": "cardinality", "min": 1, "max": 2},
                         {"type": "cardinality", "min": 1, "max": 3}],
        )
        assert "2 cardinality rows → select 1–2" in created["constraints_merged_note"]
        assert any("apply intersected" in i["message"] for i in created["validation_issues"])
        # A create without constraints keeps its lean response.
        bare = srv.model(action="create", options=[{"name": "A"}])
        assert "constraints_merged_note" not in bare and "validation_issues" not in bare

    def test_option_removal_cascade_carries_constraints_note(self):
        """An options replacement that cascade-drops referencing constraints is
        named too — rules must never vanish without a callout."""
        pid = srv.model(
            action="create",
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            objectives=[{"name": "Value", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
            constraints=[{"type": "cardinality", "min": 1, "max": 2},
                         {"type": "force_include", "option": "A"}],
        )["problem_id"]

        result = srv.model(action="update", problem_id=pid,
                           options=[{"name": "B"}, {"name": "C"}])
        assert result["status"]["constraints"] == 1
        assert "removed options/objectives" in result["constraints_note"]
        assert "force_include:A" in result["constraints_note"]

    def test_combined_update_notes_shrink_against_pre_call_count(self):
        """The shrink baseline is the pre-call count: an objectives cascade in the
        same call must not mask a partial constraints send (the note fires and
        reports the true 'was')."""
        pid = srv.model(
            action="create",
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            objectives=[{"name": "Value", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"},
                        {"name": "Risk", "direction": "minimize"}],
            constraints=[{"type": "cardinality", "min": 1, "max": 2},
                         {"type": "force_include", "option": "A"},
                         {"type": "objective_bound", "objective": "Risk",
                          "operator": "max", "value": 100}],
        )["problem_id"]

        # Dropping Risk cascades away its objective_bound (3 → 2); the partial
        # constraints list then holds the count at 2 — pre-fix, the branch-local
        # baseline read 2 and the note never fired.
        result = srv.model(
            action="update", problem_id=pid,
            objectives=[{"name": "Value", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
            constraints=[{"type": "cardinality", "min": 1, "max": 2},
                         {"type": "force_include", "option": "A"}],
        )
        assert result["status"]["constraints"] == 2
        assert "(was 3)" in result["constraints_note"]

    def test_update_scores_merge(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[
                      {"name": "Rev", "direction": "maximize"},
                      {"name": "Eff", "direction": "minimize"},
                  ],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}])

        # First batch
        srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "Rev", "value": 5},
        ])
        # Second batch — should merge, not replace
        srv.model(action="update", problem_id=pid, scores=[
            {"option": "B", "objective": "Rev", "value": 7},
        ])
        problem = srv.model(action="get", problem_id=pid, section="full")
        assert len(problem["scores"]) == 2

    def test_update_scores_upsert(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}])
        srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "Rev", "value": 5},
        ])
        srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "Rev", "value": 99},
        ])
        problem = srv.model(action="get", problem_id=pid, section="full")
        rev_scores = [s for s in problem["scores"] if s["option"] == "A" and s["objective"] == "Rev"]
        assert len(rev_scores) == 1
        assert rev_scores[0]["value"] == 99

    def test_update_clears_run(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        # Build a solvable problem
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
                  scores=[
                      {"option": "A", "objective": "Rev", "value": 8},
                      {"option": "A", "objective": "Eff", "value": 5},
                      {"option": "B", "objective": "Rev", "value": 6},
                      {"option": "B", "objective": "Eff", "value": 3},
                      {"option": "C", "objective": "Rev", "value": 9},
                      {"option": "C", "objective": "Eff", "value": 7},
                  ],
                  constraints=[{"type": "cardinality", "min": 1, "max": 2}])
        srv.solve(action="run", problem_id=pid)
        p = srv.model(action="get", problem_id=pid, section="full")
        assert p["run"] is not None

        # Updating scores should mark results stale (not clear the run)
        srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "Rev", "value": 1},
        ])
        p = srv.model(action="get", problem_id=pid, section="full")
        assert p["run"] is not None  # run preserved for comparison
        assert p["results_stale"] is True

    def test_cascading_removes_scores_on_objective_removal(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
                  scores=[
                      {"option": "A", "objective": "Rev", "value": 5},
                      {"option": "A", "objective": "Eff", "value": 3},
                  ])
        # Remove "Eff" objective
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Impact", "direction": "maximize"}])
        p = srv.model(action="get", problem_id=pid, section="full")
        # Only Rev score should remain
        assert all(s["objective"] != "Eff" for s in p["scores"])

    def test_cascading_removes_constraints_on_option_removal(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
                  constraints=[{"type": "force_include", "option": "A"}])
        # Remove option A
        srv.model(action="update", problem_id=pid,
                  options=[{"name": "B"}, {"name": "C"}, {"name": "D"}])
        p = srv.model(action="get", problem_id=pid, section="full")
        assert len(p["constraints"]) == 0

    def test_update_nonexistent_problem(self):
        result = srv.model(action="update", problem_id="nonexistent")
        assert "error" in result

    def test_update_surfaces_validation_issues_but_not_incomplete_scores(self):
        """Adjacent fix #6: model/update returns validation_issues list on structural
        change, but filters out the incomplete-scores noise. A genuinely broken
        constraint (force_include on unknown option) is surfaced."""
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}])

        # No scores yet — would be an "incomplete score matrix" validation error.
        # That should NOT appear in validation_issues (it's expected during setup).
        result = srv.model(action="update", problem_id=pid,
            constraints=[{"type": "cardinality", "min": 1, "max": 2}])
        # incomplete-scores noise filtered out
        if "validation_issues" in result:
            for issue in result["validation_issues"]:
                assert "Score matrix incomplete" not in issue["message"]

        # Now trigger a genuine error: force_include on an unknown option
        result = srv.model(action="update", problem_id=pid,
            constraints=[{"type": "force_include", "option": "NONEXISTENT"}])
        assert "validation_issues" in result
        messages = [i["message"] for i in result["validation_issues"]]
        assert any("NONEXISTENT" in m for m in messages)

    def test_update_missing_problem_id(self):
        result = srv.model(action="update")
        assert "error" in result

    def test_score_references_invalid_option(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}])
        result = srv.model(action="update", problem_id=pid, scores=[
            {"option": "NONEXISTENT", "objective": "Rev", "value": 5},
        ])
        assert "error" in result

    def test_score_references_invalid_objective(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}])
        result = srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "NONEXISTENT", "value": 5},
        ])
        assert "error" in result


class TestModelList:
    def test_list_empty(self):
        result = srv.model(action="list")
        # Wrapped in {"problems": [...]} so FastMCP serializes to a TextContent
        # block — bare-list returns are dropped by FastMCP 1.25.0 and rejected
        # by the Anthropic MCP connector.
        assert result == {"problems": []}

    def test_list_returns_problems_array(self):
        srv.model(action="create", name="One")
        srv.model(action="create", name="Two")
        result = srv.model(action="list")
        assert isinstance(result, dict)
        assert "problems" in result
        assert isinstance(result["problems"], list)
        assert len(result["problems"]) == 2


class TestModelDelete:
    def test_delete(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        result = srv.model(action="delete", problem_id=pid)
        assert result["deleted"] == pid

    def test_delete_nonexistent(self):
        result = srv.model(action="delete", problem_id="nonexistent")
        assert "error" in result


class TestModelGet:
    def test_get_defaults_to_summary(self):
        created = srv.model(action="create", name="Fetch Me")
        pid = created["problem_id"]
        result = srv.model(action="get", problem_id=pid)   # no section -> summary
        assert result["section"] == "summary"

    def test_get_full_dump(self):
        created = srv.model(action="create", name="Fetch Me")
        pid = created["problem_id"]
        result = srv.model(action="get", problem_id=pid, section="full")
        assert result["name"] == "Fetch Me"

    def test_get_summary_section(self):
        pid = _build_solvable_problem()
        result = srv.model(action="get", problem_id=pid, section="summary")
        assert result["section"] == "summary"
        assert result["objectives_count"] == 2
        assert result["options_count"] == 4
        assert result["scores_count"] == 8
        # Summary must not contain the heavy fields
        assert "scores" not in result
        assert "run" not in result

    def test_get_scores_section(self):
        pid = _build_solvable_problem()
        result = srv.model(action="get", problem_id=pid, section="scores")
        assert result["section"] == "scores"
        assert len(result["scores"]) == 8
        # Should not include other heavy fields
        assert "constraints" not in result

    def test_get_unknown_section(self):
        pid = _build_solvable_problem()
        result = srv.model(action="get", problem_id=pid, section="nonsense")
        assert "error" in result


class TestUnknownAction:
    def test_unknown_model_action(self):
        result = srv.model(action="foobar")
        assert "error" in result


# ─── solve tool ───


def _build_solvable_problem():
    created = srv.model(action="create")
    pid = created["problem_id"]
    srv.model(action="update", problem_id=pid,
              objectives=[{"name": "Rev", "direction": "maximize"},
                          {"name": "Eff", "direction": "minimize"}],
              options=[{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}],
              scores=[
                  {"option": "A", "objective": "Rev", "value": 8},
                  {"option": "A", "objective": "Eff", "value": 5},
                  {"option": "B", "objective": "Rev", "value": 6},
                  {"option": "B", "objective": "Eff", "value": 3},
                  {"option": "C", "objective": "Rev", "value": 9},
                  {"option": "C", "objective": "Eff", "value": 7},
                  {"option": "D", "objective": "Rev", "value": 4},
                  {"option": "D", "objective": "Eff", "value": 2},
              ],
              constraints=[{"type": "cardinality", "min": 2, "max": 3}])
    return pid


class TestSolveValidate:
    def test_validate_ready(self):
        pid = _build_solvable_problem()
        result = srv.solve(action="validate", problem_id=pid)
        assert result["ready"] is True
        assert result["readiness"]["verdict"] == "ready"
        assert result["readiness"]["next_steps"] == "solve run"

    def test_validate_not_ready(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        result = srv.solve(action="validate", problem_id=pid)
        assert result["ready"] is False
        # An empty problem is a structure problem, not a data problem
        assert result["readiness"]["verdict"] == "framing_gap"

    def test_validate_missing_scores_capped(self):
        # readiness.score_gaps carries the per-objective truth, so the raw cell list
        # ships a capped head + total instead of flooding the wire at portfolio scale.
        created = srv.model(action="create", name="unscored portfolio")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": f"P{i}"} for i in range(30)])
        result = srv.solve(action="validate", problem_id=pid)
        assert len(result["missing_scores"]) == 20
        assert result["missing_scores_total"] == 60
        assert result["readiness"]["score_gaps"][0]["missing"] == 30  # rollup stays exact

    def test_validate_data_gap(self):
        created = srv.model(action="create", name="scores pending")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
                  scores=[{"option": "A", "objective": "Rev", "value": 5}])
        result = srv.solve(action="validate", problem_id=pid)
        assert result["ready"] is False
        r = result["readiness"]
        assert r["verdict"] == "data_gap"
        assert {g["objective"] for g in r["score_gaps"]} == {"Rev", "Eff"}
        assert next(g for g in r["score_gaps"] if g["objective"] == "Eff")["unscored"] is True

    def test_validate_nonexistent(self):
        result = srv.solve(action="validate", problem_id="nonexistent")
        assert "error" in result


class TestSolveRun:
    def test_run_scenarios_flag_redirects_when_contradicting(self):
        """run_scenarios=true on another action is the wrong-name miscall — redirect."""
        pid = _build_solvable_problem()
        result = srv.solve(action="run", problem_id=pid, run_scenarios=True)
        assert 'action="run_scenarios"' in result["error"]

    def test_run_scenarios_flag_tolerated_beside_its_action(self):
        """run_scenarios=true beside action="run_scenarios" states the right intent
        twice — the call proceeds (here into normal validation) instead of bouncing."""
        pid = srv.model(
            action="create",
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            objectives=[{"name": "Value", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
        )["problem_id"]
        result = srv.solve(action="run_scenarios", problem_id=pid, run_scenarios=True)
        assert "solve has no" not in result.get("error", "")

    def test_run_success(self):
        pid = _build_solvable_problem()
        result = srv.solve(action="run", problem_id=pid)
        assert "run_id" in result
        assert result["solutions_found"] > 0
        assert "quality" in result
        # Compact response: no full solutions array, but objective_ranges + preview present
        assert "solutions" not in result
        assert "objective_ranges" in result
        assert "preview" in result
        assert "extremes" in result["preview"]

    def test_run_echoes_seed_used_when_random(self):
        pid = _build_solvable_problem()
        result = srv.solve(action="run", problem_id=pid)
        assert "seed_used" in result
        assert isinstance(result["seed_used"], int)

    def test_run_reproducible_with_fixed_seed(self):
        pid1 = _build_solvable_problem()
        pid2 = _build_solvable_problem()
        r1 = srv.solve(action="run", problem_id=pid1, seed=1234)
        r2 = srv.solve(action="run", problem_id=pid2, seed=1234)
        assert r1["seed_used"] == 1234
        assert r2["seed_used"] == 1234
        # Same seed on identical problem → same objective ranges
        assert r1["objective_ranges"] == r2["objective_ranges"]
        assert r1["solutions_found"] == r2["solutions_found"]

    def test_run_different_seeds_may_differ(self):
        pid1 = _build_solvable_problem()
        pid2 = _build_solvable_problem()
        r1 = srv.solve(action="run", problem_id=pid1, seed=1)
        r2 = srv.solve(action="run", problem_id=pid2, seed=9999)
        assert r1["seed_used"] == 1
        assert r2["seed_used"] == 9999
        # Both must succeed; frontier shape may vary
        assert r1["solutions_found"] > 0
        assert r2["solutions_found"] > 0

    def test_run_writes_full_result_file(self):
        """Fix 1: every solve writes the complete result to disk and returns the path."""
        import json as _json
        from pathlib import Path

        pid = _build_solvable_problem()
        result = srv.solve(action="run", problem_id=pid)
        assert "full_result_path" in result
        path = Path(result["full_result_path"])
        assert path.exists(), f"{path} was not written"
        payload = _json.loads(path.read_text())
        # File contains all solutions the response omits
        assert "solutions" in payload
        assert len(payload["solutions"]) == result["solutions_found"]
        # First solution has full structure
        sol = payload["solutions"][0]
        assert "solution_id" in sol
        assert "objective_values" in sol

    def test_run_validation_failure_includes_missing_scores(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}])
        result = srv.solve(action="run", problem_id=pid)
        assert result["ready"] is False
        assert "missing_scores" in result
        assert len(result["missing_scores"]) == 6  # 3 options * 2 objectives

    def test_run_includes_frontier_complete_and_quality(self):
        """1.10 + 1.11: solve response carries frontier_complete + frontier_quality."""
        pid = _build_solvable_problem()
        result = srv.solve(action="run", problem_id=pid)
        # 1.10
        assert "frontier_complete" in result
        assert isinstance(result["frontier_complete"], bool)
        # No pruning expected on this small problem → complete
        assert result["frontier_complete"] is True
        assert result["total_pareto_found"] == result["solutions_found"]
        # 1.11
        assert "frontier_quality" in result
        fq = result["frontier_quality"]
        assert fq["status"] in ("GOOD", "WARNING", "POOR")
        assert "gates" in fq and "issues" in fq
        assert set(fq["gates"]) == {"frontier_returned", "non_trivial", "diverse"}

    def test_run_frontier_complete_false_when_pruned(self):
        """1.10: frontier_complete is False when max_solutions truncates the frontier."""
        pid = _build_solvable_problem()
        # Tight cap forces pruning if there's any frontier larger than 1
        result = srv.solve(action="run", problem_id=pid, max_solutions=1)
        if result.get("solutions_found", 0) > 0 and result.get("total_pareto_found", 0) > 1:
            assert result["frontier_complete"] is False
            assert result["solutions_found"] < result["total_pareto_found"]


# ─── explore tool ───


class TestExploreTradeoffs:
    def test_tradeoffs(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="tradeoffs", problem_id=pid)
        assert "total_solutions" in result
        assert "balanced_solution" in result

    def test_tradeoffs_no_run(self):
        created = srv.model(action="create")
        result = srv.explore(action="tradeoffs", problem_id=created["problem_id"])
        assert "error" in result


class TestExploreCompare:
    def test_compare_requires_min_2(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="compare", problem_id=pid, solution_ids=[1])
        assert "error" in result
        assert "2+" in result["error"]

    def test_compare_success(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="compare", problem_id=pid, solution_ids=[1, 2])
        assert "shared_options" in result

    def test_detail_is_honored_by_the_solution_ids_form(self):
        """`detail` was advertised on compare but only the signatures form read it —
        the id form dumped every plan whole and dropped the flag."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        compact = srv.explore(action="compare", problem_id=pid, solution_ids=[1, 2])
        assert compact["detail"] is False
        assert all("selected_options" not in s for s in compact["solutions"])
        # The structural difference survives the compact form — that's why it can be default.
        assert "shared_options" in compact and "differentiating_options" in compact

        full = srv.explore(action="compare", problem_id=pid, solution_ids=[1, 2], detail=True)
        assert full["detail"] is True
        assert all("selected_options" in s for s in full["solutions"])


class TestExploreSolutions:
    def test_solutions_listing(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="solutions", problem_id=pid)
        assert "total_solutions" in result
        assert "solutions" in result

    def test_single_solution(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="solutions", problem_id=pid, solution_id=1)
        assert result["solution_id"] == 1

    def test_single_solution_missing_id(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        # `solution` was consolidated into `solutions(solution_id=)` — the old name
        # is a clean break, not an alias.
        result = srv.explore(action="solution", problem_id=pid)
        assert "error" in result

    def test_single_solution_not_found(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="solutions", problem_id=pid, solution_id=9999)
        assert "error" in result


class TestRunHistory:
    def test_structural_change_sets_stale(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        p = srv.model(action="get", problem_id=pid, section="full")
        assert p["results_stale"] is False

        # Structural change sets stale
        result = srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "Rev", "value": 1},
        ])
        assert result["status"]["results_stale"] is True
        p = srv.model(action="get", problem_id=pid, section="full")
        assert p["results_stale"] is True
        assert p["run"] is not None  # run preserved

    def test_solve_clears_stale(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "Rev", "value": 1},
        ])
        p = srv.model(action="get", problem_id=pid, section="full")
        assert p["results_stale"] is True

        # Re-solve clears stale
        srv.solve(action="run", problem_id=pid)
        p = srv.model(action="get", problem_id=pid, section="full")
        assert p["results_stale"] is False

    def test_solve_archives_previous_run(self):
        pid = _build_solvable_problem()
        result1 = srv.solve(action="run", problem_id=pid)
        run1_id = result1["run_id"]

        # Second solve
        result2 = srv.solve(action="run", problem_id=pid)
        run2_id = result2["run_id"]

        p = srv.model(action="get", problem_id=pid, section="full")
        assert p["run"]["run_id"] == run2_id
        assert len(p["runs"]) == 1
        assert p["runs"][0]["run_id"] == run1_id

    def test_run_has_constraints_snapshot(self):
        pid = _build_solvable_problem()
        result = srv.solve(action="run", problem_id=pid)
        p = srv.model(action="get", problem_id=pid, section="full")
        assert len(p["run"]["constraints_snapshot"]) > 0
        assert p["run"]["constraints_snapshot"][0]["type"] == "cardinality"

    def test_multiple_runs_accumulate(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        srv.solve(action="run", problem_id=pid)
        srv.solve(action="run", problem_id=pid)
        p = srv.model(action="get", problem_id=pid, section="full")
        assert len(p["runs"]) == 2  # 2 archived + 1 current
        assert p["run"] is not None

    def test_status_shows_total_runs(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        srv.solve(action="run", problem_id=pid)
        result = srv.model(action="update", problem_id=pid, name="Updated")
        assert result["status"]["total_runs"] == 2


class TestReferencePoints:
    def test_set_reference_points(self):
        pid = _build_solvable_problem()
        result = srv.model(action="update", problem_id=pid, reference_points=[
            {"type": "baseline", "name": "Current", "objective_values": {"Rev": 10, "Eff": 8}},
            {"type": "aspirational", "name": "Target", "objective_values": {"Rev": 20, "Eff": 3}},
        ])
        p = srv.model(action="get", problem_id=pid, section="full")
        assert len(p["reference_points"]) == 2
        assert p["reference_points"][0]["type"] == "baseline"

    def test_setting_reference_points_points_to_narration(self):
        # B-P2: setting reference points surfaces the read-side narration playbook.
        pid = _build_solvable_problem()
        r = srv.model(action="update", problem_id=pid, reference_points=[
            {"type": "baseline", "name": "Current", "objective_values": {"Rev": 10, "Eff": 8}}])
        assert r["guidance_pointer"]["section"] == "Reference Point Narration"

    def test_setting_scenarios_points_to_sweep_discipline(self):
        # Setting scenario_config surfaces the sweep-construction discipline — it lives in
        # references (not the injected core), and its checks apply to the config just written.
        pid = _build_solvable_problem()
        r = srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [{"name": "downturn", "score_adjustments": []}],
        })
        assert r["guidance_pointer"]["section"] == "Sweep Discipline — Constructing Scenarios"
        # And the pointed-at section resolves as exactly one section.
        body = srv.get_skill("optimization_strategy",
                             section="Sweep Discipline — Constructing Scenarios")
        assert body.startswith("## Sweep Discipline")

    def test_reference_in_tradeoffs(self):
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, reference_points=[
            {"type": "baseline", "name": "Current", "objective_values": {"Rev": 10, "Eff": 8}},
        ])
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="tradeoffs", problem_id=pid)
        assert "balanced_vs_references" in result
        assert len(result["balanced_vs_references"]) == 1
        assert result["balanced_vs_references"][0]["type"] == "baseline"

    def test_reference_in_solution(self):
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, reference_points=[
            {"type": "aspirational", "name": "Goal", "objective_values": {"Rev": 25, "Eff": 2}},
        ])
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="solutions", problem_id=pid, solution_id=1)
        assert "vs_references" in result
        ref = result["vs_references"][0]
        assert "Rev" in ref["objectives"]
        assert "better" in ref["objectives"]["Rev"]

    def test_no_reference_no_field(self):
        """Without reference points, no extra fields in output."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="tradeoffs", problem_id=pid)
        assert "balanced_vs_references" not in result


class TestCompareRuns:
    def test_compare_two_runs(self):
        pid = _build_solvable_problem()
        r1 = srv.solve(action="run", problem_id=pid)
        r2 = srv.solve(action="run", problem_id=pid)
        result = srv.explore(
            action="compare_runs", problem_id=pid,
            run_ids=[r1["run_id"], r2["run_id"]],
        )
        assert "runs_compared" in result
        assert len(result["runs_compared"]) == 2
        assert "frontier_diffs" in result
        assert "option_coverage" in result

    def test_compare_runs_criteria_diff(self):
        pid = _build_solvable_problem()
        r1 = srv.solve(action="run", problem_id=pid)
        # Add a constraint and re-solve
        srv.model(action="update", problem_id=pid, constraints=[
            {"type": "cardinality", "min": 2, "max": 3},
            {"type": "force_include", "option": "A"},
        ])
        r2 = srv.solve(action="run", problem_id=pid)
        result = srv.explore(
            action="compare_runs", problem_id=pid,
            run_ids=[r1["run_id"], r2["run_id"]],
        )
        diffs = result["criteria_diffs"]
        assert len(diffs) == 1
        assert len(diffs[0]["added"]) > 0  # force_include was added

    def test_compare_runs_invalid_id(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(
            action="compare_runs", problem_id=pid,
            run_ids=["nonexistent1", "nonexistent2"],
        )
        assert "error" in result

    def test_compare_runs_needs_two(self):
        pid = _build_solvable_problem()
        r1 = srv.solve(action="run", problem_id=pid)
        result = srv.explore(
            action="compare_runs", problem_id=pid,
            run_ids=[r1["run_id"]],
        )
        assert "error" in result

    def test_small_problem_coverage_passes_through_whole(self):
        """The elision is a portfolio-scale trim — a 4-option model ships every row.
        (The elision itself is gated in tests/test_explorer.py at engine level.)"""
        pid = _build_solvable_problem()
        r1 = srv.solve(action="run", problem_id=pid, seed=1)
        r2 = srv.solve(action="run", problem_id=pid, seed=2)
        out = srv.explore(action="compare_runs", problem_id=pid,
                          run_ids=[r1["run_id"], r2["run_id"]])
        assert "option_coverage_elided" not in out
        assert all(len(cov) == 4 for cov in out["option_coverage"].values())


class TestScenarios:
    def test_set_scenario_config(self):
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base", "probability": 0.6, "score_overrides": []},
                {"name": "Growth", "probability": 0.4, "score_overrides": [
                    {"option": "A", "objective": "Rev", "value": 20},
                ]},
            ],
        })
        p = srv.model(action="get", problem_id=pid, section="full")
        assert p["scenario_config"]["enabled"] is True
        assert len(p["scenario_config"]["scenarios"]) == 2

    def test_scenario_motivated_by_survives_update(self):
        # The sensitivity→scenario handoff provenance must survive the wire: the update
        # path constructs Scenario field-by-field, so a missing kwarg silently drops it.
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "stress", "motivated_by": "shadow_price:Rev",
                 "score_adjustments": [{"objective": "Rev", "multiply": 0.9}]},
            ],
        })
        p = srv.model(action="get", problem_id=pid, section="full")
        assert p["scenario_config"]["scenarios"][0]["motivated_by"] == "shadow_price:Rev"

    def test_malformed_scenario_errors_clearly(self):
        # Model-validated: a bad nested value errors with its location, never a crash.
        pid = _build_solvable_problem()
        r = srv.model(action="update", problem_id=pid, scenario_config={
            "scenarios": [{"name": "bad", "score_adjustments": [{"objective": "Rev", "multiply": "x"}]}],
        })
        assert "invalid scenario_config" in r["error"] and "score_adjustments" in r["error"]

    def test_unknown_scenario_field_errors_not_silently_dropped(self):
        # The motivated_by bug class: a typoed field must error, not vanish.
        pid = _build_solvable_problem()
        r = srv.model(action="update", problem_id=pid, scenario_config={
            "scenarios": [{"name": "typo", "motivatedby": "shadow_price:Rev"}],
        })
        assert "invalid scenario_config" in r["error"] and "motivatedby" in r["error"]

    def test_run_scenarios(self):
        import json as _json
        from pathlib import Path

        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base", "probability": 0.5, "score_overrides": []},
                {"name": "Alt", "probability": 0.5, "score_overrides": [
                    {"option": "A", "objective": "Rev", "value": 1},
                    {"option": "C", "objective": "Rev", "value": 20},
                ]},
            ],
        })
        result = srv.solve(action="run_scenarios", problem_id=pid)
        assert "error" not in result
        assert result["scenarios_optimized"] == 2
        assert "Base" in result["results"]
        assert "Alt" in result["results"]
        assert result["results"]["Base"]["solutions_found"] > 0
        # Fix 1: per-scenario full result files written to disk
        assert "full_result_paths" in result
        assert set(result["full_result_paths"].keys()) == {"Base", "Alt"}
        for name, path_str in result["full_result_paths"].items():
            path = Path(path_str)
            assert path.exists()
            payload = _json.loads(path.read_text())
            assert payload["scenario"] == name
            assert len(payload["solutions"]) == result["results"][name]["solutions_found"]

    def test_scenario_results(self):
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base", "probability": 0.6, "score_overrides": []},
                {"name": "Growth", "probability": 0.4, "score_overrides": [
                    {"option": "A", "objective": "Rev", "value": 50},
                ]},
            ],
        })
        srv.solve(action="run_scenarios", problem_id=pid)
        result = srv.explore(action="scenario_results", problem_id=pid)
        assert "robust_options" in result
        assert "scenario_specific" in result
        assert "expected_values" in result
        assert "per_scenario" in result
        assert "visualization" in result

    def test_scenario_results_viz_data_carries_scenarios_and_regret(self):
        """viz_data must carry the scenario names (panel visibility gate) and the
        minimax-regret block (the regret section). Regret needs a base run, so
        this solves the base case before run_scenarios."""
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base", "probability": 0.6, "score_overrides": []},
                {"name": "Growth", "probability": 0.4, "score_overrides": [
                    {"option": "A", "objective": "Rev", "value": 50},
                ]},
            ],
        })
        srv.solve(action="run", problem_id=pid)            # base run → regret available
        srv.solve(action="run_scenarios", problem_id=pid)
        viz = srv.explore(action="scenario_results", problem_id=pid)["viz_data"]

        assert viz["type"] == "scenario_summary"
        # Panel visibility gate — the original bug emitted [].
        assert viz["scenarios"] == ["Base", "Growth"]
        # Minimax-regret block surfaced for the panel's regret section.
        regret = viz["regret"]
        assert regret["available"] is True
        assert regret["minimax_choice"]["solution_id"] is not None
        assert len(regret["per_solution"]) > 0
        first = regret["per_solution"][0]
        assert {"solution_id", "max_regret", "mean_regret", "feasible_in_all", "by_scenario"} <= set(first)
        # by_scenario drives the panel's per-solution tooltip — keyed by scenario name.
        assert set(first["by_scenario"]) == {"Base", "Growth"}
        # Unsaturated regret keeps its minimax pick and reports the full count.
        assert regret.get("saturated") is not True
        assert regret["per_solution_total"] >= len(regret["per_solution"])

    def test_scenario_regret_saturation_flagged(self):
        """When every base solution is infeasible under a scenario, regret saturates at
        1.0 across the board — the payload must say so and omit the meaningless minimax
        pick instead of nominating an arbitrary all-tied solution (user test, finding P5)."""
        pid = _build_solvable_problem()                  # base cardinality 2-3
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Tight", "constraint_overrides": [
                    {"type": "cardinality", "min": 1, "max": 1},   # every base slate infeasible
                ]},
            ],
        })
        srv.solve(action="run", problem_id=pid)
        srv.solve(action="run_scenarios", problem_id=pid)
        regret = srv.explore(action="scenario_results", problem_id=pid)["regret"]
        assert regret["available"] is True
        assert regret["saturated"] is True
        assert "Tight" in regret["saturation_note"]
        assert regret["minimax_choice"] is None
        assert all(ps["feasible_in_all"] is False for ps in regret["per_solution"])

    def test_scenario_regret_partial_wipeout_excluded_from_ranking(self):
        """A scenario NO base solution survives (total wipeout) must not saturate the whole
        metric: it is excluded from the minimax ranking and named in wipeout_note, while the
        surviving scenarios still produce a meaningful pick (supplier-selection eval finding —
        one wipeout scenario turned the regret table into an all-100%/infeasible wall)."""
        pid = _build_solvable_problem()                  # base cardinality 2-3
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Mild", "score_overrides": [
                    {"option": "A", "objective": "Rev", "value": 50},
                ]},
                {"name": "Wipeout", "constraint_overrides": [
                    {"type": "cardinality", "min": 1, "max": 1},   # every base slate infeasible
                ]},
            ],
        })
        srv.solve(action="run", problem_id=pid)
        srv.solve(action="run_scenarios", problem_id=pid)
        out = srv.explore(action="scenario_results", problem_id=pid)
        regret = out["regret"]
        assert regret["available"] is True
        assert regret.get("saturated") is not True
        assert regret["wipeout_scenarios"] == ["Wipeout"]
        assert "Wipeout" in regret["wipeout_note"]
        assert regret["minimax_choice"] is not None
        # The note carries the ranked-scope qualifier so consumers know what max/mean span.
        assert "ranked scenarios" in regret["note"]
        # The ranking ignores the wipeout scenario; per-scenario detail still shows it as total.
        first = regret["per_solution"][0]
        assert first["by_scenario"]["Wipeout"] == 1.0
        assert first["max_regret"] < 1.0
        assert all(ps["feasible_in_all"] is False for ps in regret["per_solution"])
        # feasible_in_ranked distinguishes "fails only the excluded wipeout" per row.
        assert first["feasible_in_ranked"] is True
        # Survivor counts: one source of truth, lifted into the per_scenario layer too.
        assert regret["survivors_by_scenario"]["Wipeout"] == 0
        assert regret["survivors_by_scenario"]["Mild"] > 0
        assert out["per_scenario"]["Wipeout"]["base_plans_feasible"] == 0
        assert out["per_scenario"]["Mild"]["base_plans_feasible"] > 0
        assert out["per_scenario"]["Mild"]["base_plans_total"] == regret["per_solution_total"]
        # per-objective regrets share the ranked scope: the wipeout's uniform 1.0 must not
        # leak into them, and the note names them alongside max/mean/minimax.
        assert regret["per_objective"]
        assert all(v["min_max_regret"] < 1.0 for v in regret["per_objective"].values())
        assert "per-objective" in regret["note"]

    def test_round_regret_reserves_one_for_the_clamp(self):
        """A displayed 1.0 always means total regret (infeasible or fully dominated) —
        a raw 0.9996 must render 0.999, keeping saturation naming and the minimax
        headline consistent with the raw saturated check."""
        from engine.explorer import _round_regret
        assert _round_regret(0.9996) == 0.999
        assert _round_regret(0.9994) == 0.999
        assert _round_regret(1.0) == 1.0
        assert _round_regret(0.5) == 0.5
        assert _round_regret(0.0) == 0.0

    def test_scenario_regret_wipeout_and_saturation_co_occur(self):
        """One total-wipeout scenario (excluded, named) can coexist with a saturated
        ranking over the surviving scenarios: every Pareto base plan holds A or B, so
        excluding each in its own ranked scenario clamps every plan somewhere ranked,
        while each ranked scenario keeps survivors of its own."""
        pid = _build_solvable_problem()                  # base cardinality 2-3 over A-D
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "NoA", "constraint_overrides": [
                    {"type": "cardinality", "min": 2, "max": 3},
                    {"type": "force_exclude", "option": "A"}]},
                {"name": "NoB", "constraint_overrides": [
                    {"type": "cardinality", "min": 2, "max": 3},
                    {"type": "force_exclude", "option": "B"}]},
                {"name": "Wipeout", "constraint_overrides": [
                    {"type": "cardinality", "min": 1, "max": 1}]},
            ],
        })
        srv.solve(action="run", problem_id=pid, seed=42)
        srv.solve(action="run_scenarios", problem_id=pid, seed=42)
        regret = srv.explore(action="scenario_results", problem_id=pid)["regret"]
        assert regret["wipeout_scenarios"] == ["Wipeout"]
        assert "Wipeout" in regret["wipeout_note"]
        assert regret["survivors_by_scenario"]["NoA"] > 0
        assert regret["survivors_by_scenario"]["NoB"] > 0
        assert regret["saturated"] is True
        assert regret["saturation_note"]
        assert regret["minimax_choice"] is None

    def test_explore_declines_stale_run_after_objective_change(self):
        """The iterate loop: edit the model's objectives after a solve, then peek at the
        old results — a worded stale decline, never a KeyError from deep inside."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"},
                              {"name": "Risk", "direction": "minimize"}])
        out = srv.explore(action="tradeoffs", problem_id=pid)
        assert "error" in out and "stale" in out["error"] and "Risk" in out["error"]

    def test_model_update_unknown_constraint_type_is_worded_error(self):
        pid = _build_solvable_problem()
        out = srv.model(action="update", problem_id=pid,
                        constraints=[{"type": "min_allocation", "min": 5}])
        assert "error" in out and "Valid types" in out["error"]

    def test_model_create_bad_approach_is_worded_error(self):
        out = srv.model(action="create", approach="alloc")
        assert "error" in out

    def test_scenario_results_without_run(self):
        pid = _build_solvable_problem()
        result = srv.explore(action="scenario_results", problem_id=pid)
        assert "error" in result

    def test_scenario_risk_and_cvar(self):
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base", "probability": 0.6, "score_overrides": []},
                {"name": "Growth", "probability": 0.4, "score_overrides": [
                    {"option": "A", "objective": "Rev", "value": 50},
                ]},
            ],
        })
        srv.solve(action="run_scenarios", problem_id=pid)
        # Default alpha = 0.2
        default_result = srv.explore(action="scenario_results", problem_id=pid)
        assert "scenario_risk" in default_result
        assert default_result["cvar_alpha"] == 0.2
        for obj, risk in default_result["scenario_risk"].items():
            assert "expected" in risk
            assert "worst_case" in risk
            assert "best_case" in risk
            assert "cvar_20" in risk
            assert "range" in risk
            assert risk["range"][0] <= risk["range"][1]

        # Custom alpha
        alt = srv.explore(action="scenario_results", problem_id=pid, cvar_alpha=0.5)
        assert alt["cvar_alpha"] == 0.5
        for risk in alt["scenario_risk"].values():
            assert "cvar_50" in risk

    def test_scenario_results_invalid_alpha(self):
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [{"name": "A"}, {"name": "B"}],
        })
        srv.solve(action="run_scenarios", problem_id=pid)
        result = srv.explore(action="scenario_results", problem_id=pid, cvar_alpha=1.5)
        assert "error" in result

    def test_scenarios_without_probabilities(self):
        """Scenarios should run fine without probabilities — each is an independent solve."""
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "A"},
                {"name": "B"},
            ],
        })
        result = srv.solve(action="run_scenarios", problem_id=pid)
        assert "error" not in result
        assert result["scenarios_optimized"] == 2

    def test_scenarios_with_score_adjustments(self):
        """Score adjustments (multiply/add) should modify scores per scenario."""
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base"},
                {"name": "Downside", "score_adjustments": [
                    {"objective": "Revenue", "multiply": 0.8},
                ]},
            ],
        })
        result = srv.solve(action="run_scenarios", problem_id=pid)
        assert "error" not in result
        assert result["scenarios_optimized"] == 2

    def test_scenarios_with_constraint_overrides(self):
        """Constraint overrides should replace base constraints for that scenario."""
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base"},
                {"name": "Constrained", "constraint_overrides": [
                    {"type": "cardinality", "min": 1, "max": 2},
                    {"type": "force_include", "option": "A"},
                ]},
            ],
        })
        result = srv.solve(action="run_scenarios", problem_id=pid)
        assert "error" not in result
        assert result["scenarios_optimized"] == 2

    def test_scenarios_with_interaction_matrix_overrides(self):
        """Per-scenario interaction matrix overrides should change portfolio-level quadratic
        aggregation independently per scenario. Correlation regime shifts (e.g. recession)
        can be modelled by supplying a different covariance matrix for that scenario.
        """
        # Build a 3-option, 2-objective proportional problem with quadratic Volatility.
        created = srv.model(action="create", approach="proportional")
        pid = created["problem_id"]
        srv.model(
            action="update", problem_id=pid,
            objectives=[
                {"name": "Return", "direction": "maximize", "aggregation": "sum"},
                {"name": "Volatility", "direction": "minimize", "aggregation": "quadratic"},
            ],
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            scores=[
                {"option": "A", "objective": "Return", "value": 10},
                {"option": "B", "objective": "Return", "value": 8},
                {"option": "C", "objective": "Return", "value": 6},
                {"option": "A", "objective": "Volatility", "value": 20},
                {"option": "B", "objective": "Volatility", "value": 20},
                {"option": "C", "objective": "Volatility", "value": 20},
            ],
            constraints=[
                {"type": "cardinality", "min": 2, "max": 3},
                {"type": "max_allocation", "max": 60},
            ],
        )
        # Base: diagonal-only matrix (low correlation → diversification possible)
        base_matrix = {
            "objective": "Volatility",
            "entries": {
                "A": {"A": 400.0, "B": 0.0, "C": 0.0},
                "B": {"A": 0.0, "B": 400.0, "C": 0.0},
                "C": {"A": 0.0, "B": 0.0, "C": 400.0},
            },
        }
        srv.model(action="update", problem_id=pid, interaction_matrices=[base_matrix])

        # Full-correlation override: off-diagonals = variance (correlation = 1)
        full_corr_matrix = {
            "objective": "Volatility",
            "entries": {
                "A": {"A": 400.0, "B": 400.0, "C": 400.0},
                "B": {"A": 400.0, "B": 400.0, "C": 400.0},
                "C": {"A": 400.0, "B": 400.0, "C": 400.0},
            },
        }
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Diversifiable"},  # uses base matrix
                {"name": "FullCorrelation", "interaction_matrix_overrides": [full_corr_matrix]},
            ],
        })
        result = srv.solve(action="run_scenarios", problem_id=pid, mode="fast")
        assert "error" not in result
        assert result["scenarios_optimized"] == 2

        # Fetch min-vol extremes from each scenario.
        diverse = srv.explore(action="tradeoffs", problem_id=pid, scenario="Diversifiable")
        correlated = srv.explore(action="tradeoffs", problem_id=pid, scenario="FullCorrelation")

        min_vol_diverse = diverse["objective_ranges"]["Volatility"]["min"]
        min_vol_correlated = correlated["objective_ranges"]["Volatility"]["min"]

        # With zero correlation a diversified portfolio achieves sqrt(0.4^2 + 0.4^2 + 0.2^2)*sqrt(400) ≈ 12;
        # with full correlation every portfolio hits sqrt(400) = 20. Correlation should raise min vol.
        assert min_vol_correlated > min_vol_diverse, (
            f"Full-correlation scenario should have higher min vol than diversifiable base. "
            f"Got diverse={min_vol_diverse}, correlated={min_vol_correlated}"
        )

    def test_scenarios_with_scale_groups_override(self):
        """Fix 3: scale_groups on a scenario scales off-diagonal interactions in-group
        without requiring a full-matrix re-upload.

        Shape the problem so equities are correlated modestly in base; scale their
        correlations ×3 in a 'stress' scenario and confirm min-vol rises.
        """
        created = srv.model(action="create", approach="proportional")
        pid = created["problem_id"]
        srv.model(
            action="update", problem_id=pid,
            objectives=[
                {"name": "Return", "direction": "maximize", "aggregation": "sum"},
                {"name": "Volatility", "direction": "minimize", "aggregation": "quadratic"},
            ],
            options=[{"name": "E1"}, {"name": "E2"}, {"name": "E3"}, {"name": "Bond"}],
            scores=[
                # Similar returns on equities, lower on bond
                {"option": "E1", "objective": "Return", "value": 10},
                {"option": "E2", "objective": "Return", "value": 10},
                {"option": "E3", "objective": "Return", "value": 10},
                {"option": "Bond", "objective": "Return", "value": 4},
                # Dummy individual vols
                {"option": "E1", "objective": "Volatility", "value": 20},
                {"option": "E2", "objective": "Volatility", "value": 20},
                {"option": "E3", "objective": "Volatility", "value": 20},
                {"option": "Bond", "objective": "Volatility", "value": 5},
            ],
            constraints=[
                {"type": "cardinality", "min": 2, "max": 4},
                {"type": "max_allocation", "max": 50},
            ],
        )
        # Base matrix: modest equity-equity covariance; equity-bond near zero
        base_matrix = {
            "objective": "Volatility",
            "entries": {
                "E1": {"E1": 400.0, "E2": 100.0, "E3": 100.0, "Bond": 0.0},
                "E2": {"E1": 100.0, "E2": 400.0, "E3": 100.0, "Bond": 0.0},
                "E3": {"E1": 100.0, "E2": 100.0, "E3": 400.0, "Bond": 0.0},
                "Bond": {"E1": 0.0, "E2": 0.0, "E3": 0.0, "Bond": 25.0},
            },
        }
        srv.model(action="update", problem_id=pid, interaction_matrices=[base_matrix])

        # Scenario: scale equity-equity off-diagonals ×3 (stress). Base → factor 1.0 (no change).
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base"},
                {"name": "Stress", "interaction_matrix_overrides": [{
                    "objective": "Volatility",
                    "mode": "upsert",
                    "entries": {},  # don't change any specific cells
                    "scale_groups": [{"options": ["E1", "E2", "E3"], "factor": 3.0}],
                }]},
            ],
        })
        result = srv.solve(action="run_scenarios", problem_id=pid, mode="fast")
        assert "error" not in result

        base = srv.explore(action="tradeoffs", problem_id=pid, scenario="Base")
        stress = srv.explore(action="tradeoffs", problem_id=pid, scenario="Stress")

        min_vol_base = base["objective_ranges"]["Volatility"]["min"]
        min_vol_stress = stress["objective_ranges"]["Volatility"]["min"]

        # Higher equity-equity correlations → higher min-vol frontier for equity-heavy portfolios
        # The stress min-vol may still route through bonds, but cannot be lower than base.
        assert min_vol_stress >= min_vol_base - 0.01, (
            f"Stress scenario should not produce lower min vol than base. "
            f"Base={min_vol_base}, Stress={min_vol_stress}"
        )


class TestCuration:
    def test_curate_solution(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(
            action="curate", problem_id=pid, solution_id=1, custom_name="Pick A",
        )
        assert result.get("curated") is True
        assert result["custom_name"] == "Pick A"
        assert result["total_curated"] == 1

    def test_curate_duplicate(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        srv.explore(action="curate", problem_id=pid, solution_id=1, custom_name="Pick A")
        result = srv.explore(action="curate", problem_id=pid, solution_id=1, custom_name="Dup")
        assert "error" in result  # duplicate

    def test_list_curated(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        srv.explore(action="curate", problem_id=pid, solution_id=1, custom_name="First")
        srv.explore(action="curate", problem_id=pid, solution_id=2, custom_name="Second")
        result = srv.explore(action="curated", problem_id=pid)
        assert result["total_curated"] == 2
        assert all(c["in_current_frontier"] for c in result["curated_solutions"])
        assert "guidance_pointer" not in result  # a bare listing is navigation, not a handoff

    def test_export_curated_markdown(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        srv.explore(action="curate", problem_id=pid, solution_id=1, custom_name="First")
        srv.explore(action="curate", problem_id=pid, solution_id=2, custom_name="Second")
        result = srv.explore(action="curated", problem_id=pid, format="markdown")
        assert result["format"] == "markdown"
        assert result["total_curated"] == 2
        content = result["content"]
        assert "| name |" in content
        assert "content_signature" in content
        assert "First" in content and "Second" in content
        # B-P2: the export (handoff moment) carries the Stakeholder Writeup pointer.
        assert result["guidance_pointer"]["section"] == "Stakeholder Writeup & the Why-Triplet"

    def test_export_curated_csv(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        srv.explore(action="curate", problem_id=pid, solution_id=1, custom_name="First")
        result = srv.explore(action="curated", problem_id=pid, format="csv")
        assert result["format"] == "csv"
        content = result["content"]
        assert "name,content_signature" in content
        assert "First" in content

    def test_export_curated_empty(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="curated", problem_id=pid, format="markdown")
        assert result["total_curated"] == 0
        assert result["content"] == ""
        # An empty export is nothing to hand off — no Stakeholder Writeup pointer.
        assert "guidance_pointer" not in result

    def test_export_curated_bad_format(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        srv.explore(action="curate", problem_id=pid, solution_id=1)
        result = srv.explore(action="curated", problem_id=pid, format="xml")
        assert "error" in result

    def test_uncurate(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        curate_result = srv.explore(
            action="curate", problem_id=pid, solution_id=1, custom_name="Pick A",
        )
        sig = curate_result["content_signature"]
        result = srv.explore(action="curate", problem_id=pid, content_signature=sig, remove=True)
        assert result["total_curated"] == 0

    def test_rename_curated(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        curate_result = srv.explore(
            action="curate", problem_id=pid, solution_id=1, custom_name="Old",
        )
        sig = curate_result["content_signature"]
        result = srv.explore(
            action="curate", problem_id=pid,
            content_signature=sig, rename="New Name",
        )
        assert result["custom_name"] == "New Name"

    def test_compare_curated(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        r1 = srv.explore(action="curate", problem_id=pid, solution_id=1, custom_name="A")
        r2 = srv.explore(action="curate", problem_id=pid, solution_id=2, custom_name="B")
        result = srv.explore(
            action="compare", problem_id=pid,
            signatures=[r1["content_signature"], r2["content_signature"]],
        )
        assert "shared_options" in result
        assert "differentiating_options" in result
        assert len(result["solutions"]) == 2
        assert result["solutions"][0]["custom_name"] == "A"
        # Default compact: no selected_options or allocations per solution
        assert result["detail"] is False
        for sol in result["solutions"]:
            assert "selected_options" not in sol
            assert "allocations" not in sol

    def test_compare_curated_detail(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        r1 = srv.explore(action="curate", problem_id=pid, solution_id=1, custom_name="A")
        r2 = srv.explore(action="curate", problem_id=pid, solution_id=2, custom_name="B")
        result = srv.explore(
            action="compare", problem_id=pid,
            signatures=[r1["content_signature"], r2["content_signature"]],
            detail=True,
        )
        assert result["detail"] is True
        for sol in result["solutions"]:
            assert "selected_options" in sol

    def test_curated_survives_resolve(self):
        """Curated solutions persist across runs and track survival."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        r = srv.explore(action="curate", problem_id=pid, solution_id=1, custom_name="Survivor")
        sig = r["content_signature"]

        # Re-solve (same problem, same constraints — solution should survive)
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="curated", problem_id=pid)
        assert result["total_curated"] == 1
        assert result["curated_solutions"][0]["content_signature"] == sig

    def test_content_signatures_on_solutions(self):
        """Solutions have content_signature field after solving."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="solutions", problem_id=pid)
        for sol in result["solutions"]:
            assert sol["content_signature"] != ""
            assert len(sol["content_signature"]) == 12


class TestFeedbackLoop:
    """Feedback links to content_signature and attaches to curated solutions."""

    @staticmethod
    def _overlay_disjoint_exact_run(pid):
        """Stamp an exact overlay whose plans all differ from the base run's — so a
        signature resolved from the wrong frontier is provably wrong, not coincidentally
        right. Both frontiers index from 1, which is exactly the trap."""
        from engine.models import Run, Solution, _content_signature

        p = srv.store.load(pid)
        base_sigs = {s.content_signature for s in p.run.solutions}
        exact_sols = []
        sid = 0
        for opts in (["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"],
                     ["C", "D"], ["A", "B", "C"], ["A", "B", "D"], ["A", "C", "D"],
                     ["B", "C", "D"]):
            sig = _content_signature(opts)
            if sig in base_sigs:
                continue
            sid += 1
            exact_sols.append(Solution(
                solution_id=sid, selected_options=opts,
                objective_values={"Rev": 10.0 + sid, "Eff": 1.0 + sid},
                content_signature=sig,
            ))
        assert len(exact_sols) >= 3, "base run left too few distinct plans to overlay"
        p.exact_run = Run(solver="highs", solutions=exact_sols)
        srv.store.save(p)
        return exact_sols

    def test_feedback_solution_id_resolves_against_the_source_frontier(self):
        """Observed live (corpus capture, rev 9dceb63): dislikes recorded with
        solution_id + source="exact" were resolved against the BASE run — a different
        plan's signature — so `composition source="exact"` read n_disliked: 0 despite
        the recorded dislikes. Feedback must resolve solution_id on the same frontier
        every other explore action selects with `source`."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        exact_sols = self._overlay_disjoint_exact_run(pid)

        like = srv.explore(action="feedback", problem_id=pid, solution_id=1,
                           source="exact", rating=5, notes="cheap, clears every floor")
        d1 = srv.explore(action="feedback", problem_id=pid, solution_id=2,
                         source="exact", rating=1, notes="integration cannot absorb that")
        d2 = srv.explore(action="feedback", problem_id=pid, solution_id=3,
                         source="exact", rating=2)
        # The wire-level truth: each rating landed on the exact-run plan it named.
        assert like["content_signature"] == exact_sols[0].content_signature
        assert d1["content_signature"] == exact_sols[1].content_signature
        assert d2["content_signature"] == exact_sols[2].content_signature
        # And the response says which frontier the id resolved against.
        assert like["frontier_source"]["kind"] == "exact"

        fr = srv.explore(action="composition", problem_id=pid,
                         source="exact")["feedback_rules"]
        assert fr["n_liked"] == 1
        assert fr["n_disliked"] == 2

    def test_feedback_unknown_solution_id_errors_instead_of_recording(self):
        """A solution_id that resolves to nothing used to record a null-signature row —
        rule induction skips it and nothing can ever link to it. Decline in words, the
        way curate does."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="feedback", problem_id=pid, solution_id=999, rating=2)
        assert "error" in result and "recorded" not in result
        assert srv.explore(action="feedback", problem_id=pid, rating=3)["feedback_count"] == 1

    def test_a_dislike_is_not_a_selection(self):
        """Observed live: rating a plan 1 ("non-starter") echoed
        outcome.user_selected_solution as that plan. Selection requires endorsement."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        r = srv.explore(action="feedback", problem_id=pid, solution_id=1, rating=1,
                        notes="non-starter")
        assert r["outcome"]["user_selected_solution"] is None
        assert r["outcome"]["user_rating"] == 1
        r2 = srv.explore(action="feedback", problem_id=pid, solution_id=2, rating=5)
        assert r2["outcome"]["user_selected_solution"] == 2

    def test_feedback_computes_signature_from_solution_id(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(
            action="feedback", problem_id=pid, solution_id=1, rating=4, notes="Looks good",
        )
        assert result["recorded"] is True
        assert result["content_signature"] is not None
        assert len(result["content_signature"]) == 12

    def test_feedback_without_solution_has_no_signature(self):
        """Process-level feedback — a rating with no solution — stays serveable."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(
            action="feedback", problem_id=pid, rating=3, notes="General thoughts",
        )
        assert result["recorded"] is True
        assert result["content_signature"] is None

    def test_feedback_without_solution_or_rating_is_refused(self):
        """Observed live: no solution_id, no content_signature, no rating recorded a
        null-signature row and bumped feedback_count. Nothing attaches to it and rule
        induction skips it, so it's a can't-serve — the error names all three params."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="feedback", problem_id=pid, notes="Nice tool", stage="exploration")
        assert "error" in result and "recorded" not in result
        for param in ("solution_id", "content_signature", "rating"):
            assert param in result["error"]
        # And nothing was written.
        assert srv.explore(action="feedback", problem_id=pid, rating=3)["feedback_count"] == 1

    def test_notes_only_feedback_on_a_solution_is_allowed(self):
        """A rating-less annotation on a named solution is a legitimate shape — it has a
        signature, so it travels with the pin."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="feedback", problem_id=pid, solution_id=1,
                             notes="Board would ask about the vendor here")
        assert result["recorded"] is True
        assert result["content_signature"] is not None

    def test_feedback_attaches_to_curated_solution(self):
        """Feedback on a curated solution gets attached to its feedback list."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        # Curate first
        curate_r = srv.explore(
            action="curate", problem_id=pid, solution_id=1, custom_name="Fav",
        )
        sig = curate_r["content_signature"]
        # Then give feedback on same solution
        srv.explore(
            action="feedback", problem_id=pid, solution_id=1, rating=5, notes="Love it",
        )
        # Check curated solution has the feedback
        curated = srv.explore(action="curated", problem_id=pid)
        cs = curated["curated_solutions"][0]
        assert cs["feedback_count"] == 1
        assert cs["avg_rating"] == 5.0

    def test_existing_feedback_pulled_into_curation(self):
        """If feedback exists before curation, it's pulled in when curating."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        # Give feedback first
        srv.explore(
            action="feedback", problem_id=pid, solution_id=1, rating=4, notes="Promising",
        )
        # Then curate same solution
        srv.explore(
            action="curate", problem_id=pid, solution_id=1, custom_name="Promising Pick",
        )
        # Check feedback was pulled in
        curated = srv.explore(action="curated", problem_id=pid)
        cs = curated["curated_solutions"][0]
        assert cs["feedback_count"] == 1
        assert cs["avg_rating"] == 4.0

    def test_feedback_survives_rerun(self):
        """Feedback persists across re-optimization via content_signature."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        # Curate and give feedback
        curate_r = srv.explore(
            action="curate", problem_id=pid, solution_id=1, custom_name="Keeper",
        )
        sig = curate_r["content_signature"]
        srv.explore(
            action="feedback", problem_id=pid, solution_id=1, rating=5, notes="Best option",
        )
        # Re-solve
        srv.solve(action="run", problem_id=pid)
        # Curated solution still has its feedback
        curated = srv.explore(action="curated", problem_id=pid)
        cs = next(c for c in curated["curated_solutions"] if c["content_signature"] == sig)
        assert cs["feedback_count"] == 1
        assert cs["avg_rating"] == 5.0

    def test_feedback_by_content_signature_directly(self):
        """Feedback can be given via content_signature without solution_id."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        curate_r = srv.explore(
            action="curate", problem_id=pid, solution_id=1, custom_name="Direct",
        )
        sig = curate_r["content_signature"]
        # Give feedback by signature
        result = srv.explore(
            action="feedback", problem_id=pid, content_signature=sig,
            rating=3, notes="Reconsidering",
        )
        assert result["content_signature"] == sig
        curated = srv.explore(action="curated", problem_id=pid)
        assert curated["curated_solutions"][0]["feedback_count"] == 1

    def test_multiple_feedback_accumulates(self):
        """Multiple feedback entries accumulate on a curated solution."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        curate_r = srv.explore(
            action="curate", problem_id=pid, solution_id=1, custom_name="Multi",
        )
        sig = curate_r["content_signature"]
        srv.explore(action="feedback", problem_id=pid, solution_id=1, rating=3, stage="exploration")
        srv.explore(action="feedback", problem_id=pid, content_signature=sig, rating=5, stage="decision")
        curated = srv.explore(action="curated", problem_id=pid)
        cs = curated["curated_solutions"][0]
        assert cs["feedback_count"] == 2
        assert cs["avg_rating"] == 4.0  # (3+5)/2


class TestUnknownActions:
    def test_unknown_solve_action(self):
        pid = _build_solvable_problem()
        result = srv.solve(action="foobar", problem_id=pid)
        assert "error" in result

    def test_unknown_explore_action(self):
        pid = _build_solvable_problem()
        result = srv.explore(action="foobar", problem_id=pid)
        assert "error" in result


class TestMarginalAnalysis:
    def test_marginal_analysis_returns_pairs(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="marginal_analysis", problem_id=pid)
        assert "pairs" in result

    def test_marginal_analysis_no_run(self):
        created = srv.model(action="create")
        result = srv.explore(action="marginal_analysis", problem_id=created["problem_id"])
        assert "error" in result

    def test_marginal_analysis_has_rates(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="marginal_analysis", problem_id=pid)
        if result["pairs"]:
            pair = result["pairs"][0]
            assert "objectives" in pair
            assert "steepest_transitions" in pair  # summary mode (default)
            assert "visualization" in pair
        # detail=True returns full rates
        detail_result = srv.explore(action="marginal_analysis", problem_id=pid, detail=True)
        if detail_result["pairs"]:
            pair = detail_result["pairs"][0]
            assert "rates" in pair

    def test_small_frontier_detail_stays_inline(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.explore(action="marginal_analysis", problem_id=pid, detail=True)
        assert "full_result_path" not in result
        assert all("truncated" not in pair for pair in result["pairs"])

    def test_large_frontier_detail_writes_full_result_path(self):
        # One row per adjacent-solution transition per pair runs to tens of thousands of
        # tokens on a portfolio-scale frontier, so detail=true dumps to disk and keeps a
        # window inline — the claim the explore docstring makes for large-result actions.
        import json
        from pathlib import Path

        pid = srv.model(action="load", source="scarce_supply_rationing")["problem_id"]
        result = srv.explore(action="marginal_analysis", problem_id=pid, detail=True)
        capped = [p for p in result["pairs"] if "truncated" in p]
        assert capped, "a 300+ point frontier must exceed the per-pair detail cap"
        for pair in capped:
            assert len(pair["rates"]) == srv.EXPLORE_DETAIL_CAP
            assert pair["truncated"]["total_transitions"] > srv.EXPLORE_DETAIL_CAP
        on_disk = json.loads(Path(result["full_result_path"]).read_text())
        by_pair = {tuple(p["objectives"]): p for p in on_disk["pairs"]}
        for pair in capped:
            full = by_pair[tuple(pair["objectives"])]
            assert len(full["rates"]) == pair["truncated"]["total_transitions"]
            assert "truncated" not in full


# ─── Quadratic aggregation ───


class TestQuadraticAggregation:
    """Test quadratic (interaction matrix) aggregation end-to-end."""

    def test_quadratic_solve_produces_solutions(self):
        """Proportional solve with a quadratic objective uses the interaction matrix."""
        pid = srv.model(action="create", name="Quad Test", approach="proportional")["problem_id"]
        srv.model(action="update", problem_id=pid, objectives=[
            {"name": "Return", "direction": "maximize"},
            {"name": "Risk", "direction": "minimize", "aggregation": "quadratic"},
        ], options=["A", "B", "C"])
        # Scores: A high return/high risk, B medium, C low return/low risk
        srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "Return", "value": 20},
            {"option": "B", "objective": "Return", "value": 10},
            {"option": "C", "objective": "Return", "value": 5},
            # Individual risk scores (for display/marginal analysis)
            {"option": "A", "objective": "Risk", "value": 25},
            {"option": "B", "objective": "Risk", "value": 15},
            {"option": "C", "objective": "Risk", "value": 5},
        ])
        # Covariance matrix: A and C are negatively correlated (diversification benefit)
        cov = {
            "A": {"A": 625, "B": 200, "C": -100},  # var(A)=25^2=625
            "B": {"A": 200, "B": 225, "C": 50},     # var(B)=15^2=225
            "C": {"A": -100, "B": 50, "C": 25},     # var(C)=5^2=25
        }
        srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "entries": cov},
        ])
        result = srv.solve(action="run", problem_id=pid)
        assert "error" not in result
        assert result["solutions_found"] >= 2

    def test_quadratic_validation_missing_matrix(self):
        """Quadratic objective without interaction matrix fails validation."""
        pid = srv.model(action="create", name="Missing Matrix", approach="proportional")["problem_id"]
        srv.model(action="update", problem_id=pid, objectives=[
            {"name": "Return", "direction": "maximize"},
            {"name": "Risk", "direction": "minimize", "aggregation": "quadratic"},
        ], options=["A", "B", "C"])
        srv.model(action="update", problem_id=pid, scores=[
            {"option": o, "objective": obj, "value": 10}
            for o in ["A", "B", "C"] for obj in ["Return", "Risk"]
        ])
        result = srv.solve(action="validate", problem_id=pid)
        assert result["ready"] is False
        assert any("interaction matrix" in i["message"] for i in result["issues"])

    def test_quadratic_diversification_benefit(self):
        """With negative correlation, mixed portfolio should have lower risk than concentrated."""
        pid = srv.model(action="create", name="Diversification", approach="proportional")["problem_id"]
        srv.model(action="update", problem_id=pid, objectives=[
            {"name": "Return", "direction": "maximize"},
            {"name": "Risk", "direction": "minimize", "aggregation": "quadratic"},
        ], options=["Stock", "Bond", "Cash"])
        srv.model(action="update", problem_id=pid, scores=[
            {"option": "Stock", "objective": "Return", "value": 15},
            {"option": "Bond", "objective": "Return", "value": 5},
            {"option": "Cash", "objective": "Return", "value": 2},
            {"option": "Stock", "objective": "Risk", "value": 20},
            {"option": "Bond", "objective": "Risk", "value": 8},
            {"option": "Cash", "objective": "Risk", "value": 1},
        ])
        # Covariance: stock-bond negatively correlated
        cov = {
            "Stock": {"Stock": 400, "Bond": -80, "Cash": 0},
            "Bond": {"Stock": -80, "Bond": 64, "Cash": 0},
            "Cash": {"Stock": 0, "Bond": 0, "Cash": 1},
        }
        srv.model(action="update", problem_id=pid, interaction_matrices=[
            {"objective": "Risk", "entries": cov},
        ], constraints=[{"type": "cardinality", "min": 2, "max": 3}])
        result = srv.solve(action="run", problem_id=pid)
        assert "error" not in result
        # Get solutions and verify some have Risk below individual stock risk (20)
        solutions = srv.explore(action="solutions", problem_id=pid)
        risk_values = [s["objective_values"]["Risk"] for s in solutions["solutions"]]
        # At least one solution should show diversification benefit (risk < 20)
        assert min(risk_values) < 20, f"No diversification benefit found: {risk_values}"


# ─── Skill auto-injection ───


class TestCreateAcceptsObjectivesAndOptions:
    """Bug fix: model/create now accepts objectives and options params."""

    def test_create_with_objectives(self):
        result = srv.model(action="create", name="Test", objectives=[
            {"name": "Rev", "direction": "maximize"},
            {"name": "Eff", "direction": "minimize"},
        ])
        assert result["objectives"] == 2
        p = srv.model(action="get", problem_id=result["problem_id"], section="full")
        assert len(p["objectives"]) == 2

    def test_create_with_options(self):
        result = srv.model(action="create", name="Test", options=[
            {"name": "A"}, {"name": "B"}, {"name": "C"},
        ])
        assert result["options"] == 3
        p = srv.model(action="get", problem_id=result["problem_id"], section="full")
        assert len(p["options"]) == 3

    def test_create_with_objectives_and_options(self):
        result = srv.model(action="create", name="Test",
            objectives=[
                {"name": "Rev", "direction": "maximize"},
                {"name": "Eff", "direction": "minimize"},
            ],
            options=[{"name": "A"}, {"name": "B"}],
        )
        assert result["objectives"] == 2
        assert result["options"] == 2


class TestInjectSkillThrottle:
    """The once-per-problem throttle lives in _inject_skill itself (B-P0), so every call
    site inherits 'inject once, then it's in the agent's context' without re-implementing
    the guard. These pin that contract directly; the integration paths below exercise it
    through the real tool flow."""

    def test_injects_once_then_noops(self):
        pid = "throttle-unit-once"
        srv._reset_all_injections(pid)
        first = {}
        assert srv._inject_skill(first, "data_collection", "why", pid) is True
        assert first["_skill_guidance"]["skill"] == "data_collection"
        assert srv._was_injected(pid, "data_collection")
        # Second call for the same problem is a no-op — the core is already in context.
        second = {}
        assert srv._inject_skill(second, "data_collection", "again", pid) is False
        assert "_skill_guidance" not in second

    def test_reset_rearms_injection(self):
        pid = "throttle-unit-reset"
        srv._reset_all_injections(pid)
        assert srv._inject_skill({}, "optimization_strategy", "r", pid) is True
        assert srv._inject_skill({}, "optimization_strategy", "r", pid) is False
        # Explicit re-arm (e.g. an objectives/options shape change) re-enables injection.
        srv._reset_injection(pid, "optimization_strategy")
        rearmed = {}
        assert srv._inject_skill(rearmed, "optimization_strategy", "r", pid) is True
        assert rearmed["_skill_guidance"]["skill"] == "optimization_strategy"

    def test_throttle_is_per_skill(self):
        pid = "throttle-unit-perskill"
        srv._reset_all_injections(pid)
        assert srv._inject_skill({}, "data_collection", "a", pid) is True
        # A different skill for the same problem still injects.
        assert srv._inject_skill({}, "optimization_strategy", "b", pid) is True


class TestSolveGuidancePointer:
    """Solve responses carry a signal-keyed read pointer (B-P1a): a quality warning or
    structural diagnostics route to the matching playbook; a clean solve gets none, since
    the solution_interpreter core (injected on solve) already covers presentation."""

    def test_warning_quality_points_to_quality_signals(self):
        r = {"frontier_quality": {"status": "WARNING"}, "metrics": {"diagnostics": []}}
        guidance._attach_solve_guidance_pointer(r)
        assert r["guidance_pointer"]["skill"] == "solution_interpreter"
        assert r["guidance_pointer"]["section"] == "Frontier Quality and Completeness Signals"

    def test_poor_quality_points_to_quality_signals(self):
        r = {"frontier_quality": {"status": "POOR"}}
        guidance._attach_solve_guidance_pointer(r)
        assert r["guidance_pointer"]["section"] == "Frontier Quality and Completeness Signals"

    def test_actionable_diagnostics_point_to_diagnostic_patterns(self):
        r = {"frontier_quality": {"status": "GOOD"},
             "metrics": {"diagnostics": [{"pattern": "clustered_solutions", "severity": "warning"}]}}
        guidance._attach_solve_guidance_pointer(r)
        assert r["guidance_pointer"]["section"] == "Diagnostic Patterns"

    def test_info_only_diagnostics_get_no_pointer(self):
        # info patterns (binding_constraint, option_never_selected) are on most healthy
        # solves — pointing on them would fire on nearly every call. Gate to warning/error.
        r = {"frontier_quality": {"status": "GOOD"},
             "metrics": {"diagnostics": [{"pattern": "binding_constraint", "severity": "info"},
                                         {"pattern": "option_never_selected", "severity": "info"}]}}
        guidance._attach_solve_guidance_pointer(r)
        assert "guidance_pointer" not in r

    def test_quality_takes_priority_over_diagnostics(self):
        r = {"frontier_quality": {"status": "WARNING"},
             "metrics": {"diagnostics": [{"pattern": "p"}]}}
        guidance._attach_solve_guidance_pointer(r)
        assert r["guidance_pointer"]["section"] == "Frontier Quality and Completeness Signals"

    def test_exact_overlay_points_to_denoting_certification(self):
        # B-P2: an exact overlay (no more-urgent signal) routes to the denotation playbook.
        r = {"frontier_quality": {"status": "GOOD"}, "metrics": {"diagnostics": []},
             "solver_used": "highs"}
        guidance._attach_solve_guidance_pointer(r)
        assert r["guidance_pointer"]["section"] == "Denoting Certification — Prose & Tables"

    def test_quality_takes_precedence_over_denoting(self):
        r = {"frontier_quality": {"status": "WARNING"}, "solver_used": "highs"}
        guidance._attach_solve_guidance_pointer(r)
        assert r["guidance_pointer"]["section"] == "Frontier Quality and Completeness Signals"

    def test_nsga_solve_gets_no_denoting_pointer(self):
        r = {"frontier_quality": {"status": "GOOD"}, "metrics": {"diagnostics": []},
             "solver_used": "nsga"}
        guidance._attach_solve_guidance_pointer(r)
        assert "guidance_pointer" not in r

    def test_clean_solve_gets_no_pointer(self):
        r = {"frontier_quality": {"status": "GOOD"}, "metrics": {"diagnostics": []}}
        guidance._attach_solve_guidance_pointer(r)
        assert "guidance_pointer" not in r

    def test_error_and_scenario_shapes_pass_through(self):
        err = {"error": "boom", "frontier_quality": {"status": "POOR"}}
        guidance._attach_solve_guidance_pointer(err)
        assert "guidance_pointer" not in err
        scenario = {"scenarios_optimized": 3, "summary": {}}  # no top-level signals
        guidance._attach_solve_guidance_pointer(scenario)
        assert "guidance_pointer" not in scenario

    def test_cited_sections_resolve(self):
        # The cited sections must be fetchable via get_skill — guard against heading drift.
        txt = "".join(p.read_text() for p in guidance._skill_files("solution_interpreter"))
        for section in ("Frontier Quality and Completeness Signals", "Diagnostic Patterns"):
            assert guidance._extract_section(txt, section), section


class TestSkillInjectionOnCreate:
    def test_create_injects_data_collection(self):
        result = srv.model(action="create", name="Test")
        assert "_skill_guidance" in result
        assert result["_skill_guidance"]["skill"] == "data_collection"
        assert "content" in result["_skill_guidance"]
        assert len(result["_skill_guidance"]["content"]) > 100  # actual skill content


class TestSkillInjectionOnUpdate:
    def test_update_objectives_injects_data_collection_when_not_yet_injected(self):
        """If data_collection wasn't already injected (e.g. tracking cleared),
        an objectives update should inject it."""
        created = srv.model(action="create")
        pid = created["problem_id"]
        # Create already injected data_collection — clear it to test update path
        srv._injected_skills[pid].discard("data_collection")
        result = srv.model(action="update", problem_id=pid, objectives=[
            {"name": "Rev", "direction": "maximize"},
            {"name": "Eff", "direction": "minimize"},
        ])
        assert "_skill_guidance" in result
        assert result["_skill_guidance"]["skill"] == "data_collection"

    def test_update_options_injects_data_collection_when_not_yet_injected(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv._injected_skills[pid].discard("data_collection")
        result = srv.model(action="update", problem_id=pid, options=[
            {"name": "A"}, {"name": "B"}, {"name": "C"},
        ])
        assert "_skill_guidance" in result
        assert result["_skill_guidance"]["skill"] == "data_collection"

    def test_create_already_injects_so_update_does_not_reinject(self):
        """model/create injects data_collection, so a subsequent update with
        objectives/options should NOT re-inject."""
        created = srv.model(action="create")
        pid = created["problem_id"]
        assert created["_skill_guidance"]["skill"] == "data_collection"
        # Update should NOT re-inject
        result = srv.model(action="update", problem_id=pid, objectives=[
            {"name": "Rev", "direction": "maximize"},
            {"name": "Eff", "direction": "minimize"},
        ])
        assert "_skill_guidance" not in result

    def test_scores_complete_injects_optimization_strategy(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
            objectives=[{"name": "Rev", "direction": "maximize"},
                        {"name": "Eff", "direction": "minimize"}],
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}])
        # Partial scores — no injection
        result = srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "Rev", "value": 5},
            {"option": "A", "objective": "Eff", "value": 3},
        ])
        assert "_skill_guidance" not in result

        # Complete scores — should inject optimization_strategy
        result = srv.model(action="update", problem_id=pid, scores=[
            {"option": "B", "objective": "Rev", "value": 7},
            {"option": "B", "objective": "Eff", "value": 2},
            {"option": "C", "objective": "Rev", "value": 9},
            {"option": "C", "objective": "Eff", "value": 6},
        ])
        assert "_skill_guidance" in result
        assert result["_skill_guidance"]["skill"] == "optimization_strategy"

    def test_optimization_strategy_not_reinjected_on_more_scores(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
            objectives=[{"name": "Rev", "direction": "maximize"},
                        {"name": "Eff", "direction": "minimize"}],
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            scores=[
                {"option": "A", "objective": "Rev", "value": 5},
                {"option": "A", "objective": "Eff", "value": 3},
                {"option": "B", "objective": "Rev", "value": 7},
                {"option": "B", "objective": "Eff", "value": 2},
                {"option": "C", "objective": "Rev", "value": 9},
                {"option": "C", "objective": "Eff", "value": 6},
            ])
        # Upsert a score — should NOT re-inject
        result = srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "Rev", "value": 99},
        ])
        assert "_skill_guidance" not in result

    def test_structural_change_resets_optimization_strategy(self):
        """After objectives change, optimization_strategy should re-fire when scores hit 100%."""
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
            objectives=[{"name": "Rev", "direction": "maximize"},
                        {"name": "Eff", "direction": "minimize"}],
            options=[{"name": "A"}, {"name": "B"}],
            scores=[
                {"option": "A", "objective": "Rev", "value": 5},
                {"option": "A", "objective": "Eff", "value": 3},
                {"option": "B", "objective": "Rev", "value": 7},
                {"option": "B", "objective": "Eff", "value": 2},
            ])
        # optimization_strategy was injected at 100%. Now change objectives.
        srv.model(action="update", problem_id=pid, objectives=[
            {"name": "Rev", "direction": "maximize"},
            {"name": "Risk", "direction": "minimize"},
        ])
        # Re-score to 100%
        result = srv.model(action="update", problem_id=pid, scores=[
            {"option": "A", "objective": "Rev", "value": 5},
            {"option": "A", "objective": "Risk", "value": 3},
            {"option": "B", "objective": "Rev", "value": 7},
            {"option": "B", "objective": "Risk", "value": 2},
        ])
        assert "_skill_guidance" in result
        assert result["_skill_guidance"]["skill"] == "optimization_strategy"

    def test_non_shape_changes_do_not_reset_optimization_strategy(self):
        """Adding a constraint, interaction matrix, or scenario_config between scores→100%
        and solve/validate should NOT re-fire optimization_strategy. Only objective/option
        shape changes invalidate the methodology guidance."""
        created = srv.model(action="create")
        pid = created["problem_id"]
        srv.model(action="update", problem_id=pid,
            objectives=[{"name": "Rev", "direction": "maximize"},
                        {"name": "Eff", "direction": "minimize"}],
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            scores=[
                {"option": "A", "objective": "Rev", "value": 5},
                {"option": "A", "objective": "Eff", "value": 3},
                {"option": "B", "objective": "Rev", "value": 7},
                {"option": "B", "objective": "Eff", "value": 2},
                {"option": "C", "objective": "Rev", "value": 9},
                {"option": "C", "objective": "Eff", "value": 6},
            ])  # optimization_strategy injected here (scores → 100%)

        # Add a constraint — must NOT re-fire
        result = srv.model(action="update", problem_id=pid,
            constraints=[{"type": "cardinality", "min": 1, "max": 2}])
        assert "_skill_guidance" not in result

        # Add a reference point — must NOT re-fire
        result = srv.model(action="update", problem_id=pid,
            reference_points=[{"type": "baseline", "objective_values": {"Rev": 5, "Eff": 3}}])
        assert "_skill_guidance" not in result

        # Validate — must NOT re-fire either (already injected)
        result = srv.solve(action="validate", problem_id=pid)
        assert "_skill_guidance" not in result

    def test_scores_and_objectives_same_call_prioritizes_data_collection(self):
        """When objectives AND scores arrive in the same call and data_collection
        hasn't been injected yet, data_collection takes priority over optimization_strategy."""
        created = srv.model(action="create")
        pid = created["problem_id"]
        # Clear the data_collection injection from create so we can test the elif logic
        srv._injected_skills[pid].discard("data_collection")
        result = srv.model(action="update", problem_id=pid,
            objectives=[{"name": "Rev", "direction": "maximize"},
                        {"name": "Eff", "direction": "minimize"}],
            options=[{"name": "A"}, {"name": "B"}],
            scores=[
                {"option": "A", "objective": "Rev", "value": 5},
                {"option": "A", "objective": "Eff", "value": 3},
                {"option": "B", "objective": "Rev", "value": 7},
                {"option": "B", "objective": "Eff", "value": 2},
            ])
        # data_collection takes priority because objectives were added in same call
        assert "_skill_guidance" in result
        assert result["_skill_guidance"]["skill"] == "data_collection"

    def test_scores_and_objectives_same_call_when_data_collection_already_injected(self):
        """When data_collection was already injected (from create), a combined
        objectives+scores update that hits 100% should inject optimization_strategy."""
        created = srv.model(action="create")
        pid = created["problem_id"]
        # data_collection already injected by create
        result = srv.model(action="update", problem_id=pid,
            objectives=[{"name": "Rev", "direction": "maximize"},
                        {"name": "Eff", "direction": "minimize"}],
            options=[{"name": "A"}, {"name": "B"}],
            scores=[
                {"option": "A", "objective": "Rev", "value": 5},
                {"option": "A", "objective": "Eff", "value": 3},
                {"option": "B", "objective": "Rev", "value": 7},
                {"option": "B", "objective": "Eff", "value": 2},
            ])
        # data_collection already marked → elif falls through to optimization_strategy
        assert "_skill_guidance" in result
        assert result["_skill_guidance"]["skill"] == "optimization_strategy"


class TestFramingSurfacing:
    """problem_framing isn't auto-injected, so B-P1b gives it signal-driven surfacing:
    a reframe injection when a model is scored with <2 objectives, and a validate-time
    pointer to the framing checkpoint when the model is structurally thin."""

    def _create(self):
        return srv.model(action="create", name="F")["problem_id"]

    def test_scoring_thin_model_injects_problem_framing(self):
        pid = self._create()
        # One objective + options + scores: scoring before there's a real tradeoff.
        r = srv.model(action="update", problem_id=pid,
            objectives=[{"name": "Rev", "direction": "maximize"}],
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            scores=[{"option": o, "objective": "Rev", "value": v}
                    for o, v in [("A", 5), ("B", 7), ("C", 9)]])
        assert r["_skill_guidance"]["skill"] == "problem_framing"

    def test_scoring_complete_two_objective_model_does_not_inject_framing(self):
        pid = self._create()
        r = srv.model(action="update", problem_id=pid,
            objectives=[{"name": "Rev", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            scores=[{"option": o, "objective": ob, "value": v}
                    for o, ob, v in [("A", "Rev", 5), ("A", "Cost", 3), ("B", "Rev", 7),
                                     ("B", "Cost", 5), ("C", "Rev", 9), ("C", "Cost", 6)]])
        assert r["_skill_guidance"]["skill"] == "optimization_strategy"

    def test_validate_thin_model_points_to_framing_checkpoint(self):
        pid = self._create()
        srv.model(action="update", problem_id=pid,
            objectives=[{"name": "Rev", "direction": "maximize"}],
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}])
        r = srv.solve(action="validate", problem_id=pid)
        assert r["guidance_pointer"]["skill"] == "problem_framing"
        assert r["guidance_pointer"]["section"] == "Formalization Checkpoint"

    def test_validate_ready_model_has_no_framing_pointer(self):
        pid = self._create()
        srv.model(action="update", problem_id=pid,
            objectives=[{"name": "Rev", "direction": "maximize"},
                        {"name": "Cost", "direction": "minimize"}],
            options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            scores=[{"option": o, "objective": ob, "value": v}
                    for o, ob, v in [("A", "Rev", 5), ("A", "Cost", 3), ("B", "Rev", 7),
                                     ("B", "Cost", 5), ("C", "Rev", 9), ("C", "Cost", 6)]])
        r = srv.solve(action="validate", problem_id=pid)
        assert "guidance_pointer" not in r


class TestSkillInjectionOnSolve:
    def test_solve_run_injects_solution_interpreter(self):
        pid = _build_solvable_problem()
        result = srv.solve(action="run", problem_id=pid)
        assert "_skill_guidance" in result
        assert result["_skill_guidance"]["skill"] == "solution_interpreter"

    def test_solve_rerun_does_not_reinject_solution_interpreter(self):
        """solution_interpreter injects once per problem; a re-solve without
        structural change does not re-inject (throttled)."""
        pid = _build_solvable_problem()
        result1 = srv.solve(action="run", problem_id=pid)
        assert result1["_skill_guidance"]["skill"] == "solution_interpreter"
        result2 = srv.solve(action="run", problem_id=pid)
        assert "_skill_guidance" not in result2

    def test_structural_update_rearms_solution_interpreter(self):
        """Structural model edits reset the solution_interpreter flag so the
        next solve re-injects fresh guidance."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        assert srv._was_injected(pid, "solution_interpreter")
        # Structural edit (constraints replacement) should clear the flag.
        srv.model(
            action="update",
            problem_id=pid,
            constraints=[{"type": "cardinality", "min": 1, "max": 2}],
        )
        assert not srv._was_injected(pid, "solution_interpreter")
        result = srv.solve(action="run", problem_id=pid)
        assert result["_skill_guidance"]["skill"] == "solution_interpreter"

    def test_scores_only_update_does_not_rearm_solution_interpreter(self):
        """Score-value refinements mark results stale but keep interpretation guidance
        armed: the tweak→re-solve iteration loop must not re-pay the full core
        (docstring contract; post-streamlining user test, finding P2)."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        assert srv._was_injected(pid, "solution_interpreter")
        updated = srv.model(action="update", problem_id=pid,
                            scores=[{"option": "A", "objective": "Rev", "value": 9}])
        assert updated["status"]["results_stale"] is True       # staleness still marked
        assert srv._was_injected(pid, "solution_interpreter")   # but no re-arm
        result = srv.solve(action="run", problem_id=pid)
        assert "_skill_guidance" not in result                  # and no re-inject

    def test_solve_delivery_does_not_rearm_optimization_strategy(self):
        """A completed solve must not re-arm optimization_strategy: before the fix, a
        post-solve score tweak at a still-complete matrix re-fired the full ~5.5k-token
        core every iteration cycle (user test, finding P2)."""
        pid = _build_solvable_problem()                         # injects optimization_strategy at 100%
        assert srv._was_injected(pid, "optimization_strategy")
        srv.solve(action="run", problem_id=pid)
        assert srv._was_injected(pid, "optimization_strategy")  # delivery left it marked
        tweaked = srv.model(action="update", problem_id=pid,
                            scores=[{"option": "B", "objective": "Eff", "value": 4}])
        assert "_skill_guidance" not in tweaked                 # no re-fire after solve

    def test_solve_validation_failure_no_injection(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        result = srv.solve(action="run", problem_id=pid)
        assert "_skill_guidance" not in result

    def test_validate_ready_injects_optimization_strategy(self):
        pid = _build_solvable_problem()
        # _build_solvable_problem already triggers optimization_strategy via
        # model/update at 100% scores. Clear it to test the validate path.
        srv._injected_skills[pid].discard("optimization_strategy")
        result = srv.solve(action="validate", problem_id=pid)
        assert result["ready"] is True
        assert "_skill_guidance" in result
        assert result["_skill_guidance"]["skill"] == "optimization_strategy"

    def test_validate_not_ready_no_injection(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        result = srv.solve(action="validate", problem_id=pid)
        assert result["ready"] is False
        assert "_skill_guidance" not in result

    def test_validate_does_not_reinject_if_already_injected(self):
        """If optimization_strategy was already injected (e.g. from 100% scores),
        validate should NOT re-inject."""
        pid = _build_solvable_problem()
        # optimization_strategy already injected by _build_solvable_problem
        result = srv.solve(action="validate", problem_id=pid)
        assert result["ready"] is True
        assert "_skill_guidance" not in result

    def test_solve_does_not_reset_optimization_strategy(self):
        """A completed solve keeps optimization_strategy marked — post-solve validate
        or score tweaks must not re-pay the core. Only an objectives/options shape
        change re-arms it (docstring contract; user test, finding P2)."""
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        result = srv.solve(action="validate", problem_id=pid)
        assert result["ready"] is True
        assert "_skill_guidance" not in result
        # A shape change still re-arms via the model-update path.
        srv.model(action="update", problem_id=pid,
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"},
                           {"name": "E"}])
        assert not srv._was_injected(pid, "optimization_strategy")

    def test_run_scenarios_injects_solution_interpreter(self):
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base", "probability": 0.5, "score_overrides": []},
                {"name": "Alt", "probability": 0.5, "score_overrides": [
                    {"option": "A", "objective": "Rev", "value": 1},
                    {"option": "C", "objective": "Rev", "value": 20},
                ]},
            ],
        })
        result = srv.solve(action="run_scenarios", problem_id=pid)
        assert "_skill_guidance" in result
        assert result["_skill_guidance"]["skill"] == "solution_interpreter"


class TestSkillInjectionOnDelete:
    def test_delete_clears_injection_tracking(self):
        created = srv.model(action="create")
        pid = created["problem_id"]
        # Create injects data_collection, marking it as injected
        assert srv._was_injected(pid, "data_collection")
        srv.model(action="delete", problem_id=pid)
        assert not srv._was_injected(pid, "data_collection")


class TestModelLoadSave:
    @pytest.fixture
    def tmp_saved(self, tmp_path, monkeypatch):
        """Redirect saves to a temp library; leave examples/ pointed at the repo."""
        monkeypatch.setenv("FRONTIER_SAVED_DIR", str(tmp_path / "saved"))
        return tmp_path / "saved"

    def _scored_problem(self) -> str:
        pid = srv.model(action="create", name="Roundtrip")["problem_id"]
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Cost", "direction": "minimize"}],
                  options=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
                  scores=[{"option": o, "objective": obj, "value": v}
                          for o, (r, c) in {"A": (8, 3), "B": (5, 2), "C": (9, 5)}.items()
                          for obj, v in (("Rev", r), ("Cost", c))])
        return pid

    def test_save_then_load_round_trip(self, tmp_saved):
        pid = self._scored_problem()
        saved = srv.model(action="save", problem_id=pid, save_as="my_decision")
        assert saved["saved"] is True
        assert saved["name"] == "my_decision"
        assert set(saved["files"]) == {"problem", "scores"}  # unsolved → no solutions

        loaded = srv.model(action="load", source="my_decision")
        assert loaded["problem_id"] != pid  # fresh id
        assert loaded["status"]["objectives"] == 2
        assert loaded["status"]["options"] == 3
        assert loaded["status"]["scores_complete"] == 1.0

    def test_save_name_defaults_to_slug_of_problem_name(self, tmp_saved):
        pid = self._scored_problem()
        saved = srv.model(action="save", problem_id=pid)
        assert saved["name"] == "Roundtrip"

    def test_save_includes_solutions_after_solve(self, tmp_saved):
        pid = self._scored_problem()
        srv.solve(action="run", problem_id=pid)
        saved = srv.model(action="save", problem_id=pid, save_as="solved")
        assert saved["includes_solutions"] is True
        assert "solutions" in saved["files"]

        loaded = srv.model(action="load", source="solved")
        assert loaded["status"]["has_run"] is True
        assert loaded["status"]["results_stale"] is False
        # A loaded, solved problem is ready to explore.
        assert loaded["_skill_guidance"]["skill"] == "solution_interpreter"

    def test_load_unsolved_injects_optimization_strategy(self, tmp_saved):
        pid = self._scored_problem()
        srv.model(action="save", problem_id=pid, save_as="unsolved")
        loaded = srv.model(action="load", source="unsolved")
        assert loaded["_skill_guidance"]["skill"] == "optimization_strategy"

    def test_load_bundled_example_keeps_scenarios(self, tmp_saved):
        loaded = srv.model(action="load", source="investment_portfolio")
        assert loaded["status"]["scenarios"] == 3
        assert loaded["status"]["interaction_matrices"] == 1

    def test_load_names_the_library_and_flags_a_shadowing_save(self, tmp_saved):
        """saved/ shadows a bundled example of the same name (resolve_source) — the load
        response must say WHICH library served the bundle, and call out the shadow, or a
        stale saved copy silently stands in for the pristine example."""
        example = srv.model(action="load", source="investment_portfolio")
        assert example["library"] == "examples"
        assert "note" not in example

        pid = self._scored_problem()
        srv.model(action="save", problem_id=pid, save_as="investment_portfolio")
        shadowing = srv.model(action="load", source="investment_portfolio")
        assert shadowing["library"] == "saved"
        assert "shadows" in shadowing["note"]
        # The shadow really served the user's problem, not the example.
        assert shadowing["status"]["options"] == 3

    def test_load_without_source_lists_available(self, tmp_saved):
        result = srv.model(action="load")
        assert "error" in result
        assert "investment_portfolio" in result["available"]["examples"]

    def test_load_unknown_name_errors_with_available(self, tmp_saved):
        result = srv.model(action="load", source="nope_not_here")
        assert "error" in result
        assert "available" in result

    def test_save_missing_problem_errors(self, tmp_saved):
        result = srv.model(action="save", problem_id="does-not-exist")
        assert "error" in result

    def test_load_rejects_unsafe_source(self, tmp_saved):
        result = srv.model(action="load", source="../../etc/passwd")
        assert "error" in result and "Invalid" in result["error"]


class TestCertify:
    """The explore `certify` action: audit an NSGA frontier against an exact-solver run through
    the full MCP path (build → solve NSGA → solve HiGHS → certify)."""

    @staticmethod
    def _run(**kw):
        """solve run → run_id, tolerating a backgrounded solve.

        `solve run` returns the result inline when it finishes inside the wait budget and a
        {status:"running", job_id} handle otherwise, so which branch you get depends on how
        fast the machine is. Subscripting ["run_id"] therefore passes locally and fails on a
        slower CI runner with KeyError — a race, not a behavior change. These exact HiGHS
        solves sit right at the boundary (~10s of the inline budget in CI timings), so poll
        when handed a handle instead of assuming the fast path.
        """
        r = srv.solve(action="run", **kw)
        while r.get("status") == "running" and r.get("job_id"):
            r = srv.solve(action="status", job_id=r["job_id"], wait_seconds=10)
        assert "run_id" in r, f"solve produced no run: {r}"
        return r["run_id"]

    def _binary_pid(self):
        created = srv.model(action="create", name="Certify")
        pid = created["problem_id"]
        names = ["A", "B", "C", "D", "E", "F"]
        table = {"NPV": [9, 7, 8, 4, 6, 5], "Cost": [5, 3, 6, 2, 4, 3], "Fit": [8, 6, 5, 9, 7, 4]}
        scores = [{"option": n, "objective": o, "value": table[o][i]}
                  for i, n in enumerate(names) for o in table]
        srv.model(action="update", problem_id=pid, approach="binary",
                  objectives=[{"name": "NPV", "direction": "maximize"},
                              {"name": "Cost", "direction": "minimize"},
                              {"name": "Fit", "direction": "maximize"}],
                  options=[{"name": n} for n in names], scores=scores,
                  constraints=[{"type": "cardinality", "min": 2, "max": 4}])
        return pid

    def test_certify_default_overlay(self):
        """The primary flow: solve NSGA → solve exact (stored as the exact_run overlay) → certify
        with NO run_ids, auditing `run` against `exact_run`."""
        pytest.importorskip("highspy")
        pid = self._binary_pid()
        srv.solve(action="run", problem_id=pid, seed=42)                       # → p.run (NSGA)
        srv.solve(action="run", problem_id=pid, seed=42, solver="highs")       # → p.exact_run (overlay)
        cert = srv.explore(action="certify", problem_id=pid)                   # no run_ids
        assert "error" not in cert
        assert cert["exact_solver"] == "highs"
        assert cert["invariant"]["holds"] is True            # MILP: integer, never rounding-dominated
        assert "nsga_dominated_by_exact" in cert["dominance_audit"]
        assert set(cert["corner_sharpening"]) == {"NPV", "Cost", "Fit"}
        assert isinstance(cert["recommendation"], str) and cert["recommendation"]

    def test_certify_needs_an_exact_overlay(self):
        """No exact_run yet (only an NSGA run) → certify (no run_ids) asks for the exact solve."""
        pid = self._binary_pid()
        srv.solve(action="run", problem_id=pid, seed=1)
        r = srv.explore(action="certify", problem_id=pid)
        assert "error" in r and "exact overlay" in r["error"]

    def test_certify_explicit_run_ids(self):
        """run_ids overrides the default, pulling the NSGA run and the exact overlay by id."""
        pytest.importorskip("highspy")
        pid = self._binary_pid()
        nsga = self._run(problem_id=pid, seed=42)
        exact = self._run(problem_id=pid, seed=42, solver="highs")
        cert = srv.explore(action="certify", problem_id=pid, run_ids=[nsga, exact])
        assert "error" not in cert and cert["exact_solver"] == "highs"
        assert set(cert["corner_sharpening"]) == {"NPV", "Cost", "Fit"}

    def test_certify_order_free(self):
        """The exact run is detected by its solver, so run_ids order does not matter."""
        pytest.importorskip("highspy")
        pid = self._binary_pid()
        nsga = self._run(problem_id=pid, seed=1)
        exact = self._run(problem_id=pid, seed=1, solver="highs")
        a = srv.explore(action="certify", problem_id=pid, run_ids=[nsga, exact])
        b = srv.explore(action="certify", problem_id=pid, run_ids=[exact, nsga])
        assert a["dominance_audit"] == b["dominance_audit"]

    def test_certify_requires_two_runs(self):
        pid = self._binary_pid()
        one = self._run(problem_id=pid, seed=1)
        r = srv.explore(action="certify", problem_id=pid, run_ids=[one])
        assert "error" in r and "exactly 2" in r["error"]

    def test_certify_needs_one_exact_one_nsga(self):
        """Two NSGA runs (no exact) is rejected — certify is an exact-vs-heuristic audit."""
        pid = self._binary_pid()
        r1 = self._run(problem_id=pid, seed=1)
        r2 = self._run(problem_id=pid, seed=2)
        r = srv.explore(action="certify", problem_id=pid, run_ids=[r1, r2])
        assert "error" in r and "one NSGA run and one exact" in r["error"]

    def test_certify_rejects_signatures_scope(self):
        """certify is frontier-level, not per-solution — passing signatures (solution scope)
        must redirect, not be silently ignored. Surfaced by a model eval: a capable agent
        passed the curated finalists' signatures and got the whole-frontier audit anyway."""
        pid = self._binary_pid()
        r = srv.explore(action="certify", problem_id=pid, signatures=["abc123", "def456"])
        assert "error" in r
        assert "frontier-level" in r["error"] and "compare" in r["error"]

    def test_certify_rejects_solution_ids_scope(self):
        """The same guard covers the sibling solution-scope param (solution_ids)."""
        pid = self._binary_pid()
        r = srv.explore(action="certify", problem_id=pid, solution_ids=[1, 2])
        assert "error" in r and "frontier-level" in r["error"]


class TestAuditPropertyShape:
    """explore audit — a malformed audit_property names the shape, not a cryptic
    discriminated-union error. Surfaced by a model eval: a capable agent needed 3 tries
    (bare string → dict without `type` → correct) before landing the objective_bound shape."""

    def _pid(self):
        return srv.model(action="create", name="A",
                         objectives=[{"name": "V", "direction": "maximize"}],
                         options=["A", "B", "C"])["problem_id"]

    def test_bare_string_names_shape(self):
        pid = self._pid()
        r = srv.explore(action="audit", problem_id=pid, audit_property="Risk <= 3000")
        assert "error" in r
        assert "objective_bound" in r["error"] and '"value"' in r["error"]

    def test_missing_type_names_vocabulary(self):
        # The exact trap: a constraint-shaped dict with no `type` discriminator.
        pid = self._pid()
        r = srv.explore(action="audit", problem_id=pid,
                        audit_property={"objective": "V", "operator": "<=", "threshold": 3000})
        assert "error" in r
        assert "type" in r["error"] and "objective_bound" in r["error"]

    def test_unknown_type_names_vocabulary(self):
        pid = self._pid()
        r = srv.explore(action="audit", problem_id=pid,
                        audit_property={"type": "not_a_constraint"})
        assert "error" in r and "objective_bound" in r["error"]

    def test_valid_type_bad_fields_keeps_per_field_detail(self):
        # Valid `type`, wrong field names/values → precise per-field detail + shape reminder.
        pid = self._pid()
        r = srv.explore(action="audit", problem_id=pid,
                        audit_property={"type": "objective_bound", "objective": "V",
                                        "operator": "<=", "threshold": 3000})
        assert "error" in r
        assert ("operator" in r["error"] or "value" in r["error"])  # precise field detail
        assert "objective_bound" in r["error"]                       # + shape reminder


class TestDecisionGuidancePointers:
    """A2 — every explore/decision action names the skill section that governs reading it
    plus the get_skill() re-fetch path, so guidance survives a compacted long session."""

    def _solved_pid(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        return pid

    def _assert_pointer(self, result, section):
        gp = result.get("guidance_pointer")
        assert gp is not None, f"expected a guidance_pointer, got keys {list(result)}"
        assert gp["skill"] == "solution_interpreter"
        assert gp["section"] == section
        # The note must carry the exact section-fetch path — the compaction-survivor trigger.
        assert f"get_skill('solution_interpreter', section='{section}')" in gp["note"]

    def test_tradeoffs_points_to_presentation_order(self):
        pid = self._solved_pid()
        result = srv.explore(action="tradeoffs", problem_id=pid)
        self._assert_pointer(
            result, "Presentation Order: Extremes → Balanced → Inflection → Risk → Preference")

    def test_tradeoffs_flagged_redundancy_names_its_playbook(self):
        # When the payload flags a redundant/dependent pair, the pointer note must also name
        # the Objective Redundancy section (it lives in references, not the injected core).
        # Unit-level: drive _attach_guidance_pointer with synthetic classifications.
        from mcp_server.guidance import _attach_guidance_pointer
        flagged = _attach_guidance_pointer(
            {"objective_redundancy": [{"classification": "linear_redundant"}]}, "tradeoffs")
        assert ("get_skill('solution_interpreter', section='Objective Redundancy')"
                in flagged["guidance_pointer"]["note"])
        # strong_tradeoff / independent are healthy — no extra fetch demanded.
        clean = _attach_guidance_pointer(
            {"objective_redundancy": [{"classification": "strong_tradeoff"},
                                      {"classification": "independent"}]}, "tradeoffs")
        assert "Objective Redundancy" not in clean["guidance_pointer"]["note"]

    def test_compare_points_to_differentiating_options(self):
        pid = self._solved_pid()
        ids = [s["solution_id"] for s in srv.explore(action="solutions", problem_id=pid)["solutions"]]
        result = srv.explore(action="compare", problem_id=pid, solution_ids=ids[:2])
        self._assert_pointer(result, "Differentiating Options")

    def test_marginal_analysis_points_to_its_section(self):
        pid = self._solved_pid()
        result = srv.explore(action="marginal_analysis", problem_id=pid)
        self._assert_pointer(result, "Marginal Analysis Interpretation")

    def test_curate_points_to_solution_curation(self):
        pid = self._solved_pid()
        sid = srv.explore(action="solutions", problem_id=pid)["solutions"][0]["solution_id"]
        result = srv.explore(action="curate", problem_id=pid, solution_id=sid, custom_name="Pick")
        self._assert_pointer(result, "Solution Curation")

    def test_sensitivity_fallback_points_to_binding_analysis(self):
        """No exact duals on a binary NSGA run → frontier-inferred fallback, so the pointer
        must cite Binding Analysis (not Exact Sensitivity) to match the output."""
        pid = self._solved_pid()
        result = srv.explore(action="sensitivity", problem_id=pid)
        assert result["source"] == "frontier_inferred"
        self._assert_pointer(result, "Binding Analysis")
        # And no duals-language caption on a duals-less response — the server's
        # "duals are local" next_steps is solver-exact only.
        assert "duals" not in result.get("next_steps", "").lower()

    def test_scenario_results_points_to_its_section(self):
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base", "probability": 0.5, "score_overrides": []},
                {"name": "Alt", "probability": 0.5, "score_overrides": [
                    {"option": "A", "objective": "Rev", "value": 1}]},
            ],
        })
        srv.solve(action="run_scenarios", problem_id=pid)
        result = srv.explore(action="scenario_results", problem_id=pid)
        self._assert_pointer(result, "Scenario Results Presentation")

    def test_compare_curated_points_to_differentiating_options(self):
        pid = self._solved_pid()
        sols = srv.explore(action="solutions", problem_id=pid)["solutions"]
        for s, nm in zip(sols[:2], ("A", "B")):
            srv.explore(action="curate", problem_id=pid, solution_id=s["solution_id"], custom_name=nm)
        sigs = [c["content_signature"]
                for c in srv.explore(action="curated", problem_id=pid)["curated_solutions"]]
        result = srv.explore(action="compare", problem_id=pid, signatures=sigs[:2])
        self._assert_pointer(result, "Differentiating Options")

    def test_certify_points_to_reading_the_certificate(self):
        """certify is the only branch that ALSO injects the full skill — assert the section
        pointer still attaches alongside that injection."""
        from solvers import available_solvers
        if not available_solvers().get("highs"):
            import pytest
            pytest.skip("highs backend not installed")
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=1)
        srv.solve(action="run", problem_id=pid, solver="highs", seed=1)
        result = srv.explore(action="certify", problem_id=pid)
        assert "error" not in result, result
        self._assert_pointer(result, "Reading the Certificate (explore certify)")

    def test_non_decision_action_gets_no_pointer(self):
        """`solutions` is navigation, not a decision step — no pointer."""
        pid = self._solved_pid()
        result = srv.explore(action="solutions", problem_id=pid)
        assert "guidance_pointer" not in result

    def test_helper_is_noop_on_error_and_unknown_action(self):
        assert "guidance_pointer" not in srv._attach_guidance_pointer({"error": "x"}, "tradeoffs")
        assert "guidance_pointer" not in srv._attach_guidance_pointer({"ok": 1}, "list")

    def test_mapped_sections_are_real_skill_headings(self):
        """Drift guard: every section the map cites (plus the sensitivity fallback) must exist
        as an actual heading in its skill, so a pointer never sends the agent to a dead anchor."""
        sections_by_skill: dict[str, set[str]] = {}
        for skill, section in srv._DECISION_GUIDANCE.values():
            sections_by_skill.setdefault(skill, set()).add(section)
        sections_by_skill["solution_interpreter"].add("Binding Analysis")  # sensitivity fallback
        for skill, sections in sections_by_skill.items():
            # Headings span the skill core AND its references — a pointer target may
            # live in either; _section_titles is the same resolver get_skill uses.
            headings = set(srv._section_titles(srv._SKILL_MAP[skill]))
            missing = sections - headings
            assert not missing, f"{skill} is missing heading(s): {missing}"

    def test_get_skill_section_fetch(self):
        """The recovery loop the split depends on: every pointer-cited section is
        retrievable as exactly one section via get_skill(name, section=...)."""
        body = srv.get_skill("solution_interpreter", section="Reading the Certificate (explore certify)")
        assert body.startswith("### Reading the Certificate")
        assert "dominance_audit" in body.lower() or "dominance audit" in body.lower()
        # core fetch stays lean and does NOT inline the reference depth
        core = srv.get_skill("solution_interpreter")
        assert "Never Say \"Best\"" in core
        assert "### Reading the Certificate" not in core
        # depth section from the other split skill resolves too
        depth = srv.get_skill("optimization_strategy", section="Exact Solvers — Depth")
        assert "three certifiable shapes" in depth
        # unknown section errors with the available titles, not silence
        miss = srv.get_skill("solution_interpreter", section="No Such Heading")
        assert "Unknown section" in miss and "Solution Curation" in miss


class TestComposition:
    def test_composition_returns_blocks_and_pointer(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        r = srv.explore(action="composition", problem_id=pid)
        assert "option_selection" in r
        assert "design_principles" in r
        assert "clusters" in r
        assert "feedback_rules" in r
        assert r["feedback_rules"]["available"] is False
        assert r["scope"]["set"] == "frontier"
        assert r["guidance_pointer"]["skill"] == "solution_interpreter"
        assert r["guidance_pointer"]["section"] == "Mining the Solution Set"

    def test_composition_curated_subset(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        sols = srv.explore(action="solutions", problem_id=pid)["solutions"]
        sid = sols[0]["solution_id"]
        r = srv.explore(action="composition", problem_id=pid, solution_ids=[sid])
        assert r["scope"]["set"] == "curated"
        assert r["scope"]["n_solutions"] == 1

    def test_small_frontier_detail_stays_inline(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)
        r = srv.explore(action="composition", problem_id=pid, detail=True)
        assert "full_result_path" not in r
        assert "truncated" not in r

    def test_large_frontier_detail_writes_full_result_path(self):
        # detail=true lifts the co-occurrence top-N to every sometimes-but-not-always
        # pair — O(options²), ~400 KB on the 300-option showcase — so detail=true dumps
        # to disk and keeps the top-ranked pairs inline, exactly as marginal_analysis
        # detail=true does (the claim the explore docstring makes for large-result actions).
        import json
        from pathlib import Path

        pid = srv.model(action="load", source="capital_project_selection_300")["problem_id"]
        r = srv.explore(action="composition", problem_id=pid, detail=True)
        assert len(r["co_occurrence"]) == srv.EXPLORE_DETAIL_CAP
        assert r["truncated"]["total_pairs"] > srv.EXPLORE_DETAIL_CAP
        assert r["truncated"]["shown"] == srv.EXPLORE_DETAIL_CAP
        # The inline response fits a tool-result budget the full dump blows past.
        assert len(json.dumps(r)) < 100_000
        on_disk = json.loads(Path(r["full_result_path"]).read_text())
        assert len(on_disk["co_occurrence"]) == r["truncated"]["total_pairs"]
        assert "truncated" not in on_disk
        # The inline pairs are the head of the ranked full list — a window, not a re-rank.
        assert on_disk["co_occurrence"][:srv.EXPLORE_DETAIL_CAP] == r["co_occurrence"]


class TestRegret:
    def _add_scenarios(self, pid):
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [
                {"name": "Base", "probability": 0.6, "score_overrides": []},
                {"name": "Down", "probability": 0.4, "score_overrides": [
                    {"option": "A", "objective": "Rev", "value": 1},
                ]},
            ],
        })

    def test_scenario_results_includes_regret(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid)   # base frontier needed for regret
        self._add_scenarios(pid)
        srv.solve(action="run_scenarios", problem_id=pid)
        regret = srv.explore(action="scenario_results", problem_id=pid)["regret"]
        assert regret["available"] is True
        assert regret["method"] == "scenario_minimax"
        assert regret["minimax_choice"] is not None
        for entry in regret["per_solution"]:
            assert 0.0 <= entry["max_regret"] <= 1.0  # normalized + clamped
            assert all(0.0 <= v <= 1.0 for v in entry["by_scenario"].values())
            assert "feasible_in_all" in entry

    def test_regret_absent_without_base_run(self):
        pid = _build_solvable_problem()
        self._add_scenarios(pid)
        srv.solve(action="run_scenarios", problem_id=pid)
        regret = srv.explore(action="scenario_results", problem_id=pid)["regret"]
        assert regret["available"] is False


# ─── HTTP transport gating (fail-closed) ───


class TestHttpTransportAction:
    """The auth decision for the SSE / streamable-http transports.

    A token always gates. Without a token, a loopback bind serves (local dev)
    but a routable bind fails closed rather than exposing an ungated engine.
    """

    def test_token_always_gates(self):
        for host in ("127.0.0.1", "0.0.0.0", "10.0.0.5", "example.com"):
            assert srv._http_transport_action("secret", host) == "gated"

    def test_no_token_loopback_serves_ungated(self):
        for host in ("127.0.0.1", "::1", "localhost"):
            assert srv._http_transport_action(None, host) == "ungated"

    def test_no_token_routable_bind_refuses(self):
        # "" is INADDR_ANY (all interfaces) in uvicorn — exposed, not loopback.
        for host in ("0.0.0.0", "10.0.0.5", "192.168.1.10", "example.com", ""):
            assert srv._http_transport_action(None, host) == "refuse"
            assert srv._http_transport_action("", host) == "refuse"


def test_option_removal_prunes_allocation_bound():
    """Removing an option drops its allocation_bound like every other option-referencing
    constraint — a survivor would wedge validate/solve on a dangling reference."""
    import mcp_server.server as srv

    r = srv.model(action="create", name="prune", approach="proportional",
                  domain="t", context="t")
    pid = r["problem_id"]
    srv.model(action="update", problem_id=pid,
              objectives=[{"name": "V", "direction": "maximize"},
                          {"name": "C", "direction": "minimize"}],
              options=[{"name": n} for n in ("A", "B", "C", "D")],
              scores=[{"option": n, "objective": o, "value": 1}
                      for n in ("A", "B", "C", "D") for o in ("V", "C")],
              constraints=[{"type": "allocation_bound", "option": "D", "min": 10, "max": 50}])
    srv.model(action="update", problem_id=pid, options=[{"name": n} for n in ("A", "B", "C")])
    g = srv.model(action="get", problem_id=pid, section="constraints")
    assert all(c.get("option") != "D" for c in g["constraints"])


class TestStaleGuardCoverage:
    """The objective-drift guard must cover every consumer — including the paths that
    would otherwise LAUNDER stale state (certify) or zero-fill it (scenario regret)."""

    def _solved_then_grown(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"},
                              {"name": "Risk", "direction": "minimize"}],
                  scores=[{"option": o, "objective": "Risk", "value": 1.0}
                          for o in ("A", "B", "C", "D")])
        return pid

    def test_progressive_certify_declines_stale_source_frontier(self):
        pid = self._solved_then_grown()
        out = srv.solve(action="run", problem_id=pid, solver="highs")
        assert "predates the current objectives" in out.get("error", "")

    def test_explore_certify_declines_stale_runs(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        srv.solve(action="run", problem_id=pid, solver="highs", scope="full")
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Effort", "direction": "minimize"}],  # Eff renamed
                  scores=[{"option": o, "objective": "Effort", "value": 1.0}
                          for o in ("A", "B", "C", "D")])
        out = srv.explore(action="certify", problem_id=pid)
        assert "stale" in out.get("error", "")

    def test_scenario_results_declines_stale_scenario_runs(self):
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [{"name": "Base", "score_overrides": []}],
        })
        srv.solve(action="run", problem_id=pid, seed=42)
        srv.solve(action="run_scenarios", problem_id=pid, seed=42)
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"},
                              {"name": "Risk", "direction": "minimize"}],
                  scores=[{"option": o, "objective": "Risk", "value": 1.0}
                          for o in ("A", "B", "C", "D")])
        out = srv.explore(action="scenario_results", problem_id=pid)
        assert "stale" in out.get("error", "")
        assert "run_scenarios" in out["error"]

    def test_stale_scenario_decline_names_run_scenarios(self):
        pid = _build_solvable_problem()
        srv.model(action="update", problem_id=pid, scenario_config={
            "enabled": True,
            "scenarios": [{"name": "Base", "score_overrides": []}],
        })
        srv.solve(action="run", problem_id=pid, seed=42)
        srv.solve(action="run_scenarios", problem_id=pid, seed=42)
        # base re-solve AFTER the objective change clears the base run's staleness…
        srv.model(action="update", problem_id=pid,
                  objectives=[{"name": "Rev", "direction": "maximize"},
                              {"name": "Eff", "direction": "minimize"},
                              {"name": "Risk", "direction": "minimize"}],
                  scores=[{"option": o, "objective": "Risk", "value": 1.0}
                          for o in ("A", "B", "C", "D")])
        srv.solve(action="run", problem_id=pid, seed=42)
        # …so the scenario read must name the remedy that actually clears ITS decline.
        out = srv.explore(action="tradeoffs", problem_id=pid, scenario="Base")
        assert "stale" in out.get("error", "")
        assert "run_scenarios" in out["error"]


class TestScenarioParamGuards:
    """Wrong scenario-param names must return a redirect, not be silently dropped.

    Surfaced by a model eval: capable agents guessed `scenarios=` on model / `scenario=`
    / `run_scenarios=` on solve, which FastMCP silently dropped — a plain run then returned
    the BASE frontier as if it were the scenario. The guards turn that into a clear error.
    """

    def test_model_scenarios_argument_redirects_to_scenario_config(self):
        out = srv.model(action="update", problem_id="x", scenarios=[{"name": "s"}])
        assert "error" in out
        assert "scenario_config" in out["error"]

    def test_solve_scenario_argument_redirects_to_run_scenarios(self):
        out = srv.solve(action="run", problem_id="x", scenario="fab_outage")
        assert "error" in out
        assert "run_scenarios" in out["error"]

    def test_solve_run_scenarios_flag_redirects_to_action(self):
        out = srv.solve(action="run", problem_id="x", run_scenarios=True)
        assert "error" in out
        assert "run_scenarios" in out["error"]

    def test_normal_solve_and_model_unaffected(self):
        # No guard args → normal path (regression): create → update → run must still work.
        pid = srv.model(action="create", name="G", objectives=[
            {"name": "V", "direction": "maximize"},
            {"name": "W", "direction": "minimize"}], options=["A", "B", "C"])["problem_id"]
        srv.model(action="update", problem_id=pid, scores=[
            {"option": o, "objective": obj, "value": v}
            for o in ("A", "B", "C") for obj, v in (("V", 1.0), ("W", 2.0))])
        assert "error" not in srv.solve(action="run", problem_id=pid, seed=1)


class TestSensitivityAnchorAndLabelGuard:
    """Two more eval-surfaced silent traps: sensitivity ignoring content_signature (and
    silently anchoring on the balanced default instead), and `label` vs `custom_name`."""

    def test_sensitivity_honors_content_signature(self):
        pid = srv.model(action="load", source="scarce_supply_rationing")["problem_id"]
        sols = srv.explore(action="solutions", problem_id=pid, source="exact")
        sig = (sols.get("solutions") or [])[0]["content_signature"]
        out = srv.explore(action="sensitivity", problem_id=pid, source="exact",
                          content_signature=sig)
        assert "error" not in out

    def test_sensitivity_unmatched_signature_errors_not_silent(self):
        pid = srv.model(action="load", source="scarce_supply_rationing")["problem_id"]
        out = srv.explore(action="sensitivity", problem_id=pid, source="exact",
                          content_signature="deadbeefdeadbeef")
        assert "error" in out and "content_signature" in out["error"]

    def test_explore_label_argument_redirects_to_custom_name(self):
        out = srv.explore(action="curate", problem_id="x", solution_id=1, label="Balanced")
        assert "error" in out and "custom_name" in out["error"]

    def test_curate_solution_ids_redirects_to_singular(self):
        # Plural solution_ids on curate (compare's param) → clear redirect, not the generic
        # "solution_id required" that reads as if nothing was passed.
        pid = srv.model(action="create", name="C", objectives=[
            {"name": "V", "direction": "maximize"}], options=["A", "B"])["problem_id"]
        out = srv.explore(action="curate", problem_id=pid, solution_ids=[1, 2])
        assert "error" in out and "per call" in out["error"]


class TestDroppedFrontierDisclosure:
    """A stale-flagged solve clears the OTHER stored frontier by design (it predates the
    edit, so clearing results_stale mustn't vouch for it). Sound, and previously silent —
    the next `explore` just found a frontier gone. The drop now says itself."""

    def _both_frontiers_then_edit(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        srv.solve(action="run", problem_id=pid, solver="highs", scope="full")
        # Structural edit (constraints ride the solve fingerprint) → results_stale.
        srv.model(action="update", problem_id=pid,
                  constraints=[{"type": "cardinality", "min": 1, "max": 3}])
        return pid

    def test_nsga_resolve_names_the_dropped_exact_overlay(self):
        pid = self._both_frontiers_then_edit()
        out = srv.solve(action="run", problem_id=pid, seed=42)
        note = out["dropped_frontier_note"]
        assert "exact overlay" in note and "predates" in note
        assert 'solve(solver="highs")' in note  # the re-run that restores it
        assert srv.store.load(pid).exact_run is None  # the note describes what happened

    def test_exact_resolve_names_the_dropped_nsga_frontier(self):
        pid = self._both_frontiers_then_edit()
        out = srv.solve(action="run", problem_id=pid, solver="highs", scope="full")
        note = out["dropped_frontier_note"]
        assert "NSGA" in note and "predates" in note and "solve run" in note
        assert srv.store.load(pid).run is None

    def test_fresh_solve_drops_nothing_and_says_nothing(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        out = srv.solve(action="run", problem_id=pid, solver="highs", scope="full")
        assert "dropped_frontier_note" not in out
        assert srv.store.load(pid).run is not None  # nothing to drop: results were fresh

    def test_stale_solve_with_no_other_frontier_stays_silent(self):
        """The branch the `is not None` guards exist for: stale, but only ONE frontier
        stored. Nothing is dropped, so nothing may be claimed — a note here would report
        a loss that never happened. Probed in both directions."""
        # NSGA only → re-solve with NSGA.
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        srv.model(action="update", problem_id=pid,
                  constraints=[{"type": "cardinality", "min": 1, "max": 3}])
        assert srv.store.load(pid).results_stale is True
        out = srv.solve(action="run", problem_id=pid, seed=42)
        assert "dropped_frontier_note" not in out

        # Exact overlay only (no NSGA run to drop) → re-solve with the exact backend.
        pid2 = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid2, solver="highs", scope="full")
        p = srv.store.load(pid2)
        assert p.run is None and p.exact_run is not None
        srv.model(action="update", problem_id=pid2,
                  constraints=[{"type": "cardinality", "min": 1, "max": 3}])
        assert srv.store.load(pid2).results_stale is True
        out = srv.solve(action="run", problem_id=pid2, solver="highs", scope="full")
        assert "dropped_frontier_note" not in out
        assert srv.store.load(pid2).exact_run is not None  # the overlay was refreshed


class TestOverlaySeedInheritance:
    """`scope="curated"`/`"fill_gaps"` re-solve an EXISTING frontier's points, so the seed
    is the source run's. Passing one there was accepted and ignored, and `seed_used` came
    back a number the caller never sent."""

    def test_curated_overlay_declines_a_passed_seed_out_loud(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        out = srv.solve(action="run", problem_id=pid, solver="highs", seed=7712)
        assert out["overlay_scope"] == "curated"
        assert out["seed_used"] == 42, "the overlay inherits the source run's seed"
        note = out["seed_note"]
        assert "7712" in note and "seed_used=42" in note and 'scope="full"' in note

    def test_full_scope_uses_the_seed_and_stays_quiet(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        out = srv.solve(action="run", problem_id=pid, solver="highs", scope="full", seed=7712)
        assert out["seed_used"] == 7712 and "seed_note" not in out

    def test_no_seed_no_note(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        out = srv.solve(action="run", problem_id=pid, solver="highs")
        assert "seed_note" not in out


class TestSingularOptionConstraintShape:
    """`{"type": "force_exclude", "options": [...]}` answered with pydantic's bare
    "option: Field required" — the missing field named, the SHAPE unexplained."""

    def test_plural_options_names_the_singular_shape(self):
        pid = _build_solvable_problem()
        out = srv.model(action="update", problem_id=pid,
                        constraints=[{"type": "force_exclude", "options": ["B", "C"]}])
        err = out["error"]
        assert "`option` (singular)" in err and "one constraint row per option" in err
        assert '{"type": "force_exclude", "option": "B"}' in err  # the payload to send
        assert "group_limit" in err  # the type that DOES take a set of options
        assert len(srv.store.load(pid).constraints) == 1, "nothing was written"

    def test_guard_covers_every_single_option_type(self):
        # Derived from the models, so a new single-option type inherits the guard.
        assert srv._SINGULAR_OPTION_TYPES == {"force_include", "force_exclude",
                                              "allocation_bound"}

    def test_set_shaped_sibling_matches_what_the_type_constrains(self):
        """group_limit is a selection-count rule, so it is the wrong sibling for an
        allocation box — an allocation_bound miscall is pointed at max_allocation.
        (Several allocation_bound rows on ONE option are legitimate: they intersect,
        per the merge landed in #132 — which is exactly why the row-per-option shape
        is the fix here, not a set-shaped rewrite.)"""
        pid = _build_solvable_problem()
        err = srv.model(action="update", problem_id=pid,
                        constraints=[{"type": "allocation_bound", "options": ["B", "C"],
                                      "min": 5, "max": 40}])["error"]
        assert "one constraint row per option" in err
        assert "max_allocation" in err and "group_limit" not in err

    def test_correct_singular_rows_still_parse(self):
        pid = _build_solvable_problem()
        out = srv.model(action="update", problem_id=pid,
                        constraints=[{"type": "force_exclude", "option": "B"},
                                     {"type": "force_exclude", "option": "C"}])
        assert out["status"]["constraints"] == 2


class TestExploreFormatParamHonesty:
    """`format` is the curated handoff export's renderer. Elsewhere it was accepted and
    dropped, so format="summary" returned the FULL payload while reading as a summary."""

    def test_format_on_tradeoffs_redirects(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        out = srv.explore(action="tradeoffs", problem_id=pid, format="summary")
        err = out["error"]
        assert "curated" in err and "detail=true" in err and "_elided" in err

    def test_format_on_scenario_results_redirects(self):
        pid = _build_solvable_problem()
        out = srv.explore(action="scenario_results", problem_id=pid, format="compact")
        assert "format" in out["error"] and "curated" in out["error"]

    def test_curated_export_still_renders(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        out = srv.explore(action="curated", problem_id=pid, format="markdown")
        assert "error" not in out


class TestCompositionClusterQuality:
    def test_clusters_summary_rides_with_the_clusters(self):
        pid = _build_solvable_problem()
        srv.solve(action="run", problem_id=pid, seed=42)
        out = srv.explore(action="composition", problem_id=pid)
        # Present exactly when there are families to price — never a lone empty block.
        assert ("clusters_summary" in out) == bool(out["clusters"])
        if out["clusters"]:
            s = out["clusters_summary"]
            assert s["n_families"] == len(out["clusters"])
            assert 0 < s["largest_family_share"] <= 1
