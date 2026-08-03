"""Schema-level steering guards: the wrong-name params advertise their redirect.

The guard params (model.scenarios, solve.scenario/run_scenarios, explore.label)
are schema-valid by design — FastMCP silently drops unknown arguments, so each
guard must exist in the signature to catch its miscall. But an undescribed guard
param is an attractive nuisance: `run_scenarios: boolean` in the schema reads as
a flag and invites the very call it rejects (both 2026-07-10 demo-replay agents
made exactly these miscalls before the runtime guard corrected them). These
tests pin the fix — every guard param, and the merge-vs-replace contract on the
model data params, carries its steering description in the advertised
inputSchema — and gate it against an SDK upgrade dropping Field descriptions.
Assertions are keyword-level, not exact-phrase: wording may be polished freely
as long as the steering target / semantic stays named.

The runtime half of the constraints contract (count echo + shrink note) lives
with the other update-behavior tests in test_server.py::TestModelUpdate.
"""

import mcp_server.server as srv

# Built once: same private-API introspection as test_wire_compactness.py — an SDK
# rename breaks both files together, one seam to fix.
_TOOLS = {t.name: t for t in srv.mcp._tool_manager.list_tools()}


def _param_desc(tool: str, param: str) -> str:
    return _TOOLS[tool].parameters["properties"][param].get("description", "")


class TestGuardParamSteering:
    def test_explore_label_steers_to_custom_name(self):
        assert "custom_name" in _param_desc("explore", "label")

    def test_solve_run_scenarios_steers_to_action(self):
        assert 'action="run_scenarios"' in _param_desc("solve", "run_scenarios")

    def test_solve_scenario_steers_to_action(self):
        desc = _param_desc("solve", "scenario")
        assert 'action="run_scenarios"' in desc and "explore" in desc

    def test_model_scenarios_steers_to_scenario_config(self):
        assert "scenario_config" in _param_desc("model", "scenarios")

    def test_guard_tails_stay_identical(self):
        """The strict guards end with the shared _GUARD_TAIL — no drift between copies.
        (run_scenarios documents its own two-mode behavior: redundant-consistent use
        is ignored, contradicting use redirects.)"""
        for tool, param in (("model", "scenarios"), ("solve", "scenario"),
                            ("explore", "label")):
            assert _param_desc(tool, param).endswith(srv._GUARD_TAIL)
        desc = _param_desc("solve", "run_scenarios")
        assert "ignored" in desc and "redirect" in desc


class TestReplaceVsMergeContract:
    def test_constraints_declare_full_replacement(self):
        desc = _param_desc("model", "constraints").upper()
        assert "FULL REPLACEMENT" in desc and "COMPLETE" in desc

    def test_objectives_and_options_declare_full_replacement(self):
        for param in ("objectives", "options"):
            assert "FULL REPLACEMENT" in _param_desc("model", param).upper()

    def test_objectives_name_the_matrix_cascade(self):
        assert "interaction matrices" in _param_desc("model", "objectives")

    def test_scores_declare_merge(self):
        assert "merge" in _param_desc("model", "scores").lower()

    def test_reference_points_declare_full_replacement(self):
        assert "FULL REPLACEMENT" in _param_desc("model", "reference_points").upper()

    def test_interaction_matrices_declare_upsert(self):
        """The description must say upsert — and the behavior must actually upsert.

        These schema guards assert what a description CLAIMS, which is only half a
        contract: this one passed for a long stretch while the update path replaced
        the whole matrix per objective, so the docs promised incremental writes the
        code didn't do. A claim test needs a behavior test behind it — here
        tests/test_server.py::TestInteractionMatrixIncrementalWrite. Pair them for
        any new guard, or the guard just pins the wording of something untrue.
        """
        assert "upsert" in _param_desc("model", "interaction_matrices").lower()

    def test_interaction_matrices_docstring_agrees(self):
        """The tool docstring must state the same contract as the param description
        (it once said 'everything else is full replacement', contradicting the upsert)."""
        doc = _TOOLS["model"].description
        assert "interaction_matrices merge" in doc

    def test_scenario_config_declares_override_replacement(self):
        desc = _param_desc("model", "scenario_config").lower()
        assert "constraint_overrides" in desc
        assert "replace" in desc and "inherit" in desc

    def test_source_declares_load_only(self):
        assert 'action="load"' in _param_desc("model", "source")


class TestFormatIsExportOnly:
    """`format` renders the curated handoff export — one word, one meaning. It was
    accepted and ignored elsewhere, so format="summary" returned a full payload dressed
    as a summary. The schema now says so, and the runtime redirect is the behavior half
    (tests/test_server.py::TestExploreFormatParamHonesty)."""

    def test_format_declares_curated_scope_and_the_real_levers(self):
        desc = _param_desc("explore", "format")
        assert "curated" in desc and "markdown" in desc and "csv" in desc
        assert "summary" in desc and "detail=" in desc

    def test_explore_docstring_agrees(self):
        doc = _TOOLS["explore"].description
        assert "not a verbosity switch" in doc.lower()
