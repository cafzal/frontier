/**
 * Types and extractor for the structured viz_data payloads that Frontier's
 * `explore/*` tools attach alongside ASCII visualizations.
 *
 * Engine source: `engine/viz.py` `_viz_data_*` builders (plus
 * `mcp_server/server.py` `_viz_data_formulation` for the formulation card).
 * Hosts that can render charts (this web UI) consume viz_data; chat / coding-
 * agent surfaces ignore it and render the ASCII `visualization` field instead.
 */

export type ObjectiveMeta = {
  name: string;
  direction: "minimize" | "maximize";
  min: number;
  max: number;
};

export type ScatterPoint = {
  solution_id: number;
  values: Record<string, number>;
  name?: string | null; // the solution's curated custom_name, if curated
};

// Which frontier the scatter was computed over. `kind` distinguishes the heuristic NSGA
// frontier from an exact-certified one; `exact_certified` means every point carries an
// optimality certificate (always on the continuous LP/QP path, on MILP only at a zero gap —
// engine-side single classifier: solvers.run_is_certified).
// Mirrors the engine's `frontier_source`, but lives inside viz_data so the chart can denote it.
export type ScatterProvenance = {
  kind: "heuristic" | "exact";
  solver: string;
  exact_certified: boolean;
};

// Attached only on a heuristic base-case frontier that has an exact-solver overlay. `points`
// are the exact-certified solutions to draw on top; `dominated_ids` are the heuristic points
// the exact front strictly dominates ("looked efficient, but exact beats them at their cost").
export type ScatterExactOverlay = {
  solver: string;
  // Overlay provenance — matches certify's `exact_run_id`, so stale scatter reads are detectable.
  exact_run_id: string;
  exact_certified: boolean;
  points: ScatterPoint[];
  dominated_ids: number[];
};

export type ScatterVizData = {
  type: "scatter";
  objectives: ObjectiveMeta[];
  points: ScatterPoint[];
  extremes: Record<string, { best_id: number; worst_id: number }>;
  balanced_id: number;
  inflection_ids: number[];
  provenance?: ScatterProvenance; // present on the real explore path; absent in isolated builds
  exact_overlay?: ScatterExactOverlay | null;
  // Optional: render as a 2D scatter colored by this objective (the two remaining objectives
  // become the x/y axes). Lets a 3-objective frontier show as a clean color-encoded 2D scatter
  // instead of 3D. Set by the render route (?color=<objective>); absent on the normal chat path.
  color_objective?: string | null;
};

export type ParallelCoordsSeries = {
  id: number | string;
  label: string;
  values: Record<string, number>;
};

export type ParallelCoordsVizData = {
  type: "parallel_coords";
  axes: ObjectiveMeta[];
  series: ParallelCoordsSeries[];
};

export type MarginalRate = {
  from_id: number;
  to_id: number;
  rate: number;
  [k: string]: unknown;
};

export type MarginalRatesVizData = {
  type: "marginal_rates";
  from_objective: { name: string; direction: "minimize" | "maximize" };
  to_objective: { name: string; direction: "minimize" | "maximize" };
  rates: MarginalRate[];
  inflection: { solution_id: number; position: number; jump_factor: number } | null;
};

// Minimax-regret lens from scenario_results (engine: explorer.py `scenario_regret`).
// Per-solution, so it's a distinct view from the per-objective scenario_risk table.
// `available` is false when there's no base run to regret against — or when the lens was
// declined: `reason` names that case, and `note` carries the engine's wording.
export type ScenarioRegret = {
  available: boolean;
  // "base_scenario_mismatch" — the base run and the scenario runs were solved against
  // different models, so re-scoring base solutions against those scenario frontiers
  // would join two models. The ranking is withheld rather than served wrong.
  reason?: string;
  method?: string;
  normalization?: string;
  minimax_choice?: {
    solution_id: number;
    content_signature?: string;
    max_regret: number;
  } | null;
  per_solution?: Array<{
    solution_id: number;
    content_signature?: string;
    max_regret: number;
    mean_regret: number | null;
    by_scenario?: Record<string, number>;
    feasible_in_all: boolean;
    // Feasible in every RANKED scenario (wipeouts excluded) — the universe the
    // max/mean/minimax figures are computed over. Distinguishes "fails only the
    // excluded wipeout" from "fails a scenario that drives the ranking".
    feasible_in_ranked?: boolean;
  }>;
  per_objective?: Record<string, { min_max_regret: number; achieved_by_solution_id: number }>;
  // Full base-frontier count — per_solution is the compact lowest-regret slice (≤20),
  // so "how many more" must come from here, not from per_solution.length.
  per_solution_total?: number;
  note?: string;
  // True when every base-frontier solution hits max regret (infeasible or fully
  // dominated) in some scenario — the ranking is a meaningless all-1.0 tie, the
  // engine omits minimax_choice, and saturation_note names the scenarios + the fix.
  saturated?: boolean;
  saturation_note?: string;
  // Scenarios in which NO base solution is feasible: excluded from the minimax ranking
  // (their regret is uniformly total); wipeout_note names them and the way forward.
  wipeout_scenarios?: string[];
  wipeout_note?: string;
  // Per-scenario count of base-frontier solutions feasible in that scenario.
  survivors_by_scenario?: Record<string, number>;
};

export type ScenarioSummaryVizData = {
  type: "scenario_summary";
  // Set when the stored scenario runs were solved against an earlier model — the panel
  // leads with the note, since the ranges below describe the model as it was.
  scenario_results_stale?: boolean;
  scenario_stale_note?: string;
  scenarios: string[];
  option_robustness: Array<Record<string, unknown>>;
  expected_values: Record<string, unknown>;
  scenario_risk: Record<string, unknown>;
  regret?: ScenarioRegret;
};

export type CuratedScenarioPick = {
  name: string;
  // true when the pick's objective vector is identical across scenarios (e.g.
  // constraint-only scenarios); then `lines` holds one entry with scenario = -1.
  invariant: boolean;
  // one line per distinct profile: scenario = -1 for an invariant pick, else the
  // scenario index it was evaluated under (drift across score-based scenarios).
  lines: Array<{ scenario: number; values: Record<string, number> }>;
  // scenario indices the slate is feasible in — drives presence coloring.
  present: number[];
};

export type ScenarioParcoordsVizData = {
  type: "scenario_parcoords";
  axes: ObjectiveMeta[];
  scenarios: string[];
  lines: Array<{ scenario: number; values: Record<string, number> }>;
  curated?: CuratedScenarioPick[];
};

export type FormulationVizData = {
  type: "formulation";
  name: string;
  domain: string;
  approach: string;
  options_count: number;
  scores_complete: number;
  objectives: Array<{ name: string; direction: string; aggregation: string; unit?: string | null }>;
  constraints: string[];
  scenarios: string[];
};

export type VizData =
  | ScatterVizData
  | ParallelCoordsVizData
  | MarginalRatesVizData
  | ScenarioSummaryVizData
  | ScenarioParcoordsVizData
  | FormulationVizData;

/**
 * Walk a tool_result's text payload and pull out every viz_data block.
 * Handles:
 *   - top-level `viz_data` (tradeoffs, compare — incl. signatures= for curated, scenario_results)
 *   - nested `pairs[].viz_data` (marginal_analysis returns multiple pairs)
 * Returns [] if not parseable JSON or no viz_data present.
 */
export function extractVizData(text: string): VizData[] {
  let parsed: any;
  try {
    parsed = JSON.parse(text);
  } catch {
    return [];
  }
  const out: VizData[] = [];
  if (parsed && typeof parsed === "object") {
    if (parsed.viz_data && parsed.viz_data.type) {
      out.push(parsed.viz_data as VizData);
    }
    if (Array.isArray(parsed.pairs)) {
      for (const p of parsed.pairs) {
        if (p && p.viz_data && p.viz_data.type) {
          out.push(p.viz_data as VizData);
        }
      }
    }
  }
  return out;
}
