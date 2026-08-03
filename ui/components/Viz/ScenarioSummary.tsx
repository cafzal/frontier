"use client";

import type { ScenarioSummaryVizData } from "@/lib/viz-data";

type OptionRobustness = {
  option: string;
  tier?: "core" | "common" | "marginal";
  avg_frequency?: number;
  avg_weight?: number;
};

type ScenarioRiskEntry = {
  expected?: number;
  worst_case?: number;
  best_case?: number;
  range?: [number, number];
  [k: string]: unknown; // accommodates dynamic cvar_<α%> key
};

const TIER_STYLES: Record<string, string> = {
  core: "bg-emerald-100 text-emerald-800 border-emerald-300",
  common: "bg-sky-100 text-sky-800 border-sky-300",
  marginal: "bg-stone-100 text-stone-600 border-stone-300",
};

// Cap the per-solution regret table; the full ranking lives in `explore scenario_results`.
const MAX_REGRET_ROWS = 8;

const BANNER_TONES: Record<"amber" | "indigo", string> = {
  amber: "border-amber-300 bg-amber-50 text-amber-900",
  indigo: "border-indigo-200 bg-indigo-50 text-indigo-900",
};

// One callout shape for every engine finding rendered in this panel.
function Banner({ tone, label, children }: { tone: "amber" | "indigo"; label: string; children: React.ReactNode }) {
  return (
    <div className={`mb-2 rounded border px-2 py-1.5 text-xs ${BANNER_TONES[tone]}`}>
      <span className="font-semibold">{label}:</span> {children}
    </div>
  );
}

// Three-state feasibility: ✓ everywhere; ✗ in a RANKED scenario (drives the ranking);
// amber ⨯ when the only failure is an excluded wipeout scenario (harmless to the ranking).
function FeasibilityMark({ row }: { row: { feasible_in_all: boolean; feasible_in_ranked?: boolean } }) {
  if (row.feasible_in_all) return <span className="text-emerald-600">✓</span>;
  if (row.feasible_in_ranked) {
    return (
      <span className="text-amber-600" title="feasible in every ranked scenario — infeasible only in the excluded wipeout scenario(s)">
        ⨯<span className="ml-0.5 text-[9px] uppercase">wipeout only</span>
      </span>
    );
  }
  return (
    <span className="text-red-600" title="infeasible in a ranked scenario (drives max regret to 100%)">
      ✗
    </span>
  );
}

export function ScenarioSummary({ data }: { data: ScenarioSummaryVizData }) {
  // No scenarios → nothing meaningful to show; suppress the empty panel.
  if (!data.scenarios?.length) return null;
  const options = (data.option_robustness ?? []) as OptionRobustness[];
  const risks = (data.scenario_risk ?? {}) as Record<string, ScenarioRiskEntry>;
  const expected = (data.expected_values ?? {}) as Record<string, number>;
  const scenarios = data.scenarios ?? [];

  // Minimax-regret: per-solution, ranked ascending by max_regret upstream, so the
  // first rows are the most robust. Show the lowest-regret few and flag the rest.
  const regret = data.regret;
  const minimaxId = regret?.minimax_choice?.solution_id ?? null;
  const allRegretRows = regret?.per_solution ?? [];
  const regretRows = allRegretRows.slice(0, MAX_REGRET_ROWS);
  // per_solution is the engine's compact ≤20 slice; the true frontier size ships in
  // per_solution_total (a 300-solution frontier hides 292 rows here, not 12).
  const hiddenRegretCount = (regret?.per_solution_total ?? allRegretRows.length) - regretRows.length;
  const wipeoutList = regret?.wipeout_scenarios?.length ? regret.wipeout_scenarios.join(", ") : null;

  // Group options by tier for visual hierarchy
  const byTier: Record<string, OptionRobustness[]> = {
    core: [],
    common: [],
    marginal: [],
  };
  for (const opt of options) {
    const t = opt.tier ?? "marginal";
    (byTier[t] ?? byTier.marginal).push(opt);
  }

  return (
    <div className="my-3 rounded border border-stone-200 bg-white p-3 text-sm">
      <div className="mb-2 flex items-center justify-between text-[10px] text-stone-500">
        <span>
          scenario summary · {scenarios.length} scenarios
          {options.length > 0 && ` · ${options.length} options classified`}
        </span>
      </div>

      {/* Staleness leads the panel for the same reason it leads the payload: these
          numbers describe the model as it was, and nothing below says so on its own. */}
      {data.scenario_results_stale && data.scenario_stale_note && (
        <Banner tone="amber" label="Stale scenario results">
          {data.scenario_stale_note}
        </Banner>
      )}

      {scenarios.length > 0 && (
        <div className="mb-3 text-xs">
          <span className="text-stone-500">scenarios:</span>{" "}
          <span className="font-mono text-stone-700">{scenarios.join(", ")}</span>
        </div>
      )}

      {/* Option robustness by tier */}
      {options.length > 0 && (
        <div className="mb-3">
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-stone-600">
            Option robustness
          </div>
          <div className="space-y-1.5">
            {(["core", "common", "marginal"] as const).map((tier) =>
              byTier[tier].length === 0 ? null : (
                <div key={tier} className="flex flex-wrap items-center gap-1.5">
                  <span
                    className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase ${TIER_STYLES[tier]}`}
                  >
                    {tier}
                  </span>
                  {byTier[tier].map((o) => (
                    <span
                      key={o.option}
                      className="rounded bg-stone-50 px-1.5 py-0.5 text-xs text-stone-700"
                      title={
                        o.avg_frequency !== undefined
                          ? `freq: ${(o.avg_frequency * 100).toFixed(0)}%${
                              o.avg_weight !== undefined
                                ? `, weight: ${o.avg_weight.toFixed(2)}`
                                : ""
                            }`
                          : undefined
                      }
                    >
                      {o.option}
                      {o.avg_frequency !== undefined && (
                        <span className="ml-1 text-[10px] text-stone-400">
                          {(o.avg_frequency * 100).toFixed(0)}%
                        </span>
                      )}
                    </span>
                  ))}
                </div>
              )
            )}
          </div>
        </div>
      )}

      {/* Scenario risk table */}
      {Object.keys(risks).length > 0 && (
        <div>
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-stone-600">
            Scenario risk per objective
          </div>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-xs">
              <thead className="text-stone-500">
                <tr>
                  <th className="border-b border-stone-200 px-2 py-1 text-left font-normal">
                    objective
                  </th>
                  <th className="border-b border-stone-200 px-2 py-1 text-right font-normal">
                    expected
                  </th>
                  <th className="border-b border-stone-200 px-2 py-1 text-right font-normal">
                    worst
                  </th>
                  <th className="border-b border-stone-200 px-2 py-1 text-right font-normal">
                    best
                  </th>
                  <th className="border-b border-stone-200 px-2 py-1 text-right font-normal">
                    CVaR
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(risks).map(([obj, r]) => {
                  const cvarKey = Object.keys(r).find((k) => k.startsWith("cvar_"));
                  const cvarVal = cvarKey ? (r[cvarKey] as number | null) : null;
                  return (
                    <tr key={obj} className="hover:bg-stone-50">
                      <td className="border-b border-stone-100 px-2 py-1 font-mono text-stone-700">
                        {obj}
                      </td>
                      <td className="border-b border-stone-100 px-2 py-1 text-right font-mono text-stone-800">
                        {fmt(r.expected ?? expected[obj])}
                      </td>
                      <td className="border-b border-stone-100 px-2 py-1 text-right font-mono text-red-700">
                        {fmt(r.worst_case)}
                      </td>
                      <td className="border-b border-stone-100 px-2 py-1 text-right font-mono text-emerald-700">
                        {fmt(r.best_case)}
                      </td>
                      <td className="border-b border-stone-100 px-2 py-1 text-right font-mono text-amber-700">
                        {fmt(cvarVal)}
                        {cvarKey && (
                          <span className="ml-1 text-[9px] text-stone-400">
                            {cvarKey.replace("cvar_", "")}%
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Regret declines the join outright when the base and scenario runs describe
          different models — say so, rather than let the block vanish as if the lens
          simply didn't apply. */}
      {regret?.available === false && regret.reason && regret.note && (
        <div className="mt-3 border-t border-stone-100 pt-3">
          <Banner tone="amber" label="Regret withheld">
            {regret.note}
          </Banner>
        </div>
      )}

      {/* Minimax-regret robustness — a distinct lens from CVaR (per-solution, not
          per-objective): how much worse than the best achievable in hindsight. */}
      {regret?.available && (allRegretRows.length > 0 || regret.minimax_choice) && (
        <div className="mt-3 border-t border-stone-100 pt-3">
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-stone-600">
            Minimax-regret robustness
          </div>

          {/* Degradation findings from the engine — rendered verbatim (the engine owns
              the wording; both notes are always set alongside their flags). Saturation:
              the ranking is an all-1.0 tie, so the table is suppressed. Wipeout: a
              scenario no base plan survives is excluded from the ranking; the table
              below stays meaningful. The two can co-occur. */}
          {regret.saturated && regret.saturation_note && (
            <Banner tone="amber" label="Regret saturated">
              {regret.saturation_note}
            </Banner>
          )}

          {wipeoutList && regret.wipeout_note && (
            <Banner tone="amber" label={`No base plan survives ${wipeoutList}`}>
              {regret.wipeout_note}
            </Banner>
          )}

          {regret.minimax_choice && (
            <Banner tone="indigo" label="Minimax choice">
              solution #{regret.minimax_choice.solution_id} · worst-case regret{" "}
              <span className="font-mono">{pct(regret.minimax_choice.max_regret)}</span>
              <span className="ml-1 text-[10px] text-indigo-500">
                {wipeoutList
                  ? `(lowest achievable across the ranked scenarios — excludes ${wipeoutList})`
                  : "(lowest achievable)"}
              </span>
            </Banner>
          )}

          {!regret.saturated && regretRows.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs">
                <thead className="text-stone-500">
                  <tr>
                    <th className="border-b border-stone-200 px-2 py-1 text-left font-normal">
                      solution
                    </th>
                    <th className="border-b border-stone-200 px-2 py-1 text-right font-normal">
                      max regret
                    </th>
                    <th className="border-b border-stone-200 px-2 py-1 text-right font-normal">
                      mean regret
                    </th>
                    <th className="border-b border-stone-200 px-2 py-1 text-center font-normal">
                      feasible
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {regretRows.map((s) => {
                    const isPick = minimaxId !== null && s.solution_id === minimaxId;
                    return (
                      <tr
                        key={s.solution_id}
                        className={isPick ? "bg-indigo-50/60" : "hover:bg-stone-50"}
                      >
                        <td className="border-b border-stone-100 px-2 py-1 font-mono text-stone-700">
                          #{s.solution_id}
                          {isPick && (
                            <span className="ml-1 text-[9px] uppercase text-indigo-500">minimax</span>
                          )}
                        </td>
                        <td
                          className="border-b border-stone-100 px-2 py-1 text-right font-mono text-amber-700"
                          title={byScenarioTitle(s.by_scenario)}
                        >
                          {pct(s.max_regret)}
                        </td>
                        <td className="border-b border-stone-100 px-2 py-1 text-right font-mono text-stone-700">
                          {pct(s.mean_regret)}
                        </td>
                        <td className="border-b border-stone-100 px-2 py-1 text-center">
                          <FeasibilityMark row={s} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {hiddenRegretCount > 0 && (
                <div className="mt-1 text-[10px] text-stone-400">
                  +{hiddenRegretCount} more · showing the {MAX_REGRET_ROWS} lowest-regret solutions
                </div>
              )}
            </div>
          )}

          {regret.per_objective && Object.keys(regret.per_objective).length > 0 && (
            <div className="mt-2 text-[11px] leading-relaxed text-stone-500">
              <span className="text-stone-400">
                lowest worst-case regret per objective{wipeoutList ? ` (ranked scenarios — excludes ${wipeoutList})` : ""}:
              </span>{" "}
              {Object.entries(regret.per_objective).map(([obj, r], i) => (
                <span key={obj}>
                  {i > 0 && " · "}
                  <span className="text-stone-600">{obj}</span> → #{r.achieved_by_solution_id}{" "}
                  <span className="font-mono text-amber-700">{pct(r.min_max_regret)}</span>
                </span>
              ))}
            </div>
          )}

          {regret.note && (
            <div className="mt-1.5 text-[10px] leading-snug text-stone-400">{regret.note}</div>
          )}
        </div>
      )}
    </div>
  );
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toString();
    return v.toFixed(2);
  }
  return String(v);
}

// Regret is a [0,1] fraction of each scenario's achievable objective range given up.
function pct(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

// Per-scenario regret breakdown, surfaced as a hover title on the max-regret cell.
function byScenarioTitle(byScenario?: Record<string, number>): string | undefined {
  if (!byScenario || Object.keys(byScenario).length === 0) return undefined;
  return Object.entries(byScenario)
    .map(([name, v]) => `${name}: ${pct(v)}`)
    .join(" · ");
}
