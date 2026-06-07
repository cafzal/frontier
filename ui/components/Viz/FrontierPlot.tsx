"use client";

import { useMemo, useState } from "react";
import dynamic from "next/dynamic";
import type { ScatterVizData, ScatterPoint, ScatterProvenance, ObjectiveMeta } from "@/lib/viz-data";
import { useChatAction } from "@/lib/chat-action";

// Plotly needs window/WebGL — load client-side only.
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const ROLE_COLOR = {
  balanced: "#a855f7",
  inflection: "#f59e0b",
  extreme: "#ef4444",
  other: "#78716c",
} as const;
type Role = keyof typeof ROLE_COLOR;

const SELECTED_COLOR = "#111827";
// Exact-certified accent — emerald reads as "verified" and is none of the role hues, so the
// certified overlay never collides with balanced/inflection/extreme/other or the curated ring.
const CERTIFIED_COLOR = "#059669";
const CERTIFIED_RING = "#047857";

// Opacity is the primary certified signal (solid = certified, faded = not-yet-certified). It
// only carries meaning when both layers coexist (a heuristic frontier with an exact overlay);
// a frontier with no overlay renders solid so it isn't needlessly washed out.
const ALPHA_CERTIFIED = 1;
const ALPHA_HEURISTIC = 0.7; // not-yet-certified — faded but clearly legible
const ALPHA_DOMINATED = 0.45; // exact strictly beats it — the faintest tier, still visible

// Bake alpha into the fill color so per-point opacity works identically across scattergl,
// scatter3d, and parcoords (marker.opacity-as-an-array isn't reliable on scattergl).
function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

type Selected = {
  id: number;
  name: string | null;
  values: Record<string, number>;
  certified?: boolean; // point is exact-certified (overlay diamond, or any point in an exact view)
} | null;
type Effective = "scatter2d" | "scatter3d" | "parcoords";
// Selectable chart view. "color2d" (3-objective only) is a 2D scatter with the chosen objective
// encoded as marker color and the other two as axes — a first-class alternative to 3D / parcoords.
type ViewMode = "scatter" | "color2d" | "parcoords";

/**
 * Frontier solutions, rendered by dimensionality:
 *   2 objectives  → 2D scatter (WebGL, zoom + click-select)
 *   3 objectives  → 3D scatter (orbit/rotate, zoom, click-select)
 *   ≥4 objectives → parallel coordinates (axis brushing)
 * A toggle lets the user force parallel coordinates at any dimensionality (≤3),
 * since PC is often the most interpretable multi-objective view. Points are
 * role-colored (balanced / inflection / extreme / other); clicking a scatter
 * point reads out its id + curated name.
 *
 * Exact certification shows as solidity: when the rendered frontier is the exact run
 * (`provenance.kind === "exact"`) every point is solid; when a heuristic frontier carries an
 * `exact_overlay`, the heuristic field fades, the exact-certified points overlay as solid
 * emerald diamonds, and the heuristic points the exact front dominates fade the most. Opacity is
 * orthogonal to the role hue, so the two channels compose (a certified balanced point is a solid
 * purple; an uncertified one a faded purple).
 */
function inRanges(v: number, cr: number[] | number[][] | undefined): boolean {
  if (!cr || cr.length === 0) return true;
  const ranges = (Array.isArray(cr[0]) ? cr : [cr]) as number[][];
  return ranges.some((r) => v >= Math.min(r[0], r[1]) && v <= Math.max(r[0], r[1]));
}

export function FrontierPlot({ data }: { data: ScatterVizData }) {
  const [selected, setSelected] = useState<Selected>(null);
  const [brushIds, setBrushIds] = useState<number[]>([]);
  const chat = useChatAction();
  const objs = data.objectives;
  const nObj = objs.length;

  const prov = data.provenance;
  const overlay = data.exact_overlay ?? null;
  const isExactView = prov?.kind === "exact";
  const hasOverlay = !!overlay && overlay.points.length > 0;
  const dominated = useMemo(() => new Set(overlay?.dominated_ids ?? []), [overlay]);
  // A base point's fill opacity: solid on a certified (or overlay-less) frontier; faded when an
  // exact overlay exists, and faintest when the exact front dominates it.
  const alphaOf = (id: number): number =>
    isExactView || !hasOverlay
      ? ALPHA_CERTIFIED
      : dominated.has(id)
        ? ALPHA_DOMINATED
        : ALPHA_HEURISTIC;

  const canScatter = nObj <= 3;
  const canColor2d = nObj === 3; // need exactly one objective to encode as color + two as axes
  // Color-by view: encode one objective as marker color on a 2D scatter, the other two as axes —
  // a first-class view alongside 3D / parcoords for any 3-objective frontier, switchable in the UI
  // via the toggle + a color-objective dropdown. viz_data.color_objective (e.g. from the render
  // route) just seeds the initial selection; absent it, a 3-obj frontier opens in 3D as before.
  const colorHint = data.color_objective ?? null;
  const [mode, setMode] = useState<ViewMode>(colorHint && canColor2d ? "color2d" : "scatter");
  const [colorSel, setColorSel] = useState<string>(
    colorHint && objs.some((o) => o.name === colorHint)
      ? colorHint
      : canColor2d
        ? objs[nObj - 1].name
        : "",
  );
  const colorActive = canColor2d && mode === "color2d" && objs.some((o) => o.name === colorSel);
  const colorObj = colorActive ? colorSel : null;
  const axisObjs = useMemo(
    () => (colorObj ? objs.filter((o) => o.name !== colorObj) : objs),
    [objs, colorObj],
  );
  const effective: Effective = colorActive
    ? "scatter2d"
    : !canScatter || mode === "parcoords"
      ? "parcoords"
      : nObj <= 2
        ? "scatter2d"
        : "scatter3d";

  const roleOf = useMemo(() => {
    const extremeIds = new Set<number>();
    for (const o of objs) {
      const e = data.extremes[o.name];
      if (e) {
        extremeIds.add(e.best_id);
        extremeIds.add(e.worst_id);
      }
    }
    const inflSet = new Set(data.inflection_ids);
    return (id: number): Role =>
      id === data.balanced_id
        ? "balanced"
        : inflSet.has(id)
          ? "inflection"
          : extremeIds.has(id)
            ? "extreme"
            : "other";
  }, [data, objs]);

  const nameOf = (p: ScatterPoint) => p.name ?? null;

  const { plotData, layout } = useMemo(() => {
    const is3d = effective === "scatter3d";
    // A point with a name is a curated pick — emphasize it (larger + bold ring)
    // so curation reads at a glance, same intent as the bold lines in the PC views.
    const isCurated = (p: ScatterPoint) => !!nameOf(p);
    // Fill carries both role hue and certified opacity; selection overrides to a solid dark dot.
    const colors = data.points.map((p) =>
      p.solution_id === selected?.id
        ? SELECTED_COLOR
        : withAlpha(ROLE_COLOR[roleOf(p.solution_id)], alphaOf(p.solution_id))
    );
    const sizes = data.points.map((p) => {
      if (p.solution_id === selected?.id) return is3d ? 7 : 16;
      if (isCurated(p)) return is3d ? 6.5 : 15;
      const big = roleOf(p.solution_id) !== "other";
      return is3d ? (big ? 4.5 : 3) : big ? 11 : 7;
    });
    const lineWidths = data.points.map((p) =>
      p.solution_id === selected?.id ? 2.5 : isCurated(p) ? 2 : 0.5
    );
    const lineColors = data.points.map((p) =>
      p.solution_id === selected?.id ? "#000000" : isCurated(p) ? SELECTED_COLOR : "#ffffff"
    );
    const customdata = data.points.map((p) => [p.solution_id, nameOf(p) ?? "—", "base"]);

    // Color-by view: one 2D scattergl trace, the chosen objective encoded as a continuous
    // marker color (with a colorbar); the two remaining objectives are the axes. Short-circuits
    // the dimensional branches below so a 3-objective frontier reads as a clean colored 2D.
    if (colorActive) {
      const xName = axisObjs[0].name;
      const yName = axisObjs[1].name;
      const cvals = data.points.map((p) => p.values[colorObj!]);
      return {
        plotData: [
          {
            type: "scattergl",
            mode: "markers",
            x: data.points.map((p) => p.values[xName]),
            y: data.points.map((p) => p.values[yName]),
            customdata,
            marker: {
              color: cvals,
              colorscale: "Viridis",
              showscale: true,
              colorbar: { title: { text: colorObj, side: "right" }, thickness: 12, len: 0.9, outlinewidth: 0 },
              size: sizes,
              line: { width: lineWidths, color: lineColors },
            },
            hovertemplate:
              `#%{customdata[0]} %{customdata[1]}<br>` +
              `${xName}: %{x:.2f}<br>${yName}: %{y:.2f}<br>${colorObj}: %{marker.color:.2f}<extra></extra>`,
          },
        ],
        layout: {
          xaxis: { title: { text: `${xName} (${axisObjs[0].direction})` } },
          yaxis: { title: { text: `${yName} (${axisObjs[1].direction})` } },
          margin: { l: 60, r: 20, t: 10, b: 50 },
          dragmode: "zoom",
          hovermode: "closest",
        },
      };
    }

    // The exact-certified overlay drawn over a faded heuristic field — a separate solid
    // emerald-diamond trace (built for scatter only; parcoords can't overlay a second trace).
    // Prominent and intuitive, but a lighter touch than curated (thin ring, no heavy dark ring).
    const overlayTrace2d =
      hasOverlay && effective === "scatter2d"
        ? {
            type: "scattergl",
            mode: "markers",
            x: overlay!.points.map((p) => p.values[objs[0].name]),
            y: overlay!.points.map((p) => p.values[objs[1]?.name ?? objs[0].name]),
            customdata: overlay!.points.map((p) => [p.solution_id, "exact-certified", "exact"]),
            marker: {
              color: CERTIFIED_COLOR,
              size: 11,
              symbol: "diamond",
              line: { width: 1.5, color: CERTIFIED_RING },
            },
            hovertemplate:
              `✓ exact-certified #%{customdata[0]}<br>` +
              `${objs[0].name}: %{x:.2f}<br>${objs[1]?.name ?? objs[0].name}: %{y:.2f}<extra></extra>`,
          }
        : null;

    if (effective === "scatter2d") {
      const yName = objs[1]?.name ?? objs[0].name;
      return {
        plotData: [
          {
            type: "scattergl",
            mode: "markers",
            x: data.points.map((p) => p.values[objs[0].name]),
            y: data.points.map((p) => p.values[yName]),
            customdata,
            marker: { color: colors, size: sizes, line: { width: lineWidths, color: lineColors } },
            hovertemplate:
              `#%{customdata[0]} %{customdata[1]}<br>` +
              `${objs[0].name}: %{x:.2f}<br>${yName}: %{y:.2f}<extra></extra>`,
          },
          ...(overlayTrace2d ? [overlayTrace2d] : []),
        ],
        layout: {
          xaxis: { title: { text: `${objs[0].name} (${objs[0].direction})` } },
          yaxis: { title: { text: `${yName} (${objs[1]?.direction ?? objs[0].direction})` } },
          margin: { l: 60, r: 20, t: 10, b: 50 },
          dragmode: "zoom",
          hovermode: "closest",
        },
      };
    }

    if (effective === "scatter3d") {
      const overlayTrace3d = hasOverlay
        ? {
            type: "scatter3d",
            mode: "markers",
            x: overlay!.points.map((p) => p.values[objs[0].name]),
            y: overlay!.points.map((p) => p.values[objs[1].name]),
            z: overlay!.points.map((p) => p.values[objs[2].name]),
            customdata: overlay!.points.map((p) => [p.solution_id, "exact-certified", "exact"]),
            marker: { color: CERTIFIED_COLOR, size: 5, symbol: "diamond", line: { width: 0 } },
            hovertemplate:
              `✓ exact-certified #%{customdata[0]}<br>` +
              `${objs[0].name}: %{x:.2f}<br>${objs[1].name}: %{y:.2f}<br>${objs[2].name}: %{z:.2f}<extra></extra>`,
          }
        : null;
      return {
        plotData: [
          {
            type: "scatter3d",
            mode: "markers",
            x: data.points.map((p) => p.values[objs[0].name]),
            y: data.points.map((p) => p.values[objs[1].name]),
            z: data.points.map((p) => p.values[objs[2].name]),
            customdata,
            marker: { color: colors, size: sizes, line: { width: 0 } },
            hovertemplate:
              `#%{customdata[0]} %{customdata[1]}<br>` +
              `${objs[0].name}: %{x:.2f}<br>${objs[1].name}: %{y:.2f}<br>${objs[2].name}: %{z:.2f}<extra></extra>`,
          },
          ...(overlayTrace3d ? [overlayTrace3d] : []),
        ],
        layout: {
          scene: {
            xaxis: { title: { text: objs[0].name } },
            yaxis: { title: { text: objs[1].name } },
            zaxis: { title: { text: objs[2].name } },
          },
          margin: { l: 0, r: 0, t: 0, b: 0 },
        },
      };
    }

    // parallel coordinates (role-colored lines, axis brushing) — any nObj ≥ 2.
    // PC can't overlay the exact trace, so certification rides on the chip + the scatter view;
    // here the one available cue is to grey the heuristic lines the exact front dominates.
    const roleRank: Record<Role, number> = { other: 0, extreme: 1, inflection: 2, balanced: 3 };
    const dimDominated = hasOverlay && !isExactView;
    return {
      plotData: [
        {
          type: "parcoords",
          labelfont: { size: 12 },
          dimensions: objs.map((o) => ({
            label: `${o.name} ${o.direction === "maximize" ? "↑" : "↓"}`,
            range: [o.min, o.max],
            values: data.points.map((p) => p.values[o.name]),
          })),
          line: dimDominated
            ? {
                // Reserve the scale's bottom stop for dominated (greyed); roles occupy 0.25–1.
                color: data.points.map((p) =>
                  dominated.has(p.solution_id) ? 0 : 0.25 + 0.25 * roleRank[roleOf(p.solution_id)]
                ),
                cmin: 0,
                cmax: 1,
                colorscale: [
                  [0, "#d6d3d1"],
                  [0.25, ROLE_COLOR.other],
                  [0.5, ROLE_COLOR.extreme],
                  [0.75, ROLE_COLOR.inflection],
                  [1, ROLE_COLOR.balanced],
                ],
              }
            : {
                color: data.points.map((p) => roleRank[roleOf(p.solution_id)]),
                colorscale: [
                  [0, ROLE_COLOR.other],
                  [0.33, ROLE_COLOR.extreme],
                  [0.66, ROLE_COLOR.inflection],
                  [1, ROLE_COLOR.balanced],
                ],
              },
        },
      ],
      layout: { margin: { l: 80, r: 60, t: 56, b: 20 } },
    };
  }, [data, objs, effective, roleOf, selected, hasOverlay, isExactView, overlay, dominated, colorActive, axisObjs, colorObj]);

  function onClick(e: { points?: Array<{ customdata?: [number, string, string] }> }) {
    const pt = e?.points?.[0];
    if (!pt?.customdata) return;
    const id = Number(pt.customdata[0]);
    const layer = pt.customdata[2];
    if (layer === "exact") {
      // An exact-overlay diamond — its values live in the overlay, not data.points.
      const op = overlay?.points.find((p) => p.solution_id === id);
      setSelected({ id, name: null, values: op ? { ...op.values } : {}, certified: true });
      return;
    }
    const rawName = pt.customdata[1];
    const found = data.points.find((p) => p.solution_id === id);
    setSelected({
      id,
      name: rawName && rawName !== "—" ? String(rawName) : null,
      values: found ? { ...found.values } : {},
      certified: isExactView,
    });
  }

  // Parallel-coords selection is brushing: read the per-axis constraint ranges
  // and keep the solutions inside all of them (for the "curate selected" action).
  function onUpdate(figure: {
    data?: Array<{ dimensions?: Array<{ constraintrange?: number[] | number[][] }> }>;
  }) {
    if (effective !== "parcoords") return;
    const dims = figure?.data?.[0]?.dimensions;
    if (!dims) return;
    const next = dims.some((d) => d?.constraintrange)
      ? data.points
          .filter((p) => objs.every((o, i) => inRanges(p.values[o.name], dims[i]?.constraintrange)))
          .map((p) => p.solution_id)
      : [];
    const same = next.length === brushIds.length && next.every((v, i) => v === brushIds[i]);
    if (!same) setBrushIds(next);
  }

  const kind = colorActive
    ? `2D scatter · color ${colorObj}`
    : effective === "scatter2d"
      ? "2D scatter"
      : effective === "scatter3d"
        ? "3D scatter"
        : "parallel coordinates";

  // An exact-overlay diamond was selected (vs. a base point in an exact view) — curating it must
  // target the exact overlay, so the agent picks the certified solution, not a same-id base one.
  const overlayPick = !!selected?.certified && !isExactView;

  return (
    <div data-viz="frontier" className="my-3 rounded border border-stone-200 bg-white p-3">
      <div className="mb-1 flex items-center justify-between text-[10px] text-stone-500">
        <div className="flex items-center gap-2">
          <span>
            {kind} · {data.points.length} solutions · {nObj} objectives
          </span>
          <CertChip prov={prov} hasOverlay={hasOverlay} />
        </div>
        <div className="flex items-center gap-3">
          {canScatter && <Toggle mode={mode} setMode={setMode} nObj={nObj} />}
          {colorActive ? (
            <ColorPicker objs={objs} value={colorSel} onChange={setColorSel} />
          ) : (
            <Legend
              hasCurated={data.points.some((p) => !!p.name)}
              showCertified={hasOverlay}
              showDominated={hasOverlay && dominated.size > 0}
            />
          )}
        </div>
      </div>
      <Plot
        data={plotData as never}
        layout={
          {
            autosize: true,
            height: effective === "scatter3d" ? 460 : 380,
            showlegend: false,
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            font: { size: 11, color: "#292524" },
            ...layout,
          } as never
        }
        config={
          {
            displaylogo: false,
            responsive: true,
            modeBarButtonsToRemove: ["lasso2d", "select2d"],
          } as never
        }
        useResizeHandler
        style={{ width: "100%" }}
        onClick={onClick as never}
        onUpdate={onUpdate as never}
      />
      {selected && effective !== "parcoords" && (
        <div className="mt-1 flex items-center gap-2 text-[11px] text-stone-700">
          <span>
            <span className="font-semibold">
              Selected #{selected.id}
              {selected.certified ? (
                <span style={{ color: CERTIFIED_RING }}> ✓ exact-certified</span>
              ) : null}
              {selected.name ? ` · ${selected.name}` : ""}
            </span>
            {" — "}
            {objs.map((o) => `${o.name} ${selected.values[o.name]?.toFixed(2) ?? "—"}`).join(" · ")}
          </span>
          {chat && (
            <button
              type="button"
              disabled={chat.streaming}
              onClick={() =>
                chat.sendMessage(
                  selected.name
                    ? `Remove the curated solution "${selected.name}" (#${selected.id}) from curation.`
                    : overlayPick
                      ? `Curate the exact-certified solution #${selected.id} from the exact overlay (source="exact").`
                      : `Curate solution #${selected.id}.`
                )
              }
              className="shrink-0 rounded bg-stone-800 px-1.5 py-0.5 text-[10px] text-white hover:bg-stone-700 disabled:bg-stone-300"
            >
              {selected.name ? "− Uncurate" : overlayPick ? "+ Curate (exact)" : "+ Curate"}
            </button>
          )}
        </div>
      )}
      {effective === "parcoords" && (
        <div className="mt-1 flex items-center gap-2 text-[10px] text-stone-400">
          <span>Drag along an axis to brush; double-click to clear.</span>
          {hasOverlay && (
            <span>· exact-certified overlay reads best in the scatter view; dominated lines greyed</span>
          )}
          {chat && brushIds.length > 0 && (
            <button
              type="button"
              disabled={chat.streaming}
              onClick={() =>
                chat.sendMessage(
                  `Curate the ${brushIds.length} brushed solution${brushIds.length > 1 ? "s" : ""}: ` +
                    brushIds
                      .slice(0, 12)
                      .map((id) => `#${id}`)
                      .join(", ") +
                    (brushIds.length > 12 ? `, and ${brushIds.length - 12} more` : "") +
                    "."
                )
              }
              className="shrink-0 rounded bg-stone-800 px-1.5 py-0.5 text-[10px] text-white hover:bg-stone-700 disabled:bg-stone-300"
            >
              + Curate selected ({brushIds.length})
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// Frontier provenance at a glance: exact-certified (emerald) vs heuristic, with the overlay state.
function CertChip({ prov, hasOverlay }: { prov?: ScatterProvenance; hasOverlay: boolean }) {
  if (!prov) return null;
  if (prov.kind === "exact") {
    return (
      <span
        className="rounded px-1 py-0.5 font-medium"
        style={{ background: "#d1fae5", color: CERTIFIED_RING }}
      >
        ✓ Exact-certified · {prov.solver}
      </span>
    );
  }
  return (
    <span
      className={`rounded bg-stone-100 px-1 py-0.5 ${hasOverlay ? "text-stone-600" : "text-stone-400"}`}
    >
      {hasOverlay ? "Heuristic · exact overlay" : "Heuristic"}
    </span>
  );
}

function Toggle({
  mode,
  setMode,
  nObj,
}: {
  mode: ViewMode;
  setMode: (m: ViewMode) => void;
  nObj: number;
}) {
  const btn = (m: ViewMode, label: string) => (
    <button
      type="button"
      onClick={() => setMode(m)}
      className={`rounded px-1.5 py-0.5 ${
        mode === m ? "bg-stone-800 text-white" : "bg-stone-100 text-stone-600 hover:bg-stone-200"
      }`}
    >
      {label}
    </button>
  );
  return (
    <div className="flex items-center gap-1 text-[10px]">
      {btn("scatter", nObj <= 2 ? "2D" : "3D")}
      {/* 3 objectives can also be a 2D scatter colored by the 3rd (axes = the other two). */}
      {nObj === 3 && btn("color2d", "2D")}
      {btn("parcoords", "‖ PC")}
    </div>
  );
}

// Pick which objective the color2d view encodes as marker color (the other two become the axes).
function ColorPicker({
  objs,
  value,
  onChange,
}: {
  objs: ObjectiveMeta[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center gap-1 text-[10px] text-stone-500">
      color
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-stone-200 bg-white px-1 py-0.5 text-stone-700"
      >
        {objs.map((o) => (
          <option key={o.name} value={o.name}>
            {o.name}
          </option>
        ))}
      </select>
    </label>
  );
}

function Legend({
  hasCurated = false,
  showCertified = false,
  showDominated = false,
}: {
  hasCurated?: boolean;
  showCertified?: boolean;
  showDominated?: boolean;
}) {
  const items: Array<[Role, string]> = [
    ["balanced", "balanced"],
    ["inflection", "inflection"],
    ["extreme", "extreme"],
    ["other", "other"],
  ];
  return (
    <div className="flex items-center gap-3 text-[10px] text-stone-600">
      {items.map(([role, label]) => (
        <span key={role} className="flex items-center gap-1">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: ROLE_COLOR[role] }}
          />
          {label}
        </span>
      ))}
      {showCertified && (
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-2 w-2 rotate-45"
            style={{ background: CERTIFIED_COLOR, border: `1px solid ${CERTIFIED_RING}` }}
          />
          exact-certified
        </span>
      )}
      {showDominated && (
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: withAlpha("#78716c", ALPHA_DOMINATED) }}
          />
          exact dominates
        </span>
      )}
      {hasCurated && (
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-2.5 w-2.5 rounded-full bg-stone-200"
            style={{ border: `2px solid ${SELECTED_COLOR}` }}
          />
          curated
        </span>
      )}
    </div>
  );
}
