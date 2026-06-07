"use client";

/**
 * Render-only chart page for demo capture — the real FrontierPlot with no chat chrome.
 * Drive it from the capture harness:
 *   /render?problem_id=<id>              → heuristic frontier + exact overlay
 *                                          (2-obj → emerald-certified diamonds over faded dominated)
 *   /render?problem_id=<id>&source=exact → exact-only certified frontier
 *   /render?problem_id=<id>&scenario=<s> → a scenario's frontier
 *
 * Reads the query string client-side (window.location.search) so no Suspense
 * boundary is needed. See `.claude/plans/demo-capture-lessons.md` §C.
 */
import { useEffect, useState } from "react";
import { FrontierPlot } from "@/components/Viz/FrontierPlot";
import type { ScatterVizData } from "@/lib/viz-data";

export default function RenderPage() {
  const [viz, setViz] = useState<ScatterVizData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/api/render${window.location.search}`)
      .then((r) => r.json())
      .then((d) => (d.error ? setErr(d.error) : setViz(d.vizData)))
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) {
    return (
      <div style={{ padding: 24, color: "#b91c1c", fontFamily: "ui-monospace, monospace" }}>
        render error: {err}
      </div>
    );
  }
  if (!viz) {
    return <div style={{ padding: 24, color: "#78716c" }}>loading…</div>;
  }

  return (
    <div
      data-render-root
      style={{ padding: 24, maxWidth: 920, margin: "0 auto", background: "#fff" }}
    >
      <FrontierPlot data={viz} />
    </div>
  );
}
