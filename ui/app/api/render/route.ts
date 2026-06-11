/**
 * Render-only data endpoint for demo capture.
 *
 * Decouples COMPUTE (the engine, driven deterministically) from RENDER (the real
 * FrontierPlot component), so a capture harness can screenshot any solved run in a
 * chosen view without the chat agent's stochastic viz_data choices. See
 * `.claude/plans/demo-capture-lessons.md` §C.
 *
 *   GET /api/render?problem_id=<id>               → heuristic frontier + exact overlay
 *                                                    (the combined "emerald diamonds over faded
 *                                                     dominated" view, on a ≤3-obj problem)
 *   GET /api/render?problem_id=<id>&source=exact  → exact-only certified frontier
 *   GET /api/render?problem_id=<id>&scenario=<s>  → a scenario's frontier
 *
 * Returns { vizData: ScatterVizData }. Talks to the same engine the chat UI uses
 * (FRONTIER_MCP_URL), via the same MCP client + auth, calling `explore tradeoffs`.
 */
import { Client as MCPClient } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { extractVizData } from "@/lib/viz-data";
import { renderRateLimiter, clientKey, tooManyRequests } from "@/lib/rate-limit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// This endpoint is READ-ONLY: only these explore actions (which return viz_data) are
// allowed. The `explore` tool also has state-mutating actions (feedback, curate) that
// must never be reachable from an unsanitized query param.
const RENDER_ACTIONS = new Set(["tradeoffs", "scenario_results"]);

const FRONTIER_MCP_URL =
  process.env.FRONTIER_MCP_URL ??
  (process.env.FRONTIER_MCP_HOST
    ? `https://${process.env.FRONTIER_MCP_HOST}/sse`
    : "http://localhost:8000/sse");
const FRONTIER_MCP_TOKEN = process.env.FRONTIER_MCP_TOKEN;

// Mirror agent-runtime's authedFetch: attach the engine bearer token when set.
function authedFetch(token?: string) {
  return (url: string | URL, init: RequestInit = {}) => {
    const headers = new Headers(init.headers ?? {});
    if (token) headers.set("authorization", `Bearer ${token}`);
    return fetch(url, { ...init, headers });
  };
}

export async function GET(req: Request) {
  const limit = renderRateLimiter.check(clientKey(req));
  if (!limit.ok) return tooManyRequests(limit);

  const url = new URL(req.url);
  const problem_id = url.searchParams.get("problem_id");
  const action = url.searchParams.get("action") ?? "tradeoffs"; // "tradeoffs" → scatter; "scenario_results" → scenario_summary panel
  const source = url.searchParams.get("source") ?? undefined; // "exact" → exact-only; omit → heuristic + overlay (combined)
  const scenario = url.searchParams.get("scenario") ?? undefined;
  const color = url.searchParams.get("color") ?? undefined; // a 3rd objective to encode as marker color (2D color-by view)
  if (!problem_id) {
    return Response.json({ error: "problem_id query param required" }, { status: 400 });
  }
  if (!RENDER_ACTIONS.has(action)) {
    return Response.json(
      { error: `unsupported action '${action}'; this endpoint serves: ${[...RENDER_ACTIONS].join(", ")}` },
      { status: 400 },
    );
  }

  let mcp: MCPClient | null = null;
  try {
    mcp = new MCPClient({ name: "frontier-render", version: "0.1.0" });
    const transport = new SSEClientTransport(new URL(FRONTIER_MCP_URL), {
      fetch: authedFetch(FRONTIER_MCP_TOKEN) as any,
    });
    await mcp.connect(transport);

    const args: Record<string, unknown> = { action, problem_id };
    if (source) args.source = source;
    if (scenario) args.scenario = scenario;

    const result = await mcp.callTool({ name: "explore", arguments: args });
    const text = (Array.isArray(result.content) ? result.content : [])
      .map((c: any) => (typeof c?.text === "string" ? c.text : ""))
      .join("\n");

    // Which viz_data block to surface depends on the explore action driving this render.
    const wantType = action === "scenario_results" ? "scenario_summary" : "scatter";
    const vizList = extractVizData(text);
    const viz = vizList.find((v) => v.type === wantType) ?? vizList[0];
    if (!viz) {
      return Response.json(
        {
          error: `no ${wantType} viz_data — the run may not exist, or the problem shape doesn't produce it`,
          raw: text.slice(0, 400),
        },
        { status: 404 },
      );
    }
    // Optional 2D color-by view (scatter only): encode a chosen objective as marker color (the
    // route's way to get a 2D-colored-by-3rd-objective scatter the chat agent's viz_data can't express).
    if (color && viz.type === "scatter" && viz.objectives.some((o) => o.name === color)) {
      viz.color_objective = color;
    }
    return Response.json({ vizData: viz });
  } catch (err) {
    return Response.json(
      { error: err instanceof Error ? err.message : String(err) },
      { status: 500 },
    );
  } finally {
    if (mcp) {
      try {
        await mcp.close();
      } catch {
        /* ignore */
      }
    }
  }
}
