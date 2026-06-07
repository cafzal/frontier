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
import { extractVizData, type ScatterVizData } from "@/lib/viz-data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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
  const url = new URL(req.url);
  const problem_id = url.searchParams.get("problem_id");
  const source = url.searchParams.get("source") ?? undefined; // "exact" → exact-only; omit → heuristic + overlay (combined)
  const scenario = url.searchParams.get("scenario") ?? undefined;
  const color = url.searchParams.get("color") ?? undefined; // a 3rd objective to encode as marker color (2D color-by view)
  if (!problem_id) {
    return Response.json({ error: "problem_id query param required" }, { status: 400 });
  }

  let mcp: MCPClient | null = null;
  try {
    mcp = new MCPClient({ name: "frontier-render", version: "0.1.0" });
    const transport = new SSEClientTransport(new URL(FRONTIER_MCP_URL), {
      fetch: authedFetch(FRONTIER_MCP_TOKEN) as any,
    });
    await mcp.connect(transport);

    const args: Record<string, unknown> = { action: "tradeoffs", problem_id };
    if (source) args.source = source;
    if (scenario) args.scenario = scenario;

    const result = await mcp.callTool({ name: "explore", arguments: args });
    const text = (Array.isArray(result.content) ? result.content : [])
      .map((c: any) => (typeof c?.text === "string" ? c.text : ""))
      .join("\n");

    const viz = extractVizData(text).find((v) => v.type === "scatter") as
      | ScatterVizData
      | undefined;
    if (!viz) {
      return Response.json(
        {
          error:
            "no scatter viz_data — the run may not exist, or the problem has >3 objectives (parcoords only)",
          raw: text.slice(0, 400),
        },
        { status: 404 },
      );
    }
    // Optional 2D color-by view: encode a chosen objective as marker color (the route's way to
    // get a 2D-colored-by-3rd-objective scatter the chat agent's viz_data can't express).
    if (color && viz.objectives.some((o) => o.name === color)) {
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
