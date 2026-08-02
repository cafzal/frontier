/**
 * Pluggable agent runtime — single env var (`AGENT_BACKEND`) selects the adapter.
 *
 *   messages-api        Anthropic Messages API + server-side MCP connector (default)
 *   anthropic-local     Anthropic Messages API + client-side MCP loop
 *   openai-compatible   Any OpenAI Chat Completions-compatible provider +
 *                       client-side MCP loop (point OPENAI_BASE_URL anywhere)
 *
 * Provider lock-in is bounded to this file. The chat shell, MCP server,
 * skills, and every other surface are provider-agnostic. Adapters emit
 * Anthropic-shaped SSE events (content_block_*) so the UI is uniform across
 * backends — switching providers is one env var.
 */

import Anthropic from "@anthropic-ai/sdk";
import { Client as MCPClient } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { SYSTEM_PROMPT } from "./system-prompt";
import { compactHistory } from "./compaction";
import { cachedSystem, cachedTools, withCacheBreakpoint, contextManagement } from "./anthropic-request";
import {
  buildResponsesTools,
  translateMessagesToResponsesInput,
  type ResponsesInputItem,
} from "./openai-responses";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string | unknown;
};

export type AgentRuntime = {
  stream: (messages: ChatMessage[]) => Promise<ReadableStream<Uint8Array>>;
};

// ─── history compaction budget ──────────────────────────────────────────────
// The UI resends the whole transcript every turn with no compaction of its own, so a long
// session grows until it hits the model's context window and the request hard-fails. Before
// each call we trim older middle turns (compaction.ts) when (system + history) approaches a
// budget, pinning the durable index (system prompt), the first user turn, and the recent
// exchange. All knobs are env-tunable; defaults target the prod Anthropic path (Opus, 200k).
const CONTEXT_WINDOW = Number(process.env.AGENT_CONTEXT_WINDOW ?? "200000");
const COMPACTION_OUTPUT_RESERVE = Number(process.env.AGENT_OUTPUT_RESERVE ?? "24000");
const KEEP_RECENT_MESSAGES = Number(process.env.AGENT_KEEP_RECENT_MESSAGES ?? "8");

// Trim history to fit the context window, reserving room for the system prompt (counted
// inside) and `maxTokens` of response. No-op until the budget is approached.
function applyCompaction(
  messages: ChatMessage[],
  systemText: string,
  maxTokens: number,
): ChatMessage[] {
  const budget = Math.max(20000, CONTEXT_WINDOW - maxTokens - COMPACTION_OUTPUT_RESERVE);
  const { messages: out, compacted, dropped } = compactHistory(
    messages,
    systemText,
    budget,
    KEEP_RECENT_MESSAGES,
  );
  if (compacted) {
    console.warn(`[frontier] history compaction: elided ${dropped} middle turn(s) to fit context`);
  }
  return out;
}

const FRONTIER_MCP_URL =
  process.env.FRONTIER_MCP_URL ??
  (process.env.FRONTIER_MCP_HOST
    ? `https://${process.env.FRONTIER_MCP_HOST}/sse`
    : "http://localhost:8000/sse");
// Single shared bearer token gating a hosted engine. Unset = local/ungated.
const FRONTIER_MCP_TOKEN = process.env.FRONTIER_MCP_TOKEN;
const MODEL = process.env.ANTHROPIC_MODEL ?? "claude-opus-4-8";

// ─── shared engine connection ───────────────────────────────────────────────

// Attach the engine bearer token to every MCP request (SSE GET + POST) when
// set; unset = local/ungated engine.
function authedFetch(token: string | undefined) {
  return (url: string | URL, init: RequestInit = {}): Promise<Response> => {
    const headers = new Headers(init.headers ?? {});
    if (token) headers.set("authorization", `Bearer ${token}`);
    return fetch(url, { ...init, headers });
  };
}

// Open an authenticated MCP client to the Frontier engine.
async function openMcpClient(name: string): Promise<MCPClient> {
  const mcp = new MCPClient({ name, version: "0.1.0" });
  const transport = new SSEClientTransport(new URL(FRONTIER_MCP_URL), {
    fetch: authedFetch(FRONTIER_MCP_TOKEN) as any,
  });
  await mcp.connect(transport);
  return mcp;
}

// The MCP server's `instructions` field is the canonical workflow/skill guidance
// a real MCP host (e.g. a coding agent) surfaces on connect. The messages-api
// connector does NOT surface it, so we fetch it ourselves and fold it into the
// system prompt — giving the model the same framing checklist, workflow, and
// aggregation guidance every other surface gets. Instructions are static for
// the server's lifetime, so cache across requests; a failed fetch falls back to
// the bare prompt (non-fatal) and is not cached, so it retries next request.
let cachedServerInstructions: string | null = null;
async function getServerInstructions(): Promise<string> {
  if (cachedServerInstructions !== null) return cachedServerInstructions;
  let mcp: MCPClient | null = null;
  try {
    mcp = await openMcpClient("frontier-web-instructions");
    cachedServerInstructions = mcp.getInstructions() ?? "";
    return cachedServerInstructions;
  } catch {
    return "";
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

// ─── messages-api adapter (default) ─────────────────────────────────────────

const messagesApiAdapter: AgentRuntime = {
  async stream(messages) {
    const apiKey = process.env.ANTHROPIC_API_KEY || process.env.CLAUDE_API_KEY;
    if (!apiKey) throw new Error("ANTHROPIC_API_KEY (or CLAUDE_API_KEY) not set");
    const client = new Anthropic({ apiKey });
    const encoder = new TextEncoder();

    // Use the beta MCP-connector path. Anthropic runs the agent loop server-side,
    // including calling Frontier MCP tools and looping back with results.
    // Per mcp-client-2025-11-20 schema: mcp_servers defines connections,
    // tools array references them via mcp_toolset entries.
    // SDK 0.40.1 partially types these; cast as any to keep the surface clean.
    // Opus 4.7/4.8 use adaptive thinking + output_config.effort; the older
    // budget_tokens form 400s. Effort: low | medium | high | xhigh | max.
    const effort = process.env.ANTHROPIC_EFFORT ?? "high";
    const maxTokens = Number(process.env.ANTHROPIC_MAX_TOKENS ?? "32000");
    // Fold the engine's workflow/skill instructions into the system prompt — the
    // connector doesn't surface them, so without this the model lacks the framing
    // checklist and workflow that coding agents get on connect.
    const serverInstructions = await getServerInstructions();
    const system = serverInstructions
      ? `${SYSTEM_PROMPT}\n\n${serverInstructions}`
      : SYSTEM_PROMPT;
    // Trim older middle turns if the transcript is approaching the context window
    // (client-side floor), and let the engine's native context management clear old tool
    // results / compact server-side (the coding-agent-grade path).
    const history = applyCompaction(messages, system, maxTokens);
    const cm = contextManagement();
    const params: any = {
      model: MODEL,
      max_tokens: maxTokens,
      system: cachedSystem(system),
      thinking: { type: "adaptive" },
      output_config: { effort },
      mcp_servers: [
        {
          type: "url",
          url: FRONTIER_MCP_URL,
          name: "frontier",
          ...(FRONTIER_MCP_TOKEN ? { authorization_token: FRONTIER_MCP_TOKEN } : {}),
        },
      ],
      tools: cachedTools([
        { type: "mcp_toolset", mcp_server_name: "frontier" },
      ]),
      messages: withCacheBreakpoint(history) as Anthropic.MessageParam[],
      ...(cm ? { context_management: cm.context_management } : {}),
      betas: ["mcp-client-2025-11-20", ...(cm ? [cm.beta] : [])],
    };
    const upstream = await client.beta.messages.stream(params);

    return new ReadableStream<Uint8Array>({
      async start(controller) {
        try {
          for await (const event of upstream) {
            // Surface cache effectiveness in server logs (read ≈ 0.1x input price; write ≈ 1.25x).
            const u = (event as any)?.message?.usage ?? (event as any)?.usage;
            if (u && (u.cache_read_input_tokens != null || u.cache_creation_input_tokens != null)) {
              console.log(
                `[frontier] prompt cache — read:${u.cache_read_input_tokens ?? 0} ` +
                  `write:${u.cache_creation_input_tokens ?? 0} uncached:${u.input_tokens ?? 0}`,
              );
            }
            const chunk = `data: ${JSON.stringify(event)}\n\n`;
            controller.enqueue(encoder.encode(chunk));
          }
          controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
          controller.close();
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: "error", message: msg })}\n\n`)
          );
          controller.close();
        }
      },
    });
  },
};

// ─── openai-compatible adapter ──────────────────────────────────────────────
// Runs the agent loop client-side against any provider that speaks the OpenAI
// Chat Completions API:
//   1. open an MCP client to the Frontier server
//   2. translate MCP tools to OpenAI function schemas
//   3. POST /chat/completions with streaming
//   4. execute returned tool calls against MCP, feed results back, loop
// Emits Anthropic-shaped SSE events (content_block_*) so the UI is provider-
// agnostic — switch endpoints by changing one env var.
//
// Configure via:
//   OPENAI_BASE_URL          endpoint URL (default: https://api.openai.com/v1)
//   OPENAI_API_KEY           bearer token
//   OPENAI_MODEL             model id (default: gpt-4o-mini)
//   OPENAI_REASONING_EFFORT  optional, passed through only if set
//   OPENAI_WIRE              "chat" (default) or "responses" — OpenAI's reasoning models
//                            (GPT-5.x) only support function tools WITH reasoning on
//                            /v1/responses; chat/completions rejects the combination.

const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1";
const OPENAI_MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";
const OPENAI_REASONING_EFFORT = process.env.OPENAI_REASONING_EFFORT;
const OPENAI_WIRE = process.env.OPENAI_WIRE ?? "chat";

type OpenAIMessage =
  | { role: "system"; content: string }
  | { role: "user"; content: string }
  | { role: "assistant"; content: string | null; tool_calls?: OpenAIToolCall[] }
  | { role: "tool"; content: string; tool_call_id: string };

type OpenAIToolCall = {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
};

type OpenAITool = {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
  };
};

function translateMessagesToOpenAI(messages: ChatMessage[]): OpenAIMessage[] {
  // The UI sends Anthropic-shaped content (string or array of blocks). Flatten
  // each turn into OpenAI's flat schema: assistant tool_calls + separate tool
  // messages keyed by tool_call_id.
  const out: OpenAIMessage[] = [];
  for (const m of messages) {
    if (typeof m.content === "string") {
      out.push({ role: m.role, content: m.content });
      continue;
    }
    const blocks = m.content as Array<Record<string, unknown>>;
    if (m.role === "user") {
      // User turns may contain mcp_tool_result blocks (round-tripped from a prior
      // assistant turn). Surface those as `tool` messages; concat any text.
      const textParts: string[] = [];
      for (const b of blocks) {
        if (b.type === "text" && typeof b.text === "string") textParts.push(b.text);
      }
      if (textParts.length) out.push({ role: "user", content: textParts.join("\n") });
      for (const b of blocks) {
        if (b.type === "mcp_tool_result") {
          const content = Array.isArray(b.content)
            ? (b.content as Array<{ text?: string }>).map((c) => c.text ?? "").join("\n")
            : typeof b.content === "string"
              ? (b.content as string)
              : "";
          out.push({
            role: "tool",
            tool_call_id: String(b.tool_use_id ?? ""),
            content,
          });
        }
      }
      continue;
    }
    // assistant
    let text = "";
    const toolCalls: OpenAIToolCall[] = [];
    for (const b of blocks) {
      if (b.type === "text" && typeof b.text === "string") text += b.text;
      if (b.type === "mcp_tool_use") {
        toolCalls.push({
          id: String(b.id ?? ""),
          type: "function",
          function: {
            name: String(b.name ?? ""),
            arguments: JSON.stringify(b.input ?? {}),
          },
        });
      }
    }
    out.push({
      role: "assistant",
      content: text || null,
      ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
    });
    // Also surface any tool results embedded in the assistant turn (current UI
    // appends results to the same assistant message).
    for (const b of blocks) {
      if (b.type === "mcp_tool_result") {
        const content = Array.isArray(b.content)
          ? (b.content as Array<{ text?: string }>).map((c) => c.text ?? "").join("\n")
          : typeof b.content === "string"
            ? (b.content as string)
            : "";
        out.push({
          role: "tool",
          tool_call_id: String(b.tool_use_id ?? ""),
          content,
        });
      }
    }
  }
  return out;
}

async function callMcpTool(
  mcp: MCPClient,
  name: string,
  args: Record<string, unknown>,
): Promise<{ text: string; isError: boolean }> {
  try {
    const result = await mcp.callTool({ name, arguments: args });
    const parts = Array.isArray(result.content) ? result.content : [];
    const text = parts
      .map((c: any) => (typeof c?.text === "string" ? c.text : ""))
      .filter(Boolean)
      .join("\n");
    return { text, isError: Boolean(result.isError) };
  } catch (err) {
    return {
      text: err instanceof Error ? err.message : String(err),
      isError: true,
    };
  }
}

const openAICompatibleAdapter: AgentRuntime = {
  async stream(messages) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error("OPENAI_API_KEY not set");

    const encoder = new TextEncoder();

    return new ReadableStream<Uint8Array>({
      async start(controller) {
        const emit = (event: unknown) => {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
        };
        const emitError = (msg: string) => emit({ type: "error", message: msg });

        let mcp: MCPClient | null = null;
        try {
          // 1. Open MCP client to the Frontier server.
          mcp = await openMcpClient("frontier-web");

          // 2. List + translate tools.
          const { tools: mcpTools } = await mcp.listTools();
          const openAITools: OpenAITool[] = mcpTools.map((t: any) => ({
            type: "function",
            function: {
              name: t.name,
              description: t.description,
              parameters: t.inputSchema ?? { type: "object", properties: {} },
            },
          }));

          // 3. Seed conversation. The MCP server's `instructions` field is the
          // canonical agent guidance (workflow, framing checklist, style rules).
          // A real MCP host like Claude Code surfaces these to the model on
          // connect — we mirror that by prepending them to the system prompt.
          const serverInstructions = mcp.getInstructions() ?? "";
          const systemContent = serverInstructions
            ? `${SYSTEM_PROMPT}\n\n${serverInstructions}`
            : SYSTEM_PROMPT;
          // Trim older middle turns if the transcript is approaching the context window.
          // Provider context windows vary — set AGENT_CONTEXT_WINDOW to match the target.
          const history = applyCompaction(
            messages,
            systemContent,
            Number(process.env.OPENAI_MAX_TOKENS ?? "16000"),
          );

          // Responses wire: same MCP loop, /v1/responses request/event shapes.
          if (OPENAI_WIRE === "responses") {
            await runResponsesLoop({ apiKey, emit, mcp, mcpTools, systemContent, history });
            controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
            return;
          }

          const convo: OpenAIMessage[] = [
            { role: "system", content: systemContent },
            ...translateMessagesToOpenAI(history),
          ];

          // 4. Tool-call loop. Block index is monotonically increasing across
          // all emitted content blocks (text + tool_use + tool_result) so the UI
          // reducer can address them.
          let blockIndex = 0;
          const MAX_ITERATIONS = 12;

          for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
            const body: Record<string, unknown> = {
              model: OPENAI_MODEL,
              messages: convo,
              tools: openAITools.length ? openAITools : undefined,
              stream: true,
            };
            // Only pass reasoning_effort when set — many providers reject the
            // field; supported providers treat absence as "use default".
            if (OPENAI_REASONING_EFFORT) body.reasoning_effort = OPENAI_REASONING_EFFORT;

            const res = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
              method: "POST",
              headers: {
                "content-type": "application/json",
                authorization: `Bearer ${apiKey}`,
              },
              body: JSON.stringify(body),
            });

            if (!res.ok || !res.body) {
              const errText = await res.text().catch(() => "");
              // Reasoning models that gate tools behind the Responses API name it in the
              // error — point at the switch instead of leaving a bare 400.
              const hint = errText.includes("/v1/responses")
                ? " — this model requires the Responses API for tools+reasoning: set OPENAI_WIRE=responses (or OPENAI_REASONING_EFFORT=none to stay on chat/completions without reasoning)."
                : "";
              throw new Error(
                `OpenAI-compatible provider (${OPENAI_BASE_URL}) returned ${res.status}` +
                  (errText ? `: ${errText}` : "") +
                  hint,
              );
            }

            // Stream + accumulate this turn's output.
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buf = "";

            let textBlockIdx: number | null = null;
            let textAccum = "";
            // Some reasoning models (Kimi K3 on Together, observed 2026-08-01) stream the
            // user-facing answer in `reasoning_content` and leave `content` empty. Buffer it
            // so a turn that produces only reasoning can still be shown rather than rendering
            // blank; flushed at end-of-stream only when no content arrived.
            let reasoningAccum = "";
            // tool calls accumulate by chunk index → {id, name, args}
            const toolAccum = new Map<
              number,
              { id: string; name: string; args: string; blockIdx: number; emittedStart: boolean }
            >();
            let finishReason: string | null = null;

            outer: while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              buf += decoder.decode(value, { stream: true });
              const lines = buf.split("\n");
              buf = lines.pop() ?? "";
              for (const raw of lines) {
                const line = raw.trim();
                if (!line.startsWith("data:")) continue;
                const payload = line.slice(5).trim();
                if (payload === "[DONE]") break outer;
                let chunk: any;
                try {
                  chunk = JSON.parse(payload);
                } catch {
                  continue;
                }
                const choice = chunk.choices?.[0];
                if (!choice) continue;
                const delta = choice.delta ?? {};
                if (typeof delta.content === "string" && delta.content.length) {
                  if (textBlockIdx === null) {
                    textBlockIdx = blockIndex++;
                    emit({
                      type: "content_block_start",
                      index: textBlockIdx,
                      content_block: { type: "text", text: "" },
                    });
                  }
                  textAccum += delta.content;
                  emit({
                    type: "content_block_delta",
                    index: textBlockIdx,
                    delta: { type: "text_delta", text: delta.content },
                  });
                }
                if (typeof delta.reasoning_content === "string" && delta.reasoning_content.length) {
                  reasoningAccum += delta.reasoning_content;
                }
                if (Array.isArray(delta.tool_calls)) {
                  for (const tc of delta.tool_calls) {
                    const idx: number = tc.index ?? 0;
                    let entry = toolAccum.get(idx);
                    if (!entry) {
                      entry = {
                        id: tc.id ?? `call_${idx}_${Date.now()}`,
                        name: tc.function?.name ?? "",
                        args: "",
                        blockIdx: -1,
                        emittedStart: false,
                      };
                      toolAccum.set(idx, entry);
                    }
                    if (tc.id) entry.id = tc.id;
                    if (tc.function?.name) entry.name = tc.function.name;
                    // Emit content_block_start as soon as we know the name+id.
                    if (!entry.emittedStart && entry.name && entry.id) {
                      entry.blockIdx = blockIndex++;
                      entry.emittedStart = true;
                      emit({
                        type: "content_block_start",
                        index: entry.blockIdx,
                        content_block: {
                          type: "mcp_tool_use",
                          id: entry.id,
                          name: entry.name,
                          server_name: "frontier",
                          input: {},
                        },
                      });
                    }
                    const argPiece = tc.function?.arguments ?? "";
                    if (argPiece && entry.emittedStart) {
                      entry.args += argPiece;
                      emit({
                        type: "content_block_delta",
                        index: entry.blockIdx,
                        delta: { type: "input_json_delta", partial_json: argPiece },
                      });
                    } else if (argPiece) {
                      entry.args += argPiece;
                    }
                  }
                }
                if (choice.finish_reason) finishReason = choice.finish_reason;
              }
            }

            // No content but reasoning arrived: surface the reasoning as the turn's text.
            // Without this the user sees an empty assistant turn even though the model
            // answered — the answer just came down the other channel.
            if (!textAccum && reasoningAccum) {
              textBlockIdx = blockIndex++;
              emit({
                type: "content_block_start",
                index: textBlockIdx,
                content_block: { type: "text", text: "" },
              });
              textAccum = reasoningAccum;
              emit({
                type: "content_block_delta",
                index: textBlockIdx,
                delta: { type: "text_delta", text: reasoningAccum },
              });
            }

            // Flush any tool_use blocks whose start was deferred (rare — only if
            // name/id never arrived before args; emit start now so reducer can
            // parse args).
            for (const entry of toolAccum.values()) {
              if (!entry.emittedStart) {
                entry.blockIdx = blockIndex++;
                entry.emittedStart = true;
                emit({
                  type: "content_block_start",
                  index: entry.blockIdx,
                  content_block: {
                    type: "mcp_tool_use",
                    id: entry.id,
                    name: entry.name,
                    server_name: "frontier",
                    input: safeParseJson(entry.args),
                  },
                });
              }
              emit({ type: "content_block_stop", index: entry.blockIdx });
            }
            if (textBlockIdx !== null) {
              emit({ type: "content_block_stop", index: textBlockIdx });
            }

            // Update conversation with this assistant turn.
            const assistantToolCalls: OpenAIToolCall[] = [...toolAccum.values()].map((e) => ({
              id: e.id,
              type: "function",
              function: { name: e.name, arguments: e.args || "{}" },
            }));
            convo.push({
              role: "assistant",
              content: textAccum || null,
              ...(assistantToolCalls.length ? { tool_calls: assistantToolCalls } : {}),
            });

            // No tool calls → we're done.
            if (assistantToolCalls.length === 0 || finishReason === "stop") break;

            // 5. Execute tools, emit mcp_tool_result blocks, feed back into convo.
            for (const entry of toolAccum.values()) {
              const args = safeParseJson(entry.args);
              const result = await callMcpTool(mcp, entry.name, args);
              const resultBlockIdx = blockIndex++;
              emit({
                type: "content_block_start",
                index: resultBlockIdx,
                content_block: {
                  type: "mcp_tool_result",
                  tool_use_id: entry.id,
                  content: [{ type: "text", text: result.text }],
                  is_error: result.isError,
                },
              });
              emit({ type: "content_block_stop", index: resultBlockIdx });
              convo.push({
                role: "tool",
                tool_call_id: entry.id,
                content: result.text,
              });
            }
          }

          controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
        } catch (err) {
          emitError(err instanceof Error ? err.message : String(err));
        } finally {
          if (mcp) {
            try {
              await mcp.close();
            } catch {
              /* ignore */
            }
          }
          controller.close();
        }
      },
    });
  },
};

// ─── OpenAI Responses wire loop ─────────────────────────────────────────────
// Same client-side MCP tool loop as the chat path, over /v1/responses shapes:
// input items (function_call / function_call_output) instead of tool messages,
// and item-keyed streaming events instead of choice deltas. Wire validated
// against gpt-5.6-terra (tools run WITH reasoning here; chat/completions
// rejects that combination). Pure translation helpers live in openai-responses.ts.
async function runResponsesLoop(opts: {
  apiKey: string;
  emit: (event: unknown) => void;
  mcp: MCPClient;
  mcpTools: Array<{ name: string; description?: string; inputSchema?: Record<string, unknown> }>;
  systemContent: string;
  history: ChatMessage[];
}) {
  const { apiKey, emit, mcp } = opts;
  const tools = buildResponsesTools(opts.mcpTools);
  const input: ResponsesInputItem[] = translateMessagesToResponsesInput(opts.history);
  let blockIndex = 0;
  const MAX_ITERATIONS = 12;

  for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
    const body: Record<string, unknown> = {
      model: OPENAI_MODEL,
      instructions: opts.systemContent,
      input,
      tools: tools.length ? tools : undefined,
      stream: true,
    };
    if (OPENAI_REASONING_EFFORT) body.reasoning = { effort: OPENAI_REASONING_EFFORT };

    const res = await fetch(`${OPENAI_BASE_URL}/responses`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) {
      const errText = await res.text().catch(() => "");
      throw new Error(
        `OpenAI Responses provider (${OPENAI_BASE_URL}) returned ${res.status}` +
          (errText ? `: ${errText}` : ""),
      );
    }

    // Stream: events are item-keyed. Track each output item → UI block.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    type Item =
      | { kind: "text"; blockIdx: number; started: boolean }
      | { kind: "fc"; blockIdx: number; callId: string; name: string; args: string };
    const items = new Map<string, Item>();
    const calls: Array<{ callId: string; name: string; args: string }> = [];

    outer: while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const raw of lines) {
        const line = raw.trim();
        if (!line.startsWith("data:")) continue;
        let ev: any;
        try {
          ev = JSON.parse(line.slice(5).trim());
        } catch {
          continue;
        }
        switch (ev.type) {
          case "response.output_item.added": {
            const item = ev.item ?? {};
            if (item.type === "function_call") {
              const blockIdx = blockIndex++;
              items.set(String(item.id), {
                kind: "fc",
                blockIdx,
                callId: String(item.call_id ?? item.id),
                name: String(item.name ?? ""),
                args: "",
              });
              emit({
                type: "content_block_start",
                index: blockIdx,
                content_block: {
                  type: "mcp_tool_use",
                  id: String(item.call_id ?? item.id),
                  name: String(item.name ?? ""),
                  server_name: "frontier",
                  input: {},
                },
              });
            } else if (item.type === "message") {
              // Text block opens lazily on the first delta (message items can be empty).
              items.set(String(item.id), { kind: "text", blockIdx: -1, started: false });
            }
            break; // reasoning items: not surfaced
          }
          case "response.output_text.delta": {
            const entry = items.get(String(ev.item_id));
            if (!entry || entry.kind !== "text") break;
            if (!entry.started) {
              entry.blockIdx = blockIndex++;
              entry.started = true;
              emit({
                type: "content_block_start",
                index: entry.blockIdx,
                content_block: { type: "text", text: "" },
              });
            }
            emit({
              type: "content_block_delta",
              index: entry.blockIdx,
              delta: { type: "text_delta", text: String(ev.delta ?? "") },
            });
            break;
          }
          case "response.function_call_arguments.delta": {
            const entry = items.get(String(ev.item_id));
            if (!entry || entry.kind !== "fc") break;
            entry.args += String(ev.delta ?? "");
            emit({
              type: "content_block_delta",
              index: entry.blockIdx,
              delta: { type: "input_json_delta", partial_json: String(ev.delta ?? "") },
            });
            break;
          }
          case "response.output_item.done": {
            const item = ev.item ?? {};
            const entry = items.get(String(item.id));
            if (!entry) break;
            if (entry.kind === "fc") {
              // The done item carries the authoritative full arguments string.
              if (typeof item.arguments === "string" && item.arguments) entry.args = item.arguments;
              emit({ type: "content_block_stop", index: entry.blockIdx });
              calls.push({ callId: entry.callId, name: entry.name, args: entry.args });
            } else if (entry.started) {
              emit({ type: "content_block_stop", index: entry.blockIdx });
            }
            break;
          }
          case "response.failed":
          case "error": {
            const msg =
              ev.message ?? ev.response?.error?.message ?? "Responses stream reported failure";
            throw new Error(`OpenAI Responses stream error: ${msg}`);
          }
          case "response.completed":
            break outer;
        }
      }
    }

    // No tool calls → final answer delivered; done.
    if (calls.length === 0) break;

    // Execute tools against MCP, surface results, and extend the input for the next turn
    // (function_call must precede its function_call_output).
    for (const call of calls) {
      const result = await callMcpTool(mcp, call.name, safeParseJson(call.args));
      const resultBlockIdx = blockIndex++;
      emit({
        type: "content_block_start",
        index: resultBlockIdx,
        content_block: {
          type: "mcp_tool_result",
          tool_use_id: call.callId,
          content: [{ type: "text", text: result.text }],
          is_error: result.isError,
        },
      });
      emit({ type: "content_block_stop", index: resultBlockIdx });
      input.push({
        type: "function_call",
        call_id: call.callId,
        name: call.name,
        arguments: call.args || "{}",
      });
      input.push({ type: "function_call_output", call_id: call.callId, output: result.text });
    }
  }
}

function safeParseJson(s: string): Record<string, unknown> {
  if (!s) return {};
  try {
    const v = JSON.parse(s);
    return v && typeof v === "object" ? (v as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

// ─── anthropic-local adapter ────────────────────────────────────────────────
// Same engine as messages-api but runs the agent loop CLIENT-SIDE against a
// local MCP server. Required because Anthropic's MCP connector only accepts
// https:// URLs, so we can't point the server-side loop at localhost. Use this
// backend for evals against a local MCP, and to bypass the connector's 300s
// per-tool-call timeout. Also enables extended thinking (private scratchpad).

type AnthropicMessage =
  | { role: "user"; content: string | Array<Record<string, unknown>> }
  | { role: "assistant"; content: Array<Record<string, unknown>> };

// Translate one stored UI block to its Anthropic API shape (or null to drop).
function translateBlock(b: Record<string, unknown>): Record<string, unknown> | null {
  if (b.type === "text") return { type: "text", text: b.text };
  if (b.type === "mcp_tool_use")
    return { type: "tool_use", id: b.id, name: b.name, input: b.input };
  if (b.type === "mcp_tool_result")
    return { type: "tool_result", tool_use_id: b.tool_use_id, content: b.content, is_error: b.is_error };
  return null;
}

// Anthropic requires tool_result blocks in a user message that FOLLOWS the
// assistant's tool_use. The UI flattens an entire multi-iteration turn (text +
// tool_use + tool_result, interleaved) into one assistant message, so replaying
// it verbatim puts tool_result in an assistant turn → API 400 on the next turn.
// Walk the flattened blocks and emit the alternation the API expects:
// assistant (text + tool_use) → user (tool_result) → assistant …
function splitAssistantTurn(blocks: Array<Record<string, unknown>>): AnthropicMessage[] {
  const out: AnthropicMessage[] = [];
  let assistant: Array<Record<string, unknown>> = [];
  let results: Array<Record<string, unknown>> = [];
  const flushResults = () => {
    if (!results.length) return;
    if (assistant.length) {
      out.push({ role: "assistant", content: assistant });
      assistant = [];
    }
    out.push({ role: "user", content: results });
    results = [];
  };
  for (const b of blocks) {
    const t = translateBlock(b);
    if (!t) continue;
    if (t.type === "tool_result") {
      results.push(t);
    } else {
      flushResults(); // assistant content after a result batch starts a new turn
      assistant.push(t);
    }
  }
  flushResults();
  if (assistant.length) out.push({ role: "assistant", content: assistant });
  return out;
}

const anthropicLocalAdapter: AgentRuntime = {
  async stream(messages) {
    const apiKey = process.env.ANTHROPIC_API_KEY || process.env.CLAUDE_API_KEY;
    if (!apiKey) throw new Error("ANTHROPIC_API_KEY (or CLAUDE_API_KEY) not set");
    const client = new Anthropic({ apiKey });
    const encoder = new TextEncoder();
    // Opus 4.7 uses adaptive thinking + output_config.effort (not the older
    // budget_tokens form, which 400s). Effort: low | medium | high | xhigh | max.
    const effort = process.env.ANTHROPIC_EFFORT ?? "high";
    const maxTokens = Number(process.env.ANTHROPIC_MAX_TOKENS ?? "32000");

    return new ReadableStream<Uint8Array>({
      async start(controller) {
        const emit = (event: unknown) => {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
        };

        let mcp: MCPClient | null = null;
        try {
          // 1. Open local MCP client.
          mcp = await openMcpClient("frontier-web-anthropic-local");

          // 2. List + translate tools to Anthropic format.
          const { tools: mcpTools } = await mcp.listTools();
          const anthropicTools = mcpTools.map((t: any) => ({
            name: t.name,
            description: t.description,
            input_schema: t.inputSchema ?? { type: "object", properties: {} },
          }));

          // 2b. Pull server instructions and fold them into the system prompt
          // — this is the canonical agent guidance (workflow, framing checklist,
          // style rules). A real MCP host like Claude Code surfaces these on
          // connect; we mirror that here so the model runs like Frontier.
          const serverInstructions = mcp.getInstructions() ?? "";
          const systemPrompt = serverInstructions
            ? `${SYSTEM_PROMPT}\n\n${serverInstructions}`
            : SYSTEM_PROMPT;

          // Trim older middle turns if the transcript is approaching the context window.
          const history = applyCompaction(messages, systemPrompt, maxTokens);
          const cm = contextManagement();

          // 3. Seed conversation. Translate UI's Anthropic-shaped history into
          //    the API's shape (mcp_tool_use → tool_use, mcp_tool_result → tool_result).
          const convo: AnthropicMessage[] = [];
          for (const m of history) {
            if (typeof m.content === "string") {
              if (m.role === "user") {
                convo.push({ role: "user", content: m.content });
              } else {
                convo.push({ role: "assistant", content: [{ type: "text", text: m.content }] });
              }
              continue;
            }
            const blocks = m.content as Array<Record<string, unknown>>;
            if (m.role === "assistant") {
              // Split the flattened assistant turn into assistant/user alternation
              // (tool_result must live in a following user message).
              convo.push(...splitAssistantTurn(blocks));
            } else {
              const translated = blocks
                .map(translateBlock)
                .filter(Boolean) as Array<Record<string, unknown>>;
              convo.push({ role: "user", content: translated });
            }
          }

          // 4. Tool loop. Each iteration is a streaming call. We rewrite
          //    Anthropic's native tool_use blocks into our UI's mcp_tool_use
          //    wire format on the fly, and append assistant + tool_result turns
          //    to the convo for the next iteration.
          const MAX_ITERATIONS = 16;
          let uiBlockIndex = 0;
          // Track per-server-index mapping (Anthropic emits its own indices) →
          // our re-numbered indices, since we collapse thinking blocks.
          for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
            const stream = await client.beta.messages.stream({
              model: MODEL,
              max_tokens: maxTokens,
              system: cachedSystem(systemPrompt),
              thinking: { type: "adaptive" },
              output_config: { effort },
              tools: cachedTools(anthropicTools) as any,
              messages: withCacheBreakpoint(convo) as any,
              ...(cm ? { context_management: cm.context_management } : {}),
              betas: ["interleaved-thinking-2025-05-14", ...(cm ? [cm.beta] : [])],
            } as any);

            type Pending =
              | { kind: "text"; uiIdx: number; text: string }
              | { kind: "tool_use"; uiIdx: number; id: string; name: string; input: string }
              | { kind: "thinking" }; // not surfaced to UI
            const pending = new Map<number, Pending>();
            const turnContent: Array<Record<string, unknown>> = [];
            const turnToolCalls: Array<{ id: string; name: string; input: any }> = [];
            let stopReason: string | null = null;

            for await (const event of stream) {
              const ev = event as any;
              if (ev.type === "content_block_start") {
                const cb = ev.content_block;
                if (cb.type === "text") {
                  const uiIdx = uiBlockIndex++;
                  pending.set(ev.index, { kind: "text", uiIdx, text: "" });
                  emit({
                    type: "content_block_start",
                    index: uiIdx,
                    content_block: { type: "text", text: "" },
                  });
                } else if (cb.type === "tool_use") {
                  const uiIdx = uiBlockIndex++;
                  pending.set(ev.index, {
                    kind: "tool_use",
                    uiIdx,
                    id: cb.id,
                    name: cb.name,
                    input: "",
                  });
                  emit({
                    type: "content_block_start",
                    index: uiIdx,
                    content_block: {
                      type: "mcp_tool_use",
                      id: cb.id,
                      name: cb.name,
                      server_name: "frontier",
                      input: {},
                    },
                  });
                } else {
                  // thinking blocks (and any future block types) — swallow,
                  // they're not surfaced to the UI.
                  pending.set(ev.index, { kind: "thinking" });
                }
              } else if (ev.type === "content_block_delta") {
                const p = pending.get(ev.index);
                if (!p) continue;
                if (p.kind === "text" && ev.delta?.type === "text_delta") {
                  p.text += ev.delta.text;
                  emit({
                    type: "content_block_delta",
                    index: p.uiIdx,
                    delta: { type: "text_delta", text: ev.delta.text },
                  });
                } else if (p.kind === "tool_use" && ev.delta?.type === "input_json_delta") {
                  p.input += ev.delta.partial_json ?? "";
                  emit({
                    type: "content_block_delta",
                    index: p.uiIdx,
                    delta: { type: "input_json_delta", partial_json: ev.delta.partial_json },
                  });
                }
              } else if (ev.type === "content_block_stop") {
                const p = pending.get(ev.index);
                if (!p) continue;
                if (p.kind === "text") {
                  turnContent.push({ type: "text", text: p.text });
                  emit({ type: "content_block_stop", index: p.uiIdx });
                } else if (p.kind === "tool_use") {
                  let parsedInput: any = {};
                  try {
                    parsedInput = p.input ? JSON.parse(p.input) : {};
                  } catch {
                    parsedInput = {};
                  }
                  turnContent.push({
                    type: "tool_use",
                    id: p.id,
                    name: p.name,
                    input: parsedInput,
                  });
                  turnToolCalls.push({ id: p.id, name: p.name, input: parsedInput });
                  emit({ type: "content_block_stop", index: p.uiIdx });
                }
              } else if (ev.type === "message_delta") {
                if (ev.delta?.stop_reason) stopReason = ev.delta.stop_reason;
              }
            }

            // Append the assistant turn to convo.
            if (turnContent.length) {
              convo.push({ role: "assistant", content: turnContent });
            }

            // If no tool calls, we're done.
            if (turnToolCalls.length === 0 || stopReason !== "tool_use") {
              break;
            }

            // 5. Execute tools, emit mcp_tool_result blocks, append to convo.
            const userToolResults: Array<Record<string, unknown>> = [];
            for (const tc of turnToolCalls) {
              const result = await callMcpTool(mcp, tc.name, tc.input ?? {});
              const uiIdx = uiBlockIndex++;
              emit({
                type: "content_block_start",
                index: uiIdx,
                content_block: {
                  type: "mcp_tool_result",
                  tool_use_id: tc.id,
                  content: [{ type: "text", text: result.text }],
                  is_error: result.isError,
                },
              });
              emit({ type: "content_block_stop", index: uiIdx });
              userToolResults.push({
                type: "tool_result",
                tool_use_id: tc.id,
                content: result.text,
                is_error: result.isError,
              });
            }
            convo.push({ role: "user", content: userToolResults });
          }

          controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          emit({ type: "error", message: msg });
        } finally {
          if (mcp) {
            try {
              await mcp.close();
            } catch {
              /* ignore */
            }
          }
          controller.close();
        }
      },
    });
  },
};

// ─── selector ───────────────────────────────────────────────────────────────

const adapters: Record<string, AgentRuntime> = {
  "messages-api": messagesApiAdapter,
  "anthropic-local": anthropicLocalAdapter,
  "openai-compatible": openAICompatibleAdapter,
};

const backend = process.env.AGENT_BACKEND ?? "messages-api";
const selected = adapters[backend];
if (!selected) {
  // Fail at startup, not on the first request — and never silently fall back:
  // a typo'd backend quietly running messages-api would spend the API key.
  throw new Error(
    `Unknown AGENT_BACKEND "${backend}". Valid: ${Object.keys(adapters).join(", ")}.`
  );
}
export const agentRuntime: AgentRuntime = selected;
