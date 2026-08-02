## Overview
<p align="center">
  <img src="assets/frontier-logo.png" alt="Frontier" width="160" />
</p>

<p align="center">
  <strong>Multi-objective decision optimization agent toolkit. An architecture pattern for complex decisions LLMs can’t solve alone. Use in any MCP client or via the integrated web UI.</strong>
</p>

Frontier helps you make hard decisions that have many options and conflicting goals: which projects to fund, how to split a budget, who to source from. You describe the decision to an AI agent in plain language; it models the problem, optimizes it, and walks you through the **full set of optimal tradeoffs**. You make the final call.

Under the hood it maps the **Pareto frontier** (every outcome where improving one goal means giving up another) with evolutionary search plus optional exact solvers. Every number it reports is computed, not guessed, so the decision stays explainable and auditable, and framing the problem and picking the tradeoff stay yours.

**Try it:** the [hosted demo](https://frontier-ui.onrender.com/) (ask @cafzal for access), or [set up your own](#setup).

## Examples

[Worked examples](examples/) you can load and adapt: each ships a paste-ready prompt journey that reproduces the result. Two shown:

<table>
<tr>
<td width="50%" valign="top">
<img src="assets/example-portfolio.png" alt="ETF efficient frontier with a plain-language tradeoff read" /><br/>
<sub>Investment portfolio: risk / return / yield frontier with a plain-language tradeoff read</sub>
</td>
<td width="50%" valign="top">
<img src="assets/example-capital.png" alt="Capital project formulation and the exact-certified frontier" /><br/>
<sub>Capital project selection: model formulation and the exact-certified frontier</sub>
</td>
</tr>
</table>

## Workflow

<p align="center"><img src="assets/workflow-progression-icon.svg" alt="Frontier decision workflow: frame, score, explore, curate, certify, examine, decide" width="100%" /></p>

Drive Frontier in plain language, from a coding agent or the hosted web chat. You describe the decision in business terms; the agent translates it into an optimization model, solves it, and translates the results back into a decision. A typical sequence:

1. **Frame.** Name the objectives (what to maximize or minimize), the options to choose among, the hard constraints, and any scenarios; the agent translates this from business language into the model. *e.g. "Which projects should we fund for the most value, at the least cost and risk?"*
2. **Score.** Hand over the numbers, or let the agent estimate and flag what's shaky. *e.g. "Score the projects from these briefs."*
3. **Explore.** The agent runs the approximate solver and maps the frontier broadly for *coverage* (how much of the tradeoff space you hold): its shape, the extremes, the balanced/knee, the marginal cost of pushing an objective, the per-scenario frontiers. *e.g. "Show the tradeoffs and suggest a balanced one."*
4. **Curate.** Narrow the frontier to the handful you'd defend; curated picks track survival across re-runs. *e.g. "Keep the balanced and the safest options."*
5. **Certify.** Where the shape supports it, the agent runs an exact solver to prove the curated finalists, each optimal for its own tradeoff target, to a measured *optimality gap* (how far from the proven optimum), catching plans that only looked efficient. *e.g. "Confirm these hold up."*
6. **Examine.** The agent reads the results back in the language of the decision: why this mix over the alternatives, the one lever that moves the outcome most, which picks survive your scenarios, and a guarantee that holds across every feasible plan. *e.g. "What's holding us back, and what would more budget buy?"*
7. **Decide.** Commit to the pick that fits your tradeoffs: the engine lays out the options and leaves the final call to you. *e.g. "Go with the balanced one."*

Two loops run throughout. **Iterate** to refine the model and re-solve as you go: tighten a constraint, add a scenario, compare against the last run. **Update** to revisit the decision as data or conditions change, with curated picks and feedback carrying across runs, so the decision stays live.

## Purpose

Spreadsheets hit a complexity wall once options and constraints in a decision multiply. Generative AI models reason about tradeoffs but can't *solve* them: reliably enumerating a huge option space, enforcing hard constraints, and producing the frontier are beyond them. Frontier fills the gap: the LLM translates and narrates, an optimizer does the math, and the judgment stays with you.

**When Frontier helps:** any decision that picks a subset from many options or splits an allocation across them, under conflicting objectives and hard constraints, with data to score the options – the territory just past what a spreadsheet answers. Pairwise interactions between options, where one's value depends on another, make the problem genuinely nonlinear: beyond what a ranking or weighted sum can capture.

**Target problem types.** The [examples table](examples/) maps all fourteen:

| Shape | Structure | Examples |
|---|---|---|
| **Select a subset** | • linear totals<br>• pairwise interactions | • [Capital projects (300)](examples/capital_project_selection_300/)<br>• [Claims triage](examples/claims_investigation_triage/)<br>• [Charging network siting](examples/charging_network_siting/) |
| **Split an allocation** | • linear<br>• mean-variance risk<br>• pairwise interactions | • [Scarce supply rationing](examples/scarce_supply_rationing/)<br>• [Investment portfolio](examples/investment_portfolio/)<br>• [Marketing channel budget](examples/channel_budget/) |

**What it adds beyond an LLM alone** (its design principles):
- **The full frontier**: every Pareto-optimal plan, yours to weigh.
- **Explore broadly, certify selectively**: the heuristic maps the whole space; an exact solver then proves the finalists you'd commit to on supported shapes, catching dominated points the heuristic showed as efficient. Each finalist you shortlisted gets its own verdict — optimal, or dominated with the certified plan that beats it. It can only confirm or improve them — and on two-objective continuous problems the certificate also bounds how much better anything *between* the certified points could possibly be.
- **Constraints enforced, then checked**: nine hard types (cardinality, force include, force exclude, objective bounds, exclusion pairs, dependencies, group limits with per-group floors and caps, a global allocation cap, per-option allocation floors/caps), respected by every plan the search returns — and every solve re-reads its own returned plans against them, so that guarantee is reported rather than assumed. Each rule can carry the reason it exists, echoed wherever its cost is priced.
- **Governance guarantees**: on selection problems, a proof that a guardrail holds for *every* feasible plan, or a concrete counterexample.
- **Grounded and reproducible**: every number traces to a score, an objective value, a dual, or a binding constraint, and the same inputs + seed reproduce the exact frontier.
- **Scenarios & risk**: independent frontiers per scenario, plus CVaR / worst-case / expected / minimax-regret per objective.
- **Knowledge discovery**: mine the frontier for selection rates, recurring rules, and strategy families.
- **Durable decisions**: problems persist across sessions; curated picks and feedback attach to the decision and survive re-runs.

## Architecture

<p align="center"><img src="assets/architecture.png" alt="Frontier architecture: agent clients and the web UI speak MCP to one server exposing four tools, with skills and solvers attached and a deterministic engine beneath" width="560" /></p>

Frontier is a Python MCP server (built on the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)) wrapping pymoo's NSGA-II/III evolutionary solvers, with two exact-solver backends (HiGHS on CPU, cuOpt on GPU). State persists per-problem as JSON; the optimizer produces a Pareto frontier with indicators to guide the decision. Domain expertise lives in skill markdown files that the server auto-injects at workflow transitions.

For full schemas, action parameters, data model, persistence layout, and the skill auto-injection mechanism, see [`architecture.md`](architecture.md). For skill, prompt, and MCP design principles, see [`best-practices.md`](best-practices.md).

### Tools

Full action lists and parameters in [`architecture.md`](architecture.md):

- **`model`**: define and edit the problem (objectives, options, scores, 9 constraint types, interaction matrices, scenarios): `create` / `update` / `get`, plus `save` / `load` for named problems.
- **`solve`**: validate and optimize (NSGA-II/III, fast/thorough, seeded): `run`, `run_scenarios`, and `status` to poll background runs; opt-in `solver="highs"|"cuopt"` exact backends on supported shapes.
- **`explore`**: navigate results, or probe the feasible region before a solve: `tradeoffs`, `certify` (audit the exact overlay), `audit` (selection problems: prove a guardrail holds across the *whole* feasible space, or return a counterexample), `sensitivity` (solver-exact shadow prices + near-misses), `composition` (mine selection rates, principles, strategy families), plus `compare` / `solutions` / `scenario_results` / `curate`.
- **`get_skill`**: fetch the workflow guidance below.

### Skills

Markdown guides the server auto-injects at workflow transitions (also fetchable via `get_skill`) – the domain judgment for each phase:

- **`problem_framing`**: objectives vs constraints, approach + aggregation, scenario definition.
- **`data_collection`**: score elicitation without anchoring bias, quality signals.
- **`optimization_strategy`**: iteration, constraint strategy, infeasibility, re-run judgment.
- **`solution_interpreter`**: presenting tradeoffs without a "best", eliciting preferences, curation.

### Saving & loading

Every problem is auto-persisted in the engine's store (`data/`, keyed by id) – session state the engine manages for you. Separately, `model save` writes a **named, portable copy** in the [examples](examples/) format, to reload or share by name:

- **`model save problem_id=… save_as="<name>"`**: save to your gitignored `saved/` library (override with `FRONTIER_SAVED_DIR`), bundling the solved frontier when present.
- **`model load source="<name>"`**: rebuild a problem, resolving `saved/` first, then bundled `examples/`; omit `source` to list available names.

## Setup

Two ways to use Frontier:

- **Web UI**: a browser chat shell over the engine, with interactive charts and curation. Try the hosted app at **[frontier-ui.onrender.com](https://frontier-ui.onrender.com/)** (password-gated – ask @cafzal for access), or run/deploy your own (requires an API key; see [`ui/`](ui/) and [Deploy your own](#deploy-your-own)).
- **MCP client**: connect any MCP-compatible client (Claude Code, Claude Desktop, claude.ai, Cursor, Codex). The hosted beta engine (`https://frontier-592q.onrender.com/sse`) is gated by a token – ask @cafzal for the `FRONTIER_MCP_TOKEN` value; or [self-host](#self-host) your own (ungated by default).

The MCP-client snippets below assume the hosted engine.

### Claude Code (terminal)

```bash
claude mcp add frontier --transport sse \
  --url https://frontier-592q.onrender.com/sse \
  --header "Authorization: Bearer $FRONTIER_MCP_TOKEN"
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "frontier": {
      "transport": "sse",
      "url": "https://frontier-592q.onrender.com/sse",
      "headers": { "Authorization": "Bearer YOUR_FRONTIER_MCP_TOKEN" }
    }
  }
}
```

### claude.ai (MCP integrations)

Add Frontier as a remote MCP server in claude.ai settings using the SSE URL `https://frontier-592q.onrender.com/sse`, with an `Authorization: Bearer <FRONTIER_MCP_TOKEN>` header.

### Self-host

Run your own instance. Requires Python 3.11+.

```bash
git clone https://github.com/cafzal/frontier.git
cd frontier
pip install -e .

# stdio transport (for Claude Desktop / coding agents on the same machine)
python -m mcp_server.server

# SSE transport (for remote MCP clients)
MCP_TRANSPORT=sse python -m mcp_server.server

# Gate a public instance with a shared bearer token – clients must then send
# `Authorization: Bearer <token>`. Leave unset for an open local instance.
FRONTIER_MCP_TOKEN=your-secret MCP_TRANSPORT=sse python -m mcp_server.server
```

**Exact solvers (optional).** Install `highspy` (CPU; `pip install highspy`) or cuOpt (NVIDIA GPU; `pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12`) to unlock `solver="highs"|"cuopt"`: exact certification (`explore certify`) and solver-exact sensitivity (`explore sensitivity`) on supported shapes. See [`architecture.md`](architecture.md#solver-backends-pluggable).

Point your MCP client at the local server – for SSE that's `http://localhost:8000/sse`. The optional web UI lives in [`ui/`](ui/) – see its [README](ui/README.md).

### Deploy your own

Both pieces are plain web services – host them anywhere (Render, Fly, Railway, a VPS, Docker):

- **Engine** (Python) – `pip install ".[sse]"` (add `,highs` for the CPU exact backend), then run the SSE server as in [Self-host](#self-host), bound publicly: set `MCP_HOST=0.0.0.0` and `FRONTIER_MCP_TOKEN`, with the host supplying `$PORT`. Must be publicly reachable – Anthropic's MCP connector calls it.
- **Web UI** (Node, in `ui/`) – `npm install && npm run build`, then `npm start`. Set `FRONTIER_MCP_URL` (the engine's `/sse`), `FRONTIER_MCP_TOKEN`, your model-provider credentials (see **Model provider** below), and `UI_ACCESS_PASSWORD`. Long-session context management and prompt caching are env-tunable (`AGENT_CONTEXT_WINDOW`, `AGENT_CONTEXT_MANAGEMENT`, `AGENT_PROMPT_CACHE`, and related); [`architecture.md`](architecture.md#5-web-ui--hosting) documents the knobs and defaults.

`FRONTIER_MCP_TOKEN` must match on both – that's what authenticates the UI to the engine.

**Model provider.** The web UI is provider-pluggable through one env var, `AGENT_BACKEND` — pick a backend, set its credentials, done. No code changes to switch providers.

| Provider | `AGENT_BACKEND` | Credentials |
|---|---|---|
| **Anthropic** (default) | `messages-api` (public-https engine) or `anthropic-local` (localhost engine) | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` (default `claude-opus-4-8`) |
| **OpenAI** | `openai-compatible` | `OPENAI_API_KEY`, `OPENAI_MODEL`; reasoning models (GPT-5.x) also set `OPENAI_WIRE=responses` |
| **Any OpenAI-compatible endpoint** (NVIDIA NIM / [build.nvidia.com](https://build.nvidia.com), Groq, Together, a local server, …) | `openai-compatible` | `OPENAI_BASE_URL` (the provider's URL), `OPENAI_API_KEY`, `OPENAI_MODEL` |

Any OpenAI-compatible endpoint drops in by pointing `OPENAI_BASE_URL` at it — e.g. NVIDIA NIM at `https://integrate.api.nvidia.com/v1` with `OPENAI_MODEL=nvidia/nemotron-3-super-120b-a12b`. [`ui/.env.example`](ui/.env.example) has a copy-paste block per backend; keep real keys there (it's gitignored), never in the repo.

**Render (one-click example):** [`render.yaml`](render.yaml) provisions both as a blueprint.

## Background

Optional background – the thinking behind Frontier and how it's evolved:

- [Building an AI-Powered Decision Tool Prototype: A Product Manager's Journey](https://camafzal.substack.com/p/building-an-ai-powered-decision-tool) – May 2025
- [Lowering the Barriers to Decision Optimization with AI](https://camafzal.substack.com/p/lowering-the-barriers-to-decision) – Sep 2025
- [Making optimization accessible: AI as the translation layer](https://camafzal.substack.com/p/making-optimization-accessible-ai) – Jan 2026
- [Agents have a convergent reasoning gap](https://camafzal.substack.com/p/agents-have-a-convergent-reasoning) – Apr 2026
- [From decision exploration to certification and explanation](https://camafzal.substack.com/p/from-decision-exploration-to-certification) – Jun 2026

## Contributing

Contributions welcome – start with the developer docs:

- [`architecture.md`](architecture.md) – system architecture & data flow
- [`best-practices.md`](best-practices.md) – skill & prompt design guidelines

## Acknowledgements

Frontier builds on open-source optimization work, with thanks to:

- **[pymoo](https://github.com/anyoptimization/pymoo)** (Apache-2.0) – the NSGA-II / NSGA-III evolutionary solvers at Frontier's core. Blank, J. & Deb, K. (2020). *pymoo: Multi-Objective Optimization in Python.* IEEE Access, 8, 89497–89509. The underlying algorithms are Deb et al., NSGA-II (2002) and Deb & Jain, NSGA-III (2014).
- **[HiGHS](https://github.com/ERGO-Code/HiGHS)** (MIT) – CPU exact-solver backend (`solver="highs"`). Huangfu, Q. & Hall, J.A.J. (2018). *Parallelizing the dual revised simplex method.* Mathematical Programming Computation, 10(1), 119–142.
- **[NVIDIA cuOpt](https://github.com/NVIDIA/cuopt)** (Apache-2.0) – GPU exact-solver backend (`solver="cuopt"`).

The MCP surface is built on the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk); the web UI's charts render via [Plotly](https://plotly.com/javascript/) and [D3](https://d3js.org/).

## License

Apache License 2.0 – see [LICENSE](LICENSE) and [NOTICE](NOTICE).
