/**
 * Best-effort, in-memory request rate limiting for the API routes.
 *
 * A per-instance floor against runaway spend: even a holder of the shared
 * UI_ACCESS_PASSWORD (or an ungated local instance) can't drive unbounded
 * Anthropic / engine calls in a tight loop. Two fixed-window counters — one
 * per client key (IP) and one global — bound the request rate; the global
 * counter is the cost circuit-breaker that also covers a many-IP burst.
 *
 * In-memory means per-instance (not shared across Render instances) and reset
 * on redeploy — a deliberate floor, not a distributed quota. Per-user tokens
 * (D.1) are the real accounting layer; this just stops the obvious abuse. Pure
 * logic with an injectable clock, so it unit-tests under `node --test` without
 * the Next.js runtime.
 */

export interface RateLimitRule {
  /** Max requests allowed per window. */
  limit: number;
  /** Window length in milliseconds. */
  windowMs: number;
}

export interface RateLimiterOptions {
  perKey: RateLimitRule;
  global: RateLimitRule;
  /** Injectable clock (defaults to Date.now), for deterministic tests. */
  now?: () => number;
  /** Hard ceiling on tracked client keys (memory bound). Defaults to 4096. */
  maxKeys?: number;
}

export interface RateLimitResult {
  ok: boolean;
  /** Which limit was hit (only set when !ok). */
  scope?: "key" | "global";
  /** Milliseconds until the hit window resets (only set when !ok). */
  retryAfterMs?: number;
}

interface CounterWindow {
  count: number;
  resetAt: number;
}

/** Reset the window if the current one has elapsed. Idempotent for a fixed `t`. */
function roll(w: CounterWindow, rule: RateLimitRule, t: number): void {
  if (t >= w.resetAt) {
    w.count = 0;
    w.resetAt = t + rule.windowMs;
  }
}

export class RateLimiter {
  private readonly keyWindows = new Map<string, CounterWindow>();
  private readonly globalWindow: CounterWindow = { count: 0, resetAt: 0 };
  private readonly perKey: RateLimitRule;
  private readonly global: RateLimitRule;
  private readonly maxKeys: number;
  private readonly now: () => number;

  constructor(opts: RateLimiterOptions) {
    this.perKey = opts.perKey;
    this.global = opts.global;
    this.now = opts.now ?? (() => Date.now());
    this.maxKeys = opts.maxKeys ?? 4096;
  }

  /** Number of currently-tracked client keys (for observability/tests). */
  get trackedKeys(): number {
    return this.keyWindows.size;
  }

  /**
   * Record one request against `key`. Returns ok=false (without consuming a slot
   * from either window) when either limit is already exhausted.
   *
   * The global cap is checked BEFORE a per-key entry is allocated, so a flood of
   * distinct keys can't grow the Map past the global throughput (≈global.limit new
   * keys per window, which then expire) — this is the memory-exhaustion fix. A hard
   * `maxKeys` ceiling with eviction is the belt-and-suspenders bound for any config
   * where global.limit is set very high.
   */
  check(key: string): RateLimitResult {
    const t = this.now();
    roll(this.globalWindow, this.global, t);

    if (this.globalWindow.count >= this.global.limit) {
      // Reject before touching keyWindows — no allocation on the global-rejected path.
      return { ok: false, scope: "global", retryAfterMs: this.globalWindow.resetAt - t };
    }

    const keyWindow = this.acquireKeyWindow(key, t);
    roll(keyWindow, this.perKey, t);
    if (keyWindow.count >= this.perKey.limit) {
      return { ok: false, scope: "key", retryAfterMs: keyWindow.resetAt - t };
    }

    this.globalWindow.count++;
    keyWindow.count++;
    return { ok: true };
  }

  private acquireKeyWindow(key: string, t: number): CounterWindow {
    const existing = this.keyWindows.get(key);
    if (existing) return existing;
    if (this.keyWindows.size >= this.maxKeys) this.evict(t);
    const w: CounterWindow = { count: 0, resetAt: 0 };
    this.keyWindows.set(key, w);
    return w;
  }

  /** Hard-bound the Map: drop expired windows, then evict oldest-inserted until under cap. */
  private evict(t: number): void {
    for (const [k, w] of this.keyWindows) {
      if (t >= w.resetAt) this.keyWindows.delete(k);
    }
    while (this.keyWindows.size >= this.maxKeys) {
      const oldest = this.keyWindows.keys().next().value;
      if (oldest === undefined) break;
      this.keyWindows.delete(oldest);
    }
  }
}

/** Parse a positive-integer env var, falling back to `fallback`. */
function intEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

const WINDOW_MS = 60_000;

/**
 * Build a limiter from env. Defaults are generous for a human-driven beta but
 * cap a runaway loop. A per-route prefix gives each route an independent budget,
 * so a burst on one route can't exhaust another's global cap. Tunable via
 * <PREFIX>_RATE_LIMIT_PER_MIN / <PREFIX>_RATE_LIMIT_GLOBAL_PER_MIN.
 */
export function createRateLimiter(
  envPrefix: string,
  perKeyDefault: number,
  globalDefault: number,
): RateLimiter {
  return new RateLimiter({
    perKey: { limit: intEnv(`${envPrefix}_RATE_LIMIT_PER_MIN`, perKeyDefault), windowMs: WINDOW_MS },
    global: { limit: intEnv(`${envPrefix}_RATE_LIMIT_GLOBAL_PER_MIN`, globalDefault), windowMs: WINDOW_MS },
  });
}

// Independent per-instance counters. /api/chat drives paid Anthropic calls (the
// tighter budget); /api/render is read-only engine compute, no model spend, so a
// render burst gets its own looser budget and can't starve the chat cap.
export const chatRateLimiter = createRateLimiter("CHAT", 30, 90);
export const renderRateLimiter = createRateLimiter("RENDER", 60, 180);

// Number of trusted proxy hops in front of the app (Render appends one). The client
// IP is the hop appended by the *innermost trusted* proxy = the Nth-from-right XFF
// entry. Tune via TRUSTED_PROXY_HOPS if deployed behind a different topology.
const TRUSTED_PROXY_HOPS = (() => {
  const n = Number.parseInt(process.env.TRUSTED_PROXY_HOPS ?? "", 10);
  return Number.isFinite(n) && n > 0 ? n : 1;
})();

/**
 * Best-effort client identifier for the PER-KEY limit (the global cap is the real
 * bound). Picks the Nth-from-right x-forwarded-for hop — the address the trusted
 * proxy appended, which a client can't spoof from the left. When XFF is missing or
 * too short, returns "unknown" (a shared bucket, still bounded by the global cap)
 * rather than keying off a client-spoofable header.
 */
export function clientKey(req: Request): string {
  const xff = req.headers.get("x-forwarded-for");
  if (xff) {
    const hops = xff.split(",").map((s) => s.trim()).filter(Boolean);
    const pick = hops[hops.length - TRUSTED_PROXY_HOPS];
    if (pick) return pick;
  }
  return "unknown";
}

/** Build a 429 response with a Retry-After header from a failed check. */
export function tooManyRequests(result: RateLimitResult): Response {
  const retryAfterSec = Math.max(1, Math.ceil((result.retryAfterMs ?? WINDOW_MS) / 1000));
  return new Response(
    JSON.stringify({ error: "rate limit exceeded; slow down and retry shortly" }),
    {
      status: 429,
      headers: { "content-type": "application/json", "retry-after": String(retryAfterSec) },
    },
  );
}
