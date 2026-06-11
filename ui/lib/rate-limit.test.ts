/**
 * Unit tests for the API rate limiter. Pure logic with an injectable clock, so it runs
 * with Node's built-in test runner + TypeScript type-stripping — no extra dev deps:
 *
 *   node --test --experimental-strip-types lib/rate-limit.test.ts
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { RateLimiter, clientKey } from "./rate-limit.ts";

// A controllable clock so windows advance deterministically.
function fakeClock(start = 1_000_000) {
  let t = start;
  return { now: () => t, advance: (ms: number) => (t += ms) };
}

test("per-key limit allows up to the cap, then blocks", () => {
  const clock = fakeClock();
  const rl = new RateLimiter({
    perKey: { limit: 3, windowMs: 1000 },
    global: { limit: 100, windowMs: 1000 },
    now: clock.now,
  });

  for (let i = 0; i < 3; i++) assert.equal(rl.check("a").ok, true, `request ${i} should pass`);

  const blocked = rl.check("a");
  assert.equal(blocked.ok, false);
  assert.equal(blocked.scope, "key");
  assert.ok((blocked.retryAfterMs ?? 0) > 0 && (blocked.retryAfterMs ?? 0) <= 1000);
});

test("per-key window resets after windowMs", () => {
  const clock = fakeClock();
  const rl = new RateLimiter({
    perKey: { limit: 2, windowMs: 1000 },
    global: { limit: 100, windowMs: 1000 },
    now: clock.now,
  });

  assert.equal(rl.check("a").ok, true);
  assert.equal(rl.check("a").ok, true);
  assert.equal(rl.check("a").ok, false);

  clock.advance(1000);
  assert.equal(rl.check("a").ok, true, "window should reset after windowMs");
});

test("per-key limits are independent across keys", () => {
  const clock = fakeClock();
  const rl = new RateLimiter({
    perKey: { limit: 1, windowMs: 1000 },
    global: { limit: 100, windowMs: 1000 },
    now: clock.now,
  });

  assert.equal(rl.check("a").ok, true);
  assert.equal(rl.check("a").ok, false);
  assert.equal(rl.check("b").ok, true, "a different key has its own budget");
});

test("global cap blocks across keys and is the cost circuit-breaker", () => {
  const clock = fakeClock();
  const rl = new RateLimiter({
    perKey: { limit: 100, windowMs: 1000 },
    global: { limit: 2, windowMs: 1000 },
    now: clock.now,
  });

  assert.equal(rl.check("a").ok, true);
  assert.equal(rl.check("b").ok, true);

  const blocked = rl.check("c");
  assert.equal(blocked.ok, false);
  assert.equal(blocked.scope, "global", "a fresh key is still blocked by the global cap");
});

test("a rejected request consumes no slot in either window (check before commit)", () => {
  const clock = fakeClock();
  const rl = new RateLimiter({
    perKey: { limit: 1, windowMs: 1000 },
    global: { limit: 3, windowMs: 1000 },
    now: clock.now,
  });

  assert.equal(rl.check("a").ok, true); // a: 1/1, global: 1/3
  assert.equal(rl.check("a").ok, false); // blocked on key — must NOT also burn a global slot

  // If the rejected call had consumed a global slot, only one more key would fit
  // before the global cap (limit 3). Two more distinct keys must still pass.
  assert.equal(rl.check("b").ok, true); // global: 2/3
  assert.equal(rl.check("c").ok, true); // global: 3/3
  assert.equal(rl.check("d").scope, "global"); // now exhausted
});

test("clientKey uses the trusted (rightmost) x-forwarded-for hop, else a shared bucket", () => {
  // Rightmost wins: a client-spoofed leftmost ("1.2.3.4") is ignored in favor of
  // the trusted-proxy-appended "5.6.7.8".
  assert.equal(
    clientKey(new Request("http://x", { headers: { "x-forwarded-for": "1.2.3.4, 5.6.7.8" } })),
    "5.6.7.8",
  );
  // x-real-ip is NOT used as the bucket key (a client could spoof it) — fall back to a
  // shared "unknown" bucket, which is still bounded by the global cap.
  assert.equal(
    clientKey(new Request("http://x", { headers: { "x-real-ip": "9.9.9.9" } })),
    "unknown",
  );
  assert.equal(clientKey(new Request("http://x")), "unknown");
});

test("global cap bounds Map growth: distinct keys past the global limit don't allocate", () => {
  const clock = fakeClock();
  const rl = new RateLimiter({
    perKey: { limit: 100, windowMs: 1000 },
    global: { limit: 5, windowMs: 1000 },
    now: clock.now,
  });
  for (let i = 0; i < 1000; i++) rl.check(`ip-${i}`);
  // Only the 5 requests that passed the global cap allocated a key window; the other
  // 995 were rejected at the global check before touching the Map.
  assert.equal(rl.trackedKeys, 5);
});

test("Map is hard-capped: maxKeys bounds tracked keys even with a huge global limit", () => {
  const clock = fakeClock();
  const rl = new RateLimiter({
    perKey: { limit: 1000, windowMs: 1000 },
    global: { limit: 1_000_000, windowMs: 1000 },
    maxKeys: 10,
    now: clock.now,
  });
  for (let i = 0; i < 500; i++) rl.check(`ip-${i}`);
  assert.ok(rl.trackedKeys <= 10, `trackedKeys=${rl.trackedKeys} must stay <= maxKeys (10)`);
});
