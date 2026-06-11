import { NextRequest, NextResponse } from "next/server";

/**
 * Shared-password gate for the hosted web UI (HTTP Basic Auth).
 *
 * Covers the page and every API route — /api/chat and /api/render alike (the
 * matcher excludes only Next internals) — the APIs must be gated too, or a request
 * could hit chat and spend the ANTHROPIC_API_KEY, or render and spend engine
 * compute, straight past the page gate. This is distinct
 * from the engine's FRONTIER_MCP_TOKEN: that token authenticates the app→engine
 * connector call and is never exposed to UI users, so the UI needs its own gate.
 *
 * Open when UI_ACCESS_PASSWORD is unset in development (local / single-user
 * self-host). In production the gate fails CLOSED: an unset password makes the
 * app refuse to serve (503) rather than run unauthenticated, since an open
 * public deploy lets anyone with the URL spend the operator's ANTHROPIC_API_KEY
 * and drive engine compute.
 * Username is optional: unset UI_ACCESS_USER = any username + correct password.
 */
export function middleware(req: NextRequest) {
  const expectedPass = process.env.UI_ACCESS_PASSWORD;
  if (!expectedPass) {
    if (process.env.NODE_ENV === "production") {
      return new NextResponse(
        "Server misconfigured: UI_ACCESS_PASSWORD is not set; refusing to serve " +
          "an unauthenticated production deployment.",
        { status: 503 },
      );
    }
    return NextResponse.next();
  }

  const expectedUser = process.env.UI_ACCESS_USER;
  const header = req.headers.get("authorization");

  if (header) {
    const [scheme, encoded] = header.split(" ");
    if (scheme === "Basic" && encoded) {
      let decoded = "";
      try {
        decoded = atob(encoded);
      } catch {
        decoded = ""; // malformed base64 → treat as failed auth (401), never a 500
      }
      const sep = decoded.indexOf(":");
      const user = decoded.slice(0, sep);
      const pass = decoded.slice(sep + 1);
      if (pass === expectedPass && (!expectedUser || user === expectedUser)) {
        return NextResponse.next();
      }
    }
  }

  return new NextResponse("Authentication required", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="Frontier", charset="UTF-8"' },
  });
}

export const config = {
  // Gate everything except Next.js internals and static assets.
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
