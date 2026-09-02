import * as crypto from "node:crypto";
import { Hono, type Context, type MiddlewareHandler } from "hono";

/**
 * How long after the last content request the server still reports itself
 * active. kubimo samples `/api/status/connections` on a poll interval (10s by
 * default) and only refreshes the runner's `lastActive` when it sees
 * `active > 0`, so the window must comfortably exceed any sane poll interval or
 * page loads between two polls would go unseen. It only ever adds to the
 * runner's `deleteAfterSecsInactive` (hours), so its exact value is not
 * sensitive and it is deliberately not configurable.
 */
export const ACTIVITY_WINDOW_MS = 5 * 60 * 1000;

/** Remembers when content was last served, for kubimo's idle collector. */
export class ActivityTracker {
  private lastContentAt: number | undefined;
  private readonly clock: () => number;
  private readonly windowMs: number;

  constructor(clock: () => number = Date.now, windowMs = ACTIVITY_WINDOW_MS) {
    this.clock = clock;
    this.windowMs = windowMs;
  }

  touch(): void {
    this.lastContentAt = this.clock();
  }

  /** 0 or 1: the shape kubimo reads as `Connections { active: usize }`. */
  activeConnections(): number {
    return this.lastContentAt !== undefined &&
      this.clock() - this.lastContentAt <= this.windowMs
      ? 1
      : 0;
  }
}

/**
 * Per-route middleware for content routes. Only a served response counts: a
 * rejected token or a missing notebook must not keep a renderer alive.
 */
export function trackActivity(activity: ActivityTracker): MiddlewareHandler {
  return async (c, next) => {
    await next();
    if (c.res.status < 400) {
      activity.touch();
    }
  };
}

export function tokensEqual(
  expected: string,
  actual: string | undefined,
): boolean {
  if (actual === undefined) {
    return false;
  }
  const digest = (value: string) =>
    crypto.createHash("sha256").update(value).digest();
  return crypto.timingSafeEqual(digest(expected), digest(actual));
}

/** `Authorization: Bearer` first (what kubimo sends), else `?access_token=` like marimo. */
export function requestToken(c: Context): string | undefined {
  const match = /^Bearer\s+(\S+)$/i.exec(c.req.header("Authorization") ?? "");
  return match?.[1] ?? c.req.query("access_token");
}

export function requireToken(
  accessToken: string | undefined,
): MiddlewareHandler {
  return async (c, next) => {
    if (
      accessToken !== undefined &&
      !tokensEqual(accessToken, requestToken(c))
    ) {
      c.header("WWW-Authenticate", 'Bearer realm=""');
      return c.text("Unauthorized", 401);
    }
    await next();
  };
}

export interface StatusApiOptions {
  version: string;
  accessToken?: string | undefined;
  activity: ActivityTracker;
}

/**
 * The endpoints kubimo's runner status poll expects from any runner, mirroring
 * marimo's own `/api/status/connections` and `/api/version`. Mount with
 * `app.route("/", statusApi(...))` so they sit under the app's base path.
 */
export function statusApi({
  version,
  accessToken,
  activity,
}: StatusApiOptions): Hono {
  const auth = requireToken(accessToken);
  const api = new Hono();
  api.get("/api/status/connections", auth, (c) =>
    c.json({ active: activity.activeConnections() }),
  );
  api.get("/api/version", auth, (c) => c.text(version));
  return api;
}
