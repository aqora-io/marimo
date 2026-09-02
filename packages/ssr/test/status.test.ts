import { describe, expect, test } from "vitest";
import { Hono } from "hono";
import {
  ACTIVITY_WINDOW_MS,
  ActivityTracker,
  statusApi,
  tokensEqual,
  trackActivity,
} from "../src/cli/status";

function setup({ accessToken }: { accessToken?: string } = {}) {
  let now = 1_000_000;
  const activity = new ActivityTracker(() => now);
  const app = new Hono().basePath("/runner/abc");
  app.route("/", statusApi({ version: "1.2.3", accessToken, activity }));
  app.get("/health", (c) => c.text("ok"));
  app.on(["HEAD", "GET"], "/:filename", trackActivity(activity), (c) =>
    c.req.param("filename") === "missing.py"
      ? c.notFound()
      : c.body(null, 304),
  );
  return {
    app,
    advance: (ms: number) => {
      now += ms;
    },
  };
}

const bearer = { headers: { Authorization: "Bearer t" } };

async function active(app: Hono): Promise<{ active: number }> {
  const res = await app.request("/runner/abc/api/status/connections", bearer);
  expect(res.status).toBe(200);
  return (await res.json()) as { active: number };
}

describe("status api", () => {
  test("serves the version under the base path", async () => {
    const { app } = setup({ accessToken: "t" });
    const res = await app.request("/runner/abc/api/version", bearer);
    expect(res.status).toBe(200);
    expect(await res.text()).toBe("1.2.3");
  });

  test("requires the token when one is configured", async () => {
    const { app } = setup({ accessToken: "t" });
    const missing = await app.request("/runner/abc/api/version");
    expect(missing.status).toBe(401);
    expect(missing.headers.get("WWW-Authenticate")).toContain("Bearer");
    const wrong = await app.request("/runner/abc/api/version", {
      headers: { Authorization: "Bearer nope" },
    });
    expect(wrong.status).toBe(401);
    const query = await app.request("/runner/abc/api/version?access_token=t");
    expect(query.status).toBe(200);
  });

  test("is open when no token is configured", async () => {
    const { app } = setup();
    const res = await app.request("/runner/abc/api/status/connections");
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ active: 0 });
  });

  test("reports content requests within the window as active", async () => {
    const { app, advance } = setup({ accessToken: "t" });
    expect(await active(app)).toEqual({ active: 0 });
    await app.request("/runner/abc/nb.py");
    expect(await active(app)).toEqual({ active: 1 });
    advance(ACTIVITY_WINDOW_MS + 1);
    expect(await active(app)).toEqual({ active: 0 });
    await app.request("/runner/abc/nb.py", { method: "HEAD" });
    expect(await active(app)).toEqual({ active: 1 });
  });

  test("probes and polls are not activity", async () => {
    const { app } = setup({ accessToken: "t" });
    for (let i = 0; i < 50; i++) {
      expect((await app.request("/runner/abc/health")).status).toBe(200);
      await active(app);
    }
    expect(await active(app)).toEqual({ active: 0 });
  });

  test("a rejected content request is not activity", async () => {
    const { app } = setup({ accessToken: "t" });
    const res = await app.request("/runner/abc/missing.py");
    expect(res.status).toBe(404);
    expect(await active(app)).toEqual({ active: 0 });
  });
});

describe("tokensEqual", () => {
  test("rejects mismatches of any length and a missing token", () => {
    expect(tokensEqual("secret", "secret")).toBe(true);
    expect(tokensEqual("secret", "secre")).toBe(false);
    expect(tokensEqual("secret", "secreT")).toBe(false);
    expect(tokensEqual("secret", undefined)).toBe(false);
  });
});
