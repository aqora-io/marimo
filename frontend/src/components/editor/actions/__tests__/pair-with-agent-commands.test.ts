/* Copyright 2026 Marimo. All rights reserved. */

import { describe, expect, it } from "vitest";
import {
  type ConnectionInfo,
  getRunnerIdFromURL,
  getTerminalCommand,
} from "../pair-with-agent-commands";
import { shellQuote } from "@/utils/shell";

const RUNNER_ID = "AFdvcmtzcGFjZVJ1bm5lcgGgYjug5XSjsOG8qHR9wM0";

const CONNECTION: ConnectionInfo = {
  runnerId: RUNNER_ID,
  sessionId: "s_abc123",
};

describe("shellQuote", () => {
  it("quotes an empty string", () => {
    expect(shellQuote("")).toBe("''");
  });

  it("leaves shell-safe values untouched", () => {
    expect(shellQuote("http://localhost:8000")).toBe("http://localhost:8000");
    expect(shellQuote("notebooks/example.py")).toBe("notebooks/example.py");
  });

  it("quotes values with shell metacharacters", () => {
    expect(shellQuote("http://host:8000?a=1&b=2")).toBe(
      "'http://host:8000?a=1&b=2'",
    );
    expect(shellQuote("has space")).toBe("'has space'");
    expect(shellQuote("$(rm -rf /)")).toBe("'$(rm -rf /)'");
  });

  it("escapes embedded single quotes without breaking out", () => {
    // Closes the quote, emits a literal ' via "'", then reopens: '"'"'
    expect(shellQuote("a'b")).toBe(`'a'"'"'b'`);
  });

  it.each([
    ["/tmp/my notebook.py", "'/tmp/my notebook.py'"],
    [
      String.raw`C:\Users\Jane Doe\notebook.py`,
      String.raw`'C:\Users\Jane Doe\notebook.py'`,
    ],
    [
      String.raw`\\server\share\my notebook.py`,
      String.raw`'\\server\share\my notebook.py'`,
    ],
  ])("quotes non-portable path %s as one argument", (path, expected) => {
    expect(shellQuote(path)).toBe(expected);
  });
});

describe("getRunnerIdFromURL", () => {
  it("returns the last path segment of a runner url", () => {
    expect(getRunnerIdFromURL(`https://kubimo.org/runner/${RUNNER_ID}/`)).toBe(
      RUNNER_ID,
    );
    expect(
      getRunnerIdFromURL(`https://ovh.ancon-moth.ts.net/runner/${RUNNER_ID}/`),
    ).toBe(RUNNER_ID);
  });

  it("ignores the query string and a missing trailing slash", () => {
    expect(
      getRunnerIdFromURL(`https://kubimo.org/runner/${RUNNER_ID}/?file=x.py`),
    ).toBe(RUNNER_ID);
    expect(getRunnerIdFromURL(`https://kubimo.org/runner/${RUNNER_ID}`)).toBe(
      RUNNER_ID,
    );
  });

  it("is undefined when the url has no path", () => {
    expect(getRunnerIdFromURL("http://localhost:2718/")).toBeUndefined();
    expect(
      getRunnerIdFromURL("http://localhost:2718/?file=x.py"),
    ).toBeUndefined();
  });
});

describe("getTerminalCommand", () => {
  it("selects the agent for each tab", () => {
    expect(getTerminalCommand("claude", CONNECTION)).toBe(
      `aqora pair ${RUNNER_ID} --session s_abc123 --claude`,
    );
    expect(getTerminalCommand("codex", CONNECTION)).toBe(
      `aqora pair ${RUNNER_ID} --session s_abc123 --codex`,
    );
    expect(getTerminalCommand("opencode", CONNECTION)).toBe(
      `aqora pair ${RUNNER_ID} --session s_abc123 --opencode`,
    );
  });

  it("prints the prompt for any other agent", () => {
    expect(getTerminalCommand("prompt", CONNECTION)).toBe(
      `aqora pair ${RUNNER_ID} --session s_abc123 --prompt-only`,
    );
  });

  it("shell-escapes ids containing metacharacters", () => {
    expect(
      getTerminalCommand("claude", { runnerId: "a b", sessionId: "s;1" }),
    ).toBe(`aqora pair 'a b' --session 's;1' --claude`);
  });
});
