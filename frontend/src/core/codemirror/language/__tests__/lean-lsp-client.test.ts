/* Copyright 2026 Marimo. All rights reserved. */

import { RPCError } from "@marimo-team/codemirror-languageserver";
import { describe, expect, it, vi } from "vitest";
import type * as LSP from "vscode-languageserver-protocol";
import {
  LeanDocumentSession,
  type LeanLanguageServerClient,
} from "../languages/lean/lsp-client";

type Notification = Parameters<
  Parameters<LeanLanguageServerClient["onNotification"]>[0]
>[0];

function createSession() {
  let notify: (notification: Notification) => void = () => undefined;
  const client = {
    initializePromise: Promise.resolve(),
    initialize: vi.fn(() => Promise.resolve()),
    sendNotification: vi.fn((_method: string, _params: unknown) =>
      Promise.resolve(),
    ),
    onNotification: vi.fn((listener: (n: Notification) => void) => {
      notify = listener;
      return () => undefined;
    }),
    rpcConnect: vi.fn((_uri: string) =>
      Promise.resolve({
        sessionId: `session${client.rpcConnect.mock.calls.length}`,
      }),
    ),
    rpcCall: vi.fn(
      (
        _sessionId: string,
        _method: string,
        _params: unknown,
        _signal?: AbortSignal,
      ): Promise<unknown> =>
        Promise.resolve({
          goals: [{ hyps: [], type: { text: "True" }, ctx: { p: "7" } }],
        }),
    ),
    plainTermGoal: vi.fn(),
    hover: vi.fn(),
  };
  const session = new LeanDocumentSession(
    client as unknown as LeanLanguageServerClient,
  );
  return { session, client, notify: (n: Notification) => notify(n) };
}

const diagnostic: LSP.Diagnostic = {
  range: { start: { line: 0, character: 0 }, end: { line: 0, character: 1 } },
  message: "unknown identifier",
  severity: 1,
};

describe("LeanDocumentSession", () => {
  it("should open the document, then send changes with new versions", async () => {
    const { session, client } = createSession();

    expect(await session.sync("def x := 1")).toBe(1);
    expect(await session.sync("def x := 1")).toBe(1);
    expect(await session.sync("def x := 2")).toBe(2);

    expect(client.sendNotification.mock.calls).toEqual([
      [
        "textDocument/didOpen",
        {
          textDocument: {
            uri: LeanDocumentSession.URI,
            languageId: "lean4",
            version: 1,
            text: "def x := 1",
          },
        },
      ],
      [
        "textDocument/didChange",
        {
          textDocument: { uri: LeanDocumentSession.URI, version: 2 },
          contentChanges: [{ text: "def x := 2" }],
        },
      ],
    ]);
  });

  it("should re-open the current document after reconnecting", async () => {
    const { session, client } = createSession();
    await session.reopen();
    expect(client.sendNotification).not.toHaveBeenCalled();

    await session.sync("def x := 1");
    await session.sync("def x := 2");
    client.sendNotification.mockClear();
    await session.reopen();

    expect(client.initialize).toHaveBeenCalledTimes(2);
    expect(client.sendNotification).toHaveBeenCalledWith(
      "textDocument/didOpen",
      {
        textDocument: {
          uri: LeanDocumentSession.URI,
          languageId: "lean4",
          version: 2,
          text: "def x := 2",
        },
      },
    );
  });

  it("should pass on diagnostics for its document only", () => {
    const { session, notify } = createSession();
    const listener = vi.fn();
    const unsubscribe = session.onDiagnostics(listener);

    notify({
      jsonrpc: "2.0",
      method: "textDocument/publishDiagnostics",
      params: {
        uri: LeanDocumentSession.URI,
        version: 3,
        diagnostics: [diagnostic],
      },
    });
    notify({
      jsonrpc: "2.0",
      method: "textDocument/publishDiagnostics",
      params: { uri: "file:///other.lean", version: 3, diagnostics: [] },
    });
    expect(listener.mock.calls).toEqual([[3, [diagnostic]]]);

    unsubscribe();
    notify({
      jsonrpc: "2.0",
      method: "textDocument/publishDiagnostics",
      params: { uri: LeanDocumentSession.URI, version: 4, diagnostics: [] },
    });
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("should get interactive goals over one RPC session and release its refs", async () => {
    const { session, client } = createSession();
    const position = { line: 1, character: 2 };

    expect(await session.interactiveGoals(position)).toEqual({
      goals: [{ hyps: [], type: { text: "True" }, ctx: { p: "7" } }],
    });
    await session.interactiveGoals(position);

    expect(client.rpcConnect).toHaveBeenCalledTimes(1);
    expect(client.rpcCall).toHaveBeenCalledWith(
      "session1",
      "Lean.Widget.getInteractiveGoals",
      { textDocument: { uri: LeanDocumentSession.URI }, position },
      undefined,
    );
    expect(client.sendNotification).toHaveBeenCalledWith("$/lean/rpc/release", {
      uri: LeanDocumentSession.URI,
      sessionId: "session1",
      refs: [{ p: "7" }],
    });
  });

  it("should reconnect when the RPC session expired", async () => {
    const { session, client } = createSession();
    const position = { line: 0, character: 0 };
    await session.interactiveGoals(position);
    client.rpcCall.mockRejectedValueOnce(
      new RPCError(-32900, "Outdated RPC session"),
    );

    await session.interactiveGoals(position);

    expect(client.rpcConnect).toHaveBeenCalledTimes(2);
    expect(client.rpcCall).toHaveBeenLastCalledWith(
      "session2",
      "Lean.Widget.getInteractiveGoals",
      { textDocument: { uri: LeanDocumentSession.URI }, position },
      undefined,
    );
  });

  it("should not reconnect on other errors", async () => {
    const { session, client } = createSession();
    client.rpcCall.mockRejectedValueOnce(
      new RPCError(-32801, "Content modified"),
    );

    await expect(
      session.interactiveGoals({ line: 0, character: 0 }),
    ).rejects.toThrow("Content modified");
    expect(client.rpcConnect).toHaveBeenCalledTimes(1);
  });
});
