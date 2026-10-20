/* Copyright 2026 Marimo. All rights reserved. */

import {
  LanguageServerClient,
  RPCError,
} from "@marimo-team/codemirror-languageserver";
import type * as LSP from "vscode-languageserver-protocol";
import { once } from "@/utils/once";
import { createTransport } from "../../../lsp/transports";
import { assertHasLanguageServerRpc } from "../../../lsp/types";
import { getLspRootUri, getLspWorkspaceFolders } from "../../../lsp/utils";
import { collectRpcRefs, type InteractiveGoals } from "./interactive-goals";

// Lean answers goal requests only once it has elaborated up to the position,
// which can take a while (e.g. when importing Mathlib).
const REQUEST_TIMEOUT_MS = 120_000;

// Lean's error for an RPC session that expired or whose worker restarted
const RPC_NEEDS_RECONNECT = -32900;

export interface PlainTermGoal {
  goal: string;
  range: LSP.Range;
}

/**
 * A client for the Lean language server, with Lean's custom requests.
 */
export class LeanLanguageServerClient extends LanguageServerClient {
  public rpcConnect(uri: string): Promise<{ sessionId: string }> {
    return this.leanRequest("$/lean/rpc/connect", { uri });
  }

  public rpcCall<T>(
    sessionId: string,
    method: string,
    params: LSP.TextDocumentPositionParams,
    signal?: AbortSignal,
  ): Promise<T> {
    return this.leanRequest(
      "$/lean/rpc/call",
      { ...params, sessionId, method, params },
      signal,
    );
  }

  public plainTermGoal(
    params: LSP.TextDocumentPositionParams,
    signal?: AbortSignal,
  ): Promise<PlainTermGoal | null> {
    return this.leanRequest("$/lean/plainTermGoal", params, signal);
  }

  public hover(
    params: LSP.TextDocumentPositionParams,
    signal?: AbortSignal,
  ): Promise<LSP.Hover | null> {
    return this.leanRequest("textDocument/hover", params, signal);
  }

  public sendNotification(method: string, params: unknown): Promise<void> {
    assertHasLanguageServerRpc(this);
    return this.notify(method, params);
  }

  private async leanRequest<T>(
    method: string,
    params: unknown,
    signal?: AbortSignal,
  ): Promise<T> {
    assertHasLanguageServerRpc(this);
    return (await this.request(
      method,
      params,
      REQUEST_TIMEOUT_MS,
      signal,
    )) as T;
  }
}

type LeanClient = Pick<
  LeanLanguageServerClient,
  | "initializePromise"
  | "initialize"
  | "sendNotification"
  | "onNotification"
  | "rpcConnect"
  | "rpcCall"
  | "plainTermGoal"
  | "hover"
>;

type DiagnosticsListener = (
  version: number,
  diagnostics: LSP.Diagnostic[],
) => void;

/**
 * The one document open on the Lean server: the Lean code of the focused
 * cell and its ancestors. Switching cells replaces the text, and Lean reuses
 * the elaboration of any unchanged prefix.
 */
export class LeanDocumentSession {
  public static readonly URI = "untitled:marimo-lean";

  /** Version of the latest text sent to the server */
  public version = 0;
  public readonly client: LeanClient;
  private text: string | null = null;
  private listeners = new Set<DiagnosticsListener>();
  private rpcSession: Promise<string> | null = null;

  public constructor(client: LeanClient) {
    this.client = client;
    client.onNotification((notification) => {
      if (notification.method !== "textDocument/publishDiagnostics") {
        return;
      }
      const { uri, version, diagnostics } = notification.params;
      if (uri !== LeanDocumentSession.URI || version === undefined) {
        return;
      }
      for (const listener of this.listeners) {
        listener(version, diagnostics);
      }
    });
  }

  /**
   * Make `text` the document's content; returns the version it gets.
   */
  public async sync(text: string): Promise<number> {
    if (text === this.text) {
      return this.version;
    }
    const opened = this.text !== null;
    this.text = text;
    this.version += 1;
    const version = this.version;
    await this.client.initializePromise;
    await (opened
      ? this.client.sendNotification("textDocument/didChange", {
          textDocument: { uri: LeanDocumentSession.URI, version },
          contentChanges: [{ text }],
        })
      : this.sendDidOpen(text, version));
    return version;
  }

  /**
   * Re-open the document after the connection to the server was lost.
   */
  public async reopen(): Promise<void> {
    this.rpcSession = null;
    await this.client.initialize();
    if (this.text !== null) {
      await this.sendDidOpen(this.text, this.version);
    }
  }

  /**
   * The goals at `position`, marked with how the tactic there changes them.
   */
  public async interactiveGoals(
    position: LSP.Position,
    signal?: AbortSignal,
  ): Promise<InteractiveGoals | null> {
    try {
      return await this.callGoals(position, signal);
    } catch (error) {
      if (!(error instanceof RPCError && error.code === RPC_NEEDS_RECONNECT)) {
        throw error;
      }
      this.rpcSession = null;
      return this.callGoals(position, signal);
    }
  }

  public onDiagnostics(listener: DiagnosticsListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private async callGoals(
    position: LSP.Position,
    signal?: AbortSignal,
  ): Promise<InteractiveGoals | null> {
    this.rpcSession ??= this.client
      .rpcConnect(LeanDocumentSession.URI)
      .then(({ sessionId }) => sessionId)
      .catch((error: unknown) => {
        this.rpcSession = null;
        throw error;
      });
    const sessionId = await this.rpcSession;
    const textDocument = { uri: LeanDocumentSession.URI };
    const goals = await this.client.rpcCall<InteractiveGoals | null>(
      sessionId,
      "Lean.Widget.getInteractiveGoals",
      { textDocument, position },
      signal,
    );
    // We only render the goals, so the server can drop its references at once
    const refs = collectRpcRefs(goals);
    if (refs.length > 0) {
      void this.client.sendNotification("$/lean/rpc/release", {
        uri: LeanDocumentSession.URI,
        sessionId,
        refs,
      });
    }
    return goals;
  }

  private sendDidOpen(text: string, version: number) {
    return this.client.sendNotification("textDocument/didOpen", {
      textDocument: {
        uri: LeanDocumentSession.URI,
        languageId: "lean4",
        version,
        text,
      },
    });
  }
}

/**
 * The session for the Lean server, connected on first use.
 */
export const getLeanSession = once((): LeanDocumentSession => {
  // oxlint-disable-next-line prefer-const -- reassigned after closure capture
  let session: LeanDocumentSession | undefined;
  const transport = createTransport("lean", async () => {
    await session?.reopen();
  });
  session = new LeanDocumentSession(
    new LeanLanguageServerClient({
      transport,
      rootUri: getLspRootUri(),
      workspaceFolders: getLspWorkspaceFolders(),
    }),
  );
  return session;
});
