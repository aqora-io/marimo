/* Copyright 2026 Marimo. All rights reserved. */

import { type Diagnostic, setDiagnostics } from "@codemirror/lint";
import {
  type Extension,
  StateEffect,
  StateField,
  type Text,
} from "@codemirror/state";
import {
  type EditorView,
  hoverTooltip,
  type Tooltip,
  ViewPlugin,
  type ViewUpdate,
} from "@codemirror/view";
import { debounce } from "lodash-es";
import type * as LSP from "vscode-languageserver-protocol";
import { notebookAtom } from "@/core/cells/cells";
import type { CellId } from "@/core/cells/ids";
import { store } from "@/core/state/jotai";
import { sanitizeHtml } from "@/plugins/core/sanitize-html";
import { Logger } from "@/utils/Logger";
import type { NotebookLens } from "../../../lsp/lens";
import { createLspMarkdownRenderer } from "../../../lsp/markdown-renderer";
import { languageMetadataField } from "../../metadata";
import { buildLeanDocument } from "./document";
import { type GoalView, toGoalViews } from "./interactive-goals";
import { getLeanSession, LeanDocumentSession } from "./lsp-client";

/** The goals at the cursor of the focused Lean cell */
export type LeanGoals =
  | { kind: "goals"; goals: GoalView[] }
  | { kind: "term"; goal: string };

const setLeanGoals = StateEffect.define<LeanGoals | null>();

export const leanGoalsField = StateField.define<LeanGoals | null>({
  create: () => null,
  update(value, tr) {
    for (const effect of tr.effects) {
      if (effect.is(setLeanGoals)) {
        value = effect.value;
      }
    }
    return value;
  },
});

const SEVERITIES: Record<LSP.DiagnosticSeverity, Diagnostic["severity"]> = {
  1: "error",
  2: "warning",
  3: "info",
  4: "hint",
};

const renderMarkdown = createLspMarkdownRenderer();

// The last focused Lean editor, whose document is the one open on the
// server. It only changes when another Lean editor gets focus, so clicking
// into the panel keeps the goals.
let active: { view: EditorView; deactivate: () => void } | null = null;

class LeanLspPlugin {
  private readonly view: EditorView;
  private readonly cellId: CellId;
  private lens: NotebookLens | null = null;
  private goalsRequest: AbortController | null = null;
  private unsubscribes: (() => void)[] = [];
  private readonly refresh = debounce(() => void this.refreshNow(), 150);

  public constructor(view: EditorView, cellId: CellId) {
    this.view = view;
    this.cellId = cellId;
    if (view.hasFocus) {
      this.activate();
    }
  }

  public update(update: ViewUpdate) {
    if (update.focusChanged && this.view.hasFocus && !this.isActive()) {
      this.activate();
    } else if (this.isActive() && (update.docChanged || update.selectionSet)) {
      this.refresh();
    }
  }

  public destroy() {
    this.stop();
    if (this.isActive()) {
      active = null;
    }
  }

  public async hover(pos: number): Promise<Tooltip | null> {
    if (!this.isActive() || this.isFString()) {
      return null;
    }
    const session = getLeanSession();
    const version = await this.sync();
    const lens = this.lens;
    if (!lens) {
      return null;
    }
    const result = await session.client
      .hover({
        textDocument: { uri: LeanDocumentSession.URI },
        position: lens.transformPosition(
          toPosition(this.view.state.doc, pos),
          this.cellId,
        ),
      })
      .catch((error) => {
        Logger.debug("[lean] hover request failed", error);
        return null;
      });
    if (!result || version !== session.version) {
      return null;
    }

    const dom = document.createElement("div");
    dom.classList.add("documentation", "cm-lsp-hover-tooltip");
    dom.innerHTML = sanitizeHtml(
      renderMarkdown(hoverMarkdown(result.contents)),
    );
    const doc = this.view.state.doc;
    const range =
      result.range && lens.isInRange(result.range, this.cellId)
        ? lens.reverseRange(result.range, this.cellId)
        : null;
    return {
      pos: range ? toOffset(doc, range.start) : pos,
      end: range ? toOffset(doc, range.end) : undefined,
      above: true,
      create: () => ({ dom }),
    };
  }

  private isActive() {
    return active?.view === this.view;
  }

  /**
   * Whether the cell is an f-string, whose Lean code depends on Python values
   * the server can't see.
   */
  private isFString() {
    const { quotePrefix } = this.view.state.field(languageMetadataField);
    return typeof quotePrefix === "string" && quotePrefix.includes("f");
  }

  private activate() {
    active?.deactivate();
    active = { view: this.view, deactivate: () => this.deactivate() };
    this.unsubscribes = [
      getLeanSession().onDiagnostics((version, diagnostics) =>
        this.showDiagnostics(version, diagnostics),
      ),
      // Edits to ancestor cells change this cell's document
      store.sub(notebookAtom, () => this.refresh()),
    ];
    this.refresh();
  }

  private deactivate() {
    this.stop();
    this.clear();
  }

  private clear() {
    // Defer dispatch to avoid triggering during an editor update cycle
    queueMicrotask(() => {
      this.view.dispatch(setDiagnostics(this.view.state, []), {
        effects: setLeanGoals.of(null),
      });
    });
  }

  private stop() {
    this.refresh.cancel();
    this.goalsRequest?.abort();
    for (const unsubscribe of this.unsubscribes) {
      unsubscribe();
    }
    this.unsubscribes = [];
  }

  /**
   * Send this cell's document to the server; returns its version.
   */
  private sync(): Promise<number> {
    const { cellIds, cellData } = store.get(notebookAtom);
    const { parentEnv } = this.view.state.field(languageMetadataField);
    this.lens = buildLeanDocument(
      {
        id: this.cellId,
        code: this.view.state.doc.toString(),
        parentEnv: typeof parentEnv === "string" ? parentEnv : "",
      },
      cellIds.inOrderIds.map((id) => ({ id, code: cellData[id].code })),
    );
    return getLeanSession().sync(this.lens.mergedText);
  }

  private async refreshNow() {
    if (this.isFString()) {
      this.goalsRequest?.abort();
      this.clear();
      return;
    }
    const session = getLeanSession();
    const version = await this.sync();
    const lens = this.lens;
    if (!lens) {
      return;
    }

    this.goalsRequest?.abort();
    const controller = new AbortController();
    this.goalsRequest = controller;
    const params = {
      textDocument: { uri: LeanDocumentSession.URI },
      position: lens.transformPosition(
        toPosition(this.view.state.doc, this.view.state.selection.main.head),
        this.cellId,
      ),
    };
    try {
      const interactive = await session.interactiveGoals(
        params.position,
        controller.signal,
      );
      let goals: LeanGoals | null = interactive
        ? { kind: "goals", goals: toGoalViews(interactive) }
        : null;
      if (!goals) {
        const term = await session.client.plainTermGoal(
          params,
          controller.signal,
        );
        goals = term ? { kind: "term", goal: term.goal } : null;
      }
      // Lean answers requests made before an edit from the old version
      if (controller.signal.aborted || version !== session.version) {
        return;
      }
      queueMicrotask(() => {
        this.view.dispatch({ effects: setLeanGoals.of(goals) });
      });
    } catch (error) {
      if (!controller.signal.aborted) {
        Logger.debug("[lean] goal request failed", error);
      }
    }
  }

  private showDiagnostics(version: number, diagnostics: LSP.Diagnostic[]) {
    const lens = this.lens;
    if (!lens || version !== getLeanSession().version || this.isFString()) {
      return;
    }
    const doc = this.view.state.doc;
    const cellDiagnostics = diagnostics
      .filter(({ range }) => lens.isInRange(range, this.cellId))
      .map((diagnostic): Diagnostic => {
        const range = lens.reverseRange(diagnostic.range, this.cellId);
        return {
          from: toOffset(doc, range.start),
          to: toOffset(doc, range.end),
          severity: SEVERITIES[diagnostic.severity ?? 1],
          message:
            typeof diagnostic.message === "string"
              ? diagnostic.message
              : diagnostic.message.value,
          source: "lean",
        };
      });
    queueMicrotask(() => {
      this.view.dispatch(setDiagnostics(this.view.state, cellDiagnostics));
    });
  }
}

function toPosition(doc: Text, offset: number): LSP.Position {
  const line = doc.lineAt(offset);
  return { line: line.number - 1, character: offset - line.from };
}

function toOffset(doc: Text, { line, character }: LSP.Position): number {
  if (line >= doc.lines) {
    return doc.length;
  }
  const docLine = doc.line(line + 1);
  return Math.min(docLine.from + character, docLine.to);
}

function hoverMarkdown(contents: LSP.Hover["contents"]): string {
  if (typeof contents === "string") {
    return contents;
  }
  if (Array.isArray(contents)) {
    return contents.map(hoverMarkdown).join("\n\n");
  }
  return "language" in contents
    ? `\`\`\`${contents.language}\n${contents.value}\n\`\`\``
    : contents.value;
}

/**
 * Goals, diagnostics and hover from the Lean language server for the focused
 * Lean cell.
 */
export function leanLsp(cellId: CellId): Extension[] {
  const plugin = ViewPlugin.define((view) => new LeanLspPlugin(view, cellId));
  return [
    leanGoalsField,
    plugin,
    hoverTooltip((view, pos) => view.plugin(plugin)?.hover(pos) ?? null, {
      hideOnChange: true,
    }),
  ];
}
