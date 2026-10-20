/* Copyright 2026 Marimo. All rights reserved. */
// @vitest-environment jsdom

import { EditorSelection, EditorState } from "@codemirror/state";
import { EditorView } from "@codemirror/view";
import { afterEach, describe, expect, it } from "vitest";
import { LeanLanguageAdapter } from "../languages/lean/lean";
import { replaceAbbreviation } from "../languages/lean/unicode-input";

const adapter = new LeanLanguageAdapter();

describe("LeanLanguageAdapter", () => {
  it("should round trip Lean code and env names", () => {
    const pythonCode = 'env1 = lean(r"""\n#check Nat\n""", env=env0)';
    const [code, offset, metadata] = adapter.transformIn(pythonCode);
    expect(code).toBe("#check Nat");
    expect(metadata).toEqual({
      envName: "env1",
      parentEnv: "env0",
      commentLines: [],
    });
    const [out, outOffset] = adapter.transformOut(code, metadata);
    expect(out).toBe(pythonCode);
    expect(outOffset).toBe(offset);
  });
});

describe("replaceAbbreviation", () => {
  let view: EditorView | null = null;

  afterEach(() => {
    view?.destroy();
    view = null;
  });

  function run(doc: string): [boolean, string, number] {
    view = new EditorView({
      state: EditorState.create({
        doc,
        selection: EditorSelection.cursor(doc.length),
      }),
    });
    const replaced = replaceAbbreviation(view);
    return [
      replaced,
      view.state.doc.toString(),
      view.state.selection.main.head,
    ];
  }

  it("should replace an abbreviation before the cursor", () => {
    expect(run("p \\and")).toEqual([true, "p ∧", 3]);
    expect(run("\\N")).toEqual([true, "ℕ", 1]);
  });

  it("should place the cursor inside paired symbols", () => {
    expect(run("exact \\<>")).toEqual([true, "exact ⟨⟩", 7]);
  });

  it("should do nothing without an abbreviation", () => {
    expect(run("p and")).toEqual([false, "p and", 5]);
    expect(run("p \\")).toEqual([false, "p \\", 3]);
  });
});
