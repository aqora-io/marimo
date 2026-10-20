/* Copyright 2026 Marimo. All rights reserved. */

import { describe, expect, it } from "vitest";
import type { CellId } from "@/core/cells/ids";
import { buildLeanDocument } from "../languages/lean/document";

const id = (name: string) => name as CellId;

function leanCell(name: string, envName: string, code: string, env = "") {
  const envParam = env ? `, env=${env}` : "";
  return {
    id: id(name),
    code: `${envName} = lean(r"""\n${code}\n"""${envParam})`,
  };
}

const cells = [
  { id: id("import"), code: "from marimo_lean import lean" },
  leanCell("a", "env0", "theorem t : True := trivial"),
  leanCell("b", "env1", "def x := 1\ndef y := 2", "env0"),
  leanCell("c", "env2", "#check t", "env1"),
  leanCell("private", "_env", "#check x", "env1"),
];

describe("buildLeanDocument", () => {
  it("should put ancestors before the cell", () => {
    const lens = buildLeanDocument(
      { id: id("c"), code: "#check y", parentEnv: "env1" },
      cells,
    );
    expect(lens.cellIds).toEqual(["a", "b", "c"]);
    // The live editor text wins over the stored Python code
    expect(lens.mergedText).toBe(
      "theorem t : True := trivial\ndef x := 1\ndef y := 2\n#check y",
    );
    expect(lens.transformPosition({ line: 0, character: 7 }, id("c"))).toEqual({
      line: 3,
      character: 7,
    });
    expect(
      lens.reverseRange(
        { start: { line: 3, character: 0 }, end: { line: 3, character: 6 } },
        id("c"),
      ),
    ).toEqual({
      start: { line: 0, character: 0 },
      end: { line: 0, character: 6 },
    });
  });

  it("should undo the doubled braces of f-string ancestors", () => {
    const lens = buildLeanDocument(
      { id: id("g"), code: "#check f", parentEnv: "envF" },
      [
        {
          id: id("f"),
          code: 'envF = lean(rf"""\ndef f {{α : Type}} (a : α) := {x}\n""")',
        },
      ],
    );
    expect(lens.mergedText).toBe("def f {α : Type} (a : α) := {x}\n#check f");
  });

  it("should branch from a shared ancestor", () => {
    const lens = buildLeanDocument(
      { id: id("d"), code: "#check t", parentEnv: "env0" },
      cells,
    );
    expect(lens.cellIds).toEqual(["a", "d"]);
  });

  it("should stop at cells without a parent env", () => {
    const lens = buildLeanDocument(
      { id: id("a"), code: "theorem t : True := trivial", parentEnv: "" },
      cells,
    );
    expect(lens.cellIds).toEqual(["a"]);
  });

  it("should stop at unknown, private and non-name parents", () => {
    for (const parentEnv of ["missing", "_env", 'envs["x"]']) {
      const lens = buildLeanDocument(
        { id: id("d"), code: "#check t", parentEnv },
        cells,
      );
      expect(lens.cellIds).toEqual(["d"]);
    }
  });

  it("should stop at cycles", () => {
    const cyclic = [
      leanCell("a", "env0", "def a := 1", "env1"),
      leanCell("b", "env1", "def b := 1", "env0"),
    ];
    const lens = buildLeanDocument(
      { id: id("a"), code: "def a := 1", parentEnv: "env1" },
      cyclic,
    );
    expect(lens.cellIds).toEqual(["b", "a"]);
  });
});
