/* Copyright 2026 Marimo. All rights reserved. */

import { describe, expect, it } from "vitest";
import { LeanParser } from "../parsers/lean-parser.js";

const parser = new LeanParser();

const SWAP = `theorem swap (p q : Prop) : p ∧ q → q ∧ p := by
  intro h; exact ⟨h.2, h.1⟩`;

describe("LeanParser", () => {
  describe("transformIn", () => {
    it("should extract Lean code and the env names", () => {
      const pythonCode = `env1 = lean(r"""\n${SWAP}\n""", env=env0)`;
      const { code, offset, metadata } = parser.transformIn(pythonCode);
      expect(code).toBe(SWAP);
      expect(offset).toBe(17);
      expect(pythonCode.slice(offset)).toMatch(/^theorem swap/);
      expect(metadata).toEqual({
        envName: "env1",
        parentEnv: "env0",
        quotePrefix: "r",
        commentLines: [],
      });
    });

    it.each([
      ['"""', ""],
      ['r"""', "r"],
      ['f"""', "f"],
      ['rf"""', "rf"],
      ['fr"""', "fr"],
    ] as const)("should read the %s prefix", (quote, quotePrefix) => {
      const pythonCode = `e = lean(${quote}\n#eval {x} + 1\n""")`;
      const { code, offset, metadata } = parser.transformIn(pythonCode);
      expect(code).toBe("#eval {x} + 1");
      expect(pythonCode.slice(offset)).toMatch(/^#eval/);
      expect(metadata.quotePrefix).toBe(quotePrefix);
    });

    it("should handle a missing env", () => {
      const { code, metadata } = parser.transformIn(
        `env0 = lean(r"""\n${SWAP}\n""")`,
      );
      expect(code).toBe(SWAP);
      expect(metadata.parentEnv).toBe("");
    });

    it("should keep arbitrary env expressions", () => {
      const { metadata } = parser.transformIn(
        `e = lean(r"""\n#check Nat\n""", env=envs["base"])`,
      );
      expect(metadata.parentEnv).toBe('envs["base"]');
    });

    it("should dedent indented Lean code", () => {
      const { code } = parser.transformIn(
        `e = lean(r"""\n    def x := 1\n      + 2\n""")`,
      );
      expect(code).toBe("def x := 1\n  + 2");
    });

    it("should keep leading comments", () => {
      const { code, metadata } = parser.transformIn(
        `# A comment\n# Another\ne = lean(r"""\n#check Nat\n""")`,
      );
      expect(code).toBe("#check Nat");
      expect(metadata.commentLines).toEqual(["# A comment", "# Another"]);
    });

    it("should unescape non-raw strings", () => {
      const { code } = parser.transformIn(
        String.raw`e = lean("""
#eval \"\"\" ++ \"\\\"
""")`,
      );
      expect(code).toBe(String.raw`#eval """ ++ "\"`);
    });

    it("should return the code unchanged when unsupported", () => {
      const pythonCode = "x = 1";
      const { code, offset, metadata } = parser.transformIn(pythonCode);
      expect(code).toBe(pythonCode);
      expect(offset).toBe(0);
      expect(metadata).toEqual(parser.defaultMetadata);
    });
  });

  describe("transformOut", () => {
    it("should wrap Lean code in a raw string", () => {
      const { code, offset } = parser.transformOut(SWAP, {
        envName: "env1",
        parentEnv: "env0",
        quotePrefix: "r",
        commentLines: [],
      });
      expect(code).toBe(`env1 = lean(r"""\n${SWAP}\n""", env=env0)`);
      expect(code.slice(offset)).toMatch(/^theorem swap/);
    });

    it("should omit a missing env", () => {
      const { code } = parser.transformOut("#check Nat", {
        envName: "_env",
        parentEnv: "",
        quotePrefix: "r",
        commentLines: ["# comment"],
      });
      expect(code).toBe(`# comment\n_env = lean(r"""\n#check Nat\n""")`);
    });

    it.each([
      [
        "",
        String.raw`_env = lean("""
#eval \"\\n\" ++ {x}
""")`,
      ],
      [
        "f",
        String.raw`_env = lean(f"""
#eval \"\\n\" ++ {x}
""")`,
      ],
      [
        "rf",
        String.raw`_env = lean(rf"""
#eval "\n" ++ {x}
""")`,
      ],
    ] as const)("should write the %j prefix", (quotePrefix, expected) => {
      const { code } = parser.transformOut(String.raw`#eval "\n" ++ {x}`, {
        ...parser.defaultMetadata,
        quotePrefix,
      });
      expect(code).toBe(expected);
    });

    it("should escape code that contains triple quotes", () => {
      const { code } = parser.transformOut(String.raw`#eval """ ++ "\"`, {
        ...parser.defaultMetadata,
      });
      expect(code).toBe(
        String.raw`_env = lean("""
#eval \"\"\" ++ \"\\\"
""")`,
      );
      // Only the raw prefix is dropped
      expect(
        parser.transformOut('#eval """', {
          ...parser.defaultMetadata,
          quotePrefix: "rf",
        }).code,
      ).toBe(String.raw`_env = lean(f"""
#eval \"\"\"
""")`);
    });
  });

  describe("round trip", () => {
    it.each([
      [SWAP, "env1", "env0"],
      ["#check Nat", "_env", ""],
      ["", "_env", ""],
      [String.raw`def s := "\n"`, "e", "base"],
      [String.raw`#eval """ ++ "\"`, "e", ""],
      ['def s := ""', "e", ""],
      ["def f {α : Type} (a : α) := a", "e", ""],
    ])("should round trip %j", (leanCode, envName, parentEnv) => {
      for (const quotePrefix of ["", "r", "f", "rf"] as const) {
        const metadata = { envName, parentEnv, quotePrefix, commentLines: [] };
        const out = parser.transformOut(leanCode, metadata);
        expect(parser.isSupported(out.code)).toBe(true);
        const back = parser.transformIn(out.code);
        expect(back.code).toBe(leanCode);
        expect(back.metadata).toEqual({
          ...metadata,
          // Raw strings cannot contain `"""`
          quotePrefix: leanCode.includes('"""')
            ? quotePrefix.replace("r", "")
            : quotePrefix,
        });
      }
    });

    it("should round trip the default code", () => {
      const { code, metadata } = parser.transformIn(parser.defaultCode);
      expect(code).toBe("");
      expect(parser.transformOut(code, metadata).code).toBe(
        '_env = lean(r"""\n\n""")',
      );
    });
  });

  describe("isSupported", () => {
    it.each([
      "",
      `e = lean(r"""\n#check Nat\n""")`,
      `e = lean(r"""#check Nat""", env=base)`,
      `# comment\ne = lean(r"""\n#check Nat\n""")`,
      `e = lean(f"""#check {x}""")`,
      `e = lean(rf"""#check {x}""", env=base)`,
      `e = lean(fr"""#check {x}""")`,
      `e = lean(f"""\n#check {x}\n""",\n    env=base,\n)`,
    ])("should support %j", (pythonCode) => {
      expect(parser.isSupported(pythonCode)).toBe(true);
    });

    it.each([
      "x = 1",
      `lean(r"""#check Nat""")`,
      `e = foo(r"""#check Nat""")`,
      `e = mo.lean(r"""#check Nat""")`,
      `e = lean(r"""#check Nat""")\nprint(e)`,
      `e = lean(r"""#check Nat""") # trailing`,
      `e = lean(r"""#check Nat""", other=1)`,
      `e = lean(code)`,
      `e = lean(b"""#check Nat""")`,
      `e = lean(f"#check {x}")`,
      `e = lean("""\\n#check Nat""")`,
      `a = b = lean(r"""#check Nat""")`,
      `e = lean(r"""#check Nat""") if x else None`,
    ])("should not support %j", (pythonCode) => {
      expect(parser.isSupported(pythonCode)).toBe(false);
    });
  });
});
