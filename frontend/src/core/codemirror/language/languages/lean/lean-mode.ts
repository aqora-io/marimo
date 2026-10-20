/* Copyright 2026 Marimo. All rights reserved. */

import type { StreamParser, StringStream } from "@codemirror/language";

const KEYWORDS = new Set([
  "abbrev",
  "at",
  "attribute",
  "axiom",
  "by",
  "calc",
  "class",
  "decreasing_by",
  "def",
  "deriving",
  "do",
  "else",
  "end",
  "example",
  "export",
  "extends",
  "for",
  "from",
  "fun",
  "have",
  "if",
  "import",
  "in",
  "inductive",
  "instance",
  "lemma",
  "let",
  "local",
  "macro",
  "match",
  "mutual",
  "namespace",
  "noncomputable",
  "notation",
  "obtain",
  "opaque",
  "open",
  "partial",
  "private",
  "protected",
  "return",
  "section",
  "set_option",
  "show",
  "structure",
  "suffices",
  "syntax",
  "termination_by",
  "then",
  "theorem",
  "universe",
  "unsafe",
  "variable",
  "where",
  "with",
]);

const TYPES = new Set(["Prop", "Sort", "Type"]);

// Lean identifiers may contain subscripts, primes, `!` and `?`, and be dotted;
// λ, Π and Σ are binders rather than letters.
const IDENT =
  /^(?![λΠΣ])[\p{L}_][\p{L}\p{N}_'!?₀-₉ₐ-ₜ]*(?:\.(?![λΠΣ])[\p{L}_][\p{L}\p{N}_'!?₀-₉ₐ-ₜ]*)*/u;

interface LeanState {
  /** Nesting depth of `/- -/` comments */
  commentDepth: number;
}

function blockComment(stream: StringStream, state: LeanState): string {
  while (!stream.eol()) {
    if (stream.match("/-")) {
      state.commentDepth++;
    } else if (stream.match("-/")) {
      state.commentDepth--;
      if (state.commentDepth === 0) {
        break;
      }
    } else {
      stream.next();
    }
  }
  return "comment";
}

/**
 * Syntax highlighting for Lean 4.
 */
export const leanMode: StreamParser<LeanState> = {
  name: "lean",
  startState: () => ({ commentDepth: 0 }),
  copyState: (state) => ({ ...state }),
  token(stream, state) {
    if (state.commentDepth > 0) {
      return blockComment(stream, state);
    }
    if (stream.eatSpace()) {
      return null;
    }
    if (stream.match("/-")) {
      state.commentDepth = 1;
      return blockComment(stream, state);
    }
    if (stream.match("--")) {
      stream.skipToEnd();
      return "comment";
    }
    if (stream.match(/^"(?:[^"\\]|\\.)*"?/)) {
      return "string";
    }
    if (stream.match(/^'(?:[^'\\]|\\.)'/)) {
      return "string";
    }
    if (stream.match(/^(?:#[a-z_]+|@\[)/)) {
      return "meta";
    }
    if (stream.match(/^(?:0x[\da-fA-F]+|0b[01]+|\d+(?:\.\d+)?)/)) {
      return "number";
    }
    if (stream.match(IDENT)) {
      const word = stream.current();
      if (KEYWORDS.has(word)) {
        return "keyword";
      }
      if (TYPES.has(word)) {
        return "typeName";
      }
      return word === "sorry" ? "invalid" : null;
    }
    if (stream.match(/^(?::=|=>|->|<-|[∀∃λΠΣ→←↔∧∨¬≠≤≥∈∉⊆×∘·⟨⟩|])/)) {
      return "operator";
    }
    stream.next();
    return null;
  },
  languageData: {
    commentTokens: { line: "--", block: { open: "/-", close: "-/" } },
  },
};
