/* Copyright 2026 Marimo. All rights reserved. */

import { Prec } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { AbbreviationProvider } from "@leanprover/unicode-input";

const provider = new AbbreviationProvider({
  abbreviationCharacter: "\\",
  customTranslations: {},
  eagerReplacementEnabled: true,
});

// Marks where the cursor goes in replacements like `\<>` -> `⟨⟩`
const CURSOR = "$CURSOR";

/**
 * Replace an abbreviation like `\and` right before the cursor with its
 * symbol (`∧`), as in the Lean VS Code extension.
 */
export function replaceAbbreviation(view: EditorView): boolean {
  const { main } = view.state.selection;
  if (!main.empty) {
    return false;
  }
  const line = view.state.doc.lineAt(main.head);
  const match = /\\([^\s\\]+)$/.exec(line.text.slice(0, main.head - line.from));
  const symbol = match && provider.getReplacementText(match[1]);
  if (!match || !symbol) {
    return false;
  }
  const from = main.head - match[0].length;
  const insert = symbol.replace(CURSOR, "");
  const cursor = symbol.indexOf(CURSOR);
  view.dispatch({
    changes: { from, to: main.head, insert },
    selection: { anchor: from + (cursor === -1 ? insert.length : cursor) },
    userEvent: "input",
  });
  return true;
}

/**
 * Unicode input for Lean: abbreviations are replaced on space or Tab.
 */
export function leanUnicodeInput() {
  return [
    EditorView.inputHandler.of((view, _from, _to, text) => {
      if (text !== " " || !replaceAbbreviation(view)) {
        return false;
      }
      view.dispatch(view.state.replaceSelection(" "));
      return true;
    }),
    Prec.highest(keymap.of([{ key: "Tab", run: replaceAbbreviation }])),
  ];
}
