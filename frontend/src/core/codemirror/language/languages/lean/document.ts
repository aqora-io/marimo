/* Copyright 2026 Marimo. All rights reserved. */

import { LeanParser } from "@marimo-team/smart-cells";
import type { CellId } from "@/core/cells/ids";
import { createNotebookLens, type NotebookLens } from "../../../lsp/lens";

const parser = new LeanParser();

// Only plain names can refer to another cell's env; `_`-prefixed names are
// private to their cell.
const ENV_NAME = /^[A-Za-z]\w*$/;

interface LeanCell {
  id: CellId;
  /** Lean code, as shown in the editor */
  code: string;
  parentEnv: string;
}

/**
 * The Lean document for a cell: the Lean code of its ancestors (following
 * `env=` to the cell defining that env) followed by its own code, mirroring
 * how `lean(code, env=parent)` elaborates on top of `parent`.
 *
 * @param cell - the cell, with its live editor text and parent env
 * @param cells - all cells, with their Python code
 */
export function buildLeanDocument(
  cell: LeanCell,
  cells: { id: CellId; code: string }[],
): NotebookLens {
  const envs = new Map<string, LeanCell>();
  for (const { id, code } of cells) {
    if (id === cell.id || code.trim() === "" || !parser.isSupported(code)) {
      continue;
    }
    const { code: leanCode, metadata } = parser.transformIn(code);
    envs.set(metadata.envName, {
      id,
      // The best guess for an f-string: its literal braces, but placeholders
      // as written
      code: metadata.quotePrefix.includes("f")
        ? leanCode.replaceAll(/\{\{|\}\}/g, (braces) => braces[0])
        : leanCode,
      parentEnv: metadata.parentEnv,
    });
  }

  const chain = [cell];
  const seen = new Set([cell.id]);
  let parentEnv = cell.parentEnv;
  while (ENV_NAME.test(parentEnv)) {
    const parent = envs.get(parentEnv);
    if (!parent || seen.has(parent.id)) {
      break;
    }
    seen.add(parent.id);
    chain.unshift(parent);
    parentEnv = parent.parentEnv;
  }

  return createNotebookLens(
    chain.map(({ id }) => id),
    Object.fromEntries(chain.map(({ id, code }) => [id, code])),
  );
}
