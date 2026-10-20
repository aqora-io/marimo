/* Copyright 2026 Marimo. All rights reserved. */

import type { FormatResult, LanguageParser, ParseResult } from "../types.js";
import { parseArgsKwargs, parsePythonAST, safeDedent } from "../utils/index.js";

export interface LeanMetadata {
  /** Variable the resulting Lean environment is assigned to */
  envName: string;
  /** Python expression for the parent environment, or "" for none */
  parentEnv: string;
  commentLines: readonly string[];
}

interface LeanParseInfo {
  envName: string;
  parentEnv: string;
  leanCode: string;
  commentLines: string[];
  startPosition: number;
}

/**
 * Parser for Lean cells (`lean()` from the marimo-lean package).
 *
 * Converts between Python code like `env1 = lean(r"""...""", env=env0)` and
 * plain Lean code.
 */
export class LeanParser implements LanguageParser<LeanMetadata> {
  readonly type = "lean";
  readonly defaultCode = '_env = lean(r"""\n""")';
  readonly defaultMetadata: LeanMetadata = {
    envName: "_env",
    parentEnv: "",
    commentLines: [],
  };

  transformIn(pythonCode: string): ParseResult<LeanMetadata> {
    pythonCode = pythonCode.trim();
    const info = parseLeanStatement(pythonCode);
    if (!info) {
      return {
        code: pythonCode,
        offset: 0,
        metadata: { ...this.defaultMetadata },
      };
    }
    return {
      code: info.leanCode,
      offset: info.startPosition,
      metadata: {
        envName: info.envName,
        parentEnv: info.parentEnv,
        commentLines: info.commentLines,
      },
    };
  }

  transformOut(code: string, metadata: LeanMetadata): FormatResult {
    const { envName, parentEnv, commentLines } = metadata;
    // Raw strings keep Lean code verbatim, but cannot contain `"""`.
    const [quote, body] = code.includes('"""')
      ? ['"""', code.replaceAll("\\", "\\\\").replaceAll('"', '\\"')]
      : ['r"""', code];
    const start = [...commentLines, `${envName} = lean(${quote}\n`].join("\n");
    const envParam = parentEnv ? `, env=${parentEnv}` : "";
    return {
      code: `${start}${body}\n"""${envParam})`,
      offset: start.length,
    };
  }

  isSupported(pythonCode: string): boolean {
    if (pythonCode.trim() === "") {
      return true;
    }
    if (!pythonCode.includes("lean(")) {
      return false;
    }
    return parseLeanStatement(pythonCode.trim()) !== null;
  }
}

/**
 * Parse `name = lean(<str>[, env=<expr>])`, optionally preceded by comments.
 * Only the string forms produced by `transformOut` are accepted.
 */
function parseLeanStatement(code: string): LeanParseInfo | null {
  const tree = parsePythonAST(code);
  const commentLines: string[] = [];
  let assign = null;
  for (let node = tree.topNode.firstChild; node; node = node.nextSibling) {
    if (assign) {
      return null;
    }
    if (node.name === "AssignStatement") {
      assign = node;
    } else if (node.name === "Comment") {
      commentLines.push(code.slice(node.from, node.to));
    } else {
      return null;
    }
  }

  const target = assign?.firstChild;
  const op = target?.nextSibling;
  const call = op?.nextSibling;
  if (
    target?.name !== "VariableName" ||
    op?.name !== "AssignOp" ||
    call?.name !== "CallExpression" ||
    call.nextSibling
  ) {
    return null;
  }

  const callee = call.firstChild;
  const argList = callee?.nextSibling;
  if (
    callee?.name !== "VariableName" ||
    code.slice(callee.from, callee.to) !== "lean" ||
    argList?.name !== "ArgList"
  ) {
    return null;
  }

  const { args, kwargs } = parseArgsKwargs(argList.cursor(), code);
  if (
    args.length !== 1 ||
    args[0].name !== "String" ||
    kwargs.some(({ key }) => key !== "env")
  ) {
    return null;
  }

  const literal = code.slice(args[0].from, args[0].to);
  let content: string | null = null;
  let prefixLength = 0;
  if (literal.startsWith('r"""') && literal.endsWith('"""')) {
    prefixLength = 4;
    content = literal.slice(prefixLength, -3);
  } else if (literal.startsWith('"""') && literal.endsWith('"""')) {
    prefixLength = 3;
    content = unescape(literal.slice(prefixLength, -3));
  }
  if (content === null) {
    return null;
  }

  return {
    envName: code.slice(target.from, target.to),
    parentEnv: kwargs.find(({ key }) => key === "env")?.value ?? "",
    leanCode: safeDedent(content),
    commentLines,
    // Where the Lean code starts, after the newline following the quotes.
    startPosition:
      args[0].from + prefixLength + (content.startsWith("\n") ? 1 : 0),
  };
}

/**
 * Undo the escaping of `transformOut`; null for any other escape sequence.
 */
function unescape(content: string): string | null {
  let supported = true;
  const result = content.replaceAll(/\\(.)/gs, (_, char: string) => {
    if (char !== "\\" && char !== '"') {
      supported = false;
    }
    return char;
  });
  return supported ? result : null;
}
