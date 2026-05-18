import assert from "node:assert";

import parse, { Element, Text } from "html-react-parser";
import { renderToString as renderKatex } from "katex";
import "katex/contrib/mhchem";
import { createElement, type ReactElement, type ReactNode } from "react";

import * as api from "@marimo-team/marimo-api";
import { UI_PLUGINS } from "@marimo-team/frontend/unstable_internal/plugins/plugins.ts";
import type { IPlugin } from "@marimo-team/frontend/unstable_internal/plugins/types.ts";
import type { PluginFunctions } from "@marimo-team/frontend/unstable_internal/plugins/core/rpc.ts";
import { parseAttrValue } from "@marimo-team/frontend/unstable_internal/core/dom/htmlUtils.ts";
import { sanitizeHtml } from "@marimo-team/frontend/unstable_internal/plugins/core/sanitize-html.ts";

/// Pre-render cells which contain HTML outputs.
export function prerenderNotebook({
  cells,
  ...session
}: api.Session["NotebookSessionV1"]): api.Session["NotebookSessionV1"] {
  return {
    ...session,
    cells: cells.map(({ outputs, ...cell }) => {
      return {
        ...cell,
        outputs: outputs.map((output0) => {
          if (output0.type !== "data") return output0;

          const { data, ...output } = output0;
          return {
            ...output,
            data: Object.fromEntries(
              Object.entries(data).map((entry) => {
                const [mimeType, data] = entry;
                switch (mimeType) {
                  case "text/html":
                  case "text/markdown":
                  case "text/latex":
                    if (typeof data === "string") {
                      return [mimeType, prerenderNotebookCell(cell.id, data)];
                    } else {
                      return entry;
                    }

                  default:
                    return entry;
                }
              }),
            ),
          };
        }),
      };
    }),
  };
}

export function prerenderNotebookCell(cellId: string, html: string): ReactNode {
  html = sanitizeHtml(html);
  return parse(html, {
    replace: (domNode) => {
      if (!(domNode instanceof Element)) return;

      if (domNode.tagName === "marimo-ui-element") {
        for (const child of domNode.children) {
          if (child instanceof Element) {
            const plugin = UI_PLUGINS.find(
              (plg) => plg.tagName === child.tagName,
            );
            if (plugin) {
              return prenderCellDomNode(cellId, child, plugin) as ReactElement;
            }
          }
        }
      }

      if (domNode.tagName === "marimo-tex") {
        return prerenderTex(extractInnerText(domNode)) as ReactElement;
      }
    },
  });
}

export function prenderCellDomNode<S, D, F extends PluginFunctions>(
  cellId: string,
  elm: Element,
  plugin: IPlugin<S, D, F>,
): ReactNode {
  const data = plugin.validator.decode(extractElementDataset(elm));

  assert(elm.parent instanceof Element, "parent is not element");
  const parentAttribs = elm.parent.attribs;
  const host = new Proxy(
    {},
    {
      get: (_target, prop) => {
        switch (prop) {
          case "closest":
            return () => host;
          case "getAttribute":
            return (attr: string) => parentAttribs[attr];
          case "id":
            return `cell-${cellId}`;
          default:
            console.log("host proxy get", { prop });
            break;
        }
      },
    },
  ) as HTMLElement;

  const functions: F = new Proxy(
    {},
    {
      get: (_target, prop) => {
        return (...args: unknown[]) => {
          console.log("plugin function call", {
            plugin: plugin.tagName,
            prop,
            args,
          });
        };
      },
    },
  ) as F;

  const value = parseAttrValue<S>(elm.attribs["data-initial-value"]);

  return plugin.render({
    host,
    functions,
    data,
    value,
    setValue: () => {},
  });
}

function extractElementDataset(elm: Element): Record<string, unknown> {
  const dataset: Record<string, unknown> = {};
  for (const [attr, value] of Object.entries(elm.attribs)) {
    const name = attribIntoDatasetName(attr);
    if (name === null) continue;
    dataset[name] = parseAttrValue(value);
  }
  return dataset;
}

function attribIntoDatasetName(attr: string): string | null {
  if (!attr.startsWith("data-")) return null;
  const [head, ...tail] = attr.slice("data-".length).split("-");
  return head + tail.map((t) => t[0].toUpperCase() + t.slice(1)).join("");
}

function extractInnerText(elm: Element): string {
  function* walkText(elm: Element): Generator<string> {
    for (const child of elm.children) {
      if (child instanceof Element) {
        yield* walkText(child);
      } else if (child instanceof Text) {
        yield child.data;
      }
    }
  }

  return Array.from(walkText(elm)).join("");
}

export function prerenderTex(tex: string): ReactNode {
  // Required, even if empty. (see https://github.com/KaTeX/KaTeX/issues/2513)
  const macros = {
    // KaTeX doesn't support \mbox; map it to the equivalent \text
    "\\mbox": "\\text{#1}",
  };

  let html: string;
  if (tex.startsWith("||(||(") && tex.endsWith("||)||)")) {
    // when $$...$$ is used without newlines before/after the $$.
    html = renderKatex(tex.slice(6, -6), {
      displayMode: true,
      globalGroup: true,
      throwOnError: false,
      macros: macros,
    });
  } else if (tex.startsWith("||(") && tex.endsWith("||)")) {
    // Inline math, via $...$
    html = renderKatex(tex.slice(3, -3), {
      displayMode: false,
      globalGroup: true,
      throwOnError: false,
      macros: macros,
    });
  } else if (tex.startsWith("||[") && tex.endsWith("||]")) {
    // Display math, via $$...$$
    html = renderKatex(tex.slice(3, -3), {
      displayMode: true,
      globalGroup: true,
      throwOnError: false,
      macros: macros,
    });
  } else {
    return;
  }

  return createElement("span", { dangerouslySetInnerHTML: { __html: html } });
}
