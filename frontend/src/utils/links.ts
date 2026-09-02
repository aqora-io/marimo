/* Copyright 2026 Marimo. All rights reserved. */
import { asURL } from "./url";

/**
 * Open a notebook in a new tab.
 *
 * When embedded in an iframe, the host page is asked to open it instead: a
 * new top-level tab on the notebook origin would not carry the embedding
 * context's (partitioned) session cookie and would land on the login page.
 * @param path - The path to the notebook.
 */
export function openNotebook(path: string) {
  if (typeof window !== "undefined" && window.parent !== window) {
    window.parent.postMessage({ message: "open-notebook", path }, "*");
    return;
  }
  // There is no leading `/` in the path in order to work when marimo is at a subpath.
  window.open(asURL(`?file=${encodeURIComponent(path)}`).toString(), "_blank");
}
