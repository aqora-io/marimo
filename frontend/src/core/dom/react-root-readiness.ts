/* Copyright 2026 Marimo. All rights reserved. */

import { atom } from "jotai";
import { store } from "@/core/state/jotai";

// Plugins render in separate React roots; notebook state alone cannot certify
// that their controls have committed, including roots inside shadow DOM.
export const pendingReactRootsAtom = atom<ReadonlySet<HTMLElement>>(
  new Set<HTMLElement>(),
);

export function setReactRootPending(element: HTMLElement, pending: boolean) {
  if (window.parent === window) {
    return;
  }
  store.set(pendingReactRootsAtom, (previous) => {
    if (previous.has(element) === pending) {
      return previous;
    }
    const next = new Set(previous);
    if (pending) {
      next.add(element);
    } else {
      next.delete(element);
    }
    return next;
  });
}

export function containsComposedNode(root: HTMLElement, element: HTMLElement) {
  let node: Node | null = element;
  while (node) {
    if (node === root) {
      return true;
    }
    node = node.parentNode ?? (node instanceof ShadowRoot ? node.host : null);
  }
  return false;
}
