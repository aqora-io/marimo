import { notebookAtom, type NotebookState } from "@/core/cells/cells";
import { SCRATCH_CELL_ID } from "@/core/cells/ids";
import { connectionAtom } from "@/core/network/connection";
import { Logger } from "@/utils/Logger";
import { atom, useStore } from "jotai";
import { useEffect, useRef } from "react";

export function useResponsiveEmbedRef<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const store = useStore();

  useEffect(() => {
    if (window.parent === window) return;

    const root = ref.current;
    if (!root) return;

    let unsub: (() => void) | undefined;

    const measureHeight = () => {
      // const children: HTMLElement[] = Array.prototype.slice.call(root.children);
      // return Math.max(
      //   root.scrollHeight,
      //   ...children.map((child) => child.scrollHeight),
      // );
      return root.scrollHeight;
    };

    const ro = new ResizeObserver(() => {
      window.parent.postMessage(
        { message: "resize", height: measureHeight() },
        "*",
      );
    });

    const mo = new MutationObserver(() => {
      window.parent.postMessage(
        { message: "resize", height: measureHeight() },
        "*",
      );
    });

    const onWindowMessage = (event: MessageEvent<unknown>) => {
      if (event.data === "hello") {
        window.parent.postMessage(
          { message: "hello", height: measureHeight() },
          event.origin,
        );
        ro.observe(root);
        mo.observe(root, { childList: true, subtree: true });
        unsub?.();
        unsub = store.sub(readinessAtom, () => {
          const readiness = store.get(readinessAtom);
          window.parent.postMessage(readiness, "*");
        });
      } else if (event.data === "bye") {
        mo.disconnect();
        ro.disconnect();
        unsub?.();
        unsub = undefined;
      } else {
        Logger.error(
          `Invalid message=${JSON.stringify(event.data)} from origin=${event.origin}`,
        );
      }
    };

    window.addEventListener("message", onWindowMessage);
    return () => {
      window.removeEventListener("message", onWindowMessage);
      mo.disconnect();
      ro.disconnect();
      unsub?.();
    };
  }, []);

  return ref;
}

type Readiness = "connecting" | "running" | "ready";

const readinessAtom = atom<Readiness>((get) => {
  const connection = get(connectionAtom);
  const notebook = get(notebookAtom);

  if (connection.state !== "OPEN") {
    return "connecting";
  } else if (!notebookHasCompleted(notebook)) {
    return "running";
  }
  return "ready";
});

function notebookHasCompleted(notebook: NotebookState): boolean {
  const runtimes = Object.entries(notebook.cellRuntime).filter(([id]) => id !== SCRATCH_CELL_ID);
  return runtimes.length > 0 && runtimes.every(([_id, cell]) => cell.output !== null || cell.errored);
}
