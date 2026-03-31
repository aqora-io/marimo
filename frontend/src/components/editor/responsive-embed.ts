import { hasCellsAtom } from "@/core/cells/cells";
import { isConnectingAtom } from "@/core/network/connection";
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
      } else if (event.data === "bye") {
        mo.disconnect();
        ro.disconnect();
      } else {
        Logger.error(
          `Invalid message=${JSON.stringify(event.data)} from origin=${event.origin}`,
        );
      }
    };

    const abort = new AbortController();

    void (async () => {
      await untilMarimoReady(store, abort.signal);
      window.addEventListener("message", onWindowMessage);
    })().catch((error) => {
      if (!abort.signal.aborted) {
        Logger.error(error);
      }
    });


    return () => {
      abort.abort();
      window.removeEventListener("message", onWindowMessage);
      mo.disconnect();
      ro.disconnect();
    };
  }, []);

  return ref;
}

type JotaiStore = ReturnType<typeof useStore>;

const readyAtom = atom((get) => {
  const isConnecting = get(isConnectingAtom);
  const hasCells = get(hasCellsAtom);
  return !isConnecting && hasCells;
});

function untilMarimoReady(store: JotaiStore, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const unsub = store.sub(readyAtom, () => {
      const isReady = store.get(readyAtom);
      if (isReady) {
        unsub();
        resolve();
      }
    });

    signal?.addEventListener("abort", () => {
      unsub();
      reject(new Error("Aborted"));
    });

  });
}
