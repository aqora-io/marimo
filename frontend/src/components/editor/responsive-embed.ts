import { Logger } from "@/utils/Logger";
import { useEffect, useRef } from "react";

export function useResponsiveEmbedRef<T extends HTMLElement>() {
  const ref = useRef<T>(null);

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
    window.addEventListener("message", onWindowMessage);

    return () => {
      window.removeEventListener("message", onWindowMessage);
      mo.disconnect();
      ro.disconnect();
    };
  }, []);

  return ref;
}
