import { literal, object, string } from "zod";
import type { UIElementId } from "@/core/cells/ids";
import { UI_ELEMENT_REGISTRY } from "@/core/dom/uiregistry";

type HTMLButtonSelector = "button";

export function initializeEmbedded() {
  // if (window.parent === window) return;

  window.addEventListener("message", (event) => {
    const data = Message.safeParse(event.data);
    if (!data.success) return;

    switch (data.data.message) {
      case "click":
        const entry = UI_ELEMENT_REGISTRY.entries.get(
          data.data.objectId as UIElementId,
        );
        if (!entry) break;

        const element = entry.elements.values().next().value;
        if (!element || !element.shadowRoot) break;

        switch (element.nodeName) {
          case "MARIMO-BUTTON":
          case "MARIMO-CHECKBOX":
            element.shadowRoot.querySelector("button")?.click();
            break;

          case "MARIMO-TEXT":
            element.shadowRoot.querySelector("input")?.focus();
            break;

          case "MARIMO-RADIO": {
            if (!data.data.value) break;
            element.shadowRoot
              .querySelector(
                `button[value="${CSS.escape(data.data.value)}"]` as HTMLButtonSelector,
              )
              ?.click();
            break;
          }

          default: {
            if (!data.data.testid) break;
            const matches = element.shadowRoot.querySelectorAll(
              `button[data-testid="${CSS.escape(data.data.testid)}"]` as HTMLButtonSelector,
            );
            if (matches.length !== 1) break;
            matches[0].click();
            break;
          }
        }
        break;
    }
  });
}

const Message = object({
  message: literal("click"),
  objectId: string(),
  value: string().optional(),
  testid: string().optional(),
});
