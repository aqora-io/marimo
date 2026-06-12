import { literal, object, string } from "zod";
import type { UIElementId } from "@/core/cells/ids";
import { UI_ELEMENT_REGISTRY } from "@/core/dom/uiregistry";
import { assertNever } from "@/utils/assertNever";
import { Logger } from "@/utils/Logger";

type HTMLButtonSelector = "button";

export function initializeEmbedded() {
  if (window.parent === window) return;

  window.addEventListener("message", (event) => {
    const data = Message.safeParse(event.data);
    if (!data.success) return;

    const entry = UI_ELEMENT_REGISTRY.entries.get(
      data.data.objectId as UIElementId,
    );
    if (!entry) return;

    const element = entry.elements.values().next().value;
    if (!element || !element.shadowRoot) return;

    switch (data.data.message) {
      case "click":
        switch (element.nodeName) {
          case "MARIMO-BUTTON":
          case "MARIMO-CHECKBOX":
          case "MARIMO-DROPDOWN":
            element.shadowRoot.querySelector("button")?.click();
            break;

          case "MARIMO-TEXT":
            element.shadowRoot.querySelector("input")?.focus();
            break;

          case "MARIMO-RADIO":
            if (data.data.value) {
              element.shadowRoot
                .querySelector(
                  `button[value="${CSS.escape(data.data.value)}"]` as HTMLButtonSelector,
                )
                ?.click();
            }
            break;

          default:
            if (data.data.testid) {
              const matches = element.shadowRoot.querySelectorAll(
                `button[data-testid="${CSS.escape(data.data.testid)}"]` as HTMLButtonSelector,
              );
              if (matches.length !== 1) break;
              matches[0].click();
            }
            break;
        }
        break;

      case "change":
        switch (element.nodeName) {
          case "MARIMO-DROPDOWN":
            const $select = element.shadowRoot.querySelector("select");
            if ($select) {
              $select.value = data.data.value;
              $select.dispatchEvent(new Event("change", { bubbles: true }));
            }
            break;

          default:
            Logger.error("Cannot change %s", element.nodeName);
            break;
        }
        break;

      default:
        assertNever(data.data);
    }
  });
}

const Message = object({
  message: literal("click"),
  objectId: string(),
  value: string().optional(),
  testid: string().optional(),
}).or(
  object({
    message: literal("change"),
    objectId: string(),
    value: string(),
  }),
);
