import { flushSync } from "react-dom";
import { boolean, literal, number, object, string } from "zod";
import type { UIElementId } from "@/core/cells/ids";
import { MarimoValueUpdateEvent } from "@/core/dom/events";
import { parseDataset } from "@/core/dom/htmlUtils";
import { SliderPlugin } from "@/plugins/impl/SliderPlugin";
import { UI_ELEMENT_REGISTRY } from "@/core/dom/uiregistry";
import { assertNever } from "@/utils/assertNever";
import { Logger } from "@/utils/Logger";

type HTMLButtonSelector = "button";

export function initializeEmbedded() {
  if (window.parent === window) {
    return;
  }

  window.addEventListener("message", (event) => {
    if (event.source !== window.parent) {
      return;
    }
    const data = Message.safeParse(event.data);
    if (!data.success) {
      return;
    }

    const entry = UI_ELEMENT_REGISTRY.entries.get(
      data.data.objectId as UIElementId,
    );
    if (!entry) {
      return;
    }

    const element = entry.elements.values().next().value;
    if (!element || !element.shadowRoot) {
      return;
    }

    switch (data.data.message) {
      case "click":
        switch (element.nodeName) {
          case "MARIMO-BUTTON":
          case "MARIMO-CHECKBOX":
          case "MARIMO-DROPDOWN":
            clickButton(element.shadowRoot.querySelector("button"));
            break;

          case "MARIMO-TEXT":
          case "MARIMO-TEXT-AREA":
            element.shadowRoot
              .querySelector<HTMLElement>("input, textarea")
              ?.focus({ preventScroll: true });
            break;

          case "MARIMO-SLIDER":
            changeSlider(
              element,
              data.data.objectId as UIElementId,
              data.data,
              true,
            );
            break;

          case "MARIMO-RADIO":
            if (data.data.value) {
              clickButton(
                element.shadowRoot.querySelector(
                  `button[value="${CSS.escape(data.data.value)}"]` as HTMLButtonSelector,
                ),
              );
            }
            break;

          default:
            if (data.data.testid) {
              const matches = element.shadowRoot.querySelectorAll(
                `button[data-testid="${CSS.escape(data.data.testid)}"]` as HTMLButtonSelector,
              );
              if (matches.length !== 1) {
                break;
              }
              clickButton(matches[0]);
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
              if (data.data.focus) {
                $select.focus({ preventScroll: true });
              }
            }
            break;

          case "MARIMO-SLIDER": {
            const field = editableField(element.shadowRoot);
            if (field) {
              changeField(field, data.data);
            } else {
              changeSlider(
                element,
                data.data.objectId as UIElementId,
                data.data,
                data.data.focus ?? false,
              );
            }
            break;
          }

          default: {
            const field = editableField(element.shadowRoot);
            if (field) {
              changeField(field, data.data);
            } else {
              Logger.error("Cannot change %s", element.nodeName);
            }
            break;
          }
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
  position: number().min(0).max(1).optional(),
}).or(
  object({
    message: literal("change"),
    objectId: string(),
    value: string(),
    focus: boolean().optional(),
    selectionStart: number().int().nonnegative().optional(),
    selectionEnd: number().int().nonnegative().optional(),
  }),
);

function clickButton(button: HTMLButtonElement | null) {
  button?.focus({ preventScroll: true });
  button?.click();
}

function editableField(root: ShadowRoot) {
  const fields = [
    ...root.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>(
      "input:not([type=hidden]):not([hidden]), textarea:not([hidden])",
    ),
  ].filter((field) => field.style.display !== "none");
  return fields.length === 1 ? fields[0] : undefined;
}

function changeField(
  field: HTMLInputElement | HTMLTextAreaElement,
  change: {
    value: string;
    focus?: boolean;
    selectionStart?: number;
    selectionEnd?: number;
  },
) {
  if (field.disabled || field.readOnly) {
    return;
  }
  if (
    field instanceof HTMLInputElement &&
    ["file", "checkbox", "radio", "button", "submit", "reset"].includes(
      field.type,
    )
  ) {
    return;
  }
  // React tracks assignments to the instance's value property. Use the native
  // setter so its normal input/change handlers observe the replayed edit.
  const prototype =
    field instanceof HTMLInputElement
      ? HTMLInputElement.prototype
      : HTMLTextAreaElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value")?.set?.call(
    field,
    change.value,
  );
  flushSync(() => {
    field.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
    field.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
  });
  if (change.focus) {
    field.focus({ preventScroll: true });
    if (
      field instanceof HTMLTextAreaElement ||
      ["text", "search", "tel", "url", "password"].includes(field.type)
    ) {
      const start = change.selectionStart ?? change.value.length;
      field.setSelectionRange(start, change.selectionEnd ?? start);
    }
  } else {
    // An earlier queued field was already left in the preview. Its buffered
    // edit must commit before the next replayed action without taking focus.
    field.dispatchEvent(
      new FocusEvent("focusout", { bubbles: true, composed: true }),
    );
  }
}

const sliderValidator = new SliderPlugin().validator;
function changeSlider(
  element: HTMLElement,
  objectId: UIElementId,
  change: { position?: number; value?: string },
  focus: boolean,
) {
  const slider =
    element.shadowRoot?.querySelector<HTMLElement>("[role=slider]");
  if (!slider) {
    return;
  }
  const parsed = sliderValidator.safeParse(parseDataset(element));
  if (!parsed.success || parsed.data.disabled) {
    return;
  }
  if (focus) {
    slider.focus({ preventScroll: true });
  }
  const { start, stop, step } = parsed.data;
  const value =
    change.position === undefined
      ? change.value?.trim()
        ? Number(change.value)
        : Number.NaN
      : start + (stop - start) * change.position;
  if (!Number.isFinite(value)) {
    return;
  }
  // Match Radix's step rounding and clamping, including the index space
  // used by sliders with an explicit list of values.
  const stepSize = step ?? 1;
  if (stepSize <= 0) {
    return;
  }
  const [coefficient, exponent = "0"] = String(stepSize).split("e");
  const precision = Math.max(
    0,
    (coefficient.split(".")[1]?.length ?? 0) - Number(exponent),
  );
  const factor = 10 ** precision;
  const snapped = Math.round((value - start) / stepSize) * stepSize + start;
  const nextValue = Math.min(
    stop,
    Math.max(start, Math.round(snapped * factor) / factor),
  );
  if (
    !Number.isFinite(nextValue) ||
    Object.is(UI_ELEMENT_REGISTRY.lookupValue(objectId), nextValue)
  ) {
    return;
  }
  // Radix sliders have no editable native input. Update the source instance
  // before broadcasting the same committed value to its siblings and kernel.
  element.dispatchEvent(
    MarimoValueUpdateEvent.create({ detail: { value: nextValue, element } }),
  );
  UI_ELEMENT_REGISTRY.broadcastValueUpdate(element, objectId, nextValue);
}
