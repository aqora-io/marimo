/* Copyright 2026 Marimo. All rights reserved. */
import { act, waitFor } from "@testing-library/react";
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { uiElementId } from "@/__tests__/branded";
import { MarimoValueReadyEvent } from "@/core/dom/events";
import { initializeUIElement } from "@/core/dom/ui-element";
import { UI_ELEMENT_REGISTRY } from "@/core/dom/uiregistry";
import { ButtonPlugin } from "@/plugins/impl/ButtonPlugin";
import { DropdownPlugin } from "@/plugins/impl/DropdownPlugin";
import { RadioPlugin } from "@/plugins/impl/RadioPlugin";
import { SliderPlugin } from "@/plugins/impl/SliderPlugin";
import { TextAreaPlugin } from "@/plugins/impl/TextAreaPlugin";
import { TextInputPlugin } from "@/plugins/impl/TextInputPlugin";
import { initializeEmbedded } from "../embedded";
import { registerReactComponent } from "../registerReactComponent";

const OBJECT_ID = uiElementId("test-input");
let parent: Window;
let stop: (() => void) | undefined;
const mounted: HTMLElement[] = [];

beforeAll(() => {
  initializeUIElement();
  registerReactComponent(new TextInputPlugin());
  registerReactComponent(new TextAreaPlugin());
  registerReactComponent(new SliderPlugin());
  registerReactComponent(new ButtonPlugin());
  registerReactComponent(new DropdownPlugin());
  registerReactComponent(new RadioPlugin());
});

beforeEach(() => {
  const parentFrame = document.createElement("iframe");
  document.body.append(parentFrame);
  mounted.push(parentFrame);
  if (!parentFrame.contentWindow) {
    throw new Error("Missing parent frame");
  }
  parent = parentFrame.contentWindow;
  vi.spyOn(window, "parent", "get").mockReturnValue(parent);
  const addedListeners = vi.spyOn(window, "addEventListener");
  initializeEmbedded();
  const listener = addedListeners.mock.calls.find(
    ([type]) => type === "message",
  )?.[1];
  if (!listener) {
    throw new Error("Missing embedded listener");
  }
  stop = () => window.removeEventListener("message", listener);
  addedListeners.mockRestore();
});

afterEach(() => {
  stop?.();
  act(() => mounted.splice(0).forEach((element) => element.remove()));
  UI_ELEMENT_REGISTRY.entries.clear();
  vi.restoreAllMocks();
});

async function mount(tag: string, data: Record<string, unknown>) {
  const wrapper = document.createElement("marimo-ui-element");
  wrapper.setAttribute("object-id", OBJECT_ID);
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(data)) {
    element.dataset[key] = JSON.stringify(value);
  }
  wrapper.append(element);
  mounted.push(wrapper);
  await act(async () => {
    document.body.append(wrapper);
  });
  const shadow = element.shadowRoot;
  if (!shadow) {
    throw new Error("Missing widget shadow root");
  }
  await waitFor(() =>
    expect(
      shadow.querySelector("input, textarea, button, select, [role=slider]"),
    ).not.toBeNull(),
  );
  return { element, shadow };
}

function send(message: Record<string, unknown>, source: Window = parent) {
  act(() =>
    window.dispatchEvent(
      new MessageEvent("message", {
        source,
        origin: "https://aqora.test",
        data: { objectId: OBJECT_ID, ...message },
      }),
    ),
  );
}

function textData(extra: Record<string, unknown> = {}) {
  return { initialValue: "initial", placeholder: "", label: null, ...extra };
}

function sliderData(extra: Record<string, unknown> = {}) {
  return {
    initialValue: 4,
    start: 1,
    stop: 10,
    step: 1,
    steps: null,
    label: "Amount",
    ...extra,
  };
}

describe("embedded input handoff", () => {
  it.each(["marimo-text", "marimo-text-area"])(
    "restores a typed value and caret in %s without duplicate kernel updates",
    async (tag) => {
      const { shadow } = await mount(tag, textData());
      const field = shadow.querySelector("input, textarea");
      if (
        !(
          field instanceof HTMLInputElement ||
          field instanceof HTMLTextAreaElement
        )
      ) {
        throw new Error("Missing text field");
      }
      const ready = vi.fn();
      document.addEventListener(MarimoValueReadyEvent.TYPE, ready);
      try {
        send({
          message: "change",
          value: "typed before ready",
          focus: true,
          selectionStart: 5,
          selectionEnd: 11,
        });
        await waitFor(() =>
          expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(
            "typed before ready",
          ),
        );
        expect(field.value).toBe("typed before ready");
        expect(shadow.activeElement).toBe(field);
        expect([field.selectionStart, field.selectionEnd]).toEqual([5, 11]);
        expect(ready).toHaveBeenCalledTimes(1);
      } finally {
        document.removeEventListener(MarimoValueReadyEvent.TYPE, ready);
      }
    },
  );

  it("updates a React-controlled debounced input through its native event handler", async () => {
    const { shadow } = await mount("marimo-text", textData({ debounce: 10 }));
    send({ message: "change", value: "retained draft" });
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe("retained draft"),
    );
    expect(shadow.querySelector("input")?.value).toBe("retained draft");
  });

  it.each(["marimo-text", "marimo-text-area"])(
    "commits an earlier blur-debounced %s edit without stealing focus",
    async (tag) => {
      const { shadow } = await mount(tag, textData({ debounce: true }));
      send({ message: "change", value: "finished field", focus: false });
      await waitFor(() =>
        expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(
          "finished field",
        ),
      );
      expect(shadow.activeElement).toBeNull();
    },
  );

  it("keeps the focused blur-debounced draft editable until the user leaves it", async () => {
    const { shadow } = await mount("marimo-text", textData({ debounce: true }));
    const field = shadow.querySelector("input");
    if (!field) {
      throw new Error("Missing text field");
    }
    send({ message: "change", value: "unfinished draft", focus: true });
    expect(field.value).toBe("unfinished draft");
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe("initial");
    act(() => field.blur());
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(
        "unfinished draft",
      ),
    );
  });

  it("does not steal focus for an earlier queued field", async () => {
    const { shadow } = await mount("marimo-text", textData());
    send({ message: "change", value: "retained" });
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe("retained"),
    );
    expect(shadow.activeElement).toBeNull();
  });

  it("replays the preview track position through the actual slider and registry", async () => {
    const { shadow } = await mount("marimo-slider", sliderData());
    const ready = vi.fn();
    document.addEventListener(MarimoValueReadyEvent.TYPE, ready);
    try {
      send({ message: "click", testid: "track", position: 0.8 });
      await waitFor(() =>
        expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(8),
      );
      expect(shadow.querySelector("[role=slider]")).toHaveAttribute(
        "aria-valuenow",
        "8",
      );
      expect(shadow.activeElement).toBe(shadow.querySelector("[role=slider]"));
      expect(ready).toHaveBeenCalledTimes(1);
    } finally {
      document.removeEventListener(MarimoValueReadyEvent.TYPE, ready);
    }
  });

  it("synchronizes duplicate slider instances with one kernel notification", async () => {
    const first = await mount("marimo-slider", sliderData());
    const second = await mount("marimo-slider", sliderData());
    const ready = vi.fn();
    document.addEventListener(MarimoValueReadyEvent.TYPE, ready);
    try {
      send({ message: "change", value: "7" });
      await waitFor(() => {
        expect(first.shadow.querySelector("[role=slider]")).toHaveAttribute(
          "aria-valuenow",
          "7",
        );
        expect(second.shadow.querySelector("[role=slider]")).toHaveAttribute(
          "aria-valuenow",
          "7",
        );
      });
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(7);
      expect(ready).toHaveBeenCalledTimes(1);
    } finally {
      document.removeEventListener(MarimoValueReadyEvent.TYPE, ready);
    }
  });

  it("uses index space for mapped slider steps and preserves numeric changes", async () => {
    const { shadow } = await mount(
      "marimo-slider",
      sliderData({
        initialValue: 0,
        start: 0,
        stop: 2,
        steps: [10, 50, 100],
        showValue: true,
      }),
    );
    send({ message: "click", testid: "track", position: 1 });
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(2),
    );
    expect(shadow.textContent).toContain("100");
    send({ message: "change", value: "1", focus: true });
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(1),
    );
    expect(shadow.textContent).toContain("50");
  });

  it("replays an included slider input in mapped value space", async () => {
    const { shadow } = await mount(
      "marimo-slider",
      sliderData({
        initialValue: 0,
        start: 0,
        stop: 2,
        steps: [10, 50, 100],
        includeInput: true,
      }),
    );
    send({ message: "change", value: "50", focus: false });
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(1),
    );
    expect(shadow.querySelector("[role=slider]")).toHaveAttribute(
      "aria-valuenow",
      "1",
    );
  });

  it("rounds decimal slider positions to the same step as live interaction", async () => {
    const { shadow } = await mount(
      "marimo-slider",
      sliderData({ initialValue: 0, start: 0, stop: 1, step: 0.1 }),
    );
    send({ message: "click", testid: "track", position: 0.34 });
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(0.3),
    );
    expect(shadow.querySelector("[role=slider]")).toHaveAttribute(
      "aria-valuenow",
      "0.3",
    );
    send({ message: "change", value: "NaN" });
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(0.3);
  });

  it("ignores disabled sliders and malformed positions", async () => {
    await mount("marimo-slider", sliderData({ disabled: true }));
    send({ message: "click", testid: "track", position: 0.8 });
    send({ message: "change", value: "NaN" });
    send({ message: "click", testid: "track", position: 2 });
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(4);
  });

  it("does not modify a disabled text field", async () => {
    const { shadow } = await mount("marimo-text", textData({ disabled: true }));
    send({ message: "change", value: "ignored", focus: true });
    expect(shadow.querySelector("input")?.value).toBe("initial");
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe("initial");
  });

  it("preserves existing button clicks and rejects messages from other windows", async () => {
    const { shadow } = await mount("marimo-button", {
      initialValue: 0,
      label: "Run",
      kind: "neutral",
    });
    send({ message: "click" }, window);
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(0);
    send({ message: "click" });
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(1),
    );
    expect(shadow.activeElement).toBe(shadow.querySelector("button"));
  });

  it("preserves existing dropdown changes", async () => {
    const { shadow } = await mount("marimo-dropdown", {
      initialValue: ["first"],
      label: null,
      options: ["first", "second"],
      allowSelectNone: false,
    });
    send({ message: "change", value: "second" });
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toEqual(["second"]),
    );
    expect(shadow.querySelector("select")?.value).toBe("second");
  });
  it("restores keyboard focus to a button without activating it", async () => {
    const { shadow } = await mount("marimo-button", {
      initialValue: 0,
      label: "Run",
      kind: "neutral",
    });
    const ready = vi.fn();
    document.addEventListener(MarimoValueReadyEvent.TYPE, ready);
    try {
      send({ message: "focus", testid: "marimo-plugin-button" });
      expect(shadow.activeElement).toBe(shadow.querySelector("button"));
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(0);
      expect(ready).not.toHaveBeenCalled();
    } finally {
      document.removeEventListener(MarimoValueReadyEvent.TYPE, ready);
    }
  });

  it.each(["marimo-text", "marimo-text-area"])(
    "restores %s focus and caret without replacing its current value",
    async (tag) => {
      const { shadow } = await mount(tag, textData());
      const field = shadow.querySelector("input, textarea");
      if (
        !(
          field instanceof HTMLInputElement ||
          field instanceof HTMLTextAreaElement
        )
      ) {
        throw new Error("Missing text field");
      }
      send({
        message: "focus",
        value: "must not overwrite",
        selectionStart: 1,
        selectionEnd: 4,
      });
      expect(shadow.activeElement).toBe(field);
      expect(field.value).toBe("initial");
      expect([field.selectionStart, field.selectionEnd]).toEqual([1, 4]);
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe("initial");
    },
  );

  it("focuses the live slider without changing its value", async () => {
    const { shadow } = await mount("marimo-slider", sliderData());
    send({ message: "focus", testid: "thumb" });
    expect(shadow.activeElement).toBe(shadow.querySelector("[role=slider]"));
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(4);
  });

  it("focuses a select without changing its selection", async () => {
    const { shadow } = await mount("marimo-dropdown", {
      initialValue: ["first"],
      label: null,
      options: ["first", "second"],
      allowSelectNone: false,
    });
    send({ message: "focus" });
    expect(shadow.activeElement).toBe(shadow.querySelector("select"));
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toEqual(["first"]);
  });

  it("ignores disabled controls and focus requests from unrelated windows", async () => {
    const { shadow } = await mount("marimo-button", {
      initialValue: 0,
      label: "Run",
      kind: "neutral",
      disabled: true,
    });
    send({ message: "focus", testid: "marimo-plugin-button" });
    expect(shadow.activeElement).toBeNull();
    send({ message: "focus" }, window);
    expect(shadow.activeElement).toBeNull();
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(0);
  });
  it("resolves a radio button by value without selecting it", async () => {
    const { shadow } = await mount("marimo-radio", {
      initialValue: "first",
      label: null,
      options: ["first", "second"],
    });
    send({ message: "focus", value: "second" });
    expect(shadow.activeElement).toBe(
      shadow.querySelector('button[value="second"]'),
    );
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe("first");
  });

  it("rejects unrelated-window focus requests on an enabled button", async () => {
    const { shadow } = await mount("marimo-button", {
      initialValue: 0,
      label: "Run",
      kind: "neutral",
    });
    send({ message: "focus", testid: "marimo-plugin-button" }, window);
    expect(shadow.activeElement).toBeNull();
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(0);
  });

  it("distinguishes a slider's native value input from its thumb during focus restoration", async () => {
    const { shadow } = await mount(
      "marimo-slider",
      sliderData({ includeInput: true }),
    );
    send({ message: "focus", value: "4" });
    expect(shadow.activeElement).toBe(
      shadow.querySelector('input:not([style*="display: none"])'),
    );
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(4);
  });
  it("replays a queued button click once without taking focus from outside the notebook", async () => {
    const { shadow } = await mount("marimo-button", {
      initialValue: 0,
      label: "Run",
      kind: "neutral",
    });
    const outside = document.createElement("input");
    document.body.append(outside);
    mounted.push(outside);
    outside.focus();
    send({ message: "click", focus: false });
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(1),
    );
    expect(document.activeElement).toBe(outside);
    expect(shadow.activeElement).toBeNull();
  });

  it("replays a slider position without taking focus from outside the notebook", async () => {
    const { shadow } = await mount("marimo-slider", sliderData());
    const outside = document.createElement("input");
    document.body.append(outside);
    mounted.push(outside);
    outside.focus();
    send({ message: "click", testid: "track", position: 0.8, focus: false });
    await waitFor(() =>
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toBe(8),
    );
    expect(document.activeElement).toBe(outside);
    expect(shadow.activeElement).toBeNull();
  });

  it("does not steal outside focus when replaying a searchable dropdown click", async () => {
    HTMLElement.prototype.scrollIntoView ??= vi.fn();
    await mount("marimo-dropdown", {
      initialValue: ["first"],
      label: null,
      options: ["first", "second"],
      allowSelectNone: false,
      searchable: true,
    });
    const outside = document.createElement("input");
    document.body.append(outside);
    mounted.push(outside);
    outside.focus();
    send({ message: "click", focus: false });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(document.activeElement).toBe(outside);
    expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toEqual(["first"]);
  });
  it.each([undefined, true])(
    "opens a searchable dropdown with default/owned focus intent=%s",
    async (focus) => {
      HTMLElement.prototype.scrollIntoView ??= vi.fn();
      await mount("marimo-dropdown", {
        initialValue: ["first"],
        label: null,
        options: ["first", "second"],
        allowSelectNone: false,
        searchable: true,
      });
      send({ message: "click", ...(focus === undefined ? {} : { focus }) });
      await waitFor(() => {
        const search = document.querySelector("input[role=combobox]");
        expect(search).not.toBeNull();
        expect(document.activeElement).toBe(search);
      });
      expect(UI_ELEMENT_REGISTRY.lookupValue(OBJECT_ID)).toEqual(["first"]);
    },
  );
});
