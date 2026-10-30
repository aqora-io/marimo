import { createElement, use } from "react";
import { z } from "zod";
import { registerReactComponent } from "@/plugins/core/registerReactComponent";
import { act, render } from "@testing-library/react";
import { createStore, Provider, useAtomValue } from "jotai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MockNotebook } from "@/__mocks__/notebook";
import { cellId } from "@/__tests__/branded";
import { notebookAtom } from "@/core/cells/cells";
import { connectionAtom } from "@/core/network/connection";
import { useResponsiveEmbedRef } from "../responsive-embed";

function EmbeddedNotebook() {
  const ref = useResponsiveEmbedRef<HTMLDivElement>();
  return <div ref={ref} />;
}

function setup(ready = false) {
  const parentFrame = document.createElement("iframe");
  document.body.append(parentFrame);
  const parent = parentFrame.contentWindow;
  if (!parent) {
    throw new Error("Missing parent window");
  }
  vi.spyOn(window, "parent", "get").mockReturnValue(parent);
  const postMessage = vi
    .spyOn(parent, "postMessage")
    .mockImplementation(() => undefined);
  const store = createStore();
  const id = cellId("cell");
  const completed = MockNotebook.notebookState({
    cellData: { [id]: {} },
    cellRuntime: { [id]: { errored: true } },
  });
  if (ready) {
    store.set(connectionAtom, { state: "OPEN" });
    store.set(notebookAtom, completed);
  }
  const view = render(
    <Provider store={store}>
      <EmbeddedNotebook />
    </Provider>,
  );
  const send = (data: string, source: Window = parent) =>
    act(() => {
      window.dispatchEvent(
        new MessageEvent("message", {
          data,
          source,
          origin: "https://aqora.test",
        }),
      );
    });
  return { store, completed, postMessage, send, view };
}

afterEach(() => vi.restoreAllMocks());

describe("responsive embed handshake", () => {
  it("replays readiness when the notebook completed before hello", () => {
    const { send, postMessage } = setup(true);
    send("hello");
    expect(postMessage.mock.calls).toEqual([
      [{ message: "hello", height: 0 }, "https://aqora.test"],
      ["ready", "https://aqora.test"],
    ]);
  });

  it("reports current and future state once after repeated handshakes", () => {
    const { send, postMessage, store, completed, view } = setup();
    send("hello");
    send("hello");
    act(() => store.set(connectionAtom, { state: "OPEN" }));
    act(() => store.set(notebookAtom, completed));
    expect(postMessage.mock.calls.map(([message]) => message)).toEqual([
      { message: "hello", height: 0 },
      "connecting",
      { message: "hello", height: 0 },
      "connecting",
      "running",
      "ready",
    ]);
    send("bye");
    postMessage.mockClear();
    act(() => store.set(connectionAtom, { state: "CONNECTING" }));
    expect(postMessage).not.toHaveBeenCalled();
    view.unmount();
  });

  it("ignores messages from other windows", () => {
    const { send, postMessage } = setup(true);
    send("hello", window);
    expect(postMessage).not.toHaveBeenCalled();
  });
});

describe("committed embed readiness", () => {
  it("does not publish ready while completed notebook content is hidden", () => {
    const { store, completed, postMessage, send, view } = setup();
    view.rerender(
      <Provider store={store}>
        <HiddenNotebook />
      </Provider>,
    );
    send("hello");
    act(() => {
      store.set(connectionAtom, { state: "OPEN" });
      store.set(notebookAtom, completed);
    });
    expect(postMessage.mock.calls.map(([message]) => message)).not.toContain(
      "ready",
    );
    view.rerender(
      <Provider store={store}>
        <HiddenNotebook visible={true} />
      </Provider>,
    );
    expect(postMessage.mock.calls.at(-1)?.[0]).toBe("ready");
  });

  it("waits for a React plugin's content instead of its Suspense fallback", async () => {
    const { promise, resolve } = Promise.withResolvers<boolean>();
    const tag = "marimo-readiness-test-plugin";
    function AsyncButton() {
      use(promise);
      return <button type="button">Committed plugin</button>;
    }
    registerReactComponent({
      tagName: tag,
      validator: z.any(),
      render: () => <AsyncButton />,
    });
    const { store, postMessage, send, view } = setup(true);
    view.rerender(
      <Provider store={store}>
        <NotebookPlugin tag={tag} />
      </Provider>,
    );
    send("hello");
    expect(postMessage.mock.calls.map(([message]) => message)).not.toContain(
      "ready",
    );
    await act(async () => resolve(true));
    expect(postMessage.mock.calls.at(-1)?.[0]).toBe("ready");
    const host = view.container.querySelector(tag);
    expect(host?.shadowRoot?.querySelector("button")?.textContent).toBe(
      "Committed plugin",
    );
    view.unmount();
  });

  it("waits for nested shadow-root plugins and releases disconnected roots", async () => {
    const { promise, resolve } = Promise.withResolvers<boolean>();
    const inner = "marimo-readiness-nested-plugin";
    const outer = "marimo-readiness-parent-plugin";
    function AsyncButton() {
      use(promise);
      return <button type="button">Nested control</button>;
    }
    registerReactComponent({
      tagName: inner,
      validator: z.any(),
      render: () => <AsyncButton />,
    });
    registerReactComponent({
      tagName: outer,
      validator: z.any(),
      render: () => createElement(inner),
    });
    const { store, postMessage, send, view } = setup(true);
    view.rerender(
      <Provider store={store}>
        <NotebookPlugin tag={outer} />
      </Provider>,
    );
    await act(async () => {
      await Promise.resolve();
    });
    send("hello");
    expect(
      view.container.querySelector(outer)?.shadowRoot?.querySelector(inner),
    ).not.toBeNull();
    expect(postMessage.mock.calls.map(([message]) => message)).not.toContain(
      "ready",
    );
    view.rerender(
      <Provider store={store}>
        <NotebookPlugin />
      </Provider>,
    );
    expect(postMessage.mock.calls.at(-1)?.[0]).toBe("ready");
    await act(async () => resolve(true));
    view.unmount();
  });

  it("does not replay stale ready between a disconnect and its React commit", () => {
    const { store, postMessage, send } = setup(true);
    send("hello");
    postMessage.mockClear();
    act(() => {
      store.set(connectionAtom, { state: "CONNECTING" });
      send("hello");
    });
    expect(postMessage.mock.calls.map(([message]) => message)).not.toContain(
      "ready",
    );
  });

  it("publishes ready after the output DOM is committed", () => {
    const { store, completed, postMessage, send, view } = setup();
    view.rerender(
      <Provider store={store}>
        <NotebookOutput />
      </Provider>,
    );
    send("hello");
    expect(view.container.querySelector("h1")).toBeNull();
    const committed: boolean[] = [];
    postMessage.mockImplementation((message) => {
      if (message === "ready") {
        committed.push(!!view.container.querySelector("h1"));
      }
    });
    act(() => {
      store.set(connectionAtom, { state: "OPEN" });
      store.set(notebookAtom, completed);
    });
    expect(committed).toEqual([true]);
  });
});

function HiddenNotebook({ visible = false }: { visible?: boolean }) {
  const ref = useResponsiveEmbedRef<HTMLDivElement>(visible);
  return (
    <div ref={ref}>
      <div hidden={!visible}>Notebook</div>
    </div>
  );
}

function NotebookOutput() {
  const notebook = useAtomValue(notebookAtom);
  const ref = useResponsiveEmbedRef<HTMLDivElement>();
  return (
    <div ref={ref}>
      {notebook.cellRuntime[cellId("cell")] && <h1>Output</h1>}
    </div>
  );
}

function NotebookPlugin({ tag }: { tag?: string }) {
  const ref = useResponsiveEmbedRef<HTMLDivElement>();
  return <div ref={ref}>{tag && createElement(tag)}</div>;
}
