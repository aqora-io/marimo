import { act, render } from "@testing-library/react";
import { createStore, Provider } from "jotai";
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
