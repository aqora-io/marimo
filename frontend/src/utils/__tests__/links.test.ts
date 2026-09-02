/* Copyright 2026 Marimo. All rights reserved. */
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { openNotebook } from "../links";

describe("openNotebook", () => {
  beforeAll(() => {
    Object.defineProperty(document, "baseURI", {
      value: "https://example.com/runner/abc/?file=readme.py",
      writable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("opens the notebook in a new tab when not embedded", () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null);

    openNotebook("dir/a b.py");

    expect(open).toHaveBeenCalledWith(
      "https://example.com/runner/abc/?file=dir%2Fa%20b.py",
      "_blank",
    );
  });

  it("asks the parent frame to open the notebook when embedded", () => {
    const postMessage = vi.fn();
    vi.spyOn(window, "parent", "get").mockReturnValue({
      postMessage,
    } as unknown as Window);
    const open = vi.spyOn(window, "open").mockImplementation(() => null);

    openNotebook("dir/a b.py");

    expect(postMessage).toHaveBeenCalledWith(
      { message: "open-notebook", path: "dir/a b.py" },
      "*",
    );
    expect(open).not.toHaveBeenCalled();
  });
});
