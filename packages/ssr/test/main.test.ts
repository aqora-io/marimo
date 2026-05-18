import { describe, test } from "vitest";
import { renderNotebook } from "../src/index";
import { sampleNotebook } from "./samples";

describe("renderNotebook", () => {
  test("sample notebook", async ({ expect }) => {
    expect(await renderNotebook(sampleNotebook)).toMatchSnapshot();
  });
});
