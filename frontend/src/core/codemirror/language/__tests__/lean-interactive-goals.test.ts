/* Copyright 2026 Marimo. All rights reserved. */

import { describe, expect, it } from "vitest";
import {
  collectRpcRefs,
  type InteractiveGoals,
  toGoalViews,
} from "../languages/lean/interactive-goals";

// `Lean.Widget.getInteractiveGoals` at the start of `intro h` in
// `theorem swap (p q : Prop) : p ∧ q → q ∧ p := by intro h; ...`
const BEFORE_INTRO: InteractiveGoals = {
  goals: [
    {
      goalPrefix: "⊢ ",
      hyps: [
        {
          names: ["p", "q"],
          type: {
            tag: [{ info: { p: "0" }, subexprPos: "/" }, { text: "Prop" }],
          },
        },
      ],
      isInserted: false,
      type: {
        tag: [
          { info: { p: "1" }, subexprPos: "/" },
          {
            append: [
              {
                tag: [
                  {
                    diffStatus: "willDelete",
                    info: { p: "2" },
                    subexprPos: "/0",
                  },
                  {
                    append: [
                      {
                        tag: [
                          { info: { p: "3" }, subexprPos: "/0/0/1" },
                          { text: "p" },
                        ],
                      },
                      { text: " ∧ " },
                      {
                        tag: [
                          { info: { p: "4" }, subexprPos: "/0/1" },
                          { text: "q" },
                        ],
                      },
                    ],
                  },
                ],
              },
              { text: " → " },
              {
                tag: [
                  { info: { p: "5" }, subexprPos: "/1" },
                  { text: "q ∧ p" },
                ],
              },
            ],
          },
        ],
      },
    },
  ],
};

// ...and after `intro h`
const AFTER_INTRO: InteractiveGoals = {
  goals: [
    {
      goalPrefix: "⊢ ",
      hyps: [
        { names: ["p", "q"], type: { text: "Prop" } },
        { names: ["h"], isInserted: true, type: { text: "p ∧ q" } },
      ],
      type: { text: "q ∧ p" },
    },
  ],
};

describe("toGoalViews", () => {
  it("should mark what the tactic at the cursor is about to change", () => {
    expect(toGoalViews(BEFORE_INTRO)).toEqual([
      {
        caseName: undefined,
        prefix: "⊢ ",
        hyps: [
          {
            names: "p q",
            type: [{ text: "Prop" }],
            value: undefined,
            diff: undefined,
          },
        ],
        type: [{ text: "p ∧ q", diff: "removed" }, { text: " → q ∧ p" }],
        diff: undefined,
      },
    ]);
  });

  it("should mark what the tactic before the cursor changed", () => {
    const [goal] = toGoalViews(AFTER_INTRO);
    expect(goal.hyps.map(({ names, diff }) => [names, diff])).toEqual([
      ["p q", undefined],
      ["h", "inserted"],
    ]);
    expect(goal.type).toEqual([{ text: "q ∧ p" }]);
  });
});

describe("collectRpcRefs", () => {
  it("should find every RPC reference", () => {
    expect(collectRpcRefs(BEFORE_INTRO).map(({ p }) => p)).toEqual([
      "0",
      "1",
      "2",
      "3",
      "4",
      "5",
    ]);
    expect(collectRpcRefs(null)).toEqual([]);
  });
});
