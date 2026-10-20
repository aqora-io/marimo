/* Copyright 2026 Marimo. All rights reserved. */

/**
 * Types for Lean's `Lean.Widget.getInteractiveGoals` RPC, and their
 * conversion into what the goals panel renders.
 *
 * Lean marks how the goals differ between before and after the tactic at the
 * cursor: at the start of a tactic it shows the goals before it, marking what
 * the tactic is about to change; inside or after a tactic it shows the goals
 * after it, marking what changed.
 */

type DiffTag =
  | "wasChanged"
  | "willChange"
  | "wasDeleted"
  | "willDelete"
  | "wasInserted"
  | "willInsert";

/** A reference to an object held by the Lean server's RPC session */
interface RpcRef {
  p: string;
}

interface SubexprInfo {
  info: RpcRef;
  subexprPos: string;
  diffStatus?: DiffTag;
}

type TaggedText =
  | { text: string }
  | { append: TaggedText[] }
  | { tag: [SubexprInfo, TaggedText] };

interface InteractiveHypothesisBundle {
  names: string[];
  type: TaggedText;
  val?: TaggedText;
  isInserted?: boolean;
  isRemoved?: boolean;
}

interface InteractiveGoal {
  hyps: InteractiveHypothesisBundle[];
  type: TaggedText;
  userName?: string;
  goalPrefix?: string;
  isInserted?: boolean;
  isRemoved?: boolean;
}

export interface InteractiveGoals {
  goals: InteractiveGoal[];
}

/** Whether something was inserted (green) or is removed (red), as in VS Code */
export type DiffStatus = "inserted" | "removed";

const DIFF_STATUS: Record<DiffTag, DiffStatus> = {
  wasChanged: "inserted",
  willChange: "removed",
  wasInserted: "inserted",
  willInsert: "inserted",
  willDelete: "removed",
  wasDeleted: "removed",
};

export interface DiffSegment {
  text: string;
  diff?: DiffStatus;
}

export interface HypothesisView {
  names: string;
  type: DiffSegment[];
  value?: DiffSegment[];
  diff?: DiffStatus;
}

export interface GoalView {
  caseName?: string;
  prefix: string;
  hyps: HypothesisView[];
  type: DiffSegment[];
  diff?: DiffStatus;
}

function bundleDiff({
  isInserted,
  isRemoved,
}: {
  isInserted?: boolean;
  isRemoved?: boolean;
}): DiffStatus | undefined {
  if (isInserted) {
    return "inserted";
  }
  return isRemoved ? "removed" : undefined;
}

/**
 * Flatten tagged text into segments, merging neighbours with the same diff.
 */
function toSegments(text: TaggedText): DiffSegment[] {
  const segments: DiffSegment[] = [];
  const visit = (node: TaggedText, diff: DiffStatus | undefined) => {
    if ("text" in node) {
      const last = segments.at(-1);
      if (last && last.diff === diff) {
        last.text += node.text;
      } else {
        segments.push(diff ? { text: node.text, diff } : { text: node.text });
      }
    } else if ("append" in node) {
      for (const child of node.append) {
        visit(child, diff);
      }
    } else {
      const [info, child] = node.tag;
      visit(child, info.diffStatus ? DIFF_STATUS[info.diffStatus] : diff);
    }
  };
  visit(text, undefined);
  return segments;
}

export function toGoalViews({ goals }: InteractiveGoals): GoalView[] {
  return goals.map((goal) => ({
    caseName: goal.userName,
    prefix: goal.goalPrefix ?? "⊢ ",
    hyps: goal.hyps.map((hyp) => ({
      names: hyp.names.join(" "),
      type: toSegments(hyp.type),
      value: hyp.val && toSegments(hyp.val),
      diff: bundleDiff(hyp),
    })),
    type: toSegments(goal.type),
    diff: bundleDiff(goal),
  }));
}

/**
 * All RPC references in an RPC response, so they can be released.
 */
export function collectRpcRefs(value: unknown): RpcRef[] {
  if (Array.isArray(value)) {
    return value.flatMap(collectRpcRefs);
  }
  if (value === null || typeof value !== "object") {
    return [];
  }
  const keys = Object.keys(value);
  if (
    keys.length === 1 &&
    keys[0] === "p" &&
    typeof (value as RpcRef).p === "string"
  ) {
    return [value as RpcRef];
  }
  return Object.values(value).flatMap(collectRpcRefs);
}
