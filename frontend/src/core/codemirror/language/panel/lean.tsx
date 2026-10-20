/* Copyright 2026 Marimo. All rights reserved. */

import type { QuotePrefixKind } from "@marimo-team/smart-cells";
import { InfoIcon } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Tooltip } from "@/components/ui/tooltip";
import { normalizeName } from "@/core/cells/names";
import { useAutoGrowInputProps } from "@/hooks/useAutoGrowInputProps";
import { cn } from "@/utils/cn";
import type {
  DiffSegment,
  DiffStatus,
  GoalView,
} from "../languages/lean/interactive-goals";
import type { LeanGoals } from "../languages/lean/lsp-extension";
import { getQuotePrefix } from "./markdown";

/**
 * Input for the name of a Lean environment variable.
 * When `optional`, an empty value means no environment.
 */
export const LeanEnvInput: React.FC<{
  label: string;
  value: string;
  optional?: boolean;
  onChange: (name: string) => void;
}> = ({ label, value, optional, onChange }) => {
  const { spanProps, inputProps } = useAutoGrowInputProps({ minWidth: 50 });

  const sanitizeAndUpdate = (e: React.SyntheticEvent<HTMLInputElement>) => {
    const raw = e.currentTarget.value.trim();
    // Normalize the name to a valid variable name
    const name = optional && raw === "" ? "" : normalizeName(raw, false);
    e.currentTarget.value = name;
    onChange(name);
  };

  return (
    <label className="flex gap-2 items-center">
      <span className="select-none">{label}</span>
      <input
        {...inputProps}
        defaultValue={value}
        placeholder={optional ? "none" : undefined}
        onBlur={sanitizeAndUpdate}
        onKeyDown={(e) => {
          if (e.key === "Enter" && e.shiftKey) {
            sanitizeAndUpdate(e);
          }
        }}
        className="min-w-14 w-auto border border-border rounded px-1 focus-visible:outline-hidden focus-visible:ring-1 focus-visible:ring-ring"
      />
      <span {...spanProps} />
    </label>
  );
};

const LeanQuotePrefixTooltip: React.FC = () => (
  <div className="flex flex-col gap-3.5">
    <section className="flex flex-col gap-0.5">
      <header className="flex items-center gap-1">
        <code className="text-xs px-1 py-0.5 bg-(--slate-2) rounded">r</code>
        <span className="font-semibold">Raw String</span>
      </header>
      <p className="text-sm text-muted-foreground">
        Write backslashes without escaping them
      </p>
      <pre className="text-xs bg-(--slate-2) p-2 rounded">#eval "a\nb"</pre>
    </section>
    <section className="flex flex-col gap-0.5">
      <header className="flex items-center gap-1">
        <code className="text-xs px-1 py-0.5 bg-(--slate-2) rounded">f</code>
        <span className="font-semibold">Format String</span>
      </header>
      <p className="text-sm text-muted-foreground">
        Interpolate Python values, doubling Lean&apos;s own braces. The editor
        shows no goals or errors for these cells.
      </p>
      <pre className="text-xs bg-(--slate-2) p-2 rounded">
        {"#eval {n.value} + 1\ndef id' {{α : Type}} (a : α) := a"}
      </pre>
    </section>
  </div>
);

/**
 * Checkboxes for the `r` and `f` prefixes of the Lean code's Python string.
 */
export const LeanQuotePrefixToggles: React.FC<{
  quotePrefix: QuotePrefixKind;
  onChange: (quotePrefix: QuotePrefixKind) => void;
}> = ({ quotePrefix, onChange }) => {
  const toggle = (prefix: QuotePrefixKind, checked: boolean | string) => {
    if (typeof checked === "boolean") {
      onChange(
        getQuotePrefix({ currentQuotePrefix: quotePrefix, checked, prefix }),
      );
    }
  };

  return (
    <div className="flex items-center gap-1.5 ml-auto">
      <span>r</span>
      <Checkbox
        aria-label="Toggle raw string"
        className="w-3 h-3"
        checked={quotePrefix.includes("r")}
        onCheckedChange={(checked) => toggle("r", checked)}
      />
      <span>f</span>
      <Checkbox
        aria-label="Toggle f-string"
        className="w-3 h-3"
        checked={quotePrefix.includes("f")}
        onCheckedChange={(checked) => toggle("f", checked)}
      />
      <Tooltip content={<LeanQuotePrefixTooltip />}>
        <InfoIcon className="w-3 h-3" />
      </Tooltip>
    </div>
  );
};

const DIFF_CLASSES: Record<DiffStatus, string> = {
  inserted: "rounded-sm bg-(--grass-6)",
  removed: "rounded-sm bg-(--red-6)",
};

const Segments: React.FC<{ segments: DiffSegment[] }> = ({ segments }) =>
  segments.map((segment, index) => (
    <span
      // Segments have no identity besides their position
      // oxlint-disable-next-line react/no-array-index-key
      key={index}
      className={segment.diff && DIFF_CLASSES[segment.diff]}
    >
      {segment.text}
    </span>
  ));

const Goal: React.FC<{ goal: GoalView }> = ({ goal }) => (
  <div
    className={cn(
      "border-l-2 border-transparent pl-1",
      goal.diff === "inserted" && "border-(--grass-8)",
      goal.diff === "removed" && "border-(--red-8)",
    )}
  >
    {goal.caseName && (
      <div>
        <strong>case</strong> {goal.caseName}
      </div>
    )}
    {goal.hyps.map((hyp, index) => (
      // Hypotheses have no identity besides their position
      // oxlint-disable-next-line react/no-array-index-key
      <div key={index}>
        <strong className={hyp.diff && DIFF_CLASSES[hyp.diff]}>
          {hyp.names}
        </strong>
        {" : "}
        <Segments segments={hyp.type} />
        {hyp.value && (
          <>
            {" := "}
            <Segments segments={hyp.value} />
          </>
        )}
      </div>
    ))}
    <div>
      <strong>{goal.prefix}</strong>
      <Segments segments={goal.type} />
    </div>
  </div>
);

/**
 * The goals at the cursor, from the Lean language server. As in VS Code,
 * green marks what the tactic before the cursor changed, and red what the
 * tactic starting at the cursor is about to change.
 */
export const LeanGoalsView: React.FC<{ goals: LeanGoals | null }> = ({
  goals,
}) => {
  if (!goals) {
    return null;
  }
  return (
    <div className="flex flex-col gap-1 px-2 pt-2 font-mono text-xs font-normal text-foreground whitespace-pre-wrap">
      {goals.kind === "term" ? (
        <>
          <span className="font-sans text-muted-foreground">Expected type</span>
          <div>{goals.goal}</div>
        </>
      ) : goals.goals.length === 0 ? (
        <span className="font-sans text-muted-foreground">No goals</span>
      ) : (
        goals.goals.map((goal, index) => (
          // Goals have no identity besides their position
          // oxlint-disable-next-line react/no-array-index-key
          <Goal key={index} goal={goal} />
        ))
      )}
    </div>
  );
};
