/* Copyright 2026 Marimo. All rights reserved. */

import { normalizeName } from "@/core/cells/names";
import { useAutoGrowInputProps } from "@/hooks/useAutoGrowInputProps";

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
