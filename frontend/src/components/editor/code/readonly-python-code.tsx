/* Copyright 2026 Marimo. All rights reserved. */

import { LightAsync as SyntaxHighlighter } from "react-syntax-highlighter";
import lightSyntaxHighlight from "react-syntax-highlighter/dist/esm/styles/hljs/stackoverflow-dark";
import darkSyntaxHighlight from "react-syntax-highlighter/dist/esm/styles/hljs/stackoverflow-light";
import { CopyIcon, EyeIcon, EyeOffIcon, PlusIcon } from "lucide-react";
import { memo, useState } from "react";
import { useAddCodeToNewCell } from "@/components/editor/cell/useAddCell";
import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/ui/tooltip";
import { toast } from "@/components/ui/use-toast";
import { cn } from "@/utils/cn";
import { copyToClipboard } from "@/utils/copy";
import { Events } from "@/utils/events";
import { useTheme } from "@/theme/useTheme";

const supportedLanguages = Object.freeze(["python", "sql", "markdown"] as const);
export type SupportedLanguage = (typeof supportedLanguages)[number];

export function isLanguageSupported(language: string): language is SupportedLanguage {
  return (supportedLanguages as readonly string[]).includes(language);
}

/**
 * A readonly code component that can be used to display code in a readonly state.
 *
 * @param props.className - The class name to apply to the component.
 * @param props.code - The code to display.
 * @param props.initiallyHideCode - Whether to initially hide the code.
 * @param props.showHideCode - Whether to show the hide code button.
 * @param props.insertNewCell - Whether to add a insert new cell button; when clicked will add a new cell next to the current cell or at the end of the file
 * @param props.language - The language of the code. Default is "python".
 */
export const ReadonlyCode = memo(
  (props: {
    className?: string;
    code: string;
    initiallyHideCode?: boolean;
    showHideCode?: boolean;
    showCopyCode?: boolean;
    insertNewCell?: boolean;
    language?: SupportedLanguage;
    minHeight?: string | number;
    maxHeight?: string | number;
  }) => {
    const { theme } = useTheme();
    const {
      code,
      className,
      initiallyHideCode,
      showHideCode = true,
      showCopyCode = true,
      insertNewCell,
      language = "python",
    } = props;
    const [hideCode, setHideCode] = useState(!!initiallyHideCode);

    return (
      <div
        className={cn(
          "relative hover-actions-parent w-full overflow-hidden pb-1",
          className,
        )}
      >
        <div className="absolute top-0 right-0 my-1 mx-2 z-10 hover-action flex gap-2">
          {showCopyCode && <CopyButton text={code} />}
          {insertNewCell && <InsertNewCell code={code} />}
          {showHideCode && (
            <ToggleCodeButton
              hidden={hideCode}
              onClick={() => setHideCode((prev) => !prev)}
            />
          )}
        </div>
        {!hideCode && (
          <SyntaxHighlighter
            language={language}
            style={
              theme === "light" ? lightSyntaxHighlight : darkSyntaxHighlight
            }
          >
            {code}
          </SyntaxHighlighter>
        )}
      </div>
    );
  },
);
ReadonlyCode.displayName = "ReadonlyCode";

const CopyButton = (props: { text: string }) => {
  const copy = Events.stopPropagation(async () => {
    await copyToClipboard(props.text);
    toast({ title: "Copied to clipboard" });
  });

  return (
    <Tooltip content="Copy code" usePortal={false}>
      <Button
        onClick={copy}
        size="xs"
        className="py-0"
        variant="secondary"
        aria-label="Copy code"
      >
        <CopyIcon size={14} strokeWidth={1.5} />
      </Button>
    </Tooltip>
  );
};

const ToggleCodeButton = (props: { hidden: boolean; onClick: () => void }) => {
  return (
    <Tooltip
      content={props.hidden ? "Show code" : "Hide code"}
      usePortal={false}
    >
      <Button
        onClick={props.onClick}
        aria-label={props.hidden ? "Show code" : "Hide code"}
        size="xs"
        className="py-0"
        variant="secondary"
      >
        {props.hidden ? (
          <EyeIcon size={14} strokeWidth={1.5} />
        ) : (
          <EyeOffIcon size={14} strokeWidth={1.5} />
        )}
      </Button>
    </Tooltip>
  );
};

const InsertNewCell = (props: { code: string }) => {
  const addCodeToNewCell = useAddCodeToNewCell();

  const handleClick = () => {
    addCodeToNewCell(props.code);
  };

  return (
    <Tooltip content="Add code to notebook" usePortal={false}>
      <Button
        onClick={handleClick}
        size="xs"
        className="py-0"
        variant="secondary"
        aria-label="Add code to notebook"
      >
        <PlusIcon size={14} strokeWidth={1.5} />
      </Button>
    </Tooltip>
  );
};
