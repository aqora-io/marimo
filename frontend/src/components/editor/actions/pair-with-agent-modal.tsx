/* Copyright 2026 Marimo. All rights reserved. */

import { CheckIcon, CopyIcon } from "lucide-react";
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { copyToClipboard } from "@/utils/copy";
import { Events } from "@/utils/events";
import { Tooltip } from "@/components/ui/tooltip";
import { useRuntimeManager } from "@/core/runtime/config";
import { getSessionId } from "@/core/kernel/session";
import {
  AGENT_LABELS,
  AGENT_TABS,
  type AgentTab,
  CLI_INSTALL,
  getRunnerIdFromURL,
  getTerminalCommand,
  SKILL_INSTALL,
} from "./pair-with-agent-commands";

export const PairWithAgentModal: React.FC<{
  onClose: () => void;
}> = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState<AgentTab>("claude");
  const runtimeManager = useRuntimeManager();
  const runnerId = getRunnerIdFromURL(runtimeManager.httpURL.toString());
  const sessionId = getSessionId();

  return (
    <DialogContent className="sm:max-w-2xl">
      <DialogHeader>
        <DialogTitle>Pair with an agent</DialogTitle>
        <DialogDescription>
          Use an AI coding agent to pair-program on this notebook.{" "}
          <a
            href="https://links.marimo.app/marimo-pair"
            target="_blank"
            rel="noopener noreferrer"
            className="underline"
          >
            Learn more
            <span className="sr-only"> about pairing marimo with an agent</span>
          </a>
          .
        </DialogDescription>
      </DialogHeader>

      <div className="flex flex-col gap-4 py-2">
        {runnerId === undefined ? (
          <p className="text-sm text-muted-foreground">
            This notebook is not served by an aqora workspace runner.
          </p>
        ) : (
          <Tabs
            value={activeTab}
            onValueChange={(v) => setActiveTab(v as AgentTab)}
          >
            <TabsList className="w-full">
              {AGENT_TABS.map((tab) => (
                <TabsTrigger key={tab} value={tab} className="flex-1">
                  {AGENT_LABELS[tab]}
                </TabsTrigger>
              ))}
            </TabsList>

            {AGENT_TABS.map((tab) => (
              <TabsContent
                key={tab}
                value={tab}
                className="mt-4 flex flex-col gap-4"
              >
                <Step
                  index={1}
                  title="Install the aqora CLI and the marimo-pair skill"
                  hint="Run once per machine."
                >
                  <CommandBlock command={CLI_INSTALL} />
                  <CommandBlock command={SKILL_INSTALL} />
                </Step>
                <Step
                  index={2}
                  title={
                    tab === "prompt"
                      ? "Run in your terminal, then paste the printed prompt into your agent"
                      : "Run in your terminal"
                  }
                  hint={
                    tab === "prompt"
                      ? "For any agent that has the marimo-pair skill."
                      : undefined
                  }
                >
                  <CommandBlock
                    command={getTerminalCommand(tab, { runnerId, sessionId })}
                  />
                </Step>
              </TabsContent>
            ))}
          </Tabs>
        )}
      </div>

      <DialogFooter>
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </DialogFooter>
    </DialogContent>
  );
};

const Step: React.FC<{
  index: number;
  title: string;
  hint?: string;
  children: React.ReactNode;
}> = ({ index, title, hint, children }) => (
  <div className="flex flex-col gap-2">
    <div className="flex items-baseline gap-2">
      <span className="text-sm font-medium">
        {index}. {title}
      </span>
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </div>
    {children}
  </div>
);

const CommandBlock: React.FC<{ command: string }> = ({ command }) => {
  const [copied, setCopied] = useState(false);

  const copy = Events.stopPropagation(async (e) => {
    e.preventDefault();
    await copyToClipboard(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  });

  return (
    <div className="flex items-center gap-2 rounded-md bg-muted px-3 py-2 font-mono text-xs">
      <code className="flex-1 select-all wrap-break-word">{command}</code>
      <Tooltip content="Copied!" open={copied}>
        <Button onClick={copy} size="xs" variant="ghost">
          {copied ? (
            <CheckIcon size={14} strokeWidth={1.5} />
          ) : (
            <CopyIcon size={14} strokeWidth={1.5} />
          )}
        </Button>
      </Tooltip>
    </div>
  );
};
