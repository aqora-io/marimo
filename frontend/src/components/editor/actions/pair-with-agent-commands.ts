/* Copyright 2026 Marimo. All rights reserved. */

import { shellQuote } from "@/utils/shell";

export type AgentTab = "claude" | "codex" | "opencode" | "prompt";

export const AGENT_TABS = ["claude", "codex", "opencode", "prompt"] as const;

export const AGENT_LABELS: Record<AgentTab, string> = {
  claude: "Claude",
  codex: "Codex",
  opencode: "OpenCode",
  prompt: "Prompt",
};

/**
 * The `aqora pair` flag selecting each tab's agent. "prompt" prints the prompt
 * instead, for any other agent that has the marimo-pair skill.
 */
const AGENT_FLAGS: Record<AgentTab, string> = {
  claude: "--claude",
  codex: "--codex",
  opencode: "--opencode",
  prompt: "--prompt-only",
};

export const CLI_INSTALL = "uv tool install aqora";

export const SKILL_INSTALL = "npx skills add marimo-team/marimo-pair";

/**
 * The aqora runner id: the last path segment of the notebook server URL,
 * e.g. `https://kubimo.org/runner/<id>/`.
 */
export function getRunnerIdFromURL(href: string): string | undefined {
  return new URL(href).pathname
    .split("/")
    .findLast((segment) => segment !== "");
}

/** Identifies the specific running notebook to pair on. */
export interface ConnectionInfo {
  /** The aqora workspace runner serving the notebook. */
  runnerId: string;
  /** The marimo session of this browser tab. */
  sessionId: string;
}

/** The terminal command that pairs the tab's agent with this notebook. */
export function getTerminalCommand(
  tab: AgentTab,
  { runnerId, sessionId }: ConnectionInfo,
): string {
  return `aqora pair ${shellQuote(runnerId)} --session ${shellQuote(sessionId)} ${AGENT_FLAGS[tab]}`;
}
