import { createStore as createJotaiStore } from "jotai";

import {
  AppConfigSchema,
  defaultUserConfig,
  type AppConfig,
  type UserConfig,
} from "@marimo-team/frontend/unstable_internal/core/config/config-schema.ts";
import {
  appConfigAtom,
  userConfigAtom,
} from "@marimo-team/frontend/unstable_internal/core/config/config.ts";
import { notebookAtom } from "@marimo-team/frontend/unstable_internal/core/cells/cells.ts";
import { notebookStateFromSession } from "@marimo-team/frontend/unstable_internal/core/cells/session.ts";
import { requestClientAtom } from "@marimo-team/frontend/unstable_internal/core/network/requests.ts";
import { createStaticRequests } from "@marimo-team/frontend/unstable_internal/core/network/requests-static.ts";
import {
  initialModeAtom,
  viewStateAtom,
} from "@marimo-team/frontend/unstable_internal/core/mode.ts";
import {
  DEFAULT_RUNTIME_CONFIG,
  runtimeConfigAtom,
} from "@marimo-team/frontend/unstable_internal/core/runtime/config.ts";
import { showCodeInRunModeAtom } from "@marimo-team/frontend/unstable_internal/core/meta/state.ts";
import type { NotebookSnapshot } from "./types";

export function createStore(
  snapshot: NotebookSnapshot,
  appConfig: AppConfig,
  userConfig: UserConfig,
  hideCode: boolean,
) {
  const notebookState = notebookStateFromSession(
    snapshot.session,
    snapshot.notebook,
  );
  if (!notebookState) {
    throw new Error("Notebook appears empty");
  }

  const store = createJotaiStore();

  store.set(notebookAtom, notebookState);
  store.set(appConfigAtom, appConfig);
  store.set(userConfigAtom, userConfig);
  store.set(runtimeConfigAtom, {
    ...DEFAULT_RUNTIME_CONFIG,
    lazy: false,
  });
  store.set(requestClientAtom, createStaticRequests());
  store.set(initialModeAtom, "read");
  store.set(viewStateAtom, { mode: "read", cellAnchor: null });
  store.set(showCodeInRunModeAtom, !hideCode);

  return store;
}

export function defaultAppConfig(): AppConfig {
  return AppConfigSchema.decode({});
}

export { defaultUserConfig };
