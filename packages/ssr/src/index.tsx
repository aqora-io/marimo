import { Provider, useAtomValue } from "jotai";
import ReactDOM, { type RenderToReadableStreamOptions } from "react-dom/server";

import {
  type AppConfig,
  type UserConfig,
} from "@marimo-team/frontend/unstable_internal/core/config/config-schema.ts";
import { appConfigAtom } from "@marimo-team/frontend/unstable_internal/core/config/config.ts";
import { VerticalLayoutRenderer } from "@marimo-team/frontend/unstable_internal/components/editor/renderers/vertical-layout/vertical-layout.tsx";
import { ErrorBoundary } from "@marimo-team/frontend/unstable_internal/components/editor/boundary/ErrorBoundary.tsx";
import { slotsController } from "@marimo-team/frontend/unstable_internal/core/slots/slots.ts";
import { Provider as SlotzProvider } from "@marimo-team/react-slotz";
import { TooltipProvider } from "@marimo-team/frontend/unstable_internal/components/ui/tooltip.tsx";
import { ThemeProvider } from "@marimo-team/frontend/unstable_internal/theme/ThemeProvider.tsx";
import { ModalProvider } from "@marimo-team/frontend/unstable_internal/components/modal/ImperativeModal.tsx";
import { LocaleProvider } from "@marimo-team/frontend/unstable_internal/core/i18n/locale-provider.tsx";
import {
  flattenTopLevelNotebookCells,
  useNotebook,
} from "@marimo-team/frontend/unstable_internal/core/cells/cells.ts";

import { prerenderNotebook } from "./cells";
import { VirtualBrowserEnvironment } from "./utils";
import type { NotebookSnapshot } from "./types";
import { createStore, defaultAppConfig, defaultUserConfig } from "./store";

import "@marimo-team/frontend/unstable_internal/css/index.css";
import "@marimo-team/frontend/unstable_internal/css/app/App.css";

export interface RenderNotebookProps extends RenderToReadableStreamOptions {
  appConfig?: AppConfig | undefined;
  userConfig?: UserConfig | undefined;
  hideCode?: boolean | undefined;
}

/// Render a notebook to a static html string.
export async function renderNotebook(
  snapshot: NotebookSnapshot,
  {
    appConfig = defaultAppConfig(),
    userConfig = defaultUserConfig(),
    hideCode = false,
    ...props
  }: RenderNotebookProps = {},
): Promise<string> {
  {
    using _env = new VirtualBrowserEnvironment();
    snapshot.session = prerenderNotebook(snapshot.session);
  }

  const store = createStore(snapshot, appConfig, userConfig, hideCode);

  const app = (
    <Provider store={store}>
      <ThemeProvider>
        <ErrorBoundary>
          <TooltipProvider>
            <LocaleProvider>
              <ModalProvider>
                <SlotzProvider controller={slotsController}>
                  <NotebookCells />
                </SlotzProvider>
              </ModalProvider>
            </LocaleProvider>
          </TooltipProvider>
        </ErrorBoundary>
      </ThemeProvider>
    </Provider>
  );

  let html = "";
  {
    using _env = new VirtualBrowserEnvironment({ signal: props.signal });
    const stream = await ReactDOM.renderToReadableStream(app, props);
    await stream.allReady;
    const decoder = new TextDecoder();
    for await (const chunk of stream) {
      html += decoder.decode(chunk);
    }
  }

  return html;
}

function NotebookCells() {
  const appConfig = useAtomValue(appConfigAtom);
  const notebook = useNotebook();
  const cells = flattenTopLevelNotebookCells(notebook);
  return (
    <VerticalLayoutRenderer
      appConfig={appConfig}
      cells={cells}
      layout={null}
      setLayout={console.error}
      mode="read"
    />
  );
}
