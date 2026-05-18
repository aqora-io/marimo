import { JSDOM } from "jsdom";

export interface VirtualBrowserEnvironmentInit {
  html?: string | undefined;
  signal?: AbortSignal | undefined;
}

export class VirtualBrowserEnvironment {
  private readonly prev: Partial<typeof globalThis>;
  readonly jsdom: JSDOM;

  constructor({ html, signal }: VirtualBrowserEnvironmentInit = {}) {
    this.jsdom = new JSDOM(html);
    this.prev = poluteGlobals({
      window: this.jsdom.window,
      document: this.jsdom.window.document,
      InputEvent: class InputEvent {},
    });
    signal?.addEventListener("abort", this.dispose);
  }

  dispose = () => {
    this.jsdom.window.close();
    poluteGlobals(this.prev);
  };

  [Symbol.dispose] = this.dispose;
}

function poluteGlobals(globals: object): object {
  const prev: Partial<typeof globalThis> = {};
  for (const attr in globals) {
    // @ts-ignore
    prev[attr] = globalThis[attr];
    // @ts-ignore
    globalThis[attr] = globals[attr];
  }
  return prev;
}
