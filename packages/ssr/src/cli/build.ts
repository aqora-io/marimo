import { readFile, writeFile, copyFile, mkdir } from "node:fs/promises";
import { dirname, basename, join } from "node:path";

import { message } from "@optique/core";
import { print } from "@optique/run";
import type { NotebookSnapshot } from "../types";
import { renderNotebook } from "..";
import { OutputType, type BuildCommand } from "./parser";
import { fileURLToPath } from "node:url";
import { trimEnd, trimStart } from "lodash-es";

export async function build(options: BuildCommand) {
  const sessionFile = await readFile(
    join(
      dirname(options.name),
      "__marimo__",
      "session",
      `${basename(options.name)}.json`,
    ),
    { encoding: "utf8" },
  );
  const notebookFile = await readFile(
    join(
      dirname(options.name),
      "__marimo__",
      "notebook",
      `${basename(options.name)}.json`,
    ),
    { encoding: "utf8" },
  );

  const manifest = await readViteManifest();

  const snapshot: NotebookSnapshot = {
    session: JSON.parse(sessionFile),
    notebook: JSON.parse(notebookFile),
  };

  const html = await renderNotebook(snapshot, { hideCode: options.hideCode });
  await writeFile(
    options.output,
    writeHtml(html, options.outputType, manifest, options.assetsBase),
  );

  print(message`Created ${options.output}`);

  if (options.copyAssets) {
    const outputDir = dirname(options.output);
    const distDir = dirname(fileURLToPath(import.meta.url));
    for (const style of iterManifestAssets(manifest)) {
      const assetPath = join(distDir, style);
      const targetAssetPath = join(outputDir, style);
      await mkdir(dirname(targetAssetPath), { recursive: true });
      await copyFile(assetPath, targetAssetPath);

      print(message`Created ${targetAssetPath}`);
    }
  }
}

export type ViteManifest = Record<string, { file: string }>;

export async function readViteManifest(): Promise<ViteManifest> {
  return JSON.parse(
    await readFile(
      join(dirname(fileURLToPath(import.meta.url)), ".vite", "manifest.json"),
      {
        encoding: "utf8",
      },
    ),
  );
}

export function writeHtml(
  html: string,
  outputType: OutputType,
  bundle: ViteManifest,
  assetsBase: string | undefined,
): string {
  switch (outputType) {
    case "raw":
      return html;
    case "page":
      return writeHtmlPage(html, bundle, assetsBase);
    case "dsd":
      return writeHtmlDsd(html, bundle, assetsBase);
    case "dsd-page":
      return writeHtmlDsdPage(html, bundle, assetsBase);
  }
}

function writeHtmlPage(
  html: string,
  bundle: ViteManifest,
  assetsBase: string | undefined,
): string {
  return `<!doctype html>
<html>
  <head>
    ${writeHtmlLinkStylesheets(bundle, assetsBase)}
  </head>
  <body>
    <div id="root" class="marimo light">${html}</div>
  </body>
</html>`;
}

function writeHtmlDsd(
  html: string,
  bundle: ViteManifest,
  assetsBase: string | undefined,
): string {
  return `${writeHtmlLinkStylesheets(bundle, assetsBase)}
<div id="root" class="marimo light" style="min-height:auto;position:relative;--tw-border-style:solid">${html}</div>`;
}

function writeHtmlDsdPage(
  html: string,
  bundle: ViteManifest,
  assetsBase: string | undefined,
): string {
  return `<!doctype html>
<head>
  <style>
    * {
      margin: 0;
      padding: 0;
      border: 0;
    }
    body {
      display: flex;
      justify-content: center;
    }
    #marimo {
      width: 67vw;
      border: 1px solid #eee;
    }
  </style>
</head>
<html>
  <body>
    <div id="marimo">
      <template shadowrootmode="open">
        ${writeHtmlDsd(html, bundle, assetsBase)}
      </template>
    </div>
  </body>
</html>`;
}

function writeHtmlLinkStylesheets(
  manifest: ViteManifest,
  assetsBase: string | undefined,
): string {
  return Array.from(iterManifestAssets(manifest))
    .map(
      (style) =>
        `<link rel="stylesheet" href="${joinUrl(assetsBase, style)}" />`,
    )
    .join("");
}

function joinUrl(base: string | undefined, path: string): string {
  if (!base) return path;
  return trimEnd(base, "/") + "/" + trimStart(path, "/");
}

function* iterManifestAssets(manifest: ViteManifest) {
  const extensions = [".css", ".ttf", ".woff", ".woff2", ".png", ".svg"];
  for (const { file } of Object.values(manifest)) {
    if (extensions.some((ext) => file.endsWith(ext))) {
      yield file;
    }
  }
}
