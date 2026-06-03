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
    for (const [asset, _extension] of iterManifestAssets(manifest)) {
      const assetPath = join(distDir, asset.file);
      const targetAssetPath = join(outputDir, asset.file);
      await mkdir(dirname(targetAssetPath), { recursive: true });
      await copyFile(assetPath, targetAssetPath);

      print(message`Created ${targetAssetPath}`);
    }
  }
}

export type ViteManifest = Record<string, ViteManifestEntry>;
export interface ViteManifestEntry {
  file: string;
  isEntry?: boolean | undefined;
  src: string;
  name?: string | undefined;
  names?: string[] | undefined;
}

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
    ${writeHtmlLinks(bundle, assetsBase)}
    <style>
      body {
        margin: 0;
        padding: 0;
        border: 0;
      }
      #root {
        padding-top: 28.8px;
      }
    </style>
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
  return `${writeDsdLinks(bundle, assetsBase, false)}
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

  ${writeDsdLinks(bundle, assetsBase, true)}
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

function writeHtmlLinks(
  manifest: ViteManifest,
  assetsBase: string | undefined,
): string {
  return Array.from(iterManifestAssets(manifest))
    .map(([asset, extension]) => writeAssetLink(assetsBase, asset, extension))
    .join("\n");
}

function writeDsdLinks(
  manifest: ViteManifest,
  assetsBase: string | undefined,
  fonts: boolean,
): string {
  return Array.from(iterManifestAssets(manifest))
    .filter(([asset, extension]) => isAssetFont(asset, extension) === fonts)
    .map(([asset, extension]) => writeAssetLink(assetsBase, asset, extension))
    .join("\n");
}

function joinUrl(base: string | undefined, path: string): string {
  if (!base) return path;
  return trimEnd(base, "/") + "/" + trimStart(path, "/");
}

const ASSET_EXTENSIONS = Object.freeze([
  ".css",
  ".ttf",
  ".woff",
  ".woff2",
  ".png",
  ".svg",
] as const);
type AssetExtension = (typeof ASSET_EXTENSIONS)[number];

function isAssetFont(
  asset: ViteManifestEntry,
  extension: AssetExtension,
): boolean {
  switch (extension) {
    case ".ttf":
    case ".woff":
    case ".woff2":
      return true;
    case ".css":
      return asset.name === "fonts";
    default:
      return false;
  }
}

function* iterManifestAssets(
  manifest: ViteManifest,
): Generator<[ViteManifestEntry, AssetExtension]> {
  for (const entry of Object.values(manifest)) {
    if (entry.file.endsWith(".js")) continue;

    const extension = ASSET_EXTENSIONS.find((ext) => entry.file.endsWith(ext));
    if (!extension) {
      console.error(`Asset "${entry.file}": unrecognized extension`);
      continue;
    }
    yield [entry, extension];
  }
}

function writeAssetLink(
  base: string | undefined,
  { file }: ViteManifestEntry,
  extension: AssetExtension,
): string {
  switch (extension) {
    case ".css":
      return `<link rel="stylesheet" crossorigin href="${joinUrl(base, file)}" />`;

    case ".ttf":
    case ".woff":
    case ".woff2":
      return `<link rel="preload" as="font" crossorigin href="${joinUrl(base, file)}" />`;

    case ".png":
    case ".svg":
      return `<link rel="preload" as="image" crossorigin href="${joinUrl(base, file)}" />`;
  }
}
