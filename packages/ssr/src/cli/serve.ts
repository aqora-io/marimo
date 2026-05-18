import * as crypto from "node:crypto";
import { type Stats, createReadStream } from "node:fs";
import * as fs from "node:fs/promises";
import { type AddressInfo } from "node:net";
import * as path from "node:path";

import { Hono, type Handler as HonoHandler, type HonoRequest } from "hono";
import { getMimeType } from "hono/utils/mime";
import { trimStart } from "lodash-es";
import {
  serve as startServer,
  type ServerType as HonoServer,
} from "@hono/node-server";
import { print } from "@optique/run";
import { message } from "@optique/core";
import { isOutputType, OutputType, type ServeCommand } from "./parser";
import { fileURLToPath } from "node:url";
import { renderNotebook } from "..";
import type { NotebookSnapshot } from "../types";
import { readViteManifest, writeHtml } from "./build";
import { Readable } from "node:stream";

export async function serve(options: ServeCommand) {
  print(message`Initializing server...`);
  const app = await createApp(options);

  const serverOptions = {
    fetch: app.fetch,
    hostname: options.host,
    port: options.port,
  };
  const server = startServer(serverOptions, (addr) => {
    print(message`Listening on ${formatAddrInfo(addr)}...`);
  });

  try {
    await Promise.any([signal("SIGINT"), signal("SIGTERM")]);
  } finally {
    print(message`Shutting down...`);
    await closeServer(server);
  }
}

function signal<T extends NodeJS.Signals>(sig: T): Promise<T> {
  return new Promise((resolve) => process.once(sig, resolve));
}

function formatAddrInfo(addr: AddressInfo): string {
  switch (addr.family) {
    case "IPv6":
      return `[${addr.address}]:${addr.port}`;
    default:
      return `${addr.address}:${addr.port}`;
  }
}

function closeServer(server: HonoServer): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
      } else {
        resolve();
      }
    });
  });
}

async function createApp({
  basePath = "",
  origin,
  directory,
  timeout,
  accessToken,
}: ServeCommand): Promise<Hono> {
  directory = path.resolve(directory);
  const manifest = await readViteManifest();
  const assetDir = path.join(
    path.dirname(fileURLToPath(import.meta.url)),
    "assets",
  );

  const app = new Hono().basePath(basePath);

  /// CORS middleware
  app.use(async (c, next) => {
    c.header("Access-Control-Allow-Origin", "*");
    c.header("Access-Control-Allow-Methods", "HEAD, GET");
    if (c.req.method === "OPTIONS") {
      return c.body(null, 204);
    }
    await next();
  });

  /// Health endpoint
  app.get("/health", (c) => c.text("ok"));

  /// Static assets
  app.on(
    ["HEAD", "GET"],
    "/assets/*",
    serveStatic(assetDir, `${basePath}/assets`),
  );

  /// Notebook rendering
  app.on(["HEAD", "GET"], "/:filename", async (c) => {
    if (
      accessToken !== undefined &&
      c.req.query("access_token") !== accessToken
    ) {
      return c.notFound();
    }

    const outputType = c.req.query("t") ?? "page";
    if (!isOutputType(outputType)) {
      return c.text(`Invalid output type: ${outputType}`, 400);
    }

    const assetsBase = getAssetsBase(c.req, outputType, origin, basePath);

    const hideCode = c.req.query("hide-code") == "true";

    const filename = c.req.param("filename");
    const filepath = path.resolve(directory, filename);
    if (!filepath.startsWith(directory)) {
      return c.notFound();
    }

    const file = await statOpt(filepath);
    if (!file?.isFile()) {
      return c.notFound();
    }

    const sessionPath = path.join(
      directory,
      "__marimo__",
      "session",
      `${filename}.json`,
    );
    const notebookPath = path.join(
      directory,
      "__marimo__",
      "notebook",
      `${filename}.json`,
    );

    const [session, notebook] = await Promise.all([
      statOpt(sessionPath),
      statOpt(notebookPath),
    ]);
    if (!session?.isFile()) {
      return c.text("Notebook has no session cache", 500);
    }
    if (!notebook?.isFile()) {
      return c.text("Notebook was not properly exported", 500);
    }

    const etag = await hashFile(sessionPath);

    if (c.req.method === "HEAD") {
      c.header("ETag", etag);
      c.header("Date", session.mtime.toUTCString());
      return c.body(null, 204);
    }

    if (notModified(c.req, etag, session.mtime)) {
      c.header("ETag", etag);
      c.header("Date", session.mtime.toUTCString());
      return c.body(null, 304);
    }

    const snapshot: NotebookSnapshot = {
      session: JSON.parse(await fs.readFile(sessionPath, { encoding: "utf8" })),
      notebook: JSON.parse(
        await fs.readFile(notebookPath, { encoding: "utf8" }),
      ),
    };

    let html;
    try {
      const abort = timeout > 0 ? AbortSignal.timeout(timeout) : undefined;
      html = await renderNotebook(snapshot, { signal: abort, hideCode });
    } catch (error) {
      console.error("Cannot render notebook:", error);
      return c.text("Internal Server Error", 500);
    }

    c.header("ETag", etag);
    c.header("Date", session.mtime.toUTCString());
    return c.html(writeHtml(html, outputType, manifest, assetsBase), 200);
  });

  return app;
}

async function statOpt(filepath: string): Promise<Stats | null> {
  try {
    return await fs.stat(filepath);
  } catch {
    return null;
  }
}

async function hashFile(
  filepath: string,
  signal?: AbortSignal | undefined,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha1");
    createReadStream(filepath, { autoClose: true, signal }).pipe(hash);
    hash.once("readable", () => {
      const digest: Buffer = hash.read();
      resolve(digest.toString("hex"));
    });
    hash.once("error", (error) => {
      reject(error);
    });
  });
}

function notModified(request: HonoRequest, etag: string, mtime: Date): boolean {
  const ifNoneMatch = request.header("If-None-Match");
  if (ifNoneMatch !== undefined) {
    return ifNoneMatch === etag;
  }

  const ifModifiedSince = request.header("If-Modified-Since");
  if (ifModifiedSince !== undefined) {
    const since = new Date(ifModifiedSince);
    return mtime > since;
  }

  return false;
}

function serveStatic(directory: string, prefix: string): HonoHandler {
  return async (c) => {
    const filepath = path.resolve(
      directory,
      trimStart(c.req.path.slice(prefix.length), "/"),
    );
    if (!filepath.startsWith(directory)) {
      return c.notFound();
    }

    const file = await statOpt(filepath);
    if (!file || !file.isFile()) {
      return c.notFound();
    }

    const etag = await hashFile(filepath);
    c.header("ETag", etag);
    c.header("Date", file.mtime.toUTCString());
    c.header("Content-Type", getMimeType(filepath));

    if (notModified(c.req, etag, file.mtime)) {
      return c.body(null, 304);
    }

    if (c.req.method === "HEAD") {
      c.header("Content-Length", file.size.toString());
      return c.body(null, 200);
    } else {
      return c.body(
        Readable.toWeb(
          createReadStream(filepath, { autoClose: true }),
        ) as ReadableStream,
        200,
      );
    }
  };
}

function getAssetsBase(
  request: HonoRequest,
  outputType: OutputType,
  origin: string | undefined,
  basePath: string,
): string {
  if (outputType !== "dsd") return ".";
  return `${origin ?? getRequestOrigin(request)}${basePath}`;
}

function getRequestOrigin(request: HonoRequest): string {
  const host = request.header("Host");
  if (!host) {
    return ".";
  }
  return `http://${host}`;
}
