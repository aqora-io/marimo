import {
  choice,
  option,
  withDefault,
  object,
  optional,
  string,
  argument,
  message,
  or,
  command,
  constant,
  type InferValue,
  merge,
  integer,
} from "@optique/core";
import { path } from "@optique/run";

export const OutputType = Object.freeze([
  "raw",
  "page",
  "dsd",
  "dsd-page",
  "json",
] as const);
export type OutputType = (typeof OutputType)[number];

export function isOutputType(value: string): value is OutputType {
  return (OutputType as readonly string[]).includes(value);
}

const buildParser = () =>
  object({
    name: argument(path({ metavar: "NAME", mustExist: true, type: "file" }), {
      description: message`Notebook to statically render`,
    }),
    output: option(
      "-o",
      "--output",
      path({ metavar: "OUTPUT", allowCreate: true }),
      { description: message`Path where the notebook is rendered` },
    ),
    hideCode: option("--hide-code", {
      description: message`Hide code in the render`,
    }),
    copyAssets: option("--copy-assets", {
      description: message`Copy assets in the same directory as the output`,
    }),
    assetsBase: optional(
      option("--assets-base", string({ metavar: "ASSETS_BASE" }), {
        description: message`Prefix to write when writing hrefs`,
      }),
    ),
    outputType: withDefault(
      option(
        "-t",
        "--output-type",
        choice(OutputType, { metavar: "OUTPUT_TYPE" }),
        { description: message`Type of output` },
      ),
      "raw",
    ),
  });

export type BuildCommand = InferValue<ReturnType<typeof buildParser>>;

const serveParser = () =>
  object({
    host: optional(
      option("--host", string({ metavar: "ADDRESS" }), {
        description: message`Address to listen on`,
      }),
    ),
    port: optional(
      option("--port", integer({ metavar: "PORT", min: 0, max: 65535 }), {
        description: message`Port to listen on`,
      }),
    ),
    timeout: withDefault(
      option("--timeout", integer({ metavar: "MILLIS", min: 0 }), {
        description: message`Period of time before a request is cancelled automatically`,
      }),
      30_000,
    ),
    origin: optional(
      option("--origin", string({ metavar: "URL" }), {
        description: message`URL where this server will be reachable at. Useful only for DSD output`,
      }),
    ),
    basePath: optional(
      option("--base-path", string({ metavar: "PATH" }), {
        description: message`Base request path to serve notebooks under`,
      }),
    ),
    directory: argument(
      path({ metavar: "DIRECTORY", mustExist: true, type: "directory" }),
      {
        description: message`Directory to serve notebooks from`,
      },
    ),
    accessToken: optional(
      option("--token", string({ metavar: "TOKEN" }), {
        description: message`Requests without this token in their querystring will be denied`,
      }),
    ),
  });

export type ServeCommand = InferValue<ReturnType<typeof serveParser>>;

export const parser = () =>
  or(
    command(
      "build",
      merge(buildParser(), object({ command_: constant("build") })),
      {
        description: message`Render a notebook`,
      },
    ),
    command(
      "serve",
      merge(serveParser(), object({ command_: constant("serve") })),
      {
        description: message`Start an HTTP server that renders notebooks`,
      },
    ),
  );
