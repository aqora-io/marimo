#!/usr/bin/env node
import { run } from "@optique/run";
import { parser } from "./parser";
import { build } from "./build";
import { serve } from "./serve";

void main().then(
  () => {
    process.exit(0);
  },
  (error) => {
    console.error("Fatal error", error);
    process.exit(1);
  },
);

async function main() {
  const options = run(parser(), {
    help: "option",
    colors: false,
    showDefault: true,
    showChoices: true,
    errorExitCode: 2,
  });

  switch (options.command_) {
    case "build":
      return await build(options);

    case "serve":
      return await serve(__VERSION__ || "dev", options);
  }
}
