import { readFile } from "node:fs/promises";

import { type ConfigEnv, type UserConfig } from "vite";
import react from "@vitejs/plugin-react";

export default async ({ mode }: ConfigEnv): Promise<UserConfig> => {
  process.env.NODE_ENV = mode;

  const pkg = JSON.parse(
    (await readFile("./package.json")).toString("utf8"),
  ) as PackageJson;
  const external = Object.keys(pkg.dependencies).concat(
    Object.keys(pkg.peerDependencies ?? {}),
  );

  return {
    build: {
      sourcemap: mode === "development",
      minify: mode !== "development" && "oxc",
      ssr: true,
      ssrEmitAssets: true,
      manifest: true,
      rollupOptions: {
        input: {
          index: "src/index.tsx",
          main: "src/cli/main.ts",
          fonts: "../../frontend/src/css/app/fonts.css",
        },
        output: {
          entryFileNames: `[name].${mode}.js`,
        },
      },
    },
    define: {
      "process.env.NODE_ENV": JSON.stringify(mode),
      __VERSION__: JSON.stringify(
        mode === "development" ? Date.now().toString() : pkg.version,
      ),
    },
    experimental: {
      enableNativePlugin: true,
    },
    resolve: {
      tsconfigPaths: true,
    },
    ssr: {
      target: "node",
      external,
      noExternal: true,
    },
    plugins: [
      react({
        babel: {
          presets: ["@babel/preset-typescript"],
          plugins: [
            ["@babel/plugin-proposal-decorators", { legacy: true }],
            ["babel-plugin-react-compiler", ReactCompilerConfig],
          ],
        },
      }),
    ],
  };
};

interface PackageJson {
  version: string;
  dependencies: Record<string, unknown>;
  peerDependencies?: Record<string, unknown> | undefined;
}

const ReactCompilerConfig = {
  target: "19",
};
