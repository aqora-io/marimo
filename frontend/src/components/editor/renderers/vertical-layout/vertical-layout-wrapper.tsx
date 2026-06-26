/* Copyright 2026 Marimo. All rights reserved. */
import type { PropsWithChildren } from "react";
import type { AppConfig } from "@/core/config/config-schema";
import { cn } from "@/utils/cn";
import { useResponsiveEmbedRef } from "../../responsive-embed";
import type { AppMode } from "@/core/mode";

interface Props {
  mode: AppMode;
  className?: string;
  innerClassName?: string;
  appConfig: AppConfig;
  invisible?: boolean;
}

export const VerticalLayoutWrapper: React.FC<PropsWithChildren<Props>> = ({
  mode,
  invisible,
  appConfig,
  className,
  children,
  innerClassName,
}) => {
  const ref = useResponsiveEmbedRef<HTMLDivElement>();
  const width = getAppWidth(appConfig, mode);
  return (
    <div
      className={cn(
        mode === "read"
          ? "lg:px-24"
          : "px-1 sm:px-16 md:px-20 xl:px-24 print:px-0 print:pb-0",
        // // Large mobile bottom padding due to mobile browser navigation bar
        // "pb-24 sm:pb-12",
        className,
      )}
      ref={ref}
    >
      <div
        className={cn(
          "m-auto",
          // // This padding needs to be the same from above to be correctly applied
          // "pb-24 sm:pb-12",
          width === "compact" &&
          "max-w-(--content-width) sm:min-w-[400px]",
          width === "medium" &&
          "max-w-(--content-width-medium) sm:min-w-[400px]",
          width === "columns" && "w-fit",
          width === "full" && "max-w-full",
          // Hide the cells for a fake loading effect, to avoid flickering
          invisible && "invisible",
          innerClassName,
        )}
      >
        {children}
      </div>
    </div>
  );
};

function getAppWidth(config: AppConfig, mode: AppMode) {
  return mode !== "read" ? config.width : "full";
}
