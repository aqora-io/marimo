/* Copyright 2026 Marimo. All rights reserved. */
import { Handler } from "vega-tooltip";

let handler: Handler | undefined;

const call: Handler["call"] = (...args) => {
  if (!handler) {
    handler = new Handler();
  }
  return handler.call(...args);
};

// Create a tooltip handler that supports HTML content (including images)
export const tooltipHandler = Object.freeze({ call });
