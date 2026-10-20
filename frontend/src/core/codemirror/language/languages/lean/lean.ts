/* Copyright 2026 Marimo. All rights reserved. */

import { StreamLanguage } from "@codemirror/language";
import type { Extension } from "@codemirror/state";
import { type LeanMetadata, LeanParser } from "@marimo-team/smart-cells";
import type { LanguageAdapter } from "../../types";
import { leanMode } from "./lean-mode";
import { leanUnicodeInput } from "./unicode-input";

export type LeanLanguageAdapterMetadata = LeanMetadata;

/**
 * Language adapter for Lean, backed by `lean()` from marimo-lean.
 */
export class LeanLanguageAdapter implements LanguageAdapter<LeanLanguageAdapterMetadata> {
  private parser = new LeanParser();

  public readonly type = "lean";
  public readonly defaultCode = this.parser.defaultCode;
  public readonly defaultMetadata: LeanLanguageAdapterMetadata =
    this.parser.defaultMetadata;

  public transformIn(
    pythonCode: string,
  ): [string, number, LeanLanguageAdapterMetadata] {
    const result = this.parser.transformIn(pythonCode);
    return [result.code, result.offset, result.metadata];
  }

  public transformOut(
    code: string,
    metadata: LeanLanguageAdapterMetadata,
  ): [string, number] {
    const result = this.parser.transformOut(code, metadata);
    return [result.code, result.offset];
  }

  public isSupported(pythonCode: string): boolean {
    return this.parser.isSupported(pythonCode);
  }

  public getExtension(): Extension[] {
    return [StreamLanguage.define(leanMode), leanUnicodeInput()];
  }
}
