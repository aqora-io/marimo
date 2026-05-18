import * as api from "@marimo-team/marimo-api";

export interface NotebookSnapshot {
  /// Snapshot of the notebook outputs. Can be exported with the marimo CLI:
  ///
  /// $ marimo export session notebook.py
  session: api.Session["NotebookSessionV1"];

  /// Snapshot of the notebook code Can be exported with the marimo CLI:
  ///
  /// $ marimo export json-notebook notebook.py
  notebook: api.Notebook["NotebookV1"];
}
