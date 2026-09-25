# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

import pytest

from marimo._config.settings import GLOBAL_SETTINGS
from marimo._server.start import _configure_sandbox
from marimo._session.model import SessionMode


@pytest.fixture
def _clean_sandbox_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARIMO_SANDBOX_MODE", raising=False)
    monkeypatch.delenv("MARIMO_SANDBOX_BACKEND", raising=False)
    monkeypatch.setattr(GLOBAL_SETTINGS, "SANDBOX_MODE", None)
    monkeypatch.setattr(GLOBAL_SETTINGS, "SANDBOX_BACKEND", None)


@pytest.mark.usefixtures("_clean_sandbox_settings")
def test_edit_sandbox_enables_multi_mode() -> None:
    _configure_sandbox("uv", SessionMode.EDIT, {})
    assert GLOBAL_SETTINGS.SANDBOX_MODE == "multi"
    assert GLOBAL_SETTINGS.SANDBOX_BACKEND == "uv"


@pytest.mark.usefixtures("_clean_sandbox_settings")
def test_edit_sandbox_with_configured_venv_keeps_project_mode() -> None:
    """Notebooks in a configured venv are not script sandboxes; leaving the
    mode unset keeps the packages panel on the project dependency tree."""
    _configure_sandbox("uv", SessionMode.EDIT, {"path": "/opt/venv"})
    assert GLOBAL_SETTINGS.SANDBOX_MODE is None
    assert GLOBAL_SETTINGS.SANDBOX_BACKEND == "uv"


@pytest.mark.usefixtures("_clean_sandbox_settings")
def test_run_sandbox_never_enables_multi_mode() -> None:
    _configure_sandbox("uv", SessionMode.RUN, {})
    assert GLOBAL_SETTINGS.SANDBOX_MODE is None
