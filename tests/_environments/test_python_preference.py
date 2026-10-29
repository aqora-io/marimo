# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

import asyncio
import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from marimo._environments import environment
from marimo._environments.environment import Environment
from marimo._environments.uv import UvCommandError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


REPORT = json.dumps(
    {
        "sync": {
            "environment": {
                "path": "/env",
                "python": {"path": "/env/bin/python"},
            },
            "action": "check",
        }
    }
)


@pytest.fixture
def script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("UV_PYTHON_PREFERENCE", "system")
    monkeypatch.delenv("UV_PYTHON", raising=False)
    monkeypatch.setattr(environment, "ensure_supported_uv", lambda: None)
    path = tmp_path / "nb.py"
    path.write_text(
        '# /// script\n# requires-python = ">=3.13"\n# dependencies = []\n# ///\n'
    )
    return path


def _record_uv(
    monkeypatch: pytest.MonkeyPatch, *, failure: Exception | None = None
):
    calls: list[list[str]] = []

    def invoke(args: Sequence[str], **_kwargs: object):
        calls.append(list(args))
        if args[0] == "python":
            if failure is not None:
                raise failure
            return subprocess.CompletedProcess(
                args, 0, "/system/python3.13\n", ""
            )
        return subprocess.CompletedProcess(args, 0, REPORT, "")

    monkeypatch.setattr(environment, "uv", invoke)
    return calls


@pytest.mark.parametrize(
    "requirement", [">=3.13", ">=3.14", "==3.15.*", "==3.12.*"]
)
def test_sync_selects_an_installed_python_matching_the_manifest(
    script: Path, monkeypatch: pytest.MonkeyPatch, requirement: str
) -> None:
    original = script.read_text().replace(">=3.13", requirement)
    script.write_text(original)
    calls = _record_uv(monkeypatch)

    environment.sync(str(script))

    assert calls[0] == ["python", "find", "--system", requirement]
    assert calls[1][-2:] == ["--python", "/system/python3.13"]
    assert script.read_text() == original


def test_relative_script_metadata_uses_the_sync_working_directory(
    script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = script.parent / "other"
    other.mkdir()
    (other / script.name).write_text(
        script.read_text().replace(">=3.13", ">=3.14")
    )
    monkeypatch.chdir(script.parent)
    calls = _record_uv(monkeypatch)

    environment.sync(script.name, cwd=str(other))

    assert calls[0] == ["python", "find", "--system", ">=3.14"]


@pytest.mark.parametrize(
    "bypass", ["default", "explicit", "active", "environment"]
)
def test_sync_preserves_existing_interpreter_overrides(
    script: Path, monkeypatch: pytest.MonkeyPatch, bypass: str
) -> None:
    calls = _record_uv(monkeypatch)
    kwargs = {}
    if bypass == "default":
        monkeypatch.delenv("UV_PYTHON_PREFERENCE")
    elif bypass == "explicit":
        kwargs["python_override"] = "3.15"
    elif bypass == "active":
        kwargs["active_environment"] = Environment(
            "/live/python", "/live", "unchanged"
        )
    else:
        monkeypatch.setenv("UV_PYTHON", "3.15")

    environment.sync(str(script), **kwargs)

    assert len(calls) == 1
    assert calls[0][0] == "sync"
    if bypass == "explicit":
        assert calls[0][-2:] == ["--python", "3.15"]
    else:
        assert "--python" not in calls[0]


@pytest.mark.parametrize(
    "failure",
    [
        UvCommandError(["uv", "python", "find"], 2, "", "missing interpreter"),
        subprocess.TimeoutExpired(["uv", "python", "find"], 5),
    ],
)
def test_failed_discovery_leaves_resolution_to_sync(
    script: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    calls = _record_uv(monkeypatch, failure=failure)

    environment.sync(str(script))

    assert len(calls) == 2
    assert "--python" not in calls[1]


@pytest.mark.parametrize(
    "content", ["print(42)\n", "# /// script\n# requires-python = [\n# ///\n"]
)
def test_unreadable_metadata_is_left_to_uv(
    script: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    script.write_text(content)
    calls = _record_uv(monkeypatch)

    environment.sync(str(script))

    assert len(calls) == 1
    assert calls[0][0] == "sync"


@pytest.mark.asyncio
async def test_async_discovery_is_cancellable(
    script: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovering = asyncio.Event()
    calls: list[list[str]] = []

    async def invoke(args: Sequence[str], **_kwargs: object):
        calls.append(list(args))
        if args == ["--version"]:
            return subprocess.CompletedProcess(args, 0, "uv 0.13.6", "")
        if args[0] == "python":
            discovering.set()
            await asyncio.Event().wait()
        pytest.fail("Cancellation must not start a sync")

    monkeypatch.setattr(environment, "uv_async", invoke)
    task = asyncio.create_task(environment.sync_async(str(script)))
    try:
        await asyncio.wait_for(discovering.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert calls == [["--version"], ["python", "find", "--system", ">=3.13"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, TimeoutError()])
async def test_async_discovery_selects_or_falls_back(
    script: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception | None
) -> None:
    calls: list[list[str]] = []

    async def invoke(args: Sequence[str], **_kwargs: object):
        calls.append(list(args))
        if args == ["--version"]:
            return subprocess.CompletedProcess(args, 0, "uv 0.13.6", "")
        if args[0] == "python":
            if failure is not None:
                raise failure
            return subprocess.CompletedProcess(
                args, 0, "/system/python3.13\n", ""
            )
        return subprocess.CompletedProcess(args, 0, REPORT, "")

    monkeypatch.setattr(environment, "uv_async", invoke)

    await environment.sync_async(str(script))

    assert calls[1] == ["python", "find", "--system", ">=3.13"]
    if failure is None:
        assert calls[2][-2:] == ["--python", "/system/python3.13"]
    else:
        assert "--python" not in calls[2]
