# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from marimo._session.app_host.pool import AppHostPool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.requires("zmq")
class TestAppHostPoolRespawnDead:
    def test_respawns_dead_host(self) -> None:
        """A dead host is shut down before its replacement is returned."""
        pool = AppHostPool(sandbox=False)

        dead_host = MagicMock()
        dead_host.is_alive.return_value = False

        new_host = MagicMock()
        new_host.is_alive.return_value = True

        abs_path = os.path.abspath("/tmp/test_respawn.py")
        pool._workers[abs_path] = dead_host

        with patch(
            "marimo._session.app_host.pool.AppHost",
            return_value=new_host,
        ):
            result = pool.get_or_create("/tmp/test_respawn.py")

        dead_host.shutdown.assert_called_once()
        assert result is new_host
        assert pool._workers[abs_path] is new_host


def test_startup_does_not_block_other_notebooks_or_duplicate_hosts() -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    entered = threading.Event()
    release = threading.Event()
    slow = MagicMock()
    fast = MagicMock()
    pool = AppHostPool()

    def start():
        entered.set()
        assert release.wait(timeout=10)

    slow.start.side_effect = start
    with patch(
        "marimo._session.app_host.pool.AppHost", side_effect=[slow, fast]
    ) as create:
        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(pool.get_or_create, "slow.py")
            try:
                assert entered.wait(timeout=5)
                same = executor.submit(pool.get_or_create, "slow.py")
                other = executor.submit(pool.get_or_create, "fast.py")
                assert other.result(timeout=5) is fast
                with pytest.raises(TimeoutError):
                    same.result(timeout=0.1)
            finally:
                release.set()
            assert first.result(timeout=5) is slow
            assert same.result(timeout=5) is slow
        assert create.call_count == 2
    pool.shutdown()


def test_shutdown_during_startup_does_not_publish_host() -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from marimo._session.managers.ipc import KernelStartupError

    entered = threading.Event()
    release = threading.Event()
    host = MagicMock()
    pool = AppHostPool()

    def start():
        entered.set()
        assert release.wait(timeout=10)

    host.start.side_effect = start
    with patch("marimo._session.app_host.pool.AppHost", return_value=host):
        with ThreadPoolExecutor() as executor:
            pending = executor.submit(pool.get_or_create, "nb.py")
            try:
                assert entered.wait(timeout=5)
                pool.shutdown()
                with pytest.raises(KernelStartupError, match="shut down"):
                    pool.get_or_create("nb.py")
            finally:
                release.set()
            with pytest.raises(KernelStartupError, match="shut down"):
                pending.result(timeout=5)
    host.shutdown.assert_called_once()
    assert not pool._workers


def test_failed_startup_is_cleaned_up_and_can_be_retried() -> None:
    failed = MagicMock()
    failed.start.side_effect = RuntimeError("startup failed")
    replacement = MagicMock()
    pool = AppHostPool()
    with patch(
        "marimo._session.app_host.pool.AppHost",
        side_effect=[failed, replacement],
    ):
        with pytest.raises(RuntimeError, match="startup failed"):
            pool.get_or_create("nb.py")
        assert pool.get_or_create("nb.py") is replacement
    failed.shutdown.assert_called_once()
    pool.shutdown()


def test_old_host_callback_does_not_remove_replacement() -> None:
    callbacks = []

    def create(_path, *, plan, on_empty):
        del plan
        host = MagicMock()
        callbacks.append(on_empty)
        return host

    pool = AppHostPool()
    with patch("marimo._session.app_host.pool.AppHost", side_effect=create):
        first = pool.get_or_create("nb.py")
        first.is_alive.return_value = False
        replacement = pool.get_or_create("nb.py")
        callbacks[0]()
        assert pool.get_or_create("nb.py") is replacement
    replacement.shutdown.assert_not_called()
    pool.shutdown()


@pytest.mark.parametrize("backend", ["uv", "pixi"])
def test_manifestless_host_clears_inherited_sandbox_identity(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    from marimo._environments.pixi import PixiMissingScriptMetadataError
    from marimo._environments.uv import UvMissingScriptMetadataError

    error = (
        PixiMissingScriptMetadataError(["pixi"], 1, "missing")
        if backend == "pixi"
        else UvMissingScriptMetadataError(["uv"], 1, "", "missing")
    )
    monkeypatch.setenv("MARIMO_SANDBOX_MODE", "multi")
    pool = AppHostPool(sandbox=True)
    with (
        patch(
            "marimo._environments.backends.sync_notebook",
            side_effect=error,
        ),
        patch(
            "marimo._session.app_host.pool.runtime_overlay", return_value=[]
        ),
        patch("marimo._session.app_host.pool.AppHost") as create,
    ):
        pool.get_or_create("nb.py")
    plan = create.call_args.kwargs["plan"]
    assert "MARIMO_SANDBOX_MODE" not in plan.env
    assert plan.env["MARIMO_MANAGE_SCRIPT_METADATA"] == "true"
    assert os.environ["MARIMO_SANDBOX_MODE"] == "multi"
    pool.shutdown()


def test_a_manifestless_notebook_falls_back_before_the_backend_runs(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "nb.py"
    notebook.write_text("import marimo\n\napp = marimo.App()\n")
    pool = AppHostPool(sandbox=True)
    with (
        patch("marimo._environments.backends.sync_notebook") as sync,
        patch(
            "marimo._session.app_host.pool.runtime_overlay", return_value=[]
        ),
        patch("marimo._session.app_host.pool.AppHost") as create,
    ):
        pool.get_or_create(str(notebook))
    sync.assert_not_called()
    plan = create.call_args.kwargs["plan"]
    assert "MARIMO_SANDBOX_MODE" not in plan.env
    pool.shutdown()


def test_a_notebook_with_a_manifest_is_synced(tmp_path: Path) -> None:
    notebook = tmp_path / "nb.py"
    notebook.write_text(
        "# /// script\n# dependencies = []\n# ///\nimport marimo\n"
    )
    pool = AppHostPool(sandbox=True)
    launched = MagicMock()
    launched.env = {}
    with (
        patch("marimo._environments.backends.sync_notebook") as sync,
        patch("marimo._environments.backends.launch", return_value=launched),
        patch(
            "marimo._session.app_host.pool.runtime_overlay", return_value=[]
        ),
        patch("marimo._session.app_host.pool.AppHost") as create,
    ):
        pool.get_or_create(str(notebook))
    sync.assert_called_once()
    assert (
        create.call_args.kwargs["plan"].env["MARIMO_SANDBOX_MODE"] == "multi"
    )
    pool.shutdown()


@pytest.mark.parametrize(
    "content",
    [
        b"# /// script\n# ...\n# ///\nimport marimo\n",
        b"# /// script\n# dependencies = []\n# ///\n" * 2,
        b"# /// script\n# dependencies = []\n# ///\n# \xff\n",
    ],
    ids=["malformed", "two-blocks", "not-utf-8"],
)
def test_an_unparseable_manifest_is_left_to_the_backend_to_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    import random

    from marimo._environments.uv import UvError
    from marimo._session.managers.ipc import KernelStartupError

    monkeypatch.setattr(random, "uniform", lambda _lo, _hi: 0.0)
    notebook = tmp_path / "nb.py"
    notebook.write_bytes(content)
    pool = AppHostPool(sandbox=True)
    with patch(
        "marimo._environments.backends.sync_notebook",
        side_effect=UvError("backend diagnostic"),
    ):
        with pytest.raises(KernelStartupError, match="backend diagnostic"):
            pool.get_or_create(str(notebook))
    pool.shutdown()


def test_environment_failure_is_reported_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    from marimo._environments.uv import UvError
    from marimo._session.managers.ipc import KernelStartupError

    monkeypatch.setattr(random, "uniform", lambda _lo, _hi: 0.0)
    pool = AppHostPool(sandbox=True)
    with (
        patch(
            "marimo._environments.backends.sync_notebook",
            side_effect=UvError("solver diagnostic"),
        ) as sync,
        patch("marimo._environments.backends.launch_fallback") as fallback,
    ):
        with pytest.raises(KernelStartupError, match="solver diagnostic"):
            pool.get_or_create("nb.py")
    # Retried once, like an IPC kernel's sync, before giving up.
    assert sync.call_count == 2
    fallback.assert_not_called()


def test_a_failed_environment_sync_is_retried_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two pods racing one environment (a gVisor sandbox's file locks are
    not shared across pods) must not fail a run-mode host any more than an
    IPC kernel: its sync is retried once too."""
    import random

    from marimo._environments.pixi import PixiCommandError

    monkeypatch.setattr(random, "uniform", lambda _lo, _hi: 0.0)
    environment = MagicMock()
    pool = AppHostPool(sandbox=True)
    with (
        patch(
            "marimo._environments.backends.sync_notebook",
            side_effect=[PixiCommandError(["pixi"], 1, "busy"), environment],
        ) as sync,
        patch(
            "marimo._environments.backends.launch",
            return_value=MagicMock(env={}),
        ) as launch,
        patch(
            "marimo._session.app_host.pool.runtime_overlay", return_value=[]
        ),
        patch("marimo._session.app_host.pool.AppHost") as create,
    ):
        pool.get_or_create("nb.py")
    assert sync.call_count == 2
    assert launch.call_args.args[0] is environment
    create.assert_called_once()
    pool.shutdown()
