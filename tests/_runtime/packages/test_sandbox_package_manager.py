# Copyright 2026 Marimo. All rights reserved.
"""Recording a cell's imports in a sandboxed notebook's manifest.

Registering a cell blocks its run, so recording must never ask the backend
to solve: pixi takes seconds for that, on every import a cell adds.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from marimo._environments import script_metadata
from marimo._environments.environment import Environment
from marimo._environments.sandbox import NotebookSandbox
from marimo._runtime.packages.sandbox_package_manager import (
    SandboxPackageManager,
)
from tests._environments.test_sandbox_interface import FakeBackend

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class NeverSolves(FakeBackend):
    def packages(self, target: Any, environment: Any) -> Any:
        del target, environment
        raise AssertionError("recording an import asked the backend")


def install(
    site_packages: Path, name: str, version: str, installer: str
) -> None:
    info = site_packages / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )
    (info / "INSTALLER").write_text(f"{installer}\n")


def sandboxed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    installed: list[tuple[str, str, str]],
) -> tuple[SandboxPackageManager, NeverSolves, Path]:
    """A manager for a notebook whose kernel runs in an environment holding
    `installed` (name, version, INSTALLER), plus `overlaypkg` 7.8.9 from the
    runtime overlay outside it."""
    prefix = tmp_path / "env"
    site_packages = prefix / "lib" / "site-packages"
    site_packages.mkdir(parents=True)
    for name, version, installer in installed:
        install(site_packages, name, version, installer)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    install(overlay, "overlaypkg", "7.8.9", "uv")
    monkeypatch.syspath_prepend(str(overlay))
    monkeypatch.syspath_prepend(str(site_packages))

    notebook = tmp_path / "notebook.py"
    notebook.write_text(
        script_metadata.dumps({"dependencies": ["marimo", "declared>=1"]})
        + "\nimport marimo\n"
    )
    backend = NeverSolves(tmp_path)
    backend.name = backend_name  # type: ignore[misc]
    sandbox = NotebookSandbox(
        str(notebook),
        backend_name,  # type: ignore[arg-type]
        environment=Environment(
            python=sys.executable, root=str(prefix), action="unchanged"
        ),
        adapter=backend,  # type: ignore[arg-type]
    )
    return SandboxPackageManager(sandbox), backend, notebook


def dependencies(notebook: Path) -> list[str]:
    project = script_metadata.loads(notebook.read_text())
    assert project is not None
    return list(project["dependencies"])


def test_imports_are_pinned_to_the_versions_the_kernel_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, backend, notebook = sandboxed(
        tmp_path,
        monkeypatch,
        "pixi",
        [
            ("pypkg", "1.2.3", "uv-pixi"),
            ("condapkg", "4.5.6", "conda"),
            ("declared", "2.0.0", "uv-pixi"),
        ],
    )

    assert manager.update_notebook_script_metadata(
        str(notebook),
        import_namespaces_to_add=[
            "json",
            "os",
            "marimo",
            "pypkg",
            "condapkg",
            "declared",
            "overlaypkg",
            "not_installed",
        ],
        upgrade=False,
    )

    # PyPI packages from the environment and the overlay are pinned; the
    # standard library, marimo, what is already declared, pixi's conda
    # packages and what is not installed are left alone.
    assert backend.add_requests == ["overlaypkg==7.8.9", "pypkg==1.2.3"]
    assert dependencies(notebook) == [
        "marimo",
        "declared>=1",
        "overlaypkg==7.8.9",
        "pypkg==1.2.3",
    ]


def test_standard_library_imports_leave_the_manifest_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, backend, notebook = sandboxed(tmp_path, monkeypatch, "pixi", [])
    before = notebook.read_bytes()

    assert manager.update_notebook_script_metadata(
        str(notebook),
        import_namespaces_to_add=["json", "os", "re", "__future__"],
        upgrade=False,
    )

    assert backend.add_requests == []
    assert notebook.read_bytes() == before


def test_a_uv_environment_holds_pypi_packages_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, backend, notebook = sandboxed(
        tmp_path, monkeypatch, "uv", [("pypkg", "1.2.3", "uv")]
    )

    assert manager.update_notebook_script_metadata(
        str(notebook), import_namespaces_to_add=["pypkg"], upgrade=False
    )

    assert backend.add_requests == ["pypkg==1.2.3"]


def test_an_import_whose_package_takes_extras_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `ibis` maps to `ibis-framework[duckdb]`; the record pins the package.
    manager, backend, notebook = sandboxed(
        tmp_path, monkeypatch, "uv", [("ibis_framework", "10.0.0", "uv")]
    )

    assert manager.update_notebook_script_metadata(
        str(notebook), import_namespaces_to_add=["ibis"], upgrade=False
    )

    assert backend.add_requests == ["ibis-framework==10.0.0"]
