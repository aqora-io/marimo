# Copyright 2026 Marimo. All rights reserved.
"""What marimo layers into sandbox launches."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from marimo import _loggers
from marimo._environments.errors import EnvironmentManagerError
from marimo._utils.versions import is_editable
from marimo._version import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = _loggers.marimo_logger()

DepExtras = Literal["lsp", "recommended"]


def marimo_dir() -> Path:
    """The repository root when marimo runs from a checkout."""
    return Path(__file__).parent.parent.parent


@dataclass(frozen=True)
class RuntimeOverlay:
    """The requirements a launch adds on top of a script environment.

    A script environment belongs to the notebook. A backend builds it from
    the notebook's manifest, and both `uv run notebook.py` and `marimo edit
    --sandbox notebook.py` address the same cached environment for the same
    path. But a marimo process launched into that environment needs packages
    the manifest does not describe -- at minimum, the marimo doing the
    launching. An overlay is how those packages reach the process:

    * The backend resolves the overlay into an ephemeral environment created
      from the script environment's interpreter. That environment chains the
      script environment's packages, so the notebook's dependencies are still
      importable.
    * An entry here wins over the same package in the script environment, so
      a version stated in the overlay is the version the process imports.
    * Nothing is installed into the script environment and the manifest is
      never edited. The environment the notebook's own tool builds stays
      byte-identical to the one marimo launches into, which is what lets the
      two share a cache entry.

    # Properties

    * `runtime` is the marimo the launched process must import: a direct
      reference to the wheel named by `MARIMO_RUNTIME_WHEEL` when set,
      `marimo[<extras>]==<version>` otherwise, or `-e <path>` when marimo
      runs from a development checkout. Every overlay has one; see below.
    * `command` holds requirements of the marimo *command* being launched,
      such as `nbformat` for an ipynb export or `playwright` for a
      thumbnail. These exist only because such a command runs inside the
      notebook's environment today, and they go away as those commands move
      into environments marimo owns.

    # What belongs in an overlay

    Only marimo's own requirements. A package the notebook's code imports
    belongs in the manifest, where a reader, a reviewer, and a plain
    `uv run notebook.py` can all see it -- add it with `NotebookSandbox.add`
    instead. If a package is being put here so that some notebook's cells
    work, it is in the wrong place.

    # Why every launch carries a runtime

    The server and the kernel are one program in two processes: they share
    message schemas, session semantics, and code paths, so the kernel has to
    be the same marimo that launched it. The manifest cannot express that.
    It says `marimo` loosely, because it is a shared file that has no way to
    know which marimo will open it, and writing `marimo==<version>` there
    would put one machine's state into a document meant to outlive it. So
    the version binding lives in the overlay, and it is not optional: a
    launch without one would run whatever marimo the manifest happened to
    resolve. A versioned kernel protocol could someday relax this and retire
    the injection entirely; until then, every launch goes through the layer.

    # Warning

    A backend resolves the overlay against PyPI without seeing what the
    script environment already contains. For a pixi environment that means
    the overlay's transitive dependencies arrive as PyPI wheels and shadow
    any conda-forge build of the same package. This is acceptable for
    marimo's own dependency tree, which is small and pure-Python; an overlay
    is not a general bridge between conda and PyPI packaging.
    """

    runtime: str
    command: tuple[str, ...] = ()

    @property
    def requirements(self) -> tuple[str, ...]:
        """The overlay as requirement strings, runtime first.

        Backends translate these into their own layering flags: uv passes
        each as `--with` (or `--with-editable` for a local path), and pixi
        borrows the same uv verb through `pixi exec`.
        """
        return (self.runtime, *self.command)


def runtime_overlay(
    extras: Sequence[DepExtras] = (),
    command: Sequence[str] = (),
) -> RuntimeOverlay:
    """The overlay binding a launch to the running marimo.

    Resolves the runtime requirement in precedence order: a direct
    reference to the wheel named by `MARIMO_RUNTIME_WHEEL` when set (for a
    distribution that installs marimo from a wheel rather than PyPI, where
    a same-version PyPI marimo would otherwise satisfy `marimo==<version>`
    without ever installing the wheel); the installed version; or an
    editable install of the checkout when marimo is running from source,
    so that a contributor's sandbox runs their working tree rather than
    the last release. `extras` are marimo's own extras, such as `lsp` for
    a process that hosts the language server.

    Requirements are never written to the notebook's manifest; see
    `RuntimeOverlay`.
    """
    suffix = f"[{','.join(extras)}]" if extras else ""
    wheel_env = os.environ.get("MARIMO_RUNTIME_WHEEL")
    if wheel_env:
        wheel = _runtime_wheel(wheel_env)
        runtime = f"marimo{suffix} @ {wheel.resolve().as_uri()}"
    elif is_editable("marimo"):
        LOGGER.info("Using editable of marimo for sandbox")
        runtime = f"-e {marimo_dir()}{suffix}"
    else:
        runtime = f"marimo{suffix}=={__version__}"
    return RuntimeOverlay(runtime=runtime, command=tuple(command))


def _runtime_wheel(value: str) -> Path:
    """Validate `MARIMO_RUNTIME_WHEEL` and return the wheel it names.

    `value` is either a wheel file or a directory containing exactly one
    `marimo-*.whl`. Either way, the wheel's filename must name `marimo`
    at this process's version -- the overlay always binds the launched
    process to the running marimo; see `RuntimeOverlay`.
    """
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
    from packaging.version import Version

    path = Path(value)
    if path.is_dir():
        matches = sorted(path.glob("marimo-*.whl"))
        if len(matches) != 1:
            raise EnvironmentManagerError(
                f"MARIMO_RUNTIME_WHEEL={value} must contain exactly one "
                f"marimo-*.whl wheel; found {len(matches)}"
            )
        wheel = matches[0]
    elif path.is_file():
        wheel = path
    else:
        raise EnvironmentManagerError(
            f"MARIMO_RUNTIME_WHEEL={value} does not exist"
        )

    try:
        name, version, _build, _tags = parse_wheel_filename(wheel.name)
    except InvalidWheelFilename as error:
        raise EnvironmentManagerError(
            f"MARIMO_RUNTIME_WHEEL={value} is not a valid wheel filename: "
            f"{error}"
        ) from error
    if name != "marimo":
        raise EnvironmentManagerError(
            f"MARIMO_RUNTIME_WHEEL={value} must be a marimo wheel, found "
            f"{name!r}"
        )
    if version != Version(__version__):
        raise EnvironmentManagerError(
            f"MARIMO_RUNTIME_WHEEL={value} is marimo {version}, but this "
            f"process is {__version__}"
        )
    return wheel
