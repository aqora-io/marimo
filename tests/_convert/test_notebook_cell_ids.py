# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

from textwrap import dedent

from marimo._ast.cell_id import CellIdGenerator
from marimo._ast.names import SETUP_CELL_NAME
from marimo._convert.converters import MarimoConvert


def _kernel_cell_ids(source: str) -> list[str]:
    """Simulate the cell IDs the kernel would generate for a notebook."""
    from marimo._ast.parse import parse_notebook
    from marimo._schemas.serialization import SetupCell

    ir = parse_notebook(source)
    assert ir is not None
    gen = CellIdGenerator()
    ids = []
    for i, cell_def in enumerate(ir.cells):
        if isinstance(cell_def, SetupCell) or (
            i == 0 and cell_def.name == SETUP_CELL_NAME
        ):
            ids.append(SETUP_CELL_NAME)
        else:
            ids.append(gen.create_cell_id())
    return ids


def test_snapshot_ids_match_kernel_ids():
    source = dedent(
        """
        import marimo

        __generated_with = "0.1.0"
        app = marimo.App()

        @app.cell
        def hello():
            x = 1
            return (x,)

        @app.cell
        def world(x):
            y = x + 1
            return (y,)

        if __name__ == "__main__":
            app.run()
    """
    ).strip()

    notebook = MarimoConvert.from_py(source).to_notebook_v1()
    snapshot_ids = [c["id"] for c in notebook["cells"]]
    kernel_ids = _kernel_cell_ids(source)

    assert snapshot_ids == kernel_ids
    assert all(snapshot_id is not None for snapshot_id in snapshot_ids)


def test_snapshot_ids_match_kernel_ids_with_setup_cell():
    source = dedent(
        """
        import marimo

        __generated_with = "0.1.0"
        app = marimo.App()

        with app.setup:
            import numpy as np

        @app.cell
        def hello():
            x = 1
            return (x,)

        @app.cell
        def world(x):
            y = x + 1
            return (y,)

        if __name__ == "__main__":
            app.run()
    """
    ).strip()

    notebook = MarimoConvert.from_py(source).to_notebook_v1()
    snapshot_ids = [c["id"] for c in notebook["cells"]]
    kernel_ids = _kernel_cell_ids(source)

    assert snapshot_ids == kernel_ids
    assert snapshot_ids[0] == SETUP_CELL_NAME


def test_snapshot_ids_are_deterministic():
    source = dedent(
        """
        import marimo

        __generated_with = "0.1.0"
        app = marimo.App()

        @app.cell
        def _():
            x = 1
            return (x,)

        @app.cell
        def _():
            y = 2
            return (y,)

        @app.cell
        def _():
            z = 3
            return (z,)

        if __name__ == "__main__":
            app.run()
    """
    ).strip()

    ids_1 = [
        c["id"]
        for c in MarimoConvert.from_py(source).to_notebook_v1()["cells"]
    ]
    ids_2 = [
        c["id"]
        for c in MarimoConvert.from_py(source).to_notebook_v1()["cells"]
    ]

    assert ids_1 == ids_2
    assert len(set(ids_1)) == 3  # all unique


def test_snapshot_ids_unaffected_by_cell_manager_operations():
    """Snapshot IDs are generated from a fresh CellIdGenerator, not shared
    with the kernel's CellManager. Adding/deleting cells mid-session
    advances the CellManager's generator but must not affect snapshot
    generation."""
    from marimo._ast.load import load_notebook_ir
    from marimo._ast.parse import parse_notebook

    source = dedent(
        """
        import marimo

        __generated_with = "0.1.0"
        app = marimo.App()

        @app.cell
        def hello():
            x = 1
            return (x,)

        @app.cell
        def world(x):
            y = x + 1
            return (y,)

        @app.cell
        def foo(y):
            z = y + 1
            return (z,)

        if __name__ == "__main__":
            app.run()
    """
    ).strip()

    # 1. Load notebook via the kernel path (fresh CellManager)
    ir = parse_notebook(source)
    assert ir is not None
    app = load_notebook_ir(ir, filepath="notebook.py")
    initial_kernel_ids = list(app._cell_manager.cell_ids())

    # 2. Simulate mid-session add/delete: create several cells
    #    (advances the CellManager's CellIdGenerator RNG state)
    extra_ids = [app._cell_manager.create_cell_id() for _ in range(10)]
    # The generator has now been called 10 extra times
    assert len(app._cell_manager.seen_ids) > len(initial_kernel_ids)

    # The extra IDs are distinct from the original ones
    for eid in extra_ids:
        assert eid not in initial_kernel_ids

    # 3. Generate a snapshot from the same source (fresh generator)
    snapshot = MarimoConvert.from_py(source).to_notebook_v1()
    snapshot_ids = [c["id"] for c in snapshot["cells"]]

    # 4. Snapshot IDs must match the initial kernel IDs, unaffected
    #    by the CellManager's advanced generator state
    assert snapshot_ids == initial_kernel_ids

    # 5. Also matches a completely fresh kernel simulation
    assert snapshot_ids == _kernel_cell_ids(source)
