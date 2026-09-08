"""Tests for the tutorial environment preparation."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import call, patch

from setup import notebook


class TestNotebookSetup(unittest.TestCase):

    def test_project_root(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qachat").mkdir()
            (root / "pyproject.toml").touch()
            nested = root / "tutorials/examples"
            nested.mkdir(parents=True)

            self.assertEqual(
                notebook._project_root(nested),
                root.resolve(),
            )

    def test_prepare_notebook_when_ready(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qachat").mkdir()
            (root / "pyproject.toml").touch()

            with patch.object(notebook, "_same_python", return_value=True):
                with patch.object(
                    notebook,
                    "_missing_modules",
                    return_value=(),
                ):
                    actual = notebook.prepare_notebook(root)

            self.assertEqual(actual, root.resolve())

    def test_prepare_notebook_repairs_missing_packages(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qachat").mkdir()
            (root / "pyproject.toml").touch()

            with patch.object(notebook, "_same_python", return_value=True):
                with patch.object(
                    notebook,
                    "_missing_modules",
                    side_effect=[("mlx_embeddings",), ()],
                ):
                    with patch.object(
                        notebook,
                        "_sync_environment",
                    ) as sync:
                        notebook.prepare_notebook(root)

            sync.assert_called_once_with(root.resolve())

    def test_wrong_kernel_is_repaired_then_reported(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qachat").mkdir()
            (root / "pyproject.toml").touch()

            with patch.object(notebook, "_same_python", return_value=False):
                with patch.object(
                    notebook,
                    "_sync_environment",
                ) as sync:
                    with self.assertRaisesRegex(RuntimeError, "Select"):
                        notebook.prepare_notebook(root)

            sync.assert_called_once_with(root.resolve())

    def test_uv_sync_command(self) -> None:
        with patch.object(
            notebook.shutil,
            "which",
            return_value="/tools/uv",
        ):
            with patch.object(
                notebook.platform,
                "system",
                return_value="Darwin",
            ):
                with patch.object(notebook.subprocess, "run") as run:
                    notebook._sync_environment(Path("/project"))

        self.assertEqual(
            run.call_args,
            call(
                [
                    "/tools/uv",
                    "sync",
                    "--extra",
                    "mlx",
                    "--extra",
                    "torch",
                ],
                cwd=Path("/project"),
                check=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
