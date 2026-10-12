"""Tests for the tutorial environment preparation."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import call, patch

from setup import notebook


class TestNotebookSetup(unittest.TestCase):
    def setUp(self) -> None:
        # Keep temporary repository paths out of subsequent tests.
        path_patch = patch.object(
            notebook.sys, "path", list(notebook.sys.path)
        )
        path_patch.start()
        self.addCleanup(path_patch.stop)

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

    def test_project_root_rejects_missing_repository(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "QAChat project"):
                notebook._project_root(Path(directory))

    def test_project_python_for_both_platforms(self) -> None:
        root = Path("/project")
        with patch.object(notebook.platform, "system", return_value="Windows"):
            self.assertEqual(
                notebook._project_python(root),
                root / ".venv/Scripts/python.exe",
            )
        with patch.object(notebook.platform, "system", return_value="Linux"):
            self.assertEqual(
                notebook._project_python(root),
                root / ".venv/bin/python",
            )

    def test_same_python_requires_matching_existing_file(self) -> None:
        with TemporaryDirectory() as directory:
            expected = Path(directory) / "python"
            with patch.object(notebook.sys, "executable", str(expected)):
                self.assertFalse(notebook._same_python(expected))
                expected.touch()
                self.assertTrue(notebook._same_python(expected))
                self.assertFalse(
                    notebook._same_python(Path(directory) / "other")
                )

    def test_required_modules_are_platform_specific(self) -> None:
        choices = (
            ("Darwin", "arm64", True),
            ("Darwin", "x86_64", False),
            ("Linux", "aarch64", False),
            ("Windows", "AMD64", False),
        )
        for system, machine, needs_mlx in choices:
            with self.subTest(system=system, machine=machine):
                with patch.object(
                    notebook.platform, "system", return_value=system
                ), patch.object(
                    notebook.platform, "machine", return_value=machine
                ):
                    modules = notebook._required_modules()
                self.assertIn("torch", modules)
                self.assertEqual("mlx" in modules, needs_mlx)

    def test_missing_modules_checks_import_specs(self) -> None:
        with patch.object(
            notebook, "_required_modules", return_value=("a", "b")
        ), patch.object(
            notebook, "find_spec", side_effect=lambda name: (
                None if name == "b" else object()
            )
        ):
            self.assertEqual(notebook._missing_modules(), ("b",))

    def test_missing_packages_after_sync_raise(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qachat").mkdir()
            (root / "pyproject.toml").touch()
            with patch.object(notebook, "_same_python", return_value=True):
                with patch.object(
                    notebook, "_missing_modules", return_value=("missing",)
                ), patch.object(notebook, "_sync_environment") as sync:
                    with patch.object(notebook, "invalidate_caches") as clear:
                        with self.assertRaisesRegex(
                            RuntimeError, "still missing"
                        ):
                            notebook.prepare_notebook(root)
            sync.assert_called_once_with(root.resolve())
            clear.assert_called_once_with()

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
                    "--extra",
                    "bench",
                ],
                cwd=Path("/project"),
                check=True,
            ),
        )

    def test_uv_sync_on_windows_omits_mlx(self) -> None:
        with patch.object(notebook.shutil, "which", return_value="uv"):
            with patch.object(
                notebook.platform, "system", return_value="Windows"
            ), patch.object(notebook.subprocess, "run") as run:
                notebook._sync_environment(Path("/project"))
        run.assert_called_once_with(
            ["uv", "sync", "--extra", "torch", "--extra", "bench"],
            cwd=Path("/project"),
            check=True,
        )

    def test_setup_script_fallbacks(self) -> None:
        root = Path("/project")
        choices = (
            (
                "Windows",
                [
                    "powershell", "-NoProfile", "-ExecutionPolicy",
                    "Bypass", "-File", str(root / "setup/setup.ps1"),
                ],
            ),
            ("Linux", ["/bin/sh", str(root / "setup/setup.sh")]),
        )
        for system, command in choices:
            with self.subTest(system=system):
                with patch.object(notebook.shutil, "which", return_value=None):
                    with patch.object(
                        notebook.platform, "system", return_value=system
                    ), patch.object(notebook.subprocess, "run") as run:
                        notebook._sync_environment(root)
                run.assert_called_once_with(
                    command, cwd=root, check=True
                )


if __name__ == "__main__":
    unittest.main()
