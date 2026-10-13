"""Test the QAChat console entry point without launching desktop Qt.

Argument parsing, platform-specific dependency selection, and optional
``uv sync`` preparation are tested using mocked boundaries. No test starts
the GUI, installs packages, accesses the network, or probes real backends.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import call, patch

from qachat import cli
from qachat.runtime import (
    BACKEND_AUTO,
    BACKEND_MLX,
    BACKEND_TORCH,
)


class TestCommandLine(unittest.TestCase):
    """Test argument handling and delegation to the optional GUI."""

    def test_no_arguments_prints_help_without_starting_gui(self) -> None:
        """Show usage and avoid environment preparation by default."""

        output = StringIO()
        with (
            patch.object(cli, "_ensure_gui_environment") as ensure,
            patch("qachat.gui.app.main") as gui_main,
            redirect_stdout(output),
        ):
            result = cli.main([])

        self.assertEqual(result, 0)
        self.assertIn("usage: qachat", output.getvalue())
        self.assertIn("--gui", output.getvalue())
        self.assertIn("--backend", output.getvalue())
        ensure.assert_not_called()
        gui_main.assert_not_called()

    def test_backend_without_gui_only_displays_help(self) -> None:
        """Do not install backend extras until GUI launch is requested."""

        for backend in (BACKEND_AUTO, BACKEND_MLX, BACKEND_TORCH):
            with self.subTest(backend=backend):
                with (
                    patch.object(cli, "_ensure_gui_environment") as ensure,
                    patch("qachat.gui.app.main") as gui_main,
                    redirect_stdout(StringIO()) as output,
                ):
                    result = cli.main(["--backend", backend])
                self.assertEqual(result, 0)
                self.assertIn("--gui", output.getvalue())
                ensure.assert_not_called()
                gui_main.assert_not_called()

    def test_help_exits_zero_without_preparing_environment(self) -> None:
        """Honor argparse's help exit even if GUI was requested."""

        for arguments in (["--help"], ["--gui", "--help"]):
            with self.subTest(arguments=arguments):
                with (
                    patch.object(cli, "_ensure_gui_environment") as ensure,
                    patch("qachat.gui.app.main") as gui_main,
                    redirect_stdout(StringIO()) as output,
                ):
                    with self.assertRaises(SystemExit) as caught:
                        cli.main(arguments)
                self.assertEqual(caught.exception.code, 0)
                self.assertIn("--gui", output.getvalue())
                ensure.assert_not_called()
                gui_main.assert_not_called()

    def test_invalid_arguments_exit_before_gui_preparation(self) -> None:
        """Reject unknown options, invalid backends, and missing values."""

        cases = (
            ["--unknown"],
            ["--gui", "--backend", "invalid"],
            ["--gui", "--backend", "MLX"],
            ["--gui", "--backend"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with (
                    patch.object(cli, "_ensure_gui_environment") as ensure,
                    patch("qachat.gui.app.main") as gui_main,
                    redirect_stderr(StringIO()) as output,
                ):
                    with self.assertRaises(SystemExit) as caught:
                        cli.main(arguments)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("error:", output.getvalue())
                ensure.assert_not_called()
                gui_main.assert_not_called()

    def test_gui_option_opens_desktop_interface(self) -> None:
        """Return the GUI exit code and pass only the executable to Qt."""

        with (
            patch.object(cli, "_ensure_gui_environment") as ensure,
            patch("qachat.gui.app.main", return_value=7) as gui_main,
            patch.object(cli.sys, "argv", ["qachat-launcher", "unused"]),
        ):
            result = cli.main(["--gui", "--backend", BACKEND_MLX])

        self.assertEqual(result, 7)
        ensure.assert_called_once_with(BACKEND_MLX)
        gui_main.assert_called_once_with(["qachat-launcher"])

    def test_gui_launch_uses_auto_backend_by_default(self) -> None:
        """Prepare the auto backend without forwarding it to Qt."""

        with (
            patch.object(cli, "_ensure_gui_environment") as ensure,
            patch("qachat.gui.app.main", return_value=0) as gui_main,
            patch.object(cli.sys, "argv", ["qachat-executable"]),
        ):
            result = cli.main(["--gui"])

        self.assertEqual(result, 0)
        ensure.assert_called_once_with(BACKEND_AUTO)
        gui_main.assert_called_once_with(["qachat-executable"])

    def test_explicit_backends_are_used_for_preparation_only(self) -> None:
        """Keep backend extras separate from the GUI argument vector."""

        for backend in (BACKEND_MLX, BACKEND_TORCH):
            with self.subTest(backend=backend):
                with (
                    patch.object(cli, "_ensure_gui_environment") as ensure,
                    patch("qachat.gui.app.main", return_value=0) as gui_main,
                    patch.object(cli.sys, "argv", ["launcher"]),
                ):
                    result = cli.main(["--gui", "--backend", backend])
                self.assertEqual(result, 0)
                ensure.assert_called_once_with(backend)
                gui_main.assert_called_once_with(["launcher"])

    def test_none_arguments_use_the_process_command_line(self) -> None:
        """Allow the installed console script to read ``sys.argv``."""

        with (
            patch.object(cli, "_ensure_gui_environment") as ensure,
            patch("qachat.gui.app.main", return_value=3) as gui_main,
            patch.object(
                cli.sys,
                "argv",
                ["installed-qachat", "--gui", "--backend", "torch"],
            ),
        ):
            result = cli.main()

        self.assertEqual(result, 3)
        ensure.assert_called_once_with(BACKEND_TORCH)
        gui_main.assert_called_once_with(["installed-qachat"])

    def test_environment_failure_prevents_gui_launch(self) -> None:
        """Propagate preparation failures without starting the desktop."""

        with (
            patch.object(
                cli,
                "_ensure_gui_environment",
                side_effect=RuntimeError("missing dependencies"),
            ) as ensure,
            patch("qachat.gui.app.main") as gui_main,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing dependencies"):
                cli.main(["--gui"])

        ensure.assert_called_once_with(BACKEND_AUTO)
        gui_main.assert_not_called()

    def test_gui_startup_errors_propagate(self) -> None:
        """Do not hide exceptions from the delegated application entry."""

        with (
            patch.object(cli, "_ensure_gui_environment") as ensure,
            patch(
                "qachat.gui.app.main",
                side_effect=RuntimeError("GUI startup failed"),
            ) as gui_main,
        ):
            with self.assertRaisesRegex(RuntimeError, "GUI startup failed"):
                cli.main(["--gui"])

        ensure.assert_called_once_with(BACKEND_AUTO)
        gui_main.assert_called_once()


class TestSelectedBackend(unittest.TestCase):
    """Test CLI dependency-backend selection independently of hardware."""

    def test_explicit_backends_do_not_probe_platform(self) -> None:
        """Honor explicit requests without asking about the host CPU."""

        for backend in (BACKEND_MLX, BACKEND_TORCH):
            with self.subTest(backend=backend):
                with patch.object(cli.platform, "machine") as machine:
                    selected = cli._selected_backend(backend)
                self.assertEqual(selected, backend)
                machine.assert_not_called()

    def test_auto_prefers_mlx_on_apple_silicon(self) -> None:
        """Recognize Apple Silicon with any machine-name capitalization."""

        for machine_name in ("arm64", "ARM64", "Arm64"):
            with self.subTest(machine=machine_name):
                with (
                    patch.object(cli.sys, "platform", "darwin"),
                    patch.object(
                        cli.platform, "machine", return_value=machine_name
                    ),
                ):
                    selected = cli._selected_backend(BACKEND_AUTO)
                self.assertEqual(selected, BACKEND_MLX)

    def test_auto_uses_torch_on_other_platforms(self) -> None:
        """Choose PyTorch for Intel macOS, Linux, and Windows."""

        cases = (
            ("darwin", "x86_64"),
            ("linux", "aarch64"),
            ("linux", "x86_64"),
            ("win32", "AMD64"),
        )
        for operating_system, machine_name in cases:
            with self.subTest(platform=operating_system, machine=machine_name):
                with (
                    patch.object(cli.sys, "platform", operating_system),
                    patch.object(
                        cli.platform, "machine", return_value=machine_name
                    ),
                ):
                    selected = cli._selected_backend(BACKEND_AUTO)
                self.assertEqual(selected, BACKEND_TORCH)


class TestGUIEnvironment(unittest.TestCase):
    """Verify optional dependency preparation without executing ``uv``."""

    def test_auto_preparation_uses_mlx_for_uppercase_arm64(self) -> None:
        """Choose MLX dependencies rather than PyTorch on native macOS."""

        with (
            patch.object(cli.sys, "platform", "darwin"),
            patch.object(cli.platform, "machine", return_value="ARM64"),
            patch.object(cli, "find_spec", return_value=object()) as found,
            patch.object(cli.shutil, "which") as which,
            patch.object(cli.subprocess, "run") as sync,
        ):
            cli._ensure_gui_environment(BACKEND_AUTO)

        self.assertEqual(
            found.call_args_list,
            [call("PySide6"), call("mlx_embeddings"), call("mlx_lm")],
        )
        which.assert_not_called()
        sync.assert_not_called()

    def test_installed_dependencies_require_no_environment_changes(
        self,
    ) -> None:
        """Return immediately when Qt and the selected backend are found."""

        packages = (
            (BACKEND_MLX, ("PySide6", "mlx_embeddings", "mlx_lm")),
            (BACKEND_TORCH, ("PySide6", "torch", "transformers")),
        )
        for backend, expected in packages:
            with self.subTest(backend=backend):
                with (
                    patch.object(
                        cli, "find_spec", return_value=object()
                    ) as find,
                    patch.object(cli.shutil, "which") as which,
                    patch.object(cli.subprocess, "run") as run,
                    patch.object(
                        cli.importlib, "invalidate_caches"
                    ) as invalidate,
                    redirect_stdout(StringIO()) as output,
                ):
                    cli._ensure_gui_environment(backend)

                self.assertEqual(
                    find.call_args_list, [call(name) for name in expected]
                )
                self.assertEqual(output.getvalue(), "")
                which.assert_not_called()
                run.assert_not_called()
                invalidate.assert_not_called()

    def test_missing_uv_reports_install_command(self) -> None:
        """Explain how to install extras when ``uv`` is unavailable."""

        with (
            patch.object(cli, "find_spec", return_value=None) as find,
            patch.object(cli.shutil, "which", return_value=None) as which,
            patch.object(cli.subprocess, "run") as run,
            patch.object(
                cli.importlib, "invalidate_caches"
            ) as invalidate,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "uv sync --extra gui --extra torch"
            ):
                cli._ensure_gui_environment(BACKEND_TORCH)

        find.assert_called_once_with("PySide6")
        which.assert_called_once_with("uv")
        run.assert_not_called()
        invalidate.assert_not_called()

    def test_missing_checkout_profile_does_not_run_uv(self) -> None:
        """Require a project pyproject before attempting synchronization."""

        with TemporaryDirectory() as directory:
            fake_cli_path = Path(directory) / "qachat" / "cli.py"
            with (
                patch.object(cli, "__file__", str(fake_cli_path)),
                patch.object(cli, "find_spec", return_value=None),
                patch.object(
                    cli.shutil, "which", return_value="/fake/bin/uv"
                ),
                patch.object(cli.subprocess, "run") as run,
                patch.object(
                    cli.importlib, "invalidate_caches"
                ) as invalidate,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "uv sync --extra gui --extra mlx"
                ):
                    cli._ensure_gui_environment(BACKEND_MLX)

        run.assert_not_called()
        invalidate.assert_not_called()

    def test_missing_components_trigger_backend_specific_sync(self) -> None:
        """Invoke the selected extras from the checkout, never for real."""

        cases = (
            (BACKEND_MLX, "mlx_lm", "mlx"),
            (BACKEND_TORCH, "PySide6", "torch"),
            (BACKEND_TORCH, "transformers", "torch"),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkout"
            root.mkdir()
            (root / "pyproject.toml").write_text("[project]\n")
            fake_cli_path = root / "qachat" / "cli.py"

            for backend, missing, extra in cases:
                with self.subTest(backend=backend, missing=missing):
                    seen: list[str] = []

                    def find_module(name: str) -> object | None:
                        """Return mocked module availability and log checks.

                        Parameters
                        ----------
                        name: str
                            Dependency module requested by the CLI.

                        Returns
                        -------
                        object | None
                            None for the missing module, an object otherwise.

                        """

                        seen.append(name)
                        return None if name == missing else object()

                    with (
                        patch.object(cli, "__file__", str(fake_cli_path)),
                        patch.object(
                            cli, "find_spec", side_effect=find_module
                        ),
                        patch.object(
                            cli.shutil, "which", return_value="/fake/uv"
                        ) as which,
                        patch.object(cli.subprocess, "run") as run,
                        patch.object(
                            cli.importlib, "invalidate_caches"
                        ) as invalidate,
                        redirect_stdout(StringIO()) as output,
                    ):
                        cli._ensure_gui_environment(backend)

                    self.assertIn(missing, seen)
                    self.assertTrue(output.getvalue().startswith("Preparing"))
                    self.assertIn(extra.upper(), output.getvalue())
                    which.assert_called_once_with("uv")
                    run.assert_called_once_with(
                        [
                            "/fake/uv", "sync", "--extra", "gui",
                            "--extra", extra,
                        ],
                        cwd=root,
                        check=True,
                    )
                    invalidate.assert_called_once_with()

    def test_failed_sync_propagates_without_invalidating_caches(
        self,
    ) -> None:
        """Report failing setup commands without pretending they worked."""

        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkout"
            root.mkdir()
            (root / "pyproject.toml").write_text("[project]\n")
            error = subprocess.CalledProcessError(9, ["uv", "sync"])
            with (
                patch.object(cli, "__file__", str(root / "qachat/cli.py")),
                patch.object(cli, "find_spec", return_value=None),
                patch.object(
                    cli.shutil, "which", return_value="/fake/uv"
                ),
                patch.object(cli.subprocess, "run", side_effect=error) as run,
                patch.object(
                    cli.importlib, "invalidate_caches"
                ) as invalidate,
                redirect_stdout(StringIO()),
            ):
                with self.assertRaises(
                    subprocess.CalledProcessError
                ) as caught:
                    cli._ensure_gui_environment(BACKEND_TORCH)

        self.assertIs(caught.exception, error)
        run.assert_called_once()
        invalidate.assert_not_called()


class TestLazyCLIImport(unittest.TestCase):
    """Protect the dependency-light command-line import boundary."""

    def test_importing_cli_does_not_import_qt_or_gui_application(self) -> None:
        """Keep the CLI usable for help without installing PySide6."""

        command = (
            "import sys\n"
            "import qachat.cli\n"
            "assert 'qachat.gui.app' not in sys.modules\n"
            "assert 'PySide6' not in sys.modules\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
