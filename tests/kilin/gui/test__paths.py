"""Verify development and installed GUI data-directory resolution.

Tests simulate source checkouts, installed packages, and platform-specific
user locations without requiring Qt, GPU dependencies, or real user writes.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.gui import _paths
from kilin.gui import engine as gui_engine


class TestGUIPaths(unittest.TestCase):
    """Resolve GUI-owned paths without writing to package directories."""

    def setUp(self) -> None:
        """Provide source and installation roots without user state."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.home = self.root / "home"
        self.module = self.root / "site-packages/kilin/gui/_paths.py"
        self.module.parent.mkdir(parents=True)
        self.module.write_text("", encoding="utf-8")
        self.module_patch = patch.object(_paths, "__file__", str(self.module))
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.home_patch = patch.object(
            _paths.Path, "home", return_value=self.home
        )
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)
        self.env_patch = patch.dict("os.environ", {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_checkout_uses_repository_sandbox(self) -> None:
        """Preserve a development checkout's original workspace location."""

        checkout = self.root / "checkout"
        module = checkout / "kilin/gui/_paths.py"
        module.parent.mkdir(parents=True)
        module.write_text("", encoding="utf-8")
        (checkout / "kilin/gui/engine.py").touch()
        (checkout / "tests/kilin").mkdir(parents=True)
        (checkout / "pyproject.toml").write_text(
            "[project]\n", encoding="utf-8"
        )
        with patch.object(_paths, "__file__", str(module)):
            self.assertEqual(
                _paths.default_workspace_directory(), checkout / "sandbox"
            )
        self.assertFalse((checkout / "sandbox").exists())

    def test_installed_package_uses_user_owned_directory(self) -> None:
        """Do not construct a workspace under site-packages."""

        with patch.object(_paths.sys, "platform", "darwin"):
            workspace = _paths.default_workspace_directory()
            self.assertEqual(workspace, self.home / "Documents/KíLín")
            self.assertEqual(
                _paths.preferences_path(),
                self.home / "Library/Application Support/kilin/"
                "preferences.json",
            )
        self.assertFalse(workspace.exists())
        self.assertFalse(self.home.exists())

    def test_linux_xdg_directories_and_relative_fallback(self) -> None:
        """Use an absolute XDG config path; projects go in Documents."""

        with (
            patch.object(_paths.sys, "platform", "linux"),
            patch.dict(
                "os.environ",
                {
                    "XDG_CONFIG_HOME": str(self.root / "xdg-config"),
                    "XDG_DATA_HOME": str(self.root / "xdg-data"),
                },
            ),
        ):
            self.assertEqual(
                _paths.preferences_path(),
                self.root / "xdg-config/kilin/preferences.json",
            )
            self.assertEqual(
                _paths.default_workspace_directory(),
                self.home / "Documents/KíLín",
            )
        with (
            patch.object(_paths.sys, "platform", "linux"),
            patch.dict(
                "os.environ",
                {"XDG_CONFIG_HOME": "relative", "XDG_DATA_HOME": "foo"},
            ),
        ):
            self.assertEqual(
                _paths.preferences_path(),
                self.home / ".config/kilin/preferences.json",
            )
            self.assertEqual(
                _paths.default_workspace_directory(),
                self.home / "Documents/KíLín",
            )

    def test_windows_config_and_data_use_distinct_locations(self) -> None:
        """Use roaming preferences and Documents for projects on Windows."""

        with (
            patch.object(_paths.sys, "platform", "win32"),
            patch.dict(
                "os.environ",
                {
                    "APPDATA": str(self.root / "roaming"),
                    "LOCALAPPDATA": str(self.root / "local"),
                },
            ),
        ):
            self.assertEqual(
                _paths.preferences_path(),
                self.root / "roaming/kilin/preferences.json",
            )
            self.assertEqual(
                _paths.default_workspace_directory(),
                self.home / "Documents/KíLín",
            )

    def test_environment_overrides_are_explicit(self) -> None:
        """Allow users to choose writable paths without touching the wheel."""

        with patch.dict(
            "os.environ",
            {
                "KILIN_DEFAULTS_FILE": str(self.root / "prefs.json"),
                "KILIN_WORKSPACES_DIR": str(self.root / "workspaces"),
            },
        ):
            self.assertEqual(
                _paths.preferences_path(), self.root / "prefs.json"
            )
            self.assertEqual(
                _paths.default_workspace_directory(),
                self.root / "workspaces",
            )

    def test_projects_folder_from_settings_is_remembered(self) -> None:
        """Use the chosen folder while it exists; otherwise the standard."""

        preferences = self.root / "prefs.json"
        chosen = self.root / "Desktop/KiLin projects"
        chosen.mkdir(parents=True)
        preferences.write_text(
            json.dumps({"workspace_folder": str(chosen)}), encoding="utf-8"
        )
        standard = self.home / "Documents/KíLín"
        environment = {"KILIN_DEFAULTS_FILE": str(preferences)}
        with (
            patch.dict("os.environ", environment),
            patch.object(_paths.sys, "platform", "darwin"),
        ):
            self.assertEqual(_paths.saved_workspace_folder(), str(chosen))
            self.assertEqual(_paths.default_workspace_directory(), chosen)
            self.assertEqual(_paths.standard_workspace_directory(), standard)
            # A removed or unmounted folder (e.g. a cloud drive) is skipped.
            chosen.rmdir()
            self.assertEqual(_paths.default_workspace_directory(), standard)
            chosen.mkdir()
            # An explicit environment folder still wins.
            override = self.root / "workspaces"
            with patch.dict(
                "os.environ", {"KILIN_WORKSPACES_DIR": str(override)}
            ):
                self.assertEqual(
                    _paths.default_workspace_directory(), override
                )
            # A broken preferences file never stops KíLín from starting.
            preferences.write_text("{", encoding="utf-8")
            self.assertEqual(_paths.saved_workspace_folder(), "")
            self.assertEqual(_paths.default_workspace_directory(), standard)

    def test_relative_environment_paths_are_rejected(self) -> None:
        """Avoid silently writing relative paths under site-packages."""

        with patch.dict(
            "os.environ", {"KILIN_DEFAULTS_FILE": "preferences.json"}
        ):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                _paths.preferences_path()
        with patch.dict("os.environ", {"KILIN_WORKSPACES_DIR": "sandbox"}):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                _paths.default_workspace_directory()

    def test_backend_repair_targets_exact_installed_python(self) -> None:
        """Pip installs the selected extra without assuming the default."""

        python = self.root / "Python Folder/bin/python3.12"
        with (
            patch.object(_paths.sys, "executable", str(python)),
            patch.object(_paths, "find_spec", return_value=object()),
        ):
            for backend in ("mlx", "torch"):
                with self.subTest(backend=backend):
                    command = _paths.gui_install_command(backend)
                    self.assertEqual(
                        _paths.shlex.split(command),
                        [str(python), "-m", "pip", "install",
                         f"kilin[gui,{backend}]"],
                    )
                    self.assertNotIn("kilin[app]", command)

    def test_backend_repair_uses_bundled_uv_and_exact_wheel(
        self,
    ) -> None:
        """The macOS app repairs its own pip-less runtime from its wheel."""

        uv = self.root / "KíLín.app/Contents/Resources/uv"
        uv.parent.mkdir(parents=True)
        uv.write_text("#!/bin/sh\n", encoding="utf-8")
        uv.chmod(0o755)
        python = self.root / "Application Support/runtime/app/bin/python"
        wheel = uv.parent / "kilin-0.1.0-py3-none-any.whl"
        for spec in (str(wheel) + "[app]", "kilin[app]==0.1.0"):
            with (
                self.subTest(spec=spec),
                patch.dict(
                    "os.environ",
                    {
                        _paths.APP_INSTALL_UV: str(uv),
                        _paths.APP_INSTALL_SPEC: spec,
                    },
                ),
                patch.object(_paths.sys, "executable", str(python)),
                patch.object(_paths, "find_spec", return_value=None),
            ):
                for backend in ("mlx", "torch"):
                    command = _paths.gui_install_command(backend)
                    self.assertEqual(
                        _paths.shlex.split(command),
                        [str(uv), "pip", "install", "--python",
                         str(python), spec.replace(
                             "[app]", f"[app,{backend}]"
                         )],
                    )
                    self.assertNotIn(" -m pip ", command)

    def test_backend_repair_uses_uv_without_pip(self) -> None:
        """An ordinary uv venv needs no pip module to be repaired."""

        python = self.root / "venv/bin/python"
        uv = self.root / "bin/uv"
        with (
            patch.object(_paths.sys, "executable", str(python)),
            patch.object(_paths, "find_spec", return_value=None),
            patch.object(_paths.shutil, "which", return_value=str(uv)),
        ):
            self.assertEqual(
                _paths.shlex.split(_paths.gui_install_command("torch")),
                [str(uv), "pip", "install", "--python", str(python),
                 "kilin[gui,torch]"],
            )

    def test_backend_repair_without_installer_is_actionable(self) -> None:
        """Do not suggest python -m pip if no installer is available."""

        with (
            patch.object(_paths, "find_spec", return_value=None),
            patch.object(_paths.shutil, "which", return_value=None),
        ):
            command = _paths.gui_install_command("torch")
            self.assertIn("Install uv (https://", command)
            self.assertIn("uv pip install --python", command)
            self.assertIn("kilin[gui,torch]", command)
            self.assertNotIn("-m pip", command)

    def test_backend_repair_rejects_invalid_backend(self) -> None:
        """Auto is a selection policy, not an installable backend extra."""

        for backend in ("auto", "other", ""):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "mlx.*torch"):
                    _paths.gui_install_command(backend)

    def test_backend_repair_uses_platform_quoting(self) -> None:
        """Windows instructions quote the active executable with spaces."""

        python = r"C:\Program Files\Python312\python.exe"
        with (
            patch.object(_paths.sys, "platform", "win32"),
            patch.object(_paths.sys, "executable", python),
            patch.object(_paths, "find_spec", return_value=object()),
        ):
            self.assertEqual(
                _paths.gui_install_command("torch"),
                r'"C:\Program Files\Python312\python.exe" '
                '-m pip install kilin[gui,torch]',
            )

    def test_checkout_backend_repair_remains_unchanged(self) -> None:
        """A source checkout continues to use explicit uv extras."""

        with patch.object(
            _paths, "_development_checkout", return_value=self.root
        ):
            self.assertEqual(
                _paths.gui_install_command("mlx"),
                "uv sync --extra gui --extra mlx",
            )
            self.assertEqual(
                _paths.gui_install_command("torch"),
                "uv sync --extra gui --extra torch",
            )

    def test_engine_entry_point_uses_resolved_workspace(self) -> None:
        """Inject a user-owned default sandbox into the engine process."""

        class EngineStub:
            """Record the sandbox passed by the engine's entry point."""

            passed: Path | None = None

            def __init__(self, sandbox, emit, **kwargs) -> None:
                """Record the default without creating any workspaces."""

                EngineStub.passed = sandbox
                self.active_trace = None

            def close(self) -> None:
                """Allow engine shutdown without a real backend."""

        destination = self.root / "workspaces"
        with (
            patch.object(
                gui_engine,
                "default_workspace_directory",
                return_value=destination,
            ),
            patch.object(gui_engine, "ChatEngine", EngineStub),
            patch.object(gui_engine.sys, "stdin", []),
        ):
            gui_engine.main()
        self.assertEqual(EngineStub.passed, destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
