"""Verify development and installed GUI data-directory resolution.

Tests simulate source checkouts, installed packages, and platform-specific
user locations without requiring Qt, GPU dependencies, or real user writes.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

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
            self.assertEqual(
                workspace,
                self.home / "Library/Application Support/kilin/workspaces",
            )
            self.assertEqual(
                _paths.preferences_path(),
                self.home / "Library/Application Support/kilin/"
                "preferences.json",
            )
        self.assertFalse(workspace.exists())
        self.assertFalse(self.home.exists())

    def test_linux_xdg_directories_and_relative_fallback(self) -> None:
        """Use XDG absolute paths and ignore relative XDG locations."""

        with patch.object(_paths.sys, "platform", "linux"), patch.dict(
            "os.environ",
            {
                "XDG_CONFIG_HOME": str(self.root / "xdg-config"),
                "XDG_DATA_HOME": str(self.root / "xdg-data"),
            },
        ):
            self.assertEqual(
                _paths.preferences_path(),
                self.root / "xdg-config/kilin/preferences.json",
            )
            self.assertEqual(
                _paths.default_workspace_directory(),
                self.root / "xdg-data/kilin/workspaces",
            )
        with patch.object(_paths.sys, "platform", "linux"), patch.dict(
            "os.environ",
            {"XDG_CONFIG_HOME": "relative", "XDG_DATA_HOME": "foo"},
        ):
            self.assertEqual(
                _paths.preferences_path(),
                self.home / ".config/kilin/preferences.json",
            )
            self.assertEqual(
                _paths.default_workspace_directory(),
                self.home / ".local/share/kilin/workspaces",
            )

    def test_windows_config_and_data_use_distinct_locations(self) -> None:
        """Use roaming preferences and local workspace data on Windows."""

        with patch.object(_paths.sys, "platform", "win32"), patch.dict(
            "os.environ",
            {
                "APPDATA": str(self.root / "roaming"),
                "LOCALAPPDATA": str(self.root / "local"),
            },
        ):
            self.assertEqual(
                _paths.preferences_path(),
                self.root / "roaming/kilin/preferences.json",
            )
            self.assertEqual(
                _paths.default_workspace_directory(),
                self.root / "local/kilin/workspaces",
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

    def test_relative_environment_paths_are_rejected(self) -> None:
        """Avoid silently writing relative paths under site-packages."""

        with patch.dict(
            "os.environ", {"KILIN_DEFAULTS_FILE": "preferences.json"}
        ):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                _paths.preferences_path()
        with patch.dict(
            "os.environ", {"KILIN_WORKSPACES_DIR": "sandbox"}
        ):
            with self.assertRaisesRegex(ValueError, "absolute path"):
                _paths.default_workspace_directory()

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
        with patch.object(
            gui_engine, "default_workspace_directory", return_value=destination
        ), patch.object(
            gui_engine, "ChatEngine", EngineStub
        ), patch.object(gui_engine.sys, "stdin", []):
            gui_engine.main()
        self.assertEqual(EngineStub.passed, destination)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
