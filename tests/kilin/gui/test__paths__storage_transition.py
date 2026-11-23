"""Verify platform Documents defaults and previous-project discovery.

The tests emulate installed KíLín on each operating system, without
creating a GUI or changing the actual user's project files.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from kilin.gui import _paths
from kilin.gui.engine import ChatEngine


class TestInstalledStorageTransition(unittest.TestCase):
    """Keep defaults visible without losing previously stored projects."""

    def setUp(self) -> None:
        """Isolate filesystem paths and environment for each test."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.home = self.root / "home"
        self.home.mkdir()
        self.home_patch = patch.object(
            _paths.Path, "home", return_value=self.home
        )
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.checkout_patch = patch.object(
            _paths, "_development_checkout", return_value=None
        )
        self.checkout_patch.start()
        self.addCleanup(self.checkout_patch.stop)

    def _legacy(self, platform: str) -> Path:
        """Create a pre-transition project with a valid saved profile.

        Parameters
        ----------
        platform: str
            Platform whose old installed-workspace directory is needed.

        Returns
        -------
        pathlib.Path
            Previously used workspace directory.

        """

        with patch.object(_paths.sys, "platform", platform):
            root = _paths._previous_installed_workspace_directory()
        assert root is not None
        project = root / "existing"
        (project / "sources").mkdir(parents=True)
        (project / "profile.ini").write_text(
            "[project]\ntype_id = code\nsource = sources\n",
            encoding="utf-8",
        )
        return root

    def test_linux_localized_documents_uses_xdg_user_dirs(self) -> None:
        """Use a configured Documents directory, not its English name."""

        config = self.home / ".config"
        config.mkdir()
        (config / "user-dirs.dirs").write_text(
            'XDG_DESKTOP_DIR="$HOME/Desktop"\n'
            'XDG_DOCUMENTS_DIR="$HOME/Dokumente und Texte"\n',
            encoding="utf-8",
        )
        with patch.object(_paths.sys, "platform", "linux"):
            expected = self.home / "Dokumente und Texte/KíLín"
            self.assertEqual(_paths.standard_workspace_directory(), expected)
        self.assertFalse(expected.exists())

    def test_linux_absolute_xdg_and_invalid_fallback(self) -> None:
        """Use an absolute registered path, not relative shell syntax."""

        config = self.root / "config"
        config.mkdir()
        document_root = self.root / "Documents on data disk"
        path = config / "user-dirs.dirs"
        with (
            patch.object(_paths.sys, "platform", "linux"),
            patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config)}),
        ):
            path.write_text(
                f'XDG_DOCUMENTS_DIR="{document_root}"\n',
                encoding="utf-8",
            )
            self.assertEqual(
                _paths.standard_workspace_directory(),
                document_root / "KíLín",
            )
            path.write_text(
                'XDG_DOCUMENTS_DIR="relative/Dokumente"\n',
                encoding="utf-8",
            )
            self.assertEqual(
                _paths.standard_workspace_directory(),
                self.home / "Documents/KíLín",
            )

    def test_windows_redirected_documents_uses_shell_folder(self) -> None:
        """Honor Documents redirection such as OneDrive via the registry."""

        document_root = self.home / "OneDrive/Documents"
        registry = MagicMock()
        registry.HKEY_CURRENT_USER = object()
        registry.REG_SZ = 1
        registry.REG_EXPAND_SZ = 2
        registry.QueryValueEx.return_value = ("%USERPROFILE%/OneDrive", 2)
        registry.ExpandEnvironmentStrings.return_value = str(document_root)
        with (
            patch.object(_paths.sys, "platform", "win32"),
            patch.dict(sys.modules, {"winreg": registry}),
        ):
            self.assertEqual(
                _paths.standard_workspace_directory(),
                document_root / "KíLín",
            )
            registry.OpenKey.assert_called_once()
            registry.QueryValueEx.assert_called_once()
            registry.QueryValueEx.side_effect = OSError("unavailable")
            self.assertEqual(
                _paths.standard_workspace_directory(),
                self.home / "Documents/KíLín",
            )

    def test_legacy_candidate_is_found_on_all_platforms(self) -> None:
        """Detect the old platform data directory without migrating it."""

        for platform in ("darwin", "win32", "linux"):
            with self.subTest(platform=platform):
                root = self._legacy(platform)
                contents = (root / "existing/profile.ini").read_bytes()
                with patch.object(_paths.sys, "platform", platform):
                    self.assertEqual(
                        _paths.available_legacy_workspace_directory(), root
                    )
                    self.assertNotEqual(
                        _paths.standard_workspace_directory(), root
                    )
                self.assertEqual(
                    (root / "existing/profile.ini").read_bytes(), contents
                )
                self.assertFalse((root / "new-project").exists())

    def test_legacy_folder_reuse_discovers_existing_projects(self) -> None:
        """Selecting the previous root preserves engine project discovery."""

        root = self._legacy("darwin")
        events: list[dict[str, object]] = []
        with patch.object(_paths.sys, "platform", "darwin"):
            selected = _paths.available_legacy_workspace_directory()
        self.assertIsNotNone(selected)
        engine = ChatEngine(selected, events.append)
        engine.list_saved_projects("")
        self.assertEqual(events[-1]["event"], "projects")
        self.assertEqual(
            [item["name"] for item in events[-1]["projects"]],
            ["existing"],
        )
        self.assertTrue((root / "existing/profile.ini").is_file())

    def test_legacy_scan_ignores_symlinks_and_invalid_entries(self) -> None:
        """Do not follow project symlinks or report unrelated directories."""

        with patch.object(_paths.sys, "platform", "darwin"):
            root = _paths._previous_installed_workspace_directory()
            assert root is not None
            root.mkdir(parents=True)
            (root / "notes").mkdir()
            self.assertIsNone(
                _paths.available_legacy_workspace_directory()
            )
            actual = self.root / "outside"
            actual.mkdir()
            (actual / "profile.ini").write_text("opaque", encoding="utf-8")
            (root / "linked").symlink_to(actual, target_is_directory=True)
            self.assertIsNone(
                _paths.available_legacy_workspace_directory()
            )
            old_root = root.rename(self.root / "moved")
            root.symlink_to(old_root, target_is_directory=True)
            self.assertIsNone(
                _paths.available_legacy_workspace_directory()
            )

    def test_explicit_override_and_checkout_skip_legacy(self) -> None:
        """Never override deliberate workspace selection or a checkout."""

        root = self._legacy("darwin")
        with patch.object(_paths.sys, "platform", "darwin"):
            with patch.dict(
                os.environ,
                {"KILIN_WORKSPACES_DIR": str(self.root / "chosen")},
            ):
                self.assertIsNone(
                    _paths.available_legacy_workspace_directory()
                )
            with patch.object(
                _paths, "_development_checkout", return_value=self.root
            ):
                self.assertIsNone(
                    _paths.available_legacy_workspace_directory()
                )
            with patch.object(
                _paths, "saved_workspace_folder", return_value=str(root)
            ):
                self.assertEqual(_paths.default_workspace_directory(), root)

    def test_legacy_directory_not_created_by_discovery(self) -> None:
        """Read-only inspection does not construct directories or profiles."""

        with patch.object(_paths.sys, "platform", "darwin"):
            root = _paths._previous_installed_workspace_directory()
            self.assertIsNotNone(root)
            self.assertIsNone(
                _paths.available_legacy_workspace_directory()
            )
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
