"""Test GUI configuration ownership and sparse user preference overrides.

All tests use temporary user files. Packaged resources and existing project
profiles must remain unchanged after an application-wide preference update.

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

from qachat.gui import preferences
from qachat.project import (
    default_project_profile,
    initialize_project,
    open_project,
)


class TestGUIPreferences(unittest.TestCase):
    """Preserve immutable defaults and validate sparse user overrides."""

    def setUp(self) -> None:
        """Create an isolated location for writable preferences."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "config" / "preferences.json"
        self.override = patch.dict(
            "os.environ", {"QACHAT_DEFAULTS_FILE": str(self.path)}
        )
        self.override.start()
        self.addCleanup(self.override.stop)
        self.packaged_bytes = preferences.DEFAULTS_FILE.read_bytes()

    def tearDown(self) -> None:
        """Verify that no test operation changed packaged defaults."""

        self.assertEqual(
            preferences.DEFAULTS_FILE.read_bytes(), self.packaged_bytes
        )

    def test_missing_preferences_preserve_packaged_defaults(self) -> None:
        """Do not create preferences merely by reading defaults."""

        self.assertEqual(preferences.preferences_file(), self.path)
        self.assertEqual(
            preferences.app_defaults(), preferences.built_in_defaults()
        )
        self.assertEqual(
            preferences.default_search_depth(), [5, 8, 10, 15, 20]
        )
        self.assertFalse(self.path.exists())
        self.assertEqual(
            preferences.defaults_file(), preferences.DEFAULTS_FILE
        )

    def test_write_and_reload_sparse_overrides(self) -> None:
        """Persist only changed backends and keep the original profile."""

        original_profile = default_project_profile("code")
        saved = preferences.write_defaults(
            {
                "appearance": "dark",
                "search_depth": [1, 2, 3, 4, 5],
                "router_models": {"mlx": "Qwen/Qwen3-8B"},
            }
        )
        self.assertEqual(saved["appearance"], "dark")
        self.assertEqual(preferences.default_appearance(), "dark")
        self.assertEqual(preferences.default_search_depth(), [1, 2, 3, 4, 5])
        self.assertEqual(
            preferences.default_router_model("mlx"), "Qwen/Qwen3-8B"
        )
        self.assertEqual(
            preferences.default_router_model("torch"),
            preferences.built_in_defaults()["router_models"]["torch"],
        )
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8")),
            {
                "appearance": "dark",
                "search_depth": [1, 2, 3, 4, 5],
                "router_models": {"mlx": "Qwen/Qwen3-8B"},
            },
        )
        self.assertEqual(
            default_project_profile("code"), original_profile
        )

    def test_restoring_defaults_removes_only_matching_overrides(self) -> None:
        """Do not erase unrelated changes when a default is restored."""

        preferences.write_defaults(
            {"appearance": "dark", "query_format": "plain"}
        )
        preferences.write_defaults({"appearance": "system"})
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8")),
            {"query_format": "plain"},
        )
        preferences.write_defaults({"query_format": "qwen3"})
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8")), {}
        )

    def test_preferences_can_override_one_backend(self) -> None:
        """Retain other backend models while changing one preference."""

        self.path.parent.mkdir(parents=True)
        self.path.write_text(
            json.dumps({"embedding_models": {"mlx": "Qwen/Qwen3-4B"}}),
            encoding="utf-8",
        )
        effective = preferences.app_defaults()
        self.assertEqual(
            effective["embedding_models"]["mlx"], "Qwen/Qwen3-4B"
        )
        self.assertEqual(
            effective["embedding_models"]["torch"],
            preferences.built_in_defaults()["embedding_models"]["torch"],
        )

    def test_invalid_preferences_are_not_silently_replaced(self) -> None:
        """Preserve malformed data and identify the offending setting."""

        invalid = (
            "not JSON",
            "[]",
            '{"unrecognized": true}',
            '{"router_models": {"invalid": "Qwen/Qwen3-4B"}}',
            '{"search_depth": [0, 1, 2, 3, 4]}',
        )
        self.path.parent.mkdir(parents=True)
        for contents in invalid:
            with self.subTest(contents=contents):
                self.path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    preferences.app_defaults()
                with self.assertRaises(ValueError):
                    preferences.write_defaults({"appearance": "dark"})
                self.assertEqual(
                    self.path.read_text(encoding="utf-8"), contents
                )

    def test_invalid_update_does_not_create_or_mutate_file(self) -> None:
        """Validate all changed entries before any filesystem operation."""

        with self.assertRaises(ValueError):
            preferences.write_defaults(
                {"appearance": "dark", "search_depth": [0]}
            )
        self.assertFalse(self.path.parent.exists())
        preferences.write_defaults({"appearance": "dark"})
        original = self.path.read_bytes()
        with self.assertRaises(ValueError):
            preferences.write_defaults({"router_models": {"bad": "id"}})
        self.assertEqual(self.path.read_bytes(), original)

    def test_packaged_file_and_alias_cannot_be_write_targets(self) -> None:
        """Prevent direct and symbolic-link writes to shipped resources."""

        with self.assertRaisesRegex(ValueError, "read-only"):
            preferences.write_defaults(
                {"appearance": "dark"}, preferences.DEFAULTS_FILE
            )
        link = self.root / "packaged.json"
        try:
            link.symlink_to(preferences.DEFAULTS_FILE)
        except (NotImplementedError, OSError):
            self.skipTest("Symlinks unavailable")
        with self.assertRaisesRegex(ValueError, "read-only"):
            preferences.write_defaults({"appearance": "dark"}, link)

    def test_write_failure_preserves_previous_preferences(self) -> None:
        """Use the existing atomic writer and propagate I/O failures."""

        preferences.write_defaults({"appearance": "dark"})
        original = self.path.read_bytes()
        with patch.object(
            preferences,
            "atomic_write_text",
            side_effect=PermissionError("read-only"),
        ):
            with self.assertRaises(PermissionError):
                preferences.write_defaults({"appearance": "light"})
        self.assertEqual(self.path.read_bytes(), original)

    def test_returned_settings_do_not_mutate_cache(self) -> None:
        """Never expose a mutable reference to the cached base config."""

        values = preferences.built_in_defaults()
        values["search_depth"].append(100)
        self.assertEqual(
            preferences.built_in_defaults()["search_depth"],
            [5, 8, 10, 15, 20],
        )
        self.assertEqual(
            preferences.default_search_depth(), [5, 8, 10, 15, 20]
        )

    def test_existing_profile_survives_global_changes(self) -> None:
        """Leave saved profile bytes untouched when global choices change."""

        project_root = self.root / "existing_project"
        project_root.mkdir()
        original = initialize_project(project_root, "code")
        profile_path = project_root / "profile.ini"
        before = profile_path.read_bytes()
        preferences.write_defaults(
            {
                "router_models": {"mlx": "Qwen/Qwen3-8B"},
                "appearance": "dark",
                "search_depth": [1, 2, 3, 4, 5],
            }
        )
        after = open_project(project_root)
        self.assertEqual(after.profile, original.profile)
        self.assertEqual(profile_path.read_bytes(), before)

    def test_empty_update_does_not_write(self) -> None:
        """Do not create a preference file for a no-op update."""

        self.assertEqual(
            preferences.write_defaults({}), preferences.built_in_defaults()
        )
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
