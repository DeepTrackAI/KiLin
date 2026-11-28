"""Verify captured project-type definitions and workspace availability.

Use isolated definitions and headless GUI operations without loading models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import kilin
from kilin.gui.engine import ChatEngine
from kilin.project_behavior import (
    effective_project_type, project_behavior_settings, save_project_behavior,
)
from kilin.tools import ToolUse
from tests.kilin._project_type_support import _type_data


_types = importlib.import_module("kilin.project_type")


class TestProjectTypeCatalog(unittest.TestCase):
    """Keep runtime discovery, explicit validation and opening separate."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.types = self.root / "types"
        self.types.mkdir()
        self.workspaces = self.root / "workspaces"
        self.workspaces.mkdir()
        _types.load_project_types.cache_clear()
        self.addCleanup(_types.load_project_types.cache_clear)
        for name, value in (
            ("PROJECT_TYPES_DIR", self.types), ("_catalog", None),
        ):
            patcher = patch.object(_types, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _write(self, name, value):
        path = self.types / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _missing_tool_type(self):
        raw = _type_data()
        raw["intents"] = {
            name: {"description": name, "tool": {"name": name}}
            for name in ("not_installed", "also_missing")
        }
        return raw

    def test_isolate_failed_definitions_and_validate_explicitly(self):
        self._write("alpha", _type_data())
        (self.types / "broken.json").write_text("{", encoding="utf-8")
        self._write("missing-tools", self._missing_tool_type())
        self._write("bad-version", {**_type_data(), "schema_version": 2})
        self._write("bad-intent", {**_type_data(), "default_intent": []})
        self._write("zeta", {**_type_data(), "schema_version": 1})
        catalog = kilin.project_type_catalog()
        self.assertEqual(list(catalog), sorted(catalog))
        self.assertEqual(kilin.project_type_ids(), ("alpha", "zeta"))
        self.assertEqual(catalog["alpha"].origin, "distribution")
        self.assertEqual(catalog["alpha"].path, self.types / "alpha.json")
        self.assertEqual(catalog["alpha"].schema_version, 1)
        self.assertEqual(catalog["bad-version"].schema_version, 2)
        self.assertIn("schema_version", catalog["bad-version"].error)
        self.assertIn("not valid JSON", catalog["broken"].error)
        self.assertIn("default_intent", catalog["bad-intent"].error)
        missing = catalog["missing-tools"]
        self.assertFalse(missing.available)
        self.assertEqual(missing.required_tools,
                         ("also_missing", "not_installed"))
        for tool in missing.required_tools:
            self.assertIn(tool, missing.error)
        with self.assertRaisesRegex(ValueError, "Required tools unavailable"):
            kilin.project_type("missing-tools")
        with self.assertRaisesRegex(ValueError, "Unknown project type"):
            kilin.project_type("removed")
        with self.assertRaises(ValueError):
            _types.load_project_types(self.types)
        self.assertEqual(kilin.project_type_ids(), ("alpha", "zeta"))

    def test_schema_version_is_independent_and_strict(self):
        for index, value in enumerate((True, False, None, 1.0, "1", 0, 2)):
            with self.subTest(value=value):
                self._write(f"bad-{index}", {
                    **_type_data(), "schema_version": value,
                })
        self._write("good", {**_type_data(), "schema_version": 1})
        catalog = kilin.project_type_catalog()
        self.assertEqual(kilin.PROJECT_TYPE_SCHEMA_VERSION, 1)
        self.assertEqual(kilin.project_type_ids(), ("good",))
        self.assertTrue(all(not info.available for key, info in catalog.items()
                            if key != "good"))

    def test_nested_copies_and_source_snapshot(self):
        @dataclass(frozen=True)
        class Settings:
            categories: list[str]

        raw = _type_data()
        raw["intents"] = {"tool": {
            "description": "Use tool", "tool": {"name": "example"},
        }}
        path = self._write("alpha", raw)
        with (
            patch.object(_types, "registered_tools",
                         return_value={"example": object()}),
            patch.object(_types, "parse_tool", return_value=ToolUse(
                "example", Settings(["original"]),
            )),
        ):
            catalog = kilin.project_type_catalog()
        with self.assertRaises(TypeError):
            catalog["other"] = catalog["alpha"]
        with self.assertRaises(FrozenInstanceError):
            catalog["alpha"].origin = "user"
        available = _types.load_project_types()
        with self.assertRaises(TypeError):
            available["other"] = available["alpha"]
        copy = available["alpha"]
        copy.intents["tool"].tool.settings.categories.append("mutated")
        copy.intents.clear()
        captured = kilin.project_type_raw("alpha")
        captured["intents"].clear()
        path.unlink()
        _types.load_project_types.cache_clear()
        self.assertEqual(kilin.project_type_raw("alpha"), raw)
        self.assertEqual(
            kilin.project_type("alpha").intents["tool"].tool.settings,
            Settings(["original"]),
        )
        self.assertEqual(kilin.project_type_ids(), ("alpha",))

    def test_single_catalog_initialization_across_threads(self):
        self._write("alpha", _type_data())
        with patch.object(
            _types, "_build_project_type_catalog",
            wraps=_types._build_project_type_catalog,
        ) as builder:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(
                    lambda _: kilin.project_type_ids(), range(16),
                ))
        self.assertEqual(results, [("alpha",)] * 16)
        builder.assert_called_once()

    def test_custom_source_overrides_survive_definition_file_removal(self):
        raw = _type_data()
        path = self._write("custom-lab", raw)
        project = kilin.initialize_project(
            self.workspaces / "custom", "custom-lab",
        )
        self.assertEqual(project.profile.type_id, "custom-lab")
        self.assertEqual(project.profile.max_file_bytes, 4096)
        self.assertTrue((project.root / "templates").is_dir())
        path.unlink()
        payload = project_behavior_settings(project.root)
        values = deepcopy(payload["global"]["values"])
        values["description"] = "Local override"
        updated = save_project_behavior(
            project.root, "global", values,
            expected_revision=payload["revision"],
        )
        self.assertEqual(effective_project_type(project.root).description,
                         "Local override")
        self.assertEqual(kilin.open_project(project.root), project)
        save_project_behavior(
            project.root, "global", updated["global"]["defaults"],
            expected_revision=updated["revision"],
        )
        self.assertEqual(effective_project_type(project.root).description,
                         raw["description"])

    def test_discovery_and_preflight_preserve_unavailable_workspaces(self):
        for name in ("good", "removed", "needs-tools"):
            self._write(name, _type_data())
        for name in ("good", "removed", "needs-tools"):
            kilin.initialize_project(self.workspaces / name, name)
        # Simulate the next process after a plugin is removed or broken.
        (self.types / "removed.json").unlink()
        self._write("needs-tools", self._missing_tool_type())
        _types._catalog = None
        _types.load_project_types.cache_clear()
        for index, name in enumerate(("good", "removed", "needs-tools")):
            path = self.workspaces / name / "profile.ini"
            os.utime(path, (100 + index, 100 + index))
        before = {
            p.relative_to(self.workspaces): p.read_bytes()
            for p in self.workspaces.rglob("*") if p.is_file()
        }
        events = []
        engine = ChatEngine(self.workspaces, events.append)
        engine.list_saved_projects("")
        rows = events[-1]["projects"]
        self.assertEqual([row["type_id"] for row in rows],
                         ["needs-tools", "removed", "good"])
        self.assertFalse(rows[0]["available"])
        self.assertIn("not_installed", rows[0]["unavailable_reason"])
        self.assertFalse(rows[1]["available"])
        self.assertIn("not installed", rows[1]["unavailable_reason"])
        self.assertTrue(rows[2]["available"])
        self.assertIsNone(rows[2]["unavailable_reason"])
        engine.startup()
        self.assertEqual(events[-1]["type_id"], "good")
        engine.find_saved_project("removed", "")
        self.assertIsNone(events[-1]["project"])
        previous = object()
        engine.session = previous
        for name in ("removed", "needs-tools"):
            target = self.workspaces / name
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "project.type_id"):
                    kilin.open_project(target)
                with (
                    patch.object(engine, "close") as close,
                    patch.object(engine, "_activate_project") as activate,
                    patch.object(engine, "_open_conversation_store") as chat,
                ):
                    engine.resume_project(str(target))
                close.assert_not_called()
                activate.assert_not_called()
                chat.assert_not_called()
                self.assertIs(engine.session, previous)
                self.assertEqual(events[-1]["event"], "error")
                self.assertFalse(events[-1]["resume_released"])
                self.assertIn(name, events[-1]["message"])
        new = self.workspaces / "new"
        with self.assertRaisesRegex(ValueError, "Required tools unavailable"):
            kilin.initialize_project(new, "needs-tools")
        self.assertFalse(new.exists())
        self.assertEqual(before, {
            p.relative_to(self.workspaces): p.read_bytes()
            for p in self.workspaces.rglob("*") if p.is_file()
        })

    def test_all_unavailable_does_not_query_model_settings(self):
        self._write("missing-tools", self._missing_tool_type())
        events = []
        engine = ChatEngine(self.workspaces, events.append)
        with patch("kilin.gui.engine.downloaded_models") as models:
            engine.settings("", "")
        models.assert_not_called()
        self.assertEqual(events[-1]["event"], "settings")
        self.assertEqual(events[-1]["project_types"], [])
        self.assertIsNone(events[-1]["profile"])
        self.assertIn("No project types", events[-1]["project_type_error"])

    def test_stale_creation_selection_returns_available_choices(self):
        self._write("good", _type_data())
        events = []
        engine = ChatEngine(self.workspaces, events.append)
        with patch("kilin.gui.engine.downloaded_models") as models:
            engine.settings("removed", "")
        models.assert_not_called()
        self.assertEqual(events[-1]["event"], "settings")
        self.assertIsNone(events[-1]["profile"])
        self.assertEqual([item["type_id"] for item in
                          events[-1]["project_types"]], ["good"])
        self.assertIn("removed", events[-1]["project_type_error"])
