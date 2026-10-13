"""Test the project_inventory tool package without live model backends.

Check configuration ownership, public registry entries, settings validation,
and deterministic execution behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import unittest

from qachat.tools import project_inventory as tool
from qachat.tools._registry import TOOL_REGISTRY


class TestProjectInventory(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        entry = TOOL_REGISTRY["project_inventory"]
        self.assertIs(entry.parse_settings, tool.parse_settings)
        self.assertIs(entry.run, tool.run)
        self.assertIsNone(entry.workspace_resources)

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        package = Path(tool.__file__).parent
        data = json.loads((package / "defaults.json").read_text())
        readme = (package / "README.md").read_text()
        for section, values in data.items():
            with self.subTest(section=section):
                self.assertIn("`" + section + "`", readme)
            for key in values:
                with self.subTest(section=section, key=key):
                    self.assertIn("`" + key + "`", readme)
        defaults = tool.InventorySettings()
        self.assertEqual(defaults, tool.parse_settings({}, "example"))
        self.assertEqual(
            {field.name for field in fields(defaults)},
            {name.lower() for name in data["SETTINGS"]},
        )
        for field in fields(defaults):
            expected = data["SETTINGS"][field.name.upper()]
            if isinstance(expected, list):
                expected = tuple(expected)
            self.assertEqual(getattr(defaults, field.name), expected)

    def test_invalid_project_type_settings(self) -> None:
        """Reject out-of-range, mistyped and unknown overrides.

        """

        invalid_settings = (
            {'max_folders': 0},
            {'max_listed': 2001},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")


    def test_type_recognition_and_inventory_without_a_model(self) -> None:
        """Count files from core metadata without invoking a model.

        """

        from pathlib import Path
        from unittest.mock import Mock
        from qachat.tools._contract import ToolRequest
        from tests.qachat.tools._support import chunk

        self.assertEqual(tool.file_types_in("List PDFs"), (".pdf",))
        self.assertEqual(tool.file_types_in("text files"), (".txt",))
        context = Mock()
        context.source_files.return_value = {
            Path("a.pdf"): 123,
            Path("b.md"): 456,
        }
        context.chunks = [chunk("one", "A useful paper", path="a.pdf")]
        result = tool.run(
            tool.InventorySettings(),
            ToolRequest("list all PDFs", "list all PDFs"),
            context,
        )
        self.assertIn("a.pdf", result.answer)
        self.assertIn("PDF", result.answer)
        context.ask_model.assert_not_called()
        context.done.assert_called_once()


if __name__ == "__main__":
    unittest.main()
