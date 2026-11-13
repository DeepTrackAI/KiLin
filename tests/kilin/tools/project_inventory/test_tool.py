"""Test the project_inventory tool package without live model backends.

Check configuration ownership, public registry entries, settings validation,
and deterministic execution behavior.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.tools import project_inventory as tool

from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)


class TestProjectInventory(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        assert_registry_entry(
            self, "project_inventory", tool
        )

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        assert_packaged_defaults(
            self, tool, tool.InventorySettings,
        )

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
        from kilin.tools._contract import ToolRequest
        from tests.kilin.tools._shared._support import chunk

        self.assertEqual(tool.file_types_in("List PDFs"), (".pdf",))
        self.assertEqual(tool.file_types_in("text files"), (".txt",))
        self.assertEqual(
            tool.file_types_in("How many Word documents?"), (".doc", ".docx")
        )
        self.assertEqual(
            tool.file_types_in("List Word docs"), (".doc", ".docx")
        )
        self.assertEqual(
            tool.file_types_in("List Word doc"), (".doc", ".docx")
        )
        self.assertEqual(tool.file_types_in("List the .docx"), (".docx",))
        self.assertEqual(tool.file_types_in("List the .doc"), (".doc",))
        self.assertEqual(tool.file_types_in("Find the word MRI"), ())
        self.assertEqual(tool.file_types_in("list the LaTeX files"),
                         (".tex",))
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
