"""Test the show_source tool package without live model backends.

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

from qachat.tools import show_source as tool
from qachat.tools._registry import TOOL_REGISTRY


class TestShowSource(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        entry = TOOL_REGISTRY["show_source"]
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
        defaults = tool.SourceSettings()
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
            {'max_passages': 0},
            {'context_sentences': 11},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")


    def test_choose_linked_evidence(self) -> None:
        """Prefer linked evidence over unlinked recent evidence.

        """

        from pathlib import Path
        from qachat.tools._contract import EarlierTurn
        recent = EarlierTurn(
            3, "recent", paths=(Path("a.md"),),
            texts=("a",), linked=False,
        )
        linked = EarlierTurn(
            2, "linked", paths=(Path("b.md"),),
            texts=("b",), linked=True,
        )
        self.assertEqual(tool.newest_listed([recent, linked]), linked)
        self.assertIsNone(tool.newest_listed([]))

    def test_explanation_prompt_includes_exact_text(self) -> None:
        """Preserve the cited passage and its source label.

        """

        system, prompt = tool.explain_prompt(
            "Explain this", [(1, "paper.md", ">>Important<<")]
        )
        self.assertIn("Use only the text shown", system)
        self.assertIn("paper.md", prompt)
        self.assertIn(">>Important<<", prompt)


if __name__ == "__main__":
    unittest.main()
