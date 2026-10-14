"""Test the paper_overview tool package without live model backends.

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

from qachat.tools import paper_overview as tool
from qachat.tools._registry import TOOL_REGISTRY


class TestPaperOverview(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        entry = TOOL_REGISTRY["paper_overview"]
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
        defaults = tool.OverviewSettings()
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
            {'max_tokens': 0},
            {'answer_tokens': 30},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")


    def test_overview_prompt_preserves_evidence_and_guidance(self) -> None:
        """Include numbered passages and requested length constraints.

        """

        from tests.qachat.tools._shared._support import chunk
        _, prompt = tool.overview_prompt(
            "Summarize findings", "paper.pdf",
            [(3, chunk("a", "A result of interest."))],
            "Two sentences",
        )
        self.assertIn("[3] A result", prompt)
        self.assertIn("Two sentences", prompt)


if __name__ == "__main__":
    unittest.main()
