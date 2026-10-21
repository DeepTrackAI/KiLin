"""Test the find_passages tool package without live model backends.

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

from kilin.tools import find_passages as tool
from kilin.tools._registry import TOOL_REGISTRY


class TestFindPassages(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        entry = TOOL_REGISTRY["find_passages"]
        self.assertIs(entry.parse_settings, tool.parse_settings)
        self.assertIs(entry.run, tool.run)
        self.assertIsNone(entry.workspace_resources)

    def test_packaged_defaults_and_project_type_overrides(self) -> None:
        """Use the tool-local JSON when settings are not supplied.

        """

        package = Path(tool.__file__).parent
        data = json.loads(
            (package / "defaults.json").read_text(encoding="utf-8")
        )
        readme = (package / "README.md").read_text(encoding="utf-8")
        for section, values in data.items():
            with self.subTest(section=section):
                self.assertIn("`" + section + "`", readme)
            for key in values:
                with self.subTest(section=section, key=key):
                    self.assertIn("`" + key + "`", readme)
        defaults = tool.PassageSettings()
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
            {'batch_tokens': 0},
            {'quick_passages': 4},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_topic_and_judgement_prompts(self) -> None:
        """Generate complete, deterministic model prompts.

        """

        _, prompt = tool.topic_prompt("List papers using deep learning")
        self.assertIn("deep learning", prompt)
        from tests.kilin.tools._shared._support import chunk
        sentences = tool.split_sentences([
            chunk("1", "This sentence has enough words for testing."
                  " Another sentence has enough useful words.")
        ])
        self.assertEqual(len(sentences), 2)
        _, judge = tool.judge_prompt("testing", sentences)
        self.assertIn("testing", judge)

    def test_overlapping_passages_do_not_duplicate_sentences(self) -> None:
        """Deduplicate source sentences across overlapping chunks.

        """

        from tests.kilin.tools._shared._support import chunk
        same = "This sentence clearly contains enough words."
        pieces = [chunk("1", same), chunk("2", same)]
        self.assertEqual(len(tool.split_sentences(pieces)), 1)


if __name__ == "__main__":
    unittest.main()
