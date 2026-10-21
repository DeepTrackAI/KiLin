"""Test the batch_classify tool package without live model backends.

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

from kilin.tools import batch_classify as tool
from kilin.tools._registry import TOOL_REGISTRY


class TestBatchClassify(unittest.TestCase):
    """Verify independent configuration and offline functionality."""

    def test_registry_uses_package_entry_points(self) -> None:
        """Resolve parser and runner to the packaged implementation.

        """

        entry = TOOL_REGISTRY["batch_classify"]
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
        defaults = tool.ClassifySettings()
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
            {'batch_tokens': 100},
            {'categories': ['only_one']},
            {"unknown": True},
        )
        for values in invalid_settings:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_category_validation(self) -> None:
        """Reject malformed categories and preserve valid label order.

        """

        self.assertEqual(
            tool.validated_categories(
                '{"categories": ["method", "result"]}'
            ),
            ("method", "result"),
        )
        self.assertEqual(tool.validated_categories("broken JSON"), ())
        self.assertEqual(
            tool.validated_categories('{"categories": ["only"]}'), ()
        )

    def test_label_validation(self) -> None:
        """Discard labels outside the batch or allowed categories.

        """

        reply = (
            '[{"id": 1, "category": "method", "reason": "yes"},'
            '{"id": 3, "category": "method"}]'
        )
        parsed = tool.parse_labels(reply, [1, 2], ["method", "result"])
        self.assertEqual(set(parsed), {1})
        self.assertEqual(parsed[1].category, "method")


if __name__ == "__main__":
    unittest.main()
