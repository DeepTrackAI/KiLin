"""Test the shared loader for QAChat's packaged configuration defaults.

Tests use temporary JSON files and require no model, GUI, or network access.
They verify cached reads, defensive section copies, and strict type checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qachat import defaults


class TestPackagedDefaultsLoader(unittest.TestCase):
    """Test JSON shape validation and safe sharing of cached sections."""

    def setUp(self) -> None:
        """Create a disposable config path and an isolated reader cache."""

        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "qachat_defaults.json"
        self.path_patch = patch.object(defaults, "_DEFAULTS_PATH", self.path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        defaults._packaged_defaults.cache_clear()
        self.addCleanup(defaults._packaged_defaults.cache_clear)

    def _write(self, payload: object) -> None:
        """Write JSON and clear the per-process loader cache.

        Parameters
        ----------
        payload: object
            JSON-compatible value used as the complete config file.

        """

        self.path.write_text(json.dumps(payload), encoding="utf-8")
        defaults._packaged_defaults.cache_clear()

    def test_caches_file_read_and_returns_independent_sections(self) -> None:
        """Load once while preventing callers from mutating cached lists."""

        self._write({"REPOSITORY": {"SKIP_FILES": ["first"]}})
        first = defaults.load_defaults_section("REPOSITORY")
        first["SKIP_FILES"].append("changed")
        self.path.write_text(
            '{"REPOSITORY": {"SKIP_FILES": ["second"]}}',
            encoding="utf-8",
        )
        second = defaults.load_defaults_section("REPOSITORY")

        self.assertEqual(second, {"SKIP_FILES": ["first"]})
        self.assertIsNot(first, second)
        self.assertIsNot(first["SKIP_FILES"], second["SKIP_FILES"])

    def test_missing_section_or_nonobject_section_is_rejected(self) -> None:
        """Fail when required configuration sections are unavailable."""

        for section in (None, [], "text"):
            with self.subTest(section=section):
                self._write({"ANSWERING": section})
                with self.assertRaisesRegex(ValueError, "ANSWERING"):
                    defaults.load_defaults_section("ANSWERING")

        self._write({"ANSWERING": {}})
        with self.assertRaisesRegex(ValueError, "REPOSITORY"):
            defaults.load_defaults_section("REPOSITORY")

    def test_bad_json_and_nonobject_root_are_rejected(self) -> None:
        """Reject malformed JSON and invalid root values explicitly."""

        self.path.write_text("{invalid", encoding="utf-8")
        defaults._packaged_defaults.cache_clear()
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            defaults.load_defaults_section("ANSWERING")

        for payload in ([], "text", None):
            with self.subTest(payload=payload):
                self._write(payload)
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    defaults.load_defaults_section("ANSWERING")

    def test_missing_file_propagates_io_error(self) -> None:
        """Do not conceal missing packaged resources with fake defaults."""

        with self.assertRaises(FileNotFoundError):
            defaults.load_defaults_section("ANSWERING")


class TestDefaultValueValidation(unittest.TestCase):
    """Test typed validation used by modules owning default sections."""

    def test_integer_rejects_boolean_float_zero_and_missing_value(
        self,
    ) -> None:
        """Only positive integers are valid for numeric routing limits."""

        self.assertEqual(
            defaults.require_default_int({"MAX": 3}, "MAX", "ANSWERING"),
            3,
        )
        for value in (True, False, 0, -2, 1.5, "3", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "ANSWERING.MAX"
                ):
                    defaults.require_default_int(
                        {"MAX": value}, "MAX", "ANSWERING"
                    )
        with self.assertRaisesRegex(ValueError, "ANSWERING.MAX"):
            defaults.require_default_int({}, "MAX", "ANSWERING")

    def test_string_requires_nonempty_text(self) -> None:
        """Reject missing or blank model identifiers and regex strings."""

        self.assertEqual(
            defaults.require_default_string(
                {"MODEL": "example/model"}, "MODEL", "DEFAULT_MODELS"
            ),
            "example/model",
        )
        for value in (None, 1, "", " \t"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "DEFAULT_MODELS.MODEL"
                ):
                    defaults.require_default_string(
                        {"MODEL": value}, "MODEL", "DEFAULT_MODELS"
                    )

    def test_string_list_checks_container_and_items(self) -> None:
        """Return ordered immutable strings and reject malformed arrays."""

        self.assertEqual(
            defaults.require_default_strings(
                {"FILES": [".py", ".md"]}, "FILES", "REPOSITORY"
            ),
            (".py", ".md"),
        )
        for value in (None, [], [""], ["  "], [12], "one"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "REPOSITORY.FILES"
                ):
                    defaults.require_default_strings(
                        {"FILES": value}, "FILES", "REPOSITORY"
                    )

    def test_regex_is_compiled_with_configured_flags(self) -> None:
        """Preserve the regex semantics and report invalid syntax."""

        expression = defaults.require_default_pattern(
            {"CUE": r"\bhello\b"}, "CUE", "ANSWERING", flags=re.I
        )
        self.assertIsNotNone(expression.search("HELLO"))
        for value in ("(", "", False, 3):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "ANSWERING.CUE"
                ):
                    defaults.require_default_pattern(
                        {"CUE": value}, "CUE", "ANSWERING"
                    )

    def test_vocabulary_is_validated_and_immutable(self) -> None:
        """Validate lowercase common words independently of the router."""

        self.assertEqual(
            defaults.require_default_lowercase_words(
                {"WORDS": ["about", "after", "about"]},
                "WORDS",
                "ANSWERING",
            ),
            frozenset({"about", "after"}),
        )
        for value in ([], ["Word"], ["two words"], ["  word"], [9]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "ANSWERING.WORDS"
                ):
                    defaults.require_default_lowercase_words(
                        {"WORDS": value}, "WORDS", "ANSWERING"
                    )


class TestDefaultDocumentation(unittest.TestCase):
    """Test that package-wide settings are accounted for in the guide."""

    def test_every_default_key_is_named_in_technical_guide(self) -> None:
        """Require a documented entry when another setting is introduced."""

        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "qachat" / "qachat_defaults.json").read_text(
                encoding="utf-8"
            )
        )
        guide = (root / "docs" / "package_defaults.md").read_text(
            encoding="utf-8"
        )
        for section, settings in config.items():
            with self.subTest(section=section):
                self.assertIn(f"`{section}`", guide)
            for key in settings:
                with self.subTest(section=section, key=key):
                    self.assertIn(f"`{key}`", guide)



if __name__ == "__main__":
    unittest.main()
