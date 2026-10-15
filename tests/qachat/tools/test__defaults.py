"""Test independent tool-default loading and type/range validation.

Cover the private, general-purpose loader separately from the
paper-summary package, including caching and defensive copies.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.tools._defaults import (
    load_tool_defaults,
    require_int,
    require_int_pair,
    require_keys,
    require_number,
    require_section,
    require_string,
    require_words,
)


class TestToolDefaults(unittest.TestCase):
    """Keep independently packaged defaults strict and immutable."""

    def setUp(self) -> None:
        """Create an isolated tool directory.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "defaults.json"

    def _write(self, data: object) -> None:
        """Write temporary packaged defaults.

        Parameters
        ----------
        data: object
            JSON-compatible value to write.

        """

        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_loaded_defaults_are_independent_and_cached(self) -> None:
        """Retain original values after mutation or removal of the file.

        """

        self._write({"NESTED": {"words": ["first", "second"]}})
        first = load_tool_defaults(self.path)
        first["NESTED"]["words"].append("third")
        self.path.unlink()
        second = load_tool_defaults(self.path)
        self.assertEqual(
            second, {"NESTED": {"words": ["first", "second"]}}
        )
        self.assertIsNot(first, second)
        self.assertIsNot(first["NESTED"], second["NESTED"])

    def test_invalid_json_and_root_shape(self) -> None:
        """Report corrupt or non-object package defaults.

        """

        self.path.write_text("{bad", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            load_tool_defaults(self.path)

        self._write(["not", "an object"])
        with self.assertRaisesRegex(ValueError, "JSON object"):
            load_tool_defaults(self.path)

    def test_missing_defaults_file(self) -> None:
        """Treat missing configuration files as packaging failures.

        """

        with self.assertRaises(OSError):
            load_tool_defaults(self.path)

    def test_required_keys_and_sections(self) -> None:
        """Reject unknown and missing keys without dropping either.

        """

        valid = {"SETTINGS": {"size": 8}}
        require_keys(valid, {"SETTINGS"}, "example")
        self.assertEqual(
            require_section(valid, "SETTINGS", "example"),
            {"size": 8},
        )
        for data in ({"A": 1}, {"SETTINGS": 1}, {}):
            with self.subTest(data=data):
                if "SETTINGS" in data:
                    with self.assertRaisesRegex(ValueError, "object"):
                        require_section(data, "SETTINGS", "example")
                else:
                    with self.assertRaises(ValueError):
                        require_keys(data, {"SETTINGS"}, "example")

    def test_required_integer_bounds(self) -> None:
        """Reject bools, floats and out-of-range integers.

        """

        self.assertEqual(require_int({"x": 1}, "x", "example", 0, 3), 1)
        for value in (True, False, -1, 4, 1.0, "2", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "example.x"):
                    require_int({"x": value}, "x", "example", 0, 3)

    def test_required_finite_number(self) -> None:
        """Reject bools, infinities, NaN and invalid numeric ranges.

        """

        self.assertEqual(
            require_number({"x": 1}, "x", "example", 0, 2), 1.0
        )
        for value in (True, -0.1, 3, float("inf"), float("nan"), "1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "example.x"):
                    require_number({"x": value}, "x", "example", 0, 2)

    def test_required_strings_and_vocabulary(self) -> None:
        """Reject blank names and malformed vocabulary entries.

        """

        self.assertEqual(
            require_string({"x": "briefing"}, "x", "example"),
            "briefing",
        )
        self.assertEqual(
            require_words({"x": ["a", "b"]}, "x", "example"),
            ("a", "b"),
        )
        for value in ("", " name ", None, 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "example.x"):
                    require_string({"x": value}, "x", "example")
        for value in ([], "a", ["A"], ["a", "a"], ["a b"], [1]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "example.x"):
                    require_words({"x": value}, "x", "example")

    def test_required_range_pair(self) -> None:
        """Reject incorrectly ordered or noninteger reference bounds.

        """

        self.assertEqual(
            require_int_pair({"x": [1, 3]}, "x", "example"), (1, 3)
        )
        for value in ([2, 1], [0, 3], [True, 2], [1], (1, 2), []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "example.x"):
                    require_int_pair({"x": value}, "x", "example")


if __name__ == "__main__":
    unittest.main()
