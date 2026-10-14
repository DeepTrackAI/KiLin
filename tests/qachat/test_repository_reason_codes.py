"""Check the packaged repository-selection protocol vocabulary.

The resource is shared by the Qt-free core, GUI engine, and JavaScript page.
Tests validate the JSON definitions without loading models or Qt.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from qachat import repository_reason_codes as codes


class TestRepositoryReasonCodes(unittest.TestCase):
    """Protect the canonical JSON vocabulary and named Python constants."""

    def test_resource_and_python_identifiers_agree(self) -> None:
        """Expose every packaged identifier without changing its value."""

        path = resources.files("qachat").joinpath(
            "repository_reason_codes.json"
        )
        definitions = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(dict(codes.SELECTION_REASON_CODES), definitions)
        self.assertEqual(len(definitions), 12)
        for name, value in definitions.items():
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(codes, f"SELECTION_REASON_{name}"), value
                )
        with self.assertRaises(TypeError):
            codes.SELECTION_REASON_CODES["FILE_TOO_LARGE"] = "other"

    def test_validation_rejects_invalid_definition_sets(self) -> None:
        """Reject absent, extra, malformed, and conflicting identifiers."""

        good = dict(codes.SELECTION_REASON_CODES)
        missing = dict(good)
        del missing["FILE_TOO_LARGE"]
        extra = dict(good, UNKNOWN="unknown")
        duplicate = dict(good, EMPTY_FILE=good["FILE_TOO_LARGE"])
        invalid = dict(good, EMPTY_FILE="Not a code")
        wrong_type = dict(good, EMPTY_FILE=42)
        for payload in (
            [], None, missing, extra, duplicate, invalid, wrong_type
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    codes._validate_reason_codes(payload)

    def test_resource_errors_are_not_silently_ignored(self) -> None:
        """Fail explicitly on missing or malformed packaged resources."""

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(
                codes.resources, "files", return_value=directory
            ):
                with self.assertRaises(FileNotFoundError):
                    codes._load_reason_codes()
                resource = directory / "repository_reason_codes.json"
                resource.write_text("{invalid", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                    codes._load_reason_codes()
                resource.write_text(
                    '{"FILE_TOO_LARGE":"first",'
                    '"FILE_TOO_LARGE":"second"}',
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "repeats key"):
                    codes._load_reason_codes()
                resource.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid keys"):
                    codes._load_reason_codes()


if __name__ == "__main__":
    unittest.main()
