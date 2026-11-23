"""Verify shared tool-contract assertions with controlled false contracts.

Tool-specific tests retain their own invalid-setting and functional cases.
These checks ensure consolidation does not silently weaken registry/default
validation when a package is registered or its resources change.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from kilin.tools._registry import TOOL_REGISTRY
from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)


@dataclass(frozen=True)
class _Settings:
    """Representative packaged values, including a serialized list."""

    count: int = 3
    tags: tuple[str, ...] = ("a", "b")


class TestContractSupport(unittest.TestCase):
    """Exercise positive and negative shared contract checks."""

    def test_registry_detects_wrong_parser_and_followup_policy(self) -> None:
        package = ModuleType("sample_tool")
        package.parse_settings = lambda *args: None
        package.run = lambda *args: None
        entry = SimpleNamespace(
            parse_settings=package.parse_settings,
            run=package.run,
            workspace_resources=None,
            follows_up=False,
        )
        with patch.dict(TOOL_REGISTRY, {"test_only": entry}):
            assert_registry_entry(
                self, "test_only", package, non_followup=True
            )
            entry.parse_settings = lambda *args: None
            with self.assertRaises(AssertionError):
                assert_registry_entry(self, "test_only", package)
            entry.parse_settings = package.parse_settings
            entry.follows_up = True
            with self.assertRaises(AssertionError):
                assert_registry_entry(
                    self, "test_only", package, non_followup=True
                )

    def test_packaged_defaults_detect_value_drift(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / "tool.py"
            file.write_text("", encoding="utf-8")
            defaults = {
                "SETTINGS": {"COUNT": 3, "TAGS": ["a", "b"]},
            }
            path = root / "defaults.json"
            path.write_text(json.dumps(defaults), encoding="utf-8")
            (root / "README.md").write_text(
                "`SETTINGS` documents `COUNT` and `TAGS`.",
                encoding="utf-8",
            )
            package = ModuleType("test_only")
            package.__file__ = str(file)
            package.parse_settings = lambda *_: _Settings()
            data = assert_packaged_defaults(self, package, _Settings)
            self.assertEqual(data, defaults)
            defaults["SETTINGS"]["COUNT"] = 8
            path.write_text(json.dumps(defaults), encoding="utf-8")
            strict_case = unittest.TestCase()
            strict_case.subTest = lambda **_kwargs: nullcontext()
            with self.assertRaises(AssertionError):
                assert_packaged_defaults(strict_case, package, _Settings)
            # Reference tools retain their independent configuration tests.
            assert_packaged_defaults(
                self, package, _Settings, check_values=False
            )

    def test_packaged_defaults_require_documented_keys(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / "tool.py"
            file.write_text("", encoding="utf-8")
            (root / "defaults.json").write_text(
                '{"SETTINGS": {"COUNT": 3, "TAGS": ["a", "b"]}}',
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "Only `SETTINGS` is documented.", encoding="utf-8"
            )
            package = ModuleType("test_only")
            package.__file__ = str(file)
            package.parse_settings = lambda *_: _Settings()
            strict_case = unittest.TestCase()
            strict_case.subTest = lambda **_kwargs: nullcontext()
            with self.assertRaises(AssertionError):
                assert_packaged_defaults(strict_case, package, _Settings)


if __name__ == "__main__":
    unittest.main()
