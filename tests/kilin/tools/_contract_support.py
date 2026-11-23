"""Shared contract assertions for separately packaged built-in tools.

Each tool test retains its own discoverable test methods and package-specific
invalid settings. These helpers centralize only identical registry and
packaged-default checks, not the tools' distinct execution behavior.

"""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from types import ModuleType
import unittest

from kilin.tools._registry import TOOL_REGISTRY


def assert_registry_entry(
    case: unittest.TestCase,
    name: str,
    tool: ModuleType,
    *,
    non_followup: bool = False,
) -> None:
    """Check that a tool registry entry references its own package API.

    Parameters
    ----------
    case: unittest.TestCase
        Calling test, whose assertions report failures in that tool's suite.
    name: str
        Registered tool identifier.
    tool: types.ModuleType
        The tool package exporting ``parse_settings`` and ``run``.
    non_followup: bool
        Whether to also require ``follows_up`` to be false.

    """

    entry = TOOL_REGISTRY[name]
    case.assertIs(entry.parse_settings, tool.parse_settings)
    case.assertIs(entry.run, tool.run)
    case.assertIsNone(entry.workspace_resources)
    if non_followup:
        case.assertFalse(entry.follows_up)


def assert_packaged_defaults(
    case: unittest.TestCase,
    tool: ModuleType,
    settings_type: type,
    *,
    check_values: bool = True,
) -> dict:
    """Check packaged settings and documentation against the public parser.

    Parameters
    ----------
    case: unittest.TestCase
        Calling test for tool-local assertion attribution and subtests.
    tool: types.ModuleType
        Tool package supplying ``defaults.json``, README and parser.
    settings_type: type
        The tool's default settings dataclass.
    check_values: bool
        Verify literal values for ordinary packaged tools. Specialized
        reference tools retain their own extra configuration checks.

    Returns
    -------
    dict
        Decoded packaged sections for additional tool-specific checks.

    """

    package = Path(tool.__file__).parent
    data = json.loads(
        (package / "defaults.json").read_text(encoding="utf-8")
    )
    readme = (package / "README.md").read_text(encoding="utf-8")
    for section, values in data.items():
        with case.subTest(section=section):
            case.assertIn(f"`{section}`", readme)
        for key in values:
            with case.subTest(section=section, key=key):
                case.assertIn(f"`{key}`", readme)
    defaults = settings_type()
    case.assertEqual(defaults, tool.parse_settings({}, "example"))
    case.assertEqual(
        {field.name for field in fields(defaults)},
        {name.lower() for name in data["SETTINGS"]},
    )
    if check_values:
        for field in fields(defaults):
            expected = data["SETTINGS"][field.name.upper()]
            if isinstance(expected, list):
                expected = tuple(expected)
            case.assertEqual(getattr(defaults, field.name), expected)
    return data
