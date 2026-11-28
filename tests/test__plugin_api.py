"""Verify the versioned authoring boundary and deferred catalog creation.

Fresh interpreters prevent earlier test imports from hiding initialization
cycles or partially published registry state. No model or GUI is required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import kilin


class TestPluginAPI(unittest.TestCase):
    """Exercise the authoring facade before any executable tool is loaded."""

    def _fresh(self, script: str) -> None:
        """Run an assertion script in a fresh source-tree interpreter."""

        root = Path(kilin.__file__).resolve().parent.parent
        with TemporaryDirectory() as directory:
            process = subprocess.run(
                [sys.executable, "-B", "-c", script, str(root)],
                cwd=directory, text=True, capture_output=True,
                timeout=60, check=False,
            )
            self.assertEqual(
                process.returncode, 0, process.stdout + process.stderr
            )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_imports_do_not_initialize_catalogs(self) -> None:
        """Guard direct imports of both the public API and a tool package."""

        setup = '''
import importlib
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])

# Catch a catalog read even if a future implementation drops its cache.
opened = []
def audit(event, args):
    if event == "open" and isinstance(args[0], str):
        path = Path(args[0])
        if path.suffix == ".json" and "project_types" in path.parts:
            opened.append(path)
sys.addaudithook(audit)
'''
        assertions = '''
from kilin.tools import (
    PLUGIN_API_VERSION, Tool, ToolContext, ToolTrace, ToolRequest,
    ToolResult, ToolRouting, WorkspaceResource, EarlierTurn, Chunk,
    CodeGraph, CodeEntity, CodeRelation, HybridResult, EvidenceLimits,
    RetrievalRequest, RetrievalResult, Suggestion, GenerationCancelled,
    IncompleteModelResponse, ToolPlan, ask_choice, confirm_plan,
    partial_result, save_output_file, selected_paths, selected_source_file,
    check_bool, check_int, check_choice, reject_unknown,
)
from kilin.tools import _registry
assert PLUGIN_API_VERSION == 1
assert Tool.__module__ == "kilin.tools._contract"
assert _registry._registry is None
module = importlib.import_module("kilin.project_type")
assert module.load_project_types.cache_info().currsize == 0
assert not opened, opened
assert not any(name.startswith("PySide6") for name in sys.modules)
assert "torch" not in sys.modules and "mlx" not in sys.modules
loaded = {name.split(".")[2] for name in sys.modules
          if name.startswith("kilin.tools.")
          and not name.split(".")[2].startswith("_")}
assert loaded <= expected, loaded
'''
        for statement, expected in (
            ("import kilin", "set()"),
            ("from kilin.tools import Tool", "set()"),
            ("import kilin.tools.project_inventory",
             "{'project_inventory'}"),
            ("import kilin.system_testing", "set()"),
        ):
            with self.subTest(statement=statement):
                self._fresh(
                    setup + statement + "\nexpected = " + expected
                    + "\n" + assertions
                )

    def test_one_registry_with_legacy_accessors(self) -> None:
        """Resolve the same bundled entries from all public access paths."""

        self._fresh('''
import sys
sys.path.insert(0, sys.argv[1])
import kilin
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from kilin.tools import _registry, registered_tools, tool_choices, ToolName
from kilin.project import project_type_ids
assert ToolName is str
assert _registry._registry is None
with patch.object(_registry, "_build_tool_registry",
                  wraps=_registry._build_tool_registry) as build:
    with ThreadPoolExecutor(max_workers=8) as pool:
        catalogs = list(pool.map(lambda _: registered_tools(), range(16)))
    assert build.call_count == 1
from kilin.tools import TOOL_REGISTRY, TOOL_CHOICES
from kilin.tools._registry import TOOL_REGISTRY as old_registry
assert TOOL_REGISTRY is old_registry
assert len(TOOL_REGISTRY) == 15
assert TOOL_CHOICES == tool_choices() == kilin.TOOL_CHOICES
assert set(TOOL_CHOICES) == set(TOOL_REGISTRY)
assert all(catalog == TOOL_REGISTRY for catalog in catalogs)
assert all(catalogs[0][name] is tool
           for name, tool in TOOL_REGISTRY.items())
try:
    catalogs[0]["new"] = None
except TypeError:
    pass
else:
    raise AssertionError("Catalog view accepted a write")
assert kilin.PROJECT_TYPE_IDS == project_type_ids() == (
    "code", "creative_brainstorming", "grounded_brainstorming",
    "literature", "reviewing", "writing",
)
from kilin.tools import ClassifySettings, ScanSettings
from kilin.tools.batch_classify import ClassifySettings as Classify
from kilin.tools.scan_and_list import ScanSettings as Scan
assert ClassifySettings is Classify and ScanSettings is Scan
''')

    def test_incompatible_api_does_not_publish_registry(self) -> None:
        """Reject incompatible descriptors and allow clean subsequent use."""

        self._fresh('''
import sys
sys.path.insert(0, sys.argv[1])
from dataclasses import replace
from unittest.mock import patch
from kilin.tools import _registry, registered_tools, PLUGIN_API_VERSION
entries = _registry._build_tool_registry()
first = next(iter(entries))
for version in (PLUGIN_API_VERSION + 1, True, "1", None):
    invalid = dict(entries)
    invalid[first] = replace(entries[first], api_version=version)
    with patch.object(_registry, "_build_tool_registry", return_value=invalid):
        try:
            registered_tools()
        except ValueError as error:
            assert first in str(error) and "plugin API version" in str(error)
        else:
            raise AssertionError("Accepted incompatible API")
    assert _registry._registry is None
    assert not _registry._initializing
with patch.object(_registry, "_build_tool_registry",
                  side_effect=registered_tools):
    try:
        registered_tools()
    except RuntimeError as error:
        assert "Recursive" in str(error)
    else:
        raise AssertionError("Accepted recursive initialization")
assert _registry._registry is None
assert not _registry._initializing
assert len(registered_tools()) == 15
''')

    def test_unregistered_names_fail_configuration(self) -> None:
        """Open identifiers do not allow arbitrary imports or invalid keys."""

        from kilin.tools import parse_tool

        for name in ("unregistered_tool", "os.path", [], None, True):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "tool name"):
                    parse_tool("test.intent", {"name": name})

    def test_descriptor_and_public_helpers(self) -> None:
        """Build a descriptor using public imports and JSON-safe settings."""

        from dataclasses import asdict, dataclass
        import json

        from kilin.tools import (
            PLUGIN_API_VERSION, Tool, ToolRequest, ToolResult,
            check_int, reject_unknown,
        )

        @dataclass(frozen=True)
        class Settings:
            """One example setting."""

            count: int

        def parse(data, where):
            """Validate settings without a model or a project."""

            reject_unknown(where, data, ("count",))
            return Settings(check_int(where, data, "count", 1, 1, 10))

        def run(settings, request, context):
            """Return a deterministic result independent of host services."""

            return ToolResult(request.question * settings.count, (), {})

        descriptor = Tool(
            name="example", summary="Repeat a question.",
            parse_settings=parse, run=run, version="0.1.0",
            api_version=PLUGIN_API_VERSION,
        )
        settings = descriptor.parse_settings({"count": 2}, "test")
        self.assertEqual(
            json.loads(json.dumps(asdict(settings))), {"count": 2}
        )
        result = descriptor.run(settings, ToolRequest("a", "a"), None)
        self.assertEqual(result, ToolResult("aa", (), {}))


if __name__ == "__main__":
    unittest.main()
