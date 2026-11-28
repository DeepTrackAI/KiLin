"""Verify descriptor discovery and ownership independently of a model or GUI.

The bundled migration preserves each executable tool's contract while the
loader binds a descriptor to the package actually imported. Temporary
packages exercise that boundary without altering the installed distribution.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from importlib import import_module, util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from kilin import settings_info
from kilin.project_layout import (
    _tool_owned_source,
    install_project_resources,
    plan_project_layout,
)
from kilin.tools import Tool, ToolResult, ToolUse, registered_tools
from kilin.tools._registry import _register_tool


@dataclass(frozen=True)
class _Settings:
    """Minimal settings record used by synthetic descriptors."""

    count: int


def _parse(data, where):
    """Require explicit settings rather than assuming an empty default."""

    if type(data.get("count")) is not int:
        raise ValueError(f"{where}: count required")
    return _Settings(data["count"])


def _run(settings, request, context):
    """Return a deterministic result without calling supplied services."""

    return ToolResult(str(settings.count), (), {"count": settings.count})


class TestToolRegistration(unittest.TestCase):
    """Check registration independently of project-type discovery."""

    def test_bundled_descriptors_and_defaults(self):
        expected = (
            "project_inventory", "scan_and_list", "batch_classify",
            "batch_review", "filter_results", "find_fact", "find_passages",
            "paper_overview", "reference_check", "suggest_reference",
            "writing_check", "assistant_guide", "show_source",
            "generate_paper_summary", "project_summary",
        )
        tools = registered_tools()
        self.assertEqual(tuple(tools), expected)
        off = {
            "batch_classify", "batch_review", "filter_results",
            "find_passages", "generate_paper_summary", "writing_check",
            "project_summary",
        }
        for name, tool in tools.items():
            with self.subTest(tool=name):
                module = import_module(f"kilin.tools.{name}")
                descriptor = module.PLUGIN
                self.assertIsInstance(descriptor, Tool)
                self.assertIsNot(tool, descriptor)
                for item in fields(Tool):
                    self.assertEqual(getattr(tool, item.name),
                                     getattr(descriptor, item.name))
                self.assertEqual(tool.api_version, 1)
                self.assertEqual(tool.version, "1.0.0")
                self.assertEqual(tool.resource_root,
                                 Path(module.__file__).resolve().parent)
                self.assertIs(tool.parse_settings, module.parse_settings)
                self.assertIs(tool.run, module.run)
                self.assertEqual(tool.model_calls.thinking,
                                 "off" if name in off else "inherit")
                defaults = json.loads(
                    (tool.resource_root / "defaults.json").read_text(
                        encoding="utf-8"
                    )
                )
                data = {
                    k.lower(): v for k, v in defaults.pop("SETTINGS").items()
                }
                self.assertEqual(tool.internal_defaults, defaults)
                self.assertEqual(asdict(tool.parse_settings(data, name)),
                                 asdict(tool.parse_settings({}, name)))
                with self.assertRaises(RuntimeError):
                    _ = descriptor.resource_root
        self.assertTrue(tools["filter_results"].follows_up)
        self.assertTrue(tools["show_source"].follows_up)
        self.assertIsNotNone(tools["writing_check"].follows_up_when)
        for name in ("generate_paper_summary", "project_summary"):
            self.assertIsNotNone(tools[name].routing)
            self.assertIsNotNone(tools[name].workspace_resources)

    def test_additional_bundled_package_is_discovered(self):
        """An isolated distribution needs no central registry edit."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            installed = root / "kilin"
            source = Path(import_module("kilin").__file__).parent
            shutil.copytree(source, installed,
                            ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copy2(source.parent / "pyproject.toml", root)
            package = installed / "tools" / "z_example"
            package.mkdir()
            (package / "__init__.py").write_text('''
from dataclasses import dataclass
from kilin.tools import Tool, ToolResult
@dataclass(frozen=True)
class Settings:
    pass
def parse_settings(data, where):
    if data:
        raise ValueError(where)
    return Settings()
def run(settings, request, context):
    return ToolResult("example", (), {})
PLUGIN = Tool("z_example", "Example", parse_settings, run,
              version="1.0.0", api_version=1)
''', encoding="utf-8")
            private = installed / "tools" / "_ignored"
            private.mkdir()
            (private / "__init__.py").write_text(
                'raise AssertionError("private package imported")',
                encoding="utf-8",
            )
            script = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from kilin.tools import registered_tools, tool_choices, parse_tool, run_tool
from kilin.settings_info import get_tool_settings_info
catalog = registered_tools()
assert len(catalog) == 16
assert tuple(catalog)[-1] == tool_choices()[-1] == "z_example"
tool = catalog["z_example"]
assert tool.resource_root == Path(sys.argv[1]) / "kilin/tools/z_example"
assert tool.model_calls.thinking == "inherit"
assert tool.internal_defaults == {}
assert get_tool_settings_info("z_example") == {}
use = parse_tool("test", {"name": "z_example"})
assert run_tool(use, None, None).answer == "example"
'''
            result = subprocess.run(
                [sys.executable, "-B", "-c", script, str(root)],
                cwd=root, text=True, capture_output=True, timeout=60,
                check=False,
            )
            self.assertEqual(result.returncode, 0,
                             result.stdout + result.stderr)

    def test_relocated_report_tool_owns_its_resources(self):
        original = registered_tools()["generate_paper_summary"]
        namespace = "_kilin_test_report_copy"
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / original.name
            shutil.copytree(original.resource_root, root,
                            ignore=shutil.ignore_patterns("__pycache__"))
            defaults_path = root / "defaults.json"
            defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
            defaults["MODEL_CALLS"]["THINKING"] = "on"
            defaults["SETTINGS"]["REPAIR_ATTEMPTS"] = 2
            defaults_path.write_text(json.dumps(defaults), encoding="utf-8")
            metadata_path = root / "settings_info.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["settings"]["repair_attempts"]["title"] = "Copy"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            spec = util.spec_from_file_location(
                namespace, root / "__init__.py",
                submodule_search_locations=[str(root)],
            )
            module = util.module_from_spec(spec)
            sys.modules[namespace] = module
            try:
                spec.loader.exec_module(module)
                tool = _register_tool(module, root)
                settings = tool.parse_settings({}, "test")
                self.assertTrue(tool.parse_settings.__module__.startswith(
                    namespace + "."
                ))
                self.assertEqual(tool.resource_root, root.resolve())
                self.assertEqual(settings.repair_attempts, 2)
                self.assertEqual(tool.model_calls.thinking, "on")
                self.assertEqual(original.model_calls.thinking, "off")
                self.assertEqual(tool.internal_defaults["MODEL_CALLS"],
                                 {"THINKING": "on"})
                resources = tool.workspace_resources(settings)
                self.assertTrue(resources)
                for resource in resources:
                    self.assertTrue(resource.source.is_relative_to(root))
                    self.assertEqual(
                        _tool_owned_source(resource.source, tool),
                        resource.source,
                    )
                    bundled = original.resource_root / (
                        resource.source.relative_to(root)
                    )
                    with self.assertRaisesRegex(ValueError, "outside"):
                        _tool_owned_source(bundled, tool)
                definition = SimpleNamespace(intents={
                    "test": SimpleNamespace(tool=ToolUse(tool.name, settings))
                })
                with (
                    patch("kilin.project_layout.project_type",
                          return_value=definition),
                    patch("kilin.project_layout.registered_tools",
                          return_value={tool.name: tool}),
                    patch("kilin.tools._registry.registered_tools",
                          return_value={tool.name: tool}),
                ):
                    plan = plan_project_layout("synthetic")
                    info = settings_info._tool_info.__wrapped__(tool.name)
                self.assertEqual(info["repair_attempts"]["title"], "Copy")
                workspace = Path(temporary) / "workspace"
                workspace.mkdir()
                install_project_resources(workspace, plan)
                for resource in plan.resources:
                    self.assertEqual(
                        (workspace / resource.destination).read_bytes(),
                        resource.source.read_bytes(),
                    )
            finally:
                for name in tuple(sys.modules):
                    if (name == namespace
                            or name.startswith(namespace + ".")):
                        del sys.modules[name]

    def test_invalid_descriptors_and_optional_policies(self):
        descriptor = Tool("example", "Example", _parse, _run,
                          version="1.0.0", api_version=1)
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "example"
            root.mkdir()
            module = ModuleType("example")
            module.__file__ = str(root / "__init__.py")
            Path(module.__file__).touch()
            for change in (
                {"name": "other"}, {"api_version": 2},
                {"api_version": True}, {"version": ""},
                {"parse_settings": None}, {"routing": True},
            ):
                with self.subTest(change=change):
                    module.PLUGIN = replace(descriptor, **change)
                    with self.assertRaises(ValueError):
                        _register_tool(module, root)
            module.PLUGIN = descriptor
            tool = _register_tool(module, root)
            self.assertEqual(tool.model_calls.thinking, "inherit")
            with self.assertRaisesRegex(ValueError, "count required"):
                tool.parse_settings({}, "test")
        for data, valid in (
            ({"SETTINGS": {"COUNT": 2}}, True),
            ({"SETTINGS": {"COUNT": "bad"}}, False),
            ({"MODEL_CALLS": {}}, False),
            ({"MODEL_CALLS": {"THINKING": True}}, False),
            ({"MODEL_CALLS": {"THINKING": "off"}}, True),
        ):
            with self.subTest(data=data), TemporaryDirectory() as temporary:
                root = Path(temporary) / "example"
                root.mkdir()
                module.__file__ = str(root / "__init__.py")
                Path(module.__file__).touch()
                (root / "defaults.json").write_text(json.dumps(data),
                                                   encoding="utf-8")
                if valid:
                    tool = _register_tool(module, root)
                    self.assertEqual(tool.model_calls.thinking,
                                     data.get("MODEL_CALLS", {}).get(
                                         "THINKING", "inherit"
                                     ))
                else:
                    with self.assertRaises(ValueError):
                        _register_tool(module, root)


if __name__ == "__main__":
    unittest.main()
