"""Exercise trusted user plugins through isolated KíLín runtimes.

The tests create disposable plugin directories and use deterministic tools.
Neither models nor Qt nor the user's own plugins are required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tests.kilin._project_type_support import _type_data

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = '''from dataclasses import dataclass
from pathlib import Path
from kilin.tools import Tool, ToolResult, WorkspaceResource, check_int
from .helper import word

@dataclass(frozen=True)
class Settings:
    count: int

def parse_settings(data, where):
    return Settings(check_int(where, data, "count", 2, 1, 10))

def run(settings, request, context):
    from .lazy import suffix
    return ToolResult(word * settings.count + suffix, (), {})

def resources(settings):
    return (WorkspaceResource(Path(__file__).parent / "template.md",
                              "templates/example.md"),)

PLUGIN = Tool(name="NAME", summary="Example user tool.", version="1.2.3",
              api_version=1, parse_settings=parse_settings, run=run,
              workspace_resources=resources)
'''


class TestUserPlugins(unittest.TestCase):
    """Use a fresh registry for each integration scenario."""

    def setUp(self):
        temp = TemporaryDirectory(prefix="kilin-user-plugins-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.plugins = self.root / "plugins"
        self.tools = self.plugins / "tools"
        self.types = self.plugins / "project_types"
        self.tools.mkdir(parents=True)
        self.types.mkdir()

    def _tool(self, name, *, source=None, parent=None, defaults=True):
        folder = (parent or self.tools) / name
        folder.mkdir(parents=True)
        if source is None:
            source = _SOURCE.replace('"NAME"', repr(name))
        (folder / "__init__.py").write_text(source, encoding="utf-8")
        (folder / "helper.py").write_text('word = "a"\n', encoding="utf-8")
        (folder / "lazy.py").write_text('suffix = "!"\n', encoding="utf-8")
        (folder / "template.md").write_text("# User\n", encoding="utf-8")
        if defaults:
            (folder / "defaults.json").write_text(json.dumps({
                "SETTINGS": {"COUNT": 2},
                "MODEL_CALLS": {"THINKING": "off"},
            }), encoding="utf-8")
        return folder

    def _type(self, name, tool, *, parent=None, **settings):
        raw = _type_data()
        raw["schema_version"] = 1
        raw["intents"]["explain"]["tool"] = {"name": tool, **settings}
        raw["intents"]["explain"].pop("system_prompt")
        path = (parent or self.types) / (name + ".json")
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def _fresh(self, source, *args):
        env = dict(os.environ, KILIN_PLUGINS_DIR=str(self.plugins))
        script = "import sys\nsys.path.insert(0, sys.argv.pop(1))\n" + source
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(_ROOT),
             *map(str, args)],
            cwd=self.root, env=env, capture_output=True, text=True,
            encoding="utf-8", timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_discovery_workspace_settings_and_execution(self):
        tool = self._tool("user_repeat")
        marker = self.root / "imported"
        initial = tool / "__init__.py"
        initial.write_text(initial.read_text(encoding="utf-8") +
                           f"\nPath({str(marker)!r}).touch()\n",
                           encoding="utf-8")
        (tool / "settings_info.json").write_text(json.dumps({
            "format": "settings-info-v1", "settings": {"count": {
                "title": "Count", "summary": "Repetitions",
                "description": "Number of repetitions.",
            }},
        }), encoding="utf-8")
        definition = self._type("user-repeat", "user_repeat", count=3)
        self._fresh('''
import os
from pathlib import Path
import kilin
from kilin import plugins
from kilin.tools import (
    Tool, ToolContext, ToolRequest, CodeGraph, registered_tools, run_tool,
)
from kilin.tool_settings import project_tool_settings, save_tool_settings
from kilin.tool_settings import with_project_settings
from kilin.trace_logging import NullTraceLogger
marker, definition = map(Path, sys.argv[1:])
assert not marker.exists()
original_path = list(sys.path)
assert plugins.plugin_directory() == Path(os.environ["KILIN_PLUGINS_DIR"])
assert not marker.exists()
infos = plugins.list_tools()
assert marker.exists()
assert sys.path == original_path
info = next(x for x in infos if x.name == "user_repeat")
assert info.available and info.origin == "user" and info.version == "1.2.3"
assert info.api_version == 1 and len(info.sha256) == 64
assert "path" not in info.identity()
assert list(registered_tools())[-1] == "user_repeat"
tool = registered_tools()["user_repeat"]
assert tool.resource_root == definition.parent.parent / "tools/user_repeat"
assert tool.model_calls.thinking == "off"
assert kilin.get_tool_settings_info("user_repeat")["count"]["title"] == "Count"
assert kilin.project_type_ids()[-1] == "user-repeat"
kind = kilin.project_type_catalog()["user-repeat"]
assert kind.available and kind.required_tools == ("user_repeat",)
assert kind.origin == "user" and len(kind.sha256) == 64
root = Path("workspace").resolve()
kilin.initialize_project(root, "user-repeat")
assert (root / "templates/example.md").read_text() == "# User\\n"
assert project_tool_settings(root)[0]["settings"]["count"] == 3
save_tool_settings(root, "user_repeat", {"count": 4})
use = kilin.project_type("user-repeat").intents["explain"].tool
use = with_project_settings(root, use)

def unused(*args, **kwargs):
    raise AssertionError("Unexpected model call")

context = ToolContext(root, (), CodeGraph((), ()), lambda q: [], 1,
                      lambda p: [], lambda: {}, unused, unused, unused,
                      "test", lambda *a, **k: None, lambda *a, **k: None,
                      None, NullTraceLogger(), None)
result = run_tool(use, ToolRequest("question", "question"), context)
assert result.answer == "aaaa!"
from kilin.tools._registry import _tool_identity
from kilin.project_type import _project_type_identity
assert _tool_identity(use.name) == info.identity()
assert _project_type_identity("user-repeat") == kind.identity()
assert kilin.open_project(root).profile.type_id == "user-repeat"
raw = kilin.project_type_raw("user-repeat")
definition.unlink()
assert kilin.project_type_raw("user-repeat") == raw
assert kilin.project_type_catalog()["user-repeat"] == kind
os.environ["KILIN_PLUGINS_DIR"] = str(root)
assert plugins.plugin_directory() != root
''', marker, definition)

    def test_invalid_candidates_are_isolated(self):
        self._tool("good")
        bad_help = self._tool("bad_help")
        (bad_help / "settings_info.json").write_text("{", encoding="utf-8")
        self._tool("bad_syntax", source="def broken(:\n")
        self._tool("missing_import", source="import no_such_plugin_module\n")
        self._tool("exits", source="raise SystemExit(7)\n")
        self._tool("no_descriptor", source="pass\n")
        self._tool("wrong_api", source=_SOURCE.replace('"NAME"',
                   '"wrong_api"').replace("api_version=1", "api_version=2"))
        self._tool("wrong_name", source=_SOURCE.replace('"NAME"', '"other"'))
        bad_defaults = self._tool("bad_defaults")
        (bad_defaults / "defaults.json").write_text("{", encoding="utf-8")
        duplicate_defaults = self._tool("duplicate_defaults")
        (duplicate_defaults / "defaults.json").write_text(
            '{"SETTINGS":{"COUNT":2,"COUNT":3}}', encoding="utf-8"
        )
        self._tool("bad_parser", source=_SOURCE.replace('"NAME"',
                   '"bad_parser"').replace("return Settings(", "return dict("))
        duplicate = self.root / "duplicate-imported"
        self._tool("find_fact", source=f"open({str(duplicate)!r},'w')\n")
        required = _SOURCE.replace('"NAME"', '"requires_count"').replace(
            'return Settings(check_int(where, data, "count", 2, 1, 10))',
            'return Settings(data["count"])',
        )
        self._tool("requires_count", source=required, defaults=False)
        self._type("valid", "good", count=2)
        self._type("missing", "wrong_api")
        self._type("required", "requires_count")
        self._type("configured", "requires_count", count=3)
        self._type("literature", "good")
        (self.types / "broken.json").write_text("{", encoding="utf-8")
        self._fresh('''
from pathlib import Path
import kilin
from kilin import plugins
from kilin.tools import registered_tools
infos = {x.name: x for x in plugins.list_tools() if x.origin == "user"}
for name in ("bad_syntax", "missing_import", "exits", "no_descriptor",
             "wrong_api", "wrong_name", "bad_defaults", "bad_parser",
             "duplicate_defaults", "find_fact"):
    assert not infos[name].available and infos[name].error, name
    assert "kilin.plugins._user_" + name not in sys.modules
assert not Path(sys.argv[1]).exists()
assert not infos["find_fact"].runtime_checked
assert infos["good"].available and infos["requires_count"].available
assert infos["bad_help"].available and infos["bad_help"].warnings
assert registered_tools()["find_fact"].resource_root != infos["find_fact"].path
assert kilin.get_settings_info()
assert kilin.get_tool_settings_info("bad_help") == {}
types = kilin.project_type_catalog()
assert types["valid"].available and types["configured"].available
assert not types["required"].available and not types["broken"].available
assert types["missing"].required_tools == ("wrong_api",)
assert "wrong_api" in types["missing"].error
assert types["literature"].origin == "distribution"
conflict = [x for x in plugins.list_project_types()
            if x.type_id == "literature" and x.origin == "user"]
assert len(conflict) == 1 and not conflict[0].available
''', duplicate)

    def test_candidate_validation_does_not_replace_active_plugin(self):
        self._tool("candidate")
        marker = self.root / "validated"
        candidate = self._tool("candidate", parent=self.root / "drafts")
        initial = candidate / "__init__.py"
        initial.write_text(initial.read_text(encoding="utf-8") +
                           f"\nPath({str(marker)!r}).touch()\n",
                           encoding="utf-8")
        (candidate / "helper.py").write_text('word = "b"\n', encoding="utf-8")
        draft_type = self._type("draft", "not_installed", parent=self.root)
        broken = self._tool("broken", parent=self.root / "drafts",
                            source="import no_such_module\n")
        self._fresh('''
from pathlib import Path
from kilin import plugins
from kilin.tools import registered_tools, parse_tool, run_tool, ToolRequest
candidate, marker, draft, broken = map(Path, sys.argv[1:])
active = registered_tools()["candidate"]
structural = plugins.validate_tool(candidate)
assert not structural.error and not structural.runtime_checked
assert not structural.available and not marker.exists()
validated = plugins.validate_tool(candidate, trusted=True)
assert validated.available and marker.exists()
assert validated.sha256 != active._info.sha256
assert registered_tools()["candidate"] is active
assert run_tool(parse_tool("test", {"name": "candidate"}),
                ToolRequest("q", "q"), None).answer == "aa!"
assert plugins.validate_tool(broken).error is None
failed = plugins.validate_tool(broken, trusted=True)
assert failed.error and not failed.available and failed.runtime_checked
assert not list(candidate.rglob("*.pyc"))
structural_type = plugins.validate_project_type(draft, bind_tools=False)
assert not structural_type.error and not structural_type.bound
assert structural_type.required_tools == ("not_installed",)
assert plugins.validate_project_type(draft).error
assert "draft" not in {x.type_id for x in plugins.list_project_types()}
''', candidate, marker, draft_type, broken)

    def test_digest_and_source_only_loading(self):
        import py_compile
        from kilin.plugins._discovery import _package_digest

        tool = self._tool("stale")
        original = _package_digest(tool)
        source = tool / "helper.py"
        py_compile.compile(str(source), doraise=True)
        self.assertEqual(original, _package_digest(tool))
        stamp = source.stat()
        source.write_text('word = "b"\n', encoding="utf-8")
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        changed = _package_digest(tool)
        self.assertNotEqual(original, changed)
        self._fresh('''
from kilin import plugins
from kilin.tools import run_tool, parse_tool, ToolRequest
info = next(x for x in plugins.list_tools() if x.name == "stale")
assert info.sha256 == sys.argv[1]
assert run_tool(parse_tool("test", {"name": "stale"}),
                ToolRequest("q", "q"), None).answer == "bb!"
''', changed)

    def test_invalid_results_and_cancellation(self):
        for name, result in (
            ("bad_result", 'return "not a result"'),
            ("bad_json", 'return ToolResult("a", (), {"n": float("nan")})'),
            ("bad_exit", 'raise SystemExit(8)'),
            ("cancelled", 'raise GenerationCancelled()'),
        ):
            source = _SOURCE.replace('"NAME"', repr(name)).replace(
                'return ToolResult(word * settings.count + suffix, (), {})',
                result,
            )
            source = "from kilin.tools import GenerationCancelled\n" + source
            self._tool(name, source=source)
        self._fresh('''
from kilin.tools import GenerationCancelled, run_tool, parse_tool, ToolRequest
for name, error in (("bad_result", ValueError), ("bad_json", ValueError),
                    ("bad_exit", RuntimeError),
                    ("cancelled", GenerationCancelled)):
    try:
        run_tool(parse_tool("test", {"name": name}),
                 ToolRequest("q", "q"), None)
    except error:
        pass
    else:
        raise AssertionError(name)
''')

    def test_result_evidence_preserves_path_contract(self):
        source = _SOURCE.replace('"NAME"', '"evidence_tool"').replace(
            'return ToolResult(word * settings.count + suffix, (), {})',
            'return ToolResult("Answer [1]", ({"citation": 1, '
            '"path": Path("source.txt"), "start_line": 1, "end_line": 1, '
            '"symbol": None, "text": "Source", "cell": None},), {})',
        )
        self._tool("evidence_tool", source=source)
        self._fresh('''
from pathlib import Path
from kilin.rag import Evidence
from kilin.tools import ToolRequest, parse_tool, run_tool
use = parse_tool("test", {"name": "evidence_tool"})
result = run_tool(use, ToolRequest("q", "q"), None)
assert Evidence(**result.evidence[0]).path == Path("source.txt")
''')

    def test_path_policy(self):
        from kilin.plugins._paths import _resolve_directory

        with patch.dict(os.environ, {"KILIN_PLUGINS_DIR": "relative"}):
            with self.assertRaisesRegex(ValueError, "absolute"):
                _resolve_directory()
        with patch.dict(os.environ, {"KILIN_PLUGINS_DIR": str(self.root)}):
            self.assertEqual(_resolve_directory(), self.root)

    def test_documented_minimal_tool(self):
        text = (_ROOT / "docs/PLUGIN_API.md").read_text(encoding="utf-8")
        source = text.split("```python\n", 1)[1].split("```", 1)[0]
        self._tool("repeat_question", source=source, defaults=False)
        definition = text.split("```json\n", 1)[1].split("```", 1)[0]
        (self.types / "repetition.json").write_text(definition,
                                                   encoding="utf-8")
        self._fresh('''
import kilin
from kilin.tools import run_tool, ToolRequest
use = kilin.project_type("repetition").intents["repeat"].tool
result = run_tool(use, ToolRequest("hello", "hello"), None)
assert result.answer == "hello\\nhello" and result.summary == {"count": 2}
''')
