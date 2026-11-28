"""Exercise plugin-aware traces, offline inspection and replay preflight.

Use real workspace creation, traces, configuration references and tool
execution with deterministic model doubles, in separate interpreter lifetimes.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
from shutil import rmtree
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from kilin.system_testing._plugin_provenance import (
    compare_plugins, plugin_observations, preflight_plugins,
)
from kilin.system_testing.replay import _plugin_differences
from kilin.system_testing.timeline import TimelineOperation
from kilin.system_testing.traces import TraceEvent

_ROOT = Path(__file__).resolve().parents[3]
_TYPE = "provenance-example"
_TOOL = "provenance_example"
_SOURCE = '''from dataclasses import dataclass
from kilin.tools import Tool, ToolResult, reject_unknown

@dataclass(frozen=True)
class Settings:
    pass

def parse_settings(data, where):
    reject_unknown(where, data, ())
    return Settings()

def run(settings, request, context):
    return ToolResult("Answer A", (), {"name": "provenance_example"})

PLUGIN = Tool(name="provenance_example", summary="Deterministic example",
              version="1.0.0", api_version=1,
              parse_settings=parse_settings, run=run)
'''


def _type_data():
    """Define a minimal user type that only invokes the test plugin."""
    return {
        "schema_version": 1,
        "display_name": "Provenance example",
        "description": "A deterministic test project.",
        "extensions": [".txt"], "max_file_bytes": 4096,
        "retrieval_instruction": "Find information.",
        "system_prompt": "Use project sources.",
        "history": {"exclude_disliked_answers": True},
        "conversation_memory": {
            "enabled": False, "questions": "all", "answers": "selected",
            "max_hits": 4, "max_tokens": 2000,
        },
        "default_intent": "example",
        "intents": {"example": {"description": "Use the example tool.",
                                "tool": {"name": _TOOL}}},
    }


def _chat_model(model_id, **kwargs):
    """Stand in for loaded model weights while keeping real core execution."""
    model = Mock()
    model.model_id = model_id
    model.backend = kwargs["backend"]
    model.model_source = model_id
    model.temperature = kwargs.get("temperature", 0.0)
    model.thinking = kwargs.get("thinking", False)
    model.max_new_tokens = kwargs.get("max_new_tokens", 1024)
    model.messages = [{"role": "system",
                       "content": kwargs.get("system_prompt", "")}]
    model.input_token_total = 0
    model.output_token_total = 0
    model.last_finish_reason = None

    def configure(values, **kwargs):
        for key, value in values.items():
            setattr(model, key, value)

    def restore(messages):
        model.messages = deepcopy(messages)

    model.configure_generation.side_effect = configure
    model.restore_history.side_effect = restore
    return model


def _embedding_model(model_id, **kwargs):
    """Supply deterministic embedding identity and a local tokenizer."""
    model = Mock()
    model.model_id = model_id
    model.backend = kwargs["backend"]
    model.model_source = model_id
    model._retrieval_model_identity = "hf:fixture"
    model.count_tokens.side_effect = lambda text: len(text.split())
    model.embed_documents.side_effect = (
        lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
    )
    return model


def _event(kind, data, sequence=1):
    return TraceEvent("trace.json", "/", {"sequence": sequence,
                                         "event_type": kind}, data)


def _operation(*events, kind="create_project"):
    return TimelineOperation(kind, 1, 5, "completed", {
        "source_batch": {"type_id": _TYPE},
    }, events)


def _snapshot(root):
    return {path.relative_to(root).as_posix(): (
        path.read_bytes() if path.is_file() else None
    ) for path in root.rglob("*")}


def _models():
    """Replace only model loading, routing and router summarization."""
    from kilin import rag
    from kilin.answering import RouteDecision

    stack = ExitStack()
    for patcher in (
        patch.object(rag, "choose_backend", return_value="mlx"),
        patch.object(rag, "ChatSession", side_effect=_chat_model),
        patch.object(rag, "EmbeddingSession", side_effect=_embedding_model),
        patch.object(rag.RAGSession, "_route",
                     return_value=RouteDecision("example", True)),
        patch.object(rag, "summarize_answer", return_value="Example"),
    ):
        stack.enter_context(patcher)
    return stack


def _record(root):
    from kilin.gui.engine import ChatEngine
    from kilin.system_testing import Experiment

    events = []
    with _models():
        engine = ChatEngine(root, events.append, trace_enabled=True)
        try:
            engine.open_project([], _TYPE, str(root / "original"))
            assert events[-1]["event"] == "opened", events[-3:]
            engine.ask("Use example")
            assert len(engine.session.turns) == 1, events[-5:]
            assert engine.session.turns[0].answer == "Answer A"
        finally:
            engine.close()
        engine.resume_project(str(root / "original"))
        assert events[-1]["event"] == "opened", events[-3:]
        engine.close()
    report = Experiment(root / "original").validate()
    assert report.valid, report.issues
    observations = plugin_observations(report.events)
    assert len(observations) == 6, observations
    assert all(item["status"] == "recorded" for item in observations)
    assert all(item["identity"]["origin"] == "user" for item in observations)
    from kilin.system_testing.traces import read_recording

    recording = read_recording(root / "original")
    turns = [t for state in recording.states.values() for t in state["turns"]]
    assert turns[0]["parameters"]["tool_plugin"] == observations[3]["identity"]
    assert turns[0]["parameters"]["project_type_plugin"] == (
        observations[2]["identity"])


def _replay_changed(root):
    from kilin.system_testing import Experiment

    experiment = Experiment(root / "original")
    before = _snapshot(experiment.workspace)
    with _models():
        action = experiment.run_replay()
    assert action.status == "completed", action.error
    receipt = action.provenance["recorded"]["environment"]["plugins"]
    assert set(receipt["tools"]) == {_TOOL}, receipt
    assert receipt["project_type"]["type_id"] == _TYPE
    differences = action.result.differences
    assert any(d.field == f"plugins.execution.tool.{_TOOL}.sha256"
               for d in differences), differences
    assert any(d.field == f"plugins.open.project_type.{_TYPE}.sha256"
               for d in differences), differences
    comparison = experiment.compare()
    assert comparison.complete, comparison.data["alignment"]
    pairs = [p for op in comparison.data["operations"]
             for p in op["plugin_provenance"]]
    assert all(p["status"] == "changed" for p in pairs), pairs
    assert "Recorded plugin provenance" in comparison.markdown()
    from kilin.system_testing.analysis import _state
    from kilin.system_testing.batch import collect_issues, group_issues

    assert "Recorded plugin identities" in _state(comparison.data)
    issues = collect_issues(comparison.data, {})
    plugin_issues = [i for i in issues
                     if i["code"] == "plugin.environment.difference"]
    assert len(plugin_issues) == len(pairs)
    assert all(i["severity"] == "review" for i in plugin_issues)
    groups = group_issues(plugin_issues, comparison.data)
    assert all(g["code"] == "environment.plugins" for g in groups)
    assert "Answer B" in comparison.data["operations"][1]["question"][
        "answer"]["replayed"]
    assert before == _snapshot(experiment.workspace)


def _offline(root):
    from kilin.system_testing import Experiment, ReplayError
    from kilin.tools import _registry
    import importlib

    types = importlib.import_module("kilin.project_type")
    experiment = Experiment(root / "original")
    target = root / "original_replay"
    before, after = _snapshot(experiment.workspace), _snapshot(target)
    with patch.object(_registry, "_build_tool_registry",
                      side_effect=AssertionError("Offline import")):
        report = experiment.validate()
        assert report.valid, report.issues
        comparison = experiment.compare()
        assert comparison.complete, comparison.data["alignment"]
        assert experiment.run_replay().status == "retained"
        assert comparison.data["replay_execution"]["status"] == "recorded"
        assert _registry._registry is None and types._catalog is None
    # An execution request is different: unavailable types are rejected.
    absent = root / "never-created"
    try:
        experiment.replay(absent)
    except ReplayError as error:
        assert "Required project type" in str(error), error
        assert error.completed_operations == 0
    else:
        raise AssertionError("Missing plugin accepted")
    assert not absent.exists()
    action = experiment.run_replay(existing_replay="archive",
                                   archive_directory=root / "archive")
    assert action.status == "error" and action.archived_workspace is None
    assert action.completed_operations == 0
    assert not (root / "archive").exists()
    assert before == _snapshot(experiment.workspace)
    assert after == _snapshot(target)


class TestPluginProvenance(unittest.TestCase):
    """Preserve captured identity across changes and plugin removal."""

    def test_external_plugin_replay_and_offline_inspection(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plugins = root / "plugins"
            package = plugins / "tools" / _TOOL
            package.mkdir(parents=True)
            path = package / "__init__.py"
            path.write_text(_SOURCE, encoding="utf-8")
            definitions = plugins / "project_types"
            definitions.mkdir()
            definition = definitions / (_TYPE + ".json")
            definition.write_text(json.dumps(_type_data()), encoding="utf-8")
            env = dict(os.environ, KILIN_PLUGINS_DIR=str(plugins),
                       KILIN_DEFAULTS_FILE=str(root / "preferences.json"))
            env["PYTHONPATH"] = str(_ROOT)
            self._child(root, env, "_record")
            # Replacing code without bumping its declared version must
            # still produce a different identity; JSON bytes also change.
            path.write_text(_SOURCE.replace("Answer A", "Answer B"),
                            encoding="utf-8")
            definition.write_text(json.dumps(_type_data(), indent=2),
                                  encoding="utf-8")
            self._child(root, env, "_replay_changed")
            rmtree(plugins)
            self._child(root, env, "_offline")

    def _child(self, root, env, name):
        script = ("import sys; from tests.kilin.system_testing import "
                  "test__plugin_provenance as t; "
                  # Importing the tests package isolates its own registry.
                  "import os; os.environ['KILIN_PLUGINS_DIR']=sys.argv[2]; "
                  "getattr(t, sys.argv[3])(t.Path(sys.argv[1]))")
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(root),
             env["KILIN_PLUGINS_DIR"], name], env=env, cwd=root,
            capture_output=True, encoding="utf-8", timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_unknown_invalid_and_one_sided_observations(self):
        identity = {"name": _TOOL, "origin": "user", "version": "1.0.0",
                    "api_version": 1, "sha256": "a" * 64}
        known = _operation(_event("tool.run.started", {
            "name": _TOOL, "plugin": identity,
        }), kind="ask")
        old = _operation(_event("tool.run.started", {"name": _TOOL}),
                         kind="ask")
        self.assertEqual(compare_plugins(old, known)[0]["status"], "unknown")
        self.assertEqual(_plugin_differences((old,), (known,)), [])
        self.assertEqual(compare_plugins(known, known)[0]["status"], "equal")
        self.assertEqual(compare_plugins(known, _operation(kind="ask"))[0][
            "status"], "one_sided")
        bad = _operation(_event("tool.run.started", {
            "name": _TOOL, "plugin": {**identity, "name": "wrong"},
        }), kind="ask")
        self.assertEqual(compare_plugins(known, bad)[0]["status"], "invalid")
        observation = plugin_observations(known.events)[0]
        observation["identity"]["version"] = "changed"
        self.assertEqual(identity["version"], "1.0.0")

    def test_legacy_activation_preserves_unknown_requirements(self):
        operation = _operation(_event("gui.project.configuration.effective", {
            "profile": {"type_id": _TYPE}, "tools": [{"name": _TOOL}],
        }))
        items = plugin_observations(operation.events)
        self.assertEqual([x["name"] for x in items], [_TYPE, _TOOL])
        self.assertTrue(all(x["status"] == "unknown" for x in items))
        # Missing historical tools matter even after the current definition
        # stops referring to them.
        info = Mock(available=True, required_tools=())
        with patch("kilin.project_type.project_type_catalog",
                   return_value={_TYPE: info}), patch(
                       "kilin.tools._registry.registered_tools",
                       return_value={}), patch("kilin.plugins.list_tools",
                                               return_value=()):
            with self.assertRaisesRegex(ValueError, _TOOL):
                preflight_plugins((operation,))
