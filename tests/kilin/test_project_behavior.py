"""Test project-local behavior overrides without loading model backends.

Use real packaged type definitions with isolated workspaces to check
validation, inheritance, independent projects, revision checks, and resets.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import kilin
from kilin.project_behavior import (
    PROJECT_BEHAVIOR_FILE,
    effective_project_type,
    project_behavior_settings,
    save_project_behavior,
)
from kilin.project_type import project_type


class ProjectBehaviorFixture(unittest.TestCase):
    """Create independent literature projects with no loaded models."""

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.first = self.root / "first"
        self.second = self.root / "second"
        kilin.initialize_project(self.first, "literature")
        kilin.initialize_project(self.second, "literature")

    def save_global(self, **changes: object) -> dict:
        """Save the global editor while retaining all other values."""

        payload = project_behavior_settings(self.first)
        settings = deepcopy(payload["global"]["values"])
        settings.update(changes)
        return save_project_behavior(
            self.first, "global", settings,
            expected_revision=payload["revision"],
        )

    def save_intent(self, name: str, **changes: object) -> dict:
        """Save an existing intent with selected overridden values."""

        payload = project_behavior_settings(self.first)
        original = next(
            item for item in payload["intents"] if item["id"] == name
        )
        values = deepcopy(original["values"])
        values.update(changes)
        return save_project_behavior(
            self.first, "intent", values, intent_id=name,
            expected_revision=payload["revision"],
        )


class TestProjectBehavior(ProjectBehaviorFixture):

    def test_unedited_projects_use_packaged_configuration(self):
        base = project_type("literature")
        self.assertIs(effective_project_type(self.first), base)
        self.assertFalse((self.first / PROJECT_BEHAVIOR_FILE).exists())
        values = project_behavior_settings(self.first)
        self.assertEqual(values["global"]["values"],
                         values["global"]["defaults"])
        self.assertEqual(len(values["intents"]), len(base.intents))

    def test_global_changes_are_private_and_survive_reopening(self):
        changed = self.save_global(
            description="My project", welcome_prompts=["Welcome back"],
            system_prompt="Only refer to this project's records.",
            history={"recent_turns": 7},
        )
        actual = effective_project_type(self.first)
        self.assertEqual(actual.description, "My project")
        self.assertEqual(actual.recent_turns, 7)
        self.assertEqual(actual.welcome_prompts, ("Welcome back",))
        self.assertEqual(
            effective_project_type(self.second).description,
            project_type("literature").description,
        )
        saved = json.loads(
            (self.first / PROJECT_BEHAVIOR_FILE).read_text(encoding="utf-8")
        )
        self.assertNotIn("suggestions", saved["global"])
        self.assertEqual(saved["global"]["history"], {"recent_turns": 7})
        self.assertEqual(changed["global"]["values"]["description"],
                         "My project")

    def test_intents_are_independent_and_unchanged_fields_inherit(self):
        modified = self.save_intent(
            "question_answering", description="My personalized routing",
            thinking=True,
        )
        config = effective_project_type(self.first)
        self.assertTrue(config.intents["question_answering"].thinking)
        self.assertEqual(config.intents["question_answering"].description,
                         "My personalized routing")
        self.assertEqual(
            config.intents["fact_finding"],
            project_type("literature").intents["fact_finding"],
        )
        self.assertIsNone(
            effective_project_type(self.second)
            .intents["question_answering"].thinking
        )
        self.assertTrue(modified["revision"])
        state = json.loads(
            (self.first / PROJECT_BEHAVIOR_FILE).read_text(encoding="utf-8")
        )
        self.assertEqual(state["intents"]["question_answering"], {
            "description": "My personalized routing", "thinking": True,
        })

    def test_strategy_change_removes_incompatible_packaged_limits(self):
        payload = project_behavior_settings(self.first)
        original = next(
            item for item in payload["intents"]
            if item["id"] == "question_answering"
        )
        values = deepcopy(original["values"])
        self.assertEqual(values["retrieval"]["strategy"], "source_first")
        values["retrieval"]["strategy"] = "hybrid"
        for key in (
            "max_sources", "passages_per_source", "max_passages",
            "evidence_tokens", "coverage",
        ):
            values["retrieval"].pop(key)
        self.save_intent("question_answering", retrieval=values["retrieval"])
        self.assertEqual(
            effective_project_type(self.first)
            .intents["question_answering"].retrieval_strategy, "hybrid"
        )
        state = json.loads(
            (self.first / PROJECT_BEHAVIOR_FILE).read_text(encoding="utf-8")
        )
        removal = state["intents"]["question_answering"]["retrieval"]
        self.assertEqual(removal["max_passages"], {"$delete": True})

    def test_invalid_values_and_stale_saves_leave_file_unchanged(self):
        current = self.save_global(description="Saved")
        path = self.first / PROJECT_BEHAVIOR_FILE
        before = path.read_bytes()
        values = deepcopy(current["global"]["values"])
        values["default_intent"] = "missing_intent"
        with self.assertRaisesRegex(ValueError, "default_intent"):
            save_project_behavior(self.first, "global", values)
        self.assertEqual(path.read_bytes(), before)
        values["default_intent"] = "question_answering"
        with self.assertRaisesRegex(ValueError, "changed since"):
            save_project_behavior(
                self.first, "global", values, expected_revision="stale"
            )
        self.assertEqual(path.read_bytes(), before)

    def test_tool_binding_cannot_be_overridden(self):
        current = project_behavior_settings(self.first)
        intent = next(x for x in current["intents"]
                      if x["id"] == "fact_finding")
        values = deepcopy(intent["values"])
        values["thinking"] = True
        with self.assertRaisesRegex(ValueError, "thinking"):
            save_project_behavior(
                self.first, "intent", values, intent_id="fact_finding"
            )
        values = deepcopy(intent["values"])
        values["tool"] = "unknown"
        with self.assertRaisesRegex(ValueError, "editable intent"):
            save_project_behavior(
                self.first, "intent", values, intent_id="fact_finding"
            )
        self.assertFalse((self.first / PROJECT_BEHAVIOR_FILE).exists())

    def test_restoring_defaults_removes_sparse_overrides(self):
        self.save_global(description="Personal")
        self.save_intent("question_answering", thinking=True)
        current = project_behavior_settings(self.first)
        save_project_behavior(
            self.first, "global", current["global"]["defaults"]
        )
        current = project_behavior_settings(self.first)
        intent = next(x for x in current["intents"]
                      if x["id"] == "question_answering")
        save_project_behavior(self.first, "intent", intent["defaults"],
                              intent_id="question_answering")
        saved = json.loads(
            (self.first / PROJECT_BEHAVIOR_FILE).read_text(encoding="utf-8")
        )
        self.assertEqual(saved["global"], {})
        self.assertEqual(saved["intents"], {})
        self.assertEqual(effective_project_type(self.first),
                         project_type("literature"))

    def test_indexing_capabilities_inherit_and_remain_project_local(self):
        current = project_behavior_settings(self.first)
        defaults = current["capabilities"]["defaults"]
        values = {"code_graph": False,
                  "document_headers": not defaults["document_headers"]}
        updated = save_project_behavior(
            self.first, "capabilities", values,
            expected_revision=current["revision"],
        )
        self.assertEqual(updated["capabilities"]["values"], values)
        self.assertEqual(effective_project_type(self.first).code_graph, False)
        self.assertEqual(
            effective_project_type(self.first).document_headers,
            values["document_headers"],
        )
        self.assertEqual(effective_project_type(self.second).document_headers,
                         defaults["document_headers"])
        saved = json.loads((self.first / PROJECT_BEHAVIOR_FILE).read_text())
        self.assertEqual(saved["capabilities"], values)
        self.save_global(description="Separate global edit")
        save_project_behavior(self.first, "capabilities", defaults)
        saved = json.loads((self.first / PROJECT_BEHAVIOR_FILE).read_text())
        self.assertEqual(saved["capabilities"], {})
        self.assertEqual(effective_project_type(self.first).code_graph,
                         defaults["code_graph"])
        self.assertEqual(effective_project_type(self.first).description,
                         "Separate global edit")

    def test_indexing_capability_values_are_strictly_validated(self):
        current = project_behavior_settings(self.first)
        defaults = current["capabilities"]["values"]
        for invalid in (
            {**defaults, "code_graph": "enabled"},
            {**defaults, "document_headers": 1},
            {**defaults, "source_cards": {"enabled": False}},
            {"code_graph": False},
        ):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    save_project_behavior(
                        self.first, "capabilities", invalid,
                        expected_revision=current["revision"],
                    )
                self.assertFalse((self.first / PROJECT_BEHAVIOR_FILE).exists())
        with self.assertRaisesRegex(ValueError, "changed since"):
            save_project_behavior(self.first, "capabilities", defaults,
                                  expected_revision="obsolete")

    def test_index_compatibility_metadata_tracks_capabilities(self):
        from kilin.rag import RAGSession
        session = object.__new__(RAGSession)
        session.project = kilin.open_project(self.first)
        session.backend = "mlx"
        previous = session._index_metadata()
        current = project_behavior_settings(self.first)
        save_project_behavior(
            self.first, "capabilities",
            {"code_graph": False, "document_headers": False},
            expected_revision=current["revision"],
        )
        changed = session._index_metadata()
        self.assertEqual(changed["code_graph"], False)
        self.assertNotIn("document_headers", changed)
        self.assertNotEqual(previous, changed)
        self.assertEqual(session._index_metadata(), changed)

    def test_corrupted_or_foreign_overrides_fail_closed(self):
        path = self.first / PROJECT_BEHAVIOR_FILE
        for value in (
            '{"format":"project-behavior-v1","project_type_id":'
            '"code","global":{},"intents":{}}',
            '{"format":"project-behavior-v1","project_type_id":'
            '"literature","global":{},"intents":{"other":{}}}',
            '{"format":"project-behavior-v1","project_type_id":'
            '"literature","global":{},"intents":{},"unexpected":1}',
            '{"format":"project-behavior-v1","project_type_id":'
            '"literature","global":{},"global":{},"intents":{}}',
            '{not valid json',
        ):
            with self.subTest(value=value):
                path.write_text(value, encoding="utf-8")
                with self.assertRaises(ValueError):
                    effective_project_type(self.first)


if __name__ == "__main__":
    unittest.main()
