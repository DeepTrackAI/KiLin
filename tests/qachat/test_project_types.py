"""Check project types loaded from project_types.json."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.project import default_project_profile
from qachat.project_types import (
    PROJECT_TYPES_FILE,
    load_project_types,
    project_type,
)
from qachat.query_formats import (
    LITERATURE_REPOSITORY_INSTRUCTION,
    SOFTWARE_REPOSITORY_INSTRUCTION,
)
from qachat.rag import _SYSTEM_PROMPT


class TestProjectTypes(unittest.TestCase):
    """The packaged file reproduces the former built-in behavior."""

    def test_built_in_types_keep_previous_defaults(self) -> None:
        code = project_type("code")
        literature = project_type("literature")
        self.assertEqual(
            code.retrieval_instruction, SOFTWARE_REPOSITORY_INSTRUCTION
        )
        self.assertEqual(
            literature.retrieval_instruction,
            LITERATURE_REPOSITORY_INSTRUCTION,
        )
        self.assertEqual(code.system_prompt, _SYSTEM_PROMPT)
        self.assertEqual(literature.system_prompt, _SYSTEM_PROMPT)
        self.assertEqual(
            default_project_profile("code").extensions,
            (".py", ".ipynb", ".m", ".md"),
        )
        self.assertEqual(
            default_project_profile("literature").max_file_bytes, 20 * 2**20
        )

    def _write(self, folder: str, name: str, value: object) -> Path:
        # Loading is cached per path, so each case uses its own file.
        path = Path(folder) / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_added_type_loads_and_bad_files_explain_the_problem(self) -> None:
        packaged = json.loads(PROJECT_TYPES_FILE.read_text(encoding="utf-8"))
        with TemporaryDirectory() as folder:
            added = {
                **packaged,
                "protocols": {
                    "extensions": [".pdf", ".docx"],
                    "max_file_bytes": 1000,
                    "retrieval_instruction": "Find protocol steps.",
                    "system_prompt": "Answer from the protocols.",
                    "display_name": "Clinical protocols",
                    "description": "Clinical study protocols.",
                    "intents": {
                        "find_step": {
                            "description": "Find a protocol step.",
                            "system_prompt": "Quote the step.",
                        }
                    },
                },
            }
            types = load_project_types(
                self._write(folder, "added.json", added)
            )
            self.assertEqual(list(types)[-1], "protocols")
            self.assertEqual(types["protocols"].answer_guidance, "")

            # qachat.project owns the built-in kinds and requires them.
            from qachat.project import _check_built_in_kinds

            missing = load_project_types(
                self._write(folder, "missing.json", {"code": packaged["code"]})
            )
            with self.assertRaisesRegex(ValueError, "literature"):
                _check_built_in_kinds(missing)

            bad_shortcut = json.loads(json.dumps(packaged))
            bad_shortcut["code"]["intents"]["small_talk"]["shortcuts"] = ["("]
            with self.assertRaisesRegex(ValueError, "regular expression"):
                load_project_types(
                    self._write(folder, "shortcut.json", bad_shortcut)
                )

            budget = json.loads(json.dumps(packaged))
            reading = budget["code"]["intents"]["file_reading"]
            self.assertEqual(reading["retrieval"]["mode"], "document")
            reading["retrieval"]["max_tokens"] = 50
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                load_project_types(self._write(folder, "budget.json", budget))

            listing = packaged["literature"]["intents"]["mention_listing"]
            self.assertEqual(listing["workflow"]["kind"], "scan_and_list")
            for change, message in (
                ({"kind": "summarize"}, "workflow kind"),
                ({"max_model_calls": 3}, "unknown settings"),
                ({"context_lines": 9}, "context_lines"),
                ({"whole_words": "yes"}, "whole_words"),
            ):
                bad = json.loads(json.dumps(packaged))
                bad["literature"]["intents"]["mention_listing"][
                    "workflow"
                ].update(change)
                with self.assertRaisesRegex(ValueError, message):
                    load_project_types(
                        self._write(folder, f"{message}.json", bad)
                    )
            sorting = packaged["code"]["intents"]["mention_classification"]
            self.assertEqual(sorting["workflow"]["kind"], "batch_classify")
            for intent, change, message in (
                # Batch settings belong to batch_classify only.
                ("mention_listing", {"batch_tokens": 900}, "unknown"),
                ("mention_classification", {"categories": ["one"]}, "2-8"),
                ("mention_classification", {"confirmation": "maybe"},
                 "confirmation"),
                ("mention_classification", {"candidates": "chunks"},
                 "candidates"),
                ("mention_classification", {"max_model_calls": 0},
                 "max_model_calls"),
            ):
                bad = json.loads(json.dumps(packaged))
                bad["code"]["intents"][intent]["workflow"].update(change)
                with self.assertRaisesRegex(ValueError, message):
                    load_project_types(
                        self._write(folder, f"c-{message}.json", bad)
                    )
            fixed = json.loads(json.dumps(packaged))
            fixed["code"]["intents"]["mention_classification"]["workflow"][
                "categories"
            ] = ["tests", "library"]
            loaded = load_project_types(
                self._write(folder, "fixed.json", fixed)
            )
            self.assertEqual(
                loaded["code"]
                .intents["mention_classification"]
                .workflow.categories,
                ("tests", "library"),
            )

            searching = json.loads(json.dumps(packaged))
            searching["literature"]["intents"]["mention_listing"][
                "retrieval"
            ] = {"mode": "search"}
            with self.assertRaisesRegex(ValueError, "replaces retrieval"):
                load_project_types(
                    self._write(folder, "searching.json", searching)
                )

            broken = {**packaged, "notes": {**packaged["code"]}}
            broken["notes"]["max_file_bytes"] = "big"
            with self.assertRaisesRegex(ValueError, "max_file_bytes"):
                load_project_types(self._write(folder, "broken.json", broken))


if __name__ == "__main__":
    unittest.main()
