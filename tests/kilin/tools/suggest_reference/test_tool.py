"""Test the suggest_reference tool package without live model backends.

Check configuration ownership, the registry entry and intent, settings
validation, reading the statement, choosing the papers, and a full
model-free run on a small project.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from kilin.project_type import project_type
from kilin.tools import suggest_reference as tool
from kilin.tools._contract import ToolRequest
from kilin.tools._registry import TOOL_REGISTRY, TOOL_SUGGEST_REFERENCE
from tests.kilin.tools._shared._support import chunk

STATEMENT = "anomalous diffusion is common in living cells"
DRAFT = (
    "Introduction\nWe note that anomalous diffusion is common in living "
    "cells, as many studies show.\n"
)
REVIEW = (
    "Anomalous diffusion has been observed in the cytoplasm of many living "
    "cells. It is common in crowded environments.\n"
)
REPEAT = (
    "As shown in [12], anomalous diffusion is frequent in living cells. We "
    "use this to motivate our tracking method.\n"
)
OTHER = "Graph theory measures brain networks.\n"


class TestSuggestReferencePackage(unittest.TestCase):
    """Configuration, registry and intent."""

    def test_registry_uses_package_entry_points(self) -> None:
        entry = TOOL_REGISTRY[TOOL_SUGGEST_REFERENCE]
        self.assertIs(entry.parse_settings, tool.parse_settings)
        self.assertIs(entry.run, tool.run)
        self.assertIsNone(entry.workspace_resources)
        self.assertFalse(entry.follows_up)

    def test_packaged_defaults_are_documented(self) -> None:
        package = Path(tool.__file__).parent
        data = json.loads((package / "defaults.json").read_text())
        readme = (package / "README.md").read_text()
        for section, values in data.items():
            with self.subTest(section=section):
                self.assertIn(f"`{section}`", readme)
            for key in values:
                with self.subTest(key=f"{section}.{key}"):
                    self.assertIn(f"`{key}`", readme)
        defaults = tool.SuggestReferenceSettings()
        self.assertEqual(defaults, tool.parse_settings({}, "example"))
        self.assertEqual(
            {f.name for f in fields(defaults)},
            {name.lower() for name in data["SETTINGS"]},
        )

    def test_invalid_project_type_settings(self) -> None:
        for values in (
            {"files": 0}, {"unknown": 1}, {"passages": 21},
            {"confirmation": "sometimes"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_literature_intent_uses_the_tool(self) -> None:
        literature = project_type("literature")
        intent = literature.intents["suggest_reference"]
        self.assertEqual(intent.tool.name, TOOL_SUGGEST_REFERENCE)


class TestStatement(unittest.TestCase):
    """The statement in the question."""

    def test_quoted_or_after_a_colon(self) -> None:
        router = Mock()
        self.assertEqual(
            tool.statement_of(f"which paper supports “{STATEMENT}”?", router),
            STATEMENT,
        )
        self.assertEqual(
            tool.statement_of(f"suggest a reference for: {STATEMENT}", router),
            STATEMENT,
        )
        router.assert_not_called()

    def test_router_copy_must_be_the_users_words(self) -> None:
        question = f"which paper shows that {STATEMENT}?"
        self.assertEqual(
            tool.statement_of(question, Mock(return_value=STATEMENT)),
            STATEMENT,
        )
        # An answer in other words, or none, falls back to the question.
        for reply in ("Diffusion is weird inside bacteria and yeast", "NONE"):
            with self.subTest(reply=reply):
                self.assertEqual(
                    tool.statement_of(question, Mock(return_value=reply)),
                    question,
                )
        broken = Mock(side_effect=RuntimeError("down"))
        self.assertEqual(tool.statement_of(question, broken), question)


class TestSuggestReferenceRun(unittest.TestCase):
    """A full run on a small Markdown project."""

    def setUp(self) -> None:
        self.chunks = [
            chunk("d1", DRAFT, path="draft.md"),
            chunk("r1", REVIEW, path="review.md"),
            chunk("p1", REPEAT, path="repeat.md"),
            chunk("o1", OTHER, path="other.md"),
        ]

    def _run(
        self, question: str = f"suggest a reference for: {STATEMENT}",
        choice: str = "continue", **settings: object,
    ) -> tuple[object, SimpleNamespace]:
        def ask_model(system: str, prompt: str, tokens: int) -> str:
            if "(review.md)" in prompt:
                return (
                    "VERDICT: supported\nQUOTE: Anomalous diffusion has been "
                    "observed in the cytoplasm of many living cells.\n"
                    "COMMENT: States it [2]."
                )
            if "(repeat.md)" in prompt:
                return (
                    "VERDICT: partly supported\nQUOTE: NONE\nCOMMENT: Repeats "
                    "it citing [12]; the original source is elsewhere."
                )
            return "VERDICT: not found\nQUOTE: NONE\nCOMMENT: Off topic."

        context = SimpleNamespace(
            chunks=self.chunks,
            progress=Mock(), done=Mock(),
            # The draft ranks first, then the others in order.
            rank=lambda text: [SimpleNamespace(chunk=c) for c in self.chunks],
            ask_model=Mock(side_effect=ask_model),
            ask_router=Mock(return_value="NONE"),
            confirm=Mock(return_value=choice),
            trace=SimpleNamespace(emit=Mock()), trace_turn_id=None,
        )
        request = ToolRequest(question, question)
        return tool.run(
            tool.SuggestReferenceSettings(**settings), request, context
        ), context

    def test_suggestions_with_quotes(self) -> None:
        result, context = self._run()
        # The draft contains the statement: left out, not judged.
        self.assertEqual(context.ask_model.call_count, 3)
        self.assertEqual(result.summary["left_out"], ["draft.md"])
        self.assertEqual(result.summary["suggested"], 2)
        answer = result.answer
        self.assertIn(f"**Statement:** “{STATEMENT}”", answer)
        self.assertIn("**2 of the 3 papers checked could be cited", answer)
        self.assertIn("1. **Supported** · `review.md`", answer)
        self.assertIn("2. **Partly supported** · `repeat.md`", answer)
        self.assertLess(answer.index("review.md"), answer.index("repeat.md"))
        self.assertIn(
            "*It says:* “Anomalous diffusion has been observed in the "
            "cytoplasm of many living cells” [1]",
            answer,
        )
        # Bracketed numbers from the model are not evidence links.
        self.assertIn("States it (2).", answer)
        self.assertIn("citing (12)", answer)
        self.assertIn("Also checked: `other.md` (not found).", answer)
        self.assertIn(
            "Not checked, as they contain the statement word for word "
            "(your draft, or the paper it comes from): `draft.md`.",
            answer,
        )
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0]["path"], Path("review.md"))
        context.confirm.assert_not_called()  # 3 papers: a small plan.
        events = [c.args[0] for c in context.trace.emit.call_args_list]
        self.assertEqual(events.count("tool.suggest_reference.paper"), 3)

    def test_no_supporting_paper(self) -> None:
        self.chunks = [chunk("o1", OTHER, path="other.md")]
        result, _ = self._run()
        self.assertIn("None of the 1 papers most related", result.answer)
        self.assertIn("add the paper you have in mind", result.answer)
        self.assertEqual(result.evidence, ())

    def test_plan_quick_and_cancel(self) -> None:
        result, context = self._run(
            choice="quick", confirmation="always", quick_files=1
        )
        context.confirm.assert_called_once()
        self.assertEqual(context.ask_model.call_count, 1)
        self.assertIn("Only the 1 most related of 3 papers", result.answer)
        result, context = self._run(choice="cancel", confirmation="always")
        context.ask_model.assert_not_called()
        self.assertIn("Not checked (cancelled)", result.answer)

    def test_no_statement(self) -> None:
        result, context = self._run(question="suggest a reference")
        context.ask_model.assert_not_called()
        self.assertIn("Which sentence needs a reference?", result.answer)


if __name__ == "__main__":
    unittest.main()
