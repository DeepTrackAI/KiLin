"""Verify production tool requests do not restore hidden output ceilings.

Exercise prose, long identifier lists and report repairs with deterministic
callbacks. Only the bounded metadata requests retain local output caps.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin.tools._contract import EarlierTurn, ToolRequest
from kilin.tools._registry import TOOL_REGISTRY
from kilin.tools.find_passages import tool as passages
from kilin.tools.generate_paper_summary import tool as summary
from kilin.tools.paper_overview import tool as overview
from kilin.tools.show_source import tool as source
from tests.kilin.tools._shared._support import chunk


class TestProductionBudgets(unittest.TestCase):
    """Distinguish generation limits from content and planning limits."""

    def test_obsolete_caps_are_rejected_not_silently_ignored(self) -> None:
        obsolete = {
            "paper_overview": ("answer_tokens",),
            "show_source": ("answer_tokens",),
            "find_fact": ("answer_tokens", "note_tokens"),
            "batch_review": ("answer_tokens", "note_tokens"),
            "generate_paper_summary": ("note_tokens", "section_tokens"),
            "reference_check": ("answer_tokens",),
            "suggest_reference": ("answer_tokens",),
        }
        for name, keys in obsolete.items():
            for key in keys:
                for value in (None, 128, 8192):
                    with self.subTest(tool=name, key=key, value=value):
                        with self.assertRaises(ValueError):
                            TOOL_REGISTRY[name].parse_settings(
                                {key: value}, "custom"
                            )

    def test_overview_and_source_explanation_inherit(self) -> None:
        text = "The indexed paper reports a reproducible software result."
        piece = chunk("p1", text, path="paper.pdf")
        context = SimpleNamespace(
            chunks=[piece], progress=Mock(), done=Mock(),
            rank=Mock(return_value=[]), write=Mock(return_value="Summary [1]"),
            ask_model=Mock(return_value="Explanation [1]"),
        )
        request = ToolRequest("Summarize paper.pdf", "Summarize paper.pdf")
        with patch.object(overview, "choose_document",
                          return_value=(piece.path, "named", False)):
            result = overview.run(overview.OverviewSettings(), request,
                                  context)
        self.assertEqual(context.write.call_args.args[-1], None)
        self.assertEqual(result.answer, "Summary [1]")
        self.assertEqual(len(result.evidence), 1)
        earlier = EarlierTurn(
            1, request.question, paths=(piece.path,), texts=(text,),
            lines=(1,), answer=result.answer,
        )
        request = ToolRequest("Show its source", "Show its source",
                              earlier=(earlier,))
        result = source.run(source.SourceSettings(), request, context)
        self.assertEqual(context.ask_model.call_args.args[-1], None)
        self.assertIn("Explanation [1]", result.answer)
        self.assertEqual(len(result.evidence), 1)

    def test_long_sentence_selection_does_not_use_small_cap(self) -> None:
        sentences = passages.split_sentences([
            chunk(str(index),
                  f"Result number {index} is scientifically useful.")
            for index in range(256)
        ])
        self.assertEqual(len(sentences), 256)
        context = SimpleNamespace(
            progress=Mock(), ask_model=Mock(
                return_value=", ".join(str(i) for i in range(1, 257))
            ),
        )
        selected = passages._judge("results", [sentences], "Find", context)
        self.assertEqual(selected, sentences)
        self.assertEqual(context.ask_model.call_count, 1)
        self.assertIsNone(context.ask_model.call_args.args[-1])

    def test_report_prose_and_repairs_inherit_but_metadata_is_bounded(self):
        template = summary.load_template(
            "research_briefing", summary.SHIPPED_TEMPLATES_DIR
        )
        section = next(s for s in template.sections if s.words)
        valid = " ".join(["Evidence"] * 80) + "."
        notes = [(1, ["A supported observation [1]."])]
        context = SimpleNamespace(
            progress=Mock(), trace=Mock(), trace_turn_id=None,
            ask_model=Mock(side_effect=["Too short.", valid]),
        )
        content, repairs = summary._write_section(
            template, section, "paper.pdf", notes, [],
            summary.SummarySettings(), context,
        )
        self.assertEqual(content, valid)
        self.assertEqual(len(repairs), 1)
        self.assertEqual(
            [call.args[-1] for call in context.ask_model.call_args_list],
            [None, None],
        )
        context.ask_model = Mock(return_value="A scientific finding")
        headline, repairs = summary._write_headline(
            template, "paper.pdf", notes, context
        )
        self.assertEqual(headline, "A scientific finding")
        self.assertEqual(repairs, [])
        self.assertEqual(context.ask_model.call_args.args[-1], 128)
        context.ask_model = Mock(return_value="1, 2")
        references = next(s for s in template.sections
                          if s.source == "references")
        content, error = summary._fill_references(
            template, references, "paper.pdf", notes,
            ["First reference", "Second reference", "Third reference"],
            context,
        )
        self.assertIsNone(error)
        self.assertEqual(content, "- First reference\n- Second reference")
        self.assertEqual(context.ask_model.call_args.args[-1], 128)


if __name__ == "__main__":
    unittest.main()
