"""Test the reference_check tool package without live model backends.

Check configuration ownership, the registry entry and intent, settings
validation, and a full model-free run on a small paper and project.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from kilin.project_type import project_type
from kilin.tools import reference_check as tool
from kilin.tools._contract import ToolRequest
from kilin.tools._registry import TOOL_REFERENCE_CHECK
from kilin.tools.reference_check import _configuration as config
from kilin.tools.reference_check._resolution import offline_link
from kilin.tools._shared.sentences import Sentence
from kilin.tools.reference_check._citations import Citation
from kilin.tools.reference_check.tool import (
    _citation_context,
    _cited_for,
)
from kilin.tools._shared.sentences import split_sentences
from tests.kilin.tools._contract_support import (
    assert_packaged_defaults,
    assert_registry_entry,
)
from tests.kilin.tools._shared._support import chunk

PAPER = """Introduction
Brain networks can be analysed with graph theory [1]. Software such as
BRAPH implements these measures [2, 3]. Deep learning helps too [4].
Methods
We followed the protocol of [2].
References
1. Sporns, O. The human connectome: a structural description. PLoS
Comput. Biol. 1, e42 (2005).
2. Mijalkov, M. et al. BRAPH: A graph theory software for the analysis of
brain connectivity. PLoS ONE 12, e0178798 (2017).
3. Chang, Y.-W. et al. BRAPH 2: a flexible, open-source framework for
network analyses in neurosciences. bioRxiv (2025).
4. LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521,
436–444 (2015).
5. Never, C. A reference the text does not cite at all. Cell 1, 2 (2020).
"""
BRAPH = """BRAPH: A graph theory software for the analysis of brain
connectivity
Mite Mijalkov, Ehsan Kakaei, Joana B. Pereira
Abstract. Graph theory ...
"""
# Another version of the cited preprint: same title and server, another
# first author, so it is only probable.
BRAPH2 = """bioRxiv preprint
BRAPH 2: A flexible, open-source framework for network analyses
in neurosciences
Iok-ui Tiunn, Blanca Zufiria-Gerboles
"""


class TestReferenceCheckPackage(unittest.TestCase):
    """Configuration, registry and intent."""

    def test_registry_uses_package_entry_points(self) -> None:
        assert_registry_entry(
            self, "reference_check", tool, non_followup=True
        )

    def test_packaged_defaults_are_documented(self) -> None:
        data = assert_packaged_defaults(
            self, tool, tool.ReferenceCheckSettings, check_values=False,
        )
        self.assertEqual(
            config.REFERENCE_LAYOUT.max_characters,
            data["REFERENCES"]["REFERENCE_CHARACTERS"],
        )
        self.assertEqual(
            config.MATCH_RULES.common_words,
            frozenset(data["MATCHING"]["COMMON_WORDS"]),
        )

    def test_verdict_tokens_inherit_session_by_default(self) -> None:
        """An omitted limit never imposes the historical 250-token cap."""

        settings = tool.parse_settings({}, "test")
        self.assertNotIn("answer_tokens", {f.name for f in fields(settings)})
        for obsolete in (None, 4096, 0, -1, "250", False, 131073):
            with self.subTest(value=obsolete):
                with self.assertRaisesRegex(ValueError, "unknown settings"):
                    tool.parse_settings({"answer_tokens": obsolete}, "test")

    def test_invalid_project_type_settings(self) -> None:
        for values in (
            {"max_listed": 0}, {"unknown": 1}, {"verify": "yes"},
            {"passages": 0}, {"confirmation": "sometimes"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tool.parse_settings(values, "example")

    def test_literature_intent_uses_the_tool(self) -> None:
        literature = project_type("literature")
        intent = literature.intents["reference_check"]
        self.assertEqual(intent.tool.name, TOOL_REFERENCE_CHECK)


class TestReferenceCheckRun(unittest.TestCase):
    """A full run on a small Markdown project."""

    def setUp(self) -> None:
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.chunks = [
            chunk("p1", PAPER, path="paper.md"),
            chunk("b1", BRAPH, path="braph.md"),
            chunk("c1", BRAPH2, path="braph2.md"),
        ]

    def tearDown(self) -> None:
        self.folder.cleanup()

    def _run(
        self, chosen: object, choice: str = "continue",
        question: str = "check the references of paper.md",
        interactive: bool = True, choose: object = None,
        ask: object = None, model_attempt_count: object = None,
        **settings: object
    ) -> tuple[object, SimpleNamespace]:
        def ask_model(system: str, prompt: str, tokens: int) -> str:
            # Reference [2] is quoted from its file; [3] is called
            # contradicted without a quote.
            if "reference [2]" in prompt:
                return (
                    "VERDICT: partly supported\nQUOTE: Mite Mijalkov, Ehsan "
                    "Kakaei, Joana B. Pereira\nCOMMENT: Names the authors."
                )
            return "VERDICT: contradicted\nQUOTE: NONE\nCOMMENT: Disagrees."

        context = SimpleNamespace(
            project_root=self.root, chunks=self.chunks,
            progress=Mock(), done=Mock(), choose=choose,
            rank=Mock(side_effect=lambda text: [
                SimpleNamespace(chunk=c) for c in self.chunks
            ]),
            ask_model=Mock(side_effect=ask or ask_model),
            model_attempt_count=model_attempt_count,
            confirm=Mock(return_value=choice),
            trace=SimpleNamespace(emit=Mock()), trace_turn_id=None,
        )
        request = ToolRequest(question, question)
        if not interactive:
            context.confirm = None
        values = {"max_listed": 1, "confirmation": "always", **settings}
        with patch(
            "kilin.tools.reference_check.tool.choose_document",
            return_value=chosen,
        ):
            return tool.run(
                tool.ReferenceCheckSettings(**values), request, context
            ), context

    def test_report_and_answer(self) -> None:
        result, context = self._run((Path("paper.md"), "named", False))
        summary = result.summary
        self.assertEqual(summary["references"], 5)
        self.assertEqual(summary["numbers_printed"], 5)
        self.assertEqual(
            (summary["found"], summary["probable"], summary["missing"]),
            (1, 1, 3),
        )
        self.assertEqual(summary["citation_style"], "brackets")
        self.assertEqual(summary["uncited"], [5])
        self.assertIn("**2.** Mijalkov", result.answer)
        self.assertIn("→ `braph.md`", result.answer)
        self.assertIn("another version", result.answer)
        # One missing reference is listed, the others counted.
        self.assertIn("… and 2 more", result.answer)
        self.assertIn("Never cited in the text: 5", result.answer)
        self.assertNotIn("[2]", result.answer)
        context.done.assert_called_once()

        report = (self.root / summary["report"]).read_text(encoding="utf-8")
        self.assertTrue(report.startswith("# Reference check: paper.md"))
        section = report[report.index("### 2."):report.index("### 3.")]
        self.assertIn(
            "`braph.md` (found: first page and first author)", section
        )
        self.assertIn(
            "Software such as BRAPH implements these "
            "measures **⁽²,³⁾**.",
            section,
        )
        self.assertIn("We followed the protocol of **⁽²⁾**.", section)
        self.assertIn("not found in the text", report)

    def test_claims_are_checked_after_confirmation(self) -> None:
        result, context = self._run((Path("paper.md"), "named", False))
        context.confirm.assert_called_once()
        # [2] is cited twice, [3] once: three claims, one call each, and
        # one call splitting the sentence citing [2, 3] between them.
        self.assertEqual(context.ask_model.call_count, 4)
        self.assertEqual(result.summary["claims_checked"], 3)
        self.assertEqual(
            result.summary["verdicts"],
            {"partly supported": 2, "not supported": 1},
        )
        # The unconfirmed contradiction is listed first, weakened.
        self.assertIn(
            "**1. Not supported** · reference 3 → "
            "`braph2.md`", result.answer
        )
        self.assertIn("2 partly supported · 1 not supported", result.answer)
        self.assertIn("Judged contradicted", result.answer)
        self.assertEqual(len(result.evidence), 0)  # max_listed hides [2].
        report = (
            self.root / result.summary["report"]
        ).read_text(encoding="utf-8")
        self.assertIn("## Claims checked against the cited papers", report)
        self.assertIn(
            "**Verdict:** partly supported · ★★★☆☆ (50/100) "
            "— Names the authors.", report
        )
        self.assertIn("★★★☆☆ (50/100)", report)
        self.assertIn("☆☆☆☆☆ (unrated)", result.answer)
        self.assertEqual(len(result.summary["suitability"]), 3)
        self.assertEqual(
            [rating["stars"] for rating in result.summary["suitability"]],
            [3.0, None, 3.0],
        )
        record = next(
            call.kwargs["data"]
            for call in context.trace.emit.call_args_list
            if call.args[0] == "tool.reference_check.claim"
            and call.kwargs["data"]["reference"] == 2
        )
        self.assertIn("citation_sentence", record)
        self.assertIn("preceding_context", record)
        self.assertIn("following_context", record)
        self.assertIn("retrieval_queries", record)
        self.assertEqual(record["passages"][0]["id"], "b1")
        self.assertIn("quote_passage_id", record)
        self.assertEqual(record["suitability"]["stars"], 3.0)
        self.assertIn("*Preceding context used:*", report)
        self.assertIn("*Evidence retrieval queries:*", report)
        events = [c.args[0] for c in context.trace.emit.call_args_list]
        self.assertEqual(events.count("tool.reference_check.claim"), 3)

    def test_checked_claim_limit_and_actual_model_calls(self) -> None:
        """The claim cap is distinct from invocation counts and retries."""

        result, context = self._run(
            (Path("paper.md"), "named", False),
            max_checked_claims=1, retry_passages=0,
        )
        self.assertEqual(result.summary["claims_checked"], 1)
        self.assertEqual(result.summary["model_calls"], {
            "focusing": 1,
            "initial": 1,
            "second_look": 0,
            "total": 2,
        })
        self.assertEqual(context.ask_model.call_count, 2)
        self.assertIn("2 of 3 claims were not checked", result.answer)
        self.assertIn("**Model requests:** 2 (1 focusing, 1 initial, "
                      "0 second looks)", result.answer)
        self.assertNotIn("model_attempts", result.summary)
        plan = context.confirm.call_args.args[0]
        self.assertEqual(plan.items, 3)
        self.assertEqual(plan.batches_to_run, 2)
        self.assertEqual(plan.batches, 4)
        self.assertEqual(plan.quick_items, 1)
        self.assertEqual(plan.estimated_seconds, plan.quick_seconds)
        report = (self.root / result.summary["report"]).read_text(
            encoding="utf-8"
        )
        self.assertIn("**Model requests:** 2", report)

    def test_generation_attempts_include_recovery_by_stage(self) -> None:
        """Count adapter recovery separately from logical requests."""

        attempts = {"count": 0}

        def answer(system: str, prompt: str, tokens: object) -> str:
            """Simulate one recovery during a focusing request."""

            attempts["count"] += (
                2 if "citing several references" in prompt else 1
            )
            if "citing several references" in prompt:
                return "[2]: BRAPH implements these measures."
            return "VERDICT: not supported\nQUOTE: NONE\nCOMMENT: None."

        result, context = self._run(
            (Path("paper.md"), "named", False),
            ask=answer, model_attempt_count=lambda: attempts["count"],
            max_checked_claims=1, retry_passages=0,
        )
        self.assertEqual(result.summary["model_calls"], {
            "focusing": 1, "initial": 1, "second_look": 0, "total": 2,
        })
        self.assertEqual(result.summary["model_attempts"], {
            "focusing": 2, "initial": 1, "second_look": 0, "total": 3,
        })
        self.assertEqual(context.ask_model.call_count, 2)
        self.assertIn("**Generation attempts:** 3 (2 focusing", result.answer)
        report = (self.root / result.summary["report"]).read_text(
            encoding="utf-8"
        )
        self.assertIn("**Generation attempts:** 3", report)
        claim_events = [
            call.kwargs["data"]
            for call in context.trace.emit.call_args_list
            if call.args[0] == "tool.reference_check.claim"
        ]
        self.assertEqual(claim_events[0]["model_calls"], 2)
        self.assertEqual(claim_events[0]["model_attempts"], 3)

    def test_failed_focusing_attempt_is_still_accounted_for(self) -> None:
        """A failed hint request remains an attempted inference."""

        attempts = {"count": 0}

        def answer(system: str, prompt: str, tokens: object) -> str:
            """Fail focusing, then complete the original verification."""

            attempts["count"] += 1
            if "citing several references" in prompt:
                raise RuntimeError("Focus failed")
            return "VERDICT: not supported\nQUOTE: NONE\nCOMMENT: None."

        result, _ = self._run(
            (Path("paper.md"), "named", False),
            ask=answer, model_attempt_count=lambda: attempts["count"],
            max_checked_claims=1, retry_passages=0,
        )
        self.assertEqual(result.summary["model_attempts"], {
            "focusing": 1, "initial": 1, "second_look": 0, "total": 2,
        })

    def test_claim_cap_also_applies_to_quick_selection(self) -> None:
        """Choosing quick cannot select beyond the configured claim cap."""

        result, _ = self._run(
            (Path("paper.md"), "named", False), choice="quick",
            quick_claims=5, max_checked_claims=1, retry_passages=0,
        )
        self.assertEqual(result.summary["claims_checked"], 1)

    def test_old_model_call_limit_is_rejected(self) -> None:
        """The renamed setting must not silently acquire two meanings."""

        with self.assertRaisesRegex(ValueError, "unknown settings"):
            tool.parse_settings({"max_model_calls": 1}, "test")
        self.assertEqual(
            tool.parse_settings(
                {"max_checked_claims": 1}, "test"
            ).max_checked_claims, 1,
        )

    def test_one_claim_can_use_three_counted_model_calls(self) -> None:
        """Focusing, checking and rereading do not consume claim slots."""

        self.chunks[1:2] = [
            chunk(
                f"b{number}",
                ("BRAPH implements these graph measures."
                 if number == 10 else BRAPH if number == 1
                 else f"Unrelated passage {number}."),
                path="braph.md", line=number,
            )
            for number in range(1, 11)
        ]

        def answer(system: str, prompt: str, tokens: object) -> str:
            if "citing several references" in prompt:
                return "[2]: BRAPH implements graph measures."
            if "BRAPH implements these graph measures." in prompt:
                return (
                    "VERDICT: supported\nQUOTE: BRAPH implements these "
                    "graph measures.\nCOMMENT: Direct evidence."
                )
            return "VERDICT: not supported\nQUOTE: NONE\nCOMMENT: None."

        result, context = self._run_model(
            answer, max_checked_claims=1, focus_claims=True,
            passages=6, retry_passages=8,
        )
        self.assertEqual(result.summary["claims_checked"], 1)
        self.assertEqual(result.summary["model_calls"], {
            "focusing": 1,
            "initial": 1,
            "second_look": 1,
            "total": 3,
        })
        self.assertEqual(context.ask_model.call_count, 3)
        events = [
            event.kwargs["data"]
            for event in context.trace.emit.call_args_list
            if event.args[0] == "tool.reference_check.claim"
        ]
        self.assertEqual(events[0]["quote_passage_id"], "b10")
        self.assertEqual(events[0]["verdict_stage"], "second_look")
        self.assertEqual(events[0]["model_calls"], 3)

    def test_claim_rank_queries_are_cached_within_run(self) -> None:
        """Repeated citation queries need only one retrieval per string."""

        _, context = self._run(
            (Path("paper.md"), "named", False), max_listed=5,
        )
        queries = [
            call.args[0] for call in context.rank.call_args_list
        ]
        self.assertTrue(queries)
        self.assertEqual(len(queries), len(set(queries)))

    def test_probable_reference_and_evidence_status_are_reported(
        self,
    ) -> None:
        """A model verdict cannot hide uncertainty in the source file."""

        result, context = self._run(
            (Path("paper.md"), "named", False), max_listed=5,
        )
        probable = next(
            item for item in result.summary["suitability"]
            if item["reference"] == 3
        )
        self.assertEqual(probable["match_status"], "probable")
        self.assertEqual(probable["evidence_status"], "not required")
        self.assertIn("Provisional:", result.answer)
        trace = [
            call.kwargs["data"]
            for call in context.trace.emit.call_args_list
            if call.args[0] == "tool.reference_check.claim"
        ]
        self.assertEqual(
            next(item for item in trace if item["reference"] == 3)
            ["match_status"], "probable",
        )

    def test_all_checked_stars_appear_when_chat_limit_allows(self) -> None:
        result, _ = self._run(
            (Path("paper.md"), "named", False), max_listed=5,
        )
        self.assertIn("★★★☆☆ (50/100)", result.answer)
        self.assertIn("☆☆☆☆☆ (unrated)", result.answer)
        self.assertIn("heuristic", result.answer)

    def test_displayed_scores_match_summary_and_trace(self) -> None:
        result, context = self._run(
            (Path("paper.md"), "named", False), max_listed=5,
        )
        report = (self.root / result.summary["report"]).read_text(
            encoding="utf-8"
        )
        scores = result.summary["suitability"]
        self.assertEqual([item["score"] for item in scores], [50, None, 50])
        claims = [
            call.kwargs["data"] for call in context.trace.emit.call_args_list
            if call.args[0] == "tool.reference_check.claim"
        ]
        for item, record in zip(scores, claims):
            if item["score"] is None:
                self.assertIsNone(record["suitability"])
                continue
            self.assertEqual(item["score"], record["suitability"]["score"])
            suffix = f"({item['score']}/100)"
            self.assertIn(suffix, result.answer)
            self.assertIn(suffix, report)
        self.assertNotIn("(0/100)", result.answer)
        self.assertNotIn("(0/100)", report)
        self.assertEqual(context.ask_model.call_count, 4)

    def test_two_preceding_sentences_resolve_this_study(self) -> None:
        original = (
            "Software such as\nBRAPH implements these measures [2, 3]."
        )
        replacement = (
            "The second case study implements recursive feature "
            "elimination for schizophrenia classification. Most building "
            "blocks, including the classifiers, are reused from the "
            "standard distribution. The results of this study were "
            "published [2]. As shown in Figure 3, the project adds a "
            "single generator file."
        )
        self.chunks[0] = chunk(
            "p1", PAPER.replace(original, replacement), path="paper.md"
        )
        result, context = self._run((Path("paper.md"), "named", False))
        self.assertGreater(result.summary["claims_checked"], 0)
        prompt = next(
            call.args[1] for call in context.ask_model.call_args_list
            if '"The results of this study were published [2]."' in
            call.args[1]
        )
        self.assertIn("The second case study implements", prompt)
        self.assertIn("Most building blocks", prompt)
        self.assertIn("Following context: ", prompt)
        self.assertIn("As shown in Figure 3", prompt)
        self.assertIn("Judge only what the cited sentence", prompt)
        queries = [call.args[0] for call in context.rank.call_args_list]
        self.assertTrue(any(
            "schizophrenia classification" in query
            and "results of this study were published" not in query
            for query in queries
        ))
        report = (
            self.root / result.summary["report"]
        ).read_text(encoding="utf-8")
        self.assertIn("*Preceding context used:*", report)
        self.assertIn("*Following context used:*", report)
        self.assertIn("*Evidence retrieval queries:*", report)
        self.assertIn("schizophrenia classification", report)
        traced = [
            call.kwargs["data"]
            for call in context.trace.emit.call_args_list
            if call.args[0] == "tool.reference_check.claim"
        ]
        publication = next(
            item for item in traced
            if "The results of this study" in item["citation_sentence"]
        )
        self.assertIn(
            "schizophrenia classification",
            publication["preceding_context"],
        )
        self.assertEqual(len(publication["retrieval_queries"]), 1)
        self.assertTrue(all(
            "published" not in query
            for query in publication["retrieval_queries"]
        ))
        self.assertIn("passages", publication)

    def test_context_respects_paragraphs_and_word_budget(self) -> None:
        text = (
            "A previous paragraph discusses an unrelated experiment.\n\n"
            "The second case study implements a classification pipeline. "
            "Most building blocks are reused from the main framework. "
            "The results of this study were published [22]. "
            "As shown in the next figure, a generator file is added.\n\n"
            "An unrelated new paragraph starts here."
        )
        sentences = split_sentences((chunk("p", text),), min_words=1)
        citing = next(
            sentence for sentence in sentences
            if "results of this study" in sentence.text
        )
        before, after = _citation_context(sentences, {citing: []})
        self.assertIn("The second case study", before[citing])
        self.assertIn("Most building blocks", before[citing])
        self.assertNotIn("unrelated experiment", before[citing])
        self.assertIn("As shown in the next figure", after[citing])
        self.assertNotIn("unrelated new paragraph", after[citing])

        lengthy = " ".join("antecedent" for _ in range(145))
        long_text = (
            f"{lengthy}. This study was published [22]. "
            "An additional supporting sentence follows."
        )
        sentences = split_sentences(
            (chunk("l", long_text),), min_words=1
        )
        citing = next(
            sentence for sentence in sentences
            if "This study was published" in sentence.text
        )
        before, after = _citation_context(sentences, {citing: []})
        total = (
            len(before.get(citing, "").split())
            + len(citing.text.split())
            + len(after.get(citing, "").split())
        )
        self.assertLessEqual(total, 140)

    def test_titleless_missing_reference_links_in_report(self) -> None:
        original = (
            "Sporns, O. The human connectome: a structural description. "
            "PLoS\nComput. Biol. 1, e42 (2005)."
        )
        replacement = (
            "Smith J, Jones A, Brown B. Phys. Rev. Lett. "
            "1998;80(2):1234-1237."
        )
        self.chunks[0] = chunk(
            "p1", PAPER.replace(original, replacement), path="paper.md"
        )
        with patch(
            "kilin.tools.reference_check._resolution._json_request"
        ) as request:
            result, context = self._run(
                (Path("paper.md"), "named", False),
                question="Get web links for missing references of paper.md",
                interactive=False, verify=False, max_listed=20,
            )
        request.assert_not_called()
        context.ask_model.assert_not_called()
        self.assertEqual(result.summary["links"]["1"]["kind"], "search")
        url = result.summary["links"]["1"]["url"]
        self.assertIn("Smith%20Phys.%20Rev.%20Lett", url)
        self.assertIn("1998%2080%201234", url)
        report = (self.root / result.summary["report"]).read_text(
            encoding="utf-8"
        )
        self.assertEqual(report.count(url), 2)
        self.assertIn(url, result.answer)

    def test_short_journal_citation_still_receives_links(self) -> None:
        original = (
            "Sporns, O. The human connectome: a structural description. "
            "PLoS\nComput. Biol. 1, e42 (2005)."
        )
        self.chunks[0] = chunk(
            "p1", PAPER.replace(
                original, "Smith J. Phys Rev Lett 1998;80:1234."
            ), path="paper.md",
        )
        with patch(
            "kilin.tools.reference_check._resolution._json_request"
        ) as request:
            result, context = self._run(
                (Path("paper.md"), "named", False),
                question="Get links for missing references in paper.md",
                interactive=False, verify=False, max_listed=20,
            )
        request.assert_not_called()
        context.ask_model.assert_not_called()
        self.assertEqual(result.summary["unreadable"], 1)
        url = result.summary["links"]["1"]["url"]
        self.assertIn("Smith%20Phys%20Rev%20Lett%201998%2080%201234", url)
        report = (self.root / result.summary["report"]).read_text(
            encoding="utf-8"
        )
        self.assertEqual(report.count(url), 2)
        self.assertIn(url, result.answer)

    def test_very_compact_numbered_citation_is_not_discarded(self) -> None:
        original = (
            "Sporns, O. The human connectome: a structural description. "
            "PLoS\nComput. Biol. 1, e42 (2005)."
        )
        self.chunks[0] = chunk(
            "p1", PAPER.replace(original, "Nature 2020;12:22."),
            path="paper.md",
        )
        result, context = self._run(
            (Path("paper.md"), "named", False), verify=False,
            max_listed=20,
        )
        self.assertEqual(result.summary["references"], 5)
        self.assertEqual(result.summary["unreadable"], 1)
        url = result.summary["links"]["1"]["url"]
        self.assertIn("Nature%202020%2012%2022", url)
        self.assertIn(url, result.answer)
        context.ask_model.assert_not_called()
        report = (self.root / result.summary["report"]).read_text(
            encoding="utf-8"
        )
        self.assertEqual(report.count(url), 2)

    def test_written_only_citations_are_reported(self) -> None:
        paper = (
            PAPER.replace("[1]", "Ref. 1")
            .replace("[2, 3]", "refs. 2,3")
            .replace("[4]", "Ref. 4")
            .replace("[2]", "Ref. 2")
        )
        self.chunks[0] = chunk("p1", paper, path="paper.md")
        result, _ = self._run(
            (Path("paper.md"), "named", False), verify=False
        )
        self.assertEqual(result.summary["citation_style"], "written")
        self.assertEqual(result.summary["uncited"], [5])
        self.assertIn("written-out citations", result.answer)

    def test_quoted_passages_are_evidence(self) -> None:
        result, _ = self._run(
            (Path("paper.md"), "named", False), max_listed=5
        )
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0]["path"], Path("braph.md"))
        self.assertIn(
            "*The cited paper says:* “Mite Mijalkov, Ehsan Kakaei, Joana B. "
            "Pereira” [1]",
            result.answer,
        )

    def test_cancel_or_disable_verification(self) -> None:
        result, context = self._run(
            (Path("paper.md"), "named", False), choice="cancel"
        )
        context.ask_model.assert_not_called()
        self.assertIn("were not checked (cancelled)", result.answer)
        result, context = self._run(
            (Path("paper.md"), "named", False), verify=False
        )
        context.confirm.assert_not_called()
        context.ask_model.assert_not_called()
        self.assertNotIn("Claims checked", result.answer)

    def test_quick_version_checks_the_first_claims(self) -> None:
        result, context = self._run(
            (Path("paper.md"), "named", False), choice="quick",
            quick_claims=1,
        )
        # The first claim's sentence cites [2, 3]: it is split first.
        self.assertEqual(context.ask_model.call_count, 2)
        self.assertIn("2 of 3 claims were not checked (the quick version)",
                      result.answer)

    def test_model_sees_markers_and_what_is_cited(self) -> None:
        self.chunks[0] = chunk("p1", PAPER.replace(
            "Software such as\nBRAPH implements these measures [2, 3].",
            "All teams tracked particles using an established tracking "
            "method [2, 3] and then analysed them.",
        ), path="paper.md")
        _, context = self._run((Path("paper.md"), "named", False))
        self.assertIn("tracking method", self.chunks[0].text)
        prompt = next(
            call.args[1] for call in context.ask_model.call_args_list
            if "citing reference [2]" in call.args[1]
        )
        self.assertIn(
            'Reference [2] is cited for: "an established tracking method"',
            prompt,
        )
        self.assertIn("Preceding context:", prompt)

    def test_attribution_is_judged_as_the_whole_sentence(self) -> None:
        sentence = Sentence(
            Path("p.md"), 1,
            "Estimating it is hard, as already observed in the challenge "
            "[2].",
            chunk("s", "x"),
        )
        start = sentence.text.index("[2]")
        citation = Citation((2,), "2", sentence, start, start + 3)
        self.assertEqual(_cited_for(sentence, [citation], 2), "")

    def test_answer_has_no_bracketed_numbers(self) -> None:
        self.chunks[1] = chunk(
            "b1", BRAPH + "Earlier toolboxes [20, 21] exist.\n",
            path="braph.md",
        )
        result, _ = self._run(
            (Path("paper.md"), "named", False), max_listed=5
        )
        answer = result.answer.replace("[1]", "")
        self.assertNotRegex(answer, r"\[\d")
        self.assertNotIn("\\[", answer)
        self.assertIn("**3 claims checked:** 2 partly supported · 1 not "
                      "supported", result.answer)

    def test_one_of_several_reference_lists_is_checked(self) -> None:
        supplement = (
            "Supplementary Information\n"
            "The supplement extends the analysis [1], [2].\n"
            "Supplementary References\n"
            "1. Extra, A. A supplementary study of brain cells. Cell 1, "
            "2 (2020).\n"
            "2. Extra, B. Another supplementary study of tissue. Cell 3, "
            "4 (2021).\n"
        )
        self.chunks[0] = chunk("p1", PAPER + supplement, path="paper.md")
        cases = (
            # (answer to the question, question, references, note part)
            (1, "check the references of paper.md", 2, "you chose"),
            (None, "check the supplementary references of paper.md", 2,
             "the question names"),
            (None, "check the main references of paper.md", 5,
             "the question names"),
        )
        for answer, question, count, note in cases:
            with self.subTest(question=question, answer=answer):
                choose = Mock(return_value=answer)
                result, context = self._run_with_choice(
                    choose, question, verify=False
                )
                self.assertEqual(result.summary["references"], count)
                self.assertIn(note, result.summary["reference_list"])
                self.assertIn(note, result.answer)
                asked = "you chose" in note
                self.assertEqual(choose.called, asked)
                if asked:
                    labels = choose.call_args.args[1]
                    self.assertEqual(
                        labels,
                        ("References (5 entries, 1–5)",
                         "Supplementary References (2 entries, 1–2)"),
                    )
        # The supplement's citations cite its own list.
        result, _ = self._run_with_choice(
            Mock(return_value=1), "check paper.md", verify=False
        )
        self.assertEqual(result.summary["uncited"], [])
        # Nobody to ask: the first list, and the answer says so.
        result, _ = self._run_with_choice(None, "check paper.md",
                                          verify=False)
        self.assertEqual(result.summary["references"], 5)
        self.assertIn("the first", result.answer)
        # Cancelled: nothing is checked.
        result, _ = self._run_with_choice(
            Mock(return_value=None), "check paper.md", verify=False
        )
        self.assertIn("No reference list was chosen", result.answer)
        self.assertNotIn("references", result.summary)

    def _run_with_choice(
        self, choose: object, question: str, **settings: object
    ) -> tuple[object, SimpleNamespace]:
        """Run with a ``choose`` callback for the reference-list choice."""

        return self._run(
            (Path("paper.md"), "named", False), question=question,
            choose=choose, **settings,
        )

    def test_sentence_citing_several_references_is_focused(self) -> None:
        prompts: list[str] = []

        def ask_model(system: str, prompt: str, tokens: object) -> str:
            prompts.append(prompt)
            if "citing several references" in prompt:
                return ("[2]: BRAPH implements graph measures.\n"
                        "[3]: BRAPH 2 implements these measures too.")
            return "VERDICT: supported\nQUOTE: NONE\nCOMMENT: Fine."

        _, context = self._run_model(ask_model)
        focus = [p for p in prompts if "citing several references" in p]
        self.assertEqual(len(focus), 1)
        checked = [p for p in prompts if "VERDICT:" in p]
        self.assertTrue(any(
            "BRAPH implements these measures [2, 3]." in p
            for p in checked
        ))
        self.assertFalse(any(
            "BRAPH implements graph measures." in p for p in checked
        ))
        # The hint still affects ranking, but never determines a verdict.
        self.assertIn(
            "BRAPH implements graph measures.",
            [call.args[0] for call in context.rank.call_args_list]
        )
        # The sentence citing only [2] is checked as it is.
        self.assertTrue(any(
            "We followed the protocol of [2]." in p and "rewritten" not in p
            for p in checked
        ))
        # Turned off: no splitting call.
        prompts.clear()
        self._run_model(ask_model, focus_claims=False)
        self.assertFalse(
            any("citing several references" in p for p in prompts)
        )

    def test_second_look_reads_more_passages(self) -> None:
        # The cited paper has ten passages; the answer is in the last.
        self.chunks[1:] = [
            chunk(f"b{n}", f"Passage {n} about brain graphs and networks.\n",
                  path="braph.md", line=n)
            for n in range(1, 10)
        ] + [chunk("b10", "BRAPH implements these graph measures.\n",
                   path="braph.md", line=10)]
        self.chunks[1] = chunk("b1", BRAPH, path="braph.md")
        prompts: list[str] = []

        def ask_model(system: str, prompt: str, tokens: object) -> str:
            prompts.append(prompt)
            if "BRAPH implements these graph measures" in prompt:
                return ("VERDICT: supported\nQUOTE: BRAPH implements these "
                        "graph measures.\nCOMMENT: Stated.")
            return "VERDICT: not supported\nQUOTE: NONE\nCOMMENT: Absent."

        result, _ = self._run_model(
            ask_model, focus_claims=False, passages=6, retry_passages=8,
        )
        supported = [
            item for item in result.summary["suitability"]
            if item["verdict"] == "supported"
        ]
        self.assertEqual(len(supported), 2)
        self.assertIn("second, wider reading", "\n".join(
            (self.root / result.summary["report"]).read_text(
                encoding="utf-8").splitlines()
        ))
        # The accepted second look must be reconstructable from the trace.
        result, context = self._run_model(
            ask_model, focus_claims=False, passages=6, retry_passages=8,
        )
        records = [
            event.kwargs["data"]
            for event in context.trace.emit.call_args_list
            if event.args[0] == "tool.reference_check.claim"
        ]
        traced = [
            entry for entry in records
            if entry["quote_passage_id"] == "b10"
        ]
        self.assertEqual(len(traced), 2)
        for entry in traced:
            ids = {passage["id"] for passage in entry["passages"]}
            self.assertIn("b10", ids)
            self.assertEqual(entry["verdict_stage"], "second_look")
            self.assertEqual(entry["readings"][-1]["stage"], "second_look")
            self.assertIn("b10", entry["readings"][-1]["passage_ids"])
            self.assertEqual(entry["model_calls"], 2)
        usage = result.summary["model_calls"]
        self.assertEqual(usage["focusing"], 0)
        self.assertEqual(usage["initial"], 2)
        self.assertEqual(usage["second_look"], 2)
        self.assertEqual(usage["total"], context.ask_model.call_count)
        plan = context.confirm.call_args.args[0]
        self.assertEqual(plan.batches_to_run, 4)
        self.assertGreater(plan.estimated_seconds, plan.quick_seconds / 2)
        # Without the second look, both stay not supported.
        result, _ = self._run_model(
            ask_model, focus_claims=False, passages=6, retry_passages=0,
        )
        self.assertFalse(any(
            item["verdict"] == "supported"
            for item in result.summary["suitability"]
        ))

    def _run_model(
        self, ask_model: object, **settings: object
    ) -> tuple[object, SimpleNamespace]:
        """Run on paper.md with another fake answering model."""

        return self._run(
            (Path("paper.md"), "named", False), ask=ask_model, **settings
        )

    def test_cancelled_choice(self) -> None:
        result, context = self._run("Name the paper to check.")
        self.assertEqual(result.answer, "Name the paper to check.")
        self.assertEqual(result.summary, {"name": "reference_check"})
        context.done.assert_called_once()

    def test_missing_links_are_in_chat_and_report_without_network(
        self,
    ) -> None:
        with patch(
            "kilin.tools.reference_check._resolution._json_request"
        ) as network:
            result, context = self._run(
                (Path("paper.md"), "named", False), verify=False,
                max_listed=5,
            )
        network.assert_not_called()
        context.confirm.assert_not_called()
        self.assertEqual(len(result.summary["links"]), 3)
        self.assertIn("[Search Google Scholar](https://", result.answer)
        report = (
            self.root / result.summary["report"]
        ).read_text(encoding="utf-8")
        self.assertEqual(report.count("**Link:** [Search Google Scholar]"), 3)
        self.assertIn("## Missing from the project", report)

    def test_link_request_does_not_invoke_claim_verification(self) -> None:
        question = "get web links for the missing references in paper.md"
        with patch(
            "kilin.tools.reference_check.tool.resolve_links",
        ) as resolver:
            resolver.side_effect = lambda entries, permission: {
                number: offline_link(number, entry)
                for number, entry in entries.items()
            }
            result, context = self._run(
                (Path("paper.md"), "named", False), question=question,
                choice="continue", max_listed=5,
            )
        context.confirm.assert_called_once()
        plan = context.confirm.call_args.args[0]
        self.assertEqual(plan.operation, "network")
        self.assertIn("Crossref", plan.summary)
        self.assertEqual(len(resolver.call_args.args[0]), 3)
        context.ask_model.assert_not_called()
        self.assertEqual(result.summary["claims_checked"], 0)
        self.assertIn("Online lookup approved", result.answer)

    def test_denied_and_unavailable_approval_stay_offline(self) -> None:
        question = "give me web links for missing references in paper.md"
        for choice, interactive in (
            ("cancel", True), ("continue", False), ("invalid", True)
        ):
            with self.subTest(choice=choice, interactive=interactive):
                with patch(
                    "kilin.tools.reference_check._resolution._json_request"
                ) as network:
                    result, context = self._run(
                        (Path("paper.md"), "named", False),
                        choice=choice, question=question,
                        interactive=interactive,
                    )
                network.assert_not_called()
                context.ask_model.assert_not_called()
                self.assertTrue(result.summary["links"])
                self.assertIn("Search Google Scholar", result.answer)
                self.assertIn("Online lookup", result.answer)

    def test_online_lookup_setting_still_asks_permission(self) -> None:
        with patch(
            "kilin.tools.reference_check._resolution._json_request"
        ) as network:
            result, context = self._run(
                (Path("paper.md"), "named", False), choice="cancel",
                online_lookup=True, verify=False,
            )
        network.assert_not_called()
        self.assertEqual(context.confirm.call_count, 1)
        self.assertIn("declined", result.summary["lookup_note"])

    def test_paper_without_reference_list(self) -> None:
        self.chunks[0] = chunk("p1", "Just text [1] here.\n", path="paper.md")
        result, _ = self._run((Path("paper.md"), "named", True))
        self.assertIn("no reference list", result.answer)
        self.assertIsNone(result.summary["report"])
        self.assertIn("name the paper if you meant another", result.answer)
        self.assertFalse((self.root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
