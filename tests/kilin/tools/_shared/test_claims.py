"""Unit tests for judging claims against the cited work.

Check passage selection, the prompt, and above all the Python checks on
the model's answer: quotes must be in the passages, and a contradiction
needs a confirmed quote.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

from kilin.tools._shared.sentences import Sentence
from kilin.tools._shared.claims import (
    VERDICT_BACKGROUND,
    VERDICT_CONTRADICTED,
    VERDICT_FAILED,
    VERDICT_NOT_FOUND,
    VERDICT_NOT_SUPPORTED,
    VERDICT_NO_TEXT,
    VERDICT_PARTLY,
    VERDICT_SUPPORTED,
    VERDICT_UNCLEAR,
    Claim,
    claim_passages,
    focus_claims,
    focus_prompt,
    judge_support,
    parse_focus,
    parse_verdict,
    quote_line_range,
    support_prompt,
    verdict_label,
    verification_prompt,
    verify_claim,
)
from tests.kilin.tools._shared._support import chunk

PASSAGES = (
    chunk("a", "Mice that ran daily lived 20% longer than sedentary mice.\n"
          "The effect was absent in rats.\n", path="cited.md"),
    chunk("b", "Methods were standard.\n", path="cited.md", line=3),
)
SENTENCE = Sentence(
    Path("paper.md"), 4, "Running extends lifespan in mammals [7].",
    chunk("s", "Running extends lifespan in mammals [7].\n"),
)
CLAIM = Claim(7, SENTENCE, Path("cited.md"), (3,), "Exercise matters.")


class TestParseVerdict(unittest.TestCase):
    """Check the model's answer in Python."""

    def test_confirmed_quote(self) -> None:
        reply = (
            "VERDICT: partly supported\n"
            "QUOTE: Mice that ran daily lived 20%\n longer than sedentary "
            "mice.\nCOMMENT: Only in mice, not all mammals."
        )
        verdict, quote, found, comment = parse_verdict(reply, PASSAGES)
        self.assertEqual(verdict, VERDICT_PARTLY)
        self.assertIs(found, PASSAGES[0])
        self.assertTrue(quote.startswith("Mice that ran"))
        self.assertEqual(comment, "Only in mice, not all mammals.")

    def test_invented_quote_is_dropped(self) -> None:
        reply = (
            "VERDICT: supported\nQUOTE: Running extends lifespan in every "
            "mammal studied.\nCOMMENT: Direct statement."
        )
        verdict, quote, found, comment = parse_verdict(reply, PASSAGES)
        self.assertEqual((verdict, quote, found), (VERDICT_SUPPORTED, "",
                                                   None))
        self.assertIn("not in the passages", comment)

    def test_quote_tolerates_extraction_artifacts(self) -> None:
        passages = (chunk(
            "a",
            "Numerous solutions have been proposed and compared in open\n"
            "challenges6,7. Their perfor- mance depends on the noise.\n",
            path="cited.md",
        ),)
        for said in (
            # Citation numbers dropped, a hyphenated word mended.
            "Numerous solutions have been proposed and compared in open "
            "challenges.",
            "Their performance depends on the noise.",
            # One word changed out of twelve: still the same sentence.
            "Numerous solutions have been suggested and compared in open "
            "challenges.",
        ):
            with self.subTest(said=said):
                reply = f"VERDICT: supported\nQUOTE: {said}\nCOMMENT: ok"
                verdict, quote, found, comment = parse_verdict(
                    reply, passages
                )
                self.assertIs(found, passages[0])
                self.assertEqual(comment, "ok")
        # The paper's own words are shown, not the model's.
        reply = (
            "VERDICT: supported\nQUOTE: Numerous solutions have been "
            "suggested and compared in open challenges.\nCOMMENT: ok"
        )
        self.assertEqual(
            parse_verdict(reply, passages)[1],
            "Numerous solutions have been proposed and compared in open "
            "challenges",
        )

    def test_changed_number_is_not_a_confirmed_quote(self) -> None:
        """Reject a fabricated numerical effect despite word overlap."""

        passages = (chunk(
            "numbers", "Treatment reduced mortality by 10% in mice.",
            path="trial.md",
        ),)
        reply = (
            "VERDICT: supported\n"
            "QUOTE: Treatment reduced mortality by 90% in mice.\n"
            "COMMENT: Reports the result."
        )
        verdict, quote, found, comment = parse_verdict(reply, passages)
        self.assertEqual(verdict, VERDICT_SUPPORTED)
        self.assertEqual(quote, "")
        self.assertIsNone(found)
        self.assertIn("not in the passages", comment)

    def test_reversed_effect_and_negation_are_not_confirmed(self) -> None:
        """A near-identical quote may reverse the scientific meaning."""

        passages = (chunk(
            "direction", "Treatment reduced mortality in mice after "
            "twelve weeks of observation.", path="trial.md",
        ),)
        for model_quote in (
            "Treatment increased mortality in mice after twelve "
            "weeks of observation.",
            "Treatment did not reduce mortality in mice after twelve "
            "weeks of observation.",
        ):
            with self.subTest(model_quote=model_quote):
                verdict, quote, found, _ = parse_verdict(
                    "VERDICT: supported\nQUOTE: " + model_quote +
                    "\nCOMMENT: result", passages,
                )
                self.assertEqual(verdict, VERDICT_SUPPORTED)
                self.assertEqual(quote, "")
                self.assertIsNone(found)

    def test_changed_units_and_signed_values_are_rejected(self) -> None:
        """A matched numeral is insufficient when the units differ."""

        passages = (chunk(
            "dose", "Dose of -10 mg was applied to exposed tissue "
            "and measured after seven days.", path="trial.md",
        ),)
        for altered in (
            "Dose of 10 mg was applied to exposed tissue and "
            "measured after seven days.",
            "Dose of -10 g was applied to exposed tissue and "
            "measured after seven days.",
        ):
            with self.subTest(altered=altered):
                _, _, found, _ = parse_verdict(
                    "VERDICT: supported\nQUOTE: " + altered +
                    "\nCOMMENT: Direct.", passages,
                )
                self.assertIsNone(found)

    def test_quotation_evidence_status_and_source_lines(self) -> None:
        """Expose exact, approximate and rejected evidence separately."""

        passage = chunk(
            "lines", "Opening line.\nTreatment reduced mortality by "
            "10% in mice.\nClosing line.", path="trial.md", line=20,
        )
        exact = judge_support(
            "Treatment reduced mortality by 10% in mice.",
            Path("trial.md"), (passage,),
            lambda *_: (
                "VERDICT: supported\nQUOTE: Treatment reduced mortality "
                "by 10% in mice.\nCOMMENT: exact"
            ),
        )
        self.assertEqual(exact.evidence_status, "exact")
        self.assertEqual(quote_line_range(exact.quote, passage), (21, 21))
        approximate = judge_support(
            "Treatment reduced mortality in mice.",
            Path("trial.md"), (passage,),
            lambda *_: (
                "VERDICT: supported\nQUOTE: Treatment reduced "
                "mortality by 10% in mice\nCOMMENT: partial"
            ),
        )
        self.assertIn(approximate.evidence_status, ("exact", "approximate"))
        rejected = judge_support(
            "Treatment changed mortality by 90% in mice.",
            Path("trial.md"), (passage,),
            lambda *_: (
                "VERDICT: supported\nQUOTE: Treatment reduced mortality "
                "by 90% in mice.\nCOMMENT: changed"
            ),
        )
        self.assertEqual(rejected.evidence_status, "rejected")
        self.assertIsNone(rejected.quote_chunk)
        self.assertIsNone(quote_line_range("", passage))

    def test_no_quote_is_expected_for_negative_verdicts(self) -> None:
        """No-positive-evidence cases should not trigger missing warnings."""

        for verdict in ("not supported", "not found", "unclear"):
            with self.subTest(verdict=verdict):
                result = judge_support(
                    "Running benefits mice", Path("cited.md"), PASSAGES,
                    lambda *_: (
                        f"VERDICT: {verdict}\nQUOTE: NONE\n"
                        "COMMENT: Not demonstrated by the excerpts."
                    ),
                )
                self.assertEqual(result.evidence_status, "not required")
                self.assertIsNone(result.quote_chunk)
        positive = judge_support(
            "Running benefits mice", Path("cited.md"), PASSAGES,
            lambda *_: (
                "VERDICT: supported\nQUOTE: NONE\nCOMMENT: Supported."
            ),
        )
        self.assertEqual(positive.evidence_status, "missing")

    def test_inherited_token_limit_reaches_model_callback(self) -> None:
        """No tool-specific cap is passed to reference verification."""

        received = []
        def model(system: str, prompt: str, tokens: int | None) -> str:
            received.append(tokens)
            return (
                "VERDICT: partly supported\n"
                "QUOTE: Mice that ran daily lived 20% longer than "
                "sedentary mice.\nCOMMENT: Only in mice."
            )
        result = judge_support(
            "Running extends lifespan in mice", Path("cited.md"),
            PASSAGES, model,
        )
        self.assertEqual(received, [None])
        self.assertIs(result.quote_chunk, PASSAGES[0])

    def test_quote_broken_by_interleaved_text(self) -> None:
        margin = " ".join(f"word{index}" for index in range(30))
        passages = (chunk(
            "a",
            "We present a machine learning algorithm based on the random "
            f"forest {margin} architecture that robustly characterizes "
            "single trajectories at different levels.\n",
            path="cited.md",
        ),)
        reply = (
            "VERDICT: supported\nQUOTE: We present a machine learning "
            "algorithm based on the random forest architecture that "
            "robustly characterizes single trajectories.\nCOMMENT: ok"
        )
        verdict, quote, found, comment = parse_verdict(reply, passages)
        self.assertIs(found, passages[0])
        self.assertEqual(
            quote,
            "We present a machine learning algorithm based on the random "
            "forest … architecture that robustly characterizes single "
            "trajectories",
        )

    def test_scattered_words_are_not_a_quote(self) -> None:
        passages = (chunk(
            "a",
            "The method works. In every case the noise was low. Mice were "
            "studied, and the running wheel extends the cage. Lifespan "
            "was not measured in any mammal.\n",
            path="cited.md",
        ),)
        reply = (
            "VERDICT: supported\nQUOTE: Running extends lifespan in every "
            "mammal studied.\nCOMMENT: ok"
        )
        self.assertIsNone(parse_verdict(reply, passages)[2])

    def test_rejected_quote_is_shown_in_the_comment(self) -> None:
        reply = (
            "VERDICT: supported\nQUOTE: Running extends lifespan in every "
            "mammal studied.\nCOMMENT: Direct."
        )
        comment = parse_verdict(reply, PASSAGES)[3]
        self.assertIn(
            "“Running extends lifespan in every mammal "
            "studied.”, is not in "
            "the passages",
            comment,
        )

    def test_contradiction_needs_a_confirmed_quote(self) -> None:
        confirmed = (
            "VERDICT: contradicted\nQUOTE: The effect was absent in rats.\n"
            "COMMENT: Not all mammals."
        )
        self.assertEqual(
            parse_verdict(confirmed, PASSAGES)[0], VERDICT_CONTRADICTED
        )
        for quote in ("NONE", "Running shortens lifespan in mammals."):
            with self.subTest(quote=quote):
                reply = (
                    f"VERDICT: contradicted\nQUOTE: {quote}\nCOMMENT: No."
                )
                verdict, _, _, comment = parse_verdict(reply, PASSAGES)
                self.assertEqual(verdict, VERDICT_NOT_SUPPORTED)
                self.assertIn("Judged contradicted", comment)

    def test_verdict_names(self) -> None:
        for said, expected in (
            ("**not supported**", VERDICT_NOT_SUPPORTED),
            ("Supported.", VERDICT_SUPPORTED),
            ("background", VERDICT_BACKGROUND),
            ("not found", VERDICT_NOT_FOUND),
            ("maybe", VERDICT_UNCLEAR),
        ):
            with self.subTest(said=said):
                reply = f"VERDICT: {said}\nQUOTE: NONE\nCOMMENT: x"
                self.assertEqual(parse_verdict(reply, PASSAGES)[0], expected)
        self.assertEqual(
            parse_verdict("I think it is fine.", PASSAGES)[0],
            VERDICT_UNCLEAR,
        )


class TestVerifyClaim(unittest.TestCase):
    """Passages, prompt and model call."""

    def test_passages_come_from_the_cited_file_in_order(self) -> None:
        other = chunk("z", "Unrelated running text.\n", path="other.md")
        ranking = [SimpleNamespace(chunk=c)
                   for c in (other, PASSAGES[1], PASSAGES[0])]
        picked = claim_passages(
            ("running",), PASSAGES, lambda text: ranking, 4, 1000
        )
        self.assertEqual(picked, PASSAGES)
        self.assertEqual(
            claim_passages(
                ("running",), PASSAGES, lambda text: ranking, 1, 1
            ),
            (PASSAGES[1],),
        )

    def test_rankings_of_several_texts_are_fused(self) -> None:
        third = chunk("c", "Lifespan tables.\n", path="cited.md", line=4)
        cited = (*PASSAGES, third)
        orders = {
            # The sentence ranks the methods first, what the reference is
            # cited for ranks the result first; both rank the table last.
            "sentence": (PASSAGES[1], PASSAGES[0], third),
            "cited for": (PASSAGES[0], PASSAGES[1], third),
        }
        picked = claim_passages(
            ("sentence", "cited for", ""), cited,
            lambda text: [SimpleNamespace(chunk=c) for c in orders[text]],
            2, 1000,
        )
        self.assertEqual(picked, PASSAGES)

    def test_passages_already_read_are_skipped(self) -> None:
        third = chunk("c", "Lifespan tables.\n", path="cited.md", line=4)
        cited = (*PASSAGES, third)
        ranking = [SimpleNamespace(chunk=c) for c in cited]
        picked = claim_passages(
            ("running",), cited, lambda text: ranking, 2, 1000,
            skip={PASSAGES[0].identifier},
        )
        self.assertEqual(picked, (PASSAGES[1], third))

    def test_focus_prompt_and_reply(self) -> None:
        _, prompt = focus_prompt(
            "BRAPH uses the method [3] on the data [5].", "Before.", "",
            (3, 5),
        )
        self.assertIn('"BRAPH uses the method [3] on the data [5]."', prompt)
        self.assertIn('Preceding context: "Before."', prompt)
        self.assertIn("[3]: <claim>\n[5]: <claim>", prompt)
        reply = (
            "<think>Splitting.</think>\n[3]: BRAPH uses the method of [3].\n"
            '5: "The data analysed by BRAPH."\n[9]: Not asked.\n[3]: Again.'
        )
        self.assertEqual(
            parse_focus(reply, (3, 5)),
            {3: "BRAPH uses the method of [3].",
             5: "The data analysed by BRAPH."},
        )
        self.assertEqual(parse_focus("[3]: <claim>", (3,)), {})

    def test_focus_claims_calls_once_and_survives_failures(self) -> None:
        calls = []

        def ask(system: str, prompt: str, tokens: object) -> str:
            calls.append(prompt)
            return "[3]: The method.\n[5]: The data."

        self.assertEqual(
            focus_claims("S [3] and [5].", "", "", (3, 5), ask),
            {3: "The method.", 5: "The data."},
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(focus_claims("S [3].", "", "", (), ask), {})

        def fail(system: str, prompt: str, tokens: object) -> str:
            raise RuntimeError("offline")

        self.assertEqual(focus_claims("S [3].", "", "", (3,), fail), {})

    def test_focused_claim_is_judged(self) -> None:
        """A proposed rewrite must not replace the source claim."""

        original = verification_prompt(CLAIM, PASSAGES)
        rewritten = verification_prompt(
            replace(CLAIM, focus="Invented phenomenon in mice."),
            PASSAGES,
        )
        self.assertEqual(original, rewritten)
        self.assertIn(CLAIM.sentence.text, rewritten[1])
        self.assertNotIn("Invented phenomenon", rewritten[1])

    def test_prompt_names_reference_context_and_passages(self) -> None:
        _, prompt = verification_prompt(CLAIM, PASSAGES)
        self.assertIn("citing reference [7]", prompt)
        self.assertIn('Preceding context: "Exercise matters."', prompt)
        self.assertIn("also cites [3]", prompt)
        self.assertIn("Judge only what the cited sentence", prompt)
        self.assertIn("Passage 2:\nMethods were standard.", prompt)

    def test_publication_attribution_requires_matching_study(self) -> None:
        paper = chunk(
            "p", "The results of this study were published [22].",
            path="article.md",
        )
        citing = Sentence(Path("article.md"), 12, paper.text, paper)
        claim = Claim(
            22, citing, Path("yao25connecting.pdf"),
            before=(
                "The second case study implements recursive feature "
                "elimination for schizophrenia classification. "
                "Most building blocks are reused."
            ),
        )
        source = chunk(
            "yao", (
                "In each data set we applied a deep neural network "
                "classifier to prioritize brain networks that distinguish "
                "cases from controls. We performed five folds of "
                "parallel recursive feature elimination."
            ), path="yao25connecting.pdf",
        )
        _, prompt = verification_prompt(claim, [source])
        self.assertIn("schizophrenia classification", prompt)
        self.assertIn("treat this as attribution", prompt)
        self.assertIn("require specific evidence", prompt)
        self.assertIn("does not say", prompt)
        verified = verify_claim(
            claim, (source,),
            lambda system, user, tokens: (
                "VERDICT: supported\nQUOTE: In each data set we applied "
                "a deep neural network classifier to prioritize brain "
                "networks that distinguish cases from controls.\n"
                "COMMENT: The cited study uses the described classifier."
            ),
        )
        self.assertEqual(verified.verdict, "supported")
        self.assertIs(verified.quote_chunk, source)

    def test_prompt_rules_for_sources_and_descriptions(self) -> None:
        claim = Claim(
            40, SENTENCE, Path("cited.md"),
            cited_for="a previously established tracking method",
        )
        _, prompt = verification_prompt(claim, PASSAGES)
        self.assertIn(
            '"a previously established tracking method" (as the source '
            "of it)",
            prompt,
        )
        self.assertIn('("previously\n  established"', prompt)
        self.assertIn("verdict is background, even without", prompt)

    def test_model_call_and_failures(self) -> None:
        verdict = verify_claim(
            CLAIM, PASSAGES,
            lambda system, prompt, tokens: (
                "VERDICT: supported\nQUOTE: The effect was absent in rats.\n"
                "COMMENT: ok"
            ),
        )
        self.assertEqual(verdict.verdict, VERDICT_SUPPORTED)
        self.assertEqual(verdict.passages, PASSAGES)

        def broken(system: str, prompt: str, tokens: int) -> str:
            raise RuntimeError("out of memory")

        failed = verify_claim(CLAIM, PASSAGES, broken)
        self.assertEqual(failed.verdict, VERDICT_FAILED)
        self.assertIn("out of memory", failed.comment)
        empty = verify_claim(CLAIM, (), broken)
        self.assertEqual(empty.verdict, VERDICT_NO_TEXT)


class TestSupport(unittest.TestCase):
    """A statement judged against a work it does not cite yet."""

    def test_prompt_asks_for_the_source(self) -> None:
        system, prompt = support_prompt(
            "Running extends lifespan.", Path("cited.md"), PASSAGES
        )
        self.assertIn("cited as the source", system)
        self.assertIn('"Running extends lifespan."', prompt)
        self.assertIn("cited.md", prompt)
        self.assertIn("Passage 2:\nMethods were standard.", prompt)
        self.assertIn("the original source is elsewhere", prompt)
        self.assertIn("VERDICT:", prompt)

    def test_judged_and_checked_like_a_claim(self) -> None:
        support = judge_support(
            "Running extends lifespan in mice.", Path("cited.md"), PASSAGES,
            lambda system, prompt, tokens: (
                "VERDICT: partly supported\nQUOTE: Mice that ran daily "
                "lived 20% longer than sedentary mice.\nCOMMENT: Mice only."
            ),
        )
        self.assertEqual(support.verdict, VERDICT_PARTLY)
        self.assertIs(support.quote_chunk, PASSAGES[0])
        self.assertEqual(support.cited, Path("cited.md"))
        self.assertEqual(support.passages, PASSAGES)
        invented = judge_support(
            "Running extends lifespan.", Path("cited.md"), PASSAGES,
            lambda system, prompt, tokens: (
                "VERDICT: contradicted\nQUOTE: Running shortens the life "
                "of every mammal.\nCOMMENT: No."
            ),
        )
        self.assertEqual(invented.verdict, VERDICT_NOT_SUPPORTED)
        self.assertIsNone(invented.quote_chunk)
        empty = judge_support(
            "Running extends lifespan.", Path("cited.md"), (), None
        )
        self.assertEqual(empty.verdict, VERDICT_NO_TEXT)



class TestVerdictLabel(unittest.TestCase):
    """The words shown for a verdict."""

    def test_labels(self) -> None:
        self.assertEqual(
            verdict_label(VERDICT_NOT_FOUND), "no matching passage"
        )
        self.assertEqual(verdict_label(VERDICT_NO_TEXT), "no text")
        self.assertEqual(verdict_label(VERDICT_SUPPORTED), "supported")

if __name__ == "__main__":
    unittest.main()
