"""Unit tests for judging claims against the cited work.

Check passage selection, the prompt, and above all the Python checks on
the model's answer: quotes must be in the passages, and a contradiction
needs a confirmed quote.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

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
    judge_support,
    parse_verdict,
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
            "“Running extends lifespan in every mammal studied.”, is not in "
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

    def test_prompt_names_reference_context_and_passages(self) -> None:
        _, prompt = verification_prompt(CLAIM, PASSAGES)
        self.assertIn("citing reference [7]", prompt)
        self.assertIn('Preceding context: "Exercise matters."', prompt)
        self.assertIn("also cites [3]", prompt)
        self.assertIn("Judge only what the cited sentence", prompt)
        self.assertIn("Passage 2:\nMethods were standard.", prompt)

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
            200,
        )
        self.assertEqual(verdict.verdict, VERDICT_SUPPORTED)
        self.assertEqual(verdict.passages, PASSAGES)

        def broken(system: str, prompt: str, tokens: int) -> str:
            raise RuntimeError("out of memory")

        failed = verify_claim(CLAIM, PASSAGES, broken, 200)
        self.assertEqual(failed.verdict, VERDICT_FAILED)
        self.assertIn("out of memory", failed.comment)
        empty = verify_claim(CLAIM, (), broken, 200)
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
            200,
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
            200,
        )
        self.assertEqual(invented.verdict, VERDICT_NOT_SUPPORTED)
        self.assertIsNone(invented.quote_chunk)
        empty = judge_support(
            "Running extends lifespan.", Path("cited.md"), (), None, 200
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
