"""Test deterministic suitability ratings without models or network.

Verify that verdicts dominate lexical overlap, confirmed evidence
changes ratings, and invalid or unsupported judgements get no stars.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.tools._shared.claims import Claim, Support, Verdict
from kilin.tools._shared.sentences import Sentence
from kilin.tools._shared._rating import (
    SuitabilityRating,
    citation_suitability,
    suitability_stars,
)
from kilin.tools.suggest_reference.tool import chat_answer
from tests.kilin.tools._shared._support import chunk

STATEMENT = "Anomalous diffusion occurs frequently in living cells."
QUOTE = "Anomalous diffusion is frequently observed in living cells."
PASSAGE = chunk("a", QUOTE, path="cited.md")


class TestSuitabilityRating(unittest.TestCase):
    """Verdict-first, quote-aware and ordinal star scoring."""

    def _support(
        self, verdict: str, *, quote: bool = True,
        passages: bool = True, statement: str = STATEMENT,
    ) -> Support:
        return Support(
            statement=statement,
            cited=Path("cited.md"),
            verdict=verdict,
            quote=QUOTE if quote else "",
            quote_chunk=PASSAGE if quote else None,
            comment="",
            passages=(PASSAGE,) if passages else (),
        )

    def test_supported_and_confirmed_earns_five_stars(self) -> None:
        result = citation_suitability(self._support("supported"))
        self.assertEqual(result.stars, 5)
        self.assertGreaterEqual(result.score, 85)

    def test_unconfirmed_support_is_capped_below_five(self) -> None:
        result = citation_suitability(
            self._support("supported", quote=False)
        )
        self.assertEqual(result.stars, 4.5)
        self.assertLess(result.score, 85)

    def test_partial_and_background_have_separate_bands(self) -> None:
        partial = citation_suitability(self._support("partly supported"))
        background = citation_suitability(self._support("background"))
        self.assertEqual(partial.stars, 3.5)
        self.assertEqual(background.stars, 2.0)
        self.assertGreater(partial.score, background.score)

    def test_overlap_cannot_override_support_verdict(self) -> None:
        # An unsupported paper cannot get stars even if its entire text
        # repeats the statement word for word.
        for verdict in (
            "not supported", "contradicted", "not found", "no text",
            "failed", "unclear",
        ):
            with self.subTest(verdict=verdict):
                self.assertIsNone(citation_suitability(
                    self._support(verdict)
                ))

    def test_verified_citation_uses_antecedent_for_suitability(self) -> None:
        citing = chunk(
            "c", "The results of this study were published [22].",
            path="article.md",
        )
        sentence = Sentence(
            Path("article.md"), 8, citing.text, citing
        )
        claim = Claim(
            22, sentence, Path("cited.md"),
            before=(
                "The second case study implements anomalous diffusion "
                "in living cells."
            ),
        )
        verdict = Verdict(
            claim, "supported", QUOTE, PASSAGE, "Matching study.",
            (PASSAGE,),
        )
        rating = citation_suitability(verdict)
        self.assertIsNotNone(rating)
        self.assertGreaterEqual(rating.stars, 4)
        self.assertTrue(
            suitability_stars(rating).endswith(f"({rating.score}/100)")
        )

    def test_unrated_uses_empty_stars_not_a_false_zero_score(self) -> None:
        self.assertEqual(suitability_stars(None), "☆☆☆☆☆ (unrated)")

    def test_low_text_overlap_weakens_rating_without_downgrading_verdict(
        self,
    ) -> None:
        result = citation_suitability(self._support(
            "supported", statement="Mitochondrial fission causes epilepsy."
        ))
        self.assertEqual(result.stars, 4.5)

    def test_tied_verdicts_are_ordered_by_suitability(self) -> None:
        weaker = self._support(
            "supported", statement="Mitochondrial fission causes epilepsy."
        )
        stronger = self._support("supported")
        weaker = Support(
            statement=weaker.statement, cited=Path("weak.md"),
            verdict=weaker.verdict, quote=weaker.quote,
            quote_chunk=weaker.quote_chunk, comment="",
            passages=weaker.passages,
        )
        answer, evidence = chat_answer(
            STATEMENT, [weaker, stronger]
        )
        self.assertLess(answer.index("cited.md"), answer.index("weak.md"))
        self.assertEqual(len(evidence), 2)

    def test_formatter_prints_raw_score_not_star_fraction(self) -> None:
        for score, stars, expected in (
            (0, 0.5, "½☆☆☆☆"), (9, 0.5, "½☆☆☆☆"),
            (10, 1.0, "★☆☆☆☆"), (19, 1.0, "★☆☆☆☆"),
            (20, 1.5, "★½☆☆☆"), (29, 1.5, "★½☆☆☆"),
            (30, 2.0, "★★☆☆☆"), (39, 2.0, "★★☆☆☆"),
            (40, 2.5, "★★½☆☆"), (49, 2.5, "★★½☆☆"),
            (50, 3.0, "★★★☆☆"), (59, 3.0, "★★★☆☆"),
            (60, 3.5, "★★★½☆"), (69, 3.5, "★★★½☆"),
            (70, 4.0, "★★★★☆"), (79, 4.0, "★★★★☆"),
            (80, 4.5, "★★★★½"), (89, 4.5, "★★★★½"),
            (90, 5.0, "★★★★★"), (100, 5.0, "★★★★★"),
        ):
            with self.subTest(score=score):
                rating = SuitabilityRating(score, stars)
                self.assertEqual(
                    suitability_stars(rating), f"{expected} ({score}/100)"
                )
                self.assertEqual(len(expected), 5)
                self.assertNotIn("/5", suitability_stars(rating))
        self.assertNotIn("/100", suitability_stars(None))

    def test_no_passages_or_quote_stays_in_verdict_band(self) -> None:
        result = citation_suitability(
            self._support("partly supported", quote=False, passages=False)
        )
        self.assertEqual(result.stars, 2.5)
        self.assertGreaterEqual(result.score, 40)


if __name__ == "__main__":
    unittest.main()
