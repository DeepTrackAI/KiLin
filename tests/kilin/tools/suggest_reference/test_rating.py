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

from kilin.tools._shared.claims import Support
from kilin.tools.suggest_reference._rating import citation_suitability
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
        self.assertEqual(result.stars, 4)
        self.assertLess(result.score, 85)

    def test_partial_and_background_have_separate_bands(self) -> None:
        partial = citation_suitability(self._support("partly supported"))
        background = citation_suitability(self._support("background"))
        self.assertEqual(partial.stars, 3)
        self.assertEqual(background.stars, 2)
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

    def test_low_text_overlap_weakens_rating_without_downgrading_verdict(
        self,
    ) -> None:
        result = citation_suitability(self._support(
            "supported", statement="Mitochondrial fission causes epilepsy."
        ))
        self.assertEqual(result.stars, 4)

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

    def test_no_passages_or_quote_stays_in_verdict_band(self) -> None:
        result = citation_suitability(
            self._support("partly supported", quote=False, passages=False)
        )
        self.assertEqual(result.stars, 2)
        self.assertGreaterEqual(result.score, 40)


if __name__ == "__main__":
    unittest.main()
