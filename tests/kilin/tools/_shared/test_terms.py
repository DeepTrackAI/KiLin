"""Unit tests for exact-term parsing shared by multiple KíLín tools.

Check deterministic validation of model-produced terms and flexible
literal matching independently of registered tool implementations.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools._shared.terms import (
    MAX_TERMS,
    document_phrase,
    quoted_terms,
    term_pattern,
    terms_prompt,
    validated_document,
    validated_terms,
)


class TestSharedTerms(unittest.TestCase):
    """Verify exact search terms are grounded in the user's text."""

    def test_quoted_terms_are_distinct_and_bounded(self) -> None:
        """Ignore single quotes and deduplicate repeated phrases.

        """

        self.assertEqual(quoted_terms('"MRI" `MRI` “laser traps”'),
                         ("MRI", "laser traps"))
        terms = " ".join(f'"term{i}"' for i in range(MAX_TERMS + 3))
        self.assertEqual(len(quoted_terms(terms)), MAX_TERMS)

    def test_term_prompts_include_category_only_when_requested(self) -> None:
        """Construct category extensions without changing basic prompts.

        """

        basic = terms_prompt("Find MRI")
        category = terms_prompt("Find MRI", categories=True)
        self.assertIn('"terms"', basic[1])
        self.assertNotIn('"categories"', basic[1])
        self.assertIn('"categories"', category[1])
        self.assertIn("Find MRI", category[1])

    def test_validated_terms_reject_unwritten_or_malformed_terms(self) -> None:
        """Accept copied terms, reject extraneous model suggestions.

        """

        self.assertEqual(
            validated_terms(
                '{"terms": ["MRI", "missing", "MRI", null]}',
                "Find every MRI mention",
            ),
            ("MRI",),
        )
        self.assertEqual(validated_terms("not JSON", "MRI"), ())
        self.assertEqual(validated_terms('{"terms": "MRI"}', "MRI"), ())

    def test_document_phrases_are_user_grounded(self) -> None:
        """Validate model-produced document names against question text.

        """

        text = "Count MRI in the Smith paper"
        self.assertEqual(
            validated_document('{"document": "the Smith paper"}', text),
            "the Smith paper",
        )
        self.assertIsNone(
            validated_document('{"document": "the Jones paper"}', text)
        )
        self.assertEqual(document_phrase(text), "the Smith paper")

    def test_flexible_matching_respects_whole_words(self) -> None:
        """Treat spaces or hyphens similarly without substring matches.

        """

        pattern = term_pattern("XYZ2", False, True)
        self.assertIsNotNone(pattern.search("xyz-2"))
        self.assertIsNotNone(pattern.search("XYZ 2"))
        self.assertIsNone(pattern.search("prefixXYZ2suffix"))
        self.assertIsNone(term_pattern("MRI", True, True).search("mri"))


if __name__ == "__main__":
    unittest.main()
