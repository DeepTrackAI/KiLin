"""Test offline and permission-gated bibliography resolution.

External service responses are simulated: the suite must never access the
internet. Unknown and malformed metadata must leave search links intact.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

from kilin.tools.reference_check import _resolution as resolution

ENTRY = (
    "Smith, J. et al. Tracking individual particles with microscopy. "
    "Nature 12, 22-24 (2020)."
)
WORK = {
    "DOI": "10.1234/example.2020",
    "title": ["Tracking individual particles with microscopy"],
    "author": [{"family": "Smith"}],
    "published": {"date-parts": [[2020, 10]]},
}


class TestOfflineReferenceLinks(TestCase):
    """Construct complete, usable URLs without issuing requests."""

    def test_search_link_uses_bibliography_text(self) -> None:
        with patch.object(resolution, "_json_request") as request:
            result = resolution.offline_link(12, ENTRY)
        request.assert_not_called()
        self.assertEqual(result.number, 12)
        self.assertEqual(result.kind, "search")
        self.assertTrue(result.url.startswith(
            "https://scholar.google.com/scholar?q="
        ))
        self.assertEqual(
            parse_qs(urlsplit(result.url).query)["q"],
            ["Tracking individual particles with microscopy"],
        )
        self.assertNotIn("\n", result.url)

    def test_long_numbered_reference_searches_for_title(self) -> None:
        entry = (
            "Marek S, Tervo-Clemmens B, Calabro FJ, Montez DF, Kay BP, "
            "Hatoum AS, Donohue MR, Foran W, Miller RL, Hendrickson TJ, "
            "Lynch BJ, Tam A, Chen J, Newbold DJ, Zheng A, Seider NA, "
            "Van AN, Laumann TO, Greene DJ, Dosenbach NUF. Reproducible "
            "brain-wide association studies require thousands of "
            "individuals. Nature 2022;603(7902):654–60."
        )
        result = resolution.offline_link(6, entry)
        self.assertEqual(
            parse_qs(urlsplit(result.url).query)["q"],
            ["Reproducible brain-wide association studies require "
             "thousands of individuals"],
        )
        self.assertNotIn("Marek", result.url)

    def test_short_title_and_incomplete_citation_keep_a_query(self) -> None:
        cases = (
            ("Matthews PM, Sudlow C. The UK biobank. Brain 2015.",
             "The UK biobank"),
            ("Wang J, Wang X. GRETNA: a graph theoretical network "
             "analysis toolbox for imaging connectomics. "
             "Front Hum Neurosci 2015;9:",
             "GRETNA: a graph theoretical network analysis toolbox "
             "for imaging connectomics"),
            ("Unstructured reference with no sentence delimiters",
             "Unstructured reference with no sentence delimiters"),
        )
        for entry, expected in cases:
            with self.subTest(entry=entry):
                link = resolution.offline_link(18, entry)
                self.assertEqual(
                    parse_qs(urlsplit(link.url).query)["q"],
                    [expected],
                )

    def test_titleless_citations_use_bibliographic_coordinates(self) -> None:
        cases = (
            (
                "Smith J, Jones A. Phys Rev Lett 1998;80(2):1234-1237.",
                "Smith Phys Rev Lett 1998 80 1234",
            ),
            (
                "Smith J. Phys. Rev. Lett. 1998;80:1234-1237.",
                "Smith Phys. Rev. Lett 1998 80 1234",
            ),
            (
                "Smith J. J. Neurosci. Methods. 1998;80:1234-37.",
                "Smith J. Neurosci. Methods 1998 80 1234",
            ),
            (
                "Smith, J. (1998). Phys. Rev. Lett. 80(2), 1234-37.",
                "Smith Phys. Rev. Lett 1998 80 1234",
            ),
            (
                "Smith J. Nature 392, 1234-1237 (1998).",
                "Smith Nature 1998 392 1234",
            ),
            (
                "Nature 1998;392:1234.",
                "Nature 1998 392 1234",
            ),
            (
                "Phys Rev Lett 80:1234.",
                "Phys Rev Lett 80 1234",
            ),
            (
                "Smith J. Science 80:1234.",
                "Smith Science 80 1234",
            ),
            (
                "Smith J. Front Hum Neurosci 2015;9:e2010075118.",
                "Smith Front Hum Neurosci 2015 9 e2010075118",
            ),
        )
        with patch.object(resolution, "_json_request") as request:
            for reference, expected in cases:
                with self.subTest(reference=reference):
                    result = resolution.offline_link(8, reference)
                    query = parse_qs(urlsplit(result.url).query)["q"][0]
                    self.assertEqual(query, expected)
                    self.assertEqual(result.kind, "search")
            request.assert_not_called()

    def test_short_actual_title_precedes_journal_coordinates(self) -> None:
        cases = (
            (
                "Smith J. Cell motion. Phys. Rev. Lett. 1998;80:1234.",
                "Cell motion",
            ),
            (
                "Smith J. Deep learning. Nature 2015;521:436-444.",
                "Deep learning",
            ),
            (
                "Smith J. A network model. Phys Rev Lett 1998;80:1234.",
                "A network model",
            ),
            (
                "Smith J. Brain network communication: concepts, models "
                "and applications. Nat Rev Neurosci 2023;24:557-74.",
                "Brain network communication: concepts, models "
                "and applications",
            ),
            (
                "Lee J. The challenges of data analysis: a review. "
                "Nature 2020;12:22-24.",
                "The challenges of data analysis: a review",
            ),
        )
        for entry, title in cases:
            with self.subTest(entry=entry):
                url = resolution.offline_link(2, entry).url
                self.assertEqual(
                    parse_qs(urlsplit(url).query)["q"], [title]
                )

    def test_printed_doi_is_a_link_without_verification(self) -> None:
        entry = "Smith J. Paper. doi:10.1234/example.2020."
        link = resolution.offline_link(9, entry)
        self.assertEqual(link.kind, "printed_doi")
        self.assertEqual(link.doi, "10.1234/example.2020")
        self.assertEqual(
            link.url, "https://doi.org/10.1234/example.2020"
        )

    def test_no_permission_never_runs_requests(self) -> None:
        with patch.object(resolution, "_json_request") as request:
            for permission in (
                None,
                resolution.LookupPermission((2,)),
                resolution.LookupPermission((1,), ("https://evil.test",)),
            ):
                with self.subTest(permission=permission):
                    with self.assertRaises(PermissionError):
                        resolution.resolve_links({1: ENTRY}, permission)
        request.assert_not_called()

    def test_url_boundaries_reject_unapproved_hosts(self) -> None:
        for value in (
            "https://evil.test/works?q=x",
            "http://api.crossref.org/works",
            "https://api.crossref.org.evil.test/works",
            "https://user@api.crossref.org/works",
            "https://api.crossref.org/other",
        ):
            with self.subTest(url=value):
                with self.assertRaises(ValueError):
                    resolution._json_request(
                        value, "https://api.crossref.org",
                        resolution.LookupPermission((1,)),
                    )

    def test_network_boundary_requires_permission(self) -> None:
        with patch.object(resolution, "build_opener") as opener:
            with self.assertRaises(PermissionError):
                resolution._json_request(
                    "https://api.crossref.org/works",
                    "https://api.crossref.org", None,
                )
        opener.assert_not_called()

    def test_redirects_are_not_followed(self) -> None:
        handler = resolution._NoRedirect()
        self.assertIsNone(handler.redirect_request(
            None, None, 302, "redirect", {}, "https://evil.test"
        ))


class TestOnlineReferenceResolution(TestCase):
    """Match service responses conservatively and fall back safely."""

    def setUp(self) -> None:
        self.permission = resolution.LookupPermission((4,))

    def test_good_match_includes_doi_and_optional_pdf_metadata(self) -> None:
        def response(
            url: str, service: str, permission: object
        ) -> dict:
            if "crossref" in service:
                return {"message": {"items": [WORK]}}
            self.assertIn("10.1234/example.2020", url)
            return {
                "best_oa_location": {
                    "pdf_url": "https://repository.test/manuscript.pdf",
                    "license": "cc-by", "version": "acceptedVersion",
                }
            }

        with patch.object(
            resolution, "_json_request", side_effect=response
        ) as calls:
            result = resolution.resolve_links({4: ENTRY}, self.permission)
        self.assertEqual(calls.call_count, 2)
        link = result[4]
        self.assertEqual(link.kind, "verified_doi")
        self.assertEqual(link.doi, "10.1234/example.2020")
        self.assertEqual(link.url, "https://doi.org/10.1234/example.2020")
        self.assertEqual(
            link.pdf_url, "https://repository.test/manuscript.pdf"
        )
        self.assertEqual(link.license, "cc-by")
        self.assertEqual(link.version, "acceptedVersion")

    def test_titleless_match_with_corresponding_journal_fields(self) -> None:
        entry = "Smith J. Phys Rev Lett 1998;80(2):1234-1237."
        work = {
            **WORK,
            "title": ["A completely different real article title"],
            "container-title": ["Physical Review Letters"],
            "short-container-title": ["Phys. Rev. Lett."],
            "volume": "80", "issue": "2", "page": "1234-1237",
            "published": {"date-parts": [[1998, 2]]},
        }
        urls = []

        def response(url: str, service: str, permission: object) -> dict:
            urls.append(url)
            if "crossref" in service:
                return {"message": {"items": [work]}}
            return {"best_oa_location": None}

        with patch.object(
            resolution, "_json_request", side_effect=response
        ) as request:
            link = resolution.resolve_links({4: entry}, self.permission)[4]
        self.assertEqual(request.call_count, 2)
        self.assertEqual(link.kind, "verified_doi")
        query = parse_qs(urlsplit(urls[0]).query)[
            "query.bibliographic"
        ][0]
        self.assertEqual(query, "Smith Phys Rev Lett 1998 80 1234")

    def test_titleless_wrong_coordinates_or_author_rejected(self) -> None:
        entry = "Smith J. Phys Rev Lett 1998;80(2):1234-37."
        work = {
            **WORK, "title": ["Unrelated publication title"],
            "container-title": ["Physical Review Letters"],
            "volume": "80", "issue": "2", "page": "1234-37",
            "published": {"date-parts": [[1998]]},
        }
        variations = (
            {**work, "container-title": ["Physical Review A"]},
            {**work, "volume": "81"},
            {**work, "page": "1235-1237"},
            {**work, "page": ""},
            {**work, "issue": "3"},
            {**work, "author": [{"family": "Jones"}]},
            {**work, "published": {"date-parts": [[1999]]}},
            {**work, "container-title": []},
        )
        for candidate in variations:
            with self.subTest(candidate=candidate):
                with patch.object(
                    resolution, "_json_request",
                    return_value={"message": {"items": [candidate]}},
                ) as request:
                    link = resolution.resolve_links(
                        {4: entry}, self.permission
                    )[4]
                self.assertEqual(link.kind, "search")
                request.assert_called_once()

    def test_titleless_matches_article_number_without_page(self) -> None:
        entry = "Smith J. PLoS One 2020;15:e0123456."
        candidate = {
            **WORK, "container-title": ["PLOS ONE"],
            "volume": "15", "article-number": "e0123456",
        }
        with patch.object(
            resolution, "_json_request",
            side_effect=[
                {"message": {"items": [candidate]}},
                {"best_oa_location": None},
            ],
        ) as request:
            link = resolution.resolve_links({4: entry}, self.permission)[4]
        self.assertEqual(request.call_count, 2)
        self.assertEqual(link.kind, "verified_doi")

    def test_coordinates_without_author_must_be_unique(self) -> None:
        entry = "Nature 2020;12:22."
        candidate = {
            **WORK, "container-title": ["Nature"],
            "volume": "12", "page": "22-24",
        }
        with patch.object(
            resolution, "_json_request",
            side_effect=[
                {"message": {"items": [candidate]}},
                {"best_oa_location": None},
            ],
        ):
            link = resolution.resolve_links({4: entry}, self.permission)[4]
        self.assertEqual(link.kind, "verified_doi")
        with patch.object(
            resolution, "_json_request",
            return_value={"message": {"items": [candidate, candidate]}},
        ) as request:
            link = resolution.resolve_links({4: entry}, self.permission)[4]
        self.assertEqual(link.kind, "search")
        request.assert_called_once()

    def test_coordinates_without_year_remain_unverified(self) -> None:
        entry = "Nature 12:22."
        candidate = {
            **WORK, "container-title": ["Nature"],
            "volume": "12", "page": "22-24",
        }
        with patch.object(
            resolution, "_json_request",
            return_value={"message": {"items": [candidate]}},
        ) as request:
            link = resolution.resolve_links({4: entry}, self.permission)[4]
        self.assertEqual(link.kind, "search")
        request.assert_called_once()

    def test_wrong_author_year_or_title_remains_search(self) -> None:
        cases = (
            {**WORK, "author": [{"family": "Jones"}]},
            {**WORK, "published": {"date-parts": [[2021]]}},
            {**WORK, "title": ["Molecular genetics in diseased cells"]},
            {**WORK, "author": [{"family": ""}]},
        )
        for candidate in cases:
            with self.subTest(candidate=candidate):
                with patch.object(
                    resolution, "_json_request",
                    return_value={"message": {"items": [candidate]}},
                ) as request:
                    link = resolution.resolve_links(
                        {4: ENTRY}, self.permission
                    )[4]
                self.assertEqual(link.kind, "search")
                self.assertEqual(request.call_count, 1)

    def test_http_failure_keeps_offline_link(self) -> None:
        with patch.object(
            resolution, "_json_request", side_effect=URLError("offline")
        ):
            link = resolution.resolve_links({4: ENTRY}, self.permission)[4]
        self.assertEqual(link.kind, "search")

    def test_printed_doi_is_checked_against_crossref(self) -> None:
        entry = "Smith, J. Paper. doi: 10.1234/example.2020."
        with patch.object(
            resolution, "_json_request",
            side_effect=[
                {"message": WORK},
                {"best_oa_location": None},
            ],
        ) as request:
            link = resolution.resolve_links({4: entry}, self.permission)[4]
        self.assertEqual(request.call_count, 2)
        self.assertEqual(link.kind, "verified_doi")

    def test_mismatched_printed_doi_cannot_be_verified(self) -> None:
        entry = "Smith, J. Paper. doi: 10.1234/another.2020."
        with patch.object(
            resolution, "_json_request", return_value={"message": WORK}
        ) as request:
            link = resolution.resolve_links({4: entry}, self.permission)[4]
        self.assertEqual(request.call_count, 1)
        self.assertEqual(link.kind, "printed_doi")


if __name__ == "__main__":
    import unittest
    unittest.main()
