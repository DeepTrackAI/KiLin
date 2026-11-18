"""Unit tests for author-year citations and their reference entries.

Check which markers are found and what they cite, how entries are keyed in
the common styles, how suffixes are given, and how works resolve.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools._shared.author_year import (
    CitedWork,
    entry_key,
    entry_keys,
    find_markers,
    normalized_name,
    resolve,
)


def _cited(text: str) -> list[tuple[str, str, int, str, str]]:
    """Return what each marker in a text cites.

    Parameters
    ----------
    text: str
        The text.

    Returns
    -------
    list[tuple[str, str, int, str, str]]
        First and second author, author count, year and suffix of each
        work, in order.

    """

    return [
        (work.first, work.second, work.authors, work.year, work.suffix)
        for marker in find_markers(text)
        for work in marker.works
    ]


class TestMarkers(unittest.TestCase):

    def test_parenthetical(self) -> None:
        for text, cited in (
            ("as shown (Smith 2020).", [("smith", "", 1, "2020", "")]),
            ("as shown (Smith, 2020).", [("smith", "", 1, "2020", "")]),
            ("(Smith et al., 2020; Lee and Wu 2019a, b)",
             [("smith", "", 3, "2020", ""), ("lee", "wu", 2, "2019", "a"),
              ("lee", "wu", 2, "2019", "b")]),
            ("(e.g., Smith & Lee, 2020, p. 5; see also Brown 2018)",
             [("smith", "lee", 2, "2020", ""),
              ("brown", "", 1, "2018", "")]),
            ("(Smith 2020, Lee 2021)",
             [("smith", "", 1, "2020", ""), ("lee", "", 1, "2021", "")]),
            ("(Smith 2020 for a review)", [("smith", "", 1, "2020", "")]),
            ("(Smith, Lee and Wu 2020)", [("smith", "", 3, "2020", "")]),
            ("(García López 2015)", [("lopez", "", 1, "2015", "")]),
            ("(O’Brien 2020)", [("obrien", "", 1, "2020", "")]),
            ("(Schrödinger 1926)", [("schrodinger", "", 1, "1926", "")]),
            ("(Smith n.d.)", [("smith", "", 1, "n.d.", "")]),
            ("(Smith, in press)", [("smith", "", 1, "in press", "")]),
            ("(data from Wang et al., 2020)", [("wang", "", 3, "2020", "")]),
            ("(e.g. the unified atlas, Chon et al.,\n2019)",
             [("chon", "", 3, "2019", "")]),
            ("(GitHub repository; Musy et al., 2019)",
             [("musy", "", 3, "2019", "")]),
            ("(data from Winnubst et al., 2019, downloaded from x.org, "
             "Ascoli et al., 2007)",
             [("winnubst", "", 3, "2019", ""), ("ascoli", "", 3, "2007", "")]),
            ("(Oh et al., 2014; Arganda-\nCarreras et al., 2018)",
             [("oh", "", 3, "2014", ""),
              ("arganda-carreras", "", 3, "2018", "")]),
            ("(Tyson et al., 2020a; Fu¨rth et al., 2018)",
             [("tyson", "", 3, "2020", "a"), ("furth", "", 3, "2018", "")]),
            ("(In Tyson et al., 2020a)", [("tyson", "", 3, "2020", "a")]),
            ("(Hunter, 2007; The pandas development team, 2020)",
             [("hunter", "", 1, "2007", ""),
              ("the pandas development team", "", 1, "2020", "")]),
        ):
            with self.subTest(text=text):
                self.assertEqual(_cited(text), cited)

    def test_narrative(self) -> None:
        for text, shown, cited in (
            ("Smith et al. (2020) showed", "Smith et al. (2020)",
             [("smith", "", 3, "2020", "")]),
            ("Smith and Lee (2020a, 2021) argued", "Smith and Lee "
             "(2020a, 2021)", [("smith", "lee", 2, "2020", "a"),
                               ("smith", "lee", 2, "2021", "")]),
            ("as in van den Heuvel et al. (2016).",
             "van den Heuvel et al. (2016)", [("heuvel", "", 3, "2016", "")]),
            ("However, Smith and Lee (2020) found", "Smith and Lee (2020)",
             [("smith", "lee", 2, "2020", "")]),
            ("Recently, Smith et al. (2020) reported", "Smith et al. (2020)",
             [("smith", "", 3, "2020", "")]),
            ("Smith’s (2020) model", "Smith’s (2020)",
             [("smith", "", 1, "2020", "")]),
        ):
            with self.subTest(text=text):
                markers = find_markers(text)
                self.assertEqual([m.text for m in markers], [shown])
                self.assertTrue(markers[0].narrative)
                self.assertEqual(text[markers[0].start:markers[0].end],
                                 shown)
                self.assertEqual(_cited(text), cited)

    def test_not_citations(self) -> None:
        for text in ("(Phase 2020)", "(n = 2020)", "(Figure 2020)",
                     "In 2020 (Table 1) we", "(WHO 2020)", "(see below)",
                     "since (2020)", "(March 2020)"):
            with self.subTest(text=text):
                self.assertEqual(find_markers(text), [])

    def test_organizational_authors_keep_their_identity(self) -> None:
        for text, expected in (
            ("(The pandas development team, 2020)",
             "the pandas development team"),
            ("(The PyTorch development team, 2020)",
             "the pytorch development team"),
            ("The pandas development team (2020) reported",
             "the pandas development team"),
            ("World Health Organization (2020) reported",
             "world health organization"),
            ("(World Health Organization 2020)",
             "world health organization"),
        ):
            with self.subTest(text=text):
                markers = find_markers(text)
                self.assertEqual(len(markers), 1)
                self.assertEqual(markers[0].works[0].first, expected)
                self.assertEqual(markers[0].works[0].authors, 1)

    def test_short_and_linked_organizational_names(self) -> None:
        """Retain corporate identities without changing personal names.

        """

        for name in (
            "European Commission", "Federal Commission", "United Nations",
            "African Nations", "National Institutes of Health",
            "Department of Health", "World Bank",
            "Centers for Disease Control and Prevention",
        ):
            expected = name.casefold()
            for text in (
                f"({name}, 2020)",
                f"(e.g., {name}, 2020)",
                f"{name} (2020) reported",
            ):
                with self.subTest(text=text):
                    markers = find_markers(text)
                    self.assertEqual(len(markers), 1)
                    self.assertEqual(len(markers[0].works), 1)
                    self.assertEqual(markers[0].works[0].first, expected)
                    self.assertEqual(markers[0].works[0].authors, 1)
        for text, expected in (
            ("(García López 2015)", ("lopez", "", 1)),
            ("(van den Heuvel 2020)", ("heuvel", "", 1)),
            ("Smith and Lee (2020) found", ("smith", "lee", 2)),
        ):
            with self.subTest(text=text):
                work = find_markers(text)[0].works[0]
                self.assertEqual(
                    (work.first, work.second, work.authors), expected
                )


class TestEntries(unittest.TestCase):

    def test_styles(self) -> None:
        for entry, key in (
            ("Smith, J. A., Lee, B., & Wu, C. (2020). Title. J, 1, 2–3.",
             ("smith", "", 3, "2020", "")),
            ("Smith, J. A., & Lee, B. (2020a). Title.",
             ("smith", "lee", 2, "2020", "a")),
            ("Smith JA, Lee B. Title. J Biol. 2020;12:3-4.",
             ("smith", "lee", 2, "2020", "")),
            ("Smith, J. & Lee, B. Title of the paper. Nature 1, 2 (2020).",
             ("smith", "lee", 2, "2020", "")),
            ("Smith, John A., and Bo Lee. 2020. “Title.” Journal 1: 2.",
             ("smith", "", None, "2020", "")),
            ("van den Heuvel MP, Sporns O. Rich. J Neurosci. 2011;31:1.",
             ("heuvel", "sporns", 2, "2011", "")),
            ("Smith, J., et al. (2019). Title.", ("smith", "", 3, "2019", "")),
            ("Smith, J. (n.d.). Unpublished.", ("smith", "", 1, "n.d.", "")),
            ("Hunter JD. 2007. Matplotlib. CiSE 9:90–95.",
             ("hunter", "", 1, "2007", "")),
            ("Ding S-L, Royall JJ. 2016. An atlas. J Comp Neurol 524:3.",
             ("ding", "royall", 2, "2016", "")),
            ("Fürth D, Vaissière T, Tzortzi O. 2018. A framework. N 21:1.",
             ("furth", "", 3, "2018", "")),
            ("Tyson AL, Rousseau CV. 2020b. brainreg. Zenodo.",
             ("tyson", "rousseau", 2, "2020", "b")),
            ("The pandas development team. 2020. Pandas. Zenodo. 1.2.3.",
             ("the pandas development team", "", None, "2020", "")),
            ("The PyTorch development team. 2020. PyTorch. Zenodo.",
             ("the pytorch development team", "", None, "2020", "")),
            ("World Health Organization. 2020. Global health. Geneva.",
             ("world health organization", "", None, "2020", "")),
            ("van den Heuvel. 2020. Rich clubs in brain networks.",
             ("heuvel", "", None, "2020", "")),
            ("García López. 2015. A study of brain systems.",
             ("lopez", "", None, "2015", "")),
            ("Smith JA. 2020. A comment on (2018) findings. J Biol.",
             ("smith", "", 1, "2020", "")),
            ("Smith, J. (2020). A comment on (2018) findings.",
             ("smith", "", 1, "2020", "")),
            ("Tyson AL, Rousseau CV. 2020b. Notes on (2018) methods.",
             ("tyson", "rousseau", 2, "2020", "b")),
        ):
            with self.subTest(entry=entry):
                found = entry_key(7, entry)
                self.assertEqual(
                    (found.first, found.second, found.authors, found.year,
                     found.suffix), key,
                )
                self.assertEqual(found.number, 7)
        self.assertIsNone(entry_key(1, "Smith, J. Title without a year."))

    def test_short_organizational_entry_heads(self) -> None:
        """Keep full corporate names from common year-bearing headers.

        """

        for name in (
            "European Commission", "Federal Commission", "United Nations",
            "African Nations", "National Institutes of Health",
            "Department of Health", "World Bank",
            "Centers for Disease Control and Prevention",
        ):
            for entry in (
                f"{name}. 2020. A report on global research.",
                f"{name} (2020). A report on global research.",
            ):
                with self.subTest(entry=entry):
                    key = entry_key(7, entry)
                    self.assertIsNotNone(key)
                    self.assertEqual(key.first, name.casefold())
        self.assertEqual(
            entry_key(7, "García López. 2015. A study.").first, "lopez"
        )

    def test_suffixes_in_list_order(self) -> None:
        keys = entry_keys([
            (1, "Lee, B. (2019). One."), (2, "Lee, B. (2019). Two."),
            (3, "Lee, B., & Wu, C. (2019). Three."),
            (4, "Ode, D. (2018b). Four."), (5, "Ode, D. (2018a). Five."),
        ])
        self.assertEqual([(k.number, k.suffix) for k in keys],
                         [(1, "a"), (2, "b"), (3, ""), (4, "b"), (5, "a")])

    def test_normalized_name(self) -> None:
        self.assertEqual(normalized_name("van den Heuvel"), "heuvel")
        self.assertEqual(normalized_name("García-López"), "garcia-lopez")
        self.assertEqual(normalized_name("O’Brien"), "obrien")
        self.assertEqual(normalized_name("Fu¨rth"), "furth")
        self.assertEqual(normalized_name("Fu¨ rth"), "furth")
        self.assertEqual(normalized_name("Rı´o"), "rio")
        self.assertEqual(normalized_name("Arganda-\nCarreras"),
                         "arganda-carreras")
        self.assertEqual(normalized_name(""), "")


class TestResolve(unittest.TestCase):

    def setUp(self) -> None:
        self.keys = entry_keys([
            (1, "Lee, B. (2019). One."), (2, "Lee, B. (2019). Two."),
            (3, "Smith, J., & Lee, B. (2020). Three."),
            (4, "Smith, J., Wu, C., & Ode, D. (2020). Four."),
            (5, "Smith, J., & Wu, C. (2020). Five."),
        ])

    def _work(self, first, second, authors, year, suffix=""):
        return CitedWork(first, second, authors, year, suffix, "")

    def test_resolved_ambiguous_and_missing(self) -> None:
        for work, numbers in (
            (self._work("lee", "", 1, "2019", "b"), (2,)),
            (self._work("lee", "", 1, "2019"), (1, 2)),
            (self._work("smith", "wu", 2, "2020"), (5,)),
            (self._work("smith", "", 3, "2020"), (4,)),
            # A single-author work cannot match multi-author entries.
            (self._work("smith", "", 1, "2020"), ()),
            (self._work("smith", "", 1, "2021"), ()),
            (self._work("brown", "", 1, "2019"), ()),
        ):
            with self.subTest(work=work):
                self.assertEqual(resolve(work, self.keys), numbers)

    def test_author_disagreements_do_not_resolve(self) -> None:
        keys = entry_keys([
            (1, "Smith, J., & Wu, C. (2020). A study."),
            (2, "Smith, J. (2021). A single-author study."),
            (3, "Lee, B., Wu, C., & Ode, D. (2020). A study."),
        ])
        for work in (
            self._work("smith", "lee", 2, "2020"),
            self._work("smith", "", 3, "2020"),
            self._work("smith", "", 1, "2020"),
            self._work("lee", "", 2, "2020"),
        ):
            with self.subTest(work=work):
                self.assertEqual(resolve(work, keys), ())

    def test_unknown_author_count_is_not_a_contradiction(self) -> None:
        keys = entry_keys([
            (1, "Smith, John A., and Bo Lee. 2020. A study. J 1:2."),
            (2, "Smith, J., & Wu, C. (2020). Another study."),
        ])
        self.assertEqual(
            resolve(self._work("smith", "lee", 2, "2020"), keys),
            (1,),
        )

    def test_same_year_organizations_resolve_separately(self) -> None:
        keys = entry_keys([
            (1, "The pandas development team. 2020. Pandas. Zenodo."),
            (2, "The PyTorch development team. 2020. PyTorch. Zenodo."),
        ])
        for text, number in (
            ("(The pandas development team, 2020)", 1),
            ("(The PyTorch development team, 2020)", 2),
            ("The pandas development team (2020) found", 1),
        ):
            with self.subTest(text=text):
                work = find_markers(text)[0].works[0]
                self.assertEqual(resolve(work, keys), (number,))

    def test_organizations_do_not_resolve_to_other_organizations(self) -> None:
        """Avoid false attribution when only the final word matches.

        """

        entries = entry_keys([
            (1, "Federal Commission. 2020. A report about physics."),
            (2, "African Nations. 2020. A report about physics."),
            (3, "Department of Health. 2020. A public report."),
            (4, "European Commission. 2020. A separate report."),
        ])
        for name, expected in (
            ("European Commission", (4,)),
            ("United Nations", ()),
            ("National Institutes of Health", ()),
        ):
            with self.subTest(name=name):
                work = find_markers(f"({name}, 2020)")[0].works[0]
                self.assertEqual(resolve(work, entries), expected)


if __name__ == "__main__":
    unittest.main()
