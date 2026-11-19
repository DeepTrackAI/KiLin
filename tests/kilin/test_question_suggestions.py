"""Unit tests for suggested questions about a project's own sources.

The model is a stub, so the tests check what the module does with its
reply: which words are kept in the saved pool, when the model is asked,
how fill-in questions are completed anew on every call, and that a project
without sources gets none.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import unittest.mock
from unittest.mock import Mock

from kilin.question_suggestions import (
    SUGGESTION_WORDS_FILE,
    Suggestion,
    SuggestionWords,
    ask_for_words,
    checked_words,
    fill_templates,
    question_suggestions,
    suggestion_files,
    suggestion_words,
    welcome_prompts,
)
from tests.kilin.tools._shared._support import chunk

TEMPLATES = (
    ("paper_overview", "Give me a short summary of {file}."),
    ("compare_sources", "Compare the methods of {file} and {file2}."),
    ("question_answering", "What do the papers say about {topic}?"),
    ("mention_listing", "Which papers mention {term}?"),
    ("project_inventory", "How many papers are in this project?"),
)
CHUNKS = (
    chunk("a1", "Variational autoencoders model MRI scans.", path="a.pdf"),
    chunk("a2", "We trained the autoencoders on MRI data.", path="a.pdf"),
    chunk("b1", "Sleep quality in older adults.", path="papers/b.pdf"),
)
REPLY = (
    "Here you are:\n"
    '{"themes": ["autoencoders for MRI", "quantum gravity"], '
    '"topics": ["variational autoencoders", "quantum gravity"], '
    '"terms": ["MRI", "fMRI"]}'
)


class TestFillingQuestions(unittest.TestCase):
    """Complete fill-in questions with files, topics, and terms."""

    def test_files_are_named_in_rank_order(self) -> None:
        """Order by folder depth and passages; name duplicates by path."""

        chunks = (*CHUNKS, chunk("c1", "Other copy.", path="other/b.pdf"))
        self.assertEqual(
            suggestion_files(chunks), ("a.pdf", "other/b.pdf", "papers/b.pdf")
        )

    def test_code_projects_name_only_code_files(self) -> None:
        """Name only files with code symbols when there are any."""

        readme = chunk("r1", "Read me.", path="README.md")
        code = replace(
            chunk("p1", "def f(): pass", path="pkg/f.py"), symbol="f"
        )
        self.assertEqual(suggestion_files((readme, readme, code)), ("f.py",))

    def test_templates_take_values_in_turn(self) -> None:
        """Name different files, and skip questions without a value."""

        words = SuggestionWords(("sleep quality",), ())
        questions = fill_templates(TEMPLATES, ("a.pdf", "b.pdf"), words)
        self.assertEqual(
            questions,
            [
                Suggestion(
                    "paper_overview", "Give me a short summary of a.pdf."
                ),
                Suggestion(
                    "compare_sources",
                    "Compare the methods of b.pdf and a.pdf.",
                ),
                Suggestion(
                    "question_answering",
                    "What do the papers say about sleep quality?",
                ),
                Suggestion(
                    "project_inventory", "How many papers are in this project?"
                ),
            ],
        )

    def test_variants_name_other_files_and_topics(self) -> None:
        """Write up to three different questions per template."""

        words = SuggestionWords(("sleep", "memory"), ())
        questions = fill_templates(
            TEMPLATES, ("a.pdf", "b.pdf", "c.pdf", "d.pdf"), words, 3
        )
        overviews = [
            item.question
            for item in questions
            if item.intent == "paper_overview"
        ]
        self.assertEqual(len(overviews), 3)
        self.assertEqual(len(set(overviews)), 3)
        topics = [
            item.question
            for item in questions
            if item.intent == "question_answering"
        ]
        self.assertEqual(len(topics), 2)
        inventory = [
            item for item in questions if item.intent == "project_inventory"
        ]
        self.assertEqual(len(inventory), 1)
        for item in questions:
            if item.intent == "compare_sources":
                first, second = item.question[23:-1].split(" and ")
                self.assertNotEqual(first, second)

    def test_comparison_needs_two_files(self) -> None:
        """Leave out a two-file question when there is one file."""

        questions = fill_templates(
            TEMPLATES[:2], ("a.pdf",), SuggestionWords((), ())
        )
        self.assertEqual(
            [item.intent for item in questions], ["paper_overview"]
        )


class TestModelWords(unittest.TestCase):
    """Read the model's reply and keep only words printed in the sources."""

    def test_reply_json_is_read_from_surrounding_text(self) -> None:
        """Find the JSON object even with words around it."""

        ask = Mock(return_value=REPLY)
        words = ask_for_words(CHUNKS, (), ask)
        self.assertEqual(
            words.topics, ("variational autoencoders", "quantum gravity")
        )
        self.assertEqual(words.terms, ("MRI", "fMRI"))
        self.assertEqual(
            words.themes, ("autoencoders for MRI", "quantum gravity")
        )
        system, prompt, tokens = ask.call_args.args
        self.assertIn("JSON", system)
        self.assertIn("- a.pdf: Variational autoencoders", prompt)
        self.assertGreater(tokens, 0)

    def test_card_describes_its_file(self) -> None:
        """Describe a file by its card's title and aim when it has one."""

        card = SimpleNamespace(
            path=Path("a.pdf"),
            metadata=(("title", "Deep generative MRI"),),
            statements=(SimpleNamespace(field="aim", text="Model scans."),),
        )
        ask = Mock(return_value="{}")
        ask_for_words(CHUNKS, (card,), ask)
        self.assertIn(
            "- a.pdf: Deep generative MRI Model scans.", ask.call_args.args[1]
        )

    def test_unusable_replies_give_no_words(self) -> None:
        """Treat prose, broken JSON, and wrong types as no words."""

        for reply in ("No JSON here.", "{broken", '{"topics": "one"}', "[]"):
            with self.subTest(reply=reply):
                words = ask_for_words(CHUNKS, (), Mock(return_value=reply))
                self.assertEqual(words, SuggestionWords((), ()))

    def test_only_printed_words_are_kept(self) -> None:
        """Drop topics and terms that the sources do not contain."""

        words = checked_words(
            SuggestionWords(
                ("variational autoencoders", "quantum gravity"),
                ("MRI", "fMRI", "auto"),
            ),
            CHUNKS,
        )
        self.assertEqual(words.topics, ("variational autoencoders",))
        self.assertEqual(words.terms, ("MRI",))

    def test_one_theme_instead_of_a_list_is_read(self) -> None:
        """Read a single "theme" when a model writes no list."""

        ask = Mock(return_value='{"theme": "MRI", "topics": [], "terms": []}')
        self.assertEqual(ask_for_words(CHUNKS, (), ask).themes, ("MRI",))

    def test_themes_are_checked_and_spelled_as_printed(self) -> None:
        """Keep short printed single themes in the sources' spelling."""

        chunks = (chunk("x", "We study Alzheimer's disease and resilience."),)
        themes = (
            "alzheimer's disease and resilience",
            "disease, resilience",
            "quantum gravity",
            "study of alzheimer's disease resilience",
            "alzheimer's disease",
            "resilience",
            "disease",
            "study",
        )
        words = checked_words(SuggestionWords((), (), themes), chunks)
        self.assertEqual(
            words.themes, ("Alzheimer's disease", "resilience", "disease")
        )

    def test_topics_are_spelled_as_printed(self) -> None:
        """Keep a name's capital and lower-case an ordinary word."""

        chunks = (chunk("x", "We study Alzheimer's disease. Disease varies."),)
        words = checked_words(
            SuggestionWords(("alzheimer's Disease",), ("a.pdf",)), chunks
        )
        self.assertEqual(words.topics, ("Alzheimer's disease",))
        chunks = (chunk("y", "We count T cells. A T cell enters; a t-test."),)
        words = checked_words(SuggestionWords(("t cell entry",), ()), chunks)
        self.assertEqual(words.topics, ())
        words = checked_words(SuggestionWords(("t cell",), ()), chunks)
        self.assertEqual(words.topics, ("T cell",))


class TestWordPool(unittest.TestCase):
    """Pick the pool once, save it, and reuse it until the sources change."""

    def setUp(self) -> None:
        """Create an empty index folder."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.indexes = Path(temporary.name)

    def _pool(self, ask: object, model: str = "small") -> SuggestionWords:
        """Return the pool for the test sources."""

        return suggestion_words(self.indexes, CHUNKS, (), ask, model)

    def test_model_words_are_checked_and_saved(self) -> None:
        """Ask once; later calls read the saved pool."""

        ask = Mock(return_value=REPLY)
        pool = self._pool(ask)
        self.assertEqual(pool.topics, ("variational autoencoders",))
        self.assertEqual(pool.terms, ("MRI",))
        self.assertEqual(self._pool(ask), pool)
        ask.assert_called_once()
        self.assertEqual(pool.themes, ("autoencoders for MRI",))
        saved = json.loads((self.indexes / SUGGESTION_WORDS_FILE).read_text())
        self.assertEqual(saved["topics"], ["variational autoencoders"])
        self.assertEqual(saved["themes"], ["autoencoders for MRI"])

    def test_another_model_picks_again(self) -> None:
        """Ask again when the model changes."""

        ask = Mock(return_value=REPLY)
        self._pool(ask, "small")
        self._pool(ask, "other")
        self.assertEqual(ask.call_count, 2)

    def test_without_a_model_the_pool_is_empty_and_not_saved(self) -> None:
        """Save nothing without a model, or after it fails."""

        for ask in (None, Mock(side_effect=RuntimeError("not loaded"))):
            with self.subTest(ask=ask):
                self.assertEqual(self._pool(ask), SuggestionWords((), ()))
                self.assertFalse(
                    (self.indexes / SUGGESTION_WORDS_FILE).exists()
                )

    def test_damaged_saved_file_is_picked_again(self) -> None:
        """Ignore a saved file that cannot be read."""

        (self.indexes / SUGGESTION_WORDS_FILE).write_text("not json")
        ask = Mock(return_value=REPLY)
        self.assertTrue(self._pool(ask).topics)
        ask.assert_called_once()

    def test_no_sources_need_no_model(self) -> None:
        """Return an empty pool without asking for a project without files."""

        ask = Mock(return_value=REPLY)
        self.assertEqual(
            suggestion_words(self.indexes, (), (), ask, "m"),
            SuggestionWords((), ()),
        )
        ask.assert_not_called()

    def test_model_reads_the_first_files_and_a_random_mix(self) -> None:
        """Show the first files and another random mix on each pick."""

        chunks = [chunk("own", "My draft.", path="draft.pdf")] + [
            chunk(f"r{n}", f"Paper {n}.", path=f"refs/r{n:02d}.pdf")
            for n in range(40)
        ]
        prompts = []
        for seed in (1, 2):
            ask = Mock(return_value="{}")
            ask_for_words(chunks, (), ask, random.Random(seed))
            prompts.append(ask.call_args.args[1])
        self.assertTrue(all("- draft.pdf: My draft." in p for p in prompts))
        self.assertEqual(prompts[0].count("\n- "), prompts[1].count("\n- "))
        self.assertNotEqual(prompts[0], prompts[1])


class TestFreshQuestions(unittest.TestCase):
    """Write new questions from the pool on every call, without a model."""

    WORDS = SuggestionWords(("sleep", "memory", "attention"), ("MRI",))

    def test_questions_change_between_calls(self) -> None:
        """Name other files and topics from one call to the next."""

        chunks = [
            chunk(f"c{n}", "sleep memory attention MRI", path=f"p{n}.pdf")
            for n in range(10)
        ]
        rng = random.Random(0)
        seen = {
            item.question
            for _ in range(20)
            for item in question_suggestions(
                TEMPLATES, chunks, self.WORDS, rng
            )
            if item.intent == "paper_overview"
        }
        self.assertGreater(len(seen), 5)

    def test_without_words_only_file_questions_are_written(self) -> None:
        """Leave out questions that need a topic or term."""

        questions = question_suggestions(
            TEMPLATES, CHUNKS, SuggestionWords((), ())
        )
        intents = {item.intent for item in questions}
        self.assertEqual(
            intents,
            {"paper_overview", "compare_sources", "project_inventory"},
        )

    def test_no_sources_or_templates_give_no_questions(self) -> None:
        """Leave the fixed suggestions to the caller when nothing fits."""

        self.assertEqual(question_suggestions(TEMPLATES, (), self.WORDS), [])
        self.assertEqual(question_suggestions((), CHUNKS, self.WORDS), [])


class TestWelcomePrompts(unittest.TestCase):
    """Write welcome headings from the same pool, without a model."""

    WELCOME = (
        "What would you like to know about {topic}?",
        "Which finding in {file} should we look at first?",
        "Where should we start: {file} or {file2}?",
    )

    def test_headings_name_the_sources(self) -> None:
        """Fill one heading per template with the project's values."""

        words = SuggestionWords(("sleep quality",), ())
        headings = welcome_prompts(self.WELCOME, CHUNKS, words)
        self.assertEqual(len(headings), 3)
        self.assertIn(
            "What would you like to know about sleep quality?", headings
        )
        self.assertTrue(all("{" not in text for text in headings))

    def test_headings_change_between_calls(self) -> None:
        """Name other topics from one call to the next."""

        words = SuggestionWords(("sleep", "memory", "attention"), ())
        rng = random.Random(0)
        seen = {
            welcome_prompts(self.WELCOME[:1], CHUNKS, words, rng)[0]
            for _ in range(20)
        }
        self.assertEqual(len(seen), 3)

    def test_theme_heading_covers_the_whole_collection(self) -> None:
        """Name the collection's theme, not one file or topic."""

        words = SuggestionWords(("sleep",), (), ("memory in aging",))
        self.assertEqual(
            welcome_prompts(
                ("What would you like to understand about {theme}?",),
                CHUNKS,
                words,
            ),
            ["What would you like to understand about memory in aging?"],
        )
        self.assertEqual(
            welcome_prompts(
                ("About {theme}?",), CHUNKS, SuggestionWords((), ())
            ),
            [],
        )

    def test_headings_take_turns_naming_each_theme(self) -> None:
        """Show another theme in other chats, from the same saved pool."""

        words = SuggestionWords((), (), ("memory", "sleep", "aging"))
        seen = {
            welcome_prompts(
                ("About {theme}?",), CHUNKS, words, random.Random(seed)
            )[0]
            for seed in range(20)
        }
        self.assertEqual(
            seen, {"About memory?", "About sleep?", "About aging?"}
        )

    def test_unfillable_headings_are_left_out(self) -> None:
        """Without words or sources, leave the fixed headings to the caller."""

        headings = welcome_prompts(
            self.WELCOME, CHUNKS, SuggestionWords((), ())
        )
        self.assertNotIn(
            "What would you like to know about {topic}?", headings
        )
        self.assertEqual(len(headings), 2)
        self.assertEqual(
            welcome_prompts(self.WELCOME, (), SuggestionWords(("x",), ())),
            [],
        )
        self.assertEqual(
            welcome_prompts((), CHUNKS, SuggestionWords((), ())), []
        )


class TestSessionPool(unittest.TestCase):
    """Keep the pool in memory until the session's passages change."""

    def test_pool_is_loaded_once_per_set_of_passages(self) -> None:
        """Pick the pool once, then only after the sources change."""

        from kilin import rag

        session = SimpleNamespace(
            _closed=False,
            _suggestion_pool=None,
            chunks=list(CHUNKS),
            source_cards=None,
            project=SimpleNamespace(indexes=Path("/unused")),
            _ask_router=Mock(),
            router_model="small",
            _behavior_config=lambda: SimpleNamespace(
                suggestion_templates=TEMPLATES
            ),
        )
        session._suggestion_words = lambda: rag.RAGSession._suggestion_words(
            session
        )
        words = SuggestionWords(("sleep quality",), ("MRI",))
        with unittest.mock.patch.object(
            rag, "suggestion_words", return_value=words
        ) as pick:
            first = rag.RAGSession.question_suggestions(session)
            rag.RAGSession.question_suggestions(session)
            self.assertEqual(pick.call_count, 1)
            session.chunks = list(CHUNKS)
            rag.RAGSession.question_suggestions(session)
            self.assertEqual(pick.call_count, 2)
        self.assertIn(
            "What do the papers say about sleep quality?",
            [item.question for item in first],
        )

    def test_welcome_uses_the_pool_unless_the_user_set_headings(self) -> None:
        """Share the pool, and keep headings the user wrote in Settings."""

        from kilin import rag

        packaged = SimpleNamespace(welcome_prompts=("Fixed heading?",))
        config = SimpleNamespace(
            welcome_prompts=("Fixed heading?",),
            welcome_templates=("What about {topic}?",),
            suggestion_templates=TEMPLATES,
        )
        session = SimpleNamespace(
            _closed=False,
            _suggestion_pool=None,
            chunks=list(CHUNKS),
            source_cards=None,
            project=SimpleNamespace(
                indexes=Path("/unused"),
                profile=SimpleNamespace(type_id="literature"),
            ),
            _ask_router=Mock(),
            router_model="small",
            _behavior_config=lambda: config,
        )
        session._suggestion_words = lambda: rag.RAGSession._suggestion_words(
            session
        )
        words = SuggestionWords(("sleep quality",), ())
        with (
            unittest.mock.patch.object(
                rag, "suggestion_words", return_value=words
            ) as pick,
            unittest.mock.patch.object(
                rag, "project_type", return_value=packaged
            ),
        ):
            self.assertEqual(
                rag.RAGSession.welcome_prompts(session),
                ["What about sleep quality?"],
            )
            rag.RAGSession.question_suggestions(session)
            self.assertEqual(pick.call_count, 1)
            config.welcome_prompts = ("My own heading?",)
            self.assertEqual(rag.RAGSession.welcome_prompts(session), [])


if __name__ == "__main__":
    unittest.main()
