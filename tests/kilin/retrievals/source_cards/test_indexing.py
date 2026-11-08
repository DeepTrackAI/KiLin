"""Verify incremental card indexing with deterministic model substitutes.

The test data uses source-relative chunks and manifests. Generation, reuse,
corruption recovery, and embedding are exercised without model downloads.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from kilin.chunking import Chunk
from kilin.retrievals.source_cards import (
    CARD_STATUS_FAILED,
    CARD_STATUS_NOT_APPLICABLE,
    CARD_STATUS_VALID,
    load_source_card_vectors,
    load_source_cards,
)
from kilin.retrievals.source_cards.indexing import reconcile_source_cards
from kilin.source_card_policy import (
    SourceCardOverride,
    SourceCardPolicy,
    write_source_card_overrides,
    write_source_card_policy,
)


def _passage(name, index=1):
    """Produce one supported scientific-paper excerpt."""
    return Chunk(
        f"{name}-{index}", Path(name),
        "Abstract. We measured a physical response using microscopy.",
        index, index + 3, "pdf_page", None, f"Page {index}",
    )


def _answer(*_args):
    """Return a complete, cited scientific-paper response."""
    return json.dumps({
        "applicable": True,
        "reason": "",
        "title": "Study",
        "authors": "A. Author",
        "year": "2026",
        "venue": "Journal",
        "identifiers": "",
        "abstract": "We measured the response [1].",
        "aim": "We studied microscopy [1].",
        "methods": "We used microscopy [1].",
        "data": "",
        "findings": "Responses were measured [1].",
        "limitations": "",
    })


class _Generator:
    """Minimal isolated model adapter with an observable lifetime."""

    def __init__(self, settings, *, calls, events, response, **_kwargs):
        self.settings = settings
        self.calls = calls
        self.events = events
        self.response = response
        self.events.append("created")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.events.append("closed")

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        self.events.append("generated")
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TestCardIndexing(unittest.TestCase):

    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "profile.ini").write_text(
            "[project]\ntype_id = literature\n", encoding="utf-8"
        )
        indexes = self.root / "indexes"
        indexes.mkdir()
        self.project = SimpleNamespace(
            root=self.root, indexes=indexes,
            profile=SimpleNamespace(type_id="literature",
                                    allow_download=False),
        )
        self.chunks = [_passage("a.pdf"), _passage("b.pdf")]
        self.hashes = {name: sha256(name.encode()).hexdigest()
                       for name in ("a.pdf", "b.pdf")}
        self.embed = SimpleNamespace(
            _retrieval_model_identity="embed:v1",
            embed_documents=Mock(side_effect=lambda texts: [
                [1.0, 0.0] for text in texts
            ]),
        )
        self.calls = []
        self.events = []
        self.response = _answer()

    def _manifest(self):
        return {"files": {name: {"sha256": digest}
                          for name, digest in self.hashes.items()}}

    def _run(self, *, expected_dimension=2, progress=None):
        def factory(settings, **kwargs):
            self.assertFalse(kwargs["allow_download"])
            return _Generator(
                settings, calls=self.calls, events=self.events,
                response=self.response,
            )
        return reconcile_source_cards(
            self.project, self.chunks, self._manifest(), self.embed,
            backend="torch", expected_dimension=expected_dimension,
            progress=progress, generator_factory=factory,
        )

    def _files(self):
        return (
            self.project.indexes / "source_cards.json",
            self.project.indexes / "source_card_vectors.json",
        )

    def test_missing_cards_created_before_embedding_and_saved_separately(self):
        events = []
        cards, vectors = self._run(progress=lambda n, t: events.append(
            (n, t)
        ))
        self.assertEqual(events, [(1, 2), (2, 2)])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.events, [
            "created", "generated", "generated", "closed"
        ])
        self.assertEqual(len(cards.records), 2)
        self.assertEqual(len(vectors.records), 2)
        self.assertEqual(len(vectors.records[0].embeddings), 3)
        self.assertEqual(set(record.result.status for record in
                             cards.records), {CARD_STATUS_VALID})
        self.assertEqual(load_source_cards(self._files()[0]), cards)
        self.assertEqual(load_source_card_vectors(
            self._files()[1], cards
        ), vectors)
        self.assertFalse((self.project.indexes / "paper_cards.json").exists())

    def test_identical_sources_reuse_without_model_or_embedding_calls(self):
        old_cards, old_vectors = self._run()
        self.calls.clear()
        self.events.clear()
        self.embed.embed_documents.reset_mock()
        cards, vectors = self._run()
        self.assertEqual((cards, vectors), (old_cards, old_vectors))
        self.assertFalse(self.calls)
        self.assertFalse(self.events)
        self.embed.embed_documents.assert_not_called()

    def test_modified_source_regenerates_only_its_card(self):
        self._run()
        self.calls.clear()
        self.hashes["a.pdf"] = sha256(b"changed").hexdigest()
        self.embed.embed_documents.reset_mock()
        cards, vectors = self._run()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(cards.records), 2)
        self.assertEqual(len(vectors.records), 2)
        self.embed.embed_documents.assert_called_once()

    def test_embedding_model_change_does_not_regenerate_cards(self):
        self._run()
        self.calls.clear()
        self.embed._retrieval_model_identity = "embed:v2"
        self.embed.embed_documents.reset_mock()
        self._run()
        self.assertFalse(self.calls)
        self.embed.embed_documents.assert_called_once()

    def test_corrupt_vectors_reembed_without_regenerating_cards(self):
        self._run()
        self._files()[1].write_text("{broken", encoding="utf-8")
        self.calls.clear()
        self.embed.embed_documents.reset_mock()
        cards, vectors = self._run()
        self.assertFalse(self.calls)
        self.assertEqual(len(vectors.records), 2)
        self.assertEqual(load_source_cards(self._files()[0]), cards)
        self.embed.embed_documents.assert_called_once()

    def test_corrupt_card_content_rebuilds_without_legacy_migration(self):
        self._run()
        self._files()[0].write_text("{broken", encoding="utf-8")
        self.calls.clear()
        cards, vectors = self._run()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(cards.records), len(vectors.records))

    def test_failed_generation_is_cached_and_retried_after_source_change(self):
        self.response = RuntimeError("model failed")
        cards, vectors = self._run()
        self.assertTrue(all(item.result.status == CARD_STATUS_FAILED
                            for item in cards.records))
        self.assertFalse(vectors.records)
        self.calls.clear()
        self._run()
        self.assertFalse(self.calls)
        self.hashes["a.pdf"] = sha256(b"changed").hexdigest()
        self.response = _answer()
        cards, vectors = self._run()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(vectors.records), 1)

    def test_nonpaper_verdict_is_cached_without_card_vectors(self):
        self.response = json.dumps({
            "applicable": False, "reason": "Administrative document",
        })
        cards, vectors = self._run()
        self.assertTrue(all(item.result.status == CARD_STATUS_NOT_APPLICABLE
                            for item in cards.records))
        self.assertFalse(vectors.records)
        self.calls.clear()
        self._run()
        self.assertFalse(self.calls)

    def test_chunkless_sources_are_excluded_without_generating(self):
        self.chunks = []
        cards, vectors = self._run()
        self.assertEqual(len(cards.records), 2)
        self.assertTrue(all(item.result.status == CARD_STATUS_NOT_APPLICABLE
                            for item in cards.records))
        self.assertFalse(self.calls)
        self.assertFalse(vectors.records)

    def test_source_override_disables_one_source(self):
        write_source_card_overrides(self.root, {
            Path("b.pdf"): SourceCardOverride(enabled=False)
        })
        cards, vectors = self._run()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([item.result.path for item in cards.records],
                         [Path("a.pdf")])
        self.assertEqual(len(vectors.records), 1)

    def test_master_switch_does_not_load_or_use_cached_cards(self):
        self._run()
        write_source_card_policy(
            self.root, SourceCardPolicy(False, "scientific_paper", True)
        )
        self.calls.clear()
        self.embed.embed_documents.reset_mock()
        cards, vectors = self._run()
        self.assertFalse(cards.records)
        self.assertIsNone(vectors)
        self.assertFalse(self.calls)
        self.embed.embed_documents.assert_not_called()
        self.assertTrue(self._files()[0].is_file())

    def test_disabled_automatic_generation_keeps_valid_cached_cards(self):
        self._run()
        write_source_card_policy(
            self.root, SourceCardPolicy(True, "scientific_paper", False)
        )
        self.calls.clear()
        self.hashes["a.pdf"] = sha256(b"new").hexdigest()
        cards, vectors = self._run()
        self.assertFalse(self.calls)
        self.assertEqual(len(cards.records), 1)
        self.assertEqual(len(vectors.records), 1)

    def test_wrong_current_vector_dimension_rejects_only_embedding(self):
        self.embed.embed_documents.side_effect = lambda texts: [
            [1.0, 0.0, 0.0] for text in texts
        ]
        cards, vectors = self._run(expected_dimension=2)
        self.assertEqual(len(cards.records), 2)
        self.assertFalse(vectors.records)
        self.assertTrue(self._files()[0].is_file())
        self.embed.embed_documents.side_effect = lambda texts: [
            [1.0, 0.0] for text in texts
        ]
        self.calls.clear()
        _, vectors = self._run()
        self.assertFalse(self.calls)
        self.assertEqual(len(vectors.records), 2)

    def test_generation_guard_only_when_pending_and_releases_on_error(self):
        events = []

        @contextmanager
        def guard():
            events.append("released")
            try:
                yield
            finally:
                events.append("restored")

        def factory(settings, **_kwargs):
            events.append("generator_created")
            return _Generator(
                settings, calls=self.calls, events=events,
                response=self.response,
            )

        args = (
            self.project, self.chunks, self._manifest(), self.embed,
        )
        options = {
            "backend": "torch", "generation_guard": guard,
            "generator_factory": factory,
        }
        reconcile_source_cards(*args, **options)
        self.assertEqual(events[0:3], [
            "released", "generator_created", "created",
        ])
        self.assertEqual(events[-2:], ["closed", "restored"])
        events.clear()
        reconcile_source_cards(*args, **options)
        self.assertFalse(events)
        self.response = RuntimeError("generation failed")
        reconcile_source_cards(*args, force_regenerate=True, **options)
        self.assertEqual(events[-1], "restored")

    def test_manual_regeneration_ignores_valid_and_automatic_caches(self):
        self._run()
        self.calls.clear()
        write_source_card_policy(
            self.root, SourceCardPolicy(True, "scientific_paper", False)
        )
        self.embed.embed_documents.reset_mock()
        calls = self.calls

        def factory(settings, **_kwargs):
            return _Generator(
                settings, calls=calls, events=self.events,
                response=_answer(),
            )

        cards, vectors = reconcile_source_cards(
            self.project, self.chunks, self._manifest(), self.embed,
            backend="torch", force_regenerate=True,
            generator_factory=factory,
        )
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(cards.records), 2)
        self.assertEqual(len(vectors.records), 2)
        self.embed.embed_documents.assert_not_called()

    def test_unavailable_generation_model_records_failures_not_crash(self):
        events = []

        @contextmanager
        def guard():
            events.append("released")
            try:
                yield
            finally:
                events.append("restored")

        def factory(*_args, **_kwargs):
            raise OSError("model missing")

        cards, vectors = reconcile_source_cards(
            self.project, self.chunks, self._manifest(), self.embed,
            backend="torch", generator_factory=factory,
            generation_guard=guard,
        )
        self.assertEqual(events, ["released", "restored"])
        self.assertEqual(len(cards.records), 2)
        self.assertTrue(all(
            record.result.status == CARD_STATUS_FAILED
            for record in cards.records
        ))
        self.assertFalse(vectors.records)
        self.assertTrue(self._files()[0].exists())


if __name__ == "__main__":
    unittest.main()
