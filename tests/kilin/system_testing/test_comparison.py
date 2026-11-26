"""Exercise independent comparison using ordinary recorded workspaces."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from copy import deepcopy
import json
import re
import unittest
from unittest.mock import Mock, patch

from kilin import project, rag
from kilin.answering import RouteDecision
from kilin.system_testing import Experiment, ReplayError
from kilin.system_testing._comparison_alignment import align
from tests.kilin._rag_support import _answer_patches, _intent
from tests.kilin.system_testing import test_experiment as experiment_tests
from tests.kilin.system_testing import test_replay as replay_tests


class TestComparison(unittest.TestCase):
    def setUp(self):
        fixture = replay_tests.TestReplay()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        self.fixture = fixture
        self.experiment = fixture.experiment
        self.engine = fixture.engine
        self.target = fixture.target

    def _pair(self, *, lifecycle=False):
        if lifecycle:
            self.fixture.fixture.test_complete_lifecycle_and_interactions()
        else:
            self.fixture.fixture._ask("A question")
            self.engine.close()
        with self.fixture._model():
            return self.experiment.replay()

    def test_independent_lifecycle_comparison_and_read_only_reports(self):
        replay = self._pair(lifecycle=True)
        original = self.experiment.workspace
        before = experiment_tests._snapshot(original)
        after = experiment_tests._snapshot(self.target)
        with patch.object(rag.RAGSession, "ask", side_effect=AssertionError):
            comparison = Experiment(original).compare()
        data = comparison.data
        self.assertTrue(comparison.complete, data["alignment"])
        self.assertEqual(data["alignment"]["conversation_ids"],
                         replay.conversation_ids)
        self.assertEqual(data["alignment"]["turn_uuids"], replay.turn_uuids)
        self.assertEqual(data["summary"]["questions"], 1)
        self.assertEqual(data["summary"]["token_f1"]["mean"], 0.5)
        self.assertEqual(data["summary"]["exact_answers"], 0)
        self.assertEqual(data["summary"]["evidence_jaccard"]["available"], 0)
        self.assertEqual(data["summary"]["embedding_cosine"]["available"], 0)
        self.assertTrue(any(c["archived"] for c in
                            data["original"]["conversations"]))
        question = next(op["question"] for op in data["operations"]
                        if op["kind"] == "ask")
        self.assertEqual(question["answer"]["original"], "Tool answer")
        self.assertEqual(question["answer"]["replayed"], "Replayed answer")
        self.assertNotEqual(question["interactions"]["original"],
                            question["interactions"]["replayed"])
        paths = comparison.save()
        self.assertEqual(paths[0].parent, original.parent)
        self.assertEqual(json.loads(paths[0].read_text(encoding="utf-8")),
                         data)
        self.assertIn("Matched committed questions: 1",
                      paths[1].read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            comparison.save()
        self.assertEqual(comparison.save(overwrite=True), paths)
        self.assertEqual(Experiment(original).compare().data, data)
        self.assertEqual(before, experiment_tests._snapshot(original))
        self.assertEqual(after, experiment_tests._snapshot(self.target))

    def test_one_shared_embedding_model_and_provenance(self):
        self._pair(lifecycle=True)
        session = Mock()
        session.backend = "mlx"
        session.device = "gpu"
        session.model_source = "/cached/model/snapshot"
        session.embed_documents.return_value = [[1., 0.], [0.6, 0.8]]
        with patch("kilin.embeddings.EmbeddingSession",
                   return_value=session) as loader:
            result = self.experiment.compare(
                embedding_model="example/embedding", embedding_backend="mlx",
                embedding_revision="revision",
            )
        loader.assert_called_once_with(
            "example/embedding", backend="mlx", revision="revision",
            allow_download=False,
        )
        session.embed_documents.assert_called_once_with([
            "Tool answer", "Replayed answer"
        ])
        session.close.assert_called_once_with()
        self.assertAlmostEqual(result.data["summary"]["embedding_cosine"][
            "mean"], 0.6)
        self.assertEqual(result.data["embedding_evaluation"]["model_source"],
                         "/cached/model/snapshot")

    def test_missing_evaluation_model_does_not_invent_similarity(self):
        self._pair()
        with patch("kilin.embeddings.EmbeddingSession",
                   side_effect=OSError("Not cached")) as loader:
            result = self.experiment.compare(embedding_model="missing/model")
        loader.assert_called_once()
        self.assertTrue(result.complete)
        self.assertEqual(result.data["embedding_evaluation"]["status"],
                         "failed")
        self.assertIsNone(result.data["summary"]["embedding_cosine"]["mean"])
        self.assertIn("Not cached", result.markdown())

    def test_failed_partial_replay_has_diagnostics_not_fabricated_pairs(self):
        self.fixture.fixture._ask("First")
        self.fixture.fixture._ask("Second")
        self.engine.close()
        with self.fixture._model("extra"), self.assertRaises(ReplayError):
            self.experiment.replay()
        with patch("kilin.embeddings.EmbeddingSession") as loader:
            result = self.experiment.compare(embedding_model="missing/model")
        loader.assert_not_called()
        self.assertFalse(result.complete)
        self.assertEqual(result.data["alignment"]["status"],
                         "invalid_recording")
        self.assertEqual(result.data["alignment"]["matched_operations"], 0)
        self.assertTrue(result.data["replayed"]["validation_issues"])
        self.assertEqual(result.data["summary"]["questions"], 0)

    def test_missing_replay_is_reported_without_creating_it(self):
        self.engine.close()
        result = self.experiment.compare()
        self.assertFalse(result.complete)
        self.assertFalse(self.target.exists())
        self.assertFalse(result.data["replayed"]["valid"])
        self.assertIsNone(result.data["files"]["sources"]["changes"])

    def test_valid_but_different_questions_only_align_common_prefix(self):
        self.fixture.fixture._ask("First")
        self.engine.close()
        self.engine.open_project(
            [str(self.fixture.fixture.root / "source.txt")],
            "writing", str(self.target),
        )
        self.fixture.fixture._ask("Different question")
        self.engine.close()
        result = self.experiment.compare()
        self.assertTrue(result.data["original"]["valid"])
        self.assertTrue(result.data["replayed"]["valid"])
        self.assertFalse(result.complete)
        self.assertEqual(result.data["alignment"]["status"], "diverged")
        self.assertEqual(result.data["alignment"]["matched_operations"], 1)
        self.assertEqual(result.data["summary"]["questions"], 0)
        self.assertEqual(result.data["alignment"]["turn_uuids"], {})

    def test_reclassified_explicit_setting_remains_comparable(self):
        self.engine.close()
        root = self.fixture.fixture.root / "settings_original"
        self.engine.settings("writing", "")
        self.engine.open_project([], "writing", str(root), settings={
            "temperature": 0.13, "_gui_full_snapshot": True,
        })
        self.fixture.fixture._ask("Question")
        self.engine.close()
        original = Experiment(root)
        old_template = project._default_profile_text

        def current_template(*args, **kwargs):
            return re.sub(r"(?m)^temperature = .*?$", "temperature = 0.13",
                          old_template(*args, **kwargs))

        with patch.object(project, "_default_profile_text",
                          side_effect=current_template), self.fixture._model():
            result = original.replay()
        first = original.timeline()[0].request["settings"]
        second = result.validation.operations[0].request["settings"]
        self.assertNotEqual(first["explicit_settings"],
                            second["explicit_settings"])
        comparison = original.compare()
        self.assertTrue(comparison.complete, comparison.data["alignment"])
        self.assertNotEqual(comparison.data["operations"][0][
            "settings_provenance"]["original"]["explicit_settings"],
            comparison.data["operations"][0]["settings_provenance"][
                "replayed"]["explicit_settings"])

    def test_output_byte_differences_are_not_quality_verdicts(self):
        self._pair()
        original = self.experiment.workspace
        for root, content in ((original, b"original"),
                              (self.target, b"changed")):
            (root / "outputs" / "report.txt").write_bytes(content)
        (self.target / "outputs" / "new.txt").write_bytes(b"new")
        (original / "outputs" / "gone.txt").write_bytes(b"old")
        comparison = self.experiment.compare()
        self.assertTrue(comparison.complete)
        changes = comparison.data["files"]["outputs"]["changes"]
        self.assertEqual(changes["added"], ["outputs/new.txt"])
        self.assertEqual(changes["removed"], ["outputs/gone.txt"])
        self.assertEqual(changes["changed"], ["outputs/report.txt"])

    def test_linked_report_and_workspace_paths_are_refused(self):
        self._pair()
        comparison = self.experiment.compare()
        root = self.experiment.workspace
        sibling = root.with_name("linked_replay")
        report = root.with_name(root.name + "_comparison.json")
        sentinel = root / "profile.ini"
        before = sentinel.read_bytes()
        try:
            sibling.symlink_to(self.target, target_is_directory=True)
            report.symlink_to(sentinel)
        except OSError as error:
            self.skipTest(f"Symbolic links unavailable: {error}")
        with self.assertRaises(ValueError):
            self.experiment.compare(sibling)
        with self.assertRaises(ValueError):
            self.experiment.compare(root)
        with self.assertRaises(ValueError):
            comparison.save(overwrite=True)
        self.assertEqual(sentinel.read_bytes(), before)

    def test_real_retrieval_and_evidence_comparison(self):
        real_ask, real_retrieve = rag.RAGSession.ask, rag.RAGSession._retrieve
        intent = _intent()
        decision = RouteDecision(intent.intent_id, True)

        def ask(session, question, **options):
            session.embedding_session.embed_query.return_value = [1., 0.]
            session.chat_session.reply.return_value = "Evidence [1]"
            with _answer_patches(session, intent, decision) as mocks:
                mocks.retrieve.side_effect = lambda *args, **kw: (
                    real_retrieve(session, *args, **kw)
                )
                mocks.save.side_effect = lambda **kw: (
                    rag.RAGSession._save_conversation(session, **kw)
                )
                with patch.object(rag, "summarize_answer",
                                  return_value="Summary"), patch.object(
                    session, "_format_query", side_effect=lambda text: text
                ):
                    return real_ask(session, question, **options)

        with patch.object(rag.RAGSession, "ask", ask):
            self.engine.ask("What is the evidence?")
            self.assertEqual(self.fixture.fixture.events[-1]["event"],
                             "answered", self.fixture.fixture.events[-1])
            self.engine.close()
            self.experiment.replay()
        result = self.experiment.compare()
        self.assertTrue(result.complete, result.data["alignment"])
        row = next(op for op in result.data["operations"]
                   if op["kind"] == "ask")
        self.assertEqual(row["question"]["evidence"]["overlap"]["jaccard"], 1.)
        self.assertEqual(row["question"]["evidence"]["citations"]["overlap"][
            "jaccard"], 1.)
        ranking = next(r for r in row["rankings"]
                       if r["event_type"] == "retrieval.semantic.ranked")
        self.assertEqual(ranking["overlap"]["jaccard"], 1.)

    def test_chronological_source_additions_and_branching_align(self):
        self.fixture.test_sources_are_imported_in_order_from_retained_copies()
        result = self.experiment.compare()
        self.assertTrue(result.complete, result.data["alignment"])
        self.assertEqual(result.data["summary"]["questions"], 3)
        self.assertEqual(result.data["files"]["sources"]["changes"][
            "changed"], [])
        self.assertEqual(len(result.data["alignment"]["turn_uuids"]), 3)

    def test_collision_adjusted_sources_and_partial_branch_align(self):
        self.fixture.test_collision_adjusted_names_and_partial_branch()
        result = self.experiment.compare()
        self.assertTrue(result.complete, result.data["alignment"])
        self.assertEqual(result.data["summary"]["questions"], 3)

    def test_empty_and_manual_card_commands_align(self):
        self.fixture.test_empty_creation_and_sync_noop()
        root = self.fixture.fixture.root / "blank_original"
        result = Experiment(root).compare()
        self.assertTrue(result.complete, result.data["alignment"])
        self.assertEqual(result.data["summary"]["questions"], 0)
        self.fixture.test_manual_regeneration_without_selected_cards()
        root = self.fixture.fixture.root / "cards_original"
        result = Experiment(root).compare()
        self.assertTrue(result.complete, result.data["alignment"])

    def test_changed_source_batches_are_not_matched_by_kind_alone(self):
        self._pair()
        before = self.experiment.timeline()
        after = deepcopy(Experiment(self.target).timeline())
        event = after[0].matching("index.check.completed")[0]
        event.details["source_manifest"]["files"]["source.txt"][
            "sha256"] = "a" * 64
        aligned = align(before, after)
        self.assertEqual(len(aligned.pairs), 0)
        self.assertIn("source effects", aligned.reason)


if __name__ == "__main__":
    unittest.main()
