"""Check the GUI engine boundary without loading model weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from qachat.gui.engine import ChatEngine, _reference_context
from qachat.gui.model_catalog import MODEL_CHOICES
from qachat.project import open_project
from qachat.rag import Evidence


@dataclass
class SavedTurn:
    """Small conversation record for the engine test."""

    question: str
    answer: str
    evidence: tuple[()] = ()
    parameters: dict[str, object] = field(default_factory=dict)


class TestChatEngine(unittest.TestCase):
    """The child process delegates each operation to QAChat."""

    @patch("qachat.gui.engine.download_model")
    def test_download_requires_an_explicit_suggested_model(
        self, download
    ) -> None:
        events: list[dict[str, object]] = []
        engine = ChatEngine(Path("/sandbox"), events.append)

        engine.download(MODEL_CHOICES[0].model_id)

        download.assert_called_once_with(MODEL_CHOICES[0].model_id)
        self.assertEqual(events[0]["event"], "model_download_started")
        self.assertEqual(events[1]["event"], "model_downloaded")

        with patch("qachat.gui.engine.traceback.print_exc"):
            engine.download("someone/unknown-model")
        download.assert_called_once()
        self.assertEqual(events[-1]["event"], "error")

    @patch("qachat.gui.engine.RAGSession")
    @patch("qachat.gui.engine.prepare_sources_workspace")
    def test_open_ask_and_reset(self, prepare, rag_session) -> None:
        prepare.return_value.root = Path("/sandbox/example")
        prepare.return_value.source = Path("/sandbox/example/repository")
        session = MagicMock()
        session.index_status = "reused"
        session.turns = [SavedTurn("Old question", "Old answer")]
        rag_session.return_value = session
        events: list[dict[str, object]] = []
        engine = ChatEngine(Path("/sandbox"), events.append)

        engine.open_project(["/original"], "code", "/sandbox/example")
        session.turns.append(SavedTurn("New question", "New answer"))
        engine.ask("New question")
        engine.reset()
        engine.close()

        prepare.assert_called_once_with(
            ["/original"], "code", "/sandbox/example"
        )
        session.ask.assert_called_once_with("New question")
        session.reset.assert_called_once_with()
        session.close.assert_called_once_with()
        self.assertEqual(events[1]["event"], "opened")
        self.assertEqual(events[2]["event"], "status")
        self.assertEqual(events[3]["turn"]["answer"], "New answer")
        self.assertEqual(events[-1]["event"], "reset")

    @patch("qachat.gui.engine.RAGSession")
    @patch("qachat.gui.engine.prepare_sources_workspace")
    def test_blank_source_uses_bundled_example(
        self, prepare, rag_session
    ) -> None:
        prepare.return_value.root = Path("/sandbox/example")
        prepare.return_value.source = Path("/sandbox/example/documents")
        rag_session.return_value.turns = []
        rag_session.return_value.index_status = "rebuilt"
        engine = ChatEngine(Path("/project/sandbox"), lambda event: None)

        engine.open_project([], "literature", "", selected_files={})

        source = (
            Path(__file__).resolve().parents[2]
            / "example_repos"
            / "literature_repo"
        )
        digest = sha256(str(source).encode()).hexdigest()[:8]
        prepare.assert_called_once_with(
            [str(source)],
            "literature",
            str(Path("/project/sandbox") / f"literature_repo-{digest}"),
        )

    def test_preview_lists_eligible_files(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "keep.py").write_text("pass\n")
            (root / "paper.pdf").write_bytes(b"pdf")
            (root / "skip.png").write_bytes(b"image")
            events: list[dict[str, object]] = []
            engine = ChatEngine(root / "sandbox", events.append)

            engine.preview([str(root)], [], 1000)

            self.assertEqual(events[-1]["event"], "preview")
            self.assertEqual(
                events[-1]["roots"][0]["files"],
                ["keep.py", "paper.pdf"],
            )
            self.assertEqual(
                events[-1]["roots"][0]["skipped"][0]["path"],
                "skip.png",
            )

    @patch("qachat.gui.engine.RAGSession")
    def test_checked_types_override_profile_extensions(
        self, rag_session
    ) -> None:
        """Only checked files are copied, even from a mixed source folder."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "research"
            source.mkdir()
            (source / "code.py").write_text("pass\n")
            paper = source / "paper.pdf"
            paper.write_bytes(b"%PDF-1.4\n" + b"x" * 2048)
            (source / "page.html").write_text("<p>Notes</p>\n")
            session = rag_session.return_value
            session.turns = []
            session.index_status = "rebuilt"
            events: list[dict[str, object]] = []
            engine = ChatEngine(root / "sandbox", events.append)

            engine.preview([str(source)], [".html"], 1024)
            self.assertEqual(
                events[-1]["roots"][0]["files"],
                ["code.py", "page.html"],
            )
            self.assertEqual(
                events[-1]["roots"][0]["skipped"][0]["size_bytes"],
                paper.stat().st_size,
            )

            engine.open_project(
                [str(source)],
                "code",
                str(root / "workspace"),
                selected_files={str(source): ["page.html", "paper.pdf"]},
                settings={
                    "extensions": [".py", ".html"],
                    "max_file_bytes": paper.stat().st_size,
                },
            )

            project = root / "workspace"
            self.assertTrue((project / "sources/research/page.html").is_file())
            self.assertTrue((project / "sources/research/paper.pdf").is_file())
            self.assertFalse((project / "sources/research/code.py").exists())
            profile = open_project(project).profile
            self.assertIn(".pdf", profile.extensions)
            self.assertEqual(profile.max_file_bytes, paper.stat().st_size)

    def test_preview_explains_pdf_size_limit(self) -> None:
        """A PDF is selectable when it fits the project's file limit."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.4\n" + b"x" * (2 * 1024 * 1024))
            events: list[dict[str, object]] = []
            engine = ChatEngine(root / "sandbox", events.append)

            engine.preview([str(pdf)], [".pdf"], 1024 * 1024)
            skipped = events[-1]["roots"][0]["skipped"]
            self.assertEqual(
                skipped,
                [
                    {
                        "path": "paper.pdf",
                        "reason": "larger than the file-size limit",
                        "size_bytes": pdf.stat().st_size,
                    }
                ],
            )

            engine.preview([str(pdf)], [".pdf"], 20 * 1024 * 1024)
            self.assertEqual(events[-1]["roots"][0]["files"], ["paper.pdf"])
            self.assertEqual(events[-1]["roots"][0]["skipped"], [])

    def test_preview_excludes_protected_direct_file(self) -> None:
        """Adding an explicit path cannot bypass protected-file rules."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret = root / ".env"
            secret.write_text("TOKEN=not-real\n")
            events: list[dict[str, object]] = []

            ChatEngine(root / "sandbox", events.append).preview(
                [str(secret)], [".env"], 1024
            )

            self.assertEqual(events[-1]["roots"][0]["files"], [])
            self.assertEqual(
                events[-1]["roots"][0]["skipped"][0]["reason"],
                "potentially secret or generated file",
            )

    def test_evidence_count_is_passed_for_one_question(self) -> None:
        events: list[dict[str, object]] = []
        engine = ChatEngine(Path("/sandbox"), events.append)
        session = MagicMock()
        session.turns = [SavedTurn("Question", "Answer")]
        engine.session = session

        engine.ask("Question", retrieval_count=8)

        session.ask.assert_called_once_with("Question", retrieval_count=8)
        self.assertEqual(events[-1]["event"], "answered")

    def test_generation_settings_are_passed_with_the_question(self) -> None:
        """Chat controls affect answers without editing the project profile."""

        events: list[dict[str, object]] = []
        engine = ChatEngine(Path("/sandbox"), events.append)
        session = MagicMock()
        session.turns = [SavedTurn("Question", "Answer")]
        engine.session = session
        settings = {
            "temperature": 0.4,
            "thinking": True,
            "max_new_tokens": 128,
        }

        engine.ask("Question", 6, settings)

        session.ask.assert_called_once_with(
            "Question", retrieval_count=6, generation_settings=settings
        )

    def test_startup_finds_recent_saved_project(self) -> None:
        """Only saved workspaces are candidates for automatic reopening."""

        with TemporaryDirectory() as temporary:
            sandbox = Path(temporary)
            for index, (name, kind) in enumerate(
                (("older", "code"), ("newer", "literature"))
            ):
                folder = sandbox / name
                (folder / "sources").mkdir(parents=True)
                profile = folder / "profile.ini"
                profile.write_text(
                    f"[project]\nkind = {kind}\nsource = sources\n"
                )
                os.utime(profile, (100 + index, 100 + index))
            (sandbox / "unrelated").mkdir()
            events: list[dict[str, object]] = []
            engine = ChatEngine(sandbox, events.append)

            engine.startup()

            self.assertEqual(events[-1]["event"], "startup")
            self.assertEqual(events[-1]["project"], str(sandbox / "newer"))
            self.assertEqual(events[-1]["kind"], "literature")

    def test_saved_project_tracks_type_and_deleted_folder(self) -> None:
        """Resume follows the selected type and a removed folder disappears."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sandbox = root / "sandbox"
            for name, kind in (
                ("code-project", "code"),
                ("papers", "literature"),
            ):
                folder = sandbox / name
                (folder / "sources").mkdir(parents=True)
                (folder / "profile.ini").write_text(
                    f"[project]\nkind = {kind}\nsource = sources\n"
                )
            events: list[dict[str, object]] = []
            engine = ChatEngine(sandbox, events.append)

            engine.find_saved_project("code", "")
            self.assertEqual(
                events[-1]["project"], str(sandbox / "code-project")
            )
            engine.find_saved_project("literature", "")
            self.assertEqual(events[-1]["project"], str(sandbox / "papers"))

            (sandbox / "papers").rename(root / "removed-papers")
            engine.find_saved_project("literature", "")
            self.assertIsNone(events[-1]["project"])

    def test_saved_projects_are_newest_first(self) -> None:
        """The project-list refresh shares startup's discovery rules."""

        with TemporaryDirectory() as temporary:
            sandbox = Path(temporary)
            for index, (name, kind) in enumerate(
                (("older", "code"), ("newer", "literature"))
            ):
                folder = sandbox / name
                (folder / "sources").mkdir(parents=True)
                profile = folder / "profile.ini"
                profile.write_text(
                    f"[project]\nkind = {kind}\nsource = sources\n"
                )
                os.utime(profile, (100 + index, 100 + index))
            events: list[dict[str, object]] = []
            engine = ChatEngine(sandbox, events.append)

            engine.list_saved_projects("")

            self.assertEqual(events[-1]["event"], "projects")
            self.assertEqual(
                [project["name"] for project in events[-1]["projects"]],
                ["newer", "older"],
            )

    def test_saved_project_checks_custom_workspace(self) -> None:
        """A chosen workspace takes precedence over sandbox suggestions."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            custom = root / "my-project"
            (custom / "sources").mkdir(parents=True)
            (custom / "profile.ini").write_text(
                "[project]\nkind = literature\nsource = sources\n"
            )
            events: list[dict[str, object]] = []
            engine = ChatEngine(root / "sandbox", events.append)

            engine.find_saved_project("literature", str(custom))
            self.assertEqual(events[-1]["project"], str(custom))
            engine.find_saved_project("code", str(custom))
            self.assertIsNone(events[-1]["project"])

            custom.rename(root / "moved-project")
            engine.find_saved_project("literature", str(custom))
            self.assertIsNone(events[-1]["project"])

    def test_empty_sandbox_starts_without_workspace(self) -> None:
        """An empty sandbox does not show an empty workspace sidebar."""

        with TemporaryDirectory() as temporary:
            events: list[dict[str, object]] = []
            ChatEngine(Path(temporary), events.append).startup()

        self.assertEqual(events[-1], {"event": "startup", "project": None})

    @patch("qachat.gui.engine.RAGSession")
    @patch("qachat.gui.engine.open_project")
    def test_resume_does_not_copy_sources(
        self, open_saved, rag_session
    ) -> None:
        """Startup restores the existing project without source preparation."""

        project = open_saved.return_value
        project.root = Path("/sandbox/saved")
        project.source = project.root / "sources"
        project.profile.kind = "code"
        rag_session.return_value.turns = []
        rag_session.return_value.index_status = "reused"
        events: list[dict[str, object]] = []
        engine = ChatEngine(Path("/sandbox"), events.append)

        engine.resume_project(str(project.root))

        open_saved.assert_called_once_with(str(project.root))
        rag_session.assert_called_once()
        self.assertEqual(events[-1]["event"], "opened")


class TestReferencePreview(unittest.TestCase):
    """Source previews use saved evidence and show nearby lines."""

    def test_text_preview_highlights_only_the_cited_lines(self) -> None:
        preview = _reference_context(
            "first\nbefore\nmemory\ncapacity\nafter\n",
            "memory\ncapacity",
            3,
            4,
            False,
        )

        self.assertEqual(preview["start_line"], 3)
        self.assertEqual(preview["end_line"], 4)
        self.assertEqual(
            [
                line["number"]
                for line in preview["lines"]
                if line["highlighted"]
            ],
            [3, 4],
        )
        self.assertEqual(preview["lines"][1]["text"], "before")

    def test_pdf_preview_finds_the_matching_page(self) -> None:
        preview = _reference_context(
            "First page\n\fSecond page\nmemory\ncapacity\nlast\n",
            "memory\ncapacity",
            2,
            3,
            True,
        )

        self.assertEqual(preview["page"], 2)
        self.assertEqual(
            [
                line["number"]
                for line in preview["lines"]
                if line["highlighted"]
            ],
            [2, 3],
        )

    def test_engine_opens_only_saved_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary)
            file = source / "notes.md"
            file.write_text("before\nmemory\ncapacity\nafter\n")
            evidence = Evidence(
                1,
                Path("notes.md"),
                2,
                3,
                None,
                "memory\ncapacity",
            )
            events: list[dict[str, object]] = []
            engine = ChatEngine(source, events.append)
            engine.session = SimpleNamespace(
                project=SimpleNamespace(source=source),
                turns=[SimpleNamespace(evidence=(evidence,))],
            )

            engine.preview_reference(0, 1)
            self.assertEqual(events[-1]["event"], "reference_preview")
            self.assertEqual(events[-1]["path"], "notes.md")
            engine.preview_reference(0, 2)
            self.assertEqual(events[-1]["event"], "reference_error")

            file.unlink()
            engine.preview_reference(0, 1)
            self.assertIn("saved passage", events[-1]["note"])
            self.assertTrue(
                all(line["highlighted"] for line in events[-1]["lines"])
            )


if __name__ == "__main__":
    unittest.main()
