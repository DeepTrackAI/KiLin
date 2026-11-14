"""Characterize the desktop GUI's JSON-line command and event protocol.

The Qt-free tests exercise the engine entry point with a stub coordinator,
then sample real lightweight event producers. No model is loaded and the
actual Qt window and JavaScript page are not started.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from kilin._model_completion import GenerationCancelled
from kilin.gui import engine
from kilin.rag import ANSWER_PHASE_CHOICES


class TestGUIProtocolCommands(unittest.TestCase):
    """Protect engine command framing, dispatch, and failure responses."""

    def _run(
        self, lines: list[str]
    ) -> tuple[MagicMock, list[dict[str, object]]]:
        """Execute the JSON-line loop without opening a real GUI.

        Parameters
        ----------
        lines: list[str]
            Physical input lines, including deliberately malformed JSON.

        Returns
        -------
        tuple[MagicMock, list[dict[str, object]]]
            Stubbed engine and the parsed event lines written to stdout.

        """

        worker = MagicMock()
        output = io.StringIO()
        with (
            patch.object(engine, "ChatEngine", return_value=worker),
            patch.object(
                engine,
                "default_workspace_directory",
                return_value=Path("/unused/qa-workspaces"),
            ),
            patch(
                "sys.stdin",
                io.StringIO("\n".join(lines) + ("\n" if lines else "")),
            ),
            patch("sys.stdout", output),
        ):
            engine.main()
        parsed = [json.loads(line) for line in output.getvalue().splitlines()]
        worker.close.assert_called()
        return worker, parsed

    def test_core_command_dispatch(self) -> None:
        """Map existing command names to the corresponding engine calls."""

        samples = [
            ({"action": "startup"}, "startup"),
            ({"action": "readiness"}, "readiness"),
            (
                {"action": "install_required_models"},
                "install_required_models",
            ),
            ({"action": "use_installed_models"}, "use_installed_models"),
            ({"action": "close"}, "close"),
            ({"action": "reset"}, "reset"),
            ({"action": "new_conversation"}, "create_conversation"),
            ({"action": "settings_info"}, "settings_info"),
            ({"action": "project_behavior"}, "behavior_settings"),
            ({"action": "tool_settings"}, "tool_settings"),
            (
                {"action": "save_tool_settings", "tool": "find_fact",
                 "settings": {"part_tokens": 8000}},
                "save_tool_settings",
            ),
            ({"action": "branch_conversation"}, "create_conversation"),
            (
                {"action": "resume", "project": "/project"},
                "resume_project",
            ),
            (
                {"action": "saved_projects", "workspace": "/parent"},
                "list_saved_projects",
            ),
            (
                {
                    "action": "saved_project",
                    "type_id": "code",
                    "workspace": "/parent",
                },
                "find_saved_project",
            ),
            (
                {
                    "action": "settings",
                    "type_id": "code",
                    "workspace": "",
                },
                "settings",
            ),
            (
                {"action": "apply_models", "settings": {"x": 1}},
                "apply_model_settings",
            ),
            (
                {"action": "download_model", "model_id": "a/model"},
                "download",
            ),
            (
                {
                    "action": "preview",
                    "sources": ["/inputs"],
                    "extensions": [".md"],
                    "max_file_bytes": 1200,
                },
                "preview",
            ),
            (
                {"action": "preview_reference", "turn_index": 0,
                 "citation": 1},
                "preview_reference",
            ),
            (
                {"action": "open", "sources": ["/inputs"],
                 "type_id": "code", "workspace": ""},
                "open_project",
            ),
            (
                {"action": "ask", "question": "Why?"},
                "ask",
            ),
            (
                {"action": "select_conversation", "conversation_id": "a"},
                "select_conversation",
            ),
            (
                {"action": "rename_conversation", "conversation_id": "a",
                 "name": "New name"},
                "rename_conversation",
            ),
            (
                {"action": "delete_conversation", "conversation_id": "a"},
                "delete_conversation",
            ),
            (
                {"action": "check_sources", "sources": ["/more"]},
                "check_sources",
            ),
            (
                {"action": "add_project_sources", "sources": ["/more"]},
                "add_project_sources",
            ),
            (
                {"action": "remove_project_sources", "paths": ["a.py"]},
                "remove_project_sources",
            ),
            (
                {"action": "sync_sources",
                 "decisions": {"sources/a.md": "replace"}},
                "sync_sources",
            ),
            (
                {"action": "edit_folder", "folder": "outputs",
                 "operation": "remove", "paths": ["report.md"]},
                "edit_folder",
            ),
            (
                {"action": "rename_project", "name": "Renamed"},
                "rename_project",
            ),
            (
                {"action": "delete_project", "project": "/parent/old"},
                "delete_project",
            ),
            (
                {"action": "apply_indexing", "settings": {"x": 1}},
                "apply_indexing_settings",
            ),
            (
                {"action": "save_source_cards", "settings": {"x": 1}},
                "save_source_card_settings",
            ),
            (
                {"action": "regenerate_source_cards"},
                "regenerate_source_cards",
            ),
        ]
        for command, method in samples:
            with self.subTest(action=command["action"]):
                worker, events = self._run([json.dumps(command)])
                self.assertEqual(events, [])
                expected = 2 if method == "close" else 1
                self.assertEqual(
                    getattr(worker, method).call_count, expected
                )

    def test_obsolete_timeline_writer_is_not_an_engine_command(
        self,
    ) -> None:
        """Do not silently write presentation timings to another file."""
        worker, events = self._run([
            '{"action":"save_answer_timeline","timestamp":"t"}'
        ])
        self.assertEqual(events, [
            {"event": "error", "message": "Unknown command."}
        ])
        self.assertFalse(hasattr(
            type(worker), "save_answer_timeline"
        ))

    def test_optional_fields_keep_their_existing_defaults(self) -> None:
        """Do not require optional answer and source-selection fields."""

        worker, _ = self._run([
            json.dumps({
                "action": "ask", "question": "Why?",
                "generation_settings": {"thinking": False},
            }),
            json.dumps({
                "action": "open", "sources": ["/a"],
                "type_id": "code", "workspace": "",
            }),
        ])
        worker.ask.assert_called_once_with(
            "Why?", None, {"thinking": False}, None
        )
        worker.open_project.assert_called_once_with(
            ["/a"], "code", "", None, None
        )

    def test_answer_settings_are_saved_on_their_own(self) -> None:
        """Forward Thinking from the menu without asking a question."""

        worker, _ = self._run([
            json.dumps({
                "action": "save_answer_settings",
                "settings": {"thinking": False},
            }),
        ])
        worker.save_answer_settings.assert_called_once_with(
            {"thinking": False}, None, None
        )
        worker.ask.assert_not_called()

    def test_resume_forwards_the_requested_conversation(self) -> None:
        """Restore an explicit ID without changing ordinary project resume."""

        worker, events = self._run([
            json.dumps({
                "action": "resume", "project": "/project",
                "conversation_id": "conversation_earlier",
            }),
            json.dumps({"action": "resume", "project": "/project"}),
        ])
        self.assertEqual(events, [])
        self.assertEqual(worker.resume_project.call_args_list, [
            unittest.mock.call("/project", "conversation_earlier"),
            unittest.mock.call("/project", None),
        ])

    def test_stop_is_applied_when_it_arrives(self) -> None:
        """The input thread stops the answer; the loop then ignores it."""

        worker, events = self._run([
            json.dumps({"action": "cancel_answer"}),
            json.dumps({"action": "readiness"}),
        ])
        self.assertEqual(events, [])
        worker.cancel_answer.assert_called_once_with()
        worker.readiness.assert_called_once_with()

    def test_branch_forwards_the_number_of_turns_to_keep(self) -> None:
        """Branch from a message, or keep every turn as before."""

        worker, events = self._run([
            json.dumps({"action": "branch_conversation", "turn_count": 2}),
            json.dumps({"action": "branch_conversation"}),
            json.dumps({"action": "new_conversation"}),
        ])
        self.assertEqual(events, [])
        self.assertEqual(worker.create_conversation.call_args_list, [
            unittest.mock.call(branch=True, turn_count=2),
            unittest.mock.call(branch=True, turn_count=None),
            unittest.mock.call(),
        ])

    def test_sync_forwards_the_users_choices(self) -> None:
        """Pass per-file sync choices through; none means the defaults."""

        decisions = {"sources/a.md": "keep_both"}
        worker, events = self._run([
            json.dumps({"action": "sync_sources", "decisions": decisions}),
            json.dumps({"action": "sync_sources"}),
        ])
        self.assertEqual(events, [])
        self.assertEqual(worker.sync_sources.call_args_list, [
            unittest.mock.call(decisions),
            unittest.mock.call(None),
        ])

    def test_invalid_commands_emit_errors_and_processing_continues(
        self,
    ) -> None:
        """Malformed, non-object, and unknown commands have distinct errors."""

        worker, events = self._run([
            "not-json",
            json.dumps(["not an object"]),
            json.dumps({"action": "not_supported"}),
            json.dumps({"action": "readiness"}),
        ])
        self.assertEqual([event["event"] for event in events], [
            "error", "error", "error",
        ])
        self.assertIn("background task failed", events[0]["message"])
        self.assertIn("must be an object", events[1]["message"])
        self.assertEqual(events[2]["message"], "Unknown command.")
        worker.readiness.assert_called_once_with()
        self.assertEqual(worker.active_trace.exception.call_count, 2)

    def test_plan_choice_is_not_an_idle_action(self) -> None:
        """A late tool decision is ignored outside an active plan wait."""

        worker, events = self._run([
            json.dumps({"action": "plan_choice", "choice": "continue"}),
        ])
        self.assertEqual(events, [])
        self.assertEqual(worker.close.call_count, 1)
        self.assertEqual(
            [name for name, _, _ in worker.method_calls], ["close"]
        )

    def test_event_frames_are_separate_json_objects(self) -> None:
        """Keep an unredirected event sink and one newline per event."""

        worker = MagicMock()
        output = io.StringIO()
        factory = MagicMock(return_value=worker)

        def emit_events():
            """Send two events through the main-loop writer."""

            sink = factory.call_args.args[1]
            sink({"event": "status", "message": "working"})
            sink({"event": "answer_delta", "text": "result"})

        worker.readiness.side_effect = emit_events
        with (
            patch.object(engine, "ChatEngine", factory),
            patch.object(
                engine, "default_workspace_directory",
                return_value=Path("/unused/qa-workspaces"),
            ),
            patch("sys.stdin", io.StringIO('{"action":"readiness"}\n')),
            patch("sys.stdout", output),
        ):
            engine.main()
        self.assertEqual(output.getvalue(), (
            '{"event": "status", "message": "working"}\n'
            '{"event": "answer_delta", "text": "result"}\n'
        ))

    def test_missing_required_command_key_is_reported(self) -> None:
        """Missing dispatch keys do not break the next command frame."""

        worker, events = self._run([
            json.dumps({"action": "ask"}),
            json.dumps({"action": "readiness"}),
        ])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "error")
        self.assertIn("'question'", events[0]["message"])
        worker.readiness.assert_called_once_with()

    def test_tool_choice_is_consumed_during_answer(self) -> None:
        """Share one stdin iterator between answering and later commands."""

        worker = MagicMock()
        output = io.StringIO()
        captured: list[dict[str, object] | None] = []

        def make_worker(*_args, **kwargs):
            """Provide an answer operation that waits for one command."""

            def ask(*_args, **_kwargs):
                """Read the pending tool choice without a second iterator."""

                captured.append(kwargs["read_command"]())

            worker.ask.side_effect = ask
            return worker

        incoming = [
            json.dumps({"action": "ask", "question": "Question"}),
            "broken JSON",
            json.dumps({"action": "plan_choice", "choice": "quick"}),
            json.dumps({"action": "reset"}),
        ]
        with (
            patch.object(engine, "ChatEngine", side_effect=make_worker),
            patch.object(
                engine, "default_workspace_directory",
                return_value=Path("/unused/qa-workspaces"),
            ),
            patch("sys.stdin", io.StringIO("\n".join(incoming) + "\n")),
            patch("sys.stdout", output),
        ):
            engine.main()
        self.assertEqual(captured, [{
            "action": "plan_choice", "choice": "quick",
        }])
        self.assertEqual(
            [json.loads(line) for line in output.getvalue().splitlines()],
            [{"event": "error", "message": "Unreadable command."}],
        )
        worker.reset.assert_called_once_with()

    def test_correlated_answer_settings_command(self) -> None:
        """Forward a save ID and project without losing their identity."""

        worker, _ = self._run([json.dumps({
            "action": "save_answer_settings",
            "settings": {"thinking": True},
            "request_id": 17,
            "project": "/saved/project",
        })])
        worker.save_answer_settings.assert_called_once_with(
            {"thinking": True}, 17, "/saved/project"
        )

    def test_eof_closes_engine_once(self) -> None:
        """Release an idle coordinator when the command stream ends."""

        worker, events = self._run([])
        self.assertEqual(events, [])
        worker.close.assert_called_once_with()

    def test_backend_text_is_traced_not_sent_as_json(self) -> None:
        """Diagnostic stdout cannot contaminate the framed event channel."""

        worker = MagicMock()
        worker.readiness.side_effect = lambda: print("diagnostic text")
        output = io.StringIO()
        with (
            patch.object(engine, "ChatEngine", return_value=worker),
            patch.object(
                engine, "default_workspace_directory",
                return_value=Path("/unused/qa-workspaces"),
            ),
            patch("sys.stdin", io.StringIO('{"action":"readiness"}\n')),
            patch("sys.stdout", output),
        ):
            engine.main()
        self.assertEqual(output.getvalue(), "")
        worker.active_trace.emit.assert_called_once_with(
            "runtime.output", component="runtime",
            data={"stream": "backend.stdout", "message": "diagnostic text"},
        )


class TestGUIProtocolEvents(unittest.TestCase):
    """Check lightweight real events and shared phase identifiers."""

    def test_settings_include_canonical_selection_reason_codes(self) -> None:
        """Pass the packaged code vocabulary through the settings event."""

        from kilin.repository_reason_codes import SELECTION_REASON_CODES

        events: list[dict[str, object]] = []
        worker = engine.ChatEngine.__new__(engine.ChatEngine)
        worker._send = lambda event, **data: events.append(
            {"event": event, **data}
        )
        worker._report_error = MagicMock()
        with (
            patch.object(engine, "downloaded_models", return_value=[]),
            patch.object(engine, "installed_backends", return_value=[]),
            patch.object(engine, "choose_backend", return_value="mlx"),
        ):
            worker.settings("code", "")

        worker._report_error.assert_not_called()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "settings")
        self.assertEqual(
            events[0]["selection_reason_codes"],
            dict(SELECTION_REASON_CODES),
        )
        self.assertEqual(
            json.loads(json.dumps(events[0]))["selection_reason_codes"],
            dict(SELECTION_REASON_CODES),
        )

    def test_source_preview_and_error_payloads(self) -> None:
        """Keep preview roots, selectable files, and skipped reasons stable."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            small = root / "small.py"
            large = root / "large.py"
            small.write_text("x = 1\n", encoding="utf-8", newline="\n")
            large.write_text(
                "x = '" + "a" * 200 + "'\n",
                encoding="utf-8", newline="\n",
            )
            events: list[dict[str, object]] = []
            worker = engine.ChatEngine(root, events.append)
            try:
                worker.preview([str(small), str(large)], [], 40)
                worker.preview([str(root / "missing")], [], 40)
            finally:
                worker.close()
        self.assertEqual(events[0], {
            "event": "preview",
            "roots": [
                {"path": str(small), "files": ["small.py"],
                 "skipped": []},
                {"path": str(large), "files": [], "skipped": [{
                    "path": "large.py",
                    "reason": "larger than the file-size limit",
                    "reason_code": "file_too_large",
                    "size_bytes": len("x = '" + "a" * 200 + "'\n"),
                }]},
            ],
        })
        self.assertEqual(events[1]["event"], "error")
        self.assertIn("Could not inspect", events[1]["message"])

    def test_directory_preview_forwards_repository_reason_codes(self) -> None:
        """Forward core selection codes without classifying display text."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            sources = root / "sources"
            sources.mkdir()
            (sources / "ordinary.py").write_text(
                "x = 1\n", encoding="utf-8"
            )
            (sources / "oversized.py").write_text(
                "x = " + "a" * 100, encoding="utf-8"
            )
            (sources / "empty.py").touch()
            # A type outside the project's list cannot be ticked.
            (sources / "settings.json").write_text("{}", encoding="utf-8")
            (sources / ".env").write_text(
                "FAKE_SECRET=not-real", encoding="utf-8"
            )
            (sources / "weight.safetensors").write_text(
                "not a model", encoding="utf-8"
            )
            events: list[dict[str, object]] = []
            worker = engine.ChatEngine(root, events.append)
            try:
                worker.preview([str(sources)], [".py"], 40)
            finally:
                worker.close()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "preview")
        preview = events[0]["roots"][0]
        self.assertEqual(preview["files"], ["ordinary.py"])
        skipped = {row["path"]: row for row in preview["skipped"]}
        self.assertEqual(
            {path: row["reason_code"] for path, row in skipped.items()},
            {
                ".env": "protected_file",
                "empty.py": "empty_file",
                "oversized.py": "file_too_large",
                "settings.json": "file_type_not_selected",
                "weight.safetensors": "blocked_file_type",
            },
        )
        self.assertEqual(
            skipped["oversized.py"]["reason"],
            "larger than the file-size limit",
        )

    def test_direct_empty_file_reports_structured_reason(self) -> None:
        """Give directly chosen empty files a stable skip code."""

        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            empty = root / "empty.py"
            empty.touch()
            events: list[dict[str, object]] = []
            worker = engine.ChatEngine(root, events.append)
            try:
                worker.preview([str(empty)], [], 40)
            finally:
                worker.close()

        self.assertEqual(events[0]["roots"][0]["skipped"], [{
            "path": "empty.py",
            "reason": "empty file",
            "reason_code": "empty_file",
            "size_bytes": 0,
        }])

    def test_pending_tool_plan_rejects_unrelated_commands(self) -> None:
        """Consume unrelated commands during an interactive tool wait."""

        with TemporaryDirectory() as temporary:
            commands = iter([
                {"action": "readiness"},
                {"action": "plan_choice", "choice": "quick"},
            ])
            events: list[dict[str, object]] = []
            worker = engine.ChatEngine(
                Path(temporary), events.append,
                read_command=lambda: next(commands),
            )
            try:
                result = worker._confirm_plan(None)
            finally:
                worker.close()
        self.assertEqual(result, "quick")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "error")
        self.assertIn("Continue, Quick, or Cancel", events[0]["message"])
        self.assertIs(events[0]["pending_choice"], True)

    def test_stop_ends_a_wait_for_the_users_choice(self) -> None:
        """Stop while a plan or option is shown ends the answer."""

        with TemporaryDirectory() as temporary:
            events: list[dict[str, object]] = []
            commands = iter([{"action": "cancel_answer"}] * 2)
            worker = engine.ChatEngine(
                Path(temporary), events.append,
                read_command=lambda: next(commands),
            )
            try:
                with self.assertRaises(GenerationCancelled):
                    worker._confirm_plan(None)
                with self.assertRaises(GenerationCancelled):
                    worker._choose_option("Choose", ("one", "two"))
            finally:
                worker.close()
        self.assertEqual(events, [])

    def test_pending_tool_option_accepts_only_valid_indices(self) -> None:
        """Convert option choice strings only within their valid range."""

        with TemporaryDirectory() as temporary:
            events: list[dict[str, object]] = []
            commands = iter([
                {"action": "plan_choice", "choice": "option:1"},
                {"action": "plan_choice", "choice": "option:5"},
            ])
            worker = engine.ChatEngine(
                Path(temporary), events.append,
                read_command=lambda: next(commands),
            )
            try:
                self.assertEqual(
                    worker._choose_option("Choose", ("one", "two")), 1
                )
                self.assertIsNone(
                    worker._choose_option("Choose", ("one", "two"))
                )
            finally:
                worker.close()
        self.assertEqual(events, [])

    def test_pending_option_rejection_is_not_an_answer_failure(self):
        """Keep waiting after rejecting an unrelated command or at EOF."""

        with TemporaryDirectory() as temporary:
            for last, expected in (
                ({"action": "plan_choice", "choice": "option:0"}, 0),
                ({"action": "plan_choice", "choice": "cancel"}, None),
                (None, None),
            ):
                with self.subTest(last=last):
                    commands = iter([{"action": "readiness"}, last])
                    events = []
                    worker = engine.ChatEngine(
                        Path(temporary), events.append,
                        read_command=lambda: next(commands),
                    )
                    try:
                        result = worker._choose_option("Choose", ("one",))
                    finally:
                        worker.close()
                    self.assertEqual(result, expected)
                    self.assertEqual(len(events), 1)
                    self.assertEqual(events[0]["event"], "error")
                    self.assertIs(events[0]["pending_choice"], True)
                    self.assertIn("Answer the question", events[0]["message"])

    def test_python_and_browser_answer_phases_match(self) -> None:
        """Keep JavaScript's duplicated phase names in sync with Python."""

        script = (
            Path(engine.__file__).parent / "web" / "app.js"
        ).read_text(encoding="utf-8")
        names = dict(re.findall(
            r'const (ANSWER_PHASE_[A-Z_]+) = "([a-z_]+)";',
            script,
        ))
        self.assertEqual(set(names.values()), set(ANSWER_PHASE_CHOICES))
        self.assertEqual(
            names["ANSWER_PHASE_PROCESSING"], "processing"
        )
        self.assertIn(
            'const ENGINE_PHASE_LOADING_MODEL = "loading_model";',
            script,
        )
        self.assertEqual(engine.ENGINE_PHASE_LOADING_MODEL, "loading_model")

    def test_frontend_event_inventory(self) -> None:
        """Make changes to page-side event names an explicit review item."""

        script = (
            Path(engine.__file__).parent / "web" / "app.js"
        ).read_text(encoding="utf-8")
        handled = set(re.findall(r'data\.event === "([a-z_]+)"', script))
        self.assertEqual(handled, {
            "answer_activity", "answer_cancelled", "answer_delta",
            "answer_phase",
            "answer_settings_failed", "answer_settings_saved",
            "answered",
            "backend_failed", "backend_preference_failed",
            "backend_preference_saved", "backend_restarted",
            "conversation_delete_started",
            "conversation_opened", "conversation_viewed", "conversations",
            "error", "indexing_applied",
            "source_cards_applied", "source_cards_regenerated",
            "workspace_conflict",
            "folder_changed", "installation_complete",
            "installation_model_complete", "installation_progress",
            "installation_started", "installed_models_used",
            "model_download_progress", "model_download_started",
            "model_downloaded", "models_applied", "opened", "preview",
            "progress", "projects", "projects_folder", "readiness",
            "projects_folder_cancelled", "projects_folder_failed",
            "reference_error",
            "reference_preview", "reset", "settings", "source_changes",
            "sources_checked", "status", "stopped", "tool_settings",
            "settings_info", "settings_info_failed", "project_behavior",
            "terms_acceptance_failed", "terms_acceptance_saved",
            "trace_ready",
            "usage_progress",
        })


class TestGUIProtocolBridge(unittest.TestCase):
    """Exercise the actual Qt bridge when its dependencies are available."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import the Qt bridge without creating an application.

        Only direct QObject signals and slots are exercised here. A
        QCoreApplication would prevent a later PDF test from constructing
        the QApplication required for Qt font rendering.

        """

        try:
            cls.bridge_type = importlib.import_module(
                "kilin.gui.window"
            ).ChatBridge
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith("PySide6"):
                raise unittest.SkipTest("PySide6 is not installed") from error
            raise

    def setUp(self) -> None:
        """Record Qt signals without opening a native window."""

        self.bridge = self.bridge_type()
        self.commands: list[dict[str, object]] = []
        self.events: list[str] = []
        self.stops: list[bool] = []
        self.bridge.commandRequested.connect(self.commands.append)
        self.bridge.event.connect(self.events.append)
        self.bridge.stopRequested.connect(lambda: self.stops.append(True))

    def test_answer_and_resume_command_payloads(self) -> None:
        """Check the browser-facing arguments before engine dispatch."""

        self.bridge.askQuestion(
            "What?", "lagom", json.dumps({"temperature": 0.0})
        )
        self.bridge.resumeProjectConversation("/project", "conv-123")
        self.assertEqual(self.commands, [
            {
                "action": "ask", "question": "What?", "effort": "lagom",
                "generation_settings": {"temperature": 0.0},
            },
            {
                "action": "resume", "project": "/project",
                "conversation_id": "conv-123",
            },
        ])

    def test_invalid_open_sources_emit_local_error(self) -> None:
        """Reject non-list sources before they enter the child process."""

        self.bridge.openProject(
            json.dumps({"not": "a list"}), "code", "",
            "null", "{}",
        )
        self.assertEqual(self.commands, [])
        self.assertEqual(
            [json.loads(event) for event in self.events],
            [{"event": "error", "message": "Invalid sources."}],
        )

    def test_tool_settings_command_payloads(self) -> None:
        """Ask for the tools; save one tool's settings as an object."""

        self.bridge.requestToolSettings()
        self.bridge.saveToolSettings(
            "find_fact", json.dumps({"part_tokens": 8000})
        )
        self.assertEqual(self.commands, [
            {"action": "tool_settings"},
            {
                "action": "save_tool_settings",
                "tool": "find_fact",
                "settings": {"part_tokens": 8000},
            },
        ])

    def test_delete_project_sends_the_project_folder(self) -> None:
        self.bridge.deleteProject("/parent/old")
        self.assertEqual(self.commands, [
            {"action": "delete_project", "project": "/parent/old"},
        ])

    def test_stop_removes_the_project_being_prepared(self) -> None:
        """Remove only empty claims, not completed or partial projects."""

        cleanup = importlib.import_module(
            "kilin._project_creation"
        ).cleanup_stopped_creation
        with TemporaryDirectory() as folder:
            parent = Path(folder).resolve()
            empty = parent / "copying"
            empty.mkdir()
            (empty / ".kilin-creating").write_text(
                "pid=42\ncreated=1\n"
            )
            partial = parent / "partial"
            partial.mkdir()
            (partial / ".kilin-creating").write_text(
                "pid=42\ncreated=1\n"
            )
            (partial / "notes.txt").write_text("keep")
            other = parent / "other"
            other.mkdir()
            (other / ".kilin-creating").write_text(
                "pid=7\ncreated=1\n"
            )
            committed = parent / "indexing"
            committed.mkdir()
            (committed / "profile.ini").write_text("published")
            (committed / "notes.txt").write_text("user data")
            (parent / "link").symlink_to(committed)
            removed, preserved = cleanup(parent, committed, 42)
            self.assertEqual(removed, [str(empty)])
            self.assertEqual(
                preserved, [str(committed), str(partial)]
            )
            self.assertEqual(
                sorted(path.name for path in parent.iterdir()),
                ["indexing", "link", "other", "partial"],
            )
            self.assertEqual((committed / "notes.txt").read_text(),
                             "user data")
            self.assertEqual((partial / "notes.txt").read_text(), "keep")
            self.assertEqual(cleanup(parent, parent / "link", 0), ([], []))
            self.assertEqual(cleanup(parent, Path("/"), 0), ([], ["/"]))

    def test_stop_keeps_committed_project_before_queued_opened(self):
        """A delayed Qt event must not turn a published project into trash."""

        window = importlib.import_module("kilin.gui.window").ChatWindow
        with TemporaryDirectory() as folder:
            parent = Path(folder).resolve()
            project = parent / "finished"
            project.mkdir()
            (project / "profile.ini").write_text("published")
            (project / "notes.txt").write_text("important")
            events = []
            fake = MagicMock()
            fake._stdout = bytearray()
            fake._closing = False
            fake._switching_folder = False
            fake._stopping = True
            fake._unopened_project = project
            fake._stopped_pid = 42
            fake.process.readAllStandardOutput.return_value = (
                json.dumps({"event": "opened", "project": str(project)})
                + "\n"
            ).encode()
            fake.bridge.event.emit.side_effect = events.append
            fake._cleanup_stopped_creation.side_effect = lambda: (
                window._cleanup_stopped_creation(fake)
            )
            with patch(
                "kilin.gui.window.default_workspace_directory",
                return_value=parent,
            ):
                window._read_events(fake)
                self.assertEqual(fake._unopened_project, project)
                window._process_finished(fake)
            self.assertEqual((project / "notes.txt").read_text(),
                             "important")
            self.assertEqual(events, [json.dumps({
                "event": "stopped", "preserved_projects": [str(project)]
            })])
            fake._start_engine.assert_called_once_with()
            self.assertIsNone(fake._unopened_project)

    def test_backend_preference_is_saved_by_qt(self) -> None:
        """Refresh setup only after saving a valid preference."""

        with patch("kilin.gui.window.write_defaults") as write:
            self.bridge.setBackendPreference("torch")
            self.bridge.setBackendPreference("cuda")
        write.assert_called_once_with({"backend": "torch"})
        self.assertEqual(self.commands, [{"action": "readiness"}])
        self.assertEqual([
            json.loads(message) for message in self.events
        ], [{"event": "backend_preference_saved", "backend": "torch"}])

    def test_backend_preference_save_error_keeps_previous_status(
        self,
    ) -> None:
        """Never request readiness after a failed preferences write."""

        with patch(
            "kilin.gui.window.write_defaults",
            side_effect=OSError("read-only preferences"),
        ):
            self.bridge.setBackendPreference("torch")
        self.assertEqual(self.commands, [])
        self.assertEqual(len(self.events), 1)
        event = json.loads(self.events[0])
        self.assertEqual(event["event"], "backend_preference_failed")
        self.assertEqual(event["backend"], "torch")
        self.assertIn("read-only preferences", event["message"])

    def test_invalid_saved_preferences_report_failure(self) -> None:
        """Do not refresh readiness when preference validation fails."""

        with patch(
            "kilin.gui.window.write_defaults",
            side_effect=ValueError("invalid preferences.json"),
        ):
            self.bridge.setBackendPreference("mlx")
        self.assertEqual(self.commands, [])
        self.assertIn("invalid preferences.json", self.events[0])

    def test_indexing_settings_are_sent_as_an_object(self) -> None:
        self.bridge.applyIndexingSettings('{"target_tokens": 500}')
        self.assertEqual(self.commands, [
            {"action": "apply_indexing", "settings": {"target_tokens": 500}},
        ])

    def test_answer_stop_is_a_command_with_a_last_resort(self) -> None:
        """Stop an answer by command; the window may still restart."""

        cancels: list[bool] = []
        self.bridge.answerCancelRequested.connect(
            lambda: cancels.append(True)
        )
        self.bridge.cancelAnswer()
        self.assertEqual(self.commands, [{"action": "cancel_answer"}])
        self.assertEqual(cancels, [True])
        self.assertEqual(self.stops, [])

    def test_stop_is_a_signal_not_a_child_command(self) -> None:
        """Preserve out-of-band process termination via Qt signal."""

        self.bridge.stopOperation()
        self.assertEqual(self.stops, [True])
        self.assertEqual(self.commands, [])

    def test_branch_from_a_message_sends_the_turns_to_keep(self) -> None:
        """Each message's Branch from here names how many turns to keep."""

        self.bridge.branchConversation(2)
        self.assertEqual(self.commands, [
            {"action": "branch_conversation", "turn_count": 2},
        ])

    def test_tool_plan_choice_uses_special_action(self) -> None:
        """Carry plan decisions through the existing command envelope."""

        self.bridge.choosePlan("continue")
        self.assertEqual(self.commands, [
            {"action": "plan_choice", "choice": "continue"},
        ])


    def test_projects_folder_is_reported_and_chosen(self) -> None:
        """Report the folder in use; Use default saves the standard path."""

        chosen: list[str] = []
        self.bridge.projectsFolderChosen.connect(chosen.append)
        window = importlib.import_module("kilin.gui.window")
        with patch.object(
            window, "default_workspace_directory",
            return_value=Path("/sandbox"),
        ), patch.object(
            window, "standard_workspace_directory",
            return_value=Path("/sandbox"),
        ), patch.object(
            window, "saved_workspace_folder", return_value="/Gone/Projects",
        ), patch.object(
            window, "available_legacy_workspace_directory", return_value=None,
        ):
            self.bridge.requestProjectsFolder()
        self.assertEqual(json.loads(self.events[-1]), {
            "event": "projects_folder", "folder": "/sandbox",
            "standard": "/sandbox", "saved": "/Gone/Projects",
            "legacy": "", "missing": True,
        })
        with patch.object(
            window, "standard_workspace_directory",
            return_value=Path("/sandbox"),
        ):
            self.bridge.useDefaultProjectsFolder()
        self.assertEqual(chosen, [str(Path("/sandbox"))])
        self.assertEqual(self.commands, [])

    def test_previous_projects_choice_is_explicit_and_revalidated(
        self,
    ) -> None:
        """Only a real legacy candidate can be passed to folder selection."""

        window = importlib.import_module("kilin.gui.window")
        old = Path("/previous/kilin/workspaces")
        chosen: list[str] = []
        self.bridge.projectsFolderChosen.connect(chosen.append)
        with patch.object(
            window, "available_legacy_workspace_directory",
            return_value=old,
        ):
            self.bridge.requestProjectsFolder()
            self.bridge.useLegacyProjectsFolder()
        self.assertEqual(
            json.loads(self.events[-1])["legacy"], str(old)
        )
        self.assertEqual(chosen, [str(old)])
        self.assertEqual(self.commands, [])
        with patch.object(
            window, "available_legacy_workspace_directory",
            return_value=None,
        ):
            self.bridge.useLegacyProjectsFolder()
        self.assertEqual(chosen, [str(old)])
        self.assertEqual(
            json.loads(self.events[-1])["event"],
            "projects_folder_failed",
        )

    def test_cancelled_projects_folder_dialog_emits_its_own_event(
        self,
    ) -> None:
        """Cancelling Browse unblocks setup without changing any folder."""

        window = importlib.import_module("kilin.gui.window")
        with patch.object(window, "QFileDialog") as picker:
            picker.getExistingDirectory.return_value = ""
            self.bridge.chooseProjectsFolder()
        self.assertEqual(
            json.loads(self.events[-1]),
            {"event": "projects_folder_cancelled"},
        )
        self.assertEqual(self.commands, [])

    def test_failed_projects_folder_save_reports_recoverable_event(
        self,
    ) -> None:
        """A preference failure must not kill the running GUI engine."""

        window = importlib.import_module("kilin.gui.window")
        fake = MagicMock()
        with patch.object(
            window, "write_defaults", side_effect=OSError("read-only")
        ):
            window.ChatWindow._set_projects_folder(fake, "")
        fake.process.kill.assert_not_called()
        fake._process_finished.assert_not_called()
        emitted = json.loads(fake.bridge.event.emit.call_args.args[0])
        self.assertEqual(emitted["event"], "projects_folder_failed")
        self.assertIn("read-only", emitted["message"])

    def test_accepting_the_terms_saves_the_current_version(self) -> None:
        """Confirm only a persisted current-version agreement."""

        window = importlib.import_module("kilin.gui.window")
        with patch.object(window, "write_defaults") as write:
            self.bridge.acceptTerms("an outdated version")
            write.assert_not_called()
            self.assertEqual(self.commands, [])
            self.bridge.acceptTerms(window.TERMS_VERSION)
        write.assert_called_once_with(
            {"terms_accepted": window.TERMS_VERSION}
        )
        self.assertEqual(self.commands, [{"action": "readiness"}])
        self.assertEqual([
            json.loads(message) for message in self.events
        ], [{
            "event": "terms_acceptance_failed",
            "version": "an outdated version",
            "reason": "outdated_version",
            "current_version": window.TERMS_VERSION,
            "message": (
                "The terms of use have changed. Review the current "
                "version and accept it again."
            ),
        }, {
            "event": "terms_acceptance_saved",
            "version": window.TERMS_VERSION,
        }])

    def test_terms_write_failures_allow_retry(self) -> None:
        """Do not send readiness for unwritten acceptance."""

        window = importlib.import_module("kilin.gui.window")
        for problem in (OSError("read-only"), ValueError("invalid file")):
            with self.subTest(problem=problem):
                with patch.object(
                    window, "write_defaults", side_effect=problem,
                ):
                    self.bridge.acceptTerms(window.TERMS_VERSION)
                self.assertEqual(self.commands, [])
                event = json.loads(self.events[-1])
                self.assertEqual(event["event"], "terms_acceptance_failed")
                self.assertEqual(event["reason"], "save_failed")
                self.assertEqual(event["version"], window.TERMS_VERSION)
                self.assertIn(str(problem), event["message"])
        with patch.object(window, "write_defaults") as write:
            self.bridge.acceptTerms(window.TERMS_VERSION)
        write.assert_called_once()
        self.assertEqual(self.commands, [{"action": "readiness"}])
        self.assertEqual(
            json.loads(self.events[-1])["event"], "terms_acceptance_saved"
        )

    def test_answer_settings_command_payload(self) -> None:
        """The bridge sends the settings as a command of their own."""

        self.bridge.saveAnswerSettings(json.dumps({"thinking": True}))
        self.assertEqual(self.commands, [
            {"action": "save_answer_settings",
             "settings": {"thinking": True}},
        ])

    def test_correlated_answer_settings_bridge_payload(self) -> None:
        """Unwrap the browser envelope for the engine dispatcher."""

        self.bridge.saveAnswerSettings(json.dumps({
            "settings": {"thinking": False, "temperature": 0.4},
            "request_id": 17,
            "project": "/saved/project",
        }))
        self.assertEqual(self.commands, [{
            "action": "save_answer_settings",
            "settings": {"thinking": False, "temperature": 0.4},
            "request_id": 17,
            "project": "/saved/project",
        }])

    def test_copy_text_goes_to_the_system_clipboard(self) -> None:
        """Copy through Qt, since the page may not write the clipboard."""

        clipboard = MagicMock()
        with patch(
            "kilin.gui.window.QGuiApplication.clipboard",
            return_value=clipboard,
        ):
            self.bridge.copyText("Answer [1].")
        clipboard.setText.assert_called_once_with("Answer [1].")
        self.assertEqual(self.commands, [])

if __name__ == "__main__":
    unittest.main()
