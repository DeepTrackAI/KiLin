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
                {"action": "save_answer_timeline", "timestamp": "now",
                 "timeline": {"elapsed_ms": 10, "steps": []}},
                "save_answer_timeline",
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
            patch.object(engine, "available_models", return_value=[]),
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
            "answer_delta", "answer_phase", "answered", "backend_failed",
            "backend_restarted", "conversation_delete_started",
            "conversation_opened", "conversations", "error",
            "workspace_conflict",
            "folder_changed", "installation_complete",
            "installation_model_complete", "installation_progress",
            "installation_started", "installed_models_used",
            "model_download_progress", "model_download_started",
            "model_downloaded", "models_applied", "opened", "preview",
            "progress", "projects", "readiness", "reference_error",
            "reference_preview", "reset", "settings", "source_changes",
            "status", "stopped", "trace_ready", "usage_progress",
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

    def test_stop_is_a_signal_not_a_child_command(self) -> None:
        """Preserve out-of-band process termination via Qt signal."""

        self.bridge.stopOperation()
        self.assertEqual(self.stops, [True])
        self.assertEqual(self.commands, [])

    def test_tool_plan_choice_uses_special_action(self) -> None:
        """Carry plan decisions through the existing command envelope."""

        self.bridge.choosePlan("continue")
        self.assertEqual(self.commands, [
            {"action": "plan_choice", "choice": "continue"},
        ])


if __name__ == "__main__":
    unittest.main()
