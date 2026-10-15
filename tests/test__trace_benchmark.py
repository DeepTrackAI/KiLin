"""Regression tests for the PR4 notebook and read-only trace analysis.

Tests use synthetic QAChat traces and never load a language model, download
artifacts, or modify the production answering and tracing implementations.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import ast
from configparser import ConfigParser
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import qachat
from assets.trace_benchmark import (
    question_statistics,
    read_trace_operations,
    summarize_trace_turns,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "assets" / "benchmark_traces.ipynb"


class TestTraceBenchmark(unittest.TestCase):
    """Verify timing extraction without exposing trace diagnostic payloads."""

    def setUp(self) -> None:
        """Allocate an isolated trace directory for each test."""
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "conversation_test.json"

    def _make_trace(self) -> None:
        """Create a representative numbered trace with sensitive content."""
        logger = qachat.TraceLogger(
            qachat.TraceConfig(enabled=True, output_path=self.path),
            default_directory=self.path.parent,
        )
        try:
            with logger.span("model.chat.load", component="chat"):
                pass
            logger.emit(
                "session.ready",
                component="rag",
                duration_ms=99.0,
            )
            with logger.span(
                "turn", component="rag", turn_id="turn_test"
            ):
                with logger.span(
                    "intent.classification",
                    component="rag",
                    turn_id="turn_test",
                ):
                    pass
                with logger.span(
                    "retrieval.semantic",
                    component="retrieval",
                    turn_id="turn_test",
                ):
                    pass
                with logger.span(
                    "llm.call",
                    component="chat",
                    turn_id="turn_test",
                ):
                    logger.emit(
                        "llm.call.output",
                        component="chat",
                        turn_id="turn_test",
                        duration_ms=200.0,
                        data={
                            "output_token_count": 50,
                            "reply_duration_ms": 240.0,
                            "answer_artifact": "TOP SECRET OUTPUT",
                        },
                    )
        finally:
            logger.close()

    def test_parses_timing_fields_only(self) -> None:
        """Drop content, file paths, prompts and other diagnostic fields."""
        self._make_trace()
        rows = read_trace_operations(self.path, "benchmark_01")
        self.assertTrue(rows)
        self.assertNotIn("TOP SECRET OUTPUT", repr(rows))
        self.assertNotIn(str(self.path), repr(rows))
        self.assertEqual(
            {row["run"] for row in rows}, {"benchmark_01"}
        )
        names = {row["operation"] for row in rows}
        self.assertIn("session.ready", names)
        self.assertIn("model.chat.load", names)
        self.assertIn("retrieval.semantic", names)
        self.assertIn("llm.call.output", names)
        output = next(
            item for item in rows
            if item["operation"] == "llm.call.output"
        )
        self.assertEqual(output["duration_ms"], 200.0)
        self.assertEqual(output["reply_ms"], 240.0)
        self.assertEqual(output["output_tokens"], 50)

    def test_turn_summary_does_not_double_count_spans(self) -> None:
        """Keep inclusive turn and constituent operations separate."""
        self._make_trace()
        rows = read_trace_operations(self.path, "benchmark_01")
        summary = summarize_trace_turns(rows)
        self.assertEqual(len(summary), 1)
        turn = summary[0]
        self.assertEqual(turn["turn"], 1)
        self.assertIsNotNone(turn["turn_ms"])
        self.assertIsNotNone(turn["routing_ms"])
        self.assertIsNotNone(turn["semantic_ms"])
        self.assertIsNone(turn["lexical_ms"])
        self.assertEqual(turn["generation_ms"], 200.0)
        self.assertEqual(turn["output_tokens"], 50)
        self.assertEqual(turn["output_tokens_per_second"], 250.0)
        self.assertEqual(turn["llm_calls"], 1)
        self.assertGreaterEqual(turn["turn_ms"], 0.0)

    def test_missing_or_failed_spans_remain_unavailable(self) -> None:
        """Never interpret incomplete or failed durations as successful."""
        payload = {
            "sessions": {
                "05_conversations": {
                    "category": "conversations",
                    "turns": {
                        "turn_0001": {
                            "stages": {
                                "01_question": {
                                    "operations": {
                                        "01_turn": {
                                            "event": {"status": "failed"},
                                            "timing": {
                                                "operation_duration_ms": 30,
                                            },
                                        },
                                        "02_history_selection": {
                                            "event": {"status": "running"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        rows = read_trace_operations(self.path, "failure")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")
        turns = summarize_trace_turns(rows)
        self.assertEqual(len(turns), 1)
        self.assertIsNone(turns[0]["turn_ms"])
        self.assertIsNone(turns[0]["generation_ms"])

    def test_rejects_malformed_file_and_invalid_label(self) -> None:
        """Fail visibly for a malformed saved trace or anonymous run."""
        self.path.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "JSON object"):
            read_trace_operations(self.path, "first")
        with self.assertRaisesRegex(ValueError, "run label"):
            read_trace_operations(self.path, " ")

    def test_reopened_conversation_turns_are_distinct(self) -> None:
        """Associate turns by ordinal rather than a re-used turn ID."""
        for _ in range(2):
            logger = qachat.TraceLogger(
                qachat.TraceConfig(enabled=True, output_path=self.path),
                default_directory=self.path.parent,
            )
            try:
                with logger.span(
                    "turn", component="rag", turn_id="turn_reused"
                ):
                    pass
            finally:
                logger.close()
        rows = read_trace_operations(self.path, "reopened")
        self.assertEqual(
            [item["turn"] for item in summarize_trace_turns(rows)],
            [1, 2],
        )

    def test_question_statistics_are_grouped_by_trace_state(self) -> None:
        """Report reproducible observed ranges without false confidence."""
        results = [
            {
                "question_id": "overview",
                "trace_enabled": True,
                "wall_ms": value,
                "status": "completed",
            }
            for value in (100.0, 300.0, 200.0)
        ]
        results += [
            {
                "question_id": "overview",
                "trace_enabled": False,
                "wall_ms": 80.0,
                "status": "completed",
            },
            {
                "question_id": "overview",
                "trace_enabled": True,
                "wall_ms": 1000.0,
                "status": "failed",
            },
        ]
        data = question_statistics(results)
        self.assertEqual(len(data), 2)
        traced = next(row for row in data if row["trace_enabled"])
        self.assertEqual(traced["repeats"], 3)
        self.assertEqual(traced["median_ms"], 200.0)
        self.assertEqual(traced["min_ms"], 100.0)
        self.assertEqual(traced["max_ms"], 300.0)


class TestTraceBenchmarkNotebook(unittest.TestCase):
    """Keep the notebook runnable offline and non-mutating by default."""

    def test_clean_notebook_source_compiles(self) -> None:
        """Ship valid cells with no precomputed outputs or execution IDs."""
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            compile(source, f"cell_{index}", "exec")
            for line in source.splitlines():
                self.assertLessEqual(len(line), 79)

    def test_workspace_guard_rejects_external_or_online_projects(self) -> None:
        """Permit only explicit sandbox projects with disabled downloads."""
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        validator = None
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for node in tree.body:
                if (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "validate_benchmark_workspace"
                ):
                    validator = node
        self.assertIsNotNone(validator)
        source = ast.Module(body=[validator], type_ignores=[])
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            project = root / "sandbox" / "benchmark_project"
            project.mkdir(parents=True)
            profile = project / "profile.ini"
            profile.write_text(
                "[models]\nallow_download = false\n", encoding="utf-8"
            )
            environment = {
                "repository_root": root,
                "Path": Path,
                "ConfigParser": ConfigParser,
            }
            exec(compile(source, "workspace_guard", "exec"), environment)
            validate = environment["validate_benchmark_workspace"]
            self.assertEqual(validate(project), project)
            with self.assertRaisesRegex(ValueError, "inside QAChat"):
                validate(root / "outside_project")
            profile.write_text(
                "[models]\nallow_download = true\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "allow_download"):
                validate(project)
            profile.unlink()
            with self.assertRaises(FileNotFoundError):
                validate(project)

    def test_offline_default_does_not_start_models(self) -> None:
        """Run all notebook cells without opening QAChat sessions or files."""
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        script = (
            source + "\nassert not RUN_BENCHMARK\n"
            "assert not TRACE_WRITE_CHECK\nassert not EXPORT_CSV\n"
            "assert benchmark_samples == []\n"
            "assert operation_rows == []\n"
        )
        process = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Benchmark disabled", process.stdout)


if __name__ == "__main__":
    unittest.main()
