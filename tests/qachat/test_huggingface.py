"""Unit tests for QAChat's Hugging Face cache interface.

Tests exercise cache discovery, offline snapshot resolution, and download
coordination without accessing the network or the user's model cache.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qachat import huggingface


class TestAvailableModels(unittest.TestCase):
    """Test model repository discovery and cache metadata."""

    def test_filters_and_sorts_cached_models(self) -> None:
        """Return sorted model metadata, excluding other repository types."""

        def repository(repo_id, repo_type, revisions, size):
            """Construct minimal Hub repository metadata for a test."""

            return SimpleNamespace(
                repo_id=repo_id,
                repo_type=repo_type,
                repo_path=Path("/cache") / repo_id,
                size_on_disk=size,
                revisions=[SimpleNamespace(commit_hash=r) for r in revisions],
            )

        cache = SimpleNamespace(
            repos=[
                repository("z/model", "model", ["bbb", "aaa"], 123),
                repository("a/data", "dataset", ["ccc"], 10),
                repository("a/model", "model", [], 0),
            ]
        )
        with patch.object(
            huggingface, "scan_cache_dir", return_value=cache
        ) as scan:
            models = huggingface.available_models("/cache")

        scan.assert_called_once_with(cache_dir="/cache")
        self.assertEqual(
            models,
            [
                {
                    "model_id": "a/model",
                    "path": str(Path("/cache/a/model")),
                    "size_bytes": 0,
                    "revisions": [],
                },
                {
                    "model_id": "z/model",
                    "path": str(Path("/cache/z/model")),
                    "size_bytes": 123,
                    "revisions": ["aaa", "bbb"],
                },
            ],
        )

    def test_missing_cache_returns_empty_list(self) -> None:
        """Return no models when the Hub reports a missing cache."""

        with patch.object(
            huggingface,
            "scan_cache_dir",
            side_effect=huggingface.CacheNotFound("Missing cache", "/missing"),
        ):
            self.assertEqual(huggingface.available_models("/missing"), [])

    def test_unexpected_cache_error_propagates(self) -> None:
        """Do not suppress errors unrelated to a missing cache."""

        with patch.object(
            huggingface, "scan_cache_dir", side_effect=OSError("unreadable")
        ):
            with self.assertRaisesRegex(OSError, "unreadable"):
                huggingface.available_models()


class TestModelIsAvailable(unittest.TestCase):
    """Test exact model repository membership checks."""

    def test_checks_exact_identifier_and_passes_cache_directory(self) -> None:
        """Distinguish exact IDs and forward the requested cache path."""

        models = [{"model_id": "group/model"}]
        with patch.object(
            huggingface, "available_models", return_value=models
        ) as available:
            self.assertTrue(
                huggingface.model_is_available("group/model", "/cache")
            )
            self.assertFalse(
                huggingface.model_is_available("group/model-extra", "/cache")
            )
        self.assertEqual(available.call_count, 2)
        available.assert_any_call("/cache")

    def test_rejects_invalid_model_identifiers(self) -> None:
        """Reject empty, whitespace-only, and non-string identifiers."""

        for model_id in ("", "  ", None, 123):
            with self.subTest(model_id=model_id):
                with self.assertRaises(ValueError):
                    huggingface.model_is_available(model_id)


class TestCachedModelSnapshot(unittest.TestCase):
    """Test local-only resolution and snapshot validation."""

    def test_resolves_snapshot_with_config(self) -> None:
        """Accept a local snapshot containing config.json."""

        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            with patch.object(
                huggingface, "snapshot_download", return_value=str(snapshot)
            ) as lookup:
                result = huggingface.cached_model_snapshot(
                    "group/model", revision="main"
                )
            self.assertEqual(result, snapshot)
            lookup.assert_called_once_with(
                repo_id="group/model",
                revision="main",
                local_files_only=True,
            )

    def test_rejects_snapshot_without_config(self) -> None:
        """Reject incomplete snapshots, even when the Hub returns a path."""

        with TemporaryDirectory() as directory:
            with patch.object(
                huggingface, "snapshot_download", return_value=directory
            ):
                with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                    huggingface.cached_model_snapshot("group/model")

    def test_legacy_incomplete_snapshot_uses_same_validation(self) -> None:
        """Apply config validation to older Hub partial-snapshot errors."""

        class PartialSnapshotError(Exception):
            """Represent the legacy Hub exception with a snapshot path."""

            def __init__(self, snapshot_path):
                """Store the snapshot path reported by the Hub."""
                self.snapshot_path = snapshot_path

        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            error = PartialSnapshotError(str(snapshot))
            with (
                patch.object(
                    huggingface,
                    "_IncompleteSnapshotError",
                    PartialSnapshotError,
                ),
                patch.object(
                    huggingface, "snapshot_download", side_effect=error
                ),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                    huggingface.cached_model_snapshot("group/model")
                (snapshot / "config.json").write_text("{}", encoding="utf-8")
                self.assertEqual(
                    huggingface.cached_model_snapshot("group/model"), snapshot
                )

    def test_rejects_invalid_identifier_before_lookup(self) -> None:
        """Avoid Hub calls for invalid model identifiers."""

        with patch.object(huggingface, "snapshot_download") as lookup:
            with self.assertRaises(ValueError):
                huggingface.cached_model_snapshot(" ")
        lookup.assert_not_called()

    def test_other_hub_errors_propagate(self) -> None:
        """Preserve unexpected errors from offline snapshot resolution."""

        with patch.object(
            huggingface, "snapshot_download", side_effect=OSError("offline")
        ):
            with self.assertRaisesRegex(OSError, "offline"):
                huggingface.cached_model_snapshot("group/model")


class TestDownloadModel(unittest.TestCase):
    """Test delegated downloads and progress callback integration."""

    def test_download_delegates_to_hub(self) -> None:
        """Forward model and revision without custom progress handling."""

        expected = Path("/cache/snapshot")
        with patch.object(
            huggingface, "snapshot_download", return_value=str(expected)
        ) as download:
            result = huggingface.download_model("group/model", revision="abc")
        self.assertEqual(result, expected)
        download.assert_called_once_with(repo_id="group/model", revision="abc")

    def test_invalid_inputs_do_not_trigger_download(self) -> None:
        """Validate identifiers and callbacks before contacting the Hub."""

        with patch.object(huggingface, "snapshot_download") as download:
            with self.assertRaises(ValueError):
                huggingface.download_model("")
            with self.assertRaisesRegex(ValueError, "progress_callback"):
                huggingface.download_model(
                    "group/model", progress_callback=123
                )
        download.assert_not_called()

    def test_non_string_hub_result_is_rejected(self) -> None:
        """Require the Hub to return a string snapshot path."""

        with patch.object(
            huggingface, "snapshot_download", return_value=Path("/cache")
        ):
            with self.assertRaises(RuntimeError):
                huggingface.download_model("group/model")

    def test_progress_callback_reports_aggregate_bytes(self) -> None:
        """Forward aggregate byte updates and ignore unrelated Hub bars."""

        callback = Mock()
        with patch.object(
            huggingface, "snapshot_download", return_value="/cache/snapshot"
        ) as download:
            huggingface.download_model(
                "group/model", progress_callback=callback
            )
        progress_class = download.call_args.kwargs["tqdm_class"]
        with progress_class(
            total=100, desc="Reconstructing snapshot", unit="B"
        ) as progress:
            progress.update(25)
            progress.update(50)
        self.assertIn(unittest.mock.call(0, 100), callback.call_args_list)
        self.assertIn(unittest.mock.call(25, 100), callback.call_args_list)
        self.assertIn(unittest.mock.call(75, 100), callback.call_args_list)

        callback.reset_mock()
        with progress_class(total=2, desc="Downloading", unit="files") as bar:
            bar.update(1)
        callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
