"""Tests for the thin Hugging Face cache connection."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qachat import huggingface


class TestHuggingFace(unittest.TestCase):
    def test_available_models(self) -> None:
        revision = SimpleNamespace(commit_hash="abc123")
        repository = SimpleNamespace(
            repo_type="model",
            repo_id="group/model",
            repo_path=Path("/cache/group--model"),
            size_on_disk=123,
            revisions=[revision],
        )
        dataset = SimpleNamespace(
            repo_type="dataset",
            repo_id="group/data",
            repo_path=Path("/cache/group--data"),
            size_on_disk=50,
            revisions=[],
        )
        cache = SimpleNamespace(repos=[repository, dataset])
        with patch(
            "qachat.huggingface.scan_cache_dir",
            return_value=cache,
        ):
            models = huggingface.available_models()
        self.assertEqual(models[0]["model_id"], "group/model")
        self.assertEqual(models[0]["revisions"], ["abc123"])

        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            self.assertEqual(huggingface.available_models(missing), [])

    def test_model_is_available(self) -> None:
        models = [{"model_id": "group/model"}]
        with patch(
            "qachat.huggingface.available_models",
            return_value=models,
        ):
            self.assertTrue(huggingface.model_is_available("group/model"))
            self.assertFalse(huggingface.model_is_available("missing/model"))
        with self.assertRaises(ValueError):
            huggingface.model_is_available("")

    def test_download_model(self) -> None:
        expected = Path("/cache/snapshot")
        with patch(
            "qachat.huggingface.snapshot_download",
            return_value=str(expected),
        ) as download:
            result = huggingface.download_model(
                "group/model",
                revision="abc",
            )
        self.assertEqual(result, expected)
        download.assert_called_once_with(
            repo_id="group/model",
            revision="abc",
        )
        with self.assertRaises(ValueError):
            huggingface.download_model("")


if __name__ == "__main__":
    unittest.main()
