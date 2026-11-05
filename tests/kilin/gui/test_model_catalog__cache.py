"""Verify lightweight, offline completeness checks for cached model files.

Synthetic Hugging Face snapshots exercise interrupted downloads, alternate
checkpoint/tokenizer layouts, and revisions without loading models or Qt.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.gui.engine import ChatEngine
from kilin.gui.model_catalog import downloaded_models
from kilin.gui.preferences import required_models


class TestCachedDownloadCompleteness(unittest.TestCase):
    """Count only snapshots with plausible configuration and model files."""

    def setUp(self) -> None:
        """Create disposable cache roots for each independent scenario."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def _snapshot(
        self,
        model_id: str,
        files: dict[str, str],
        *,
        revision: str = "abc",
    ) -> dict[str, object]:
        """Create one synthetic Hub snapshot, including optional shards.

        Parameters
        ----------
        model_id: str
            Synthetic repository identifier.
        files: dict[str, str]
            Relative filenames and contents; empty strings create empty files.
        revision: str, optional
            Revision directory name. Defaults to ``"abc"``.

        Returns
        -------
        dict[str, object]
            Entry compatible with ``kilin.huggingface.available_models``.

        """

        repository = self.root / model_id
        snapshot = repository / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return {"model_id": model_id, "path": str(repository)}

    def _downloaded(self, *entries: dict[str, object]) -> set[str]:
        """Query the real local validator using a synthetic Hub listing.

        Parameters
        ----------
        *entries: dict[str, object]
            Synthetic cached-model repository entries.

        Returns
        -------
        set[str]
            Model identifiers with a complete-enough local snapshot.

        """

        with patch(
            "kilin.gui.model_catalog.available_models",
            return_value=list(entries),
        ):
            return {
                str(entry["model_id"]) for entry in downloaded_models()
            }

    def test_missing_tokenizer_is_not_downloaded(self) -> None:
        """Tokenizer configuration alone is not tokenization data."""

        entry = self._snapshot("demo/no-tokenizer", {
            "config.json": "{}", "model.safetensors": "weight",
            "tokenizer_config.json": "{}",
            "special_tokens_map.json": "{}",
        })
        self.assertEqual(self._downloaded(entry), set())

    def test_supported_single_file_tokenizers(self) -> None:
        """Accept the common tokenizer representations without parsing them."""

        entries = []
        for name in (
            "tokenizer.json", "tokenizer.model", "spiece.model",
            "sentencepiece.bpe.model", "tokenizer.tiktoken", "vocab.txt",
        ):
            with self.subTest(tokenizer=name):
                model_id = f"demo/{name.replace('.', '-')}"
                entries.append(self._snapshot(model_id, {
                    "config.json": "{}", "model.safetensors": "weight",
                    name: "tokenizer data",
                }))
        self.assertEqual(
            self._downloaded(*entries),
            {str(item["model_id"]) for item in entries},
        )

    def test_bpe_pair_needs_both_files(self) -> None:
        """A BPE vocab without merges (or vice versa) is incomplete."""

        complete = self._snapshot("demo/bpe", {
            "config.json": "{}", "pytorch_model.bin": "weight",
            "vocab.json": "{}", "merges.txt": "# merges",
        })
        no_merges = self._snapshot("demo/no-merges", {
            "config.json": "{}", "pytorch_model.bin": "weight",
            "vocab.json": "{}",
        })
        no_vocab = self._snapshot("demo/no-vocab", {
            "config.json": "{}", "pytorch_model.bin": "weight",
            "merges.txt": "# merges",
        })
        self.assertEqual(
            self._downloaded(complete, no_merges, no_vocab),
            {"demo/bpe"},
        )

    def test_empty_or_invalid_config_is_not_downloaded(self) -> None:
        """Require nonempty JSON configuration whose root is an object."""

        entries = [
            self._snapshot(f"demo/config-{index}", {
                "config.json": text, "model.safetensors": "weight",
                "tokenizer.json": "tokenizer",
            })
            for index, text in enumerate(("", "{broken", "[]", "null"))
        ]
        self.assertEqual(self._downloaded(*entries), set())

    def test_empty_weights_and_tokenizers_are_not_complete(self) -> None:
        """Reject zero-byte files without inspecting their contents."""

        entries = [
            self._snapshot("demo/empty-weight", {
                "config.json": "{}", "model.safetensors": "",
                "tokenizer.json": "tokenizer",
            }),
            self._snapshot("demo/empty-tokenizer", {
                "config.json": "{}", "model.safetensors": "weight",
                "tokenizer.json": "",
            }),
            self._snapshot("demo/empty-merges", {
                "config.json": "{}", "model.safetensors": "weight",
                "vocab.json": "{}", "merges.txt": "",
            }),
        ]
        self.assertEqual(self._downloaded(*entries), set())

    def test_safetensors_index_requires_every_nonempty_shard(self) -> None:
        """Prevent a partial multi-shard download from appearing ready."""

        index = json.dumps({"weight_map": {
            "layer1": "model-00001-of-00002.safetensors",
            "layer2": "model-00002-of-00002.safetensors",
        }})
        required = {
            "config.json": "{}", "tokenizer.json": "tokenizer",
            "model.safetensors.index.json": index,
            "model-00001-of-00002.safetensors": "weight",
        }
        complete = self._snapshot("demo/whole", {
            **required, "model-00002-of-00002.safetensors": "weight",
        })
        partial = self._snapshot("demo/partial", required)
        empty = self._snapshot("demo/empty-shard", {
            **required, "model-00002-of-00002.safetensors": "",
        })
        self.assertEqual(
            self._downloaded(complete, partial, empty), {"demo/whole"}
        )

    def test_pytorch_weight_index_has_the_same_requirements(self) -> None:
        """Validate PyTorch's indexed binary checkpoint format as well."""

        index = json.dumps({"weight_map": {
            "a": "pytorch_model-00001-of-00002.bin",
            "b": "pytorch_model-00002-of-00002.bin",
        }})
        required = {
            "config.json": "{}", "vocab.txt": "tokenizer",
            "pytorch_model.bin.index.json": index,
            "pytorch_model-00001-of-00002.bin": "weight",
        }
        complete = self._snapshot("demo/torch-complete", {
            **required, "pytorch_model-00002-of-00002.bin": "weight",
        })
        partial = self._snapshot("demo/torch-partial", required)
        self.assertEqual(
            self._downloaded(complete, partial), {"demo/torch-complete"}
        )

    def test_invalid_indexes_cannot_be_overridden_by_single_weights(
        self,
    ) -> None:
        """Malformed index metadata is not proof of a finished snapshot."""

        corrupt = [
            "{broken", "[]", "{}", '{"weight_map": {}}',
            '{"weight_map": {"a": null}}',
            '{"weight_map": {"a": "../outside.safetensors"}}',
            '{"weight_map": {"a": "tokenizer.json"}}',
        ]
        entries = [
            self._snapshot(f"demo/bad-index-{index}", {
                "config.json": "{}", "tokenizer.json": "tokenizer",
                "model.safetensors": "weight",
                "model.safetensors.index.json": data,
            })
            for index, data in enumerate(corrupt)
        ]
        self.assertEqual(self._downloaded(*entries), set())

    def test_unindexed_legacy_weight_formats_remain_supported(self) -> None:
        """Preserve simple nonempty NPZ and GGUF cache representations."""

        entries = [
            self._snapshot(f"demo/{suffix}", {
                "config.json": "{}", "tokenizer.json": "tokenizer",
                f"model.{suffix}": "weight",
            })
            for suffix in ("npz", "gguf")
        ]
        self.assertEqual(
            self._downloaded(*entries), {"demo/npz", "demo/gguf"}
        )

    def test_numbered_shards_without_index_are_not_accepted(self) -> None:
        """A lone shard cannot stand in for a complete unindexed model."""

        entry = self._snapshot("demo/no-index", {
            "config.json": "{}", "tokenizer.json": "tokenizer",
            "model-00001-of-00002.safetensors": "weight",
        })
        self.assertEqual(self._downloaded(entry), set())

    def test_one_complete_revision_is_sufficient(self) -> None:
        """Ignore interrupted revisions when another revision is complete."""

        entry = self._snapshot("demo/revisions", {
            "config.json": "{}", "model.safetensors": "weight",
        }, revision="001-incomplete")
        self._snapshot("demo/revisions", {
            "config.json": "{}", "tokenizer.json": "tokenizer",
            "model.safetensors": "weight",
        }, revision="002-complete")
        self.assertEqual(self._downloaded(entry), {"demo/revisions"})

    def test_hugging_face_snapshot_symlinks(self) -> None:
        """Follow completed blob links and reject broken snapshot links."""

        entry = self._snapshot("demo/linked", {
            "config.json": "{}", "tokenizer.json": "tokenizer",
        })
        repository = Path(str(entry["path"]))
        blob = repository / "blobs" / "weight"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"weight")
        (repository / "snapshots" / "abc" / "model.safetensors").symlink_to(
            blob
        )
        broken = self._snapshot("demo/broken-link", {
            "config.json": "{}", "tokenizer.json": "tokenizer",
        })
        (Path(str(broken["path"])) / "snapshots" / "abc" /
         "model.safetensors").symlink_to(self.root / "missing-blob")
        self.assertEqual(self._downloaded(entry, broken), {"demo/linked"})

    def test_does_not_read_weight_or_tokenizer_contents(self) -> None:
        """Only metadata JSON may be read, even for large weight files."""

        entry = self._snapshot("demo/metadata-only", {
            "config.json": "{}", "tokenizer.json": "tokenizer",
            "model.safetensors": "weight",
        })
        ordinary_read = Path.read_text
        allowed = {"config.json"}

        def guarded_read(path: Path, *args, **kwargs) -> str:
            """Reject any attempt to read tokenizer or weight contents."""

            if path.name not in allowed:
                raise AssertionError(f"Unexpected file read: {path}")
            return ordinary_read(path, *args, **kwargs)

        with (
            patch.object(Path, "read_text", guarded_read),
            patch.object(
                Path, "read_bytes", side_effect=AssertionError("binary read")
            ),
        ):
            self.assertEqual(
                self._downloaded(entry), {"demo/metadata-only"}
            )

    def test_readiness_uses_the_conservative_cache_result(self) -> None:
        """A partial default model stays missing until its tokenizer exists."""

        with patch.dict(
            "os.environ",
            {"KILIN_DEFAULTS_FILE": str(self.root / "preferences.json")},
        ):
            model_id = required_models("torch")[0]
            entry = self._snapshot(model_id, {
                "config.json": "{}", "model.safetensors": "weight",
            })
            events: list[dict[str, object]] = []
            engine = ChatEngine(self.root / "project", events.append)
            self.addCleanup(engine.close)
            with (
                patch(
                    "kilin.gui.model_catalog.available_models",
                    return_value=[entry],
                ),
                patch(
                    "kilin.gui.engine.installed_backends",
                    return_value={"mlx": False, "torch": True},
                ),
                patch(
                    "kilin.gui.engine.choose_backend", return_value="torch"
                ),
            ):
                engine.readiness()
                self.assertIn(model_id, events[-1]["missing_models"])
                snapshot = Path(str(entry["path"])) / "snapshots" / "abc"
                (snapshot / "tokenizer.json").write_text(
                    "tok", encoding="utf-8"
                )
                engine.readiness()
            self.assertNotIn(model_id, events[-1]["missing_models"])
            self.assertEqual(events[-1]["event"], "readiness")


if __name__ == "__main__":
    unittest.main()
