"""Verify offline model identities used for retrieval index reuse.

Synthetic weight and tokenizer files exercise content hashing. Hugging Face
snapshot fixtures use fake commit directories, never the network or Hub API.
The tests distinguish immutable Hub commits from mutable local directories.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.retrievals import model_identity as identity


class TestMutableModelIdentity(unittest.TestCase):
    """Ensure local model identity follows file contents, not timestamps."""

    def setUp(self) -> None:
        """Create a tiny model folder with weights and tokenizer files.

        """

        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cache_base = Path(self.temporary.name) / "xdg"
        environment = patch.dict(
            os.environ, {"XDG_CACHE_HOME": str(self.cache_base)}
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.root = Path(self.temporary.name) / "model"
        self.root.mkdir()
        (self.root / "config.json").write_text(
            '{"hidden_size":8}', encoding="utf-8"
        )
        (self.root / "weights.safetensors").write_bytes(b"AAA")
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_text('{"vocab":1}', encoding="utf-8")

    def _identity(self, root: Path | None = None) -> str:
        """Identify the fixture directory by its absolute local path.

        Parameters
        ----------
        root: pathlib.Path | None, optional
            Model directory to identify. Defaults to the fixture root.

        Returns
        -------
        str
            Content-derived embedding model identity.

        """

        selected = self.root if root is None else root
        return identity.embedding_model_identity(str(selected), selected)

    def test_stable_across_repeated_reads_and_other_locations(self) -> None:
        """Equal file trees have the same identity regardless of location.

        """

        baseline = self._identity()
        self.assertTrue(baseline.startswith("local-sha256:"))
        self.assertEqual(baseline, self._identity())
        other = Path(self.temporary.name) / "copy"
        other.mkdir()
        for file in self.root.iterdir():
            (other / file.name).write_bytes(file.read_bytes())
        self.assertEqual(baseline, self._identity(other))

    @unittest.skipIf(
        os.name == "nt",
        "Windows has no change time: a rewrite restoring size and mtime "
        "is not detectable without rehashing",
    )
    def test_weight_change_with_preserved_size_and_mtime(self) -> None:
        """Detect modified model weights even with identical stat fields.

        """

        weights = self.root / "weights.safetensors"
        before = self._identity()
        stat = weights.stat()
        weights.write_bytes(b"BBB")
        os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(weights.stat().st_size, stat.st_size)
        self.assertEqual(weights.stat().st_mtime_ns, stat.st_mtime_ns)
        self.assertNotEqual(self._identity(), before)

    def test_force_verify_with_preserved_size_and_mtime(self) -> None:
        """Rehash modified weights even when common stat fields match.

        This check applies on Windows without relying on ``st_ctime``
        detecting a same-size rewrite. The default metadata cache remains
        unchanged; callers can explicitly request full content checking.

        """

        weights = self.root / "weights.safetensors"
        before = self._identity()
        stat = weights.stat()
        weights.write_bytes(b"BBB")
        os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(weights.stat().st_size, stat.st_size)
        self.assertEqual(weights.stat().st_mtime_ns, stat.st_mtime_ns)
        checked = identity.embedding_model_identity(
            str(self.root), self.root, force_verify=True,
        )
        self.assertNotEqual(checked, before)
        self.assertEqual(self._identity(), checked)

    def test_tokenizer_change_invalidates_identity(self) -> None:
        """Tokenizer changes must invalidate document/query comparability.

        """

        before = self._identity()
        self.tokenizer.write_text('{"vocab":2}', encoding="utf-8")
        self.assertNotEqual(self._identity(), before)

    def test_added_and_removed_files_change_identity(self) -> None:
        """A model file set is part of the identity, not only contents.

        """

        baseline = self._identity()
        extra = self.root / "adapter.safetensors"
        extra.write_bytes(b"adapter")
        self.assertNotEqual(self._identity(), baseline)
        extra.unlink()
        self.assertEqual(self._identity(), baseline)

    def test_nested_files_and_bounded_reading(self) -> None:
        """Detect nested assets while reading weight files in blocks.

        """

        nested = self.root / "nested"
        nested.mkdir()
        file = nested / "part.bin"
        file.write_bytes(b"abcdefghijkl")
        with patch.object(identity, "_FILE_BLOCK_SIZE", 3):
            before = self._identity()
        file.write_bytes(b"abcdefghijkl+")
        self.assertNotEqual(self._identity(), before)

    def _manifest(self, root: Path | None = None) -> Path:
        """Return the outside-model manifest for one fixture directory.

        Parameters
        ----------
        root: pathlib.Path | None, optional
            Local model to identify. Defaults to the primary fixture.

        Returns
        -------
        pathlib.Path
            Resolved filename of the model's persistent digest cache.

        """

        selected = self.root if root is None else root
        directory = identity._manifest_directory(None)
        return identity._manifest_path(selected.resolve(), directory)

    def test_cold_cache_hashes_files_and_warm_cache_avoids_io(self) -> None:
        """Reusing an unchanged manifest requires no model-file reads.

        """

        with patch.object(
            identity, "_file_digest", wraps=identity._file_digest,
        ) as digest:
            first = self._identity()
            self.assertEqual(digest.call_count, 3)
            digest.reset_mock()
            self.assertEqual(self._identity(), first)
            digest.assert_not_called()
        payload = json.loads(self._manifest().read_text(encoding="utf-8"))
        self.assertEqual(payload["root"], str(self.root.resolve()))
        self.assertEqual(len(payload["files"]), 3)
        self.assertFalse(list(self.root.rglob("*.tmp")))
        self.assertFalse(list(self.root.rglob("*.json.json")))

    def test_metadata_change_rehashes_only_modified_file(self) -> None:
        """A changed tokenizer does not require rehashing model weights.

        """

        original = self._identity()
        self.tokenizer.write_text('{"vocab":2}', encoding="utf-8")
        with patch.object(
            identity, "_file_digest", wraps=identity._file_digest,
        ) as digest:
            updated = self._identity()
        self.assertNotEqual(original, updated)
        self.assertEqual(digest.call_count, 1)
        self.assertEqual(
            digest.call_args.args[0], self.tokenizer.resolve()
        )

    def test_metadata_only_change_preserves_content_identity(self) -> None:
        """A touched model file is rechecked without changing its identity.

        """

        original = self._identity()
        weights = self.root / "weights.safetensors"
        stat = weights.stat()
        os.utime(
            weights,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
        )
        with patch.object(
            identity, "_file_digest", wraps=identity._file_digest,
        ) as digest:
            self.assertEqual(self._identity(), original)
        self.assertEqual(digest.call_count, 1)
        self.assertEqual(digest.call_args.args[0], weights.resolve())

    def test_symlinked_model_root_uses_canonical_file_paths(self) -> None:
        """Resolve file paths when a model root is reached by a symlink.

        """

        alias = Path(self.temporary.name) / "model-alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.root = alias
        self.tokenizer = alias / "tokenizer.json"
        baseline = self._identity()
        self.tokenizer.write_text('{"vocab":2}', encoding="utf-8")
        with patch.object(
            identity, "_file_digest", wraps=identity._file_digest,
        ) as digest:
            changed = self._identity()
        self.assertNotEqual(changed, baseline)
        self.assertEqual(digest.call_count, 1)
        self.assertEqual(
            digest.call_args.args[0], self.tokenizer.resolve()
        )

    def test_incremental_manifest_tracks_deletion(self) -> None:
        """Removed model files are absent from the updated manifest.

        """

        self._identity()
        self.tokenizer.unlink()
        with patch.object(
            identity, "_file_digest", wraps=identity._file_digest,
        ) as digest:
            self._identity()
            digest.assert_not_called()
        records = json.loads(self._manifest().read_text(encoding="utf-8"))
        self.assertNotIn("tokenizer.json", records["files"])

    def test_identity_does_not_depend_on_cache_location(self) -> None:
        """A cached identity equals a fresh full-content fingerprint.

        """

        current = self._identity()
        self.assertEqual(
            current,
            "local-sha256:" + identity._directory_digest(self.root),
        )
        other_cache = Path(self.temporary.name) / "other-cache"
        self.assertEqual(
            current,
            identity.embedding_model_identity(
                str(self.root), self.root, cache_directory=other_cache,
            ),
        )

    def test_cached_digest_matches_original_uncached_algorithm(self) -> None:
        """Caching must preserve the previously saved local model identity.

        """

        original = sha256()
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            name = path.relative_to(self.root).as_posix()
            file_hash = sha256(path.read_bytes()).hexdigest()
            encoded = json.dumps(
                [name, file_hash], ensure_ascii=True, separators=(",", ":"),
            )
            original.update(encoded.encode("utf-8"))
        expected = f"local-sha256:{original.hexdigest()}"
        self.assertEqual(self._identity(), expected)
        self.assertEqual(self._identity(), expected)

    def test_corrupted_cache_is_recomputed(self) -> None:
        """Unreadable or malformed manifests never provide stale hashes.

        """

        expected = self._identity()
        self._manifest().write_text("{broken", encoding="utf-8")
        with patch.object(
            identity, "_file_digest", wraps=identity._file_digest,
        ) as digest:
            self.assertEqual(self._identity(), expected)
            self.assertEqual(digest.call_count, 3)
        repaired = json.loads(self._manifest().read_text(encoding="utf-8"))
        self.assertEqual(len(repaired["files"]), 3)

    def test_manifest_missing_file_dictionary_is_recomputed(self) -> None:
        """A parseable but incomplete cache must never be trusted.

        """

        expected = self._identity()
        cache = self._manifest()
        cache.write_text(
            json.dumps({"root": str(self.root.resolve()), "files": []}),
            encoding="utf-8",
        )
        with patch.object(
            identity, "_file_digest", wraps=identity._file_digest,
        ) as digest:
            self.assertEqual(self._identity(), expected)
        self.assertEqual(digest.call_count, 3)

    def test_wrong_cache_model_and_invalid_entry_are_not_reused(self) -> None:
        """Require the correct root and genuine SHA-256 hex strings.

        """

        expected = self._identity()
        cache = self._manifest()
        manifest = json.loads(cache.read_text(encoding="utf-8"))
        manifest["root"] = "different-model"
        cache.write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(
            identity, "_file_digest", wraps=identity._file_digest,
        ) as digest:
            self.assertEqual(self._identity(), expected)
            self.assertEqual(digest.call_count, 3)
        manifest = json.loads(cache.read_text(encoding="utf-8"))
        manifest["files"]["tokenizer.json"]["sha256"] = "not-a-hash"
        cache.write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(
            identity, "_file_digest", wraps=identity._file_digest,
        ) as digest:
            self.assertEqual(self._identity(), expected)
            self.assertEqual(digest.call_count, 1)

    def test_force_verify_detects_spoofed_metadata(self) -> None:
        """Force verification bypasses metadata-validated cached digests.

        """

        baseline = self._identity()
        weights = self.root / "weights.safetensors"
        original_metadata = identity._file_metadata(weights)
        weights.write_bytes(b"BBB")
        real_metadata = identity._file_metadata

        def spoof(path: Path) -> dict[str, int]:
            """Simulate metadata that fails to reveal modified contents.

            Parameters
            ----------
            path: pathlib.Path
                File being checked by the cache.

            Returns
            -------
            dict[str, int]
                Original metadata for weights; real metadata otherwise.

            """

            if path.resolve() == weights.resolve():
                return original_metadata
            return real_metadata(path)

        with patch.object(identity, "_file_metadata", side_effect=spoof):
            self.assertEqual(self._identity(), baseline)
            changed = identity.embedding_model_identity(
                str(self.root), self.root, force_verify=True,
            )
        self.assertNotEqual(changed, baseline)
        self.assertEqual(self._identity(), changed)

    def test_force_verify_rehashes_every_file(self) -> None:
        """Forced verification never trusts cached model-file digests.

        """

        original = self._identity()
        with patch.object(
            identity, "_file_digest", wraps=identity._file_digest,
        ) as digest:
            verified = identity.embedding_model_identity(
                str(self.root), self.root, force_verify=True,
            )
        self.assertEqual(original, verified)
        self.assertEqual(digest.call_count, 3)

    def test_cache_read_failure_falls_back_to_content_hashing(self) -> None:
        """A read-inaccessible cache cannot block model identification.

        """

        expected = self._identity()
        with patch.object(Path, "read_text", side_effect=OSError("denied")):
            with patch.object(
                identity, "_file_digest", wraps=identity._file_digest,
            ) as digest:
                self.assertEqual(self._identity(), expected)
                self.assertEqual(digest.call_count, 3)

    def test_cache_write_failure_preserves_content_identity(self) -> None:
        """A cache publication error must not prevent model reuse.

        """

        with patch.object(Path, "replace", side_effect=OSError("denied")):
            expected = self._identity()
        self.assertFalse(self._manifest().exists())
        self.assertEqual(self._identity(), expected)
        self.assertFalse(list(self._manifest().parent.glob("*.tmp")))

    def test_temporary_creation_failure_does_not_stop_identity(self) -> None:
        """A cache directory that cannot create files is optional.

        """

        with patch.object(
            identity, "NamedTemporaryFile", side_effect=OSError("denied"),
        ):
            result = self._identity()
        self.assertFalse(self._manifest().exists())
        self.assertEqual(result, self._identity())

    def test_temporary_cleanup_failure_is_nonfatal(self) -> None:
        """Failure to remove an already published temporary file is safe.

        """

        with patch.object(Path, "unlink", side_effect=OSError("denied")):
            first = self._identity()
        self.assertEqual(first, self._identity())
        self.assertTrue(self._manifest().is_file())

    def test_cache_directory_must_not_be_within_model(self) -> None:
        """An explicitly selected cache may not alter the model tree.

        """

        with self.assertRaisesRegex(ValueError, "outside"):
            identity.embedding_model_identity(
                str(self.root), self.root,
                cache_directory=self.root / ".fingerprints",
            )
        self.assertFalse((self.root / ".fingerprints").exists())

    def test_default_cache_inside_model_disables_caching(self) -> None:
        """Avoid an in-model cache even with unusual XDG settings.

        """

        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root)}):
            baseline = self._identity()
            self.assertFalse((self.root / "kilin").exists())
        self.assertEqual(self._identity(), baseline)

    def test_relative_xdg_home_falls_back_to_user_cache(self) -> None:
        """An invalid relative XDG_CACHE_HOME is ignored.

        """

        with patch.dict(os.environ, {"XDG_CACHE_HOME": "relative"}):
            directory = identity._manifest_directory(None)
        self.assertEqual(
            directory, Path.home() / ".cache" / "kilin" / "model_identity",
        )

    def test_model_change_mid_hash_is_rejected(self) -> None:
        """A race during the initial content read must fail closed.

        """

        original = identity._file_metadata
        weights = self.root / "weights.safetensors"
        counter = 0

        def unstable(path: Path) -> dict[str, int]:
            """Introduce a simulated metadata change after file hashing.

            Parameters
            ----------
            path: pathlib.Path
                File being inspected.

            Returns
            -------
            dict[str, int]
                Modified metadata on the second inspection of weights.

            """

            nonlocal counter
            metadata = original(path)
            if path.resolve() == weights.resolve():
                counter += 1
                if counter == 2:
                    metadata["size"] += 1
            return metadata

        with patch.object(identity, "_file_metadata", side_effect=unstable):
            with self.assertRaisesRegex(OSError, "changed while hashing"):
                self._identity()
        self.assertFalse(self._manifest().exists())

    def test_manifest_file_does_not_change_source_digest(self) -> None:
        """The external cache never appears in the model file inventory.

        """

        names_before = sorted(p.name for p in self.root.iterdir())
        self._identity()
        self.assertEqual(
            names_before, sorted(p.name for p in self.root.iterdir())
        )
        self.assertTrue(self._manifest().is_file())

    def test_missing_or_empty_local_source_is_rejected(self) -> None:
        """A non-existent or content-free directory cannot identify a model.

        """

        missing = Path(self.temporary.name) / "missing"
        with self.assertRaisesRegex(ValueError, "directory"):
            identity.embedding_model_identity("org/model", missing)
        empty = Path(self.temporary.name) / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "empty"):
            identity.embedding_model_identity(str(empty), empty)

    def test_unreadable_model_file_propagates_error(self) -> None:
        """An incomplete identity must never silently permit index reuse.

        """

        with patch.object(Path, "open", side_effect=OSError("no read")):
            with self.assertRaisesRegex(OSError, "no read"):
                self._identity()


class TestHubSnapshotIdentity(unittest.TestCase):
    """Use immutable Hub commit IDs without hashing large weight files."""

    def setUp(self) -> None:
        """Create a cache folder following the standard Hub layout.

        """

        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = patch.dict(
            os.environ, {"XDG_CACHE_HOME": str(self.root / "xdg")}
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.commit = "a" * 40
        self.snapshot = (
            self.root / "models--org--embedding" / "snapshots" / self.commit
        )
        self.snapshot.mkdir(parents=True)
        (self.snapshot / "weights.safetensors").write_bytes(b"AAA")

    def test_remote_revision_uses_commit_without_file_reads(self) -> None:
        """A standard Hub snapshot uses its commit rather than file I/O.

        """

        with patch.object(identity, "_directory_digest") as checksum:
            found = identity.embedding_model_identity(
                "org/embedding", self.snapshot
            )
        self.assertEqual(found, f"huggingface:org/embedding@{self.commit}")
        checksum.assert_not_called()

    def test_different_commits_produce_different_identity(self) -> None:
        """A moved repository revision invalidates previously saved vectors.

        """

        second = self.snapshot.parent / ("b" * 40)
        second.mkdir()
        self.assertNotEqual(
            identity.embedding_model_identity("org/embedding", self.snapshot),
            identity.embedding_model_identity("org/embedding", second),
        )

    def test_mismatched_repository_falls_back_to_file_hash(self) -> None:
        """Arbitrary sources cannot masquerade as the requested Hub cache.

        """

        value = identity.embedding_model_identity(
            "org/different", self.snapshot
        )
        self.assertTrue(value.startswith("local-sha256:"))

    def test_explicit_local_snapshot_path_hashes_contents(self) -> None:
        """Explicitly local directories must never assume immutability.

        """

        value = identity.embedding_model_identity(
            str(self.snapshot), self.snapshot
        )
        self.assertTrue(value.startswith("local-sha256:"))

    def test_malformed_commit_directory_falls_back_to_file_hash(self) -> None:
        """Recognize only full commit hashes at the expected location.

        """

        invalid = self.snapshot.parent / "main"
        invalid.mkdir()
        (invalid / "weights.safetensors").write_bytes(b"AAA")
        value = identity.embedding_model_identity("org/embedding", invalid)
        self.assertTrue(value.startswith("local-sha256:"))


if __name__ == "__main__":
    unittest.main()
