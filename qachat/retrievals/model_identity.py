"""Identify the embedding model that produced a retrieval index.

A standard immutable Hugging Face cache snapshot is identified by its
repository and commit. Mutable local model folders use a digest of their
relative filenames and file contents, including weights and tokenizer data.

A persistent manifest caches per-file SHA-256 digests outside the model
folder. When file metadata is unchanged, the digest can be reused without
rereading the file. This is a metadata-based optimization, not a guarantee
against changes that preserve all observed metadata. Set ``force_verify``
to rehash every file when stronger verification is required.

The identity is intended for index compatibility, not
authentication.

Key Features
------------

- **Immutable model snapshots**

    Identify standard Hugging Face snapshots by repository and commit.

- **Local model fingerprints**

    Hash the relative paths and contents of locally managed model files.

- **Metadata-assisted verification**

    Cache file digests outside the model folder and optionally verify all
    contents regardless of cached metadata.

Module Structure
----------------

- ``embedding_model_identity``

    Return a stable model identity for retrieval-index compatibility.

- Private file and manifest helpers

    Resolve local model files, cache their digests, and check metadata.

Examples
--------
>>> from kilin.retrievals.model_identity import embedding_model_identity
>>> callable(embedding_model_identity)
True

"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


__all__ = ["embedding_model_identity"]


_HF_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_FILE_BLOCK_SIZE = 1024 * 1024
_METADATA_KEYS = ("size", "mtime_ns", "ctime_ns", "dev", "ino")


def embedding_model_identity(
    model_id: str,
    source: str | Path,
    *,
    cache_directory: str | Path | None = None,
    force_verify: bool = False,
) -> str:
    """Identify an embedding model by its resolved files or HF commit.

    Parameters
    ----------
    model_id: str
        Requested local directory or Hugging Face repository identifier.
    source: str | pathlib.Path
        Directory resolved and loaded by ``EmbeddingSession``.
    cache_directory: str | pathlib.Path | None, optional
        Directory for local-file fingerprint manifests. Defaults to
        ``$XDG_CACHE_HOME/kilin/model_identity`` when XDG_CACHE_HOME is
        absolute, or ``~/.cache/kilin/model_identity`` otherwise.
    force_verify: bool, optional
        Rehash all mutable model files even when cached metadata matches.
        Does not affect immutable Hugging Face snapshot identification.
        Defaults to ``False``.

    Returns
    -------
    str
        Stable model identity. A standard Hub cache snapshot uses its
        immutable commit; other directories use content-derived digests.

    Raises
    ------
    ValueError
        If the source is missing or empty, or the selected cache directory
        is inside the model directory.
    OSError
        If model files cannot be inspected or read, or change while hashed.

    Notes
    -----
    Local fingerprints remain unchanged from the uncached implementation:
    the manifest only reduces repeated file reads. File size, modification
    time, change time, device, and inode act as the cache key. Metadata can
    be spoofed; use ``force_verify=True`` to detect such changes. A standard
    Hugging Face snapshot is assumed immutable by convention.

    Cache read/write failures are nonfatal: KíLín rehashes the model files
    and still returns their identity. No cache files are stored in the model.

    """

    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The embedding model source must be a directory.")

    local_id = Path(model_id).expanduser()
    if not local_id.is_dir():
        commit = _hub_snapshot_commit(model_id, root)
        if commit is not None:
            return f"huggingface:{model_id}@{commit}"

    cache: Path | None = _manifest_directory(cache_directory)
    if cache.resolve().is_relative_to(root):
        if cache_directory is not None:
            raise ValueError(
                "The model identity cache must be outside the model "
                "directory."
            )
        cache = None

    return f"local-sha256:{_directory_digest(root, cache, force_verify)}"


def _hub_snapshot_commit(model_id: str, root: Path) -> str | None:
    """Recognize a standard Hugging Face Hub model snapshot directory.

    Parameters
    ----------
    model_id: str
        Hugging Face repository name used to locate the model.
    root: pathlib.Path
        Resolved snapshot directory.

    Returns
    -------
    str | None
        Snapshot commit hash, or None if the directory is not recognized.

    """

    repository = "models--" + model_id.replace("/", "--")
    if (
        root.parent.name == "snapshots"
        and root.parent.parent.name == repository
        and _HF_COMMIT.fullmatch(root.name) is not None
    ):
        return root.name.lower()
    return None


def _manifest_directory(directory: str | Path | None) -> Path:
    """Resolve a persistent KíLín cache directory outside the model tree.

    Parameters
    ----------
    directory: str | pathlib.Path | None
        Explicit cache directory, or None for the platform-neutral default.

    Returns
    -------
    pathlib.Path
        Directory in which per-model cache manifests are stored.

    """

    if directory is not None:
        return Path(directory).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg and Path(xdg).is_absolute() else None
    if base is None:
        base = Path.home() / ".cache"
    return base / "kilin" / "model_identity"


def _manifest_path(root: Path, directory: Path) -> Path:
    """Map an absolute model location to its manifest filename.

    Parameters
    ----------
    root: pathlib.Path
        Resolved model directory.
    directory: pathlib.Path
        External cache directory.

    Returns
    -------
    pathlib.Path
        Cache manifest path unique to the resolved model location.

    """

    name = sha256(str(root).encode("utf-8")).hexdigest() + ".json"
    return directory / name


def _file_metadata(path: Path) -> dict[str, int]:
    """Collect file metadata used to validate a cached content digest.

    Parameters
    ----------
    path: pathlib.Path
        Current model file, with symlinks followed.

    Returns
    -------
    dict[str, int]
        Size, timestamps, device, and inode from one file stat call.

    """

    info = path.stat()
    return {
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "dev": info.st_dev,
        "ino": info.st_ino,
    }


def _cached_digest(entry: object, metadata: dict[str, int]) -> str | None:
    """Reuse a digest only for an exact, well-formed metadata match.

    Parameters
    ----------
    entry: object
        Candidate manifest record for a relative model filename.
    metadata: dict[str, int]
        Current metadata for the corresponding model file.

    Returns
    -------
    str | None
        Previously computed digest, if the cache record is acceptable.

    """

    if not isinstance(entry, dict):
        return None
    if any(
        type(entry.get(key)) is not int or entry[key] != metadata[key]
        for key in _METADATA_KEYS
    ):
        return None
    digest = entry.get("sha256")
    if not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest) is None:
        return None
    return digest


def _read_manifest(path: Path, root: Path) -> dict[str, object]:
    """Read cached file records, ignoring incomplete or invalid manifests.

    Parameters
    ----------
    path: pathlib.Path
        Per-model manifest location.
    root: pathlib.Path
        Resolved model directory the manifest must describe.

    Returns
    -------
    dict[str, object]
        Cached records, or an empty dictionary when the cache is invalid.

    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("root") != str(root):
        return {}
    files = payload.get("files")
    if not isinstance(files, dict):
        return {}
    return files


def _write_manifest(
    path: Path,
    root: Path,
    records: dict[str, dict[str, int | str]],
) -> None:
    """Atomically publish the cache, without failing model identification.

    Parameters
    ----------
    path: pathlib.Path
        Per-model cache file, outside the model tree.
    root: pathlib.Path
        Model directory whose records are being cached.
    records: dict[str, dict[str, int | str]]
        File metadata and verified SHA-256 digests to save.

    Notes
    -----
    A failed cache write merely forfeits the optimization. Each writer uses
    a unique sibling temporary file, so concurrent writers cannot publish
    a partially written JSON document.

    """

    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".model-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                {"root": str(root), "files": records}, stream,
                sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except OSError:
        # A damaged or read-only cache never prevents content verification.
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _file_digest(path: Path, metadata: dict[str, int]) -> str:
    """Hash one file and reject concurrent modifications during reading.

    Parameters
    ----------
    path: pathlib.Path
        Model file whose contents must be inspected.
    metadata: dict[str, int]
        Expected file metadata collected immediately before reading.

    Returns
    -------
    str
        SHA-256 digest of the complete file contents.

    Raises
    ------
    OSError
        If the file cannot be read or changes while it is being hashed.

    """

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_FILE_BLOCK_SIZE), b""):
            digest.update(block)
    if _file_metadata(path) != metadata:
        raise OSError(f"Model file changed while hashing: {path}")
    return digest.hexdigest()


def _directory_digest(
    root: Path,
    cache_directory: Path | None = None,
    force_verify: bool = False,
) -> str:
    """Hash a local model tree, reusing unchanged per-file content hashes.

    Parameters
    ----------
    root: pathlib.Path
        Resolved model directory.
    cache_directory: pathlib.Path | None, optional
        Persistent manifest directory. Defaults to ``None``, which
        disables caching.
    force_verify: bool, optional
        Rehash every model file instead of trusting cached metadata.
        Defaults to ``False``.

    Returns
    -------
    str
        Original location-independent SHA-256 directory digest.

    Raises
    ------
    ValueError
        If no regular model files are present.
    OSError
        If a model file cannot be inspected or read, or changes mid-read.

    Notes
    -----
    Files are read in bounded blocks. The tree digest combines JSON-encoded
    relative names and individual content hashes, preserving the prior PR 4
    model identity regardless of cache hits, misses, or cache location.

    """

    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError("The embedding model directory is empty.")

    path = (
        None if cache_directory is None else
        _manifest_path(root, cache_directory)
    )
    cached = {} if path is None else _read_manifest(path, root)
    records: dict[str, dict[str, int | str]] = {}
    digest = sha256()
    for file in files:
        name = file.relative_to(root).as_posix()
        metadata = _file_metadata(file)
        file_hash = None if force_verify else _cached_digest(
            cached.get(name), metadata
        )
        if file_hash is None:
            file_hash = _file_digest(file, metadata)
        records[name] = {**metadata, "sha256": file_hash}
        encoded = json.dumps(
            [name, file_hash], ensure_ascii=True, separators=(",", ":"),
        )
        digest.update(encoded.encode("utf-8"))

    if path is not None and records != cached:
        _write_manifest(path, root, records)
    return digest.hexdigest()
