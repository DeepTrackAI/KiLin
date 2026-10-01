"""Use Hugging Face's shared cache without owning model storage.

This module provides a thin interface to ``huggingface_hub`` for discovering
and downloading model snapshots. Model files remain in Hugging Face's shared
cache, so QAChat can use locally available models without maintaining a
separate model store or duplicating cached files.

The module delegates cache layout, revision handling, and download reuse to
Hugging Face. Its responsibility is limited to exposing a small, QAChat-facing
API for inspecting cached models, locating usable local snapshots, and
requesting downloads when needed.

Key Features
------------
- **Shared-cache discovery**

    List model repositories already present in a Hugging Face cache together
    with their local paths, sizes, and cached revisions.

- **Availability checks**

    Determine whether a model ID has at least one revision in the inspected
    cache without loading the model.

- **Local snapshot lookup**

    Resolve a cached model snapshot without downloading files. Incomplete
    snapshots are accepted when the model configuration is locally available,
    leaving model-specific file validation to the downstream loader.

- **Hugging Face-managed downloads**

    Request a model snapshot through ``huggingface_hub`` while allowing the
    library to reuse existing files and manage the cache location.

Module Structure
----------------
Functions:

- `available_models(cache_directory=None) -> list[dict[str, object]]`

    List model repositories present in a Hugging Face cache.

- `model_is_available(model_id, cache_directory=None) -> bool`

    Check whether a model ID is present in a Hugging Face cache.

- `cached_model_snapshot(model_id, revision=None) -> pathlib.Path`

    Return the path to a locally cached model snapshot without downloading.

- `download_model(model_id, revision=None) -> pathlib.Path`

    Download or reuse a model snapshot through Hugging Face.

Examples
--------
List the model repositories currently available in the shared cache:

>>> models = available_models()
>>> [model["model_id"] for model in models]
[...]

Locate a cached snapshot without allowing network downloads:

>>> path = cached_model_snapshot("Qwen/Qwen3-4B")

Request a model download. Existing cached files are reused by Hugging Face:

>>> path = download_model("Qwen/Qwen3-4B")

"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import CacheNotFound, scan_cache_dir, snapshot_download
from huggingface_hub.errors import IncompleteSnapshotError

__all__ = [
    "available_models",
    "model_is_available",
    "cached_model_snapshot",
    "download_model",
]


def available_models(
    cache_directory: str | Path | None = None,
) -> list[dict[str, object]]:
    """List models in a Hugging Face cache.

    Parameters
    ----------
    cache_directory: str | pathlib.Path | None, optional
        Cache to inspect. Defaults to ``None``, which uses Hugging Face's
        normal shared cache.

    Returns
    -------
    list[dict[str, object]]
        Model IDs, local paths, sizes, and cached revisions. The result is
        sorted by model ID.

    """

    try:
        cache = scan_cache_dir(cache_dir=cache_directory)
    except CacheNotFound:
        return []

    models: list[dict[str, object]] = []
    for repository in cache.repos:

        if repository.repo_type != "model":
            continue

        revisions = sorted(
            revision.commit_hash for revision in repository.revisions
        )
        models.append(
            {
                "model_id": repository.repo_id,
                "path": str(repository.repo_path),
                "size_bytes": repository.size_on_disk,
                "revisions": revisions,
            }
        )

    return sorted(models, key=lambda model: str(model["model_id"]))


def model_is_available(
    model_id: str,
    cache_directory: str | Path | None = None,
) -> bool:
    """Check whether a model ID is already cached.

    Parameters
    ----------
    model_id: str
        Hugging Face model ID to find.
    cache_directory: str | pathlib.Path | None, optional
        Cache to inspect. Defaults to Hugging Face's normal shared cache.

    Returns
    -------
    bool
        ``True`` when the model has at least one cached revision.

    Raises
    ------
    ValueError
        If ``model_id`` is empty.

    """

    _check_model_id(model_id)

    return any(
        model["model_id"] == model_id
        for model in available_models(cache_directory)
    )


def cached_model_snapshot(
    model_id: str,
    *,
    revision: str | None = None,
) -> Path:
    """Find locally cached model files without downloading anything.

    A snapshot can be marked incomplete when only repository documentation
    is missing. Its local model files may still be usable. The model loader
    checks the files it actually needs after receiving this directory.

    """

    _check_model_id(model_id)
    try:
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    except IncompleteSnapshotError as error:
        path = error.snapshot_path
        if not (Path(path) / "config.json").is_file():
            raise
    return Path(path)


def download_model(
    model_id: str,
    *,
    revision: str | None = None,
) -> Path:
    """Ask Hugging Face to cache a model snapshot.

    Existing correct files are reused by Hugging Face.

    Parameters
    ----------
    model_id: str
        Hugging Face model ID to download.
    revision: str | None, optional
        Branch, tag, or commit. Defaults to the repository's normal revision.

    Returns
    -------
    pathlib.Path
        Cached snapshot directory.

    Raises
    ------
    ValueError
        If ``model_id`` is empty.
    RuntimeError
        If Hugging Face does not return a path.

    """

    _check_model_id(model_id)

    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
    )

    if not isinstance(path, str):
        raise RuntimeError("Hugging Face did not return a model path.")

    return Path(path)


def _check_model_id(model_id: str) -> None:
    """Reject an empty model ID.

    Parameters
    ----------
    model_id: str
        Value to check.

    Raises
    ------
    ValueError
        If ``model_id`` is empty.

    """

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Enter a Hugging Face model ID or local model path.")
