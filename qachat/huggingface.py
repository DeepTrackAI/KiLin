"""Use Hugging Face's shared cache for QAChat model snapshots.

This module provides a small interface to ``huggingface_hub`` for discovering,
locating, and downloading model snapshots. Model files remain in Hugging
Face's shared cache, allowing QAChat to use locally available models without
maintaining a separate model store or duplicating cached files.

Cache ownership remains entirely with Hugging Face. In particular, this module
does not choose the cache layout, copy model files into QAChat, or implement
revision management. It only exposes the operations QAChat needs to inspect
the cache, resolve a local snapshot, and request a download.

Key Features
------------
- **Shared-cache discovery**

    List cached model repositories together with their local repository paths,
    sizes, and cached commit revisions.

- **Availability checks**

    Check whether a model repository has at least one revision in a Hugging
    Face cache without loading the model.

- **Local snapshot lookup**

    Resolve a model snapshot using local files only. A snapshot reported as
    incomplete by Hugging Face can still be returned when ``config.json`` is
    present, leaving model-specific validation to the downstream loader.

- **Hugging Face-managed downloads**

    Request a model snapshot through ``huggingface_hub`` while relying on
    Hugging Face to reuse existing files and manage the cache location.

Module Structure
----------------
Functions:

- `available_models(cache_directory=None) -> list[dict[str, object]]`

    Return metadata for model repositories present in a Hugging Face cache.

- `model_is_available(model_id, cache_directory=None) -> bool`

    Return whether a model repository is present in a Hugging Face cache.

- `cached_model_snapshot(model_id, revision=None) -> pathlib.Path`

    Resolve a locally cached model snapshot without downloading files.

- `download_model(model_id, revision=None) -> pathlib.Path`

    Download or reuse a model snapshot through Hugging Face.

Examples
--------
List model repositories available in the default Hugging Face cache:

>>> models = qachat.available_models()
>>> [model["model_id"] for model in models]
[...]

Check whether a particular model repository is cached:

>>> qachat.model_is_available("Qwen/Qwen3-4B")
True

Resolve a cached snapshot without allowing a network download:

>>> path = qachat.cached_model_snapshot("Qwen/Qwen3-4B")

Request a model snapshot. Hugging Face reuses files already in its cache:

>>> path = qachat.download_model("Qwen/Qwen3-4B")

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
    """Return metadata for models present in a Hugging Face cache.

    Only repositories whose Hugging Face repository type is ``model`` are
    included. Each returned dictionary contains the model ID, repository path,
    size on disk, and the commit hashes of all cached revisions.

    Parameters
    ----------
    cache_directory: str | pathlib.Path | None, optional
        Hugging Face cache directory to inspect. Defaults to ``None``, which
        uses the cache selected by ``huggingface_hub``.

    Returns
    -------
    list[dict[str, object]]
        Metadata for cached model repositories, sorted by model ID. Each
        dictionary contains ``model_id``, ``path``, ``size_bytes``, and
        ``revisions``. Returns an empty list when the cache does not exist.

    Examples
    --------
    >>> models = qachat.available_models()
    >>> [model["model_id"] for model in models]
    [...]

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
    """Return whether a model repository is present in a Hugging Face cache.

    Availability means that the cache contains a model repository with the
    exact requested model ID. This function does not inspect whether a
    particular revision is complete or whether a downstream loader can use it.

    Parameters
    ----------
    model_id: str
        Hugging Face model repository ID to find.
    cache_directory: str | pathlib.Path | None, optional
        Hugging Face cache directory to inspect. Defaults to ``None``, which
        uses the cache selected by ``huggingface_hub``.

    Returns
    -------
    bool
        ``True`` if the model repository is present; otherwise ``False``.

    Raises
    ------
    ValueError
        If ``model_id`` is not a non-empty string.

    Examples
    --------
    >>> qachat.model_is_available("Qwen/Qwen3-4B")
    True

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
    """Return a locally cached model snapshot without downloading files.

    The lookup is delegated to ``huggingface_hub.snapshot_download`` with
    ``local_files_only=True``. Hugging Face can mark a snapshot incomplete
    even when only auxiliary repository files are absent. In that case, this
    function accepts the snapshot when ``config.json`` is present and leaves
    model-specific file validation to the downstream model loader.

    Parameters
    ----------
    model_id: str
        Hugging Face model repository ID to resolve.
    revision: str | None, optional
        Branch, tag, or commit to resolve. Defaults to ``None``, which uses the
        repository's default revision.

    Returns
    -------
    pathlib.Path
        Directory containing the resolved local model snapshot.

    Raises
    ------
    ValueError
        If ``model_id`` is not a non-empty string.
    huggingface_hub.errors.IncompleteSnapshotError
        If Hugging Face reports an incomplete snapshot and ``config.json`` is
        not available in that snapshot.

    Examples
    --------
    >>> path = qachat.cached_model_snapshot("Qwen/Qwen3-4B")

    Resolve a specific revision already present in the local cache:

    >>> path = qachat.cached_model_snapshot("Qwen/Qwen3-4B", revision="main")

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
    """Download or reuse a model snapshot in Hugging Face's shared cache.

    The download is delegated to ``huggingface_hub.snapshot_download``.
    Hugging Face determines the cache location and reuses valid files already
    present in the cache. QAChat does not copy or take ownership of the model
    files.

    Parameters
    ----------
    model_id: str
        Hugging Face model repository ID to download.
    revision: str | None, optional
        Branch, tag, or commit to download. Defaults to ``None``, which uses
        the repository's default revision.

    Returns
    -------
    pathlib.Path
        Directory containing the downloaded or reused model snapshot.

    Raises
    ------
    ValueError
        If ``model_id`` is not a non-empty string.
    RuntimeError
        If Hugging Face does not return the snapshot path as a string.

    Examples
    --------
    >>> path = qachat.download_model("Qwen/Qwen3-4B")

    Download a specific revision:

    >>> path = qachat.download_model("Qwen/Qwen3-4B", revision="main")

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
    """Validate a Hugging Face model repository ID.

    Parameters
    ----------
    model_id: str
        Model repository ID to validate.

    Raises
    ------
    ValueError
        If ``model_id`` is not a non-empty string.

    """

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Enter a Hugging Face model ID or local model path.")
