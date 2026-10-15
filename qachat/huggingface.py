"""Inspect and reuse Hugging Face's model cache.

This module exposes a small cache-facing API. Listing and availability checks
inspect model repositories and cached revisions. Snapshot lookup resolves a
local path, while download requests are delegated to ``huggingface_hub`` and
may contact the Hub.

Repository presence does not prove that a particular revision is complete or
loadable. Offline lookup accepts a resolved snapshot only when ``config.json``
is present. Older ``huggingface_hub`` versions can report partial snapshots
through ``IncompleteSnapshotError``; newer versions may simply return the
cached directory. QAChat supports both behaviors while leaving model-specific
file validation to the downstream loader. Custom cache directories are
accepted by listing and availability checks, not by snapshot/download wrappers.

Key Features
------------

- **Cache discovery**

    Return sorted model identifiers, repository paths, byte sizes, and revision
    hashes.

- **Offline resolution**

    Request local files only and return a snapshot directory without loading a
    model.

- **Delegated downloads**

    Let the Hub client manage cache layout, revisions, and reuse of existing
    files.

Module Structure
----------------

- ``available_models``

    Return metadata for models present in a Hugging Face cache.

- ``model_is_available``

    Return whether a model repository is present in a Hugging Face cache.

- ``cached_model_snapshot``

    Return a locally cached model snapshot without downloading files.

- ``download_model``

    Download or reuse a model snapshot in Hugging Face's shared cache.

Private helpers implement the normalization, validation, and bookkeeping
used by these operations; they are not added to the package-level API.

Examples
--------
>>> import qachat
>>> models = qachat.available_models()
>>> cached = qachat.model_is_available("organization/model")

The identifier is illustrative. Request downloads explicitly with
``qachat.download_model(model_id)``; use
``qachat.cached_model_snapshot(model_id)`` for local-only lookup.

"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from huggingface_hub import CacheNotFound, scan_cache_dir, snapshot_download
import huggingface_hub.utils.tqdm as _HubTqdm

try:
    from huggingface_hub.errors import (
        IncompleteSnapshotError as _IncompleteSnapshotError,
    )
except ImportError:

    class _IncompleteSnapshotError(Exception):
        """Compatibility placeholder for newer huggingface_hub versions."""


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
    >>> import qachat
    >>> models = qachat.available_models()
    >>> identifiers = [item["model_id"] for item in models]

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

    Notes
    -----
    This checks whether a repository identifier appears in the cache report. It
    neither selects a requested revision nor checks model-file completeness or
    loader compatibility.

    Examples
    --------
    >>> import qachat
    >>> present = qachat.model_is_available("organization/model")

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
    ``local_files_only=True``. Across supported Hub versions, QAChat accepts
    the resolved snapshot only when ``config.json`` is available. Older Hub
    versions may signal a partial snapshot with ``IncompleteSnapshotError``;
    newer versions can return the cached snapshot directory directly.

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
    FileNotFoundError
        If Hugging Face resolves a local snapshot without ``config.json``.

    Notes
    -----
    No custom cache-directory argument is exposed here. ``config.json`` is
    the only QAChat-level completeness requirement; a returned directory may
    still lack weights or tokenizer files. Other Hub and filesystem errors
    propagate.

    Examples
    --------
    For an actual model identifier already present in the configured shared
    cache:

    >>> import qachat
    >>> path = qachat.cached_model_snapshot("organization/model")

    """

    _check_model_id(model_id)

    try:
        path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    except _IncompleteSnapshotError as error:
        path = error.snapshot_path

    snapshot = Path(path)

    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(
            f"Cached model snapshot has no config.json: {snapshot}"
        )

    return snapshot


def download_model(
    model_id: str,
    *,
    revision: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Download or reuse a model snapshot in Hugging Face's shared cache.

    The download is delegated to ``huggingface_hub.snapshot_download``. Hugging
    Face determines the cache location and reuses valid files already present
    in the cache. QAChat does not copy or take ownership of the model files.

    Parameters
    ----------
    model_id: str
        Hugging Face model repository ID to download.
    revision: str | None, optional
        Branch, tag, or commit to download. Defaults to ``None``, which uses
        the repository's default revision.
    progress_callback: Callable[[int, int], None] | None, optional
        Callback receiving completed and expected snapshot byte counts. The
        expected total can initially be zero while Hugging Face discovers
        repository files. Defaults to ``None``.

    Returns
    -------
    pathlib.Path
        Directory containing the downloaded or reused model snapshot.

    Raises
    ------
    ValueError
        If ``model_id`` is not a non-empty string.
        Also raised when ``progress_callback`` is not callable.
    RuntimeError
        If Hugging Face does not return the snapshot path as a string.

    Notes
    -----
    The identifier in this example is a placeholder. The Hub client determines
    download behavior and may access the network. This wrapper does not load a
    model or validate backend compatibility.

    Examples
    --------
    For an actual accessible repository identifier, permit a Hub download:

    >>> import qachat
    >>> path = qachat.download_model("organization/model")

    """

    _check_model_id(model_id)

    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None.")

    # Without a callback, the Hub client keeps its own progress display.
    options: dict[str, object] = {}
    if progress_callback is not None:
        options["tqdm_class"] = _progress_tqdm(progress_callback)

    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        **options,
    )

    if not isinstance(path, str):
        raise RuntimeError("Hugging Face did not return a model path.")

    return Path(path)


def _progress_tqdm(
    callback: Callable[[int, int], None],
) -> type[_HubTqdm]:
    """Build a silent Hub progress class that reports aggregate bytes.

    The Hub client creates several progress bars during a snapshot
    download. Only the aggregate byte bar (described as "Reconstructing"
    with unit ``B``) reports to ``callback``; the others stay silent.

    Parameters
    ----------
    callback: collections.abc.Callable[[int, int], None]
        Receives completed and expected byte counts.

    Returns
    -------
    type[huggingface_hub.utils.tqdm]
        Progress class to pass to ``snapshot_download`` as ``tqdm_class``.

    Notes
    -----
    This depends on the Hub client's description of its aggregate bar. If
    a later version renames it, downloads still work but report no bytes.

    """

    class DownloadProgress(_HubTqdm):
        """Forward aggregate snapshot completion without terminal output."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Disable terminal output and report the starting counts.

            Parameters
            ----------
            *args: object
                Additional positional arguments forwarded to the sink.
            **kwargs: object
                Additional keyword arguments forwarded to the sink.

            """

            self._qachat_reports_bytes = (
                str(kwargs.get("desc", "")).startswith("Reconstructing")
                and kwargs.get("unit") == "B"
            )
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self._qachat_report()

        def update(self, n: int | float | None = 1) -> bool | None:
            """Advance the counter by ``n`` and report the new counts.

            Parameters
            ----------
            n: int | float | None, optional
                Number of progress steps completed. Defaults to ``1``.

            Returns
            -------
            bool | None
                The progress-bar update status.

            """

            # tqdm skips its counter entirely while disabled. Keep the counter
            # ourselves so GUI downloads stay silent but still expose bytes.
            if self.disable:
                self.n += n or 0
                self._qachat_report()
                return True

            result = super().update(n)
            self._qachat_report()
            return result

        def _qachat_report(self) -> None:
            """Send completed and expected bytes for the aggregate bar."""

            if self._qachat_reports_bytes:
                callback(int(self.n), int(self.total or 0))

    return DownloadProgress


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
