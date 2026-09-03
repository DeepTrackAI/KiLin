"""Use Hugging Face's shared cache without owning model storage.

This module is a thin connection to ``huggingface_hub``. It lists cached
models and requests downloads. Hugging Face chooses and manages the cache
location.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import CacheNotFound
from huggingface_hub import scan_cache_dir
from huggingface_hub import snapshot_download

DEFAULT_MODELS: dict[str, str] = {
    "mlx": "mlx-community/Qwen3-0.6B-4bit",
    "torch": "Qwen/Qwen3-0.6B",
}


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
