"""Describe the model presets displayed by the desktop interface.

``MODEL_CHOICES`` stores application presets for model role, backend, and
query format. These records are selection hints, not downloaded model files
or runtime compatibility checks. ``installed_backends`` consults installed
package metadata; a listed backend is not thereby proven usable on the
current device.

Key Features
------------

- **Catalog records**

    Represent a model preset using immutable ``ModelChoice`` fields.

- **Catalog availability**

    Return all presets with cache flags; the web interface filters choices by
    role and backend.

- **Installation hints**

    Expose distribution-presence checks without loading an inference model.

Module Structure
----------------

- ``MODEL_CHOICES``

    Tuple of GUI model presets, not a runtime availability check.

- ``ModelChoice``

    One QAChat starting model and the role it is offered for.

- ``installed_backends``

    Check whether the selected computation packages are installed.

- ``model_choices``

    Mark the suggested models that are present in the local cache.

Examples
--------
>>> import qachat.gui.model_catalog
>>> backends = qachat.gui.model_catalog.installed_backends()
>>> presets = qachat.gui.model_catalog.MODEL_CHOICES

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

__all__ = [
    "DEFAULT_MODEL_IDS",
    "MODEL_CHOICES",
    "ModelChoice",
    "installed_backends",
    "model_choices",
]

ModelRole = Literal["chat", "embedding"]
ModelBackend = Literal["mlx", "torch"]


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """One QAChat starting model and the role it is offered for.

    Parameters
    ----------
    model_id: str
        Hugging Face repository identifier for this application preset.
    role: ModelRole
        Model role offered by the UI: chat or embedding.
    backend: ModelBackend
        Backend associated with this preset: mlx or torch.
    query_format: str | None, optional
        Embedding query format label when applicable, otherwise None. Defaults
        to ``None``.

    Attributes
    ----------
    model_id: str
        Hugging Face repository identifier for this application preset.
    role: ModelRole
        Model role offered by the UI: chat or embedding.
    backend: ModelBackend
        Backend associated with this preset: mlx or torch.
    query_format: str | None
        Embedding query format label when applicable, otherwise None.

    Notes
    -----
    A catalog record is an application preset, not a runtime load test. It does
    not verify a Hub repository, local cache completeness, installed backend,
    or available hardware.

    """

    model_id: str
    role: ModelRole
    backend: ModelBackend
    query_format: str | None = None


MODEL_CHOICES = (
    ModelChoice("mlx-community/Qwen3-14B-4bit", "chat", "mlx"),
    ModelChoice(
        "lukaskremla/Qwen3.8-27B-4bit-MLX-TextOnly",
        "chat",
        "mlx",
    ),
    ModelChoice("Qwen/Qwen3-0.6B-MLX-4bit", "chat", "mlx"),
    ModelChoice(
        "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        "embedding",
        "mlx",
        "qwen3",
    ),
    ModelChoice("Qwen/Qwen3-0.6B", "chat", "torch"),
    ModelChoice("Qwen/Qwen3-Embedding-0.6B", "embedding", "torch", "qwen3"),
)

DEFAULT_MODEL_IDS = {
    "mlx": (
        "Qwen/Qwen3-0.6B-MLX-4bit",
        "mlx-community/Qwen3-14B-4bit",
        "lukaskremla/Qwen3.8-27B-4bit-MLX-TextOnly",
        "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    ),
    "torch": (
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-Embedding-0.6B",
    ),
}


def installed_backends() -> dict[str, bool]:
    """Check whether the selected computation packages are installed.

    This checks packages, not model weights or device compatibility.

    Returns
    -------
    dict[str, bool]
        Installation status for the MLX and PyTorch backends.

    """

    required = {
        "mlx": ("mlx-embeddings", "mlx-lm"),
        "torch": ("torch", "transformers"),
    }
    result: dict[str, bool] = {}
    for backend, packages in required.items():
        try:
            for package in packages:
                version(package)
        except PackageNotFoundError:
            result[backend] = False
        else:
            result[backend] = True
    return result


def model_choices(cached_models: set[str]) -> list[dict[str, object]]:
    """Mark the suggested models that are present in the local cache.

    Cache presence is not a load test. The actual model is checked when QAChat
    opens the project.

    Parameters
    ----------
    cached_models: set[str]
        Model IDs returned by the local Hugging Face cache scan.

    Returns
    -------
    list[dict[str, object]]
        Suggested choices with role, backend, and local-cache status.

    Notes
    -----
    Returns every preset with a cached flag derived from the supplied
    identifier set. Filtering by role/backend/query format is performed by the
    frontend, not this function.

    """

    return [
        {**asdict(choice), "cached": choice.model_id in cached_models}
        for choice in MODEL_CHOICES
    ]
