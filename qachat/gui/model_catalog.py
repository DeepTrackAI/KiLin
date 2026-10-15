"""Describe the model presets displayed by the desktop interface.

``MODEL_CHOICES`` combines core-owned model defaults with GUI presentation
records for model role, backend, and query format. These records are selection
hints, not downloaded model files or runtime compatibility checks.
``installed_backends`` consults installed package metadata; a listed backend
is not thereby proven usable on the current device.

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
import json
from pathlib import Path
from typing import Final, TypeAlias

from qachat.model_defaults import (
    DEFAULT_CHAT_MODEL_MLX,
    DEFAULT_CHAT_MODEL_TORCH,
    DEFAULT_EMBEDDING_MODEL_MLX,
    DEFAULT_EMBEDDING_MODEL_TORCH,
    MODEL_ROLE_CHAT,
    MODEL_ROLE_EMBEDDING,
    ModelRole,
)
from qachat.query_formats import QUERY_FORMAT_QWEN3, QueryFormat
from qachat.runtime import BACKEND_MLX, BACKEND_TORCH, SelectedBackend


__all__ = [
    "DEEP_CHAT_MODEL_MLX",
    "LIGHT_CHAT_MODEL_MLX",
    "cached_model_roles",
    "GUI_REQUIRED_MODEL_IDS",
    "MODEL_CHOICES",
    "ModelChoice",
    "installed_backends",
    "model_choices",
]

ModelBackend: TypeAlias = SelectedBackend

# Cached models that are neither chat nor embedding models. Only the GUI's
# cache classification uses it; qachat.model_defaults owns the model roles.
MODEL_ROLE_OTHER: Final[str] = "other"

# The GUI's smallest and largest effort models on MLX.
LIGHT_CHAT_MODEL_MLX = "Qwen/Qwen3-0.6B-MLX-4bit"
DEEP_CHAT_MODEL_MLX = "mlx-community/Qwen3-14B-4bit"


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """One QAChat starting model and the role it is offered for.

    Parameters
    ----------
    model_id: str
        Hugging Face repository identifier for this application preset.
    role: ModelRole
        Model role offered by the UI: chat or embedding.
    backend: SelectedBackend
        Backend associated with this preset: mlx or torch.
    query_format: QueryFormat | None, optional
        Embedding query format label when applicable, otherwise None. Defaults
        to ``None``.

    Attributes
    ----------
    model_id: str
        Hugging Face repository identifier for this application preset.
    role: ModelRole
        Model role offered by the UI: chat or embedding.
    backend: SelectedBackend
        Backend associated with this preset: mlx or torch.
    query_format: QueryFormat | None
        Embedding query format label when applicable, otherwise None.

    Notes
    -----
    A catalog record is an application preset, not a runtime load test. It does
    not verify a Hub repository, local cache completeness, installed backend,
    or available hardware.

    """

    model_id: str
    role: ModelRole
    backend: SelectedBackend
    query_format: QueryFormat | None = None


MODEL_CHOICES = (
    ModelChoice(LIGHT_CHAT_MODEL_MLX, MODEL_ROLE_CHAT, BACKEND_MLX),
    ModelChoice(
        DEEP_CHAT_MODEL_MLX,
        MODEL_ROLE_CHAT,
        BACKEND_MLX,
    ),
    ModelChoice(DEFAULT_CHAT_MODEL_MLX, MODEL_ROLE_CHAT, BACKEND_MLX),
    ModelChoice(
        DEFAULT_EMBEDDING_MODEL_MLX,
        MODEL_ROLE_EMBEDDING,
        BACKEND_MLX,
        QUERY_FORMAT_QWEN3,
    ),
    ModelChoice(DEFAULT_CHAT_MODEL_TORCH, MODEL_ROLE_CHAT, BACKEND_TORCH),
    ModelChoice(
        DEFAULT_EMBEDDING_MODEL_TORCH,
        MODEL_ROLE_EMBEDDING,
        BACKEND_TORCH,
        QUERY_FORMAT_QWEN3,
    ),
)

GUI_REQUIRED_MODEL_IDS = {
    BACKEND_MLX: (
        LIGHT_CHAT_MODEL_MLX,
        DEFAULT_CHAT_MODEL_MLX,
        DEEP_CHAT_MODEL_MLX,
        DEFAULT_EMBEDDING_MODEL_MLX,
    ),
    BACKEND_TORCH: (
        DEFAULT_CHAT_MODEL_TORCH,
        DEFAULT_EMBEDDING_MODEL_TORCH,
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
        BACKEND_MLX: ("mlx-embeddings", "mlx-lm"),
        BACKEND_TORCH: ("torch", "transformers"),
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


def cached_model_roles(
    cached_models: list[dict[str, object]],
) -> dict[str, str]:
    """Classify cached models as chat, embedding, or other.

    Presets keep their catalog role. Otherwise a name containing "embed" or
    a sentence-transformers configuration marks an embedding model, and a
    causal or conditional-generation architecture marks a chat model.
    Embedding models converted from chat architectures report a chat
    architecture, so names are checked first.

    Parameters
    ----------
    cached_models: list[dict[str, object]]
        Entries from ``qachat.huggingface.available_models``.

    Returns
    -------
    dict[str, str]
        Role for each cached model ID.

    """

    presets = {choice.model_id: choice.role for choice in MODEL_CHOICES}
    roles: dict[str, str] = {}
    for entry in cached_models:
        model_id = str(entry["model_id"])
        snapshots = Path(str(entry.get("path", ""))) / "snapshots"
        if model_id in presets:
            roles[model_id] = presets[model_id]
            continue
        if "embed" in model_id.lower() or any(
            snapshots.glob("*/config_sentence_transformers.json")
        ):
            roles[model_id] = MODEL_ROLE_EMBEDDING
            continue
        role = MODEL_ROLE_OTHER
        for config in snapshots.glob("*/config.json"):
            try:
                value = json.loads(config.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            architectures = value.get("architectures") or []
            if any(
                str(name).endswith(
                    ("ForCausalLM", "ForConditionalGeneration")
                )
                for name in architectures
            ):
                role = MODEL_ROLE_CHAT
            break
        roles[model_id] = role
    return roles
