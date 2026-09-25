"""Describe the small set of model choices offered by the desktop GUI.

The Hugging Face cache lists repositories, not whether their weights can
answer questions or create embeddings. Unknown cached models remain visible
in the model panel, but only these explicit choices enter the role menus.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

__all__ = [
    "MODEL_CHOICES",
    "ModelChoice",
    "installed_backends",
    "model_choices",
]

ModelRole = Literal["chat", "embedding"]
ModelBackend = Literal["mlx", "torch"]


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """One QAChat starting model and the role it is offered for."""

    model_id: str
    role: ModelRole
    backend: ModelBackend
    query_format: str | None = None


MODEL_CHOICES = (
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

    Cache presence is not a load test. The actual model is checked when
    QAChat opens the project.

    Parameters
    ----------
    cached_models: set[str]
        Model IDs returned by the local Hugging Face cache scan.

    Returns
    -------
    list[dict[str, object]]
        Suggested choices with role, backend, and local-cache status.

    """

    return [
        {**asdict(choice), "cached": choice.model_id in cached_models}
        for choice in MODEL_CHOICES
    ]
