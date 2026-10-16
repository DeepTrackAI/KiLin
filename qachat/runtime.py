"""Choose backend and device names without loading inference models.

Automatic backend selection returns MLX on native Darwin/arm64 and PyTorch
otherwise. An explicit MLX request is rejected outside that platform. These
checks do not verify that dependencies are installed or that a model is
compatible. Device selection uses caller-supplied availability flags in CUDA,
MPS, then CPU priority order; it does not probe the hardware itself.

Key Features
------------

- **Backend policy**

    Centralize the string-valued backend selection used by model sessions.

- **Device priority**

    Choose a PyTorch device from already-probed availability flags.

Module Structure
----------------

- ``BACKEND_AUTO``, ``BACKEND_MLX``, and ``BACKEND_TORCH``

    Canonical string values for KíLín calculation backends.

- ``BACKEND_CHOICES``

    Ordered collection of accepted backend settings.

- ``Backend``

    Type alias for ``"auto"``, ``"mlx"``, or ``"torch"``.

- ``SelectedBackend``

    Type alias for a resolved ``"mlx"`` or ``"torch"`` backend.

- ``choose_backend``

    Choose MLX or PyTorch for this computer.

- ``torch_device``

    Choose the best available PyTorch device.

Examples
--------
>>> import kilin
>>> backend = kilin.choose_backend(kilin.BACKEND_AUTO)
>>> device = kilin.torch_device(False, False)

"""

from __future__ import annotations

import platform
from typing import Final, Literal, TypeAlias

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_MLX",
    "BACKEND_TORCH",
    "BACKEND_CHOICES",
    "Backend",
    "SelectedBackend",
    "choose_backend",
    "torch_device",
]

Backend: TypeAlias = Literal["auto", "mlx", "torch"]
SelectedBackend: TypeAlias = Literal["mlx", "torch"]

BACKEND_AUTO: Final[Backend] = "auto"
BACKEND_MLX: Final[SelectedBackend] = "mlx"
BACKEND_TORCH: Final[SelectedBackend] = "torch"
BACKEND_CHOICES: Final[tuple[Backend, ...]] = (
    BACKEND_AUTO,
    BACKEND_MLX,
    BACKEND_TORCH,
)


def choose_backend(requested: Backend) -> SelectedBackend:
    """Choose MLX or PyTorch for this computer.

    Parameters
    ----------
    requested: "auto" | "mlx" | "torch"
        Requested calculation backend.

    Returns
    -------
    "mlx" | "torch"
        Selected backend.

    Raises
    ------
    ValueError
        If the value is invalid or MLX is requested on another platform.

    """

    if requested not in BACKEND_CHOICES:
        raise ValueError("Choose `auto`, `mlx`, or `torch`.")

    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine().lower() == "arm64"
    )

    if requested == BACKEND_MLX and not is_apple_silicon:
        raise ValueError("`mlx` requires a native Apple Silicon Mac.")

    if requested == BACKEND_AUTO:
        return BACKEND_MLX if is_apple_silicon else BACKEND_TORCH

    return requested


def torch_device(
    cuda_available: bool,
    mps_available: bool,
) -> str:
    """Choose the best available PyTorch device.

    Parameters
    ----------
    cuda_available: bool
        Whether a CUDA device is available.
    mps_available: bool
        Whether an Apple MPS device is available.

    Returns
    -------
    str
        ``cuda``, ``mps``, or ``cpu``.

    """

    if cuda_available:
        return "cuda"

    if mps_available:
        return "mps"

    return "cpu"
