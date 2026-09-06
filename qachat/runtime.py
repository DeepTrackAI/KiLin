"""Choose a local calculation backend and device."""

from __future__ import annotations

import platform
from typing import Literal

Backend = Literal["auto", "mlx", "torch"]
SelectedBackend = Literal["mlx", "torch"]


def choose_backend(requested: Backend) -> SelectedBackend:
    """Choose MLX or PyTorch for this computer.

    Parameters
    ----------
    requested: "auto" | "mlx" | "torch"
        Requested calculation backend.

    Returns
    -------
    Literal["mlx", "torch"]
        Selected backend.

    Raises
    ------
    ValueError
        If the value is invalid or MLX is requested on another platform.

    """

    if requested not in {"auto", "mlx", "torch"}:
        raise ValueError("Choose `auto`, `mlx`, or `torch`.")

    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine().lower() == "arm64"
    )

    if requested == "mlx" and not is_apple_silicon:
        raise ValueError("`mlx` requires a native Apple Silicon Mac.")

    if requested == "auto":
        return "mlx" if is_apple_silicon else "torch"

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
