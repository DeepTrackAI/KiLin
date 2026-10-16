"""Select and verify the local calculation backend.

This auxiliary repository module checks that the local KíLín development
setup can use an appropriate numerical backend. It is intended for setup and
development workflows and is not part of the released ``kilin`` package.

The module selects MLX on native Apple Silicon and PyTorch on supported Linux
and Windows systems unless a backend is requested explicitly. It then performs
a small numerical calculation and compares the result with ordinary Python.
No language model is loaded during this check.

Key Features
------------
- **Platform-aware backend selection**

    Select MLX on Apple Silicon and PyTorch on supported Linux and Windows
    platforms, while allowing an explicit backend request.

- **PyTorch device selection**

    Prefer CUDA, then Apple MPS, and finally CPU when PyTorch is used.

- **Minimal backend verification**

    Perform a simple addition with the selected backend and verify the result
    against the corresponding Python calculation.

- **Command-line execution**

    Run the setup check directly and report the interpreter, platform,
    backend, device, inputs, and result.

Module Structure
----------------
Type aliases:

- ``Backend``

    Backend request accepted by the setup check: ``auto``, ``mlx``, or
    ``torch``.

- ``SelectedBackend``

    Concrete backend selected for a calculation: ``mlx`` or ``torch``.

Functions:

- ``choose_backend(system, machine, requested='auto') -> SelectedBackend``

    Choose the calculation backend for a supported platform.

- ``choose_torch_device(cuda_available, mps_available) -> str``

    Choose the best available PyTorch device.

- ``run_calculation(backend, a, b) -> tuple[str, float]``

    Add two numbers using the selected backend.

- ``check_setup(requested='auto', *, a=2.0, b=3.0) -> dict[str, str | float]``

    Select a backend and verify a simple numerical calculation.

- ``main(argv=None) -> int``

    Run the setup check from the command line.

Examples
--------
Run the default platform-aware setup check from the repository::

    python check_setup.py

Request PyTorch explicitly::

    python check_setup.py --backend torch

"""

from __future__ import annotations

import argparse
from importlib import import_module
import math
import platform
import sys
from typing import Literal

Backend = Literal["auto", "mlx", "torch"]
SelectedBackend = Literal["mlx", "torch"]


def choose_backend(
    system: str,
    machine: str,
    requested: Backend = "auto",
) -> SelectedBackend:
    """Choose an available calculation engine.

    Parameters
    ----------
    system: str
        Operating-system name reported by Python.
    machine: str
        Processor name reported by Python.
    requested: "auto" | "mlx" | "torch", optional
        ``auto``, ``mlx``, or ``torch``. Defaults to ``auto``.

    Returns
    -------
    Literal["mlx", "torch"]
        Selected engine.

    Raises
    ------
    ValueError
        If the platform or requested engine is not supported.

    """

    if requested not in {"auto", "mlx", "torch"}:
        raise ValueError("Choose `auto`, `mlx`, or `torch`.")

    machine = machine.lower()
    is_apple_silicon = system == "Darwin" and machine == "arm64"
    supported = (
        is_apple_silicon
        or system == "Linux"
        and machine in {"x86_64", "aarch64"}
        or system == "Windows"
        and machine in {"amd64", "x86_64"}
    )

    if not supported:
        raise ValueError(
            f"Unsupported platform: {system}/{machine}. Use Apple Silicon, "
            "Linux x64/ARM64, or Windows x64."
        )

    if requested == "mlx" and not is_apple_silicon:
        raise ValueError("`mlx` requires a native Apple Silicon Mac.")

    if requested == "auto":
        return "mlx" if is_apple_silicon else "torch"

    return requested


def choose_torch_device(
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


def run_calculation(
    backend: SelectedBackend,
    a: float,
    b: float,
) -> tuple[str, float]:
    """Add two numbers using the selected engine.

    Parameters
    ----------
    backend: "mlx" | "torch"
        ``mlx`` or ``torch``.
    a: float
        First number.
    b: float
        Second number.

    Returns
    -------
    tuple[str, float]
        Device name and calculated sum.

    Raises
    ------
    ImportError
        If the selected backend package cannot be imported.
    RuntimeError
        If MLX is selected but Metal is unavailable.
    ValueError
        If ``backend`` is not ``mlx`` or ``torch``.

    """

    if backend == "mlx":
        mx = import_module("mlx.core")

        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is unavailable.")

        with mx.stream(mx.gpu):
            numbers = mx.array([a, b], dtype=mx.float32)
            result = float(mx.sum(numbers).item())

        return "metal", result

    if backend == "torch":
        torch = import_module("torch")

        device = choose_torch_device(
            torch.cuda.is_available(),
            torch.backends.mps.is_available(),
        )

        numbers = torch.tensor(
            [a, b],
            dtype=torch.float32,
            device=device,
        )

        return device, float(numbers.sum().item())

    raise ValueError("The calculation needs `mlx` or `torch`, not `auto`.")


def check_setup(
    requested: Backend = "auto",
    *,
    a: float = 2.0,
    b: float = 3.0,
) -> dict[str, str | float]:
    """Check one engine by comparing its sum with ordinary Python.

    Parameters
    ----------
    requested: "auto" | "mlx" | "torch", optional
        ``auto``, ``mlx``, or ``torch``. Defaults to ``auto``.
    a: float, optional
        First number. Defaults to ``2.0``.
    b: float, optional
        Second number. Defaults to ``3.0``.

    Returns
    -------
    dict[str, str | float]
        Environment, backend, device, input, and calculation details.

    Raises
    ------
    ImportError
        If the selected backend package cannot be imported.
    RuntimeError
        If the backend cannot perform the calculation correctly.
    ValueError
        If the requested backend, platform, or numerical inputs are invalid.

    """

    first = float(a)
    second = float(b)
    expected = first + second

    if not all(math.isfinite(value) for value in (first, second, expected)):
        raise ValueError("Use finite numbers whose sum is also finite.")

    system = platform.system()
    machine = platform.machine()
    backend = choose_backend(system, machine, requested)
    device, result = run_calculation(backend, first, second)

    if not math.isclose(result, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(
            f"The calculation returned {result}; expected {expected}."
        )

    return {
        "python": sys.executable,
        "platform": f"{system}/{machine}",
        "backend": backend,
        "device": device,
        "a": first,
        "b": second,
        "expected": expected,
        "result": result,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the setup check from a terminal.

    Parameters
    ----------
    argv: list[str] | None, optional
        Optional command-line arguments used by tests. Defaults
        to ``None``, which reads the command-line arguments.

    Returns
    -------
    int
        ``0`` after success and ``1`` after a handled error.

    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("auto", "mlx", "torch"),
        default="auto",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=2.0,
        help="First number (default: 2).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=3.0,
        help="Second number (default: 3).",
    )
    arguments = parser.parse_args(argv)

    try:
        report = check_setup(
            arguments.backend,
            a=arguments.a,
            b=arguments.b,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Setup check failed: {error}", file=sys.stderr)
        return 1

    for label, value in report.items():
        print(f"{label}: {value}")

    print(
        f"Backend check passed: {report['a']:g} + {report['b']:g} = "
        f"{report['result']:g}. No model was loaded."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
