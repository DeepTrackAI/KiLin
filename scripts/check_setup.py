"""Choose a local backend and check a tiny calculation. No model downloads."""

import argparse
from importlib import import_module
import math
import platform
import sys


def choose_backend(system: str, machine: str, requested: str = "auto") -> str:
    """Keep the platform rules separate so we can test them without any GPU."""
    if requested not in {"auto", "mlx", "torch"}:
        raise ValueError("Choose auto, mlx, or torch.")

    machine = machine.lower()
    apple_silicon = system == "Darwin" and machine == "arm64"
    supported = (
        apple_silicon
        or (system == "Linux" and machine in {"x86_64", "aarch64"})
        or (system == "Windows" and machine in {"amd64", "x86_64"})
    )
    if not supported:
        raise ValueError(
            f"Unsupported platform: {system}/{machine}. Use native Apple Silicon, "
            "Linux x64/ARM64, or Windows x64. Intel Macs/Rosetta are not supported."
        )
    if requested == "mlx" and not apple_silicon:
        raise ValueError("This project's MLX path requires an Apple Silicon Mac.")
    if requested == "auto":
        return "mlx" if apple_silicon else "torch"
    return requested


def choose_torch_device(cuda_available: bool, mps_available: bool) -> str:
    """Use an available accelerator; otherwise use the ordinary processor."""
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def run_calculation(backend: str, a: float, b: float) -> tuple[str, float]:
    """Import only the selected library, then add the two supplied numbers."""
    if backend == "mlx":
        mx = import_module("mlx.core")
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is unavailable. Try --backend torch.")
        with mx.stream(mx.gpu):
            numbers = mx.array([a, b], dtype=mx.float32)
            result = float(mx.sum(numbers).item())
        return "metal", result

    if backend == "torch":
        torch = import_module("torch")
        device = choose_torch_device(
            torch.cuda.is_available(), torch.backends.mps.is_available()
        )
        numbers = torch.tensor([a, b], dtype=torch.float32, device=device)
        result = float(numbers.sum().item())
        return device, result

    raise ValueError("The calculation needs mlx or torch, not auto.")


def check_setup(
    requested: str = "auto", *, a: float = 2.0, b: float = 3.0
) -> dict[str, str | float]:
    """This is the main flow to follow in the debugger or notebook."""
    a, b = float(a), float(b)
    expected = a + b  # Ordinary Python calculates the answer independently.
    if not all(math.isfinite(value) for value in (a, b, expected)):
        raise ValueError("Use finite numbers whose sum is also finite.")
    system = platform.system()
    machine = platform.machine()
    backend = choose_backend(system, machine, requested)  # Breakpoint 1
    device, result = run_calculation(backend, a, b)  # Breakpoint 2: Step Into here
    # Allow tiny rounding differences when using decimal numbers on the backend.
    if not math.isclose(result, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(f"The calculation returned {result}; expected {expected}.")
    return {  # Breakpoint 3: inspect the result before returning
        "python": sys.executable,
        "platform": f"{system}/{machine}",
        "backend": backend,
        "device": device,
        "a": a,
        "b": b,
        "expected": expected,
        "result": result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "mlx", "torch"), default="auto")
    parser.add_argument("--a", type=float, default=2.0, help="First number (default: 2).")
    parser.add_argument("--b", type=float, default=3.0, help="Second number (default: 3).")
    args = parser.parse_args(argv)
    try:
        report = check_setup(args.backend, a=args.a, b=args.b)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Setup check failed: {error}", file=sys.stderr)
        print("Check your numbers and backend choice. Run the setup script if packages are missing.", file=sys.stderr)
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
