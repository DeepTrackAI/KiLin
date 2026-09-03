"""Choose a local backend and check a tiny calculation. No model downloads."""

import argparse
from importlib import import_module
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


def run_calculation(backend: str) -> tuple[str, float]:
    """Import only the selected library, then calculate 1 + 2 + 3."""
    if backend == "mlx":
        mx = import_module("mlx.core")
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is unavailable. Try --backend torch.")
        with mx.stream(mx.gpu):
            numbers = mx.array([1.0, 2.0, 3.0])
            result = float(mx.sum(numbers).item())
        return "metal", result

    if backend == "torch":
        torch = import_module("torch")
        device = choose_torch_device(
            torch.cuda.is_available(), torch.backends.mps.is_available()
        )
        numbers = torch.tensor([1.0, 2.0, 3.0], device=device)
        result = float(numbers.sum().item())
        return device, result

    raise ValueError("The calculation needs mlx or torch, not auto.")


def check_setup(requested: str = "auto") -> dict[str, str | float]:
    """This is the main flow to follow in the debugger or notebook."""
    system = platform.system()
    machine = platform.machine()
    backend = choose_backend(system, machine, requested)  # Breakpoint 1
    device, result = run_calculation(backend)  # Breakpoint 2: Step Into here
    if result != 6.0:
        raise RuntimeError(f"The calculation returned {result}; expected 6.0.")
    return {  # Breakpoint 3: inspect the result before returning
        "python": sys.executable,
        "platform": f"{system}/{machine}",
        "backend": backend,
        "device": device,
        "result": result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "mlx", "torch"), default="auto")
    args = parser.parse_args(argv)
    try:
        report = check_setup(args.backend)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"Setup check failed: {error}", file=sys.stderr)
        print("Run the setup script for your operating system, then retry.", file=sys.stderr)
        return 1
    for label, value in report.items():
        print(f"{label}: {value}")
    print("Backend check passed: 1 + 2 + 3 = 6. No model was loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
