"""Fast tests use small stand-ins: no real GPU, network, or model is needed."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import check_setup as setup


@pytest.mark.parametrize(
    "system,machine,requested,expected",
    [
        ("Darwin", "arm64", "auto", "mlx"),
        ("Darwin", "arm64", "mlx", "mlx"),
        ("Darwin", "arm64", "torch", "torch"),
        ("Windows", "AMD64", "auto", "torch"),
        ("Windows", "x86_64", "torch", "torch"),
        ("Linux", "x86_64", "auto", "torch"),
        ("Linux", "aarch64", "auto", "torch"),
    ],
)
def test_backend_selection(system, machine, requested, expected):
    assert setup.choose_backend(system, machine, requested) == expected


@pytest.mark.parametrize(
    "system,machine,requested",
    [
        ("Linux", "x86_64", "mlx"),
        ("Windows", "AMD64", "mlx"),
        ("Darwin", "x86_64", "auto"),
        ("Windows", "ARM64", "auto"),
        ("Linux", "i686", "auto"),
        ("FreeBSD", "x86_64", "auto"),
        ("Darwin", "arm64", "unknown"),
    ],
)
def test_unsupported_choices_have_clear_errors(system, machine, requested):
    with pytest.raises(ValueError):
        setup.choose_backend(system, machine, requested)


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_device_priority(cuda, mps, expected):
    assert setup.choose_torch_device(cuda, mps) == expected


@pytest.mark.parametrize("a,b", [(2.0, 3.0), (7.0, -2.0), (0.1, 0.2)])
def test_mlx_calculation_imports_only_mlx(monkeypatch, a, b):
    mx = Mock()
    mx.metal.is_available.return_value = True
    mx.stream.return_value = nullcontext()
    mx.sum.return_value.item.return_value = a + b
    importer = Mock(return_value=mx)
    monkeypatch.setattr(setup, "import_module", importer)

    assert setup.run_calculation("mlx", a, b) == ("metal", a + b)
    importer.assert_called_once_with("mlx.core")
    mx.array.assert_called_once_with([a, b], dtype=mx.float32)
    mx.sum.assert_called_once_with(mx.array.return_value)
    mx.stream.assert_called_once_with(mx.gpu)


def test_mlx_without_metal_fails_instead_of_silently_switching(monkeypatch):
    mx = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(setup, "import_module", lambda _: mx)
    with pytest.raises(RuntimeError, match="--backend torch"):
        setup.run_calculation("mlx", 2.0, 3.0)


@pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
@pytest.mark.parametrize("a,b", [(2.0, 3.0), (7.0, -2.0), (0.1, 0.2)])
def test_torch_calculation_uses_selected_device(monkeypatch, device, a, b):
    torch = Mock()
    torch.cuda.is_available.return_value = device == "cuda"
    torch.backends.mps.is_available.return_value = device == "mps"
    torch.tensor.return_value.sum.return_value.item.return_value = a + b
    importer = Mock(return_value=torch)
    monkeypatch.setattr(setup, "import_module", importer)

    assert setup.run_calculation("torch", a, b) == (device, a + b)
    importer.assert_called_once_with("torch")
    torch.tensor.assert_called_once_with([a, b], dtype=torch.float32, device=device)
    torch.tensor.return_value.sum.assert_called_once_with()


def test_calculation_rejects_unresolved_backend():
    with pytest.raises(ValueError, match="not auto"):
        setup.run_calculation("auto", 2.0, 3.0)


def fake_platform(monkeypatch):
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup.platform, "machine", lambda: "arm64")


def test_check_setup_reports_actual_calculation(monkeypatch):
    fake_platform(monkeypatch)
    calculation = Mock(return_value=("metal", 5.0))
    monkeypatch.setattr(setup, "run_calculation", calculation)
    report = setup.check_setup()
    assert report == {
        "python": sys.executable, "platform": "Darwin/arm64",
        "backend": "mlx", "device": "metal",
        "a": 2.0, "b": 3.0, "expected": 5.0, "result": 5.0,
    }
    calculation.assert_called_once_with("mlx", 2.0, 3.0)


@pytest.mark.parametrize("a,b,expected", [(8, 4, 12.0), (-2, 3, 1.0), (0, 0, 0.0), (1.5, 2.25, 3.75)])
def test_custom_numbers_reach_the_backend(monkeypatch, a, b, expected):
    fake_platform(monkeypatch)
    calculation = Mock(return_value=("metal", expected))
    monkeypatch.setattr(setup, "run_calculation", calculation)
    report = setup.check_setup(a=a, b=b)
    calculation.assert_called_once_with("mlx", float(a), float(b))
    assert report["expected"] == expected
    assert report["result"] == expected


def test_decimal_rounding_is_allowed(monkeypatch):
    fake_platform(monkeypatch)
    monkeypatch.setattr(setup, "run_calculation", Mock(return_value=("metal", 0.30000001192092896)))
    report = setup.check_setup(a=0.1, b=0.2)
    assert report["expected"] == 0.1 + 0.2
    assert report["result"] == 0.30000001192092896


@pytest.mark.parametrize("result", [5.0, 11.0, float("nan"), float("inf")])
def test_wrong_calculation_is_not_reported_as_success(monkeypatch, result):
    fake_platform(monkeypatch)
    monkeypatch.setattr(setup, "run_calculation", Mock(return_value=("metal", result)))
    with pytest.raises(RuntimeError, match="expected 12.0"):
        setup.check_setup(a=8, b=4)


@pytest.mark.parametrize("a,b", [(float("nan"), 3), (2, float("inf")), (float("-inf"), 3), (1e308, 1e308)])
def test_nonfinite_numbers_are_rejected_before_running_backend(monkeypatch, a, b):
    calculation = Mock()
    monkeypatch.setattr(setup, "run_calculation", calculation)
    with pytest.raises(ValueError, match="finite"):
        setup.check_setup(a=a, b=b)
    calculation.assert_not_called()


@pytest.mark.parametrize("error", [ImportError("missing package"), OSError("broken library"), RuntimeError("GPU error")])
def test_cli_errors_return_failure_and_help(monkeypatch, capsys, error):
    monkeypatch.setattr(setup, "check_setup", Mock(side_effect=error))
    assert setup.main([]) == 1
    output = capsys.readouterr()
    assert "Run the setup script" in output.err
    assert "passed" not in output.out


def test_cli_explicit_backend_and_success_message(monkeypatch, capsys):
    check = Mock(return_value={"backend": "torch", "device": "cpu", "a": 8.0, "b": 4.0, "result": 12.0})
    monkeypatch.setattr(setup, "check_setup", check)
    assert setup.main(["--backend", "torch", "--a", "8", "--b", "4"]) == 0
    check.assert_called_once_with("torch", a=8.0, b=4.0)
    assert "Backend check passed: 8 + 4 = 12." in capsys.readouterr().out


def test_cli_defaults_add_two_and_three(monkeypatch, capsys):
    check = Mock(return_value={"a": 2.0, "b": 3.0, "result": 5.0})
    monkeypatch.setattr(setup, "check_setup", check)
    assert setup.main([]) == 0
    check.assert_called_once_with("auto", a=2.0, b=3.0)
    assert "Backend check passed: 2 + 3 = 5." in capsys.readouterr().out


def test_cli_rejects_nonnumeric_input(monkeypatch, capsys):
    check = Mock()
    monkeypatch.setattr(setup, "check_setup", check)
    with pytest.raises(SystemExit) as error:
        setup.main(["--a", "hello"])
    assert error.value.code == 2
    assert "invalid float value" in capsys.readouterr().err
    check.assert_not_called()


def test_import_does_not_load_model_libraries():
    result = subprocess.run(
        [sys.executable, "-c", "from scripts import check_setup; import sys; "
         "assert 'torch' not in sys.modules; assert 'mlx.core' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_example_notebook_is_clean_and_uses_shared_code():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/setup.ipynb").read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code)
    source = "\n".join("".join(cell["source"]) for cell in code)
    assert "from scripts.check_setup import" in source
    assert "check_setup(BACKEND, a=A, b=B)" in source
    assert "expected = A + B" in source
    assert 'assert math.isclose(report["result"], expected' in source
