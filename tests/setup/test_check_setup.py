"""Tests for the environment and backend calculation check."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import io
from contextlib import nullcontext
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from setup import check_setup

class TestCheckSetup(unittest.TestCase):
    def test_choose_backend(self) -> None:
        choices = (
            ("Darwin", "arm64", "auto", "mlx"),
            ("Darwin", "arm64", "torch", "torch"),
            ("Linux", "x86_64", "auto", "torch"),
            ("Linux", "aarch64", "torch", "torch"),
            ("Windows", "AMD64", "auto", "torch"),
        )
        for system, machine, requested, expected in choices:
            with self.subTest(system=system, machine=machine):
                actual = check_setup.choose_backend(
                    system,
                    machine,
                    requested,
                )
                self.assertEqual(actual, expected)

        invalid = (
            ("Darwin", "x86_64", "auto"),
            ("Windows", "ARM64", "auto"),
            ("Linux", "x86_64", "mlx"),
            ("Linux", "x86_64", "bad"),
        )
        for system, machine, requested in invalid:
            with self.subTest(system=system, machine=machine):
                with self.assertRaises(ValueError):
                    check_setup.choose_backend(
                        system,
                        machine,
                        requested,
                    )

    def test_choose_torch_device(self) -> None:
        self.assertEqual(check_setup.choose_torch_device(True, True), "cuda")
        self.assertEqual(check_setup.choose_torch_device(False, True), "mps")
        self.assertEqual(check_setup.choose_torch_device(False, False), "cpu")

    def test_run_calculation(self) -> None:
        class Number:
            def __init__(self, value: float) -> None:
                self.value = value

            def sum(self) -> Number:
                return self

            def item(self) -> float:
                return self.value

        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False)
            ),
            float32=object(),
            tensor=lambda values, **kwargs: Number(sum(values)),
        )
        mlx = SimpleNamespace(
            metal=SimpleNamespace(is_available=lambda: True),
            gpu=object(),
            float32=object(),
            stream=lambda gpu: nullcontext(),
            array=lambda values, **kwargs: Number(sum(values)),
            sum=lambda number: number,
        )
        with patch.object(
            check_setup, "import_module", side_effect=lambda name: (
                mlx if name == "mlx.core" else torch
            )
        ):
            self.assertEqual(
                check_setup.run_calculation("torch", 2, 3),
                ("cpu", 5.0),
            )
            self.assertEqual(
                check_setup.run_calculation("mlx", 2, 3),
                ("metal", 5.0),
            )
            mlx.metal.is_available = lambda: False
            with self.assertRaisesRegex(RuntimeError, "Metal"):
                check_setup.run_calculation("mlx", 2, 3)

        with self.assertRaises(ValueError):
            check_setup.run_calculation("auto", 2, 3)

    def test_check_setup(self) -> None:
        with patch.object(
            check_setup.platform, "system", return_value="Linux"
        ), patch.object(
            check_setup.platform, "machine", return_value="x86_64"
        ), patch.object(
            check_setup, "run_calculation", return_value=("cpu", 5.0)
        ) as calculate:
            report = check_setup.check_setup("auto", a=2, b=3)
        self.assertEqual(report["backend"], "torch")
        self.assertTrue(math.isclose(report["result"], 5.0))
        calculate.assert_called_once_with("torch", 2.0, 3.0)

        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    check_setup.check_setup("torch", a=value, b=1)

        with patch.object(
            check_setup.platform, "system", return_value="Linux"
        ), patch.object(
            check_setup.platform, "machine", return_value="x86_64"
        ), patch.object(
            check_setup, "run_calculation", return_value=("cpu", 10.0)
        ):
            with self.assertRaises(RuntimeError):
                check_setup.check_setup("torch", a=2, b=3)

    def test_main_reports_backend_failure(self) -> None:
        with patch.object(
            check_setup, "check_setup", side_effect=RuntimeError("failed")
        ), patch("sys.stderr", new_callable=io.StringIO) as errors:
            result = check_setup.main(["--backend", "torch"])
        self.assertEqual(result, 1)
        self.assertIn("Setup check failed: failed", errors.getvalue())

    def test_main(self) -> None:
        standard_output = io.StringIO()
        with patch(
            "setup.check_setup.check_setup",
            return_value={
                "backend": "torch",
                "device": "cpu",
                "a": 2.0,
                "b": 3.0,
                "result": 5.0,
            },
        ):
            with patch("sys.stdout", standard_output):
                result = check_setup.main(
                    ["--backend", "torch", "--a", "2", "--b", "3"]
                )
        self.assertEqual(result, 0)
        self.assertIn("2 + 3 = 5", standard_output.getvalue())


if __name__ == "__main__":
    unittest.main()
