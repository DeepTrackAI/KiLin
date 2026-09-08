"""Tests for the environment and backend calculation check."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import io
import math
import platform
import unittest
from unittest.mock import patch

from setup import check_setup

APPLE_SILICON = (
    platform.system() == "Darwin" and platform.machine().lower() == "arm64"
)


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
        device, result = check_setup.run_calculation("torch", 2, 3)
        self.assertIn(device, {"cpu", "mps", "cuda"})
        self.assertTrue(math.isclose(result, 5.0))

        if APPLE_SILICON:
            device, result = check_setup.run_calculation("mlx", 2, 3)
            self.assertEqual(device, "metal")
            self.assertTrue(math.isclose(result, 5.0))

        with self.assertRaises(ValueError):
            check_setup.run_calculation("auto", 2, 3)

    def test_check_setup(self) -> None:
        report = check_setup.check_setup("torch", a=2, b=3)
        self.assertEqual(report["backend"], "torch")
        self.assertTrue(math.isclose(report["result"], 5.0))

        if APPLE_SILICON:
            report = check_setup.check_setup("mlx", a=2, b=3)
            self.assertEqual(report["backend"], "mlx")
            self.assertTrue(math.isclose(report["result"], 5.0))

        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    check_setup.check_setup("torch", a=value, b=1)

        with patch(
            "setup.check_setup.run_calculation",
            return_value=("cpu", 10.0),
        ):
            with self.assertRaises(RuntimeError):
                check_setup.check_setup("torch", a=2, b=3)

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
