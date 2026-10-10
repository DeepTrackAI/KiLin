"""Tests for shared MLX and PyTorch runtime selection."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

import unittest
from unittest.mock import patch

from qachat import runtime


class TestRuntime(unittest.TestCase):

    def test_choose_backend(self) -> None:
        choices = (
            ("Darwin", "arm64", "auto", "mlx"),
            ("Darwin", "arm64", "torch", "torch"),
            ("Linux", "x86_64", "auto", "torch"),
        )
        for system, machine, requested, expected in choices:
            with self.subTest(system=system, machine=machine):
                with patch.object(
                    runtime.platform,
                    "system",
                    return_value=system,
                ):
                    with patch.object(
                        runtime.platform,
                        "machine",
                        return_value=machine,
                    ):
                        self.assertEqual(
                            runtime.choose_backend(requested),
                            expected,
                        )

        with self.assertRaises(ValueError):
            runtime.choose_backend("invalid")

        with patch.object(runtime.platform, "system", return_value="Linux"):
            with patch.object(
                runtime.platform,
                "machine",
                return_value="x86_64",
            ):
                with self.assertRaises(ValueError):
                    runtime.choose_backend("mlx")

    def test_torch_device(self) -> None:
        self.assertEqual(runtime.torch_device(True, True), "cuda")
        self.assertEqual(runtime.torch_device(False, True), "mps")
        self.assertEqual(runtime.torch_device(False, False), "cpu")


if __name__ == "__main__":
    unittest.main()
