"""Unit tests for QAChat's backend and device selection policies.

Tests cover supported backend settings, platform detection, explicit MLX
restrictions, and PyTorch device priority without probing actual hardware.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest
from unittest.mock import patch

from qachat import runtime


class TestBackendConstants(unittest.TestCase):
    """Test the canonical backend identifiers and available choices."""

    def test_backend_identifiers(self) -> None:
        """Expose distinct identifiers in the documented choice order."""

        self.assertEqual(runtime.BACKEND_AUTO, "auto")
        self.assertEqual(runtime.BACKEND_MLX, "mlx")
        self.assertEqual(runtime.BACKEND_TORCH, "torch")
        self.assertEqual(
            runtime.BACKEND_CHOICES,
            (
                runtime.BACKEND_AUTO,
                runtime.BACKEND_MLX,
                runtime.BACKEND_TORCH,
            ),
        )
        self.assertEqual(
            len(runtime.BACKEND_CHOICES),
            len(set(runtime.BACKEND_CHOICES)),
        )


class TestChooseBackend(unittest.TestCase):
    """Test backend selection independently of the host operating system."""

    def test_automatic_selection(self) -> None:
        """Select MLX only for native Apple Silicon and PyTorch otherwise."""

        cases = (
            ("Darwin", "arm64", "mlx"),
            ("Darwin", "ARM64", "mlx"),
            ("Darwin", "x86_64", "torch"),
            ("Linux", "aarch64", "torch"),
            ("Linux", "x86_64", "torch"),
            ("Windows", "AMD64", "torch"),
        )
        for system, machine, expected in cases:
            with self.subTest(system=system, machine=machine):
                with (
                    patch.object(
                        runtime.platform, "system", return_value=system
                    ),
                    patch.object(
                        runtime.platform, "machine", return_value=machine
                    ),
                ):
                    self.assertEqual(
                        runtime.choose_backend(runtime.BACKEND_AUTO),
                        expected,
                    )

    def test_explicit_torch_is_allowed_on_all_platforms(self) -> None:
        """Honor an explicit PyTorch request regardless of the platform."""

        for system, machine in (
            ("Darwin", "arm64"),
            ("Darwin", "x86_64"),
            ("Linux", "x86_64"),
        ):
            with self.subTest(system=system, machine=machine):
                with (
                    patch.object(
                        runtime.platform, "system", return_value=system
                    ),
                    patch.object(
                        runtime.platform, "machine", return_value=machine
                    ),
                ):
                    self.assertEqual(
                        runtime.choose_backend(runtime.BACKEND_TORCH),
                        runtime.BACKEND_TORCH,
                    )

    def test_explicit_mlx_on_apple_silicon(self) -> None:
        """Accept MLX on native Darwin/arm64, regardless of case."""

        with (
            patch.object(runtime.platform, "system", return_value="Darwin"),
            patch.object(runtime.platform, "machine", return_value="ARM64"),
        ):
            self.assertEqual(
                runtime.choose_backend(runtime.BACKEND_MLX),
                runtime.BACKEND_MLX,
            )

    def test_explicit_mlx_on_unsupported_platform(self) -> None:
        """Reject explicit MLX requests outside native Apple Silicon."""

        for system, machine in (
            ("Darwin", "x86_64"),
            ("Linux", "arm64"),
            ("Windows", "AMD64"),
        ):
            with self.subTest(system=system, machine=machine):
                with (
                    patch.object(
                        runtime.platform, "system", return_value=system
                    ),
                    patch.object(
                        runtime.platform, "machine", return_value=machine
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, "mlx.*requires"):
                        runtime.choose_backend(runtime.BACKEND_MLX)

    def test_invalid_backend_is_rejected_before_platform_probe(self) -> None:
        """Reject invalid settings without inspecting the host platform."""

        for requested in ("", "invalid", "MLX", None, 123):
            with self.subTest(requested=requested):
                with (
                    patch.object(runtime.platform, "system") as system,
                    patch.object(runtime.platform, "machine") as machine,
                ):
                    with self.assertRaisesRegex(ValueError, "Choose"):
                        runtime.choose_backend(requested)
                system.assert_not_called()
                machine.assert_not_called()


class TestTorchDevice(unittest.TestCase):
    """Test device selection from supplied availability flags."""

    def test_device_priority(self) -> None:
        """Prefer CUDA, then MPS, then CPU for every flag combination."""

        cases = (
            (False, False, "cpu"),
            (False, True, "mps"),
            (True, False, "cuda"),
            (True, True, "cuda"),
        )
        for cuda, mps, expected in cases:
            with self.subTest(cuda=cuda, mps=mps):
                self.assertEqual(runtime.torch_device(cuda, mps), expected)


if __name__ == "__main__":
    unittest.main()
