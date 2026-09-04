"""Tests for the Unix setup launcher using temporary fake tools."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _command(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\nset -eu\n" + body,
        encoding="utf-8",
    )
    path.chmod(0o755)


@unittest.skipIf(os.name == "nt", "The Unix launcher needs sh.")
class TestSetupScript(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.temporary_root = Path(self.temporary_directory.name)
        self.project = self.temporary_root / "project with spaces"
        (self.project / "setup").mkdir(parents=True)
        shutil.copyfile(
            ROOT / "setup/setup.sh",
            self.project / "setup/setup.sh",
        )
        self.fake_bin = self.temporary_root / "bin"
        self.fake_bin.mkdir()
        for name in (
            "sh",
            "dirname",
            "mktemp",
            "rm",
            "mkdir",
            "cp",
            "chmod",
        ):
            (self.fake_bin / name).symlink_to(shutil.which(name))

        self.log = self.temporary_root / "calls.txt"
        self.uv_stub = self.temporary_root / "uv-stub"
        _command(
            self.uv_stub,
            'printf "uv:%s:%s\\n" "$PWD" "$*" '
            '>> "$QACHAT_TEST_LOG"\n'
            'exit "${QACHAT_TEST_SYNC_EXIT:-0}"\n',
        )
        shutil.copyfile(self.uv_stub, self.fake_bin / "uv")
        (self.fake_bin / "uv").chmod(0o755)
        _command(
            self.fake_bin / "uname",
            'if [ "$1" = "-s" ]; then echo "$QACHAT_TEST_OS"; '
            'else echo "$QACHAT_TEST_ARCH"; fi\n',
        )
        _command(
            self.project / ".venv/bin/python",
            'printf "python:%s\\n" "$*" >> "$QACHAT_TEST_LOG"\n'
            'exit "${QACHAT_TEST_CHECK_EXIT:-0}"\n',
        )
        installer = self.temporary_root / "fake-installer"
        _command(
            installer,
            'echo installer >> "$QACHAT_TEST_LOG"\n'
            'mkdir -p "$UV_INSTALL_DIR"\n'
            'cp "$QACHAT_TEST_UV_STUB" "$UV_INSTALL_DIR/uv"\n'
            'chmod +x "$UV_INSTALL_DIR/uv"\n',
        )
        _command(
            self.fake_bin / "curl",
            'echo curl >> "$QACHAT_TEST_LOG"\n'
            'cp "$QACHAT_TEST_INSTALLER" "$4"\n'
            'exit "${QACHAT_TEST_CURL_EXIT:-0}"\n',
        )
        self.environment = {
            **os.environ,
            "PATH": str(self.fake_bin),
            "UV_INSTALL_DIR": str(self.temporary_root / "uv install"),
            "QACHAT_TEST_LOG": str(self.log),
            "QACHAT_TEST_UV_STUB": str(self.uv_stub),
            "QACHAT_TEST_INSTALLER": str(installer),
            "QACHAT_TEST_OS": "Darwin",
            "QACHAT_TEST_ARCH": "arm64",
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/sh", str(self.project / "setup/setup.sh")],
            cwd=self.project.parent,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_supported_platforms(self) -> None:
        choices = (
            ("Darwin", "arm64"),
            ("Linux", "x86_64"),
            ("Linux", "aarch64"),
        )
        for system, machine in choices:
            with self.subTest(system=system, machine=machine):
                self.log.unlink(missing_ok=True)
                self.environment["QACHAT_TEST_OS"] = system
                self.environment["QACHAT_TEST_ARCH"] = machine
                result = self._run()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    self.log.read_text(encoding="utf-8").splitlines(),
                    [
                        "uv:" f"{self.project}:sync --extra mlx --extra torch",
                        "python:setup/check_setup.py",
                    ],
                )

    def test_unsupported_platform_stops(self) -> None:
        self.environment["QACHAT_TEST_OS"] = "Darwin"
        self.environment["QACHAT_TEST_ARCH"] = "x86_64"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())
        self.assertNotIn("Setup complete", result.stdout)

    def test_missing_uv_is_installed_first(self) -> None:
        (self.fake_bin / "uv").unlink()
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(calls[:2], ["curl", "installer"])

    def test_failure_does_not_claim_success(self) -> None:
        self.environment["QACHAT_TEST_SYNC_EXIT"] = "7"
        result = self._run()
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("python:", self.log.read_text(encoding="utf-8"))
        self.assertNotIn("Setup complete", result.stdout)


if __name__ == "__main__":
    unittest.main()
