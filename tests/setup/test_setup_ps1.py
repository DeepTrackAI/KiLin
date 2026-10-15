"""Tests for the Windows setup launcher when PowerShell is available."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


@unittest.skipUnless(POWERSHELL, "PowerShell is not available.")
class TestSetupPs1(unittest.TestCase):

    def test_unsupported_platform_stops_before_setup(self) -> None:
        environment = {
            **os.environ,
            "OS": "NotWindows",
            "PROCESSOR_ARCHITECTURE": "x64",
        }
        result = subprocess.run(
            [
                str(POWERSHELL),
                "-NoProfile",
                "-File",
                str(ROOT / "setup/setup.ps1"),
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Setup complete", result.stdout)


if __name__ == "__main__":
    unittest.main()
