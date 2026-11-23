"""Check the macOS launcher passes the actual installer to KíLín.

Run a preinstalled launcher with mocked architecture and application paths.
This verifies the exported repair context without macOS, Qt, or a download.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

from tests.setup._macos_launcher_support import (
    LauncherSandbox,
    executable,
)

_LAUNCHER = (
    Path(__file__).resolve().parents[2] / "setup/macos/launcher.sh"
)


class TestMacOSBackendRepair(unittest.TestCase):
    """Pass the versioned install source and bundled uv to the GUI."""

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_launcher_exports_its_original_installation_source(self) -> None:
        """Do not direct a wheel-based application to an unrelated PyPI."""

        template = _LAUNCHER.read_text(encoding="utf-8")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = root / "KíLín app" / "Contents"
            macos = app / "MacOS"
            resources = app / "Resources"
            macos.mkdir(parents=True)
            resources.mkdir()
            uv = resources / "uv"
            uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            uv.chmod(0o755)
            support = root / "Application Support" / "kilin"
            runtime = support / "runtime" / "app"
            script = runtime / "bin" / "kilin-gui"
            script.parent.mkdir(parents=True)
            script.write_text(
                '#!/bin/sh\nprintf "%s\\n%s\\n" '
                '"$KILIN_APP_UV" "$KILIN_APP_SPEC"\n',
                encoding="utf-8",
            )
            script.chmod(0o755)
            mock_bin = root / "mock-bin"
            mock_bin.mkdir()
            uname = mock_bin / "uname"
            uname.write_text(
                "#!/bin/sh\nprintf 'arm64\\n'\n", encoding="utf-8"
            )
            uname.chmod(0o755)
            # Failure-path tests must never open a native alert.
            mock_alert = mock_bin / "osascript"
            executable(mock_alert, "#!/bin/sh\nexit 0\n")
            template = template.replace(
                "/usr/bin/osascript", shlex.quote(str(mock_alert))
            )
            for spec in (
                "kilin[app]==0.1.0",
                "@RESOURCES@/kilin-0.1.0-py3-none-any.whl[app]",
            ):
                with self.subTest(spec=spec):
                    # A matching marker skips installation entirely.
                    expanded = spec.replace("@RESOURCES@", str(resources))
                    (runtime / ".installed").write_text(
                        expanded + "\n", encoding="utf-8"
                    )
                    launcher = macos / "KiLin"
                    launcher.write_text(
                        template.replace("@KILIN_SPEC@", spec).replace(
                            "@KILIN_WHEEL_SHA@", ""
                        ),
                        encoding="utf-8",
                    )
                    environment = os.environ.copy()
                    environment.update({
                        "KILIN_APP_SUPPORT": str(support),
                        "PATH": str(mock_bin) + os.pathsep
                        + environment.get("PATH", ""),
                    })
                    result = subprocess.run(
                        [str(shutil.which("bash")), str(launcher)],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                        env=environment,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        result.stdout.strip().splitlines(),
                        [str(uv), expanded],
                    )

    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_versioned_release_passes_bundled_repair_context(self) -> None:
        """The active release must inherit the app's original wheel.

        """

        spec = "@RESOURCES@/kilin-0.1.0-py3-none-any.whl[app]"
        sandbox = LauncherSandbox(spec, "b" * 64)
        try:
            installed = sandbox.run()
            self.assertEqual(installed.returncode, 0, installed.stderr)
            release = sandbox.release_runtime()
            executable(
                release / "bin/kilin-gui",
                "#!/bin/sh\n"
                'printf "%s\\n%s\\n" '
                '"$KILIN_APP_UV" "$KILIN_APP_SPEC"\n',
            )
            result = sandbox.run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.strip().splitlines(),
                [
                    str(sandbox.resources / "uv"),
                    str(sandbox.resources / "kilin-0.1.0-"
                        "py3-none-any.whl") + "[app]",
                ],
            )
            self.assertEqual(
                (sandbox.runtime_root / "active")
                .read_text(encoding="utf-8").strip(),
                "releases/" + release.name,
            )
        finally:
            sandbox.close()


if __name__ == "__main__":
    unittest.main()
