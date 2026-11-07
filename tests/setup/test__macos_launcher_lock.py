"""Exercise the macOS launcher lock and runtime error handling.

Run the real Bash launcher with a mocked ``uv`` executable and a portable
``lockf`` stand-in. No native dependencies, network, or macOS GUI are needed.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest

_LAUNCHER = (
    Path(__file__).resolve().parents[2] / "setup/macos/launcher.sh"
)


def _executable(path: Path, content: str) -> None:
    """Write a small executable used by an isolated launcher test.

    Parameters
    ----------
    path: pathlib.Path
        Destination in the temporary application or tool directory.
    content: str
        Complete script, including its interpreter line.

    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


@unittest.skipUnless(
    os.name == "posix" and shutil.which("bash"),
    "The macOS launcher tests require Bash and POSIX file locks",
)
class TestMacOSLauncherLock(unittest.TestCase):
    """Keep installs bounded, serialized, and recoverable after a crash.

    """

    def setUp(self) -> None:
        """Prepare an isolated application, runtime, and executable shims.

        """

        self.temporary = TemporaryDirectory()
        root = Path(self.temporary.name)
        self.support = root / "Application Support" / "kilin"
        self.runtime_root = self.support / "runtime"
        self.runtime = self.runtime_root / "app"
        contents = root / "KíLín.app" / "Contents"
        resources = contents / "Resources"
        self.launcher = contents / "MacOS" / "KiLin"
        template = _LAUNCHER.read_text(encoding="utf-8")
        _executable(
            self.launcher,
            template.replace("@KILIN_SPEC@", "kilin[app]==0.1.0")
            .replace("@KILIN_WHEEL_SHA@", ""),
        )
        self.uv_calls = root / "uv-calls.txt"
        self.gui_calls = root / "gui-calls.txt"
        self.started = root / "install-started"
        self.release = root / "release-install"
        self.gui_script = root / "fake-gui"
        _executable(
            self.gui_script,
            "#!/bin/sh\n"
            'printf "gui\\n" >> "$KILIN_TEST_GUI_CALLS"\n',
        )
        _executable(
            resources / "uv",
            "#!/bin/sh\n"
            'printf "%s\\n" "$1" >> "$KILIN_TEST_UV_CALLS"\n'
            'if [ "$1" = "venv" ]; then\n'
            '  mkdir -p "$4/bin"\n'
            '  cp "$KILIN_TEST_GUI_SCRIPT" "$4/bin/kilin-gui"\n'
            '  if [ -n "${KILIN_TEST_RELEASE:-}" ]; then\n'
            '    : > "$KILIN_TEST_STARTED"\n'
            '    while [ ! -e "$KILIN_TEST_RELEASE" ]; do\n'
            '      sleep 0.05\n'
            '    done\n'
            '  fi\n'
            'fi\n'
            'if [ "${KILIN_TEST_FAIL_UV:-}" = "$1" ]; then\n'
            '  exit 12\n'
            'fi\n',
        )
        mock_bin = root / "bin"
        _executable(mock_bin / "uname", "#!/bin/sh\necho arm64\n")
        _executable(
            mock_bin / "lockf",
            "#!/bin/sh\n"
            f"exec {shlex.quote(sys.executable)} - \"$@\" <<'PY'\n"
            "import fcntl\n"
            "import sys\n"
            "import time\n"
            "fd = int(sys.argv[-1])\n"
            "timeout = int(sys.argv[sys.argv.index('-t') + 1])\n"
            "deadline = time.monotonic() + timeout\n"
            "while True:\n"
            "    try:\n"
            "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "        break\n"
            "    except BlockingIOError:\n"
            "        if time.monotonic() >= deadline:\n"
            "            sys.exit(75)\n"
            "        time.sleep(0.02)\n"
            "PY\n",
        )
        self.environment = {
            **os.environ,
            "KILIN_APP_SUPPORT": str(self.support),
            "KILIN_TEST_UV_CALLS": str(self.uv_calls),
            "KILIN_TEST_GUI_CALLS": str(self.gui_calls),
            "KILIN_TEST_GUI_SCRIPT": str(self.gui_script),
            "KILIN_TEST_STARTED": str(self.started),
            "PATH": str(mock_bin) + os.pathsep
            + os.environ.get("PATH", ""),
        }

    def tearDown(self) -> None:
        """Release all temporary application files.

        """

        self.temporary.cleanup()

    def _run(
        self, environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Launch once, with an optional environment override.

        Parameters
        ----------
        environment: dict[str, str] or None
            Additional variables passed to the process.

        Returns
        -------
        subprocess.CompletedProcess[str]
            Launcher exit status and captured output.

        """

        configured = self.environment.copy()
        configured.update(environment or {})
        return subprocess.run(
            [str(self.launcher)],
            env=configured,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

    def test_first_install_and_relaunch_use_one_runtime(self) -> None:
        """Keep the inode lock but skip installation after a successful run.

        """

        first = self._run()
        second = self._run()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(
            self.uv_calls.read_text(encoding="utf-8").splitlines(),
            ["venv", "pip"],
        )
        self.assertEqual(
            self.gui_calls.read_text(encoding="utf-8").splitlines(),
            ["gui", "gui"],
        )
        self.assertTrue((self.runtime_root / ".install.lock").is_file())
        self.assertFalse((self.runtime_root / ".installing").exists())

    def test_stale_file_and_partial_runtime_are_recoverable(self) -> None:
        """A dead installer leaves no kernel lock on its lock file.

        """

        self.runtime.mkdir(parents=True)
        (self.runtime / "partial").write_text("old", encoding="utf-8")
        (self.runtime_root / ".install.lock").write_text(
            "old", encoding="utf-8"
        )
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.runtime / "partial").exists())
        self.assertTrue((self.runtime / ".installed").is_file())

    def test_runtime_directory_creation_failure_is_reported(self) -> None:
        """Stop before installing if the runtime parent cannot be made.

        """

        self.support.mkdir(parents=True)
        self.runtime_root.write_text("not a directory", encoding="utf-8")
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot create its runtime folder", result.stderr)
        self.assertFalse(self.uv_calls.exists())

    def test_lock_open_failure_is_reported(self) -> None:
        """A non-file lock path cannot silently bypass serialization.

        """

        (self.runtime_root / ".install.lock").mkdir(parents=True)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot open its installation lock", result.stderr)
        self.assertFalse(self.uv_calls.exists())

    def test_invalid_timeouts_are_rejected_before_installing(self) -> None:
        """Reject malformed or unreasonably long lock waits.

        """

        for timeout in ("nope", "-1", "601", "1000000000"):
            with self.subTest(timeout=timeout):
                result = self._run({
                    "KILIN_INSTALL_LOCK_TIMEOUT": timeout,
                })
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("lock timeout", result.stderr)
                self.assertFalse(self.uv_calls.exists())

    def test_legacy_directory_requires_explicit_inspection(self) -> None:
        """Do not delete locks belonging to a previous app version.

        """

        legacy = self.runtime_root / ".installing"
        legacy.mkdir(parents=True)
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("older KíLín installation", result.stderr)
        self.assertTrue(legacy.is_dir())
        self.assertFalse(self.uv_calls.exists())
        legacy.rmdir()
        self.assertEqual(self._run().returncode, 0)

    def test_installer_failure_releases_lock_for_retry(self) -> None:
        """A failed installation must not prevent a subsequent attempt.

        """

        failed = self._run({"KILIN_TEST_FAIL_UV": "pip"})
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(self.runtime.exists())
        self.assertTrue((self.runtime_root / ".install.lock").exists())
        retried = self._run()
        self.assertEqual(retried.returncode, 0, retried.stderr)

    def test_contending_launchers_are_bounded_and_serialized(self) -> None:
        """A second launch may time out but must not run another install.

        """

        configured = self.environment.copy()
        configured["KILIN_TEST_RELEASE"] = str(self.release)
        first = subprocess.Popen(
            [str(self.launcher)], env=configured,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 5
            while not self.started.exists() and time.monotonic() < deadline:
                self.assertIsNone(first.poll())
                time.sleep(0.02)
            self.assertTrue(
                self.started.exists(), "First install did not begin"
            )
            second = self._run({"KILIN_INSTALL_LOCK_TIMEOUT": "1"})
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already installing", second.stderr)
            self.assertEqual(
                self.uv_calls.read_text(encoding="utf-8").splitlines(),
                ["venv"],
            )
        finally:
            self.release.touch()
            stdout, stderr = first.communicate(timeout=10)
        self.assertEqual(first.returncode, 0, (stdout, stderr))
        third = self._run()
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual(
            self.uv_calls.read_text(encoding="utf-8").splitlines(),
            ["venv", "pip"],
        )

    def test_waiter_reuses_completed_install(self) -> None:
        """Waiters must recheck the version after acquiring the same lock.

        """

        configured = self.environment.copy()
        configured["KILIN_TEST_RELEASE"] = str(self.release)
        first = subprocess.Popen(
            [str(self.launcher)], env=configured,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        second = None
        try:
            deadline = time.monotonic() + 5
            while not self.started.exists() and time.monotonic() < deadline:
                self.assertIsNone(first.poll())
                time.sleep(0.02)
            self.assertTrue(
                self.started.exists(), "First install did not begin"
            )
            second = subprocess.Popen(
                [str(self.launcher)], env=self.environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            time.sleep(0.15)
            self.assertIsNone(second.poll())
            self.assertEqual(
                self.uv_calls.read_text(encoding="utf-8").splitlines(),
                ["venv"],
            )
        finally:
            self.release.touch()
            first_output = first.communicate(timeout=10)
            if second is not None:
                second_output = second.communicate(timeout=10)
        self.assertEqual(first.returncode, 0, first_output)
        self.assertEqual(second.returncode, 0, second_output)
        self.assertEqual(
            self.uv_calls.read_text(encoding="utf-8").splitlines(),
            ["venv", "pip"],
        )
        self.assertEqual(
            self.gui_calls.read_text(encoding="utf-8").splitlines(),
            ["gui", "gui"],
        )


if __name__ == "__main__":
    unittest.main()
