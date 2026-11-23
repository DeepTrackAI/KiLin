"""Exercise the macOS launcher lock and runtime error handling.

Run the real Bash launcher with a mocked ``uv`` executable and a portable
``lockf`` stand-in. No native dependencies, network, or macOS GUI are needed.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
import shutil
import subprocess
import time
import unittest

from tests.setup._macos_launcher_support import LauncherSandbox


@unittest.skipUnless(
    os.name == "posix" and shutil.which("bash"),
    "The macOS launcher tests require Bash and POSIX file locks",
)
class TestMacOSLauncherLock(unittest.TestCase):
    """Keep installs bounded, serialized, and recoverable after a crash.

    """

    def setUp(self) -> None:
        """Prepare one isolated launcher with noninteractive alert shims.

        """

        self.sandbox = LauncherSandbox()
        self.support = self.sandbox.support
        self.runtime_root = self.sandbox.runtime_root
        self.runtime = self.sandbox.runtime
        self.launcher = self.sandbox.launcher
        self.uv_calls = self.sandbox.uv_calls
        self.gui_calls = self.sandbox.gui_calls
        self.osascript_calls = self.sandbox.osascript_calls
        self.started = self.sandbox.started
        self.release = self.sandbox.release
        self.environment = self.sandbox.environment

    def tearDown(self) -> None:
        """Release all temporary application files.

        """

        self.sandbox.close()

    def _run(
        self, environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Launch once with a controlled environment and no macOS dialogs.

        Parameters
        ----------
        environment: dict[str, str] or None
            Additional variables passed to the process.

        Returns
        -------
        subprocess.CompletedProcess[str]
            Launcher exit status and captured output.

        """

        return self.sandbox.run(environment)

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
        self.assertTrue((self.runtime_root / "active").is_file())
        self.assertTrue(
            (self.sandbox.release_runtime() / ".installed").is_file()
        )
        self.assertFalse((self.runtime_root / ".installing").exists())

    def test_stale_file_and_partial_runtime_are_recoverable(self) -> None:
        """A dead installer does not block a new versioned install.

        """

        self.runtime.mkdir(parents=True)
        (self.runtime / "partial").write_text("old", encoding="utf-8")
        (self.runtime_root / ".install.lock").write_text(
            "old", encoding="utf-8"
        )
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.runtime / "partial").is_file())
        self.assertTrue(
            (self.sandbox.release_runtime() / ".installed").is_file()
        )

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
        self.assertFalse(self.sandbox.release_runtime().exists())
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
            encoding="utf-8",
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
            self.assertIn(
                "already installing",
                self.osascript_calls.read_text(encoding="utf-8"),
            )
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
            encoding="utf-8",
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
                encoding="utf-8",
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
