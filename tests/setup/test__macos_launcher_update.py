"""Verify atomic macOS runtime updates and safe recovery after failures.

Run the production launcher with isolated installation, locking, and alert
shims. No user approval prompt, network, GUI, or model download is needed.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time
import unittest

from tests.setup._macos_launcher_support import LauncherSandbox


@unittest.skipUnless(
    os.name == "posix" and shutil.which("bash"),
    "The macOS launcher tests require Bash and POSIX locks",
)
class TestMacOSLauncherUpdate(unittest.TestCase):
    """Preserve existing runtimes until activation has committed.

    """

    def setUp(self) -> None:
        """Create a clean, noninteractive launcher fixture.

        """

        self.sandbox = LauncherSandbox()
        self.runtime_root = self.sandbox.runtime_root
        self.active = self.runtime_root / "active"

    def tearDown(self) -> None:
        """Remove the temporary installation tree.

        """

        self.sandbox.close()

    def _install(self, version: str) -> Path:
        """Install a version and return its permanent environment path.

        Parameters
        ----------
        version: str
            Version of the distribution requested by this app.

        Returns
        -------
        pathlib.Path
            Environment created for the requested version.

        """

        self.sandbox.set_launcher(f"kilin[app]=={version}")
        result = self.sandbox.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        return self.sandbox.release_runtime()

    def test_first_install_and_relaunch_use_stable_runtime(self) -> None:
        """A release runs where its virtual environment was created.

        """

        release = self._install("0.1.0")
        selected = self.active.read_text(encoding="utf-8").strip()
        self.assertEqual(selected, "releases/" + release.name)
        self.assertEqual(
            self.sandbox.venv_paths.read_text(encoding="utf-8").strip(),
            str(release),
        )
        self.assertTrue((release / "bin/kilin-gui").is_file())
        self.assertTrue((release / "bin/python").is_file())
        self.assertTrue((release / ".installed").is_file())
        self.assertEqual(self.sandbox.run().returncode, 0)
        self.assertEqual(
            self.sandbox.uv_calls.read_text(encoding="utf-8")
            .splitlines(),
            ["venv", "pip"],
        )

    def test_successful_upgrade_preserves_prior_environment(self) -> None:
        """Changing the active pointer never relocates an older venv.

        """

        old = self._install("0.1.0")
        new = self._install("0.2.0")
        self.assertNotEqual(old, new)
        self.assertTrue((old / ".installed").is_file())
        self.assertTrue((old / "bin/kilin-gui").is_file())
        self.assertTrue((new / ".installed").is_file())
        self.assertEqual(
            self.active.read_text(encoding="utf-8").strip(),
            "releases/" + new.name,
        )
        self.assertEqual(
            self.sandbox.venv_paths.read_text(encoding="utf-8")
            .splitlines(),
            [str(old), str(new)],
        )

    def test_failed_update_preserves_active_runtime(self) -> None:
        """Failed package installation must not start the previous app.

        """

        old = self._install("0.1.0")
        before = self.active.read_bytes()
        self.sandbox.set_launcher("kilin[app]==0.2.0")
        new = self.sandbox.release_runtime()
        result = self.sandbox.run({"KILIN_TEST_FAIL_UV": "pip"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not be installed", result.stderr)
        self.assertEqual(self.active.read_bytes(), before)
        self.assertTrue((old / ".installed").is_file())
        self.assertFalse(new.exists())
        self.assertEqual(
            self.sandbox.gui_calls.read_text(encoding="utf-8")
            .splitlines(),
            ["gui"],
        )
        self.assertIn(
            "could not be installed",
            self.sandbox.osascript_calls.read_text(encoding="utf-8"),
        )
        self.assertEqual(self.sandbox.run().returncode, 0)
        self.assertTrue((new / ".installed").is_file())

    def test_failed_venv_creation_preserves_previous_release(self) -> None:
        """An early installer failure removes only the new attempt.

        """

        old = self._install("0.1.0")
        before = self.active.read_bytes()
        self.sandbox.set_launcher("kilin[app]==0.2.0")
        failed = self.sandbox.run({"KILIN_TEST_FAIL_UV": "venv"})
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(self.active.read_bytes(), before)
        self.assertTrue(old.exists())
        self.assertFalse(self.sandbox.release_runtime().exists())

    def test_installation_validation_failure_rolls_back(self) -> None:
        """A release with failing installed metadata is never selected.

        """

        old = self._install("0.1.0")
        before = self.active.read_bytes()
        self.sandbox.set_launcher("kilin[app]==0.2.0")
        failed = self.sandbox.run({
            "KILIN_TEST_FAIL_VALIDATION": "1",
        })
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("validation failed", failed.stderr)
        self.assertEqual(self.active.read_bytes(), before)
        self.assertTrue(old.exists())
        self.assertFalse(self.sandbox.release_runtime().exists())

    def test_failed_activation_preserves_old_pointer_and_new_release(
        self,
    ) -> None:
        """A failed atomic rename is recoverable without reinstalling.

        """

        old = self._install("0.1.0")
        before = self.active.read_bytes()
        self.sandbox.set_launcher("kilin[app]==0.2.0")
        failed = self.sandbox.run({"KILIN_TEST_FAIL_ACTIVATE": "1"})
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("cannot activate", failed.stderr)
        self.assertEqual(self.active.read_bytes(), before)
        self.assertTrue(old.is_dir())
        new = self.sandbox.release_runtime()
        self.assertTrue((new / ".installed").is_file())
        calls = self.sandbox.uv_calls.read_bytes()
        self.assertEqual(self.sandbox.run().returncode, 0)
        self.assertEqual(self.sandbox.uv_calls.read_bytes(), calls)
        self.assertEqual(
            self.active.read_text(encoding="utf-8").strip(),
            "releases/" + new.name,
        )

    def test_activation_never_deletes_an_inherited_uv_cache(self) -> None:
        """Only the app's own runtime cache may be discarded.

        """

        external_cache = self.sandbox.root / "external-cache"
        external_cache.mkdir()
        sentinel = external_cache / "valuable-data"
        sentinel.write_text("keep", encoding="utf-8")
        result = self.sandbox.run({
            "UV_CACHE_DIR": str(external_cache),
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertTrue(self.active.is_file())

    def test_partial_inactive_release_is_rebuilt(self) -> None:
        """Only the incomplete target release can be deleted and rebuilt.

        """

        old = self._install("0.1.0")
        self.sandbox.set_launcher("kilin[app]==0.2.0")
        target = self.sandbox.release_runtime()
        target.mkdir(parents=True)
        (target / "partial").write_text("incomplete", encoding="utf-8")
        result = self.sandbox.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((old / ".installed").is_file())
        self.assertTrue((target / ".installed").is_file())
        self.assertFalse((target / "partial").exists())

    def test_completed_uncommitted_release_is_reused(self) -> None:
        """A crash before pointer activation needs no new download.

        """

        target = self.sandbox.release_runtime()
        (target / "bin").mkdir(parents=True)
        shutil.copy2(
            self.sandbox.gui_script, target / "bin/kilin-gui"
        )
        shutil.copy2(
            self.sandbox.python_script, target / "bin/python"
        )
        (target / ".installed").write_text(
            self.sandbox.spec + "\n", encoding="utf-8"
        )
        result = self.sandbox.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.sandbox.uv_calls.exists())
        self.assertEqual(
            self.active.read_text(encoding="utf-8").strip(),
            "releases/" + target.name,
        )

    def test_rebuilt_wheel_has_distinct_release_identity(self) -> None:
        """Same package version with changed wheel bytes upgrades safely.

        """

        old = self._install("0.1.0")
        self.sandbox.set_launcher("kilin[app]==0.1.0", "a" * 64)
        result = self.sandbox.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        new = self.sandbox.release_runtime()
        self.assertNotEqual(old, new)
        self.assertTrue(old.exists())
        self.assertTrue((new / ".installed").is_file())
        self.assertEqual(
            self.active.read_text(encoding="utf-8").strip(),
            "releases/" + new.name,
        )

    def test_legacy_runtime_remains_usable_during_transition(self) -> None:
        """An older PR4 installation is preserved on a failed upgrade.

        """

        legacy = self.sandbox.runtime
        (legacy / "bin").mkdir(parents=True)
        shutil.copy2(
            self.sandbox.gui_script, legacy / "bin/kilin-gui"
        )
        (legacy / ".installed").write_text(
            "kilin[app]==0.1.0\n", encoding="utf-8"
        )
        self.assertEqual(self.sandbox.run().returncode, 0)
        self.assertFalse(self.active.exists())
        self.sandbox.set_launcher("kilin[app]==0.2.0")
        failed = self.sandbox.run({"KILIN_TEST_FAIL_UV": "pip"})
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse(self.active.exists())
        self.assertTrue((legacy / ".installed").is_file())
        self.sandbox.set_launcher("kilin[app]==0.1.0")
        self.assertEqual(self.sandbox.run().returncode, 0)
        self.sandbox.set_launcher("kilin[app]==0.2.0")
        self.assertEqual(self.sandbox.run().returncode, 0)
        self.assertTrue((legacy / ".installed").is_file())
        self.assertTrue(self.active.is_file())

    def test_invalid_active_pointer_is_not_followed(self) -> None:
        """Corrupted state cannot select an arbitrary filesystem path.

        """

        old = self._install("0.1.0")
        for selection in ("../../escape", "releases/../escape", ""):
            with self.subTest(selection=selection):
                self.active.write_text(selection, encoding="utf-8")
                result = self.sandbox.run()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("active", result.stderr)
                self.assertTrue(old.exists())

    def test_symlinked_active_pointer_is_rejected(self) -> None:
        """Do not dereference a modified pointer outside runtime.

        """

        old = self._install("0.1.0")
        self.active.unlink()
        destination = self.sandbox.root / "other-active"
        destination.write_text("app\n", encoding="utf-8")
        self.active.symlink_to(destination)
        result = self.sandbox.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not be a symlink", result.stderr)
        self.assertTrue(old.exists())

    def test_incomplete_active_release_is_never_deleted(self) -> None:
        """An invalid active marker is not disposable staging.

        """

        release = self._install("0.1.0")
        (release / ".installed").unlink()
        result = self.sandbox.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("active", result.stderr)
        self.assertTrue((release / "bin/kilin-gui").is_file())

    def test_update_waiter_keeps_old_active_version_until_commit(
        self,
    ) -> None:
        """A contending update cannot activate until the installer ends.

        """

        old = self._install("0.1.0")
        old_selected = self.active.read_bytes()
        self.sandbox.set_launcher("kilin[app]==0.2.0")
        configured = self.sandbox.environment.copy()
        configured["KILIN_TEST_RELEASE"] = str(self.sandbox.release)
        first = subprocess.Popen(
            [str(self.sandbox.launcher)], env=configured,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 5
            while (
                not self.sandbox.started.exists()
                and time.monotonic() < deadline
            ):
                self.assertIsNone(first.poll())
                time.sleep(0.02)
            self.assertTrue(self.sandbox.started.exists())
            second = self.sandbox.run({
                "KILIN_INSTALL_LOCK_TIMEOUT": "1",
            })
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already installing", second.stderr)
            self.assertEqual(self.active.read_bytes(), old_selected)
            self.assertTrue(old.exists())
        finally:
            self.sandbox.release.touch()
            output = first.communicate(timeout=10)
        self.assertEqual(first.returncode, 0, output)
        self.assertNotEqual(self.active.read_bytes(), old_selected)
        self.assertTrue(old.exists())


if __name__ == "__main__":
    unittest.main()
