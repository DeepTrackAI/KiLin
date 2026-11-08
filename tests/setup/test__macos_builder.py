"""Test macOS application-builder artifact safety and uv resolution.

Run the production Bash builder with isolated macOS command stand-ins.
No application launch, Finder prompt, signing identity, or network is used.

"""

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


_ROOT = Path(__file__).resolve().parents[2]
_BUILDER = _ROOT / "setup/macos/build_app.sh"
_LAUNCHER = _ROOT / "setup/macos/launcher.sh"


@unittest.skipUnless(
    os.name == "posix" and shutil.which("bash") and shutil.which("realpath"),
    "Builder tests require Bash and POSIX filesystem utilities",
)
class TestMacOSBuilder(unittest.TestCase):
    """Keep generated outputs isolated until publication succeeds.

    """

    def setUp(self) -> None:
        """Create a project fixture and mock macOS command-line tools.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "Project with spaces"
        self.bin = self.root / "mock-bin"
        self.tools = self.root / "tools arm64 with spaces"
        self.dist = self.root / "dist"
        self.builder = self.root / "setup/macos/build_app.sh"
        self.bin.mkdir(parents=True)
        self.tools.mkdir(parents=True)
        self.builder.parent.mkdir(parents=True)
        self.builder.write_bytes(_BUILDER.read_bytes())
        (self.builder.parent / "launcher.sh").write_bytes(
            _LAUNCHER.read_bytes()
        )
        (self.root / "pyproject.toml").write_text(
            '[project]\nname = "kilin"\nversion = "0.1.0"\n',
            encoding="utf-8",
        )
        self.uv_real = self.tools / "uv-real"
        self._executable(self.uv_real, self._uv_stub())
        self.uv_link = self.tools / "uv"
        self.uv_link.symlink_to(self.uv_real.name)
        self.uv = self.tools / "uv-homebrew"
        self.uv.symlink_to(self.uv_link.name)
        self._executable(self.bin / "file", self._file_stub())
        self._executable(self.bin / "mv", self._mv_stub())
        self._executable(self.bin / "macos-tools", self._macos_stub())
        for name in (
            "sips", "iconutil", "codesign", "tiffutil", "hdiutil",
            "osascript",
        ):
            (self.bin / name).symlink_to("macos-tools")
        self.log = self.root / "mock.log"
        self.environment = os.environ.copy()
        # The local shell must not override signing or error fixtures.
        for key in list(self.environment):
            if key.startswith("KILIN_TEST_") or key in {
                "CODESIGN_IDENTITY", "KILIN_SKIP_FINDER_LAYOUT",
            }:
                self.environment.pop(key)
        self.environment.update({
            "UV_BINARY": str(self.uv),
            "KILIN_MOCK_LOG": str(self.log),
            "PATH": str(self.bin) + os.pathsep
            + self.environment.get("PATH", ""),
        })

    @staticmethod
    def _executable(path: Path, content: str) -> None:
        """Create one executable fixture with exactly the given source.

        Parameters
        ----------
        path : pathlib.Path
            Destination for the executable script.
        content : str
            Script source.

        """

        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    @staticmethod
    def _uv_stub() -> str:
        """Return a fake uv that creates a single deterministic wheel.

        Returns
        -------
        str
            Source for the fake executable.

        """

        return """#!/bin/sh
printf 'uv %s\\n' "$*" >> "$KILIN_MOCK_LOG"
if [ "${KILIN_TEST_FAIL_UV:-0}" = 1 ]; then
    exit 31
fi
if [ "$1" = build ]; then
    while [ "$#" -gt 0 ]; do
        if [ "$1" = --out-dir ]; then
            shift
            mkdir -p "$1"
            printf 'test wheel' > "$1/kilin-0.1.0-py3-none-any.whl"
            break
        fi
        shift
    done
fi
"""

    @staticmethod
    def _file_stub() -> str:
        """Require architecture inspection of the resolved uv target.

        Returns
        -------
        str
            Fake ``file`` command source.

        """

        return """#!/bin/sh
printf 'file %s\\n' "$*" >> "$KILIN_MOCK_LOG"
if [ "$1" = -b ]; then
    shift
else
    printf '%s: ' "$1"
fi
if [ "${KILIN_TEST_BAD_ARCH:-0}" = 1 ]; then
    printf 'Mach-O x86_64\\n'
elif [ "${1##*/}" = uv-real ]; then
    printf 'Mach-O arm64\\n'
else
    printf 'symbolic link\\n'
fi
"""

    @staticmethod
    def _mv_stub() -> str:
        """Optionally fail just the new DMG publication operation.

        Returns
        -------
        str
            Fake ``mv`` command source.

        """

        return """#!/bin/sh
if [ "${KILIN_TEST_FAIL_PUBLISH:-0}" = 1 ] && \\
    [ "${1##*/}" = 'KíLín.dmg' ] && \\
    [ "${2##*/}" = 'KíLín.dmg' ]; then
    printf 'Simulated DMG publish failure\\n' >&2
    exit 32
fi
exec /bin/mv "$@"
"""

    @staticmethod
    def _macos_stub() -> str:
        """Replace signing, icon, DMG, and Finder tools without dialogs.

        Returns
        -------
        str
            Source of the shared fake macOS command.

        """

        return """#!/bin/sh
name=${0##*/}
printf '%s %s\\n' "$name" "$*" >> "$KILIN_MOCK_LOG"
case "$name" in
    sips|iconutil|tiffutil)
        while [ "$#" -gt 0 ]; do
            if [ "$1" = --out ] || [ "$1" = -o ] || \\
                [ "$1" = -out ]; then
                shift
                printf 'generated %s' "$name" > "$1"
                exit 0
            fi
            shift
        done
        exit 20
        ;;
    codesign)
        if [ "${KILIN_TEST_FAIL_SIGN:-0}" = 1 ]; then
            exit 21
        fi
        ;;
    hdiutil)
        action=$1
        shift
        case "$action" in
            create)
                while [ "$#" -gt 1 ]; do shift; done
                printf 'writable' > "$1"
                ;;
            attach)
                if [ "${KILIN_TEST_FAIL_ATTACH:-0}" = 1 ]; then
                    exit 22
                fi
                ;;
            detach)
                ;;
            convert)
                if [ "${KILIN_TEST_FAIL_CONVERT:-0}" = 1 ]; then
                    exit 23
                fi
                while [ "$#" -gt 0 ]; do
                    if [ "$1" = -o ]; then
                        shift
                        printf 'new dmg' > "$1"
                        exit 0
                    fi
                    shift
                done
                exit 24
                ;;
        esac
        ;;
    osascript)
        cat >/dev/null
        if [ "${KILIN_TEST_DENY_FINDER:-0}" = 1 ]; then
            exit 25
        fi
        ;;
esac
exit 0
"""

    def _run(
        self,
        *arguments: str,
        variables: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run the builder with isolated macOS tools and bounded time.

        Parameters
        ----------
        *arguments : str
            Builder command-line arguments.
        variables : dict[str, str] or None
            Additional environment values for a failure scenario.

        Returns
        -------
        subprocess.CompletedProcess[str]
            Process result, including status and captured output.

        """

        environment = self.environment.copy()
        environment.update(variables or {})
        return subprocess.run(
            ["bash", str(self.builder), *arguments],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=12,
        )

    def _previous_outputs(self) -> None:
        """Create working outputs and unrelated files that must survive.

        """

        app = self.dist / "KíLín.app"
        (app / "Contents").mkdir(parents=True)
        (app / "Contents" / "old").write_text(
            "original app", encoding="utf-8"
        )
        (self.dist / "KíLín.dmg").write_text(
            "original dmg", encoding="utf-8"
        )
        (self.dist / "kilin-0.0.9-py3-none-any.whl").write_text(
            "unrelated wheel", encoding="utf-8"
        )
        (self.dist / "kilin-0.1.0-py3-none-any.whl").write_text(
            "other wheel", encoding="utf-8"
        )
        (self.root / "build").mkdir()
        (self.root / "build" / "personal.txt").write_text(
            "keep", encoding="utf-8"
        )

    def _assert_unrelated_files(self) -> None:
        """Verify that neither older wheels nor build files changed.

        """

        self.assertEqual(
            (self.dist / "kilin-0.0.9-py3-none-any.whl")
            .read_text(encoding="utf-8"),
            "unrelated wheel",
        )
        self.assertEqual(
            (self.dist / "kilin-0.1.0-py3-none-any.whl")
            .read_text(encoding="utf-8"),
            "other wheel",
        )
        self.assertEqual(
            (self.root / "build/personal.txt")
            .read_text(encoding="utf-8"),
            "keep",
        )
        self.assertEqual(
            list(self.dist.glob(".kilin-build.*")),
            [],
        )

    def _assert_previous_outputs(self) -> None:
        """Verify that both previous public artifacts remain intact.

        """

        self.assertEqual(
            (self.dist / "KíLín.app/Contents/old")
            .read_text(encoding="utf-8"),
            "original app",
        )
        self.assertEqual(
            (self.dist / "KíLín.dmg")
            .read_text(encoding="utf-8"),
            "original dmg",
        )
        self._assert_unrelated_files()

    def test_symlinked_uv_and_wheel_build_preserve_unrelated_files(
        self,
    ) -> None:
        """Bundle the target binary, not a Homebrew-style symlink.

        """

        self._previous_outputs()
        result = self._run("--wheel")
        self.assertEqual(result.returncode, 0, result.stderr)
        app = self.dist / "KíLín.app"
        self.assertEqual(
            (app / "Contents/Resources/uv").read_bytes(),
            self.uv_real.read_bytes(),
        )
        self.assertTrue(
            (app / "Contents/Resources/kilin-0.1.0-"
             "py3-none-any.whl").is_file()
        )
        launcher = (app / "Contents/MacOS/KiLin").read_text(
            encoding="utf-8"
        )
        self.assertIn("@RESOURCES@/kilin-0.1.0-", launcher)
        self.assertNotIn("@KILIN_WHEEL_SHA@", launcher)
        self.assertEqual(
            (self.dist / "KíLín.dmg").read_text(encoding="utf-8"),
            "new dmg",
        )
        self.assertIn(
            "file -b " + str(self.uv_real.resolve()),
            self.log.read_text(),
        )
        self._assert_unrelated_files()

    def test_pypi_build_replaces_only_named_outputs(self) -> None:
        """Do not build or overwrite a wheel in PyPI install mode.

        """

        self._previous_outputs()
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        launcher = (
            self.dist / "KíLín.app/Contents/MacOS/KiLin"
        ).read_text(encoding="utf-8")
        self.assertIn("kilin[app]==0.1.0", launcher)
        self.assertNotIn("uv build", self.log.read_text())
        self._assert_unrelated_files()

    def test_wheel_failure_keeps_prior_distributions(self) -> None:
        """Do not remove last build before wheel construction succeeds.

        """

        self._previous_outputs()
        result = self._run(
            "--wheel", variables={"KILIN_TEST_FAIL_UV": "1"},
        )
        self.assertNotEqual(result.returncode, 0)
        self._assert_previous_outputs()

    def test_signing_failure_keeps_prior_distributions(self) -> None:
        """A signing error must leave the earlier app and DMG intact.

        """

        self._previous_outputs()
        result = self._run(variables={"KILIN_TEST_FAIL_SIGN": "1"})
        self.assertNotEqual(result.returncode, 0)
        self._assert_previous_outputs()

    def test_dmg_conversion_failure_keeps_prior_distributions(
        self,
    ) -> None:
        """A disk-image failure cannot overwrite earlier outputs.

        """

        self._previous_outputs()
        result = self._run(
            variables={"KILIN_TEST_FAIL_CONVERT": "1"},
        )
        self.assertNotEqual(result.returncode, 0)
        self._assert_previous_outputs()

    def test_failed_publish_restores_both_previous_outputs(self) -> None:
        """The previous app is restored if the final DMG move fails.

        """

        self._previous_outputs()
        result = self._run(
            variables={"KILIN_TEST_FAIL_PUBLISH": "1"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Simulated DMG publish failure", result.stderr)
        self._assert_previous_outputs()

    def test_missing_old_outputs_are_not_required(self) -> None:
        """A fresh build requires no pre-existing app or image.

        """

        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(
            (self.dist / "KíLín.app/Contents/MacOS/KiLin").is_file()
        )
        self.assertTrue((self.dist / "KíLín.dmg").is_file())
        self.assertEqual(list(self.dist.glob(".kilin-build.*")), [])

    def test_non_arm_binary_fails_before_creating_build_outputs(
        self,
    ) -> None:
        """Reject x86-only uv even if it was provided through a link.

        """

        self.assertIn("arm64", str(self.uv_real))
        result = self._run(variables={"KILIN_TEST_BAD_ARCH": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not an Apple-silicon uv binary", result.stderr)
        self.assertFalse(self.dist.exists())

    def test_unknown_option_is_rejected_without_building(self) -> None:
        """Reject unknown build modes rather than silently using PyPI.

        """

        result = self._run("--not-a-mode")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)
        self.assertFalse(self.dist.exists())

    def test_finder_permission_denial_still_builds_plain_dmg(self) -> None:
        """Finder customization failure preserves the plain DMG.

        """

        result = self._run(
            variables={"KILIN_TEST_DENY_FINDER": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("it keeps the plain layout", result.stderr)
        self.assertTrue((self.dist / "KíLín.dmg").is_file())

    def test_headless_build_does_not_use_finder_or_credentials(self) -> None:
        """Ad-hoc CI builds require neither Finder nor developer credentials.

        """
        result = self._run(variables={"KILIN_SKIP_FINDER_LAYOUT": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        log = self.log.read_text(encoding="utf-8")
        self.assertNotIn("osascript ", log)
        self.assertNotIn("xcrun ", log)
        self.assertIn("codesign --force --deep --sign -", log)

    def test_notarize_option_is_not_supported(self) -> None:
        """Prevent an unsupported release mode from being accepted.

        """
        result = self._run("--notarize")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)
        self.assertFalse(self.dist.exists())

    def test_mount_failure_does_not_touch_previous_outputs(self) -> None:
        """Do not publish an image that failed to mount.

        """

        self._previous_outputs()
        result = self._run(
            variables={"KILIN_TEST_FAIL_ATTACH": "1"},
        )
        self.assertNotEqual(result.returncode, 0)
        self._assert_previous_outputs()


if __name__ == "__main__":
    unittest.main()
