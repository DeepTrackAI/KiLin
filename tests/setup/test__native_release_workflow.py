"""Verify native release automation without publishing GitHub assets.

The publishing step is exercised with a local fake ``gh`` executable;
no test accesses GitHub, PyPI, or native installer build services.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = (WORKFLOWS / "native_release.yml").read_text(encoding="utf-8")
MACOS = (WORKFLOWS / "macos_app.yml").read_text(encoding="utf-8")
WINDOWS = (WORKFLOWS / "windows_app.yml").read_text(encoding="utf-8")


def _step_script(name: str) -> str:
    """Extract one YAML literal shell script for isolated testing.

    Parameters
    ----------
    name : str
        Display name of the shell-script workflow step.

    Returns
    -------
    str
        De-indented source of its shell script.

    """

    marker = f"      - name: {name}\n"
    remainder = RELEASE.split(marker, 1)[1]
    remainder = remainder.split("        run: |\n", 1)[1]
    lines = []
    for line in remainder.splitlines(keepends=True):
        if not line.strip():
            lines.append("\n")
            continue
        if not line.startswith("          "):
            break
        lines.append(line[10:])
    return "".join(lines)


class TestNativeReleaseWorkflow(unittest.TestCase):
    """Enforce trigger, permission, and publication safety contracts.

    """

    def test_release_is_tag_triggered_and_not_path_filtered(self):
        self.assertIn('  push:\n    tags:\n      - "v*"', RELEASE)
        self.assertNotIn("  release:", RELEASE)
        self.assertNotIn("    paths:", RELEASE)
        self.assertNotIn("workflow_dispatch:", RELEASE)

    def test_both_native_checks_gate_release_publication(self):
        self.assertIn("uses: ./.github/workflows/macos_app.yml", RELEASE)
        self.assertIn("uses: ./.github/workflows/windows_app.yml", RELEASE)
        self.assertIn("needs: [validate, macos, windows]", RELEASE)
        self.assertIn("python -m setup.generate_pypi_readme --check", RELEASE)
        self.assertIn("name: kilin-macos-adhoc-development", RELEASE)
        self.assertIn("name: kilin-windows-unsigned-development", RELEASE)

    def test_publication_alone_has_write_permission(self):
        self.assertEqual(RELEASE.count("contents: write"), 1)
        self.assertEqual(RELEASE.count("contents: read"), 3)
        self.assertIn("  group: kilin-native-release-", RELEASE)
        self.assertIn("  cancel-in-progress: false", RELEASE)

    def test_reusable_builds_preserve_existing_smoke_triggers(self):
        for name, source in (("macos", MACOS), ("windows", WINDOWS)):
            with self.subTest(platform=name):
                self.assertIn("  workflow_call:\n", source)
                self.assertIn("  workflow_dispatch:\n", source)
                self.assertIn("    branches: [main, develop]", source)
                self.assertIn("    paths:\n", source)
                self.assertIn("contents: read", source)
        self.assertIn("group: kilin-macos-app-", MACOS)
        self.assertIn("group: kilin-windows-app-", WINDOWS)
        self.assertIn("startsWith(github.ref, 'refs/tags/v')", MACOS)
        self.assertIn("name: kilin-macos-adhoc-development", MACOS)
        self.assertIn("name: kilin-windows-unsigned-development", WINDOWS)

    def test_tag_must_match_manifest_and_release_stage(self):
        source = _step_script("Match tag to the version in pyproject.toml")
        python = source.split("python - <<'PYTHON'\n", 1)[1]
        python = python.rsplit("\nPYTHON", 1)[0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "outputs.txt"
            for version, tag, expected in (
                ("0.1.0b1", "v0.1.0b1", "true"),
                ("0.1.0rc2", "v0.1.0rc2", "true"),
                ("0.1.0", "v0.1.0", "false"),
            ):
                with self.subTest(version=version):
                    output.write_text("", encoding="utf-8")
                    (root / "pyproject.toml").write_text(
                        f'[project]\nversion = "{version}"\n',
                        encoding="utf-8",
                    )
                    env = dict(os.environ, RELEASE_TAG=tag)
                    env["GITHUB_OUTPUT"] = str(output)
                    result = subprocess.run(
                        [sys.executable, "-c", python],
                        cwd=root, env=env, capture_output=True,
                        text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        output.read_text(encoding="utf-8"),
                        f"version={version}\nprerelease={expected}\n",
                    )

    def test_incorrect_or_unsupported_tags_fail_closed(self):
        source = _step_script("Match tag to the version in pyproject.toml")
        python = source.split("python - <<'PYTHON'\n", 1)[1]
        python = python.rsplit("\nPYTHON", 1)[0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for version, tag in (
                ("0.1.0b1", "v0.1.0b2"),
                ("0.1.0b1", "v0.1.0"),
                ("0.1.0.dev1", "v0.1.0.dev1"),
                ("0.1.0b1", "v0.1.0b1-extra"),
            ):
                with self.subTest(version=version, tag=tag):
                    (root / "pyproject.toml").write_text(
                        f'[project]\nversion = "{version}"\n',
                        encoding="utf-8",
                    )
                    env = dict(os.environ, RELEASE_TAG=tag)
                    env["GITHUB_OUTPUT"] = str(root / "outputs.txt")
                    result = subprocess.run(
                        [sys.executable, "-c", python],
                        cwd=root, env=env, capture_output=True,
                        text=True, check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)

    @unittest.skipUnless(
        os.name == "posix" and shutil.which("bash"),
        "Shell syntax validation requires Bash",
    )
    def test_embedded_shell_steps_have_valid_syntax(self):
        for name in (
            "Match tag to the version in pyproject.toml",
            "Prepare versioned assets and checksums",
            "Create release or attach assets to an existing release",
        ):
            with self.subTest(step=name):
                result = subprocess.run(
                    ["bash", "-n"], input=_step_script(name),
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and shutil.which("bash"),
        "Release packaging is executed on an Ubuntu runner",
    )
    def test_checksums_describe_both_versioned_artifacts(self):
        script = _step_script("Prepare versioned assets and checksums")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            macos = root / "release-assets" / "macos" / "KíLín.dmg"
            windows = (
                root / "release-assets" / "windows"
                / "KiLin-Setup-0.1.0b1.exe"
            )
            macos.parent.mkdir(parents=True)
            windows.parent.mkdir(parents=True)
            macos.write_bytes(b"macos development binary")
            windows.write_bytes(b"windows development binary")
            env = dict(os.environ, RELEASE_VERSION="0.1.0b1")
            result = subprocess.run(
                ["bash", "-c", script], cwd=root, env=env,
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = "".join(
                f"{hashlib.sha256(content).hexdigest()}  {name}\n"
                for name, content in (
                    ("KiLin-0.1.0b1-macos-arm64.dmg",
                     b"macos development binary"),
                    ("KiLin-Setup-0.1.0b1.exe",
                     b"windows development binary"),
                )
            )
            actual = (
                root / "release-assets" / "SHA256SUMS.txt"
            ).read_text(encoding="utf-8")
            self.assertEqual(actual, expected)

    @unittest.skipUnless(
        os.name == "posix" and shutil.which("bash"),
        "The isolated GitHub CLI stub requires a POSIX shell",
    )
    def test_publish_flags_and_existing_release_safety(self):
        script = _step_script(
            "Create release or attach assets to an existing release"
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / "gh"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json\nimport os\nimport sys\n"
                "with open(os.environ['GH_LOG'], 'a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "if sys.argv[1:3] == ['release', 'view']:\n"
                "    sys.exit(0 if os.environ['EXISTING'] == '1' else 1)\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            log = root / "gh.log"
            for version, existing in (
                ("0.1.0b1", "0"),
                ("0.1.0", "0"),
                ("0.1.0b1", "1"),
            ):
                with self.subTest(version=version, existing=existing):
                    log.write_text("", encoding="utf-8")
                    env = dict(
                        os.environ,
                        PATH=str(root) + os.pathsep + os.environ["PATH"],
                        GH_LOG=str(log),
                        GH_TOKEN="test-only-token",
                        GH_REPO="DeepTrackAI/KiLin",
                        RELEASE_TAG="v" + version,
                        RELEASE_VERSION=version,
                        IS_PRERELEASE=(
                            "true" if "b" in version else "false"
                        ),
                        EXISTING=existing,
                    )
                    result = subprocess.run(
                        ["bash", "-c", script], cwd=root, env=env,
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    calls = [
                        json.loads(line) for line in log.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                    self.assertEqual(calls[0][:2], ["release", "view"])
                    self.assertEqual(len(calls), 2)
                    command = calls[1]
                    self.assertEqual(command[0], "release")
                    self.assertEqual(
                        command[1], "upload" if existing == "1"
                        else "create",
                    )
                    self.assertIn("SHA256SUMS.txt", " ".join(command))
                    self.assertNotIn("--clobber", command)
                    if existing == "0":
                        self.assertIn("--verify-tag", command)
                        self.assertEqual(
                            "--prerelease" in command, "b" in version
                        )


if __name__ == "__main__":
    unittest.main()
