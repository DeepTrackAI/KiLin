"""Test the Windows installer's script and build contract.

Inno Setup and Windows are not needed: the checks read the installer script
and run the build script only where PowerShell exists. The real build,
silent installation and uninstallation run in the Windows CI workflow.

"""

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
SCRIPT = ROOT / "setup/windows/kilin.iss"
BUILDER = ROOT / "setup/windows/build_installer.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


class TestInstallerScript(unittest.TestCase):
    """Keep the promises the installer makes to users."""

    def setUp(self) -> None:
        """Read the Inno Setup script as Inno Setup does (UTF-8 with BOM)."""

        raw = SCRIPT.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"), "Inno needs a BOM")
        self.script = raw.decode("utf-8-sig")

    def test_installs_per_user_without_administrator_rights(self) -> None:
        for line in (
            "PrivilegesRequired=lowest",
            r"DefaultDirName={localappdata}\Programs\KiLin",
            "ArchitecturesAllowed=x64compatible",
        ):
            with self.subTest(line=line):
                self.assertIn(line, self.script)

    def test_runtime_is_versioned_and_picks_the_gpu_build(self) -> None:
        self.assertIn(r"{localappdata}\kilin\runtime", self.script)
        self.assertIn(r"RuntimeRoot() + '\{#AppVersion}'", self.script)
        self.assertIn("venv --python 3.12", self.script)
        self.assertIn("pip install --torch-backend auto", self.script)
        self.assertIn("'kilin[app]=={#AppVersion}'", self.script)

    def test_uv_runs_without_setups_redirection_guard(self) -> None:
        # Programs Setup starts inherit RedirectionGuard, which blocks the
        # folder link uv makes for Python with error 448.
        self.assertIn("RedirectionGuard=no", self.script)
        self.assertIn("PrivilegesRequired=lowest", self.script)

    def test_progress_page_follows_downloads_and_keeps_the_log(self) -> None:
        for text in (
            "CreateOutputProgressPage(",
            "ExecAndLogOutput(",
            "SaveStringToFile(InstallLog",
            "Pos('Downloading ', Line) = 1",
            "Pos('Downloaded ', Line) = 1",
            "' left'",
            "SetTimer(0, 0, 1000",
            "TNewMemo.Create(ProgressPage)",
            "'downloaded'",
            "'downloading'",
        ):
            with self.subTest(text=text):
                self.assertIn(text, self.script)

    def test_finish_page_offers_the_install_log(self) -> None:
        section = self.script.split("[Run]")[1].split("[UninstallDelete]")[0]
        log = r"{localappdata}\kilin\runtime\install.log"
        self.assertIn(f'Filename: "{log}"', section)
        self.assertIn("postinstall shellexec", section)

    def test_earlier_installation_can_be_reinstalled_or_removed(self) -> None:
        start = self.script.index("function InitializeSetup")
        body = self.script[start:self.script.index("end;\n\n", start)]
        app_id = self.script.split("AppId={{")[1].split("}")[0]
        self.assertIn(f"Uninstall\\{{{app_id}}}_is1", self.script)
        for text in (
            "WizardSilent()",
            "'UninstallString'",
            "CompareVersions('{#AppVersion}', Version)",
            "Reinstall KíLín",
            "Upgrade to KíLín",
            "Remove KíLín",
        ):
            with self.subTest(text=text):
                self.assertIn(text, body)

    def test_failure_stops_setup_before_files_are_copied(self) -> None:
        start = self.script.index("function PrepareToInstall")
        body = self.script[start:self.script.index("end;\n\n", start)]
        self.assertIn("Result := 'KíLín could not be installed.", body)
        self.assertIn("DelTree(Runtime", body)

    def test_repair_variables_match_the_application(self) -> None:
        from kilin.gui._paths import APP_INSTALL_SPEC, APP_INSTALL_UV

        for name in (APP_INSTALL_UV, APP_INSTALL_SPEC):
            with self.subTest(name=name):
                self.assertIn(f'ValueName: "{name}"', self.script)
        self.assertEqual(self.script.count("uninsdeletevalue"), 2)

    def test_uninstall_keeps_projects_and_preferences(self) -> None:
        section = self.script.split("[UninstallDelete]")[1].split("[Code]")[0]
        removed = [
            line for line in section.splitlines() if line.startswith("Type:")
        ]
        self.assertEqual(
            removed,
            [r'Type: filesandordirs; Name: "{localappdata}\kilin\runtime"'],
        )

    def test_only_directives_start_with_a_hash(self) -> None:
        # The preprocessor reads every line starting with "#", indented or
        # not, as a directive: a line such as "#13#10 + ..." fails the build.
        directives = ("#if", "#else", "#endif", "#error", "#define")
        for number, line in enumerate(self.script.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                with self.subTest(line=number):
                    self.assertTrue(line.lstrip().startswith(directives))

    def test_icon_exists(self) -> None:
        self.assertTrue((SCRIPT.parent / "kilin.ico").is_file())

    def test_terms_are_left_to_the_application(self) -> None:
        # As on macOS, the welcome dialog asks for the terms of use once.
        self.assertNotIn("LicenseFile=", self.script)
        self.assertTrue((ROOT / "kilin/gui/terms.txt").is_file())


@unittest.skipUnless(POWERSHELL, "PowerShell is not available.")
class TestInstallerBuilder(unittest.TestCase):

    @unittest.skipIf(os.name == "nt", "Checks the refusal on other systems.")
    def test_builder_refuses_other_systems(self) -> None:
        result = subprocess.run(
            [str(POWERSHELL), "-NoProfile", "-File", str(BUILDER)],
            cwd=ROOT,
            env={**os.environ, "OS": "NotWindows"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Build the Windows installer on Windows", result.stderr)


if __name__ == "__main__":
    unittest.main()
