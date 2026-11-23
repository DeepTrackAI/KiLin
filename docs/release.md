# Building and validating KíLín for macOS and Windows

## Scope and limitations

The macOS application is an Apple Silicon launcher containing `uv`, a
shell entry point, and optionally a wheel from this checkout. On first
launch, it downloads Python 3.12 and installs KíLín's dependencies.
Model downloads are separate, user-initiated operations. The runtime
and project data are not bundled into the disk image.

**The `0.1.0b1` Python package is prepared for public GitHub and PyPI
release.** Native macOS and Windows applications remain experimental
unsigned development artifacts, not ready-to-use signed public installers.
The macOS build uses ad-hoc signing only, not Developer ID signing, and is
**not notarized**. No Apple Developer credentials are required. Gatekeeper
may prevent ordinary opening of these builds. Public distribution of the
source package does not remove these platform limitations.

The release sequence is `0.1.0b1` → `0.1.0rc1` → `0.1.0`, with Git tags
`v0.1.0b1`, `v0.1.0rc1`, and `v0.1.0`. Installer specs must match the
exact PyPI version. On macOS, `CFBundleShortVersionString` uses the numeric
base (`0.1.0` for all three), while `CFBundleVersion` distinguishes beta,
RC, and stable builds using numeric release stages: `0.1.1001`, `0.1.2001`,
and `0.1.3000`, respectively. Later patch numbers occupy higher ranges.
On Windows, update comparisons use the semantic order beta < RC < stable.

To build from the published PyPI version or from a checkout wheel:

```sh
bash setup/macos/build_app.sh
bash setup/macos/build_app.sh --wheel
```

Both commands write `dist/KíLín.app` and `dist/KíLín.dmg`. They
preserve previous outputs until the replacement succeeds. Use
`KILIN_SKIP_FINDER_LAYOUT=1` in unattended builds to avoid Finder
permission prompts; the resulting DMG has the standard layout.

## Native development and CI verification

On an Apple Silicon Mac with Xcode command-line tools, `uv`, and Python
3.12:

```sh
python -m pytest -q
python -m tests
KILIN_SKIP_FINDER_LAYOUT=1 bash setup/macos/build_app.sh --wheel
bash tests/setup/check_macos_app.sh
```

The checker examines the generated application and disk image: bundled
`uv` architecture, expanded launcher, Info.plist, ad-hoc code signature,
DMG integrity, and the application and Applications shortcut in the
mounted image. It does **not** install Python packages, exercise the
real GUI, or establish Gatekeeper acceptance.

The `.github/workflows/macos_app.yml` workflow performs this native
build and inspection on an Apple Silicon runner, plus the deterministic
launcher, updater, backend-repair and builder tests. It uses no Apple
Developer credentials and does not notarize or publish the artifacts.
Automatic runs are path-filtered to relevant build, setup, GUI and
packaging files; manual execution remains available. Because skipped
path-filtered jobs may remain pending as required checks, do not make
this workflow an unconditional branch-protection requirement.
A manually downloaded CI image is a **development artifact**, not a
validated public release.

## Manual acceptance for trusted testers

Test a locally built DMG on a real Apple Silicon Mac, preferably from a
fresh user account. macOS may block the app because it is not Developer
ID signed or notarized. If the source is trusted, inspect the warning
and use the system-provided option in System Settings, Privacy &
Security, to allow that particular app. Never advise disabling
Gatekeeper or removing quarantine indiscriminately.

The public beta still includes the desktop's invitation-passkey prompt.
Before distributing a native build, confirm its behavior on a fresh user
account and communicate passkeys separately to the intended testers.
The prompt is local to the client; it is not an access restriction for
public Python source code or a security boundary.

Before sharing with trusted testers:

1. Check the package version, authors, and licence with the release
   metadata tests and installed-wheel verification in
   [Development](development.md).
2. Verify the copied `.app` and mounted DMG against the native checker.
   Record the macOS version, architecture, build command, and hash of
   the supplied artifact.
3. On a fresh user account, check the GUI passkey prompt before the main
   window. Incorrect input must stay in the dialog; Quit must close the
   application. Correct input must open the GUI and persist across restarts.
   Then check first-launch installation, network failures, and the welcome
   dialog's terms acceptance and retry behavior.
4. Test the explicitly confirmed projects folder, synchronization
   warning, alternative folders, and discovery of earlier projects.
   Confirm no existing data is moved without user action.
5. Verify MLX and PyTorch repair guidance in the app-managed runtime,
   including when `pip` is unavailable. Test an explicit model download
   and a basic model answer when hardware and weights are available.
6. Upgrade from an earlier build; check activation of the new runtime,
   preservation of the previous one, interrupted installation recovery,
   and that failed updates do not silently run an old release.
7. Stop and reopen a saved project; verify conversations, checkpoints,
   traces, and project files survive the upgrade.

These remain manual checks until real GUI and backend integration tests
are available in the native environment. Passing the CI smoke check
does not guarantee that an unnotarized app will open without a macOS
security prompt on other machines.

## Windows installer

`setup/windows/kilin.iss` is an Inno Setup script for 64-bit Windows; on
Windows, with uv 0.8 or later and Inno Setup 6, build it with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File setup\windows\build_installer.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File setup\windows\build_installer.ps1 -Wheel
```

Both write `dist\KiLin-Setup-<version>.exe`. The first installs the
matching version from PyPI; `-Wheel` bundles a wheel of this checkout.
The installer holds only uv, an icon and the optional wheel:

- It installs for the current user, without administrator rights, into
  `%LOCALAPPDATA%\Programs\KiLin`.
- Before copying any file, uv downloads Python 3.12 and installs
  `kilin[app]` into `%LOCALAPPDATA%\kilin\runtime\<version>`, with
  `--torch-backend auto`: the CUDA build of PyTorch for an NVIDIA GPU, the
  CPU build otherwise (PyPI alone has only the CPU build on Windows). A
  failure stops Setup with a message and the log
  `%LOCALAPPDATA%\kilin\runtime\install.log`; an earlier version is
  untouched. After an update, other versions' runtimes are removed.
- There is no licence page. As on macOS, the welcome dialog shows the
  terms of use on first launch and records their acceptance.
- The Start-menu shortcut opens `kilin-gui.exe` without a console window.
- `KILIN_APP_UV` and `KILIN_APP_SPEC` are set in the user environment so
  the application's backend-repair commands use the same uv and build, as
  the macOS launcher does.
- Uninstalling removes the program and the runtime, and keeps projects
  and preferences.

If KíLín is already installed, Setup first compares versions and offers
to upgrade (a newer installer), reinstall (the same version, for example
to repair it) or install the older version, or to remove KíLín.

While uv installs, a progress page shows the percentage, the size
downloaded and an estimate of the time left, above a list of the large
downloads, such as PyTorch, each marked downloading or downloaded, newest
first. uv only reports when a file starts and ends, so the progress
follows the bytes the busiest network interface receives; other traffic
at the same time makes it run slightly ahead. The first install takes
several minutes.

On the last page, **Show the install log** opens `install.log`. It lists
each large download as `Downloaded torch (2.4 GiB)`, then the installed
packages and their versions as lines starting with `+`.

**The installer is not code-signed.** Windows SmartScreen shows "Windows
protected your PC" for it; testers who trust the build choose **More
info → Run anyway**. Signing for public release needs a code-signing
certificate (or a signing service such as Azure Trusted Signing), a
separate decision like Apple notarization.

The `.github/workflows/windows_app.yml` workflow builds the installer with
a checkout wheel on a Windows runner, installs it silently, checks the
runtime, the shortcut and the repair variables, and uninstalls it. Every
run uploads the unsigned installer as a development artifact. Before
sharing a build, also test on a real Windows PC: first install, update
over an earlier version, an NVIDIA computer, a model download and a basic
answer.

## Native signing and notarization (deferred)

We are deliberately **not implementing notarization** or a credentialed
Developer ID release process. Public distribution with the usual
Gatekeeper experience would require separate decisions and access to an
Apple Developer Program account. No developer credentials or
notarization commands are needed for the documented build and
verification workflow.

For the public source/PyPI prerelease, release maintainers must also check
that GitHub's migrated `DeepTrackAI/KiLin` repository is public, the README
images load at the `raw.githubusercontent.com` URLs, and the current mascot
artwork has documented redistribution permission. GitHub/PyPI do not host
arbitrary local image paths from the wheel. Use the same tagged source for
GitHub and PyPI artifacts; follow the [release checklist](development.md#public-prerelease-sequence).


## Related

[Development](development.md), [GUI architecture](gui_architecture.md),
[Configuration](configuration.md), [Security](security.md).
