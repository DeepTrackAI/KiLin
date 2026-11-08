# Building and validating KíLín for macOS

## Scope and limitations

The macOS application is an Apple Silicon launcher containing `uv`, a
shell entry point, and optionally a wheel from this checkout. On first
launch, it downloads Python 3.12 and installs KíLín's dependencies.
Model downloads are separate, user-initiated operations. The runtime
and project data are not bundled into the disk image.

**The current application is for development and trusted internal tests.**
By default, the build uses ad-hoc rather than Developer ID signing,
and it is **not notarized**. No Apple Developer credentials are required.
macOS Gatekeeper may prevent ordinary opening of these builds. We do not
claim that they are suitable for frictionless public distribution.

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
A manually downloaded CI image is a **development artifact**, not a
validated public release.

## Manual acceptance for trusted testers

Test a locally built DMG on a real Apple Silicon Mac, preferably from a
fresh user account. macOS may block the app because it is not Developer
ID signed or notarized. If the source is trusted, inspect the warning
and use the system-provided option in System Settings, Privacy &
Security, to allow that particular app. Never advise disabling
Gatekeeper or removing quarantine indiscriminately.

Before sharing with trusted testers:

1. Check the package version, authors, and licence with the release
   metadata tests and installed-wheel verification in
   [Development](development.md).
2. Verify the copied `.app` and mounted DMG against the native checker.
   Record the macOS version, architecture, build command, and hash of
   the supplied artifact.
3. Test first-launch installation, network failures, and the welcome
   dialog's terms acceptance and failure/retry behavior.
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

## Deferred public distribution

We are deliberately **not implementing notarization** or a credentialed
Developer ID release process. Public distribution with the usual
Gatekeeper experience would require separate decisions and access to an
Apple Developer Program account. No developer credentials or notarization commands are needed for
the documented build and verification workflow.

## Related

[Development](development.md), [GUI architecture](gui_architecture.md),
[Configuration](configuration.md), [Security](security.md).
