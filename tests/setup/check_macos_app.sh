#!/bin/bash
# Inspect a built KíLín application and disk image on a native macOS host.
# This is a package smoke test, not a first-run model-installation test.
set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Usage: bash tests/setup/check_macos_app.sh" >&2
    exit 2
fi

project_dir=$(cd "$(dirname "$0")/../.." && pwd)
app="$project_dir/dist/KíLín.app"
dmg="$project_dir/dist/KíLín.dmg"
launcher="$app/Contents/MacOS/KiLin"
uv="$app/Contents/Resources/uv"
plist="$app/Contents/Info.plist"

for path in "$launcher" "$uv" "$plist" "$dmg"; do
    if [ ! -f "$path" ]; then
        echo "Missing macOS distribution component: $path" >&2
        exit 1
    fi
done
if [ ! -x "$launcher" ] || [ ! -x "$uv" ]; then
    echo "The launcher and embedded uv must be executable." >&2
    exit 1
fi
if ! file -b "$uv" | grep -Eq '^Mach-O.*arm64'; then
    echo "Bundled uv must include native Apple Silicon code." >&2
    exit 1
fi
if grep -Eq '@KILIN_SPEC@|@KILIN_WHEEL_SHA@' "$launcher"; then
    echo "Unexpanded installer substitutions remain in the launcher." >&2
    exit 1
fi
bash -n "$launcher"
plutil -lint "$plist"
short_version=$(/usr/libexec/PlistBuddy -c \
    'Print :CFBundleShortVersionString' "$plist")
build_version=$(/usr/libexec/PlistBuddy -c \
    'Print :CFBundleVersion' "$plist")
source_version=$(sed -n 's/^version = "\(.*\)"$/\1/p' \
    "$project_dir/pyproject.toml" | head -n 1)
base_version=$(printf '%s' "$source_version" | \
    sed -E 's/(b|rc)[0-9]+$//')
if [ "$short_version" != "$base_version" ] || \
    [[ ! "$short_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || \
    [[ ! "$build_version" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]]; then
    echo "Invalid or mismatched numeric macOS bundle versions." >&2
    exit 1
fi
codesign --verify --deep --strict "$app"
hdiutil verify "$dmg" >/dev/null

mount_dir=$(mktemp -d "${TMPDIR:-/tmp}/kilin-smoke.XXXXXXXX")
mounted=0
cleanup() {
    status=$?
    trap - EXIT
    if [ "$mounted" -eq 1 ]; then
        hdiutil detach "$mount_dir" >/dev/null || status=1
    fi
    rmdir "$mount_dir" || status=1
    exit "$status"
}
trap cleanup EXIT
hdiutil attach "$dmg" -readonly -nobrowse -noautoopen -noverify \
    -mountpoint "$mount_dir" >/dev/null
mounted=1
if [ ! -f "$mount_dir/KíLín.app/Contents/MacOS/KiLin" ] || \
    [ ! -L "$mount_dir/Applications" ]; then
    echo "DMG is missing its application or Applications shortcut." >&2
    exit 1
fi
cmp "$uv" "$mount_dir/KíLín.app/Contents/Resources/uv"
cmp "$launcher" "$mount_dir/KíLín.app/Contents/MacOS/KiLin"

printf 'Verified KíLín macOS app, signature and disk image.\n'
