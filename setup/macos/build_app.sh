#!/bin/bash
# Build KíLín.app and KíLín.dmg for Apple-silicon Macs.
#
#   bash setup/macos/build_app.sh           install KíLín from PyPI
#   bash setup/macos/build_app.sh --wheel   bundle a local checkout wheel
#
# Output: dist/KíLín.app and dist/KíLín.dmg. Set CODESIGN_IDENTITY for
# Developer ID signing; otherwise the app is signed ad hoc. Finder may
# request automation permission when arranging the disk image window.
set -euo pipefail

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--wheel" ]; }; then
    echo "Usage: bash setup/macos/build_app.sh [--wheel]" >&2
    exit 2
fi

# Homebrew commonly exposes uv through a symlink. Inspect and embed the
# executable itself, rather than checking the symlink's file type.
uv_candidate=${UV_BINARY:-$(command -v uv || true)}
if [ -z "$uv_candidate" ] || [ ! -x "$uv_candidate" ]; then
    echo "uv is needed to build the app; install it or set UV_BINARY." >&2
    exit 1
fi
if ! uv_binary=$(realpath "$uv_candidate"); then
    echo "Cannot resolve the uv executable: $uv_candidate" >&2
    exit 1
fi
if [ ! -x "$uv_binary" ] || ! uv_type=$(file -b "$uv_binary"); then
    echo "Cannot inspect the uv executable: $uv_candidate" >&2
    exit 1
fi
if [[ "$uv_type" != *Mach-O* || "$uv_type" != *arm64* ]]; then
    echo "$uv_candidate is not an Apple-silicon uv binary." >&2
    exit 1
fi

project_dir=$(cd "$(dirname "$0")/../.." && pwd)
cd "$project_dir"
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -n 1)
if [ -z "$version" ]; then
    echo "Cannot find the KíLín version in pyproject.toml." >&2
    exit 1
fi

dist="$project_dir/dist"
final_app="$dist/KíLín.app"
final_dmg="$dist/KíLín.dmg"
bundle_id=${KILIN_BUNDLE_ID:-"org.kilin.app"}
mkdir -p "$dist"

# Work on the same filesystem as the final outputs. A failed build leaves
# previous distributions and unrelated wheel/build artifacts untouched.
build_dir=$(mktemp -d "$dist/.kilin-build.XXXXXXXX")
app="$build_dir/KíLín.app"
dmg="$build_dir/KíLín.dmg"
mount_point="$build_dir/mounted-dmg"
mounted=0
promoting=0
old_app_saved=0
old_dmg_saved=0
new_app_moved=0
new_dmg_moved=0

cleanup() {
    status=$?
    trap - EXIT
    set +e
    if [ "$mounted" -eq 1 ]; then
        if ! hdiutil detach "$mount_point" >/dev/null; then
            echo "Cannot detach $mount_point; inspect it manually." >&2
            echo "Build files kept at $build_dir" >&2
            exit 1
        fi
    fi
    if [ "$promoting" -eq 1 ]; then
        # Ordinary promotion errors must not cost the last working build.
        [ "$new_app_moved" -eq 0 ] || rm -rf "$final_app"
        [ "$new_dmg_moved" -eq 0 ] || rm -f "$final_dmg"
        if [ "$old_app_saved" -eq 1 ]; then
            if ! mv "$build_dir/previous.app" "$final_app"; then
                echo "Restore $final_app from $build_dir/previous.app" >&2
                exit 1
            fi
        fi
        if [ "$old_dmg_saved" -eq 1 ]; then
            if ! mv "$build_dir/previous.dmg" "$final_dmg"; then
                echo "Restore $final_dmg from $build_dir/previous.dmg" >&2
                exit 1
            fi
        fi
    fi
    rm -rf "$build_dir"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
spec="kilin[app]==$version"
wheel_sha=""
if [ "${1:-}" = "--wheel" ]; then
    mkdir -p "$build_dir/wheels"
    "$uv_binary" build --wheel --out-dir "$build_dir/wheels"
    wheels=("$build_dir"/wheels/kilin-*.whl)
    if [ "${#wheels[@]}" -ne 1 ] || [ ! -f "${wheels[0]}" ]; then
        echo "Expected exactly one KíLín wheel from uv build." >&2
        exit 1
    fi
    wheel=${wheels[0]}
    cp "$wheel" "$app/Contents/Resources/"
    spec="@RESOURCES@/$(basename "$wheel")[app]"
    wheel_sha=$(shasum -a 256 "$wheel" | cut -d ' ' -f 1)
fi

sed -e "s|@KILIN_SPEC@|$spec|" \
    -e "s|@KILIN_WHEEL_SHA@|$wheel_sha|" \
    setup/macos/launcher.sh >"$app/Contents/MacOS/KiLin"
chmod +x "$app/Contents/MacOS/KiLin"
cp "$uv_binary" "$app/Contents/Resources/uv"

iconset="$build_dir/KiLin.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256 512; do
    double=$((size * 2))
    sips -z "$size" "$size" kilin/gui/icons/kilin-light.png \
        --out "$iconset/icon_${size}x${size}.png" >/dev/null
    sips -z "$double" "$double" kilin/gui/icons/kilin-light.png \
        --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$app/Contents/Resources/KiLin.icns"

cat >"$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>KíLín</string>
    <key>CFBundleDisplayName</key><string>KíLín</string>
    <key>CFBundleIdentifier</key><string>$bundle_id</string>
    <key>CFBundleVersion</key><string>$version</string>
    <key>CFBundleShortVersionString</key><string>$version</string>
    <key>CFBundleExecutable</key><string>KiLin</string>
    <key>CFBundleIconFile</key><string>KiLin</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSArchitecturePriority</key><array><string>arm64</string></array>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
codesign --force --deep --sign "${CODESIGN_IDENTITY:--}" "$app"

# Assemble and customize the DMG inside the isolated build directory.
# Finder permission is optional; a denied request keeps the plain layout.
staging="$build_dir/dmg-source"
mkdir -p "$staging/.background" "$mount_point"
cp -R "$app" "$staging/"
ln -s /Applications "$staging/Applications"
tiffutil -cathidpicheck setup/macos/dmg-background.png \
    setup/macos/dmg-background@2x.png \
    -out "$staging/.background/background.tiff" >/dev/null 2>&1
writable="$build_dir/writable.dmg"
hdiutil create -volname "KíLín" -srcfolder "$staging" -format UDRW \
    "$writable" >/dev/null
hdiutil attach "$writable" -readwrite -noverify -noautoopen \
    -mountpoint "$mount_point" >/dev/null
mounted=1
if ! osascript >/dev/null <<APPLESCRIPT
tell application "Finder"
    set dmgFolder to folder (POSIX file "$mount_point" as alias)
    open dmgFolder
    set current view of container window of dmgFolder to icon view
    set toolbar visible of container window of dmgFolder to false
    set statusbar visible of container window of dmgFolder to false
    set bounds of container window of dmgFolder to {200, 120, 840, 548}
    set options to icon view options of container window of dmgFolder
    set arrangement of options to not arranged
    set icon size of options to 112
    set text size of options to 13
    set background picture of options to file ".background:background.tiff" of dmgFolder
    set position of item "KíLín.app" of dmgFolder to {170, 170}
    set position of item "Applications" of dmgFolder to {470, 170}
    update dmgFolder without registering applications
    delay 1
    close container window of dmgFolder
end tell
APPLESCRIPT
then
    echo "Could not lay out the disk image window (Finder automation" \
        "may need permission); it keeps the plain layout." >&2
fi
sync
hdiutil detach "$mount_point" >/dev/null
mounted=0
hdiutil convert "$writable" -format UDZO -o "$dmg" >/dev/null
if [ ! -f "$dmg" ]; then
    echo "The disk image was not created: $dmg" >&2
    exit 1
fi

# Only the two named outputs belong to this builder. Back them up during
# publication so that ordinary move failures can be rolled back.
promoting=1
if [ -e "$final_app" ] || [ -L "$final_app" ]; then
    mv "$final_app" "$build_dir/previous.app"
    old_app_saved=1
fi
if [ -e "$final_dmg" ] || [ -L "$final_dmg" ]; then
    mv "$final_dmg" "$build_dir/previous.dmg"
    old_dmg_saved=1
fi
mv "$app" "$final_app"
new_app_moved=1
mv "$dmg" "$final_dmg"
new_dmg_moved=1
promoting=0

echo "Built $final_app"
echo "Built $final_dmg (installs $spec)"
