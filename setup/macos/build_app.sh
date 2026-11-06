#!/bin/bash
# Build KíLín.app and KíLín.dmg for Apple-silicon Macs.
#
#   sh setup/macos/build_app.sh           the app installs KíLín from PyPI
#   sh setup/macos/build_app.sh --wheel   builds a wheel from this checkout
#                                         and puts it inside the app, to
#                                         test before a PyPI release
#
# Output: dist/KíLín.app and dist/KíLín.dmg. Set CODESIGN_IDENTITY to sign
# with a Developer ID; otherwise the app is signed ad hoc, and users open it
# the first time with right-click > Open. The disk image's window is laid
# out through Finder, so the first build asks to let the terminal control
# Finder. dmg-background.png (and @2x) is the window's background.
set -euo pipefail

project_dir=$(cd "$(dirname "$0")/../.." && pwd)
cd "$project_dir"
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -n 1)
dist="$project_dir/dist"
app="$dist/KíLín.app"
bundle_id=${KILIN_BUNDLE_ID:-"org.kilin.app"}
uv_binary=${UV_BINARY:-$(command -v uv || true)}

if [ -z "$uv_binary" ] || [ ! -x "$uv_binary" ]; then
    echo "uv is needed to build the app; install it or set UV_BINARY." >&2
    exit 1
fi
if ! file "$uv_binary" | grep -q "arm64"; then
    echo "$uv_binary is not an Apple-silicon uv binary." >&2
    exit 1
fi

rm -rf "$app" "$dist/KíLín.dmg"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"

# What the launcher installs on first launch.
spec="kilin[app]==$version"
# A bundled wheel's checksum: a rebuilt app with the same version still
# reinstalls KíLín. A PyPI version is uploaded only once, so needs none.
wheel_sha=""
if [ "${1:-}" = "--wheel" ]; then
    rm -f "$dist"/kilin-*.whl
    "$uv_binary" build --wheel --out-dir "$dist"
    rm -rf build
    wheel=$(basename "$(ls "$dist"/kilin-*.whl)")
    cp "$dist/$wheel" "$app/Contents/Resources/"
    spec="@RESOURCES@/$wheel[app]"
    wheel_sha=$(shasum -a 256 "$dist/$wheel" | cut -d " " -f 1)
fi

sed -e "s|@KILIN_SPEC@|$spec|" -e "s|@KILIN_WHEEL_SHA@|$wheel_sha|" \
    setup/macos/launcher.sh >"$app/Contents/MacOS/KiLin"
chmod +x "$app/Contents/MacOS/KiLin"
cp "$uv_binary" "$app/Contents/Resources/uv"

# The app icon, from the same artwork as the window icon.
iconset="$(mktemp -d)/KiLin.iconset"
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

# The disk image: the app beside a link to Applications, to drag onto,
# over a background that says so. Finder keeps the window's layout in the
# image, so it is laid out on a writable copy, then compressed.
staging=$(mktemp -d)
cp -R "$app" "$staging/"
ln -s /Applications "$staging/Applications"
mkdir "$staging/.background"
tiffutil -cathidpicheck setup/macos/dmg-background.png \
    setup/macos/dmg-background@2x.png \
    -out "$staging/.background/background.tiff" >/dev/null 2>&1
writable="$staging.dmg"
hdiutil create -volname "KíLín" -srcfolder "$staging" -ov -format UDRW \
    "$writable" >/dev/null
mount_point=$(mktemp -d)
hdiutil attach "$writable" -readwrite -noverify -noautoopen \
    -mountpoint "$mount_point" >/dev/null
# 640 x 400 window, the icons centred on the background's arrow.
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
rm -f "$dist/KíLín.dmg"
hdiutil convert "$writable" -format UDZO -o "$dist/KíLín.dmg" >/dev/null
rm -rf "$staging" "$writable" "$mount_point"

echo "Built $app"
echo "Built $dist/KíLín.dmg (installs $spec)"
