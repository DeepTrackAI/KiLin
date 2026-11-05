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
# the first time with right-click > Open.
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
if [ "${1:-}" = "--wheel" ]; then
    rm -f "$dist"/kilin-*.whl
    "$uv_binary" build --wheel --out-dir "$dist"
    rm -rf build
    wheel=$(basename "$(ls "$dist"/kilin-*.whl)")
    cp "$dist/$wheel" "$app/Contents/Resources/"
    spec="@RESOURCES@/$wheel[app]"
fi

sed "s|@KILIN_SPEC@|$spec|" setup/macos/launcher.sh >"$app/Contents/MacOS/KiLin"
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

# The disk image: the app beside a link to Applications, to drag onto.
staging=$(mktemp -d)
cp -R "$app" "$staging/"
ln -s /Applications "$staging/Applications"
hdiutil create -volname "KíLín" -srcfolder "$staging" -ov -format UDZO \
    "$dist/KíLín.dmg" >/dev/null
rm -rf "$staging"

echo "Built $app"
echo "Built $dist/KíLín.dmg (installs $spec)"
