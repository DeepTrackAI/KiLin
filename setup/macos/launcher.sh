#!/bin/bash
# KíLín.app's launcher, copied into Contents/MacOS by build_app.sh.
#
# The app holds only this script, its icon and uv. On first launch, and
# after the app is updated, it installs KíLín (``kilin[app]``) from PyPI, or
# from the wheel the app was built with, into runtime/ in KíLín's folder in
# Application Support, then opens the desktop window. The models are downloaded later,
# by KíLín setup inside the window.
set -u

# Filled in by build_app.sh.
KILIN_SPEC="@KILIN_SPEC@"

contents=$(cd "$(dirname "$0")/.." && pwd)
resources="$contents/Resources"
support=${KILIN_APP_SUPPORT:-"$HOME/Library/Application Support/kilin"}
# Only what the app installs; projects and preferences stay beside it.
runtime_root="$support/runtime"
runtime="$runtime_root/app"
log="$runtime_root/install.log"

# Show a message; KíLín has no window yet while it installs.
alert() {
    /usr/bin/osascript -e "display alert \"KíLín\" message \"$1\" as critical" >/dev/null 2>&1
}

notify() {
    /usr/bin/osascript -e "display notification \"$1\" with title \"KíLín\"" >/dev/null 2>&1
}

if [ "$(uname -m)" != "arm64" ]; then
    alert "KíLín needs a Mac with Apple silicon (M1 or later)."
    exit 1
fi

# A wheel inside the app is installed from there.
spec=${KILIN_SPEC//@RESOURCES@/$resources}
mkdir -p "$runtime_root"

# Install, or reinstall after an update; a second launch meanwhile waits.
if [ "$(cat "$runtime/.installed" 2>/dev/null)" != "$spec" ]; then
    lock="$runtime_root/.installing"
    until mkdir "$lock" 2>/dev/null; do sleep 1; done
    trap 'rmdir "$lock" 2>/dev/null' EXIT
    if [ "$(cat "$runtime/.installed" 2>/dev/null)" != "$spec" ]; then
        notify "Installing KíLín. The first launch takes a few minutes."
        # Python and uv's downloads stay in runtime/, so deleting runtime/
        # removes the installed KíLín and nothing else.
        export UV_PYTHON_INSTALL_DIR="$runtime_root/python"
        export UV_CACHE_DIR="$runtime_root/uv-cache"
        {
            echo "Installing $spec on $(date)"
            rm -rf "$runtime" &&
            "$resources/uv" venv --python 3.12 "$runtime" &&
            "$resources/uv" pip install --python "$runtime/bin/python" "$spec"
        } >"$log" 2>&1
        if [ $? -ne 0 ] || [ ! -x "$runtime/bin/kilin-gui" ]; then
            rm -rf "$runtime"
            alert "KíLín could not be installed. Check the internet connection and open KíLín again. Details: $log"
            exit 1
        fi
        echo "$spec" >"$runtime/.installed"
        # The downloads are not needed once installed.
        rm -rf "$UV_CACHE_DIR"
        notify "KíLín is installed."
    fi
    rmdir "$lock" 2>/dev/null
    trap - EXIT
fi

exec "$runtime/bin/kilin-gui"
