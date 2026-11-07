#!/bin/bash
# KíLín.app's launcher, copied into Contents/MacOS by build_app.sh.
#
# The app holds only this script, its icon and uv. On first launch, and
# after the app is updated, it installs KíLín (``kilin[app]``) from PyPI, or
# from the wheel the app was built with, into runtime/ in KíLín's folder in
# Application Support, then opens the desktop window. The models are downloaded later,
# by KíLín setup inside the window.
set -u

# Filled in by build_app.sh; the checksum is empty for a PyPI install.
KILIN_SPEC="@KILIN_SPEC@"
KILIN_WHEEL_SHA="@KILIN_WHEEL_SHA@"

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

fail() {
    printf '%s\n' "$1" >&2
    alert "$1"
    exit 1
}

if [ "$(uname -m)" != "arm64" ]; then
    alert "KíLín needs a Mac with Apple silicon (M1 or later)."
    exit 1
fi

# A wheel inside the app is installed from there.
spec=${KILIN_SPEC//@RESOURCES@/$resources}
# The running GUI and engine can offer an exact backend-repair command.
# Point to this app's uv and original wheel/version; runtime has no pip.
export KILIN_APP_UV="$resources/uv"
export KILIN_APP_SPEC="$spec"
# What is installed: the spec, plus the wheel's checksum, so a rebuilt
# wheel with an unchanged version and file name is installed too.
build="$spec${KILIN_WHEEL_SHA:+ $KILIN_WHEEL_SHA}"
if ! mkdir -p "$runtime_root"; then
    fail "KíLín cannot create its runtime folder. Check permissions and disk space."
fi

# Install, or reinstall after an update. macOS lockf holds a kernel lock on
# an open descriptor, released on exit even after a crash. Keep the lock file
# itself: unlinking it would let concurrent launchers lock different inodes.
if [ "$(cat "$runtime/.installed" 2>/dev/null)" != "$build" ]; then
    lock_timeout=${KILIN_INSTALL_LOCK_TIMEOUT:-300}
    case "$lock_timeout" in
        ''|*[!0-9]*) fail "Invalid KíLín installation lock timeout." ;;
    esac
    if [ "${#lock_timeout}" -gt 3 ] || [ "$lock_timeout" -gt 600 ]; then
        fail "KíLín installation lock timeout must be 0–600 seconds."
    fi
    if ! command -v lockf >/dev/null 2>&1; then
        fail "The macOS lockf utility is required to install KíLín."
    fi
    lock="$runtime_root/.install.lock"
    if ! exec 9>>"$lock"; then
        fail "KíLín cannot open its installation lock. Check permissions."
    fi
    if ! lockf -s -t "$lock_timeout" 9; then
        fail "KíLín is already installing or its lock is unavailable. Try again later."
    fi

    # The previous launcher used an unowned directory lock. It cannot be
    # reclaimed safely: an older app may still be installing. Leave it for
    # explicit inspection, and never remove it automatically.
    if [ -d "$runtime_root/.installing" ]; then
        fail "An older KíLín installation may be running. Close it and inspect the legacy .installing lock folder."
    fi

    # Recheck while locked: another launcher may have completed the install.
    if [ "$(cat "$runtime/.installed" 2>/dev/null)" != "$build" ]; then
        if [ -e "$runtime/.installed" ]; then
            notify "Updating KíLín. This takes a few minutes; projects and settings are kept."
        else
            notify "Installing KíLín. The first launch takes a few minutes."
        fi
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
        if ! printf '%s\n' "$build" >"$runtime/.installed"; then
            rm -rf "$runtime"
            fail "KíLín could not record its installation. Check permissions."
        fi
        # The downloads are not needed once installed.
        rm -rf "$UV_CACHE_DIR"
        notify "KíLín is installed."
    fi
    # The desktop process must not hold the installer lock after startup.
    exec 9>&-
fi

exec "$runtime/bin/kilin-gui"
