#!/bin/bash
# KíLín.app's launcher, copied into Contents/MacOS by build_app.sh.
#
# The app holds only this script, its icon and uv. On first launch, and
# after the app is updated, it installs KíLín (``kilin[app]``) from PyPI, or
# from the bundled wheel, into a stable versioned runtime in KíLín's
# Application Support folder, then opens the desktop window. Model downloads
# happen later, inside the application.
set -u

# Filled in by build_app.sh; the checksum is empty for a PyPI install.
KILIN_SPEC="@KILIN_SPEC@"
KILIN_WHEEL_SHA="@KILIN_WHEEL_SHA@"

contents=$(cd "$(dirname "$0")/.." && pwd)
resources="$contents/Resources"
support=${KILIN_APP_SUPPORT:-"$HOME/Library/Application Support/kilin"}
# Only what the app installs; projects and preferences stay beside it.
runtime_root="$support/runtime"
legacy_runtime="$runtime_root/app"
releases="$runtime_root/releases"
active_file="$runtime_root/active"
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

# Release environments are never moved after installation. Python virtual
# environments can contain absolute paths in their console entry points.
# Hash the exact installation source and optional wheel checksum so an app
# rebuilt from a changed wheel gets a different, permanent runtime path.
if ! digest=$(printf '%s' "$build" | shasum -a 256); then
    fail "KíLín cannot calculate its release identifier."
fi
release_id=${digest%% *}
case "$release_id" in
    ''|*[!0-9a-f]*) fail "Invalid KíLín release identifier." ;;
esac
if [ "${#release_id}" -ne 64 ]; then
    fail "Invalid KíLín release identifier."
fi
target_name="releases/$release_id"
target_runtime="$runtime_root/$target_name"

# An active pointer contains either a relative release name or the legacy
# 'app' directory. Never use an unvalidated value as a filesystem path.
# Without a pointer, the existing legacy installation remains active
# until a complete replacement has been committed.
read_active() {
    runtime=""
    if [ -L "$active_file" ]; then
        fail "KíLín's active runtime pointer must not be a symlink."
    fi
    if [ -e "$active_file" ]; then
        if [ ! -f "$active_file" ]; then
            fail "KíLín's active runtime pointer is not a file."
        fi
        if ! selection=$(cat "$active_file"); then
            fail "KíLín cannot read its active runtime pointer."
        fi
        case "$selection" in
            app) runtime="$legacy_runtime" ;;
            releases/*)
                selected_id=${selection#releases/}
                case "$selected_id" in
                    ''|*[!0-9a-f]*)
                        fail "Invalid KíLín active release identifier."
                        ;;
                esac
                if [ "${#selected_id}" -ne 64 ]; then
                    fail "Invalid KíLín active release identifier."
                fi
                runtime="$releases/$selected_id"
                ;;
            *) fail "Invalid KíLín active runtime pointer." ;;
        esac
    elif [ -f "$legacy_runtime/.installed" ]; then
        runtime="$legacy_runtime"
    fi
    if [ -n "$runtime" ] && [ -L "$runtime" ]; then
        fail "KíLín's active runtime must not be a symlink."
    fi
}

installed_for_build() {
    [ -n "$1" ] && [ -x "$1/bin/kilin-gui" ] &&
        [ "$(cat "$1/.installed" 2>/dev/null)" = "$build" ]
}

# Atomic activation writes a relative pointer inside runtime/, never a
# replacement virtual environment. A crash before rename leaves the old
# active value intact; a crash after rename selects a complete release.
activate_release() {
    if [ -d "$active_file" ] || [ -L "$active_file" ]; then
        fail "KíLín cannot replace its active runtime pointer."
    fi
    if ! pointer_tmp=$(mktemp "$runtime_root/.active.XXXXXX"); then
        fail "KíLín cannot prepare its active runtime pointer."
    fi
    if ! printf '%s\n' "$target_name" >"$pointer_tmp"; then
        rm -f "$pointer_tmp"
        fail "KíLín cannot write its active runtime pointer."
    fi
    if ! mv -f "$pointer_tmp" "$active_file"; then
        rm -f "$pointer_tmp"
        fail "KíLín cannot activate the installed release."
    fi
}

read_active
if ! installed_for_build "$runtime"; then
    lock_timeout=${KILIN_INSTALL_LOCK_TIMEOUT:-300}
    case "$lock_timeout" in
        ''|*[!0-9]*) fail "Invalid KíLín installation lock timeout." ;;
    esac
    if [ "${#lock_timeout}" -gt 3 ] || [ "$lock_timeout" -gt 600 ]; then
        fail "KíLín installation lock timeout must be 0-600 seconds."
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

    # Do not remove an unowned directory left by an older launcher. Such a
    # launcher may still be running and does not participate in this lock.
    if [ -d "$runtime_root/.installing" ]; then
        fail "An older KíLín installation may be running. Close it and inspect the legacy .installing lock folder."
    fi

    # Another launcher may have completed and activated this version while
    # we waited, so re-read the pointer under the installation lock.
    read_active
    if ! installed_for_build "$runtime"; then
        if [ "$runtime" = "$target_runtime" ]; then
            fail "The active KíLín runtime is incomplete. Repair it manually."
        fi
        if ! mkdir -p "$releases"; then
            fail "KíLín cannot create its releases folder."
        fi
        if ! installed_for_build "$target_runtime"; then
            # Delete only an incomplete *inactive* attempt for this exact
            # build. An active runtime and every other release are untouched.
            if [ -e "$target_runtime" ] || [ -L "$target_runtime" ]; then
                if ! rm -rf "$target_runtime"; then
                    fail "KíLín cannot clean an incomplete release."
                fi
            fi
            if [ -n "$runtime" ] && [ -f "$runtime/.installed" ]; then
                notify "Updating KíLín. The previous version is kept."
            else
                notify "Installing KíLín. The first launch takes a few minutes."
            fi
            export UV_PYTHON_INSTALL_DIR="$runtime_root/python"
            export UV_CACHE_DIR="$runtime_root/uv-cache"
            if ! {
                echo "Installing $spec on $(date)"
                "$resources/uv" venv --python 3.12 "$target_runtime" &&
                    "$resources/uv" pip install \
                        --python "$target_runtime/bin/python" "$spec"
            } >"$log" 2>&1; then
                rm -rf "$target_runtime"
                fail "KíLín could not be installed. Check the internet connection and try again. Details: $log"
            fi
            # Verify both entry points and installed distribution metadata
            # before recording a release as complete.
            if [ ! -x "$target_runtime/bin/python" ] ||
                [ ! -x "$target_runtime/bin/kilin-gui" ] ||
                ! "$target_runtime/bin/python" -c \
                    'from importlib.metadata import version; version("kilin")' \
                    >>"$log" 2>&1; then
                rm -rf "$target_runtime"
                fail "KíLín installation validation failed. Details: $log"
            fi
            if ! printf '%s\n' "$build" >"$target_runtime/.installed"; then
                rm -rf "$target_runtime"
                fail "KíLín could not record its installation."
            fi
        fi
        # This is the only commit point. On failure, the old active
        # installation remains selected and must not launch silently.
        activate_release
        # Only the application-owned cache may be removed. Never delete
        # an arbitrary UV_CACHE_DIR inherited from the user environment.
        rm -rf "$runtime_root/uv-cache"
        notify "KíLín is installed."
    fi
    # Do not retain the lock while the desktop GUI is running.
    exec 9>&-
fi

read_active
if ! installed_for_build "$runtime"; then
    fail "KíLín's active runtime is unavailable or incomplete."
fi
exec "$runtime/bin/kilin-gui"
