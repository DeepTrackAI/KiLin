"""Resolve user-owned GUI data paths without importing Qt.

Packaged GUI resources are read-only. Development checkouts retain their
``sandbox/`` workspace parent; installed distributions use user data folders.
No directories are created while resolving a path.

Key Features
------------

- **Development workspace**

    Use the checkout's ``sandbox/`` when the running code belongs to a
    recognizable KíLín source tree.

- **Installed workspace and preferences**

    Use platform-specific user data and configuration directories.

Module Structure
----------------

- ``preferences_path``

    Resolve the writable application-preferences file.

- ``default_workspace_directory``

    Resolve the parent for workspaces created without an explicit path:
    the folder chosen in Settings when it exists, else the standard one.

- ``standard_workspace_directory``

    Resolve the standard parent, ignoring the folder chosen in Settings.

- ``gui_install_command``

    Tell how to install the GUI and its backend, for a checkout or an
    installed package.

Examples
--------
>>> from kilin.gui._paths import default_workspace_directory
>>> default_workspace_directory().name in ("sandbox", "workspaces")
True

"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Final

__all__ = [
    "default_workspace_directory",
    "gui_install_command",
    "preferences_path",
    "saved_workspace_folder",
    "standard_workspace_directory",
]

PREFERENCES_ENVIRONMENT: Final[str] = "KILIN_DEFAULTS_FILE"
WORKSPACES_ENVIRONMENT: Final[str] = "KILIN_WORKSPACES_DIR"


def _user_config_directory() -> Path:
    """Return the platform's writable per-user application configuration.

    Returns
    -------
    pathlib.Path
        Directory used for KíLín's user preference overrides.

    Notes
    -----
    On Linux, ``XDG_CONFIG_HOME`` takes precedence when it is absolute.
    On macOS, Application Support holds both settings and workspaces.
    On Windows, ``APPDATA`` is preferred for roaming preferences.

    """

    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "kilin"
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        return Path(base) / "kilin" if base else (
            home / "AppData" / "Roaming" / "kilin"
        )
    configured = Path(os.environ.get("XDG_CONFIG_HOME", ""))
    base = configured if configured.is_absolute() else home / ".config"
    return base / "kilin"


def _user_data_directory() -> Path:
    """Return the user-owned base directory for installed GUI workspaces.

    Returns
    -------
    pathlib.Path
        Directory containing GUI workspaces on the current platform.

    """

    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "kilin"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) / "kilin" if base else (
            home / "AppData" / "Local" / "kilin"
        )
    configured = Path(os.environ.get("XDG_DATA_HOME", ""))
    base = (
        configured if configured.is_absolute()
        else home / ".local" / "share"
    )
    return base / "kilin"


def _development_checkout() -> Path | None:
    """Find the KíLín source root only when running from its source tree.

    Returns
    -------
    pathlib.Path | None
        Validated development root, or None for an installed package.

    Notes
    -----
    A site-packages directory does not become a checkout merely because
    it contains the installed ``kilin`` package. Requiring the project
    metadata and test tree avoids writing into an ordinary wheel install.

    """

    root = Path(__file__).resolve().parents[2]
    if (
        (root / "pyproject.toml").is_file()
        and (root / "tests" / "kilin").is_dir()
        and (root / "kilin" / "gui" / "engine.py").is_file()
    ):
        return root
    return None


def preferences_path() -> Path:
    """Return the writable application-preferences path.

    Returns
    -------
    pathlib.Path
        Explicit absolute ``KILIN_DEFAULTS_FILE`` path, or a user
        configuration file. The packaged defaults are never the default.

    Raises
    ------
    ValueError
        If an explicit override is relative to the working directory.

    """

    override = os.environ.get(PREFERENCES_ENVIRONMENT)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"{PREFERENCES_ENVIRONMENT} must be an absolute path."
            )
        return path
    return _user_config_directory() / "preferences.json"


def default_workspace_directory() -> Path:
    """Return the parent directory for GUI-created project workspaces.

    Returns
    -------
    pathlib.Path
        Explicit ``KILIN_WORKSPACES_DIR`` directory, else the folder chosen
        in Settings when it still exists, else the standard folder.

    Raises
    ------
    ValueError
        If an explicit override is relative to the working directory.

    Notes
    -----
    A chosen folder that was removed or is not mounted (e.g. a cloud drive)
    is ignored, so KíLín still starts in the standard folder. An explicit
    workspace for one project overrides this default. The resolver does not
    create directories or change existing projects.

    """

    if not os.environ.get(WORKSPACES_ENVIRONMENT):
        saved = saved_workspace_folder()
        if saved:
            path = Path(saved).expanduser()
            if path.is_dir():
                return path.resolve()
    return standard_workspace_directory()


def standard_workspace_directory() -> Path:
    """Return the standard projects folder, ignoring the Settings choice.

    Returns
    -------
    pathlib.Path
        Explicit ``KILIN_WORKSPACES_DIR`` directory, checkout sandbox,
        or user-owned installed-application workspace directory.

    Raises
    ------
    ValueError
        If an explicit override is relative to the working directory.

    """

    override = os.environ.get(WORKSPACES_ENVIRONMENT)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"{WORKSPACES_ENVIRONMENT} must be an absolute path."
            )
        return path.resolve()
    checkout = _development_checkout()
    if checkout is not None:
        return checkout / "sandbox"
    return _user_data_directory() / "workspaces"


def saved_workspace_folder() -> str:
    """Return the projects folder chosen in Settings, or "" for none.

    Returns
    -------
    str
        Saved absolute folder path, possibly no longer present, or "".

    Notes
    -----
    An unreadable or invalid preferences file counts as no choice, so a
    broken file never stops KíLín from starting.

    """

    # Imported here: the preferences module imports this one.
    from kilin.gui.preferences import app_defaults

    try:
        return str(app_defaults()["workspace_folder"])
    except (OSError, ValueError, KeyError):
        return ""


def gui_install_command(backend: str) -> str:
    """Tell how to install the GUI and its backend in this KíLín.

    A source checkout installs them with uv; an installed package (from
    PyPI or the macOS app) with pip, into the same Python, using the
    ``app`` extra that picks the backend for the platform.

    Parameters
    ----------
    backend: str
        Backend that is missing: ``mlx`` or ``torch``.

    Returns
    -------
    str
        The command to run.

    """

    if _development_checkout() is not None:
        return f"uv sync --extra gui --extra {backend}"
    return f'{Path(sys.executable).name} -m pip install "kilin[app]"'
