"""Resolve user-owned GUI data paths without importing Qt.

Packaged GUI resources are read-only. Development checkouts retain their
``sandbox/`` workspace parent; installed distributions use the user's
``Documents/KíLín`` folder.
No directories are created while resolving a path.

Key Features
------------

- **Development workspace**

    Use the checkout's ``sandbox/`` when the running code belongs to a
    recognizable KíLín source tree.

- **Installed workspace and preferences**

    Keep projects in ``Documents/KíLín`` and preferences in the platform's
    configuration directory.

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

- ``projects_folder_confirmed``

    Tell whether the user confirmed a projects folder that still exists.

Examples
--------
>>> from kilin.gui._paths import default_workspace_directory
>>> default_workspace_directory().name in ("sandbox", "KíLín")
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
    "projects_folder_confirmed",
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
    On macOS, Application Support holds the settings.
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


def _user_projects_directory() -> Path:
    """Return the standard projects folder for installed copies of KíLín.

    Returns
    -------
    pathlib.Path
        ``KíLín`` in the user's Documents folder, on every platform.

    Notes
    -----
    Projects are user documents: a visible folder is easy to find, back up
    and move. The folder is created only when it is confirmed in KíLín
    setup or a project needs it.

    """

    return Path.home() / "Documents" / "KíLín"


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
        or ``Documents/KíLín`` in the user's home for an installed copy.

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
    return _user_projects_directory()


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


def projects_folder_confirmed() -> bool:
    """Return whether the user confirmed a projects folder that exists.

    Returns
    -------
    bool
        True when a folder is saved in the preferences and is present.

    Notes
    -----
    KíLín setup asks for the folder until it is confirmed. A saved folder
    that was removed or is not mounted counts as unconfirmed, so setup asks
    again instead of silently using another folder.

    """

    saved = saved_workspace_folder()
    return bool(saved) and Path(saved).expanduser().is_dir()
