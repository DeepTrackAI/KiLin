"""Resolve the user-level plugin directory without creating files.

Key Features
------------
- Platform-specific application data, independent of workspaces and Qt.
- An absolute override for explicit installations and isolated tests.
- One captured runtime directory for both plugin categories.

Module Structure
----------------
- ``plugin_directory`` resolves the configured location.
- ``_runtime_directory`` captures it on first catalog access.

Examples
--------
>>> from kilin.plugins import plugin_directory
>>> plugin_directory().is_absolute()
True

"""

from functools import lru_cache
import os
from pathlib import Path
import sys

__all__ = ["plugin_directory"]


def _resolve_directory() -> Path:
    """Resolve the user plugin directory without reading any plugin code.

    Returns
    -------
    pathlib.Path
        Absolute user-level directory; it need not exist.

    Raises
    ------
    ValueError
        If the explicit KILIN_PLUGINS_DIR override is relative.

    """

    override = os.environ.get("KILIN_PLUGINS_DIR")
    if override:
        path = Path(override)
        if not path.is_absolute():
            raise ValueError("KILIN_PLUGINS_DIR must be an absolute path.")
        return path.resolve()
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", ""))
        if not base.is_absolute():
            base = home / "AppData" / "Local"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", ""))
        if not base.is_absolute():
            base = home / ".local" / "share"
    return (base / "kilin" / "plugins").resolve()


@lru_cache(maxsize=1)
def _runtime_directory() -> Path:
    """Capture the directory for the lifetime of the runtime catalogs."""

    return _resolve_directory()


def plugin_directory() -> Path:
    """Return the configured directory captured for this process.

    Returns
    -------
    pathlib.Path
        Absolute user-level directory, not created by this query.

    Notes
    -----
    Set the override before first use of this function or either catalog.
    Changing the environment afterwards requires a process restart.

    """

    return _runtime_directory()
