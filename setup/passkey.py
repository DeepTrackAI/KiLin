"""Change or switch off the desktop window's passkey.

This auxiliary repository script is for KíLín's maintainers and is not part
of the released ``kilin`` package. It rewrites the salt and digest in
``kilin/gui/_activation.py``: a new passkey gets a fresh salt, so every user
is asked again, and switching off leaves an empty digest, so no one is
asked. The passkey itself is never written. Commit the rewritten module so
that new installs and updates follow it.

Key Features
------------
- **Fresh salt per passkey**

    A new passkey never reuses the previous salt.

- **Switch off without removing code**

    An empty digest makes ``ask_passkey`` open the window directly;
    setting a passkey again switches the prompt back on.

Module Structure
----------------
- ``ACTIVATION_MODULE``

    The module whose constants are rewritten.

- ``set_passkey``

    Write a new passkey's salt and digest, or none, into the module.

- ``main``

    Change or switch off the passkey from the command line.

Examples
--------
From the repository root::

    uv run python setup/passkey.py --passkey "new passkey"
    uv run python setup/passkey.py --off

"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import re
import secrets
import sys
from typing import Final

if __package__ in (None, ""):  # Run as a script from the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kilin.gui._activation import passkey_digest  # noqa: E402

__all__ = [
    "ACTIVATION_MODULE",
    "main",
    "set_passkey",
]

ACTIVATION_MODULE: Final[Path] = (
    Path(__file__).resolve().parents[1] / "kilin" / "gui" / "_activation.py"
)
_SALT_LINE: Final[re.Pattern[str]] = re.compile(
    r'^PASSKEY_SALT: Final\[str\] = "[0-9a-f]*"$', re.MULTILINE
)
_DIGEST_LINES: Final[re.Pattern[str]] = re.compile(
    r'^PASSKEY_DIGEST: Final\[str\] = \(\n    "[0-9a-f]*"\n\)$',
    re.MULTILINE,
)


def set_passkey(passkey: str | None, module: Path | None = None) -> Path:
    """Write a new passkey's salt and digest, or none, into the module.

    Parameters
    ----------
    passkey: str | None
        The new passkey, or None to switch the prompt off.
    module: pathlib.Path | None, optional
        The file to rewrite. Defaults to ``ACTIVATION_MODULE``.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    ValueError
        If the passkey is empty, or the file's constants are not found.

    """

    path = ACTIVATION_MODULE if module is None else module
    if passkey is not None and not passkey.strip():
        raise ValueError("The passkey cannot be empty.")
    text = path.read_text(encoding="utf-8")
    salt_line = _SALT_LINE.search(text)
    if not (salt_line and _DIGEST_LINES.search(text)):
        raise ValueError(f"No passkey constants found in {path}.")
    if passkey is None:
        salt, digest = salt_line.group(0).split('"')[1], ""
    else:
        salt = secrets.token_hex(16)
        digest = passkey_digest(passkey, salt)
    text = _SALT_LINE.sub(
        f'PASSKEY_SALT: Final[str] = "{salt}"', text, count=1
    )
    text = _DIGEST_LINES.sub(
        f'PASSKEY_DIGEST: Final[str] = (\n    "{digest}"\n)', text, count=1
    )
    path.write_text(text, encoding="utf-8")
    return path


def main(arguments: Sequence[str] | None = None) -> int:
    """Change or switch off the passkey from the command line.

    Parameters
    ----------
    arguments: Sequence[str] | None, optional
        Arguments without the program name. None reads the command line.

    Returns
    -------
    int
        Zero once the file is written.

    """

    parser = argparse.ArgumentParser(
        prog="setup/passkey.py",
        description="Change or switch off the desktop window's passkey.",
    )
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--passkey", help="the new passkey")
    choice.add_argument(
        "--off", action="store_true", help="ask no one for a passkey"
    )
    options = parser.parse_args(arguments)
    try:
        path = set_passkey(None if options.off else options.passkey)
    except ValueError as error:
        parser.error(str(error))
    print(
        f"Passkey prompt switched off in {path}; commit it." if options.off
        else f"New passkey written to {path}; commit it. Every user will "
        "be asked again."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
