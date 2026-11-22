"""Check the desktop passkey before opening KíLín's main window.

The desktop GUI requests an invitation passkey on first launch for a user
account. Successful activation is remembered in ``activation.json``
beside GUI preferences, allowing the main window to open directly on
subsequent launches. The passkey verifier uses PBKDF2-SHA256.

Key Features
------------

- **Per-account activation**

    Remember a successful passkey check between desktop launches.

- **Forgiving entry**

    Letter case and surrounding spaces do not matter.

- **Local activation record**

    Store accepted activation next to the user's GUI preferences.

- **Desktop startup integration**

    ``kilin.gui.app.main`` calls ``ask_passkey`` before constructing
    the main window.

Module Structure
----------------

- ``passkey_digest``

    Return the salted hash of a typed passkey.

- ``activation_record``

    Return the file that remembers the activation.

- ``activated``

    Return whether this user account entered the current passkey.

- ``remember_activation``

    Record the activation.

- ``ask_passkey``

    Show the passkey dialog unless this account is already activated.

Examples
--------
>>> from kilin.gui._activation import passkey_digest, PASSKEY_DIGEST
>>> passkey_digest("wrong") == PASSKEY_DIGEST
False

"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from kilin.gui._paths import preferences_path

__all__ = [
    "ACTIVATION_FILE",
    "PASSKEY_DIGEST",
    "activated",
    "activation_record",
    "ask_passkey",
    "passkey_digest",
    "remember_activation",
]

# PBKDF2-SHA256 verifier of the normalized passkey with a stored salt.
PASSKEY_SALT: Final[str] = "ca9f2b0f73fe084676c6b12b7f5d15b8"
PASSKEY_DIGEST: Final[str] = (
    "8398f6905a89a993838848ed4ff7ef960863b5cc1a0baae937387f6faed7dfae"
)
PASSKEY_ITERATIONS: Final[int] = 200_000
ACTIVATION_FILE: Final[str] = "activation.json"

_TITLE: Final[str] = "Activate KíLín"
_MESSAGE: Final[str] = (
    "<b>Welcome to KíLín.</b><br><br>Enter your passkey to activate "
    "KíLín.<br>To obtain a passkey, contact the KíLín core developers."
)
_WRONG: Final[str] = "That passkey is not right. Please try again."


def passkey_digest(passkey: str, salt: str | None = None) -> str:
    """Return the salted hash of a typed passkey.

    Parameters
    ----------
    passkey: str
        The passkey as typed.
    salt: str | None, optional
        Salt in hexadecimal. Defaults to ``PASSKEY_SALT``.

    Returns
    -------
    str
        PBKDF2-SHA256 digest, in hexadecimal, of the passkey without
        surrounding spaces and in lower case.

    """

    return hashlib.pbkdf2_hmac(
        "sha256",
        passkey.strip().casefold().encode("utf-8"),
        bytes.fromhex(PASSKEY_SALT if salt is None else salt),
        PASSKEY_ITERATIONS,
    ).hex()


def activation_record() -> Path:
    """Return the file that remembers the activation.

    Returns
    -------
    pathlib.Path
        ``activation.json`` in the folder of the user's preferences file.

    """

    return preferences_path().with_name(ACTIVATION_FILE)


def activated(record: Path | None = None) -> bool:
    """Return whether this user account entered the current passkey.

    Parameters
    ----------
    record: pathlib.Path | None, optional
        The activation file. Defaults to ``activation_record()``.

    Returns
    -------
    bool
        True when the file holds the current passkey's digest; False when
        it is missing, unreadable or holds another digest.

    """

    try:
        path = record if record is not None else activation_record()
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(saved, dict) and saved.get("digest") == PASSKEY_DIGEST


def remember_activation(record: Path | None = None) -> bool:
    """Record the activation.

    Parameters
    ----------
    record: pathlib.Path | None, optional
        The activation file. Defaults to ``activation_record()``.

    Returns
    -------
    bool
        Whether the file was written. A failure only means the passkey is
        asked again at the next launch.

    """

    try:
        path = record if record is not None else activation_record()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"digest": PASSKEY_DIGEST}), encoding="utf-8"
        )
    except (OSError, ValueError):
        return False
    return True


def ask_passkey() -> bool:
    """Show the passkey dialog unless this account is already activated.

    A ``QApplication`` must exist. The dialog stays open on a wrong
    passkey and says so.

    Returns
    -------
    bool
        True when no check is required, activation is already present,
        or the correct passkey was entered; False when the user quit.

    """

    if not PASSKEY_DIGEST or activated():
        return True

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QDialogButtonBox,
        QLabel,
        QLineEdit,
        QVBoxLayout,
    )

    dialog = QDialog()
    dialog.setWindowTitle(_TITLE)
    dialog.setMinimumWidth(420)
    icon = QApplication.windowIcon()
    if not icon.isNull():
        dialog.setWindowIcon(icon)
    layout = QVBoxLayout(dialog)
    message = QLabel(_MESSAGE)
    message.setWordWrap(True)
    message.setTextFormat(Qt.RichText)
    layout.addWidget(message)
    field = QLineEdit()
    field.setEchoMode(QLineEdit.Password)
    field.setPlaceholderText("Passkey")
    layout.addWidget(field)
    error = QLabel("")
    error.setStyleSheet("color: #c0392b;")
    layout.addWidget(error)
    buttons = QDialogButtonBox()
    activate = buttons.addButton("Activate", QDialogButtonBox.AcceptRole)
    buttons.addButton("Quit", QDialogButtonBox.RejectRole)
    activate.setDefault(True)
    layout.addWidget(buttons)

    def check() -> None:
        """Accept the right passkey, or say that it is wrong.

        Returns
        -------
        None

        """

        if passkey_digest(field.text()) == PASSKEY_DIGEST:
            dialog.accept()
            return
        error.setText(_WRONG)
        field.selectAll()
        field.setFocus()

    buttons.accepted.connect(check)
    buttons.rejected.connect(dialog.reject)
    if dialog.exec() != QDialog.Accepted:
        return False
    remember_activation()
    return True
