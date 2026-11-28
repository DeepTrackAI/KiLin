"""Verify the desktop window's passkey activation.

Tests check the passkey, the activation record and the launch gate in
``kilin.gui.app.main`` without Qt or real user files.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from kilin.gui import _activation
from kilin.gui._activation import (
    PASSKEY_DIGEST,
    activated,
    activation_record,
    ask_passkey,
    passkey_digest,
    remember_activation,
)


class TestPasskey(unittest.TestCase):
    """Accept a passkey however it is typed, and nothing else.

    The tests use their own passkey: the real one is not in the code.

    """

    def test_case_and_spaces_do_not_matter(self) -> None:
        digest = passkey_digest("Test Key")
        for typed in ("test key", "  TEST KEY \n"):
            self.assertEqual(passkey_digest(typed), digest)

    def test_other_passkeys_are_rejected(self) -> None:
        digest = passkey_digest("Test Key")
        for typed in ("", "Test", "TestKey", "Test Key 2"):
            self.assertNotEqual(passkey_digest(typed), digest)

    def test_shipped_digest_is_valid_or_disabled(self) -> None:
        self.assertRegex(PASSKEY_DIGEST, r"^(?:[0-9a-f]{64})?$")
        self.assertRegex(_activation.PASSKEY_SALT, r"^[0-9a-f]{32}$")


class TestActivationRecord(unittest.TestCase):
    """Remember an activation beside the user's preferences."""

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.record = self.folder / "kilin" / "activation.json"

    def test_record_sits_beside_preferences(self) -> None:
        preferences = self.folder / "custom" / "preferences.json"
        with patch.dict(
            "os.environ", {"KILIN_DEFAULTS_FILE": str(preferences)}
        ):
            self.assertEqual(
                activation_record(), preferences.with_name("activation.json")
            )

    def test_remembered_activation_is_kept(self) -> None:
        self.assertFalse(activated(self.record))
        self.assertTrue(remember_activation(self.record))
        self.assertTrue(activated(self.record))
        self.assertEqual(
            json.loads(self.record.read_text(encoding="utf-8")),
            {"digest": PASSKEY_DIGEST},
        )

    def test_other_or_broken_records_ask_again(self) -> None:
        self.record.parent.mkdir(parents=True)
        for text in ('{"digest": "old"}', "not json", "[]", ""):
            self.record.write_text(text, encoding="utf-8")
            self.assertFalse(activated(self.record), text)

    def test_unwritable_record_is_not_an_error(self) -> None:
        blocker = self.folder / "file"
        blocker.write_text("", encoding="utf-8")
        self.assertFalse(remember_activation(blocker / "activation.json"))

    def test_activated_account_sees_no_dialog(self) -> None:
        with patch.object(_activation, "activated", return_value=True):
            self.assertTrue(ask_passkey())  # PySide6 is never imported.

    def test_empty_digest_asks_no_one(self) -> None:
        with patch.object(_activation, "PASSKEY_DIGEST", ""), \
                patch.object(_activation, "activated", return_value=False):
            self.assertTrue(ask_passkey())


class TestLaunchGate(unittest.TestCase):
    """Ask before the window is built, and stop when the user quits."""

    def _main(self, accepted: bool) -> tuple[int, MagicMock]:
        window = MagicMock()
        qt = SimpleNamespace(QApplication=MagicMock())
        qt.QApplication.return_value.exec.return_value = 7
        modules = {
            "PySide6": SimpleNamespace(),
            "PySide6.QtWidgets": qt,
            "kilin.gui.window": SimpleNamespace(ChatWindow=window),
        }
        from kilin.gui import app

        with patch.dict(sys.modules, modules), patch.object(
            _activation, "ask_passkey", return_value=accepted
        ):
            code = app.main(["kilin"])
        connect = qt.QApplication.return_value.aboutToQuit.connect
        if accepted:
            connect.assert_called_once_with(
                window.return_value._shutdown_engine
            )
        else:
            connect.assert_not_called()
        return code, window

    def test_quitting_the_dialog_opens_no_window(self) -> None:
        code, window = self._main(accepted=False)
        self.assertEqual(code, 0)
        window.assert_not_called()

    def test_accepted_passkey_opens_the_window(self) -> None:
        code, window = self._main(accepted=True)
        self.assertEqual(code, 7)
        window.return_value.show.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
