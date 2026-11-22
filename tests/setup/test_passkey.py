"""Verify the maintainers' script that changes the desktop passkey.

Tests rewrite a temporary copy of ``kilin/gui/_activation.py``, never the
repository's own file.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.gui._activation import PASSKEY_DIGEST, passkey_digest
from setup import passkey


class TestSetPasskey(unittest.TestCase):
    """Change or switch off the passkey in a copy of the module."""

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.module = Path(temporary.name) / "_activation.py"
        self.module.write_text(
            passkey.ACTIVATION_MODULE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    def _constants(self) -> tuple[str, str]:
        """Read rewritten constants without caching executed Python.

        Returns
        -------
        tuple[str, str]
            Current salt and digest from the source file.

        """

        tree = ast.parse(self.module.read_text(encoding="utf-8"))
        values = {
            node.target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in {"PASSKEY_SALT", "PASSKEY_DIGEST"}
        }
        return values["PASSKEY_SALT"], values["PASSKEY_DIGEST"]

    def test_new_passkey_gets_a_new_salt_and_digest(self) -> None:
        old_salt, _ = self._constants()
        passkey.set_passkey("New Key", self.module)
        salt, digest = self._constants()
        self.assertNotEqual(salt, old_salt)
        self.assertNotEqual(digest, PASSKEY_DIGEST)
        self.assertEqual(passkey_digest(" new key ", salt), digest)
        self.assertNotIn("New Key", self.module.read_text(encoding="utf-8"))

    def test_off_empties_the_digest_and_on_restores_it(self) -> None:
        passkey.set_passkey(None, self.module)
        self.assertEqual(self._constants()[1], "")
        passkey.set_passkey("New Key", self.module)
        salt, digest = self._constants()
        self.assertEqual(passkey_digest("New Key", salt), digest)

    def test_reenable_after_formatted_off_digest(self) -> None:
        passkey.set_passkey(None, self.module)
        source = self.module.read_text(encoding="utf-8")
        expanded = 'PASSKEY_DIGEST: Final[str] = (\n    ""\n)'
        compact = 'PASSKEY_DIGEST: Final[str] = ""'
        self.assertIn(expanded, source)
        self.module.write_text(
            source.replace(expanded, compact), encoding="utf-8"
        )
        passkey.set_passkey("New Key", self.module)
        salt, digest = self._constants()
        self.assertEqual(passkey_digest("New Key", salt), digest)

    def test_empty_passkey_and_foreign_files_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            passkey.set_passkey("  ", self.module)
        other = self.module.with_name("other.py")
        other.write_text("x = 1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            passkey.set_passkey("New Key", other)

    def test_command_line_rewrites_the_module(self) -> None:
        with patch.object(passkey, "ACTIVATION_MODULE", self.module), \
                patch("builtins.print"):
            self.assertEqual(passkey.main(["--off"]), 0)
        self.assertEqual(self._constants()[1], "")
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            passkey.main([])

    def test_default_target_is_the_shipped_module(self) -> None:
        self.assertEqual(
            passkey.ACTIVATION_MODULE,
            Path(__file__).resolve().parents[2] / "kilin/gui/_activation.py",
        )


if __name__ == "__main__":
    unittest.main()
