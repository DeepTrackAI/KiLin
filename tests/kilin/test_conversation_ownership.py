"""Verify cross-process ownership of canonical conversation JSON files.

Tests exercise real OS file locks using local temporary directories and
short-lived Python subprocesses. No backend, network, or GUI is required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import errno
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.conversation_ownership import (
    ConversationFileLock,
    ConversationLockedError,
    conversation_write_ownership,
)


_CHILD = """
import sys
from kilin.conversation_ownership import ConversationFileLock

with ConversationFileLock(sys.argv[1]):
    print('LOCKED', flush=True)
    sys.stdin.readline()
"""


class TestConversationFileLock(unittest.TestCase):
    """Enforce single-writer ownership and reliable lease cleanup."""

    def setUp(self) -> None:
        """Create an isolated conversation location."""

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "traces" / "conversation_test.json"

    def test_acquire_release_and_companion_file(self) -> None:
        """Acquire once, release idempotently, and keep the sidecar."""

        owner = ConversationFileLock(self.path)
        self.assertFalse(owner.holds(self.path))
        with owner:
            self.assertTrue(owner.holds(self.path))
            self.assertTrue(owner.lock_path.is_file())
            self.assertEqual(
                owner.lock_path.name, "conversation_test.json.lock"
            )
            with self.assertRaisesRegex(RuntimeError, "already held"):
                owner.acquire()
        self.assertFalse(owner.holds(self.path))
        owner.release()
        self.assertTrue(owner.lock_path.exists())
        with ConversationFileLock(self.path) as reopened:
            self.assertTrue(reopened.holds(self.path))

    def test_two_instances_conflict_without_modifying_json(self) -> None:
        """Reject another in-process owner without touching saved bytes."""

        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b'{"checkpoint_id": "saved"}\n')
        before = self.path.read_bytes()
        with ConversationFileLock(self.path):
            with self.assertRaises(ConversationLockedError):
                ConversationFileLock(self.path).acquire()
            self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.path.read_bytes(), before)

    def test_canonical_alias_conflicts_with_same_owner(self) -> None:
        """Treat normalized dot and parent path components as one file."""

        alias = self.path.parent / "other" / ".." / self.path.name
        with ConversationFileLock(self.path) as owner:
            self.assertTrue(owner.holds(alias))
            with self.assertRaises(ConversationLockedError):
                ConversationFileLock(alias).acquire()

    def test_borrowed_owner_must_be_held_and_match_path(self) -> None:
        """Allow explicit borrowing but reject stale or wrong ownership."""

        owner = ConversationFileLock(self.path)
        with self.assertRaisesRegex(ValueError, "ownership is invalid"):
            with conversation_write_ownership(self.path, ownership=owner):
                pass
        with owner:
            with conversation_write_ownership(
                self.path, ownership=owner
            ):
                self.assertTrue(owner.holds(self.path))
            with self.assertRaisesRegex(ValueError, "ownership is invalid"):
                with conversation_write_ownership(
                    self.root / "other.json", ownership=owner
                ):
                    pass
        with conversation_write_ownership(self.path):
            with self.assertRaises(ConversationLockedError):
                ConversationFileLock(self.path).acquire()

    def test_failed_open_removes_process_reservation(self) -> None:
        """Leave no ownership registration if opening the sidecar fails."""

        owner = ConversationFileLock(self.path)
        with patch(
            "kilin.conversation_ownership.os.open",
            side_effect=PermissionError("not allowed"),
        ):
            with self.assertRaisesRegex(PermissionError, "not allowed"):
                owner.acquire()
        with ConversationFileLock(self.path):
            pass

    def test_open_permission_error_is_not_reported_as_contention(
        self,
    ) -> None:
        """Distinguish inaccessible companion files from occupied locks."""

        owner = ConversationFileLock(self.path)
        error = PermissionError(errno.EACCES, "permission denied")
        with patch(
            "kilin.conversation_ownership.os.open",
            side_effect=error,
        ):
            with self.assertRaises(PermissionError):
                owner.acquire()
        with ConversationFileLock(self.path):
            pass

    @unittest.skipUnless(hasattr(os, "fork"), "Requires POSIX fork")
    def test_forked_child_drops_inherited_lock_descriptor(self) -> None:
        """Verify fork cleanup inside a separate single-threaded process."""

        script = """
import os
import sys
from kilin.conversation_ownership import ConversationFileLock

with ConversationFileLock(sys.argv[1]) as owner:
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        state = b'unowned' if not owner.holds(sys.argv[1]) else b'held'
        os.write(writer, state)
        os.close(writer)
        os._exit(0)
    os.close(writer)
    state = os.read(reader, 20)
    os.close(reader)
    os.waitpid(pid, 0)
    if state != b'unowned' or not owner.holds(sys.argv[1]):
        raise RuntimeError('A child retained the parent lock')
"""
        child = subprocess.run(
            [sys.executable, "-c", script, str(self.path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(child.returncode, 0, child.stderr)

    @unittest.skipUnless(os.name == "posix", "Requires POSIX file modes")
    def test_posix_companion_file_is_private(self) -> None:
        """Create a sidecar readable and writable only by its owner."""

        with ConversationFileLock(self.path) as owner:
            self.assertEqual(
                owner.lock_path.stat().st_mode & 0o777,
                0o600,
            )

    def test_independent_process_cannot_acquire_owned_file(self) -> None:
        """Reject concurrent subprocess writers, then allow reopening."""

        script = (
            "import sys; from kilin.conversation_ownership import "
            "ConversationFileLock; "
            "lock = ConversationFileLock(sys.argv[1]); "
            "lock.acquire(); lock.release()"
        )
        with ConversationFileLock(self.path):
            rejected = subprocess.run(
                [sys.executable, "-c", script, str(self.path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("ConversationLockedError", rejected.stderr)
        accepted = subprocess.run(
            [sys.executable, "-c", script, str(self.path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)

    def test_abrupt_child_termination_releases_os_lock(self) -> None:
        """Permit a new writer after the child owning the lock exits."""

        process = subprocess.Popen(
            [sys.executable, "-u", "-c", _CHILD, str(self.path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(process.stdout.readline().strip(), "LOCKED")
            with self.assertRaises(ConversationLockedError):
                ConversationFileLock(self.path).acquire()
            process.kill()
            process.wait(timeout=10)
            with ConversationFileLock(self.path) as recovered:
                self.assertTrue(recovered.holds(self.path))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()


if __name__ == "__main__":
    unittest.main()
