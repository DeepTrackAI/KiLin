"""Test persistent conversation discovery, metadata, and branching.

Use real JSON files in isolated temporary directories without loading models.
Exercise the public ConversationStore operations and canonical storage,
file naming, recoverable archiving, and filesystem and corruption boundaries.
Regression tests cover malformed records and preservation on updates.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin import conversations as module
from kilin.conversation_format import (
    CONVERSATION_SESSION_KEY,
    conversation_state,
    new_conversation_session,
)


class ConversationStoreFixture(unittest.TestCase):
    """Supply disposable conversation files and store instances.

    """

    def setUp(self) -> None:
        """Create independent paths without implicit project creation.

        """

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.project = self.base / "project"
        self.store = module.ConversationStore(self.project)

    def _saved(self, path: Path) -> dict[str, object]:
        """Read one JSON conversation file.

        Parameters
        ----------
        path: pathlib.Path
            File containing a serialized conversation record.

        Returns
        -------
        dict[str, object]
            Decoded JSON root object.

        """

        return json.loads(path.read_text(encoding="utf-8"))

    def _state(self, item: module.ConversationInfo | str) -> dict:
        """Read resumable state for an existing registered conversation.

        Parameters
        ----------
        item: ConversationInfo | str
            Registered conversation or its identifier.

        Returns
        -------
        dict
            Saved resume state.

        """

        return conversation_state(self._saved(self.store.state_path(item)))

    def _persist(
        self,
        item: module.ConversationInfo,
        updates: dict[str, object],
    ) -> None:
        """Update persisted resume state without touching stored metadata.

        Parameters
        ----------
        item: ConversationInfo
            Previously created record.
        updates: dict[str, object]
            Resume-state fields to add or replace.

        """

        path = self.store.state_path(item)
        data = self._saved(path)
        conversation_state(data, create=True).update(updates)
        path.write_text(json.dumps(data), encoding="utf-8")

    def _candidate(
        self,
        filename: str,
        **metadata: object,
    ) -> Path:
        """Write one independently discoverable fixture conversation.

        Parameters
        ----------
        filename: str
            JSON file basename under logs/traces.
        **metadata: object
            Metadata overrides, including invalid values for negative tests.

        Returns
        -------
        pathlib.Path
            Written fixture path.

        """

        state: dict[str, object] = {
            "conversation_id": filename.removesuffix(".json"),
            "name": "Saved conversation",
            "parent_conversation_id": None,
            "origin_checkpoint_id": None,
            "created_at_utc": "2026-10-10T09:00:00.000Z",
            "updated_at_utc": "2026-10-10T10:00:00.000Z",
            "turns": [],
        }
        state.update(metadata)
        path = self.store.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessions": {
                CONVERSATION_SESSION_KEY: new_conversation_session(state)
            }
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


class TestStoreConstruction(ConversationStoreFixture):
    """Verify initialization and immutable metadata before filesystem I/O."""

    def test_constructor_does_not_create_project_or_traces(self):
        self.assertFalse(self.project.exists())
        self.assertEqual(self.store.project_directory, self.project)
        self.assertEqual(self.store.root, self.project / "logs/traces")
        self.assertFalse(self.store.root.exists())

    def test_project_path_is_expanded_and_resolved(self):
        absolute = self.project / ".." / "project"
        store = module.ConversationStore(absolute)
        self.assertEqual(store.project_directory, self.project)

    def test_active_without_ensure_or_create_fails(self):
        with self.assertRaisesRegex(RuntimeError, "No active"):
            self.store.active()

    def test_empty_registry_and_immutable_metadata(self):
        self.assertEqual(self.store.conversations(), [])
        self.assertEqual(
            module.__all__, ["ConversationInfo", "ConversationStore"]
        )
        item = self.store.create()
        with self.assertRaises(FrozenInstanceError):
            item.name = "changed"


class TestStoreDiscovery(ConversationStoreFixture):
    """Discover canonical records and skip obsolete or invalid files."""

    def test_ensure_creates_an_empty_grouped_record(self):
        item = self.store.ensure()
        self.assertEqual(self.store.active(), item)
        self.assertEqual(self.store.conversations(), [item])
        self.assertTrue(item.conversation_id.startswith("conversation_"))
        self.assertEqual(item.name, "Conversation 1")
        self.assertIsNone(item.parent_conversation_id)
        self.assertIsNone(item.origin_checkpoint_id)
        self.assertTrue(item.created_at_utc.endswith("Z"))
        self.assertEqual(item.created_at_utc, item.updated_at_utc)
        path = self.store.state_path(item)
        self.assertTrue(path.name.startswith("conversation_"))
        self.assertTrue(path.is_file())
        self.assertEqual(self._state(item)["conversation_id"],
                         item.conversation_id)
        self.assertTrue(self.store.is_empty(item.conversation_id))

    def test_ensure_existing_store_is_idempotent(self):
        first = self.store.ensure()
        name = self.store.state_path(first)
        second = self.store.ensure()
        self.assertEqual(second, first)
        self.assertEqual(self.store.state_path(second), name)
        self.assertEqual(len(self.store.conversations()), 1)

    def test_ensure_skips_obsolete_record_without_deleting_it(self):
        old = self._candidate("conversation_old.json")
        old.write_text(json.dumps({"conversation": {
            "conversation_id": "conversation_old", "name": "Old",
        }}), encoding="utf-8")
        before = old.read_bytes()
        new = self._candidate(
            "conversation_new.json", name="Grouped",
            updated_at_utc="2026-10-12T10:00:00.000Z",
        )
        selected = self.store.ensure()
        self.assertEqual(selected.name, "Grouped")
        self.assertEqual(self.store.state_path("conversation_new"), new)
        self.assertEqual(old.read_bytes(), before)
        self.assertNotIn("conversation_old", [
            item.conversation_id for item in self.store.conversations()
        ])

    def test_latest_updated_record_becomes_active(self):
        self._candidate(
            "conversation_first.json",
            updated_at_utc="2026-10-10T11:00:00Z",
            created_at_utc="2026-10-10T01:00:00Z",
        )
        self._candidate(
            "conversation_second.json",
            updated_at_utc="2026-10-11T11:00:00Z",
            created_at_utc="2026-10-10T00:00:00Z",
        )
        self.assertEqual(self.store.ensure().conversation_id,
                         "conversation_second")
        self.assertEqual(
            [item.conversation_id for item in self.store.conversations()],
            ["conversation_second", "conversation_first"],
        )

    def test_active_tie_breaks_by_identifier(self):
        self._candidate("conversation_a.json")
        self._candidate("conversation_b.json")
        self.assertEqual(self.store.ensure().conversation_id,
                         "conversation_b")

    def test_conversation_sort_ties_by_identifier(self):
        self._candidate("conversation_b.json")
        self._candidate("conversation_a.json")
        self.store.ensure()
        self.assertEqual(
            [item.conversation_id for item in self.store.conversations()],
            ["conversation_a", "conversation_b"],
        )

    def test_scan_skips_invalid_paths_and_json(self):
        valid = self._candidate("conversation_valid.json")
        (self.store.root / "conversation_folder.json").mkdir()
        (self.store.root / "conversation_bad.json").write_text("{")
        (self.store.root / "conversation_list.json").write_text("[]")
        (self.store.root / "unrelated.json").write_text("{")
        (self.store.root / "conversation_missing.json").write_text(
            json.dumps({"sessions": {}}), encoding="utf-8"
        )
        (self.store.root / "conversation_incomplete.json").write_text(
            json.dumps({"conversation": {"name": "Incomplete"}}),
            encoding="utf-8",
        )
        self.assertEqual(self.store.ensure().conversation_id,
                         "conversation_valid")
        self.assertEqual(self.store.state_path("conversation_valid"), valid)
        self.assertEqual(len(self.store.conversations()), 1)

    def test_discovery_ignores_file_symlinks(self):
        outside = self.base / "outside.json"
        outside.write_text(json.dumps({"conversation": {
            "conversation_id": "outside", "name": "Outside",
            "parent_conversation_id": None, "origin_checkpoint_id": None,
            "created_at_utc": "2026-10-10T00:00:00Z",
            "updated_at_utc": "2026-10-10T00:00:00Z",
        }}))
        self.store.root.mkdir(parents=True)
        link = self.store.root / "conversation_link.json"
        try:
            link.symlink_to(outside)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"Symbolic links unavailable: {error}")
        self.assertEqual(len(self.store.conversations()), 0)
        first = self.store.ensure()
        self.assertNotEqual(first.conversation_id, "outside")

    def test_discovery_rejects_symlinked_traces_directory(self):
        outside = self.base / "elsewhere"
        outside.mkdir()
        self.store.root.parent.mkdir(parents=True)
        try:
            self.store.root.symlink_to(outside, target_is_directory=True)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"Symbolic links unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "must not be a link"):
            self.store.ensure()
        self.assertEqual(list(outside.iterdir()), [])

    def test_duplicate_ids_keep_first_sorted_file(self):
        earlier = self._candidate(
            "conversation_01.json", conversation_id="shared"
        )
        self._candidate("conversation_02.json", conversation_id="shared")
        self.store.ensure()
        self.assertEqual(self.store.state_path("shared"), earlier)
        self.assertEqual(len(self.store.conversations()), 1)

    def test_scan_discards_non_dictionary_state_from_helper(self):
        """Retain a defensive guard if the format helper changes."""
        self._candidate("conversation_saved.json")
        with patch("kilin.conversations.conversation_state",
                   return_value=[]):
            self.store._scan()
        self.assertEqual(self.store.conversations(), [])

    def test_ensure_rebuilds_registry_after_external_removal(self):
        older = self.store.ensure()
        self.store.state_path(older).unlink()
        replacement = self.store.ensure()
        self.assertNotEqual(replacement.conversation_id,
                            older.conversation_id)
        self.assertEqual(len(self.store.conversations()), 1)

    def test_malformed_unhashable_id_does_not_interrupt_discovery(self):
        """One corrupt file must not interrupt valid record discovery."""
        self._candidate("conversation_good.json")
        self._candidate("conversation_bad.json", conversation_id=["bad"])
        selected = self.store.ensure()
        self.assertEqual(selected.conversation_id, "conversation_good")
        self.assertEqual(len(self.store.conversations()), 1)

    def test_malformed_timestamp_does_not_interrupt_discovery(self):
        """Type-invalid metadata should be skipped, not used for sorting."""
        self._candidate("conversation_good.json")
        self._candidate("conversation_bad.json", updated_at_utc=None)
        selected = self.store.ensure()
        self.assertEqual(selected.conversation_id, "conversation_good")


    def test_metadata_validator_rejects_invalid_identifiers(self):
        invalid = (
            None, False, 123, [], {}, "", " ", "bad id", "bad/id",
            "bad\nline", "_leading", "-leading", "name.with.dot",
        )
        good = self._candidate("conversation_good.json")
        for index, value in enumerate(invalid):
            with self.subTest(value=value):
                candidate = self._candidate(
                    f"conversation_bad_{index}.json",
                    conversation_id=value,
                )
                self.assertEqual(
                    self.store.ensure().conversation_id,
                    "conversation_good",
                )
                self.assertEqual(len(self.store.conversations()), 1)
                self.assertTrue(candidate.exists())
                self.assertEqual(
                    self.store.state_path("conversation_good"), good
                )

    def test_metadata_validator_rejects_missing_and_invalid_names(self):
        invalid = (None, False, 0, [], {}, "", "   ", "x" * 81)
        self._candidate("conversation_good.json")
        for index, value in enumerate(invalid):
            with self.subTest(value=value):
                self._candidate(
                    f"conversation_bad_{index}.json", name=value
                )
                selected = self.store.ensure()
                self.assertEqual(
                    selected.conversation_id, "conversation_good"
                )
                self.assertEqual(len(self.store.conversations()), 1)

    def test_metadata_validator_checks_optional_parent_and_checkpoint(self):
        self._candidate("conversation_good.json")
        invalid_fields = {
            "parent_conversation_id": (
                False, 42, [], {}, "", "bad parent", "bad/id"
            ),
            "origin_checkpoint_id": (False, 2, [], {}, ""),
        }
        number = 0
        for field, values in invalid_fields.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self._candidate(
                        f"conversation_bad_{number}.json",
                        **{field: value},
                    )
                    number += 1
                    self.assertEqual(
                        self.store.ensure().conversation_id,
                        "conversation_good",
                    )
                    self.assertEqual(len(self.store.conversations()), 1)

    def test_metadata_validator_rejects_invalid_utc_timestamps(self):
        invalid = (
            None, False, 5, [], {}, "", "not-a-date",
            "2026-10-10", "2026-10-10T10:00:00",
            "2026-10-10T10:00:00+01:00",
            "2026-10-10T10:00:00-00:00",
            "2026-10-10T10:00:00.1234567Z",
            "2026-13-10T10:00:00Z",
            "2026-02-30T10:00:00Z",
            "2026-10-10T25:00:00Z",
            " 2026-10-10T10:00:00Z",
        )
        self._candidate("conversation_good.json")
        for field in ("created_at_utc", "updated_at_utc"):
            for index, value in enumerate(invalid):
                with self.subTest(field=field, value=value):
                    self._candidate(
                        f"conversation_bad_{field}_{index}.json",
                        **{field: value},
                    )
                    self.assertEqual(
                        self.store.ensure().conversation_id,
                        "conversation_good",
                    )
                    self.assertEqual(len(self.store.conversations()), 1)

    def test_metadata_validator_skips_missing_required_fields(self):
        self._candidate("conversation_good.json")
        missing = (
            "conversation_id", "name", "parent_conversation_id",
            "origin_checkpoint_id", "created_at_utc",
            "updated_at_utc",
        )
        for index, field in enumerate(missing):
            with self.subTest(field=field):
                candidate = self._candidate(
                    f"conversation_missing_{index}.json"
                )
                payload = self._saved(candidate)
                payload["sessions"][CONVERSATION_SESSION_KEY][
                    "resume_state"
                ].pop(field)
                candidate.write_text(json.dumps(payload), encoding="utf-8")
                self.assertEqual(
                    self.store.ensure().conversation_id,
                    "conversation_good",
                )
                self.assertEqual(len(self.store.conversations()), 1)

    def test_malformed_first_duplicate_does_not_hide_valid_record(self):
        broken = self._candidate(
            "conversation_01.json", conversation_id="same",
            updated_at_utc=None,
        )
        valid = self._candidate(
            "conversation_02.json", conversation_id="same"
        )
        self.assertEqual(self.store.ensure().conversation_id, "same")
        self.assertEqual(self.store.state_path("same"), valid)
        self.assertTrue(broken.is_file())

    def test_utc_offsets_and_fractional_precisions_sort_by_actual_time(self):
        self._candidate(
            "conversation_early.json",
            created_at_utc="2026-10-10T10:00:00.050Z",
            updated_at_utc="2026-10-10T10:00:00.050Z",
        )
        self._candidate(
            "conversation_late.json",
            created_at_utc="2026-10-10T10:00:00.100+00:00",
            updated_at_utc="2026-10-10T10:00:00.100+00:00",
        )
        self._candidate(
            "conversation_first.json",
            created_at_utc="2026-10-10T10:00:00+00:00",
            updated_at_utc="2026-10-10T10:00:00+00:00",
        )
        selected = self.store.ensure()
        self.assertEqual(selected.conversation_id, "conversation_late")
        self.assertEqual(
            [item.conversation_id for item in self.store.conversations()],
            ["conversation_first", "conversation_early",
             "conversation_late"],
        )
        # Discovery must not rewrite metadata or normalize its spelling.
        saved = self._saved(self.store.state_path(selected))
        self.assertEqual(
            conversation_state(saved)["updated_at_utc"],
            "2026-10-10T10:00:00.100+00:00",
        )

    def test_archive_selects_most_recent_timestamp_not_text_order(self):
        self._candidate(
            "conversation_older.json",
            updated_at_utc="2026-10-10T10:00:00Z",
        )
        self._candidate(
            "conversation_newer.json",
            updated_at_utc="2026-10-10T10:00:00.100Z",
        )
        self._candidate(
            "conversation_archived.json",
            updated_at_utc="2026-10-10T11:00:00Z",
        )
        selected = self.store.ensure()
        self.assertEqual(selected.conversation_id, "conversation_archived")
        remaining, archived = self.store.archive(selected.conversation_id)
        self.assertEqual(remaining.conversation_id, "conversation_newer")
        self.assertTrue(archived.is_file())

    def test_valid_utc_metadata_and_unicode_names_are_preserved(self):
        original = self._candidate(
            "conversation_imported.json",
            name="Vetenskap 日本語", parent_conversation_id="parent_01",
            origin_checkpoint_id="step:12/xyz",
            created_at_utc="2026-10-10T09:00:00+00:00",
            updated_at_utc="2026-10-10T09:00:00.5Z",
        )
        selected = self.store.ensure()
        self.assertEqual(selected.name, "Vetenskap 日本語")
        self.assertEqual(selected.parent_conversation_id, "parent_01")
        self.assertEqual(selected.origin_checkpoint_id, "step:12/xyz")
        self.assertEqual(self.store.state_path(selected), original)


class TestStoreSelectionAndCreation(ConversationStoreFixture):
    """Test independent creation, selection, and name/path collisions."""

    def test_create_without_ensure_initializes_storage(self):
        item = self.store.create()
        self.assertTrue(self.store.state_path(item).is_file())
        self.assertEqual(self.store.active(), item)
        self.assertEqual(self.store.conversations(), [item])

    def test_create_twice_sets_active_and_allocates_second_name(self):
        first = self.store.create()
        second = self.store.create()
        self.assertNotEqual(first.conversation_id, second.conversation_id)
        self.assertNotEqual(self.store.state_path(first),
                            self.store.state_path(second))
        self.assertEqual((first.name, second.name),
                         ("Conversation 1", "Conversation 2"))
        self.assertEqual(self.store.active(), second)

    def test_create_writes_unicode_metadata_without_ascii_escaping(self):
        item = self.store.create()
        renamed = self.store.rename(item.conversation_id, "Fråga 日本語")
        raw = self.store.state_path(renamed).read_text(encoding="utf-8")
        self.assertIn("Fråga 日本語", raw)
        self.assertTrue(raw.endswith("\n"))
        self.assertEqual(self._state(renamed)["name"], "Fråga 日本語")

    def test_select_existing_and_unknown_does_not_change_active(self):
        first = self.store.create()
        second = self.store.create()
        self.assertEqual(self.store.select(first.conversation_id), first)
        self.assertEqual(self.store.active(), first)
        for invalid in ("missing", "", None, 5, []):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "not found"):
                    self.store.select(invalid)
                self.assertEqual(self.store.active(), first)
        self.assertEqual(len(self.store.conversations()), 2)
        self.assertIn(second, self.store.conversations())

    def test_state_path_resolves_metadata_or_id(self):
        item = self.store.create()
        self.assertEqual(self.store.state_path(item),
                         self.store.state_path(item.conversation_id))
        with self.assertRaisesRegex(ValueError, "not found"):
            self.store.state_path("not-registered")

    def test_new_path_suffixes_timestamp_collision(self):
        self.store.root.mkdir(parents=True)
        with patch("kilin.conversations.datetime") as clock:
            clock.now.return_value.strftime.return_value = "same-time"
            first = self.store._new_path()
            first.touch()
            second = self.store._new_path()
            second.touch()
            third = self.store._new_path()
        self.assertEqual(first.name, "conversation_same-time.json")
        self.assertEqual(second.name, "conversation_same-time_2.json")
        self.assertEqual(third.name, "conversation_same-time_3.json")

    def test_create_does_not_replace_an_existing_file_on_collision(self):
        self.store.root.mkdir(parents=True)
        with patch("kilin.conversations.datetime", wraps=datetime) as clock:
            clock.now.return_value = datetime(
                2026, 10, 12, 12, 0, tzinfo=timezone.utc
            )
            existing = (
                self.store.root / "conversation_2026-10-12T12-00-00.json"
            )
            existing.write_text("do not replace", encoding="utf-8")
            item = self.store.create()
        self.assertEqual(existing.read_text(), "do not replace")
        self.assertEqual(self.store.state_path(item).name,
                         "conversation_2026-10-12T12-00-00_2.json")

    def test_timestamp_has_expected_utc_format(self):
        self.assertRegex(
            module._utc_timestamp(),
            r"^\d{4}-\d{2}-\d{2}T\d\d:\d\d:\d\d\.\d{3}Z$",
        )

    def test_temporary_file_symlink_cannot_overwrite_external_file(self):
        """A malicious predictable .json.tmp link must not be followed."""
        self.store.root.mkdir(parents=True)
        target = self.base / "unrelated.txt"
        target.write_text("Do not change", encoding="utf-8")
        path = self.store.root / "conversation_fixed.json"
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.symlink_to(target)
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"Symbolic links unavailable: {error}")
        with patch.object(self.store, "_new_path", return_value=path):
            self.store.create()
        self.assertEqual(target.read_text(), "Do not change")
        self.assertTrue(temporary.is_symlink())
        self.assertTrue(path.is_file())
        self.assertEqual(
            list(self.store.root.glob(f".{path.name}.*.tmp")), []
        )


class TestStoreBranching(ConversationStoreFixture):
    """Verify provenance, isolation, and format compatibility of branches."""

    def test_branch_copies_resume_state_but_not_trace_operations(self):
        parent = self.store.create()
        self._persist(parent, {
            "turns": [{"question": "Original", "answer": "Yes"}],
            "checkpoint_id": "checkpoint-123",
            "checkpoint": {"parent_checkpoint_id": "before"},
            "system_prompt": "Do science",
            "profile": {"kind": "literature"},
        })
        payload = self._saved(self.store.state_path(parent))
        session = payload["sessions"][CONVERSATION_SESSION_KEY]
        session["turns"] = {"turn_0001": {"title": "Parent event"}}
        session["operations"] = {"01_action": {"name": "Parent action"}}
        self.store.state_path(parent).write_text(json.dumps(payload))
        branch = self.store.create(branch_from=parent.conversation_id)
        self.assertEqual(branch.parent_conversation_id,
                         parent.conversation_id)
        self.assertEqual(branch.origin_checkpoint_id, "checkpoint-123")
        self.assertEqual(branch.name, f"{parent.name} branch 1")
        self.assertEqual(self.store.active(), branch)
        saved = self._state(branch)
        self.assertEqual(saved["turns"],
                         [{"question": "Original", "answer": "Yes"}])
        self.assertEqual(saved["system_prompt"], "Do science")
        self.assertEqual(saved["checkpoint_id"], "checkpoint-123")
        self.assertEqual(saved["profile"], {"kind": "literature"})
        self.assertEqual(saved["conversation_id"], branch.conversation_id)
        branched = self._saved(self.store.state_path(branch))
        group = branched["sessions"][CONVERSATION_SESSION_KEY]
        self.assertEqual(group["turns"], {})
        self.assertEqual(group["operations"], {})
        self.assertEqual(self._state(parent)["name"], parent.name)
        saved["turns"][0]["question"] = "Changed in memory"
        self.assertEqual(self._state(parent)["turns"][0]["question"],
                         "Original")

    def test_branch_does_not_copy_parent_identity_or_metadata(self):
        parent = self.store.create()
        self._persist(parent, {"checkpoint_id": "one", "custom": 4})
        branched = self.store.create(branch_from=parent.conversation_id)
        state = self._state(branched)
        self.assertEqual(state["custom"], 4)
        for key, value in asdict(branched).items():
            self.assertEqual(state[key], value)
        self.assertEqual(self._state(parent)["conversation_id"],
                         parent.conversation_id)

    def test_branch_from_imported_conversation_uses_canonical_state(self):
        source = self._candidate(
            "conversation_external.json",
            checkpoint_id="source-checkpoint",
            turns=[{"question": "Old"}],
        )
        self.store.ensure()
        branch = self.store.create(branch_from="conversation_external")
        payload = self._saved(self.store.state_path(branch))
        self.assertNotIn("conversation", payload)
        self.assertIn(CONVERSATION_SESSION_KEY, payload["sessions"])
        self.assertEqual(branch.origin_checkpoint_id, "source-checkpoint")
        self.assertEqual(self._state(branch)["turns"],
                         [{"question": "Old"}])
        self.assertTrue(source.is_file())

    def test_multiple_branches_increment_names(self):
        parent = self.store.create()
        first = self.store.create(branch_from=parent.conversation_id)
        second = self.store.create(branch_from=parent.conversation_id)
        self.assertEqual(first.name, f"{parent.name} branch 1")
        self.assertEqual(second.name, f"{parent.name} branch 2")

    def test_branches_without_valid_checkpoint_have_no_origin(self):
        parent = self.store.create()
        for value in (None, 23, "", [], {}):
            with self.subTest(value=value):
                self._persist(parent, {"checkpoint_id": value})
                branched = self.store.create(
                    branch_from=parent.conversation_id
                )
                self.assertIsNone(branched.origin_checkpoint_id)

    def test_branch_unknown_parent_does_not_create_file(self):
        before = list(self.store.root.glob("*.json")) if (
            self.store.root.is_dir()
        ) else []
        with self.assertRaisesRegex(ValueError, "not found"):
            self.store.create(branch_from="unknown")
        self.assertEqual(list(self.store.root.glob("*.json")) if (
            self.store.root.is_dir()
        ) else [], before)

    def test_malformed_parent_prevents_branch_and_keeps_active(self):
        parent = self.store.create()
        self.store.state_path(parent).write_text("{", encoding="utf-8")
        before = set(self.store.root.iterdir())
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.store.create(branch_from=parent.conversation_id)
        self.assertEqual(set(self.store.root.iterdir()), before)
        self.assertEqual(self.store.active(), parent)

    def test_branch_parent_with_missing_checkpoint_is_valid(self):
        parent = self.store.create()
        self._persist(parent, {"turns": [1]})
        branch = self.store.create(branch_from=parent.conversation_id)
        self.assertIsNone(branch.origin_checkpoint_id)
        self.assertEqual(self._state(branch)["turns"], [1])


class TestStoreMetadata(ConversationStoreFixture):
    """Check rename, touch, state preservation, and failure boundaries."""

    def test_rename_collapses_whitespace_and_persists(self):
        item = self.store.create()
        with patch("kilin.conversations._utc_timestamp",
                   return_value="2026-10-12T12:00:00.000Z"):
            renamed = self.store.rename(
                item.conversation_id, "  New \n\t conversation  "
            )
        self.assertEqual(renamed.name, "New conversation")
        self.assertEqual(renamed.updated_at_utc,
                         "2026-10-12T12:00:00.000Z")
        self.assertEqual(item.name, "Conversation 1")
        self.assertEqual(renamed.created_at_utc, item.created_at_utc)
        self.assertEqual(self.store.active(), renamed)
        self.assertEqual(self._state(renamed)["name"], "New conversation")

    def test_rename_preserves_saved_conversation_state(self):
        item = self.store.create()
        self._persist(item, {
            "turns": [{"question": "keep"}],
            "checkpoint_id": "unchanged",
        })
        self.store.rename(item.conversation_id, "Updated")
        self.assertEqual(self._state(item)["turns"],
                         [{"question": "keep"}])
        self.assertEqual(self._state(item)["checkpoint_id"], "unchanged")

    def test_rename_rejects_grouped_legacy_field_without_rewriting(self):
        item = self.store.create()
        path = self.store.state_path(item)
        payload = self._saved(path)
        session = payload["sessions"][CONVERSATION_SESSION_KEY]
        session["conversation"] = session.pop("resume_state")
        session["turns"] = {"turn_0001": {"title": "Saved event"}}
        path.write_text(json.dumps(payload), encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.store.rename(item.conversation_id, "New name")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.store.active(), item)

    def test_touch_rejects_noncanonical_session_without_rewriting(self):
        item = self.store.create()
        path = self.store.state_path(item)
        payload = self._saved(path)
        session = payload["sessions"].pop(CONVERSATION_SESSION_KEY)
        payload["sessions"]["prior_conversation"] = session
        path.write_text(json.dumps(payload), encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.store.touch(item.conversation_id)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.store.active(), item)

    def test_rename_name_limits(self):
        item = self.store.create()
        accepted = self.store.rename(item.conversation_id, "x" * 80)
        self.assertEqual(len(accepted.name), 80)
        for invalid in ("", "   ", "\n\t", "a" * 81):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "name"):
                    self.store.rename(item.conversation_id, invalid)
                self.assertEqual(self.store.active(), accepted)
        self.assertEqual(self._state(item)["name"], "x" * 80)

    def test_rename_stringifies_non_string_names(self):
        item = self.store.create()
        renamed = self.store.rename(item.conversation_id, 321)
        self.assertEqual(renamed.name, "321")

    def test_touch_updates_metadata_and_preserves_other_fields(self):
        item = self.store.create()
        self._persist(item, {"turns": [1], "checkpoint_id": "one"})
        with patch("kilin.conversations._utc_timestamp",
                   return_value="2026-10-12T15:00:00.000Z"):
            touched = self.store.touch(item.conversation_id)
        self.assertEqual(touched.name, item.name)
        self.assertEqual(touched.created_at_utc, item.created_at_utc)
        self.assertEqual(touched.updated_at_utc,
                         "2026-10-12T15:00:00.000Z")
        self.assertEqual(self._state(touched)["turns"], [1])
        self.assertEqual(self._state(touched)["checkpoint_id"], "one")
        self.assertEqual(self.store.active(), touched)

    def test_touch_nonactive_record_leaves_selection(self):
        first = self.store.create()
        second = self.store.create()
        self.store.touch(first.conversation_id)
        self.assertEqual(self.store.active(), second)

    def test_touch_and_rename_unknown_ids_fail_without_file_creation(self):
        item = self.store.create()
        before = set(self.store.root.iterdir())
        for action in (
            lambda: self.store.rename("unknown", "Okay"),
            lambda: self.store.touch("unknown"),
        ):
            with self.assertRaisesRegex(ValueError, "not found"):
                action()
        self.assertEqual(self.store.active(), item)
        self.assertEqual(set(self.store.root.iterdir()), before)

    def test_touch_rejects_corrupted_json_without_changing_registry(self):
        item = self.store.create()
        path = self.store.state_path(item)
        path.write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.store.touch(item.conversation_id)
        self.assertEqual(self.store.active(), item)
        self.assertEqual(path.read_text(), "not json")

    def test_touch_rejects_invalid_state_container(self):
        item = self.store.create()
        path = self.store.state_path(item)
        path.write_text(json.dumps({"sessions": []}))
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.store.touch(item.conversation_id)
        self.assertEqual(self.store.active(), item)

    def test_touch_rejects_conflicting_canonical_slot_without_writing(self):
        item = self.store.create()
        path = self.store.state_path(item)
        payload = self._saved(path)
        payload["sessions"][CONVERSATION_SESSION_KEY] = {
            "category": "runtime", "important": "keep",
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        before = path.read_bytes()

        with self.assertRaisesRegex(ValueError, "reserved session"):
            self.store.touch(item.conversation_id)

        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.store.active(), item)

    def test_rename_rejects_top_level_legacy_without_rewriting(self):
        item = self.store.create()
        path = self.store.state_path(item)
        path.write_text(json.dumps({
            "conversation": self._state(item),
        }), encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.store.rename(item.conversation_id, "Other")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.store.active(), item)

    def test_touch_and_rename_persist_across_new_store(self):
        item = self.store.create()
        renamed = self.store.rename(item.conversation_id, "Reopened")
        self.store.touch(renamed.conversation_id)
        reopened = module.ConversationStore(self.project)
        selected = reopened.ensure()
        self.assertEqual(selected.name, "Reopened")
        self.assertEqual(selected.conversation_id, item.conversation_id)


class TestStoreEmptyState(ConversationStoreFixture):
    """Classify missing, populated, and corrupt turn state."""

    def test_new_grouped_conversation_is_empty(self):
        item = self.store.ensure()
        self.assertTrue(self.store.is_empty(item.conversation_id))

    def test_missing_turn_field_counts_as_empty(self):
        item = self.store.create()
        data = self._saved(self.store.state_path(item))
        state = data["sessions"][CONVERSATION_SESSION_KEY]["resume_state"]
        state.pop("turns", None)
        self.store.state_path(item).write_text(json.dumps(data))
        self.assertTrue(self.store.is_empty(item.conversation_id))

    def test_saved_turns_are_not_empty(self):
        item = self.store.create()
        self._persist(item, {"turns": [{"question": "Hello"}]})
        self.assertFalse(self.store.is_empty(item.conversation_id))
        self._persist(item, {"turns": []})
        self.assertTrue(self.store.is_empty(item.conversation_id))

    def test_non_list_turns_are_rejected(self):
        item = self.store.create()
        for invalid in ({}, "hello", None, 2, False):
            with self.subTest(invalid=invalid):
                self._persist(item, {"turns": invalid})
                with self.assertRaisesRegex(ValueError, "invalid"):
                    self.store.is_empty(item.conversation_id)

    def test_invalid_or_missing_saved_file_fails(self):
        item = self.store.create()
        path = self.store.state_path(item)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "invalid"):
            self.store.is_empty(item.conversation_id)
        with self.assertRaisesRegex(ValueError, "not found"):
            self.store.is_empty("missing")


class TestStoreArchiving(ConversationStoreFixture):
    """Preserve deleted records without losing the active conversation."""

    def test_archive_last_conversation_creates_new_empty_active(self):
        old = self.store.ensure()
        old_path = self.store.state_path(old)
        active, archived = self.store.archive(old.conversation_id)
        self.assertEqual(archived.parent.name, "deleted")
        self.assertEqual(archived.name, old_path.name)
        self.assertTrue(archived.is_file())
        self.assertEqual(self.store.state_path(active).parent,
                         old_path.parent)
        self.assertEqual(self.store.active(), active)
        self.assertNotEqual(old.conversation_id, active.conversation_id)
        self.assertTrue(self.store.is_empty(active.conversation_id))
        self.assertEqual(len(self.store.conversations()), 1)
        self.assertIn(old.conversation_id,
                      archived.read_text(encoding="utf-8"))

    def test_archive_active_selects_latest_remaining(self):
        first = self.store.create()
        second = self.store.create()
        third = self.store.create()
        with patch("kilin.conversations._utc_timestamp",
                   return_value="2099-01-01T00:00:00Z"):
            self.store.touch(first.conversation_id)
        self.store.select(third.conversation_id)
        active, archived = self.store.archive(third.conversation_id)
        self.assertEqual(active.conversation_id, first.conversation_id)
        self.assertEqual(self.store.active().conversation_id,
                         first.conversation_id)
        self.assertTrue(archived.is_file())
        self.assertIn(second, self.store.conversations())

    def test_archive_nonactive_keeps_current_selection(self):
        first = self.store.create()
        second = self.store.create()
        active, archived = self.store.archive(first.conversation_id)
        self.assertEqual(active, second)
        self.assertEqual(self.store.active(), second)
        self.assertTrue(archived.is_file())
        with self.assertRaisesRegex(ValueError, "not found"):
            self.store.state_path(first.conversation_id)

    def test_archive_collision_appends_number_without_overwrite(self):
        item = self.store.create()
        path = self.store.state_path(item)
        folder = self.store.root / "deleted"
        folder.mkdir()
        existing = folder / path.name
        existing.write_text("original archive", encoding="utf-8")
        active, archived = self.store.archive(item.conversation_id)
        self.assertEqual(existing.read_text(), "original archive")
        self.assertEqual(archived.name,
                         f"{path.stem}_2{path.suffix}")
        self.assertTrue(archived.exists())
        self.assertNotEqual(active.conversation_id,
                            item.conversation_id)

    def test_archive_collision_skips_multiple_reserved_names(self):
        item = self.store.create()
        path = self.store.state_path(item)
        folder = self.store.root / "deleted"
        folder.mkdir()
        (folder / path.name).write_text("1")
        (folder / f"{path.stem}_2.json").write_text("2")
        _, archived = self.store.archive(item.conversation_id)
        self.assertEqual(archived.name, f"{path.stem}_3.json")

    def test_archived_files_not_rediscovered(self):
        item = self.store.create()
        self.store.archive(item.conversation_id)
        reopened = module.ConversationStore(self.project)
        reopened.ensure()
        self.assertNotIn(item.conversation_id,
                         [x.conversation_id for x in reopened.conversations()])

    def test_symlinked_deleted_folder_is_rejected(self):
        item = self.store.create()
        outside = self.base / "outside"
        outside.mkdir()
        try:
            (self.store.root / "deleted").symlink_to(
                outside, target_is_directory=True
            )
        except (NotImplementedError, OSError) as error:
            self.skipTest(f"Symbolic links unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "must not be a link"):
            self.store.archive(item.conversation_id)
        self.assertTrue(self.store.state_path(item).is_file())
        self.assertEqual(list(outside.iterdir()), [])

    def test_archive_unknown_id_does_not_create_deleted_directory(self):
        item = self.store.create()
        with self.assertRaisesRegex(ValueError, "not found"):
            self.store.archive("not-there")
        self.assertFalse((self.store.root / "deleted").exists())
        self.assertEqual(self.store.active(), item)

    def test_missing_source_archive_does_not_change_registry(self):
        item = self.store.create()
        self.store.state_path(item).unlink()
        with self.assertRaises(OSError):
            self.store.archive(item.conversation_id)
        self.assertEqual(self.store.active(), item)


class TestConversationJsonHelpers(ConversationStoreFixture):
    """Test isolated JSON utilities and their failure mapping."""

    def test_read_valid_json_object(self):
        path = self.base / "simple.json"
        path.write_text('{"hello": "world"}', encoding="utf-8")
        self.assertEqual(module._read_json(path), {"hello": "world"})

    def test_read_missing_invalid_json_and_non_object(self):
        for text in ("{", "[]", "null", '"text"'):
            with self.subTest(text=text):
                path = self.base / "bad.json"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid"):
                    module._read_json(path)
        with self.assertRaisesRegex(ValueError, "invalid"):
            module._read_json(self.base / "missing.json")

    def test_write_json_creates_parent_and_formatted_unicode(self):
        path = self.base / "subfolder" / "nested.json"
        data = {"title": "Göteborg", "saved": [1, 2]}
        module._write_json(path, data)
        self.assertEqual(self._saved(path), data)
        self.assertIn('  "title": "Göteborg"', path.read_text())
        self.assertTrue(path.read_text().endswith("\n"))
        self.assertFalse((path.parent / "nested.json.tmp").exists())

    def test_write_json_replaces_existing_data(self):
        path = self.base / "saved.json"
        module._write_json(path, {"version": 1})
        module._write_json(path, {"version": 2})
        self.assertEqual(self._saved(path), {"version": 2})

    def test_write_json_rejects_unserializable_data(self):
        path = self.base / "saved.json"
        with self.assertRaises(TypeError):
            module._write_json(path, {"bad": object()})
        self.assertFalse(path.exists())

    def test_write_json_serialization_failure_preserves_existing_file(self):
        """A serialization error must not leave a partial replacement."""
        path = self.base / "saved.json"
        module._write_json(path, {"version": 1})
        before = path.read_bytes()
        with self.assertRaises(TypeError):
            module._write_json(path, {"value": object()})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(
            list(self.base.glob(f".{path.name}.*.tmp")), []
        )

    def test_write_json_replace_failure_cleans_temporary_file(self):
        """A failed replacement preserves the old JSON and removes debris."""
        path = self.base / "saved.json"
        module._write_json(path, {"version": 1})
        before = path.read_bytes()
        with patch.object(module.Path, "replace", side_effect=OSError("deny")):
            with self.assertRaisesRegex(OSError, "deny"):
                module._write_json(path, {"version": 2})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(
            list(self.base.glob(f".{path.name}.*.tmp")), []
        )

    def test_invalid_utf8_is_rejected_during_decoding(self):
        """Binary content must not be treated as a valid JSON record."""
        path = self.base / "invalid.json"
        path.write_bytes(b"\xff\xfe")
        with self.assertRaises(UnicodeDecodeError):
            module._read_json(path)


if __name__ == "__main__":
    unittest.main()


class TestStoreFileOwnership(ConversationStoreFixture):
    """Require ownership for store metadata writes, branching and archival."""

    def test_mutations_refuse_a_competing_logger(self) -> None:
        """Keep file bytes and store registry unchanged on contention."""

        from kilin.conversation_ownership import ConversationLockedError
        from kilin.trace_logging import TraceConfig, TraceLogger

        item = self.store.create()
        path = self.store.state_path(item)
        logger = TraceLogger(
            TraceConfig(enabled=True, output_path=path),
            default_directory=self.store.root,
        )
        self.addCleanup(logger.close)
        before = path.read_bytes()
        actions = (
            lambda: self.store.rename(item.conversation_id, "new"),
            lambda: self.store.touch(item.conversation_id),
            lambda: self.store.archive(item.conversation_id),
            lambda: self.store.create(branch_from=item.conversation_id),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(ConversationLockedError):
                    action()
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(self.store.active(), item)

    def test_coordinated_metadata_write_is_the_only_file_write(
        self,
    ) -> None:
        """Keep the trace snapshot current without a store replacement."""

        from kilin.trace_logging import TraceConfig, TraceLogger

        item = self.store.create()
        path = self.store.state_path(item)
        logger = TraceLogger(
            TraceConfig(enabled=True, output_path=path),
            default_directory=self.store.root,
        )
        self.addCleanup(logger.close)
        with patch.object(
            module, "_write_json", side_effect=AssertionError("store wrote")
        ):
            renamed = self.store.rename(
                item.conversation_id,
                "Updated title",
                ownership=logger._file_lock,
                state_writer=logger.update_conversation,
            )
        self.assertEqual(self.store.active().name, "Updated title")
        self.assertEqual(renamed.name, "Updated title")
        logger.emit("later.event", component="test")
        self.assertEqual(
            conversation_state(self._saved(path))["name"], "Updated title"
        )

    def test_failed_coordinated_write_leaves_registry_and_trace_intact(
        self,
    ) -> None:
        """Do not commit metadata when its active trace write fails."""

        from kilin.trace_logging import TraceConfig, TraceLogger

        item = self.store.create()
        path = self.store.state_path(item)
        logger = TraceLogger(
            TraceConfig(enabled=True, output_path=path),
            default_directory=self.store.root,
        )
        self.addCleanup(logger.close)
        before = path.read_bytes()
        with patch.object(
            logger,
            "_write_readable_trace",
            side_effect=OSError("disk full"),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.store.rename(
                    item.conversation_id,
                    "Unsaved title",
                    ownership=logger._file_lock,
                    state_writer=logger.update_conversation,
                )
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.store.active(), item)
        logger.emit("later.event", component="test")
        self.assertEqual(
            conversation_state(self._saved(path))["name"], item.name
        )

    def test_coordinated_writer_requires_borrowed_ownership(self) -> None:
        """Reject a writer callback when no file owner is supplied."""

        item = self.store.create()
        path = self.store.state_path(item)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "requires held ownership"):
            self.store.touch(
                item.conversation_id,
                state_writer=lambda metadata: None,
            )
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.store.active(), item)

    def test_borrowed_logger_owner_supports_gui_style_mutations(
        self,
    ) -> None:
        """Allow metadata updates and archival under an existing file lock."""

        from kilin.trace_logging import TraceConfig, TraceLogger

        item = self.store.create()
        path = self.store.state_path(item)
        logger = TraceLogger(
            TraceConfig(enabled=True, output_path=path),
            default_directory=self.store.root,
        )
        self.addCleanup(logger.close)
        ownership = logger._file_lock
        renamed = self.store.rename(
            item.conversation_id,
            "Revised",
            ownership=ownership,
            state_writer=logger.update_conversation,
        )
        touched = self.store.touch(
            item.conversation_id,
            ownership=ownership,
            state_writer=logger.update_conversation,
        )
        self.assertEqual(renamed.name, "Revised")
        self.assertEqual(touched.name, "Revised")
        logger.emit("metadata.updated", component="test")
        self.assertEqual(
            conversation_state(self._saved(path))["name"], "Revised"
        )
        branch = self.store.create(
            branch_from=item.conversation_id, ownership=ownership
        )
        self.assertEqual(
            branch.parent_conversation_id, item.conversation_id
        )
        active, archive = self.store.archive(
            item.conversation_id, ownership=ownership
        )
        self.assertTrue(archive.is_file())
        self.assertEqual(active, branch)
        logger.close(emit_event=False)
        self.assertTrue(archive.exists())
