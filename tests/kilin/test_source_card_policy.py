"""Verify project and case-sensitive source-level card indexing policy.

Temporary profiles and overrides exercise persistence without loading models.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kilin.project_type import project_type
from kilin.source_card_policy import (
    SourceCardOverride,
    SourceCardPolicy,
    read_source_card_overrides,
    read_source_card_policy,
    select_source_card_kinds,
    write_source_card_overrides,
    write_source_card_policy,
)


class TestSourceCardPolicy(unittest.TestCase):

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = self.root / "profile.ini"
        self.profile.write_text(
            "[project]\ntype_id = literature\n", encoding="utf-8"
        )

    def test_packaged_defaults_are_independent_of_paper_headers(self):
        literature = read_source_card_policy(
            self.root, project_type("literature")
        )
        self.assertEqual(literature, SourceCardPolicy(
            True, "scientific_paper", True
        ))
        self.assertEqual(read_source_card_policy(
            self.root, project_type("code")
        ), SourceCardPolicy(False, None, False))

    def test_project_config_changes_only_its_own_section(self):
        changed = SourceCardPolicy(True, "scientific_paper", False)
        write_source_card_policy(self.root, changed)
        self.assertEqual(read_source_card_policy(
            self.root, project_type("literature")
        ), changed)
        self.assertIn("type_id = literature", self.profile.read_text())
        self.assertIn("automatic = false", self.profile.read_text())

    def test_partial_ini_overrides_inherit_project_defaults(self):
        self.profile.write_text(
            "[source_cards]\nautomatic = false\n", encoding="utf-8"
        )
        self.assertEqual(read_source_card_policy(
            self.root, project_type("literature")
        ), SourceCardPolicy(True, "scientific_paper", False))

    def test_unknown_or_malformed_settings_are_rejected(self):
        for text in (
            "[source_cards]\nunknown = yes\n",
            "[source_cards]\nenabled = perhaps\n",
            "[source_cards]\nkind = unknown_kind\n",
            "[source_cards]\nautomatic = 123\n",
        ):
            with self.subTest(text=text):
                self.profile.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_source_card_policy(
                        self.root, project_type("literature")
                    )

    def test_unsafe_or_empty_overrides_are_rejected(self):
        for options in ({"enabled": None}, {"kind": "not_registered"}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    SourceCardOverride(**options)
        with self.assertRaises(ValueError):
            SourceCardOverride(enabled="true")
        with self.assertRaises(ValueError):
            SourceCardPolicy(1, "scientific_paper", True)

    def test_source_overrides_preserve_exact_path_case(self):
        overrides = {
            Path("Papers/Study.PDF"): SourceCardOverride(enabled=False),
            Path("papers/paper.pdf"): SourceCardOverride(
                kind="scientific_paper"
            ),
        }
        write_source_card_overrides(self.root, overrides)
        self.assertEqual(read_source_card_overrides(self.root), overrides)
        policy = SourceCardPolicy(True, "scientific_paper", True)
        selected = select_source_card_kinds((
            Path("Papers/Study.PDF"), Path("papers/paper.pdf"),
            Path("papers/Study.PDF"),
        ), policy, overrides)
        self.assertEqual(set(selected), {
            Path("papers/paper.pdf"), Path("papers/Study.PDF"),
        })
        self.assertEqual(read_source_card_overrides(
            self.root / "missing"
        ), {})

    def test_master_switch_cannot_be_bypassed_by_source_override(self):
        policy = SourceCardPolicy(False, None, True)
        self.assertEqual(select_source_card_kinds(
            (Path("paper.pdf"),), policy,
            {Path("paper.pdf"): SourceCardOverride(
                enabled=True, kind="scientific_paper"
            )},
        ), {})

    def test_source_override_accepts_registered_kind_without_default(self):
        policy = SourceCardPolicy(True, None, True)
        self.assertEqual(select_source_card_kinds(
            (Path("paper.pdf"), Path("other.pdf")), policy,
            {Path("paper.pdf"): SourceCardOverride(
                kind="scientific_paper"
            )},
        ), {Path("paper.pdf"): "scientific_paper"})

    def test_malformed_json_unknown_fields_and_unsafe_paths(self):
        file = self.root / "source_card_sources.json"
        for payload in (
            '{broken',
            '{"format":"source-card-source-settings-v1",'
            '"sources":{"../escape":{"enabled":false}}}',
            '{"format":"source-card-source-settings-v1",'
            '"sources":{"x.pdf":{"other":false}}}',
            '{"format":"source-card-source-settings-v1",'
            '"sources":{"x.pdf":{}}}',
            '{"format":"source-card-source-settings-v1",'
            '"sources":{},"sources":{}}',
            '{"format":"outdated","sources":{}}',
            '{"format":"source-card-source-settings-v1",'
            '"sources":{"a\\\\b.pdf":{"enabled":false}}}',
        ):
            with self.subTest(payload=payload):
                file.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_source_card_overrides(self.root)

    def test_writer_rejects_unsafe_paths_and_preserves_valid_json(self):
        with self.assertRaises(ValueError):
            write_source_card_overrides(self.root, {
                Path("../escape"): SourceCardOverride(enabled=False)
            })
        choices = {Path("a.pdf"): SourceCardOverride(enabled=False)}
        write_source_card_overrides(self.root, choices)
        self.assertEqual(json.loads((
            self.root / "source_card_sources.json"
        ).read_text())["sources"]["a.pdf"], {"enabled": False})


if __name__ == "__main__":
    unittest.main()
