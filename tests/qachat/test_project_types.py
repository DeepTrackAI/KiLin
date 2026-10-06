"""Check project types loaded from project_types.json."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qachat.project import default_project_profile
from qachat.project_types import (
    PROJECT_TYPES_FILE,
    load_project_types,
    project_type,
)
from qachat.query_formats import (
    LITERATURE_REPOSITORY_INSTRUCTION,
    SOFTWARE_REPOSITORY_INSTRUCTION,
)
from qachat.rag import _SYSTEM_PROMPT


class TestProjectTypes(unittest.TestCase):
    """The packaged file reproduces the former built-in behavior."""

    def test_built_in_types_keep_previous_defaults(self) -> None:
        code = project_type("code")
        literature = project_type("literature")
        self.assertEqual(
            code.retrieval_instruction, SOFTWARE_REPOSITORY_INSTRUCTION
        )
        self.assertEqual(
            literature.retrieval_instruction,
            LITERATURE_REPOSITORY_INSTRUCTION,
        )
        self.assertEqual(code.system_prompt, _SYSTEM_PROMPT)
        self.assertEqual(literature.system_prompt, _SYSTEM_PROMPT)
        self.assertEqual(
            default_project_profile("code").extensions,
            (".py", ".ipynb", ".m", ".md"),
        )
        self.assertEqual(
            default_project_profile("literature").max_file_bytes, 20 * 2**20
        )

    def _write(self, folder: str, name: str, value: object) -> Path:
        # Loading is cached per path, so each case uses its own file.
        path = Path(folder) / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_added_type_loads_and_bad_files_explain_the_problem(self) -> None:
        packaged = json.loads(PROJECT_TYPES_FILE.read_text(encoding="utf-8"))
        with TemporaryDirectory() as folder:
            added = {
                **packaged,
                "protocols": {
                    "label": "Clinical protocols",
                    "extensions": [".pdf", ".docx"],
                    "max_file_bytes": 1000,
                    "retrieval_instruction": "Find protocol steps.",
                    "system_prompt": "Answer from the protocols.",
                },
            }
            types = load_project_types(
                self._write(folder, "added.json", added)
            )
            self.assertEqual(list(types)[-1], "protocols")
            self.assertEqual(types["protocols"].answer_guidance, "")

            missing = {"code": packaged["code"]}
            with self.assertRaisesRegex(ValueError, "literature"):
                load_project_types(
                    self._write(folder, "missing.json", missing)
                )

            broken = {**packaged, "notes": {**packaged["code"]}}
            broken["notes"]["max_file_bytes"] = "big"
            with self.assertRaisesRegex(ValueError, "max_file_bytes"):
                load_project_types(self._write(folder, "broken.json", broken))


if __name__ == "__main__":
    unittest.main()
