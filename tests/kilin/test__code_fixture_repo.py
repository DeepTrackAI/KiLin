"""Read the mock code repository from scanning to passages and graph links.

``tests/fixtures/code_repo`` holds one small file per supported code
language. These tests run it through the same steps a Code project uses,
without models: scanning with the Code type's file types, reading,
splitting into units and passages, and building the code graph. Each
language's own reader is tested in detail by its unit tests; this file
checks that they work together on one repository.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import unittest

from kilin.chunking import chunk_source_units
from kilin.project_type import project_type
from kilin.repository import scan_repository
from kilin.retrievals.code_graph import CodeRelation, build_code_graph
from kilin.sources import extract_source_units, read_repository_files

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "code_repo"


def _words(text: str) -> int:
    return len(text.split())


class TestCodeFixtureRepository(unittest.TestCase):
    """Check what a Code project reads from the mock repository."""

    @classmethod
    def setUpClass(cls) -> None:
        report = scan_repository(
            FIXTURE, extensions=project_type("code").extensions
        )
        cls.selected = set(report["selected_files"])
        cls.files = read_repository_files(report)
        cls.units = extract_source_units(cls.files)
        cls.chunks = chunk_source_units(cls.units, count_tokens=_words)
        cls.relations = set(build_code_graph(cls.files, cls.units).relations)

    def test_every_file_is_selected_and_read(self) -> None:
        on_disk = {
            path.relative_to(FIXTURE).as_posix()
            for path in FIXTURE.rglob("*")
            # Hidden files, such as macOS .DS_Store, are never sources.
            if path.is_file() and not path.name.startswith(".")
        }
        self.assertEqual(self.selected, on_disk)
        read = {unit.path.as_posix() for unit in self.units}
        self.assertEqual(read, on_disk)

    def test_each_definition_is_its_own_passage(self) -> None:
        passages = {
            (chunk.path.as_posix(), chunk.symbol) for chunk in self.chunks
        }
        for expected in (
            ("python/calculator_py.py", "py_function_A"),
            ("python/calculator_py.py", "py_function_B"),
            ("python/calculator_py.py", "PyChildClass"),
            ("matlab/m_function_A.m", "m_function_A"),
            ("matlab/m_function_B.m", "m_function_B"),
            ("js/calculator_js.js", "js_function_A"),
            ("js/calculator_js.js", "js_function_B"),
            ("js/calculator_js.js", "JsBaseClass"),
            ("js/calculator_js.js", "JsChildClass"),
            ("js/calculator_js.js", "js_function_D"),
        ):
            with self.subTest(passage=expected):
                self.assertIn(expected, passages)

    def test_javascript_comments_stay_with_their_function(self) -> None:
        function = next(
            chunk for chunk in self.chunks if chunk.symbol == "js_function_A"
        )
        self.assertTrue(function.text.startswith("/**"))
        self.assertIn("Double a, then add b.", function.text)
        self.assertTrue(function.text.rstrip().endswith("}"))

    def test_graph_links_calls_and_inheritance(self) -> None:
        py = "python.calculator_py"
        js = "js.calculator_js"
        for expected in (
            CodeRelation(
                f"{py}.py_function_A", "calls", f"{py}.py_function_B"
            ),
            CodeRelation(
                f"{py}.PyChildClass.py_method_C",
                "calls",
                f"{py}.py_function_A",
            ),
            CodeRelation(
                f"{py}.PyChildClass", "inherits", f"{py}.PyBaseClass"
            ),
            CodeRelation(
                f"{js}.js_function_A", "calls", f"{js}.js_function_B"
            ),
            CodeRelation(
                f"{js}.JsChildClass.js_method_C",
                "calls",
                f"{js}.js_function_A",
            ),
            CodeRelation(f"{js}.js_function_D", "calls", f"{js}.JsChildClass"),
            CodeRelation(
                f"{js}.JsChildClass", "inherits", f"{js}.JsBaseClass"
            ),
        ):
            with self.subTest(relation=expected):
                self.assertIn(expected, self.relations)

    def test_matlab_function_calls_its_helper(self) -> None:
        targets = {
            relation.target
            for relation in self.relations
            if relation.source == "matlab.m_function_A.m_function_A"
            and relation.relation == "calls"
        }
        self.assertTrue(
            any(target.endswith("m_function_B") for target in targets)
        )

    def test_javascript_calls_through_objects_are_not_links(self) -> None:
        targets = {
            relation.target
            for relation in self.relations
            if relation.source.startswith("js.calculator_js")
            and relation.relation == "calls"
        }
        for name in ("round", "map", "js_method_C"):
            with self.subTest(name=name):
                self.assertFalse(
                    any(target.endswith(name) for target in targets)
                )


if __name__ == "__main__":
    unittest.main()
