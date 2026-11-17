"""Read the mock code repository from scanning to passages and graph links.

``tests/fixtures/code_repo`` holds one small file per supported code
language. These tests run it through the same steps a Code project uses,
without models: scanning with the Code type's file types, reading,
splitting into units and passages, and building the code graph. As in a
Code project, whose ``code_graph`` is ``true``, HTML is read as code. Each
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
from kilin.retrievals.context_expansion import (
    names_code,
    plan_context_expansion,
)
from kilin.retrievals.strategies.retrieval_code import code_search
from kilin.sources import extract_source_units, read_repository_files

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "code_repo"


def _words(text: str) -> int:
    return len(text.split())


class TestCodeFixtureRepository(unittest.TestCase):
    """Check what a Code project reads from the mock repository."""

    @classmethod
    def setUpClass(cls) -> None:
        code = project_type("code")
        report = scan_repository(FIXTURE, extensions=code.extensions)
        cls.report = report
        cls.selected = set(report["selected_files"])
        as_code = code.code_graph is True
        cls.files = read_repository_files(report, as_code=as_code)
        cls.units = extract_source_units(cls.files, as_code=as_code)
        cls.chunks = chunk_source_units(cls.units, count_tokens=_words)
        cls.graph = build_code_graph(cls.files, cls.units)
        cls.relations = set(cls.graph.relations)

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
            ("r/calculator_r.R", "r_function_A"),
            ("r/calculator_r.R", "r_function_B"),
            ("r/calculator_r.R", "RChildClass"),
            ("r/calculator_r.R", "r_function_D"),
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

    def test_r_calls_inheritance_and_package(self) -> None:
        r = "r.calculator_r"
        for expected in (
            CodeRelation(f"{r}.r_function_A", "calls", f"{r}.r_function_B"),
            CodeRelation(
                f"{r}.RChildClass.r_method_C", "calls", f"{r}.r_function_A"
            ),
            CodeRelation(f"{r}.RChildClass", "inherits", f"{r}.RBaseClass"),
            CodeRelation(r, "imports", "methods"),
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

    def test_html_is_read_as_code(self) -> None:
        page = next(
            chunk for chunk in self.chunks if chunk.path.suffix == ".html"
        )
        self.assertIn('<button onclick="js_function_A(1, 2)">', page.text)

    def test_html_is_read_as_page_text_without_the_code_graph(self) -> None:
        files = read_repository_files(self.report)
        text = next(f.text for f in files if f.path.suffix == ".html")
        self.assertIn("# html_heading_A", text)
        self.assertIn("- Press the html_button_D button.", text)
        self.assertNotIn("<", text)

    def test_html_page_links_to_its_script_and_handler(self) -> None:
        for expected in (
            CodeRelation("html.page_html", "imports", "js.calculator_js"),
            CodeRelation(
                "html.page_html", "calls", "js.calculator_js.js_function_A"
            ),
        ):
            with self.subTest(relation=expected):
                self.assertIn(expected, self.relations)

    def test_html_page_is_retrieved_through_javascript_calls(self) -> None:
        """Follow a JavaScript function's call edge to its HTML page."""

        results = code_search(
            self.graph, self.chunks, "js_function_A", count=20
        )
        self.assertIn(
            Path("html/page_html.html"),
            {result.chunk.path for result in results},
        )

    def test_r_functions_expand_to_callers_and_callees(self) -> None:
        r = 'r.calculator_r'
        plan = plan_context_expansion(
            'Which functions call r_function_A?', self.graph
        )
        self.assertTrue(names_code('Explain r_function_A', self.graph))
        self.assertEqual(plan.functions, (f'{r}.r_function_A',))
        self.assertTrue(plan.relationship)
        self.assertIn(
            f'{r}.RChildClass.r_method_C', plan.related_functions
        )
        plain = plan_context_expansion('Explain r_function_A', self.graph)
        self.assertIn(f'{r}.r_function_B', plain.related_functions)

    def test_r_class_expands_to_its_methods(self) -> None:
        plan = plan_context_expansion(
            'Which methods belong to RChildClass?', self.graph
        )
        self.assertEqual(plan.functions, ())
        self.assertIn('exact class', plan.signals)
        self.assertIn(
            'r.calculator_r.RChildClass.r_method_C',
            plan.related_functions,
        )

    def test_javascript_functions_expand_to_callers_and_callees(self) -> None:
        js = "js.calculator_js"
        plan = plan_context_expansion(
            "Which functions call js_function_A?", self.graph
        )
        self.assertTrue(names_code("Explain js_function_A", self.graph))
        self.assertEqual(plan.functions, (f"{js}.js_function_A",))
        self.assertTrue(plan.relationship)
        self.assertIn(f"{js}.JsChildClass.js_method_C", plan.related_functions)
        plain = plan_context_expansion("Explain js_function_A", self.graph)
        self.assertIn(f"{js}.js_function_B", plain.related_functions)

    def test_javascript_class_expands_to_its_methods(self) -> None:
        plan = plan_context_expansion(
            "Which methods belong to JsChildClass?", self.graph
        )
        self.assertEqual(plan.functions, ())
        self.assertIn("exact class", plan.signals)
        self.assertIn(
            "js.calculator_js.JsChildClass.js_method_C",
            plan.related_functions,
        )


    def test_html_is_read_as_code(self) -> None:
        page = next(
            chunk for chunk in self.chunks if chunk.path.suffix == ".html"
        )
        self.assertIn('<button onclick="js_function_A(1, 2)">', page.text)

    def test_html_is_read_as_page_text_without_the_code_graph(self) -> None:
        files = read_repository_files(self.report)
        text = next(f.text for f in files if f.path.suffix == ".html")
        self.assertIn("# html_heading_A", text)
        self.assertIn("- Press the html_button_D button.", text)
        self.assertNotIn("<", text)

    def test_html_page_links_to_its_script_and_handler(self) -> None:
        for expected in (
            CodeRelation("html.page_html", "imports", "js.calculator_js"),
            CodeRelation(
                "html.page_html", "calls", "js.calculator_js.js_function_A"
            ),
        ):
            with self.subTest(relation=expected):
                self.assertIn(expected, self.relations)


if __name__ == "__main__":
    unittest.main()
