"""Tests for deterministic Python and MATLAB repository graphs."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring

from __future__ import annotations

from pathlib import Path
import unittest

from qachat.graph import build_repository_graph
from qachat.source import SourceFile, SourceUnit


class TestRepositoryGraph(unittest.TestCase):

    def test_python_definitions_calls_inheritance_and_tests(self) -> None:
        files = [
            SourceFile(
                Path("package/base.py"),
                "class Base:\n"
                "    pass\n"
                "\n"
                "def helper():\n"
                "    return 2\n",
            ),
            SourceFile(
                Path("package/service.py"),
                "from package.base import Base, helper\n"
                "\n"
                "class Service(Base):\n"
                "    def prepare(self):\n"
                "        return helper()\n"
                "\n"
                "    def run(self):\n"
                "        return self.prepare()\n",
            ),
            SourceFile(
                Path("tests/test_service.py"),
                "from package.service import Service\n"
                "\n"
                "def test_service():\n"
                "    assert Service() is not None\n",
            ),
        ]

        graph = build_repository_graph(files)

        self.assertEqual(graph.count_nodes(kind="module"), 3)
        self.assertEqual(graph.count_nodes(kind="class"), 2)
        self.assertEqual(graph.count_nodes(kind="function"), 2)
        self.assertEqual(graph.count_nodes(kind="method"), 2)
        service = graph.find_definitions("Service")[0]
        prepare = graph.find_definitions("Service.prepare")[0]
        run = graph.find_definitions("package.service.Service.run")[0]
        helper = graph.find_definitions("helper")[0]

        inheritance = graph.relations_for(service.identifier, kind="inherits")
        self.assertEqual(len(inheritance), 1)
        self.assertEqual(inheritance[0].target_name, "Base")
        self.assertEqual(inheritance[0].certainty, "exact")

        prepare_calls = graph.relations_for(
            prepare.identifier, kind="calls", direction="outgoing"
        )
        self.assertEqual(
            {relation.target_identifier for relation in prepare_calls},
            {helper.identifier},
        )
        run_calls = graph.relations_for(
            run.identifier, kind="calls", direction="outgoing"
        )
        self.assertEqual(
            {relation.target_identifier for relation in run_calls},
            {prepare.identifier},
        )
        self.assertIn(prepare, graph.neighbors(run.identifier, kind="calls"))

        test_node = graph.find_definitions("test_service")[0]
        tested = graph.relations_for(
            test_node.identifier, kind="tests", direction="outgoing"
        )
        self.assertEqual(len(tested), 1)
        self.assertEqual(tested[0].target_identifier, service.identifier)

    def test_python_does_not_guess_dynamic_calls(self) -> None:
        source = SourceFile(
            Path("example.py"),
            "class MissingChild(ExternalBase):\n"
            "    def run(self, dependency):\n"
            "        text = 'fake_call()'\n"
            "        return dependency.execute(text)\n",
        )

        graph = build_repository_graph([source])

        child = graph.find_definitions("MissingChild")[0]
        inheritance = graph.relations_for(child.identifier, kind="inherits")
        self.assertEqual(len(inheritance), 1)
        self.assertEqual(inheritance[0].certainty, "unresolved")
        self.assertIsNone(inheritance[0].target_identifier)
        run = graph.find_definitions("MissingChild.run")[0]
        self.assertEqual(
            graph.relations_for(
                run.identifier, kind="calls", direction="outgoing"
            ),
            (),
        )
        self.assertNotIn(
            "fake_call",
            {relation.target_name for relation in graph.relations},
        )

    def test_matlab_units_and_uncertain_path_resolution(self) -> None:
        child = SourceFile(
            Path("Child.m"),
            "classdef Child < Base & ExternalMixin\n"
            "    methods\n"
            "        function obj = Child()\n"
            "        end\n"
            "\n"
            "        function value = run(obj)\n"
            "            import tools.Utility\n"
            "            value = helper(obj);\n"
            "            text = 'ignored_call()';\n"
            "            % commented_call()\n"
            "            missing(obj);\n"
            "        end\n"
            "    end\n"
            "end\n",
        )
        base = SourceFile(Path("Base.m"), "classdef Base\nend\n")
        first_helper = SourceFile(
            Path("first/helper.m"),
            "function value = helper(input)\nvalue = input;\nend\n",
        )
        second_helper = SourceFile(
            Path("second/helper.m"),
            "function value = helper(input)\nvalue = input;\nend\n",
        )
        child_units = [
            SourceUnit(
                path=Path("Child.m"),
                text=child.text,
                start_line=1,
                end_line=14,
                kind="matlab_class",
                symbol="Child",
                children=(
                    SourceUnit(
                        path=Path("Child.m"),
                        text="function obj = Child()\nend\n",
                        start_line=3,
                        end_line=4,
                        kind="matlab_constructor",
                        symbol="Child.Child",
                    ),
                    SourceUnit(
                        path=Path("Child.m"),
                        text="function value = run(obj)\n",
                        start_line=6,
                        end_line=12,
                        kind="matlab_method",
                        symbol="Child.run",
                    ),
                ),
            ),
            SourceUnit(
                path=Path("Base.m"),
                text=base.text,
                start_line=1,
                end_line=2,
                kind="matlab_class",
                symbol="Base",
            ),
            SourceUnit(
                path=Path("first/helper.m"),
                text=first_helper.text,
                start_line=1,
                end_line=3,
                kind="matlab_function",
                symbol="helper",
            ),
            SourceUnit(
                path=Path("second/helper.m"),
                text=second_helper.text,
                start_line=1,
                end_line=3,
                kind="matlab_function",
                symbol="helper",
            ),
        ]

        graph = build_repository_graph(
            [child, base, first_helper, second_helper],
            child_units,
        )

        self.assertEqual(graph.count_nodes(language="matlab", kind="class"), 2)
        self.assertEqual(
            graph.count_nodes(language="matlab", kind="function"),
            2,
        )
        self.assertEqual(
            graph.count_nodes(language="matlab", kind="method"),
            2,
        )
        run = graph.find_definitions("Child.run", language="matlab")[0]
        child_node = graph.find_definitions("Child", language="matlab")[0]

        inheritance = graph.relations_for(
            child_node.identifier,
            kind="inherits",
        )
        certainty = {
            relation.target_name: relation.certainty
            for relation in inheritance
        }
        self.assertEqual(certainty["Base"], "ambiguous")
        self.assertEqual(certainty["ExternalMixin"], "unresolved")

        helper_calls = [
            relation
            for relation in graph.relations_for(
                run.identifier, kind="calls", direction="outgoing"
            )
            if relation.target_name == "helper"
        ]
        self.assertEqual(len(helper_calls), 2)
        self.assertTrue(
            all(relation.certainty == "ambiguous" for relation in helper_calls)
        )
        references = graph.relations_for(
            run.identifier, kind="references", direction="outgoing"
        )
        self.assertIn("missing", {item.target_name for item in references})
        all_targets = {relation.target_name for relation in graph.relations}
        self.assertNotIn("ignored_call", all_targets)
        self.assertNotIn("commented_call", all_targets)

        imports = graph.relations_for(
            run.identifier, kind="imports", direction="outgoing"
        )
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].target_name, "tools.Utility")
        self.assertEqual(imports[0].certainty, "exact")

    def test_empty_and_invalid_source_are_safe(self) -> None:
        empty = build_repository_graph([])
        self.assertEqual(empty.nodes, ())
        self.assertEqual(empty.relations, ())

        invalid = build_repository_graph(
            [SourceFile(Path("broken.py"), "def broken(:\n")]
        )
        self.assertEqual(invalid.count_nodes(kind="module"), 1)
        self.assertEqual(invalid.count_nodes(kind="function"), 0)
        self.assertEqual(invalid.relations, ())


if __name__ == "__main__":
    unittest.main()
