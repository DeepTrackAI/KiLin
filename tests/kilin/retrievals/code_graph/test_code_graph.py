"""Test structural code-graph construction and name resolution.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from kilin.retrievals.code_graph import code_graph as cg
from kilin.retrievals.code_graph.extensions import code_graph_m as matlab
from kilin.retrievals.code_graph.extensions import code_graph_py as py
from tests.kilin.retrievals._support import source_file, source_unit


class TestCodeGraphNames(unittest.TestCase):
    """Check stable qualified naming and name resolution."""

    def test_module_name_uses_relative_path(self) -> None:
        """Convert paths without importing modules or collapsing init names.

        """
        self.assertEqual(cg._module_name(Path("pkg/helper.py")),
                         "pkg.helper")
        self.assertEqual(cg._module_name(Path("pkg/__init__.py")),
                         "pkg.__init__")

    def test_qualified_names_and_contexts(self) -> None:
        """Avoid adding an artificial name for context-only units.

        """
        parent = "pkg.code"
        cases = (
            ("method", "python_function", "pkg.code.method"),
            ("code.method", "python_method", "pkg.code.method"),
            ("code.method", "python_class", "pkg.code.method"),
            ("code", "python_class", "pkg.code.code"),
            ("context", "python_class_context", parent),
            ("context", "python_module_context", parent),
            ("method", "markdown_block", parent),
            (None, "python_function", parent),
        )
        for symbol, kind, expected in cases:
            with self.subTest(symbol=symbol, kind=kind):
                unit = source_unit(
                    "pkg/code.py", "x", symbol=symbol, kind=kind
                )
                self.assertEqual(cg._unit_name(unit, parent), expected)

    def test_known_name_resolution_is_conservative(self) -> None:
        """Resolve unique names, leave ambiguous names unchanged.

        """
        known: dict[str, set[str]] = {}
        cg._remember_name(known, "pkg.alpha")
        self.assertEqual(cg._resolve_name("alpha", known), "pkg.alpha")
        self.assertEqual(cg._resolve_name("missing", known), "missing")
        cg._remember_name(known, "other.alpha")
        self.assertEqual(cg._resolve_name("alpha", known), "alpha")
        # An explicit qualified reference names its own entity.
        self.assertEqual(cg._resolve_name("pkg.alpha", known), "pkg.alpha")

    def test_dotted_suffix_and_enclosing_scope_resolve_names(self) -> None:
        """Qualified suffixes and Python's scope lookup break ties.

        """
        known: dict[str, set[str]] = {}
        for name in (
            "pkg.py_ext.extract", "pkg.pdf_ext.extract",
            "pkg.match._candidate", "pkg.store._candidate",
            "pkg.store.Store._candidate", "pkg.other.close",
            "pkg.store.Store.close",
        ):
            cg._remember_name(known, name)
        self.assertEqual(
            cg._resolve_name("py_ext.extract", known), "pkg.py_ext.extract"
        )
        self.assertEqual(cg._resolve_name("extract", known), "extract")
        self.assertEqual(
            cg._resolve_name(
                "_candidate", known, "pkg.match.find", "pkg.match"
            ),
            "pkg.match._candidate",
        )
        scope = ("pkg.store.Store.add", "pkg.store")
        # A plain name in a method is the module's, not the class's.
        self.assertEqual(
            cg._resolve_name("_candidate", known, *scope),
            "pkg.store._candidate",
        )
        self.assertEqual(
            cg._resolve_name("self._candidate", known, *scope),
            "pkg.store.Store._candidate",
        )
        self.assertEqual(
            cg._resolve_name("self.close", known, *scope),
            "pkg.store.Store.close",
        )
        # Another object's attribute is not resolved by the caller's scope.
        self.assertEqual(
            cg._resolve_name("stream.close", known, *scope), "stream.close"
        )
        self.assertEqual(
            cg._resolve_name("_candidate", known, "pkg.other.f", "pkg.other"),
            "_candidate",
        )



class TestBuildCodeGraph(unittest.TestCase):
    """Verify structural graph content and extension boundaries."""

    def test_empty_inputs_are_rejected(self) -> None:
        """Both file records and source units are required.

        """
        f = source_file("src/example.py", "def run(): pass\n")
        u = source_unit("src/example.py", f.text, symbol="run")
        for files, units in (([], [u]), ([f], []), ([], [])):
            with self.subTest(files=files, units=units):
                with self.assertRaises(ValueError):
                    cg.build_code_graph(files, units)

    def test_nested_entities_and_containment(self) -> None:
        """Retain code hierarchy and deterministically sort graph members.

        """
        f = source_file("src/code.py", "class Example: pass\n")
        child = source_unit(
            "src/code.py", "def run(): pass", symbol="Example.run",
            kind="python_method"
        )
        parent = source_unit(
            "src/code.py", "class Example: pass", symbol="Example",
            kind="python_class", children=(child,)
        )
        result = cg.build_code_graph([f], [parent, parent])
        names = [item.name for item in result.entities]
        self.assertEqual(names, ["src.code", "src.code.Example",
                                 "src.code.Example.run"])
        relations = {(r.source, r.relation, r.target)
                     for r in result.relations}
        self.assertIn(("src.code", "contains", "src.code.Example"),
                      relations)
        self.assertIn(("src.code.Example", "contains",
                       "src.code.Example.run"), relations)
        self.assertEqual(len(result.entities), len(set(result.entities)))
        with self.assertRaises(FrozenInstanceError):
            result.entities = ()

    def test_context_only_unit_does_not_create_self_edge(self) -> None:
        """A module context may map to its parent without self containment.

        """
        f = source_file("code.py", "x = 1")
        unit = source_unit(
            "code.py", "x = 1", kind="python_module_context"
        )
        result = cg.build_code_graph([f], [unit])
        self.assertEqual(result.relations, ())

    def test_missing_extension_keeps_structural_graph(self) -> None:
        """Ignore an unsupported suffix extension, not the source content.

        """
        f = source_file("readme.unknown", "hello")
        u = source_unit("readme.unknown", "hello", symbol="text")
        result = cg.build_code_graph([f], [u])
        self.assertEqual(len(result.entities), 2)
        self.assertIn(cg.CodeRelation("readme", "contains", "readme.text"),
                      result.relations)

    def test_extension_validation(self) -> None:
        """Reject extensions that return wrong relation containers or types.

        """
        f = source_file("src/file.fake", "hello")
        u = source_unit("src/file.fake", "hello", symbol="function")
        for returned in ([], {"wrong"}):
            with self.subTest(returned=returned):
                extension = SimpleNamespace(
                    extract_code_relations=lambda *_: returned
                )
                with patch.object(cg, "import_module", return_value=extension):
                    with self.assertRaisesRegex(RuntimeError, "extension"):
                        cg.build_code_graph([f], [u])

    def test_extension_internal_import_error_propagates(self) -> None:
        """Do not conceal missing dependencies inside a recognized hook.

        """
        f = source_file("src/file.foo", "hello")
        u = source_unit("src/file.foo", "hello", symbol="function")

        def missing(_: str) -> object:
            """Simulate an extension dependency failure.

            Parameters
            ----------
            _: str
                Requested extension module.

            Returns
            -------
            object
                Never returned.

            """
            raise ModuleNotFoundError("missing helper", name="helper_dep")

        with patch.object(cg, "import_module", side_effect=missing):
            with self.assertRaises(ModuleNotFoundError):
                cg.build_code_graph([f], [u])

    def test_python_extensions_extract_import_and_call(self) -> None:
        """Use real syntax-tree extraction for code relationships.

        """
        text = (
            "from math import sqrt\n"
            "def helper():\n    return 2\n"
            "def run():\n    return helper() + sqrt(9)\n"
        )
        f = source_file("pkg/number.py", text)
        units = [
            source_unit("pkg/number.py", text, symbol="helper"),
            source_unit("pkg/number.py", text, symbol="run"),
        ]
        graph = cg.build_code_graph([f], units)
        relations = {(r.source, r.relation, r.target)
                     for r in graph.relations}
        self.assertIn(("pkg.number", "imports", "math.sqrt"), relations)
        self.assertIn(("pkg.number.run", "calls", "pkg.number.helper"),
                      relations)



class TestRemainingGraphBranches(unittest.TestCase):
    """Cover specialized naming and optional extension paths."""

    def test_suffixless_file_keeps_only_structural_entities(self) -> None:
        """A source without an extension is indexed without a language hook.

        """
        f = source_file("README", "Title")
        unit = source_unit("README", "Title", symbol="title")
        result = cg.build_code_graph([f], [unit])
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(len(result.relations), 1)

    def test_module_constant_names_resolve_across_files(self) -> None:
        """The graph builder registers module constants before visiting uses.

        """
        definition = source_file("settings.py", "MODE = 4\n")
        use = source_file(
            "client.py", "def run():\n    return MODE\n"
        )
        unit = source_unit("client.py", use.text, symbol="run")
        result = cg.build_code_graph([use, definition], [unit])
        self.assertIn(
            cg.CodeRelation("client.run", "references", "settings"),
            result.relations,
        )

    def test_class_equal_to_parent_uses_named_fallback(self) -> None:
        """A special class-name collision receives a qualified suffix.

        """
        cls = source_unit("pkg.py", "x", symbol="pkg",
                          kind="python_class")
        same = source_unit("pkg.py", "x", symbol="pkg",
                           kind="python_function")
        self.assertEqual(cg._unit_name(cls, "pkg"), "pkg.pkg")
        self.assertEqual(cg._unit_name(same, "pkg"), "pkg")

    def test_unqualified_class_without_base_and_dynamic_calls(self) -> None:
        """Skip unsupported static targets in class and function calls.

        """
        f = source_file(
            "example.py",
            "class Example(factory()):\n    pass\n"
            "def run():\n    callback()()\n"
            "class Plain:\n    pass\n",
        )
        relations = py.extract_code_relations(f, {})
        self.assertFalse(any(r.relation == "inherits" for r in relations))
        self.assertIn(cg.CodeRelation("example.run", "calls", "callback"),
                      relations)

    def test_constant_reference_within_same_module_not_redundant(self) -> None:
        """A module constant referenced at its defining module is not an edge.

        """
        f = source_file("config.py", "VALUE = 1\nprint(VALUE)\n")
        relations = py.extract_code_relations(f, {"VALUE": {"config"}})
        self.assertFalse(any(r.relation == "references" for r in relations))

    def test_matlab_class_without_inheritance(self) -> None:
        """A MATLAB class without a superclass yields no inherits edge.

        """
        f = source_file(
            "Sample.m", "classdef Sample\nfunction x = run(obj)\n"
            "helper(obj);\n",
        )
        relations = matlab.extract_code_relations(f, {})
        self.assertIn(
            cg.CodeRelation("Sample.Sample.run", "calls", "helper"),
            relations,
        )
        self.assertFalse(any(r.relation == "inherits" for r in relations))



class TestGraphDeterminismRegression(unittest.TestCase):
    """Track cross-process ordering when graph entities share a name."""

    def test_equal_name_order_independent_of_hash_seed(self) -> None:
        """Equal-name entities are reproducibly ordered across processes.

        Notes
        -----
        A module context and its source file may share a qualified name.
        Their sorted order must not depend on hash randomization because
        graph serialization contributes to the repository fingerprint.

        """
        program = (
            "from pathlib import Path; "
            "from kilin.sources.sources import SourceFile, SourceUnit; "
            "from kilin.retrievals.code_graph import build_code_graph; "
            "from kilin.retrievals.persistence import "
            "_repository_fingerprint; "
            "p=Path('x.py'); "
            "k=['python_module_context','python_class_context',"
            "'matlab_file_context','extra_context']; "
            "u=[SourceUnit(p,'hello',1,1,t) for t in k]; "
            "g=build_code_graph([SourceFile(p,'hello')],u); "
            "print(','.join(e.kind for e in g.entities)); "
            "print(_repository_fingerprint([],g))"
        )
        outputs = []
        for seed in (1, 2, 3, 4):
            env = dict(os.environ, PYTHONHASHSEED=str(seed))
            process = subprocess.run(
                [sys.executable, "-c", program],
                check=True, capture_output=True, text=True, env=env,
            )
            outputs.append(process.stdout.strip())
        self.assertEqual(len(set(outputs)), 1)
        kinds, digest = outputs[0].splitlines()
        self.assertEqual(
            kinds,
            "extra_context,matlab_file_context,python_class_context,"
            "python_module_context,source_file",
        )
        self.assertEqual(len(digest), 64)

    def test_equal_name_and_kind_order_by_path(self) -> None:
        """Path breaks remaining ties even when names and kinds match.

        Notes
        -----
        Both paths become the qualified name ``a.b``. Their distinct source
        file and module-context records must survive deduplication and keep
        their order even when source inputs arrive in a different order.

        """
        files = [
            source_file("a.b.py", "value = 1\n"),
            source_file("a/b.py", "value = 2\n"),
        ]
        units = [
            source_unit(
                "a.b.py", "value = 1\n", kind="python_module_context"
            ),
            source_unit(
                "a/b.py", "value = 2\n", kind="python_module_context"
            ),
        ]
        expected = [
            ("a.b", "python_module_context", "a.b.py"),
            ("a.b", "python_module_context", "a/b.py"),
            ("a.b", "source_file", "a.b.py"),
            ("a.b", "source_file", "a/b.py"),
        ]
        graph = cg.build_code_graph(files, units)
        actual = [
            (item.name, item.kind, item.path.as_posix())
            for item in graph.entities
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(
            graph,
            cg.build_code_graph(files[::-1], units[::-1]),
        )

