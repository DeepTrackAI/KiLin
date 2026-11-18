"""Test HTML code-graph extraction."""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.retrievals.code_graph import code_graph as cg
from kilin.retrievals.code_graph.extensions import code_graph_html as html
from tests.kilin.retrievals._support import source_file


class TestHtmlCodeGraphExtension(unittest.TestCase):
    """Link a page to the scripts it loads and its inline handlers."""

    def test_scripts_and_handlers(self) -> None:
        page = source_file(
            "web/index.html",
            '<script src="app.js?v=2"></script>\n'
            '<script src="../lib/util.js"></script>\n'
            '<script src="/static/main.js"></script>\n'
            '<script src="https://cdn.example.org/x.js"></script>\n'
            '<script src="qrc:///qtwebchannel/qwebchannel.js"></script>\n'
            '<button onclick="save()">Save</button>\n'
            '<form onsubmit="return check(this)"></form>\n',
        )
        known = {"save": {"web.app.save"}}
        self.assertEqual(
            html.extract_code_relations(page, known),
            {
                cg.CodeRelation("web.index", "imports", "web.app"),
                cg.CodeRelation("web.index", "imports", "lib.util"),
                cg.CodeRelation("web.index", "imports", "static.main"),
                cg.CodeRelation("web.index", "calls", "web.app.save"),
                cg.CodeRelation("web.index", "calls", "check"),
            },
        )

    def test_comments_templates_and_script_literals_are_not_links(
        self,
    ) -> None:
        page = source_file(
            "web/index.html",
            '<!-- <script src="ghost.js"></script> -->\n'
            '<!-- <button onclick="ghost()">x</button> -->\n'
            '<template><script src="inert.js"></script>'
            '<button onclick="inert()">x</button></template>\n'
            '<script>const html = `<script src="fake.js"></script>`;'
            'const handler = `onclick="fake()"`;</script>\n'
            '<button data-onclick="fake()">x</button>\n'
            '<button onclick="return save()">Save</button>\n',
        )
        self.assertEqual(
            html.extract_code_relations(page, {"save": {"web.save"}}),
            {cg.CodeRelation("web.index", "calls", "web.save")},
        )

    def test_unquoted_src_and_handler_entities(self) -> None:
        page = source_file(
            "web/index.html",
            '<script src=app.js></script>'
            '<button onclick="save&#40;1&#41;">Save</button>',
        )
        self.assertEqual(
            html.extract_code_relations(page, {}),
            {
                cg.CodeRelation("web.index", "imports", "web.app"),
                cg.CodeRelation("web.index", "calls", "save"),
            },
        )

    def test_page_text_has_no_relations(self) -> None:
        page = source_file("notes.html", "# Notes\n\nSave the file.\n")
        self.assertEqual(html.extract_code_relations(page, {}), set())


if __name__ == "__main__":
    unittest.main()
