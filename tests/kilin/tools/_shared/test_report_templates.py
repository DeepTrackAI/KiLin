"""Unit tests for project report templates in the shared template format.

Check the ``scope`` key, the project placeholders, per-document sections
(entries and tables), the messages for broken templates, and the report
checks for code-written sections.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import unittest

from kilin.tools._shared.report_templates import (
    SCOPE_DOCUMENT,
    SCOPE_PROJECT,
    SECTION_SOURCE_DOCUMENTS,
    SECTION_SOURCE_KEY_DOCUMENTS,
    SECTION_SOURCE_NOTES,
    SECTION_SOURCE_PER_DOCUMENT,
    TemplateRules,
    check_report,
    parse_template,
)

RULES = TemplateRules(limit_tolerance=0.1, min_total_words=50)
PROJECT = """---
title: Project Overview
description: What a collection of papers says, together.
max_words: 300
scope: project
---

# Headline (max 90 characters)

**Documents:** {documents}

## Main Themes (40-60 words)

The themes the documents share.

## Deep-Learning Models (per document)

When: the document trains or uses a deep-learning model.
Do: name the architecture and the task (max 30 words).
Write the entries in the document's own terms.

## Public Datasets (per document, table)

When: the document uses or releases a public dataset.
Do: | Dataset | Size | Licence |

## Where to Start (2-3 items)

The documents to read first, chosen from {key_documents}.
"""


def _template(body: str, scope: str = "project"):
    """Parse a small template of one scope.

    Parameters
    ----------
    body: str
        The sections.
    scope: str, optional
        The scope written in the front matter. Defaults to project.

    Returns
    -------
    ReportTemplate
        The parsed template.

    """

    return parse_template(
        f"---\ntitle: T\nscope: {scope}\n---\n\n{body}", "t", RULES
    )


class TestProjectTemplates(unittest.TestCase):

    def test_sections(self) -> None:
        template = parse_template(PROJECT, "project_overview", RULES)
        self.assertEqual(template.scope, SCOPE_PROJECT)
        self.assertEqual(
            [(s.name, s.source, s.inline) for s in template.sections],
            [("Documents", SECTION_SOURCE_DOCUMENTS, True),
             ("Main Themes", SECTION_SOURCE_NOTES, False),
             ("Deep-Learning Models", SECTION_SOURCE_PER_DOCUMENT, False),
             ("Public Datasets", SECTION_SOURCE_PER_DOCUMENT, False),
             ("Where to Start", SECTION_SOURCE_KEY_DOCUMENTS, False)],
        )
        models, datasets = template.sections[2:4]
        self.assertEqual(
            models.when,
            "the document trains or uses a deep-learning model.",
        )
        self.assertEqual(models.do, "name the architecture and the task")
        self.assertEqual(models.words, (0, 30))
        self.assertEqual(models.guidance,
                         "Write the entries in the document's own terms.")
        self.assertEqual(datasets.columns, ("Dataset", "Size", "Licence"))
        self.assertEqual(template.sections[4].items, (2, 3))

    def test_document_templates_are_unchanged(self) -> None:
        template = _template(
            "**Source:** {citation}\n\n## A (10-20 words)\n\nText.\n",
            "document",
        )
        self.assertEqual(template.scope, SCOPE_DOCUMENT)
        no_scope = parse_template("## A (10-20 words)\n\nText.\n", "a", RULES)
        self.assertEqual(no_scope.scope, SCOPE_DOCUMENT)

    def test_broken_templates(self) -> None:
        for body, scope, message in (
            ("## A (per document)\n\nWhen: x\nDo: y\n", "document",
             "per-document sections are for project templates"),
            ("## A (per document)\n\nDo: y\n", "project",
             "needs a When: line"),
            ("## A (per document)\n\nWhen: x\n", "project",
             "needs a When: line"),
            ("## A (per document, 2-3 items)\n\nWhen: x\nDo: y\n", "project",
             "has no limits in its heading"),
            ("## A (per document, table)\n\nWhen: x\nDo: list them\n",
             "project", "names its columns"),
            ("## A (table)\n\nWhen: x\nDo: | B |\n", "project",
             "(per document, table)"),
            ("## A (per document)\n\nWhen: x\nDo: y {documents}\n",
             "project", "has no placeholders"),
            ("**Source:** {citation}\n\n## A\n\nText.\n", "project",
             "{citation} is for document templates; use {documents} or "
             "{key_documents}"),
            ("## A (2 items)\n\nFrom {key_documents}.\n", "document",
             "{key_documents} is for project templates"),
            ("## A\n\nFrom {key_documents}.\n", "project",
             "{key_documents} needs an item limit"),
            ("## A\n\nText.\n", "folder", "scope must be document"),
        ):
            with self.subTest(body=body, scope=scope):
                with self.assertRaisesRegex(ValueError, message.replace(
                    "(", r"\(").replace(")", r"\)").replace(
                    "{", r"\{").replace("}", r"\}")):
                    _template(body, scope)


class TestProjectReportChecks(unittest.TestCase):

    def test_code_written_sections_are_only_checked_for_content(self) -> None:
        template = parse_template(PROJECT, "project_overview", RULES)
        themes = " ".join(["theme"] * 50)
        entries = "\n".join(
            f"- **doc{n}**: {' '.join(['word'] * 40)} [{n}]"
            for n in range(1, 30)
        )
        report = (
            "# A headline\n\n**Documents:** 29 documents.\n\n"
            f"## Main Themes\n\n{themes}\n\n"
            f"## Deep-Learning Models\n\n{entries}\n\n"
            "## Public Datasets\n\n| Dataset | Size | Licence |\n"
            "| --- | --- | --- |\n| A | 1 | MIT |\n\n"
            "## Where to Start\n\n- doc1\n- doc2\n"
        )
        check = check_report(report, template, RULES)
        # 1,160 words of entries, but max_words is 300: entries do not
        # count, and are not held to a section limit.
        self.assertTrue(check.valid, check)
        empty = report.replace(entries, "")
        self.assertIn("Deep-Learning Models",
                      check_report(empty, template, RULES).empty)


if __name__ == "__main__":
    unittest.main()
