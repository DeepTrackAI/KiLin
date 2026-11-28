"""Validate the local technical documentation without external services.

The checks keep Markdown targets, editable SVG diagrams, and the module
ownership matrix synchronized with the repository. They use only the Python
standard library, so they also run in the lightweight documentation job.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import ast
from collections import Counter
from html import unescape
import json
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SVG_NS = "{http://www.w3.org/2000/svg}"
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
# Paths in HTML inside Markdown: the README logo's <img src> and srcset.
HTML_PATTERN = re.compile(r"\b(?:src|srcset|href)=\"([^\"]+)\"")
MODULE_PATTERN = re.compile(r"\| \[`(kilin/[^`]+\.py)`\]\(")


def _without_fences(text: str) -> str:
    """Remove fenced examples while preserving surrounding Markdown.

    Parameters
    ----------
    text: str
        Markdown source, using backtick or tilde fences.

    Returns
    -------
    str
        Source without fenced blocks. Examples cannot become headings.

    """

    lines: list[str] = []
    fence = ""
    for line in text.splitlines():
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if (
                marker and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence) and not marker[2].strip()
            ):
                fence = ""
            continue
        if marker:
            fence = marker[1]
        else:
            lines.append(line)
    return "\n".join(lines)


def _heading_anchors(text: str) -> set[str]:
    """Collect local anchors for the heading forms used by these documents.

    Parameters
    ----------
    text: str
        Markdown source with ATX or setext headings and optional HTML IDs.

    Returns
    -------
    set[str]
        Heading slugs with collision suffixes, plus explicit HTML anchors.
        This is a local-reference check, not a full Markdown renderer.

    """

    body = _without_fences(text)
    used: set[str] = set()
    lines = body.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$", line)
        title = match[1] if match else ""
        if (
            not title and index and lines[index - 1].strip()
            and re.fullmatch(r" {0,3}(?:=+|-+)\s*", line)
        ):
            title = lines[index - 1].strip()
        if not title:
            continue
        title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", title)
        title = re.sub(r"<[^>]*>", "", title)
        slug = re.sub(r"[^\w\s-]", "", unescape(title).lower())
        slug = re.sub(r"\s", "-", slug)
        anchor = slug
        suffix = 0
        while anchor in used:
            suffix += 1
            anchor = f"{slug}-{suffix}"
        used.add(anchor)
    used.update(re.findall(r'<[^>]+\b(?:id|name)="([^"]+)"', body))
    return used


def _section(text: str, heading: str) -> str:
    """Read one second-level section without accepting a missing heading.

    Parameters
    ----------
    text: str
        Complete Markdown document.
    heading: str
        Exact second-level heading, without the leading hashes.

    Returns
    -------
    str
        Section body, including any third-level subsections.

    Raises
    ------
    ValueError
        If the requested heading is absent.

    """

    marker = f"\n## {heading}\n"
    if marker not in text:
        raise ValueError(f"Missing documentation section: {heading}")
    return text.split(marker, 1)[1].split("\n## ", 1)[0]


def _assignment(path: Path, name: str) -> ast.expr:
    """Read an explicit source assignment without importing the package.

    Parameters
    ----------
    path: pathlib.Path
        Python module to parse using the standard library.
    name: str
        Name of the required module-level assignment.

    Returns
    -------
    ast.expr
        Assigned expression, still unevaluated.

    Raises
    ------
    ValueError
        If no explicit assignment for the requested name exists.

    """

    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            if node.value is not None:
                return node.value
    raise ValueError(f"Missing explicit assignment: {path.name}:{name}")


class TestDocumentationAssets(unittest.TestCase):
    """Check repository-local references and maintained inventories."""

    def test_markdown_targets_exist(self) -> None:
        """Resolve local paths without requiring a network connection."""

        documents = [
            ROOT / "README.md",
            *sorted(DOCS.glob("*.md")),
            *sorted((ROOT / "kilin").rglob("README.md")),
        ]
        errors: list[str] = []
        for document in documents:
            text = _without_fences(
                document.read_text(encoding="utf-8")
            )
            for target in (
                *LINK_PATTERN.findall(text),
                *HTML_PATTERN.findall(text),
            ):
                if target.startswith(("https://", "http://", "mailto:")):
                    continue
                path = unquote(target.split("#", 1)[0])
                if not path:
                    continue
                if not (document.parent / path).exists():
                    name = document.relative_to(ROOT).as_posix()
                    errors.append(f"{name}: {target}")
        self.assertEqual(errors, [], "Invalid local Markdown targets")

    def test_public_readme_images_and_links_are_branch_relative(
        self,
    ) -> None:
        """Use current-revision paths on GitHub, not hard-coded main.

        The public repository is not contacted during this test.

        """

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("/DeepTrackAI/KiLin/blob/main/", readme)
        self.assertNotIn("/DeepTrackAI/KiLin/tree/main/", readme)
        self.assertNotIn(
            "raw.githubusercontent.com/DeepTrackAI/KiLin/main/",
            readme,
        )
        images = (
            "assets/brand/10-lockup-horizontal-light.svg",
            "assets/brand/11-lockup-horizontal-dark.svg",
        )
        self.assertIn(f'src="{images[0]}"', readme)
        self.assertIn(f'srcset="{images[1]}"', readme)
        for path in images:
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_file())

        links = LINK_PATTERN.findall(_without_fences(readme))
        self.assertTrue(links)
        for target in links:
            with self.subTest(target=target):
                parsed = urlparse(target)
                if not parsed.scheme and parsed.path:
                    relative = unquote(parsed.path)
                    self.assertTrue((ROOT / relative).exists(), target)
        anchors = _heading_anchors(readme)
        for anchor in re.findall(r'<a href="#([^"]+)">', readme):
            self.assertIn(anchor, anchors)
        self.assertIn("0.1.0b1", readme)
        self.assertIn("https://github.com/DeepTrackAI/KiLin", readme)

    def test_svg_figures_are_valid_and_documented(self) -> None:
        """Require accessible, self-contained SVGs used by owner pages."""

        figures = sorted((DOCS / "figures").glob("*.svg"))
        self.assertTrue(figures, "No architecture diagrams were found")
        owner_pages = [
            path.read_text(encoding="utf-8")
            for path in DOCS.glob("*.md")
            if path.name != "diagrams.md"
        ]
        for figure in figures:
            with self.subTest(figure=figure.name):
                diagram = ElementTree.parse(figure).getroot()
                self.assertEqual(diagram.tag, f"{SVG_NS}svg")
                for tag in ("title", "desc"):
                    description = diagram.find(f"{SVG_NS}{tag}")
                    self.assertIsNotNone(description)
                    self.assertTrue("".join(description.itertext()).strip())
                reference = f"figures/{figure.name}"
                self.assertTrue(
                    any(reference in page for page in owner_pages),
                    f"No owner document embeds {reference}",
                )

    def test_module_map_covers_every_production_module(self) -> None:
        """Assign exactly one primary documentation owner per module."""

        expected = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "kilin").rglob("*.py")
        }
        inventory = MODULE_PATTERN.findall(
            (DOCS / "module_map.md").read_text(encoding="utf-8")
        )
        counts = Counter(inventory)
        self.assertEqual(set(counts), expected)
        self.assertEqual(
            sorted(name for name, count in counts.items() if count != 1),
            [],
        )


class TestDocumentationContracts(unittest.TestCase):
    """Check documented inventories against source, without loading models.

    """

    def test_heading_parser_ignores_examples_and_disambiguates(self) -> None:
        """Cover duplicate headings, UTF-8, setext, markup and both fences."""

        source = (
            "# K\u00edL\u00edn & `API`\n## Same\n## Same\n## Same-1\n"
            "## [Linked](target.md) ###\nSetext\n------\n"
            "```python\n# Not a heading\n```\n"
            "~~~text\n# Also not a heading\n~~~\n"
            '<a id="explicit"></a>\n'
        )
        self.assertEqual(_heading_anchors(source), {
            "k\u00edl\u00edn--api", "same", "same-1", "same-1-1",
            "linked", "setext", "explicit",
        })

    def test_local_heading_fragments_resolve(self) -> None:
        """Check fragments as well as paths, excluding fenced examples."""

        checked = 0
        for document in sorted(DOCS.glob("*.md")):
            body = _without_fences(document.read_text(encoding="utf-8"))
            for target in LINK_PATTERN.findall(body):
                if target.startswith(("https://", "http://", "mailto:")):
                    continue
                path, marker, fragment = target.partition("#")
                if not marker or not fragment:
                    continue
                destination = (
                    document.parent / unquote(path) if path else document
                )
                if destination.suffix.lower() != ".md":
                    continue
                with self.subTest(document=document.name, target=target):
                    self.assertTrue(destination.is_file())
                    anchors = _heading_anchors(
                        destination.read_text(encoding="utf-8")
                    )
                    self.assertIn(unquote(fragment), anchors)
                    checked += 1
        self.assertGreater(checked, 0)

    def test_module_owner_counts_match_inventory(self) -> None:
        """Derive every owner total from the maintained module rows."""

        document = (DOCS / "module_map.md").read_text(encoding="utf-8")
        counts: Counter[str] = Counter()
        for line in document.splitlines():
            if MODULE_PATTERN.match(line):
                cell = line.split("|")[3].strip()
                owner = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", cell)
                self.assertIsNotNone(owner, line)
                label, target = owner.groups()
                if target.startswith("../kilin/tools/"):
                    label = "Per-tool README"
                counts[label] += 1
        summary = _section(document, "Module owner summary")
        rendered = {
            name.strip(): int(number)
            for name, number in re.findall(
                r"^\| ([^|]+) \| (\d+) \|$", summary, re.MULTILINE
            )
        }
        self.assertEqual(rendered, dict(counts))
        total = re.search(r"Total production Python modules: (\d+)", summary)
        self.assertIsNotNone(total)
        self.assertEqual(int(total[1]), sum(counts.values()))

    def test_engine_action_inventory_matches_dispatcher(self) -> None:
        """Discover actual dispatcher actions rather than a second list."""

        source = ROOT / "kilin/gui/engine.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        actions = set()
        for node in ast.walk(main):
            if (
                isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id == "action"
                and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq)
                and isinstance(node.comparators[0], ast.Constant)
            ):
                actions.add(node.comparators[0].value)
        self.assertTrue(actions)
        document = (DOCS / "gui_protocol.md").read_text(encoding="utf-8")
        inventory = _section(document, "Engine command inventory")
        documented = re.findall(
            r"^\| `([a-z_]+)` \|", inventory, re.MULTILINE
        )
        self.assertEqual(set(documented), actions)
        self.assertEqual(len(documented), len(set(documented)))

    def test_frontend_event_inventory_matches_handlers(self) -> None:
        """Read event names outside payload parentheses in the table."""

        source = (ROOT / "kilin/gui/web/app.js").read_text(encoding="utf-8")
        # Deliberately scoped to the current onEvent conditional syntax.
        # A switch/schema refactor must update this characterization too.
        handlers = set(re.findall(r'data\.event\s*===\s*"([a-z_]+)"', source))
        self.assertTrue(handlers)
        document = (DOCS / "gui_protocol.md").read_text(encoding="utf-8")
        inventory = _section(document, "Frontend event inventory").split(
            "\n### ", 1
        )[0]
        documented = []
        for line in inventory.splitlines():
            if not line.startswith("| "):
                continue
            while re.search(r"\([^()]*\)", line):
                line = re.sub(r"\([^()]*\)", "", line)
            # Event groups lead each semicolon-delimited payload clause;
            # later prose may name a command, not another emitted event.
            for group in line.split("|")[2].split(";"):
                names = re.match(
                    r"\s*(`(?:[a-z_]+)`(?:\s*,\s*`[a-z_]+`)*)", group
                )
                if names:
                    documented.extend(re.findall(r"`([a-z_]+)`", names[1]))
        self.assertEqual(set(documented), handlers)
        self.assertEqual(len(documented), len(set(documented)))

    def test_stop_grace_period_matches_window(self) -> None:
        """Keep the stated fallback delay tied to the Qt-owned constant."""

        path = ROOT / "kilin/gui/window.py"
        expected = ast.literal_eval(_assignment(path, "CANCEL_GRACE_MS"))
        document = (DOCS / "gui_protocol.md").read_text(encoding="utf-8")
        section = _section(document, "Errors, Stop, and restart")
        duration = re.search(
            r"`CANCEL_GRACE_MS`.*?\(([\d,]+) milliseconds\)", section, re.S
        )
        self.assertIsNotNone(duration)
        self.assertEqual(int(duration[1].replace(",", "")), expected)

    def test_workspace_operations_use_exported_names(self) -> None:
        """Validate facade and submodule examples against their exports."""

        document = (DOCS / "project_workspaces.md").read_text(
            encoding="utf-8"
        )
        operations = re.findall(
            r"^\| `(kilin\.[\w.]+)\(", document, re.MULTILINE
        )
        self.assertTrue(operations)
        for operation in operations:
            module, name = operation.rsplit(".", 1)
            path = ROOT.joinpath(*module.split("."))
            path = path / "__init__.py" if path.is_dir() else path.with_suffix(
                ".py"
            )
            with self.subTest(operation=operation):
                self.assertTrue(path.is_file())
                exports = ast.literal_eval(_assignment(path, "__all__"))
                self.assertIn(name, exports)

    def test_registered_tools_have_linked_local_references(self) -> None:
        """Check registry keys, package defaults and documentation together."""

        document = (DOCS / "module_map.md").read_text(encoding="utf-8")
        names = set()
        for path in sorted((ROOT / "kilin/tools").glob("*/__init__.py")):
            if path.parent.name.startswith("_"):
                continue
            descriptor = _assignment(path, "PLUGIN")
            self.assertIsInstance(descriptor, ast.Call)
            self.assertIsInstance(descriptor.func, ast.Name)
            self.assertEqual(descriptor.func.id, "Tool")
            keywords = {item.arg: item.value for item in descriptor.keywords}
            name = ast.literal_eval(keywords["name"])
            self.assertEqual(name, path.parent.name)
            self.assertNotIn(name, names)
            names.add(name)
            with self.subTest(tool=name):
                package = ROOT / "kilin/tools" / name
                self.assertTrue((package / "README.md").is_file())
                self.assertTrue((package / "defaults.json").is_file())
                self.assertIn(f"../kilin/tools/{name}/README.md", document)
        packaged = {
            file.parent.name
            for file in (ROOT / "kilin/tools").glob("*/defaults.json")
        }
        self.assertEqual(names, packaged)

    def test_json_examples_are_complete_json(self) -> None:
        """Keep JSON examples parseable without executing user snippets."""

        checked = 0
        for document in sorted(DOCS.glob("*.md")):
            text = document.read_text(encoding="utf-8")
            for block in re.findall(r"```json\s*\n(.*?)\n```", text, re.S):
                with self.subTest(document=document.name, block=block[:60]):
                    json.loads(block)
                    checked += 1
        self.assertGreater(checked, 0)

    def test_documented_default_values_match_packaged_json(self) -> None:
        """Validate literal values and key coverage, not prose semantics."""

        defaults = json.loads(
            (ROOT / "kilin/kilin_defaults.json").read_text(encoding="utf-8")
        )
        document = (DOCS / "package_defaults.md").read_text(encoding="utf-8")
        group = ""
        seen: dict[str, set[str]] = {}
        checked = 0
        for line in document.splitlines():
            heading = re.match(r"^## `([A-Z_]+)`", line)
            if heading:
                group = heading[1]
                seen[group] = set()
            row = re.match(r"^\| `([A-Z_]+)` \| ([^|]+) \|", line)
            if not row:
                continue
            key, cell = row.groups()
            self.assertIn(group, defaults)
            self.assertIn(key, defaults[group])
            self.assertNotIn(key, seen[group])
            seen[group].add(key)
            literal = re.fullmatch(r"`(.+)`", cell.strip())
            if literal:
                expected = defaults[group][key]
                value = (
                    literal[1] if isinstance(expected, str)
                    else json.loads(literal[1])
                )
                with self.subTest(group=group, key=key):
                    self.assertEqual(type(value), type(expected))
                    self.assertEqual(value, expected)
                checked += 1
        self.assertEqual(
            seen, {group: set(values) for group, values in defaults.items()}
        )
        self.assertGreater(checked, 0)


class TestDocumentationOrganization(unittest.TestCase):
    """Guard navigation and integration without pretending to prove prose."""

    def test_related_navigation_is_terminal(self) -> None:
        """Reject substantive sections appended after final cross-links."""

        for document in sorted(DOCS.glob("*.md")):
            body = _without_fences(document.read_text(encoding="utf-8"))
            markers = list(re.finditer(
                r"^(?:Related:|## Related\s*$)", body, re.MULTILINE
            ))
            with self.subTest(document=document.name):
                self.assertLessEqual(len(markers), 1)
                if markers:
                    remainder = body[markers[0].end():]
                    self.assertIsNone(re.search(
                        r"^#{1,6}\s+", remainder, re.MULTILINE
                    ))

    def test_top_level_sections_are_unique(self) -> None:
        """Require distinct second-level topics in each reference page."""

        for document in sorted(DOCS.glob("*.md")):
            body = _without_fences(document.read_text(encoding="utf-8"))
            headings = re.findall(r"^## (.+)$", body, re.MULTILINE)
            with self.subTest(document=document.name):
                self.assertEqual(len(headings), len(set(headings)))

    def test_reference_prose_is_not_pr_relative(self) -> None:
        """Keep evergreen descriptions independent of an unidentified PR."""

        pattern = re.compile(
            r"\bthis\s+(?:PR|refactor|simplification)\b", re.IGNORECASE
        )
        for document in sorted(DOCS.glob("*.md")):
            body = _without_fences(document.read_text(encoding="utf-8"))
            with self.subTest(document=document.name):
                self.assertIsNone(pattern.search(body))

    def test_diagram_labels_are_not_pr_stamps(self) -> None:
        """Inspect SVG text rather than flattened or rendered images."""

        for figure in sorted((DOCS / "figures").glob("*.svg")):
            diagram = ElementTree.parse(figure).getroot()
            text = " ".join(diagram.itertext())
            with self.subTest(figure=figure.name):
                self.assertIsNone(re.search(r"\bPR\s*#?\d+[a-z]*\b", text))

    def test_index_links_to_every_reference_page(self) -> None:
        """Keep a newly added reference discoverable from the docs index."""

        index = DOCS / "README.md"
        body = _without_fences(index.read_text(encoding="utf-8"))
        destinations = {
            (index.parent / unquote(target.split("#", 1)[0])).resolve()
            for target in LINK_PATTERN.findall(body)
            if not target.startswith(("https://", "http://", "mailto:"))
        }
        expected = {
            path.resolve() for path in DOCS.glob("*.md") if path != index
        }
        self.assertFalse(expected - destinations, expected - destinations)

    def test_change_impact_rows_link_contracts_and_tests(self) -> None:
        """Require actionable local references, not unverified test names."""

        text = (DOCS / "README.md").read_text(encoding="utf-8")
        navigator = _section(text, "Change-impact navigator")
        rows = [
            line for line in navigator.splitlines()
            if line.startswith("| ") and "[" in line
        ]
        self.assertTrue(rows)
        for row in rows:
            targets = LINK_PATTERN.findall(row)
            contracts = []
            tests = []
            for target in targets:
                path = (DOCS / unquote(target.split("#", 1)[0])).resolve()
                with self.subTest(row=row, target=target):
                    self.assertTrue(path.exists())
                if path.parent == DOCS.resolve() and path.suffix == ".md":
                    contracts.append(path)
                if path.is_relative_to((ROOT / "tests").resolve()):
                    tests.append(path)
            with self.subTest(row=row):
                self.assertTrue(contracts, "Missing primary contract link")
                self.assertTrue(tests, "Missing regression test link")


if __name__ == "__main__":
    unittest.main()
