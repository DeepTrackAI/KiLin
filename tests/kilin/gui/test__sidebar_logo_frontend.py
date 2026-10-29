"""Check the KíLín logo and the Projects and Conversations sections.

Read the production ``index.html``, ``app.js`` and ``styles.css``: the logo
sits at the top of the conversations panel, asks before closing the project,
is disabled while KíLín is busy, and replaces the old New project button;
each section folds and has ⋯ (sort) and ＋ (new) on hover. Run the ordering
functions and ``renderSidebarProjects`` in Node with a small DOM stand-in.
Rendering is covered only by the Qt/WebEngine smoke test and manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import subprocess
import unittest

_ROOT = Path(__file__).resolve().parents[3]
_WEB = _ROOT / "kilin/gui/web"


def _function(source: str, name: str) -> str:
    """Return one top-level function's source text."""

    start = source.index(f"function {name}(")
    return source[start : source.index("\n}\n", start) + 3]


class TestSidebarLogoFrontend(unittest.TestCase):
    """Go back to the start screen, or switch project, from the panel."""

    def test_logo_tops_the_panel_and_replaces_new_project(self):
        """First in the panel; the bottom New project button is gone."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        start = page.index('<aside id="conversations-panel"')
        panel = page[start : page.index("</aside>", start)]
        first = panel.index("<button")
        self.assertEqual(panel.index('<button id="home-logo"'), first)
        self.assertNotIn("change-project", page)
        self.assertIn('<dialog id="leave-project"', page)

    def test_logo_and_new_project_ask_then_return_to_launch(self):
        """Busy clicks are ignored; Close project returns to launch."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        start = source.index('for (const id of ["home-logo", "new-project"])')
        click = source[start : source.index("});", start)]
        self.assertIn("if (busy) return;", click)
        self.assertIn('byId("leave-project").showModal();', click)
        start = source.index('byId("leave-project-confirm").addEventListener')
        confirm = source[start : source.index("});", start)]
        self.assertIn("returnToLaunch();", confirm)
        self.assertIn(
            "bridge.closeProject();", _function(source, "returnToLaunch")
        )
        busy = _function(source, "setBusy")
        self.assertIn('byId("home-logo").disabled = value;', busy)
        self.assertIn('byId("new-project").disabled = value;', busy)
        self.assertNotIn("change-project", source)

    def test_logo_is_the_qilin(self):
        """Gold qilin on light pages, blue-black one in dark mode."""

        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        start = styles.index(".sidebar-logo-mark {")
        rule = styles[start : styles.index("}", start)]
        self.assertIn("var(--logo-image)", rule)
        self.assertEqual(
            styles.count('--logo-image: url("../icons/kilin-qilin.png");'), 1
        )
        # Both dark blocks: the chosen dark look and the system one.
        self.assertEqual(
            styles.count(
                '--logo-image: url("../icons/kilin-qilin-dark.png");'
            ),
            2,
        )
        for name in ("kilin-qilin.png", "kilin-qilin-dark.png"):
            self.assertTrue((_ROOT / "kilin/gui/icons" / name).is_file())
        # The source drawing is CC BY-SA 4.0, so the credit ships with it.
        credit = (_ROOT / "kilin/gui/icons/README.md").read_text(
            encoding="utf-8"
        )
        for text in ("RootOfAllLight", "CC BY-SA 4.0", "kilin-qilin-dark"):
            self.assertIn(text, credit)

    def test_sections_have_fold_sort_and_new(self):
        """Logo, Projects, then Conversations, each with ⋯ and ＋."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        order = [
            page.index(marker)
            for marker in (
                'id="home-logo"',
                'data-section="projects"',
                'id="sort-projects"',
                'id="new-project"',
                '<nav id="project-list"',
                'data-section="conversations"',
                'id="sort-conversations"',
                'id="new-chat"',
                '<nav id="conversation-list"',
            )
        ]
        self.assertEqual(order, sorted(order))
        self.assertEqual(page.count('class="sidebar-section-toggle"'), 2)
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        start = styles.index(".sidebar-section-action {")
        rule = styles[start : styles.index("}", start)]
        self.assertIn("opacity: 0", rule)
        self.assertIn(
            ".sidebar-section-head:hover .sidebar-section-action", styles
        )
        self.assertIn(
            ".sidebar-section.collapsed nav { display: none; }", styles
        )

    def test_conversations_sort_rename_and_hover_delete(self):
        """Sorted and draggable; right-click renames; × on hover only."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        render = _function(source, "renderConversations")
        self.assertIn("sidebarConversationOrder(", render)
        self.assertIn("makeReorderable(row, id, ids", render)
        self.assertIn(
            '["Rename", () => renameInSidebar(row, button, item)]', render
        )
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertNotIn(
            ":has(.conversation-item.active) .conversation-delete", styles
        )

    def test_header_shows_only_the_chat(self):
        """Only the chat, centred between the toggles; no project heading."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("OPEN PROJECT", page)
        self.assertNotIn('id="project-name"', page)
        self.assertIn('<h1 id="conversation-name"', page)
        self.assertIn('<strong id="topbar-chat-name"></strong>', page)
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(
            ".chat-open #open-models, .chat-open #connection "
            "{ display: none; }",
            styles,
        )
        # One panel toggle on each edge, the chat name in the middle.
        self.assertIn("#toggle-conversations { margin-right: auto; }", styles)
        self.assertIn("#toggle-workspace { margin-left: auto; }", styles)
        self.assertIn(
            ".chat-open .topbar-project { position: absolute;", styles
        )
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        summary = _function(source, "updateChatSummary")
        self.assertIn('byId("conversation-name").textContent', summary)
        start = source.index('data.event === "opened"')
        opened = source[start : source.index("} else if (data.event", start)]
        self.assertNotIn("settingsAfterOpen", opened)
        rename = _function(source, "renameProjectInSidebar")
        self.assertIn("bridge.renameProject(name);", rename)

    def test_projects_list_follows_the_open_project(self):
        """Refreshed when a project opens; disabled while busy."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        start = source.index('data.event === "opened"')
        opened = source[start : source.index("} else if (data.event", start)]
        self.assertIn("openProjectPath = data.project;", opened)
        # The chat order is kept per project, so set it before drawing.
        self.assertLess(
            opened.index("openProjectPath = data.project;"),
            opened.index("renderConversations(data);"),
        )
        self.assertIn('bridge.listSavedProjects("");', opened)
        start = source.index('data.event === "projects"')
        listed = source[start : source.index("} else if (data.event", start)]
        self.assertIn(
            "renderSidebarProjects(savedProjects, openProjectPath);", listed
        )
        self.assertIn(".project-item", _function(source, "setBusy"))
        self.assertIn(
            'openProjectPath = "";', _function(source, "returnToLaunch")
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_orders(self):
        """Last updated first; manual order puts new items on top."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = 'const assert = require("node:assert/strict");\n'
        for name in (
            "manualOrder",
            "moveInOrder",
            "sidebarProjectOrder",
            "sidebarConversationOrder",
        ):
            script += _function(source, name)
        script += r"""
assert.deepEqual(moveInOrder(["a", "b", "c"], "a", "c"), ["b", "c", "a"]);
assert.deepEqual(moveInOrder(["a", "b", "c"], "c", "a"), ["c", "a", "b"]);
const chats = [
  { conversation_id: "old", updated_at_utc: "2026-10-01T10:00:00Z" },
  { conversation_id: "new", updated_at_utc: "2026-10-29T10:00:00Z" },
  { conversation_id: "mid", updated_at_utc: "2026-10-20T10:00:00Z" },
];
const ids = (items) => items.map((item) => item.conversation_id);
assert.deepEqual(ids(sidebarConversationOrder(chats)), ["new", "mid", "old"]);
assert.deepEqual(
  ids(sidebarConversationOrder(chats, "manual", ["old", "new"])),
  ["mid", "old", "new"],
);
const projects = [{ project: "/w/a" }, { project: "/w/b" }];
const paths = (items) => items.map((item) => item.project);
// The open project keeps its place; one outside the folder comes first.
assert.deepEqual(paths(sidebarProjectOrder(projects, "/w/b")),
  ["/w/a", "/w/b"]);
assert.deepEqual(paths(sidebarProjectOrder(projects, "/x/c", "code")),
  ["/x/c", "/w/a", "/w/b"]);
assert.deepEqual(
  paths(sidebarProjectOrder(projects, "/w/a", "", "manual", ["/w/b"])),
  ["/w/a", "/w/b"],
);
assert.deepEqual(
  paths(sidebarProjectOrder(projects, "", "", "manual", ["/w/b", "/w/a"])),
  ["/w/b", "/w/a"],
);
console.log("orders passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("orders passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_projects_list_highlights_and_switches(self):
        """Open project marked; five shown, with type; More shows the rest."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        start = source.index("const SIDEBAR_PROJECTS")
        script = r"""
const assert = require("node:assert/strict");
function element() {
  return {
    children: [], attributes: {}, listeners: {}, className: "",
    classList: { set: new Set(), toggle(name, on) {
      if (on) this.set.add(name); else this.set.delete(name);
    } },
    replaceChildren() { this.children = []; },
    append(...children) { this.children.push(...children); },
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(type, listener) { this.listeners[type] = listener; },
  };
}
const list = element();
const document = { createElement: () => element() };
const byId = () => list;
const localStorage = { getItem: () => null, setItem: () => {} };
let busy = false;
const activeProjectTypeId = "code";
const projectTypes = { literature: { label: "Literature" } };
const calls = [];
const openModels = () => calls.push(["settings"]);
const showWorkspaceMenu = () => {};
const bridge = { resumeProject: (path) => calls.push(["resume", path]) };
function setBusy(value, message) {
  busy = value;
  calls.push(["busy", message]);
}
"""
        script += source[start : source.index("\n", start) + 1]
        script += "let sidebarProjectsExpanded = false;\n"
        for name in (
            "sidebarPref",
            "manualOrder",
            "sidebarProjectOrder",
            "renderSidebarProjects",
        ):
            script += _function(source, name)
        script += r"""
// A project row holds its button (and, if open, a settings icon).
const buttonOf = (index) => list.children[index].children[0];
const names = () => list.children.map((item) =>
  item.className === "project-row"
    ? item.children[0].children[0].textContent
    : item.textContent);
const projects = ["a", "b", "c", "d", "e", "f", "g"].map((name) => (
  { project: `/w/${name}`, name, type_id: "literature" }
));
// Recent use order; five shown, then More.
renderSidebarProjects(projects, "/w/b");
assert.deepEqual(names(), ["a", "b", "c", "d", "e", "More (2)"]);
assert.equal(buttonOf(1).children[1].textContent, "Literature");
assert.ok(buttonOf(1).classList.set.has("active"));
assert.equal(buttonOf(1).attributes["aria-current"], "true");
list.children[5].listeners.click();
assert.deepEqual(names(), ["a", "b", "c", "d", "e", "f", "g", "Fewer"]);
list.children[7].listeners.click();
assert.equal(list.children.length, 6);
// Clicking the open project does nothing; another one is resumed.
buttonOf(1).listeners.click();
assert.deepEqual(calls, []);
buttonOf(0).listeners.click();
assert.deepEqual(calls, [["busy", "Opening a…"], ["resume", "/w/a"]]);
// Only the open project has a settings icon; it opens Settings.
busy = false;
calls.length = 0;
assert.deepEqual(list.children.map((row) => row.children.length),
  [1, 2, 1, 1, 1, 0]);
list.children[1].children[1].listeners.click();
assert.deepEqual(calls, [["settings"]]);
// A project outside the listed folder is still shown, first.
renderSidebarProjects(projects.slice(0, 2), "/elsewhere/paper");
assert.deepEqual(names(), ["paper", "a", "b"]);
assert.equal(buttonOf(0).children[1].textContent, "code");
console.log("projects list passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("projects list passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
