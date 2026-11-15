"""Exercise the Settings tabs.

Check that Settings list their categories as tabs, that every section
belongs to exactly one tab, and that search depth, shown in the Answers
tab, is still drawn with the effort levels. Run the production
``showSettingsTab`` function in Node with a stub DOM. Rendering is
covered only by the Qt/WebEngine smoke test and manual checks. KíLín
setup holds the computation backend, Appearance and the projects folder
above the tabs, since they cover the whole app; folded, it is one row.
Below it, the Project region holds seven tabs: General (the name),
Indexing, Memory, Models, Answers, Intents and Tools. On/off settings
are switches.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import re
import shutil
import unittest

from tests.kilin.gui._frontend_support import (
    extract_function as _function,
    run_node,
)

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"
_TABS = [
    "general", "indexing", "memory", "models", "answers",
    "intents", "tools",
]


class TestSettingsTabsFrontend(unittest.TestCase):
    """Show Settings one category at a time."""

    def test_every_section_belongs_to_one_tab(self):
        """App-wide parts above seven tabs; each section has one tab."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        start = page.index('id="models-panel"')
        panel = page[start : page.index("</section>\n    <dialog", start)]
        tabs = re.findall(r'class="settings-tab"[^>]*data-tab="(\w+)"', panel)
        self.assertEqual(tabs, _TABS)
        sections = re.findall(
            r'<(?:section|details)[^>]*class="settings-section[^"]*"[^>]*>',
            panel,
        )
        # The app-wide settings, then one section per tab.
        self.assertEqual(len(sections), 8)
        self.assertNotIn("data-settings-tab", sections[0])
        self.assertIn('id="settings-app"', sections[0])
        # These cover the whole app, so they sit above the tabs, inside
        # KíLín setup, which folds them away.
        tabs_at = panel.index('class="settings-tabs"')
        app = panel[panel.index('id="settings-app"') : tabs_at]
        start = app.index('id="first-run-setup"')
        setup = app[start : app.index("</details>")]
        for part in (
            '<label for="backend" data-settings-key="interface.backend">'
            'Computation backend</label>',
            '<label for="appearance" data-settings-key="interface.appearance">'
            'Appearance</label>',
            '<label for="workspace-path" '
            'data-settings-key="interface.workspace_folder">'
            'Projects folder</label>',
        ):
            self.assertIn(part, setup)
        # The seven tabs share the full width.
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".settings-tab { position: relative; flex: 1 1 0;",
                      styles)
        for section in sections[1:]:
            tab = re.search(r'data-settings-tab="(\w+)"', section)
            self.assertIsNotNone(tab, section)
            self.assertIn(tab.group(1), _TABS)
        # Messages from Close sit below the tabs, visible on every tab.
        status = panel.index('id="model-download-status"')
        self.assertGreater(status, panel.index('id="settings-indexing"'))
        # The project's tabs are in one region, after setup.
        region = panel.index('id="settings-project-region"')
        self.assertLess(panel.index("</details>"), region)
        self.assertLess(region, tabs_at)

    def test_memory_settings_have_their_own_tab(self):
        """Memory policy is separate from source indexing controls."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        indexing_start = page.index('id="settings-indexing"')
        memory_start = page.index('id="settings-memory"')
        after_memory = page.index('<!-- Below the tabs', memory_start)
        indexing = page[indexing_start:memory_start]
        memory = page[memory_start:after_memory]
        self.assertIn('data-settings-tab="memory"', memory)
        self.assertIn('id="memory-enabled"', memory)
        self.assertIn('id="memory-index-status"', memory)
        self.assertIn('id="sync-memory-index"', memory)
        self.assertNotIn('id="memory-enabled"', indexing)
        self.assertNotIn('id="sync-memory-index"', indexing)
        self.assertIn('id="index-capabilities-title"', indexing)
        self.assertEqual(page.count('id="memory-enabled"'), 1)

    def test_search_depth_sits_in_answers(self):
        """Search depth shows in Answers but applies with the models."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        start = page.index('data-settings-tab="answers"')
        answers = page[start : page.index("</section>", start)]
        self.assertIn('<div id="search-depth"></div>', answers)
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        render = _function(source, "renderModelsPanel")
        self.assertIn("depthHost.append(depth);", render)
        self.assertNotIn("search-depth-list", source)
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".settings-section.tab-hidden { display: none; }",
                      styles)
        # The columns behind Settings cannot be resized while it is open.
        self.assertIn(
            "body:has(#models-panel:not(.hidden)) .panel-resizer "
            "{ display: none; }",
            styles,
        )

    def test_every_tab_uses_one_field_style(self):
        """Labels above full-width boxes of one size, on every tab."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '<label for="appearance" data-settings-key="interface.appearance">'
            'Appearance</label>', page,
        )
        self.assertNotIn("<h3>Appearance</h3>", page)
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(
            '#models-panel input:not([type="checkbox"]), #models-panel '
            "select { width: 100%; min-height: 36px;",
            styles,
        )
        self.assertIn(".chat-setting { display: grid; gap: 7px; }", styles)
        # On/off settings are switches in the send button's orange.
        self.assertIn(
            '.check-row input[type="checkbox"]:checked '
            "{ background: var(--accent); }",
            styles,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_one_tab_shows_at_a_time(self):
        """showSettingsTab marks one tab and hides the other sections."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
function element(data) {
  return {
    dataset: data, attributes: {}, tabIndex: 0,
    classList: { set: new Set(), toggle(name, on) {
      if (on) this.set.add(name); else this.set.delete(name);
    } },
    setAttribute(name, value) { this.attributes[name] = value; },
  };
}
const names = ["project", "models", "answers", "memory"];
const tabs = names.map((tab) => element({ tab }));
const sections = names.map((settingsTab) => element({ settingsTab }));
const document = { querySelectorAll: (selector) =>
  selector === ".settings-tab" ? tabs : sections };
let settingsTab = "project";
let ready = true, openProjectPath = "/workspace";
const statusCalls = [];
const requestProjectBehavior = () => statusCalls.push("policy");
const bridge = {requestMemoryStatus: (path) =>
  statusCalls.push(`status:${path}`)};
"""
        script += _function(source, "showSettingsTab")
        script += r"""
showSettingsTab("answers");
assert.equal(settingsTab, "answers");
assert.deepEqual(tabs.map((tab) => tab.attributes["aria-selected"]),
  ["false", "false", "true", "false"]);
assert.deepEqual(tabs.map((tab) => tab.tabIndex), [-1, -1, 0, -1]);
assert.deepEqual(sections.map((item) => item.classList.set.has("tab-hidden")),
  [true, true, false, true]);
showSettingsTab("project");
assert.deepEqual(sections.map((item) => item.classList.set.has("tab-hidden")),
  [false, true, true, true]);
showSettingsTab("memory");
assert.equal(settingsTab, "memory");
assert.deepEqual(statusCalls, ["policy", "status:/workspace"]);
assert.deepEqual(sections.map((item) =>
  item.classList.set.has("tab-hidden")), [true, true, true, false]);
console.log("settings tabs passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("settings tabs passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
