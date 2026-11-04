"""Exercise the Settings tabs.

Check that Settings list their categories as tabs, that every section
belongs to exactly one tab, and that search depth stays with the effort
levels in the Models tab. Run the
production ``showSettingsTab`` function in Node with a stub DOM. Rendering
is covered only by the Qt/WebEngine smoke test and manual checks. KíLín
setup sits above the tabs, as it covers the whole app.

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
_TABS = ["general", "models", "answers", "tools", "project"]


class TestSettingsTabsFrontend(unittest.TestCase):
    """Show Settings one category at a time."""

    def test_every_section_belongs_to_one_tab(self):
        """Setup above five tabs; each section names an existing tab."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        start = page.index('id="models-panel"')
        panel = page[start : page.index("</section>\n    <dialog", start)]
        tabs = re.findall(r'class="settings-tab"[^>]*data-tab="(\w+)"', panel)
        self.assertEqual(tabs, _TABS)
        sections = re.findall(
            r'<(?:section|details)[^>]*class="settings-section[^"]*"[^>]*>',
            panel,
        )
        # KíLín setup plus one section per tab.
        self.assertEqual(len(sections), 6)
        # KíLín setup covers the whole app, so it sits above the tabs.
        self.assertLess(
            panel.index('id="first-run-setup"'),
            panel.index('class="settings-tabs"'),
        )
        self.assertNotIn(
            "data-settings-tab",
            panel[panel.index('id="first-run-setup"') :].split(">", 1)[0],
        )
        for section in sections[1:]:
            tab = re.search(r'data-settings-tab="(\w+)"', section)
            self.assertIsNotNone(tab, section)
            self.assertIn(tab.group(1), _TABS)
        # Messages from Close sit below the tabs, visible on every tab.
        status = panel.index('id="model-download-status"')
        self.assertGreater(status, panel.index('id="settings-project"'))

    def test_search_depth_stays_with_the_models(self):
        """The five levels and their five depths are on one tab."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        render = _function(source, "renderModelsPanel")
        self.assertIn("host.append(depth);", render)
        self.assertNotIn("search-depth-list", source)
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".settings-section.tab-hidden { display: none; }",
                      styles)

    def test_every_tab_uses_one_field_style(self):
        """Labels above full-width boxes of one size, on every tab."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        self.assertIn('<label for="appearance">Appearance</label>', page)
        self.assertNotIn("<h3>Appearance</h3>", page)
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(
            '#models-panel input:not([type="checkbox"]), #models-panel '
            "select { width: 100%; min-height: 36px;",
            styles,
        )
        self.assertIn(".chat-setting { display: grid; gap: 7px; }", styles)

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
const names = ["general", "models", "answers", "project"];
const tabs = names.map((tab) => element({ tab }));
const sections = ["general", "models", "answers", "project"]
  .map((settingsTab) => element({ settingsTab }));
const document = { querySelectorAll: (selector) =>
  selector === ".settings-tab" ? tabs : sections };
let settingsTab = "general";
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
showSettingsTab("general");
assert.deepEqual(sections.map((item) => item.classList.set.has("tab-hidden")),
  [false, true, true, true]);
console.log("settings tabs passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("settings tabs passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
