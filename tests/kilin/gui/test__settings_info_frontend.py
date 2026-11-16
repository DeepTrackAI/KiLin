"""Check that GUI help consumes core-owned setting descriptions.

Frontend tests exercise the actual tooltip creator with a small DOM stub,
while protocol and metadata checks ensure tool labels, project behavior and
ordinary static settings use canonical core keys. No Qt installation or
model execution is required.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kilin.gui.engine import ChatEngine
from kilin.settings_info import get_setting_info, get_settings_info
from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestSettingsInfoFrontend(unittest.TestCase):
    """Verify the GUI uses an independent metadata source."""

    def test_static_labels_reference_existing_core_setting_ids(self):
        page = (_WEB / "index.html").read_text(encoding="utf-8")
        catalog = get_settings_info()["settings"]
        keys = re.findall(r'data-settings-key="([^"]+)"', page)
        self.assertGreater(len(keys), 20)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertIn(
            'for="thinking" data-settings-key="project.profile.thinking"',
            page,
        )
        # Every marked field must refer to a known metadata identifier.
        self.assertFalse(set(keys) - set(catalog))

    def test_dynamic_fields_are_scoped_to_core_and_tool_owners(self):
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            'label.dataset.settingsKey = `tools.${toolName}.${key}`;',
            extract_function(source, "toolSettingField"),
        )
        self.assertIn(
            '`project_behavior.${scope === "global" ? '
            '"global" : "intents"}.${path}`',
            extract_function(source, "createBehaviorEditor"),
        )
        self.assertIn(
            'heading.dataset.settingsKey = '
            '`model_settings.${groupKeys[role]}`;',
            extract_function(source, "modelGroupHeader"),
        )
        self.assertIn(
            '"model_settings.effort_levels.name"',
            extract_function(source, "renderModelsPanel"),
        )
        self.assertIn(
            '"model_settings.effort_levels.model"',
            extract_function(source, "renderModelsPanel"),
        )
        self.assertIn('bridge.requestSettingsInfo();', source)
        self.assertIn('data.event === "settings_info"', source)
        self.assertIn('data.event === "settings_info_failed"', source)

    def test_protocol_metadata_is_independent_of_an_open_project(self):
        events = []
        with TemporaryDirectory() as folder:
            manager = ChatEngine(Path(folder), events.append)
            self.addCleanup(manager.close)
            manager.settings_info()
            self.assertEqual(events[-1]["event"], "settings_info")
            metadata = events[-1]["catalog"]
            self.assertIn("project.profile.temperature", metadata["settings"])
            self.assertIn("tools.reference_check.max_online",
                          metadata["settings"])
            with patch("kilin.gui.engine.get_settings_info",
                       side_effect=ValueError("Invalid metadata")):
                manager.settings_info()
            self.assertEqual(events[-1]["event"], "settings_info_failed")
            self.assertIn("Invalid metadata", events[-1]["message"])
            self.assertIsNone(manager.session)

    def test_help_icons_are_small_and_popups_escape_dialog_clipping(self):
        """Keep the icon circular and the visible tooltip viewport-fixed."""

        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        button = re.search(
            r"\.setting-help-trigger\s*\{([^}]+)\}", styles,
        )
        popup = re.search(
            r"\.setting-help-popup\s*\{([^}]+)\}", styles,
        )
        self.assertIsNotNone(button)
        self.assertIsNotNone(popup)
        rules = button.group(1)
        self.assertRegex(rules, r"\bwidth:\s*15px;")
        self.assertRegex(rules, r"\bheight:\s*15px;")
        self.assertRegex(rules, r"\bmin-height:\s*15px;")
        self.assertRegex(rules, r"\bpadding:\s*0;")
        self.assertRegex(rules, r"\bborder-radius:\s*50%;")
        self.assertRegex(popup.group(1), r"\bposition:\s*fixed;")
        self.assertIn(
            ".setting-help-popup.is-visible { display: block; }", styles,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_help_is_populated_from_metadata_without_changing_inputs(self):
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        core = get_setting_info("project.profile.temperature")
        tool = get_setting_info("tools.reference_check.max_online")
        data = json.dumps({"format": "settings-info-v1", "settings": {
            "project.profile.temperature": core,
            "tools.reference_check.max_online": tool,
        }})
        script = r'''
const assert = require("node:assert/strict");
class FakeNode {
  constructor(tag, text = "") {
    this.tagName = tag.toUpperCase();
    this.nodeType = 1;
    this.textContent = text;
    this.dataset = {};
    this.childNodes = [];
    this.listeners = {};
    this.attrs = {};
    this.className = "";
    this.parentNode = null;
    this.style = {};
    this.isConnected = true;
    this.opened = new Set();
    this.classList = {
      contains: (value) => this.opened.has(value),
      add: (value) => this.opened.add(value),
      remove: (value) => this.opened.delete(value),
      toggle: (value, state) => {
        if (state === undefined) state = !this.opened.has(value);
        if (state) this.opened.add(value);
        else this.opened.delete(value);
      },
    };
  }
  append(...children) {
    for (const child of children) {
      if (child.parentNode) {
        child.parentNode.childNodes = child.parentNode.childNodes
          .filter((item) => item !== child);
      }
      child.parentNode = this;
      this.childNodes.push(child);
    }
  }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  getBoundingClientRect() {
    return this.className === "setting-help-popup"
      ? {left: 0, top: 0, right: 240, bottom: 60,
        width: 240, height: 60}
      : {left: 320, top: 180, right: 335, bottom: 195,
        width: 15, height: 15};
  }
  closest() { return null; }
  matches() { return false; }
  blur() {}
  querySelector(selector) {
    if (selector === ".setting-help") {
      return this.childNodes.find((item) => item.className === "setting-help")
        || this.childNodes.find((item) => item.querySelector?.(selector));
    }
    return null;
  }
  querySelectorAll() { return []; }
  replaceChild(replacement, previous) {
    const index = this.childNodes.indexOf(previous);
    assert.notEqual(index, -1);
    this.childNodes[index] = replacement;
    replacement.parentNode = this;
  }
}
function textNode(text) { return {nodeType: 3, textContent: text}; }
const first = new FakeNode("label");
first.dataset.settingsKey = "project.profile.temperature";
const input = new FakeNode("input");
input.value = "0.7";
first.childNodes = [textNode("Original"), input];
const second = new FakeNode("label");
second.dataset.settingsKey = "tools.reference_check.max_online";
second.childNodes = [new FakeNode("input"), textNode("Old")];
const labels = [first, second];
const document = {
  body: new FakeNode("body"),
  activeElement: null,
  createElement: (name) => new FakeNode(name),
  createTextNode: textNode,
  querySelectorAll: (selector) => selector === ".setting-help.is-open"
    ? labels.flatMap((label) => label.childNodes[0].childNodes)
      .filter((child) => child.className === "setting-help"
        && child.classList.contains("is-open")) : labels,
};
let settingsHelpSequence = 0;
let activeSettingsHelp = null;
const window = {innerWidth: 460, innerHeight: 240};
let settingsCatalog = '''
        script += data + ";\n"
        for function_name in (
            "dismissSettingsHelp", "positionSettingsHelp",
            "showSettingsHelp", "applySettingsHelp",
        ):
            script += extract_function(source, function_name)
        script += r'''
applySettingsHelp({querySelectorAll: () => labels});
assert.equal(input.value, "0.7");
assert.equal(first.childNodes[0].childNodes[0].textContent,
  settingsCatalog.settings["project.profile.temperature"].title);
const marker = first.childNodes[0].childNodes[1];
assert.equal(marker.childNodes[1].textContent,
  settingsCatalog.settings["project.profile.temperature"].summary);
const button = marker.childNodes[0];
assert.equal(button.attrs["aria-label"], "Help: Temperature");
const used = {prevented: false, stopped: false,
  preventDefault() {this.prevented = true;},
  stopPropagation() {this.stopped = true;}};
button.listeners.click(used);
assert.equal(used.prevented, true);
assert.equal(used.stopped, true);
assert.equal(marker.classList.contains("is-open"), true);
assert.equal(button.attrs["aria-expanded"], "true");
assert.equal(document.body.childNodes.length, 1);
assert.equal(document.body.childNodes[0], activeSettingsHelp.tooltip);
assert.equal(activeSettingsHelp.pinned, true);
assert.equal(activeSettingsHelp.tooltip.style.left, "208px");
assert.equal(activeSettingsHelp.tooltip.style.top, "114px");
button.listeners.click(used);
assert.equal(marker.classList.contains("is-open"), false);
assert.equal(document.body.childNodes.length, 0);
assert.equal(marker.childNodes.length, 2);
button.listeners.pointerenter();
assert.equal(activeSettingsHelp.pinned, false);
button.listeners.pointerleave();
assert.equal(activeSettingsHelp, null);
button.listeners.click(used);
button.listeners.keydown({...used, key: "Escape"});
assert.equal(marker.classList.contains("is-open"), false);
assert.equal(document.body.childNodes.length, 0);
applySettingsHelp({querySelectorAll: () => labels});
assert.equal(first.childNodes.length, 2);
settingsCatalog = null;
const third = new FakeNode("label");
third.dataset.settingsKey = "project.profile.temperature";
third.childNodes = [textNode("Unchanged")];
applySettingsHelp({querySelectorAll: () => [third]});
assert.equal(third.childNodes[0].textContent, "Unchanged");
console.log("metadata help passed");
'''
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("metadata help passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
