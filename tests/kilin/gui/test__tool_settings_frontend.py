"""Exercise the Settings Tools tab built from each tool's defaults.

Run the production field builders in Node with stub elements, and check
that the tab asks for the open project's tools when shown. Rendering is
covered only by manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import (
    extract_function as _function,
    run_node,
)

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


class TestToolSettingsFrontend(unittest.TestCase):
    """List the project's tools; edit their settings by value type."""

    def test_tools_tab_asks_for_the_open_projects_tools(self):
        page = (_WEB / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-tab="tools">Tools</button>', page)
        self.assertIn('data-settings-tab="tools"', page)
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            'if (name === "tools") requestToolSettings();',
            _function(source, "showSettingsTab"),
        )
        self.assertIn(
            "bridge.requestToolSettings();",
            _function(source, "requestToolSettings"),
        )
        self.assertIn(
            'data.event === "tool_settings"', source
        )

    def test_changes_save_at_once_and_defaults_can_be_restored(self):
        """No Save button: a change saves; Restore defaults resets."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        render = _function(source, "renderToolSettings")
        change = render.index('fields.addEventListener("change", () => {')
        self.assertLess(
            change,
            render.index(
                "bridge.saveToolSettings(tool.name, "
                "JSON.stringify(toolSettingValues(fields)));"
            ),
        )
        self.assertNotIn('save.textContent = "Save";', render)
        self.assertIn('restore.textContent = "Restore defaults";', render)
        self.assertIn(
            "bridge.saveToolSettings(tool.name, "
            "JSON.stringify(tool.type_settings));",
            render,
        )
        # Only enabled when something differs from the defaults.
        self.assertIn(
            "restore.disabled = JSON.stringify(tool.settings) === "
            "JSON.stringify(tool.type_settings);",
            render,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_fields_follow_each_values_type_and_read_back(self):
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
function element(tag) {
  return {
    tag, children: [], dataset: {}, className: "", type: "", value: "",
    checked: false,
    append(...items) { this.children.push(...items); },
    querySelectorAll() {
      const found = [];
      const walk = (node) => {
        for (const child of node.children || []) {
          if (child.tag === "input") found.push(child);
          walk(child);
        }
      };
      walk(this);
      return found;
    },
  };
}
const document = { createElement: (tag) => element(tag) };
"""
        for name in ("settingLabel", "toolSettingField", "toolSettingValues"):
            script += _function(source, name)
        script += r"""
assert.equal(settingLabel("max_model_calls"), "Max model calls");
const fields = element("div");
const settings = {
  part_tokens: 6000, share: 0.5, verify: true,
  confirmation: "when_large", categories: ["yes", "no"],
};
for (const [key, value] of Object.entries(settings)) {
  fields.append(toolSettingField(key, value));
}
const inputs = fields.querySelectorAll();
assert.deepEqual(
  inputs.map((input) => input.type),
  ["number", "number", "checkbox", "text", "text"],
);
assert.equal(inputs[0].step, "1");
assert.equal(inputs[1].step, "any");
assert.equal(inputs[4].value, "yes, no");
// Unchanged fields read back as the same values and types.
assert.deepEqual(toolSettingValues(fields), settings);
inputs[0].value = "8000";
inputs[2].checked = false;
inputs[4].value = "yes,  maybe , ";
assert.deepEqual(toolSettingValues(fields), {
  ...settings, part_tokens: 8000, verify: false, categories: ["yes", "maybe"],
});
console.log("tool fields passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tool fields passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
