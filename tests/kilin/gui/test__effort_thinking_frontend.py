"""Exercise Thinking and Temperature under the composer's effort name.

"Lagom ›" opens an On/Off switch that is the same setting as "Enable model
thinking, if supported" in Settings → Answers. Run the production
``showThinkingSwitch`` in Node with stubs, and check the markup and the
wiring. Rendering is covered only by the Qt/WebEngine smoke test and manual
checks.

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


class TestEffortThinkingFrontend(unittest.TestCase):
    """Turn thinking on or off from the effort menu."""

    def test_effort_name_opens_the_thinking_switch(self):
        """A button with a chevron; the switch starts folded."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        start = page.index('<div id="effort-menu"')
        menu = page[start : page.index('<div class="usage-picker">', start)]
        self.assertIn(
            '<button id="effort-name" class="effort-title" type="button" '
            'aria-expanded="false" aria-controls="effort-thinking">',
            menu,
        )
        self.assertIn('class="effort-chevron" aria-hidden="true">›', menu)
        self.assertIn(
            '<div id="effort-thinking" '
            'class="effort-section effort-thinking hidden">',
            menu,
        )
        for value in ("on", "off"):
            self.assertIn(
                f'<input type="radio" name="effort-thinking" '
                f'value="{value}">',
                menu,
            )
        # The switch sets the Answers checkbox, which is what is saved.
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            'byId("thinking").checked = input.value === "on";', source
        )
        # Thinking uses the same serialized project save as the numbers.
        start = source.index('byId("thinking").addEventListener("change"')
        changed = source[start : source.index("\n});\n", start)]
        self.assertIn("if (ready && busy) {", changed)
        self.assertIn("saveAnswerNumbers();", changed)
        self.assertIn("renderEffortPicker();", changed)
        self.assertIn('request_id: request.id', source)

    def test_temperature_is_shown_next_to_thinking(self):
        """Share the Answers temperature rather than create a new setting."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        start = page.index('<div id="effort-thinking"')
        menu = page[start:page.index('<div class="effort-section">', start)]
        self.assertIn(
            'for="effort-temperature"><span>Temperature</span>', menu
        )
        self.assertIn('<span>Thinking</span>', menu)
        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn('.effort-heading span { color: var(--muted);', styles)
        self.assertIn('id="effort-temperature" type="number"', menu)
        self.assertIn('min="0" step="0.1"', menu)
        # The menu is inside the question form; numeric validation must not
        # block a question while the user edits this separate setting.
        self.assertIn('form="effort-settings"', menu)
        self.assertNotIn('id="effort-settings"', page)
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            'byId("effort-temperature").value = byId("temperature").value;',
            _function(source, "renderEffortPicker"),
        )
        self.assertIn(
            'byId("temperature").dispatchEvent(new Event("change"));',
            source,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_temperature_uses_existing_validation_and_save_queue(self):
        """Send temperature changes through the confirmed Answers pipeline."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        begin = source.index(
            'byId("effort-temperature").addEventListener("change"'
        )
        handler = source[begin:source.index("\n});\n", begin) + 4]
        script = r"""
const assert = require("node:assert/strict");
const fields = {};
const byId = (id) => (fields[id] ||= {
  _value: "", checked: false, textContent: "",
  set value(raw) {this._value = String(raw);},
  get value() {return this._value;},
  classList: {add: () => {}, remove: () => {}},
});
let busy = false, ready = true, openProjectPath = "/project";
let savedAnswerNumbers = {temperature: 0.2, max_new_tokens: 1024};
let savedAnswerThinking = false;
let answerSettingsSave = {nextId: 0, pending: null,
  queued: null, uncertain: false};
const calls = [];
const bridge = {saveAnswerSettings: (raw) => calls.push(JSON.parse(raw))};
const updateSendButton = () => {};
const keepNewProjectValue = () => {};
const renderEffortPicker = () => {};
byId("temperature").value = "0.2";
byId("max-new-tokens").value = "1024";
byId("thinking").checked = false;
byId("temperature").dispatchEvent = () => saveAnswerNumbers();
let changeHandler = null;
byId("effort-temperature").addEventListener = (event, cb) => {
  assert.equal(event, "change");
  changeHandler = cb;
};
"""
        for name in (
            "confirmedAnswerSettings", "sameAnswerSettings",
            "queueAnswerSettings", "saveAnswerNumbers",
        ):
            script += _function(source, name)
        script += handler + r"""
byId("effort-temperature").value = "0.7";
changeHandler();
assert.equal(byId("temperature").value, "0.7");
assert.equal(calls.length, 1);
assert.deepEqual(calls[0].settings, {
  temperature: 0.7, thinking: false, max_new_tokens: 1024,
});
// Invalid edits revert both controls to the latest pending value.
byId("effort-temperature").value = "-1";
changeHandler();
assert.equal(byId("temperature").value, "0.7");
assert.equal(byId("effort-temperature").value, "0.7");
assert.equal(calls.length, 1);
// Edits made while busy also revert without starting another save.
busy = true;
byId("effort-temperature").value = "0.9";
changeHandler();
assert.equal(byId("effort-temperature").value, "0.7");
assert.equal(calls.length, 1);
console.log("effort temperature confirmed save passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("confirmed save passed", result.stdout)

    def test_settings_describes_a_template_request(self):
        """Do not imply that the model's reasoning was verified."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        self.assertIn("Request thinking via chat template", page)
        self.assertNotIn("Enable model thinking, if supported", page)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_switch_follows_the_setting_and_the_model(self):
        """On or Off as in Settings; disabled if no switch is detected."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
let busy = false;
const thinking = { checked: true };
const byId = () => thinking;
const radios = ["on", "off"].map((value) => ({
  value, checked: false, disabled: false,
}));
var document = { querySelectorAll: () => radios };
// As the engine verified from the downloaded chat templates.
const thinkingSwitches = {
  "Qwen/Qwen3-8B-MLX-4bit": true, "Qwen/Qwen3-1.7B-MLX-4bit": false,
};
"""
        for name in ("modelHasThinkingSwitch", "showThinkingSwitch"):
            script += _function(source, name)
        script += r"""
showThinkingSwitch("Qwen/Qwen3-8B-MLX-4bit");
assert.deepEqual(radios.map((radio) => radio.checked), [true, false]);
assert.deepEqual(radios.map((radio) => radio.disabled), [false, false]);
thinking.checked = false;
showThinkingSwitch("Qwen/Qwen3-8B-MLX-4bit");
assert.deepEqual(radios.map((radio) => radio.checked), [false, true]);
// No detected switch, or unknown: the switch is disabled.
for (const model of ["Qwen/Qwen3-1.7B-MLX-4bit", "other/unknown"]) {
  showThinkingSwitch(model);
  assert.deepEqual(radios.map((radio) => radio.disabled), [true, true]);
}
// Busy: nothing can change mid-answer.
busy = true;
showThinkingSwitch("Qwen/Qwen3-8B-MLX-4bit");
assert.deepEqual(radios.map((radio) => radio.disabled), [true, true]);
console.log("thinking switch passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("thinking switch passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
