"""Exercise the Thinking switch under the effort name in the composer.

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

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_switch_follows_the_setting_and_the_model(self):
        """On or Off as in Settings; disabled for models that cannot."""

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
        for name in ("modelCanThink", "showThinkingSwitch"):
            script += _function(source, name)
        script += r"""
showThinkingSwitch("Qwen/Qwen3-8B-MLX-4bit");
assert.deepEqual(radios.map((radio) => radio.checked), [true, false]);
assert.deepEqual(radios.map((radio) => radio.disabled), [false, false]);
thinking.checked = false;
showThinkingSwitch("Qwen/Qwen3-8B-MLX-4bit");
assert.deepEqual(radios.map((radio) => radio.checked), [false, true]);
// No switch in its template, or not checked: the switch is disabled.
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
