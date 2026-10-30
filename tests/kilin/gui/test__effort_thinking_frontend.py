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
import subprocess
import unittest

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web"


def _function(source: str, name: str) -> str:
    """Return one top-level function's source text."""

    start = source.index(f"function {name}(")
    return source[start : source.index("\n}\n", start) + 3]


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
        # Saved to the open project at once; not while busy or not open.
        start = source.index('byId("thinking").addEventListener("change"')
        changed = source[start : source.index("\n});\n", start)]
        self.assertIn("if (ready && !busy && bridge) {", changed)
        self.assertIn(
            "bridge.saveAnswerSettings(JSON.stringify({ thinking: "
            'byId("thinking").checked }));',
            changed,
        )
        self.assertIn("renderEffortPicker();", changed)

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
// A model that cannot think: the switch is disabled.
showThinkingSwitch("mlx-community/Llama-3.2-3B-Instruct-4bit");
assert.deepEqual(radios.map((radio) => radio.disabled), [true, true]);
// Busy: nothing can change mid-answer.
busy = true;
showThinkingSwitch("Qwen/Qwen3-8B-MLX-4bit");
assert.deepEqual(radios.map((radio) => radio.disabled), [true, true]);
console.log("thinking switch passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("thinking switch passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
