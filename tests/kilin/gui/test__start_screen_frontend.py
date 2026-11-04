"""Exercise the start screen: answer settings, thinking, and project lists.

Run the production page functions in Node with stubs: a new project is
created with the Answers settings chosen on the start screen, and a later
settings event does not reset them. Check that the home page and the left
panel show five projects and scroll for the rest. Rendering is covered only
by the Qt/WebEngine smoke test and manual checks.

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


def _rule(styles: str, selector: str) -> str:
    """Return the declarations of one single-line CSS rule."""

    start = styles.index(f"\n{selector} {{") + 1
    return styles[start : styles.index("}", start)]


class TestStartScreenFrontend(unittest.TestCase):
    """Keep the user's start-screen choices; keep the lists short."""

    def test_new_project_keeps_the_answer_settings(self):
        """Sent with the project; not reset before it is created."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            "...newProjectAnswerSettings(),",
            _function(source, "collectSettings"),
        )
        settings = _function(source, "showSettings")
        self.assertIn(
            "const keepAnswers = ready || answerSettingsEdited;", settings
        )
        self.assertIn(
            'if (!keepAnswers) byId("thinking").checked = profile.thinking;',
            settings,
        )
        # Opening a project hands the settings over to it.
        self.assertEqual(source.count("answerSettingsEdited = false;"), 3)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_answer_settings_sent_for_a_new_project(self):
        """Thinking always; temperature and token limit when valid."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const fields = {
  thinking: { checked: false },
  temperature: { value: "0.3" },
  "max-new-tokens": { value: "" },
};
const byId = (id) => fields[id];
"""
        script += _function(source, "newProjectAnswerSettings")
        script += r"""
assert.deepEqual(newProjectAnswerSettings(),
  { thinking: false, temperature: 0.3, max_new_tokens: "none" });
fields.thinking.checked = true;
fields["max-new-tokens"].value = "512";
assert.deepEqual(newProjectAnswerSettings(),
  { thinking: true, temperature: 0.3, max_new_tokens: 512 });
// Invalid values are left out, so the project keeps its defaults.
fields.temperature.value = "warm";
fields["max-new-tokens"].value = "-3";
assert.deepEqual(newProjectAnswerSettings(), { thinking: true });
console.log("answer settings passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("answer settings passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_models_that_can_think_are_marked(self):
        """Verified marks only, in the levels and intent; unknown is blank."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            "`Language model for ${level.name || `level ${index + 1}`}`, "
            "true);",
            _function(source, "renderModelsPanel"),
        )
        self.assertIn(
            '"Intent model", true);', _function(source, "renderModelsPanel")
        )
        script = r"""
const assert = require("node:assert/strict");
let busy = false;
const cachedModels = new Set(["Qwen/Qwen3-8B-MLX-4bit"]);
// Only what the engine verified from the templates; never the name.
const thinkingSwitches = {
  "Qwen/Qwen3-8B-MLX-4bit": true, "Qwen/Qwen3-1.7B-MLX-4bit": false,
};
const element = () => ({
  children: [], setAttribute() {},
  append(...items) { this.children.push(...items); },
});
var document = { createElement: element };
"""
        for name in ("modelCanThink", "thinkingMark", "modelSelect"):
            script += _function(source, name)
        script += r"""
assert.equal(modelCanThink("Qwen/Qwen3-8B-MLX-4bit"), true);
// A Qwen3 name alone is not enough: its template must switch thinking.
assert.equal(modelCanThink("Qwen/Qwen3-1.7B-MLX-4bit"), false);
const ids = [
  "Qwen/Qwen3-8B-MLX-4bit", "Qwen/Qwen3-1.7B-MLX-4bit",
  "Qwen/Qwen3-4B-Instruct-2507",
];
const texts = (select) => select.children.map((option) => option.textContent);
// Can think, no switch, and unknown (not checked: blank).
assert.deepEqual(texts(modelSelect(ids, ids[0], "Level", true)), [
  "Qwen/Qwen3-8B-MLX-4bit · can think",
  "Qwen/Qwen3-1.7B-MLX-4bit · no thinking switch (not downloaded)",
  "Qwen/Qwen3-4B-Instruct-2507 (not downloaded)",
]);
// Without the flag (embedding) nothing is marked.
assert.deepEqual(texts(modelSelect(ids, ids[0], "Embedding model")), [
  "Qwen/Qwen3-8B-MLX-4bit",
  "Qwen/Qwen3-1.7B-MLX-4bit (not downloaded)",
  "Qwen/Qwen3-4B-Instruct-2507 (not downloaded)",
]);
console.log("thinking models passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("thinking models passed", result.stdout)

    def test_project_lists_show_five_then_scroll(self):
        """Home page and left panel: five rows high, then a scrollbar."""

        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        for selector, row in (
            (".project-list", 34),
            (".saved-project-list", 26),
        ):
            rule = _rule(styles, selector)
            self.assertIn(f"max-height: calc(5 * {row}px + 4 * 3px)", rule)
            self.assertIn("overflow-y: auto", rule)
        # The rows really are that tall, so exactly five fit.
        self.assertRegex(
            styles,
            re.compile(r"\.project-item \{[^}]*min-height: 34px"),
        )
        self.assertRegex(
            styles,
            re.compile(
                r"\.saved-project-group button \{[^}]*min-height: 26px"
            ),
        )
        source = (_WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            'list.className = "saved-project-list";',
            _function(source, "renderSavedProjects"),
        )
        self.assertNotIn("SIDEBAR_PROJECTS", source)

    def test_short_window_never_overlaps_the_sidebar_lists(self):
        """Projects shrink and scroll; they never spill under Conversations.

        Regression: the projects section could shrink while its list kept
        five rows, so rows were drawn under the Conversations header.
        """

        styles = (_WEB / "styles.css").read_text(encoding="utf-8")
        projects = _rule(styles, ".project-list")
        self.assertIn("flex: 0 1 auto", projects)
        self.assertIn("min-height: 0", projects)
        self.assertIn(
            "min-height: calc(42px + 34px)",
            _rule(
                styles,
                '.sidebar-section[data-section="projects"]:not(.collapsed)',
            ),
        )
        self.assertIn(
            "flex: 1 1 0",
            _rule(styles, '.sidebar-section[data-section="conversations"]'),
        )
        self.assertIn("flex: none", _rule(styles, ".sidebar-logo"))


if __name__ == "__main__":
    unittest.main()
