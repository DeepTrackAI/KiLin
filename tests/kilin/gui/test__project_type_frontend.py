"""Exercise registry-driven browser functions without native Qt dependencies.

Run the production dropdown and welcome rendering functions in Node with a
minimal DOM adapter. This complements, but does not replace, the native
Qt/WebEngine smoke test.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import (
    extract_function,
    run_node,
)


_ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which("node"), "Node.js is not available")
class TestProjectTypeFrontend(unittest.TestCase):
    """Select arbitrary types and render source-independent welcome text."""

    def test_registry_order_selection_and_dynamic_prompt_state(self):
        """No code/literature assumptions are needed to drive the browser."""

        source = (_ROOT / "kilin/gui/web/app.js").read_text(encoding="utf-8")
        functions = [
            extract_function(source, name)
            for name in (
                "renderProjectTypes", "chooseSuggestions",
                "renderConversationEmpty",
            )
        ]
        state = []
        for prefix in (
            "let activeProjectTypeId", "const lastEmptyPrompt",
            "const lastEmptySuggestions",
            "let projectTypes", "const genericEmptyChat",
        ):
            start = source.index(prefix)
            end = source.index(";", start) + 1
            state.append(source[start:end])
        script = r'''
const assert = require("node:assert/strict");
function element() {
  return {
    children: [], value: "", textContent: "",
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    addEventListener() {},
  };
}
const select = element();
const document = { createElement: element };
const byId = (id) => {
  assert.equal(id, "type-id");
  return select;
};
const conversation = element();
const conversationScroll = {};
const updateFloatingHeader = () => {};
const renderSavedProjects = () => {};
let savedProjects = [];
let openProjectPath = "";
let projectPresentation = null;
let sourceQuestions = [];
let ready = false;
'''
        script += "\n".join([*state, *functions])
        script += r'''
assert.equal(activeProjectTypeId, "");
const types = [
  {type_id: "alpha", label: "Alpha", welcome_prompts: ["A1", "A2"]},
  {type_id: "constructor", label: "Other"},
];
renderProjectTypes(types);
assert.equal(select.value, "alpha");
assert.equal(activeProjectTypeId, "alpha");
assert.deepEqual(select.children.map((item) => item.value),
                 ["alpha", "constructor"]);
select.value = "constructor";
renderProjectTypes(types);
assert.equal(select.value, "constructor");
assert.equal(activeProjectTypeId, "constructor");
renderConversationEmpty();
assert.equal(conversation.children[0].children[0].textContent,
             "What would you like to work on?");
assert.equal(lastEmptyPrompt.constructor, 0);
activeProjectTypeId = "alpha";
Math.random = () => 0;
renderConversationEmpty();
assert.equal(conversation.children[0].children[0].textContent, "A1");
renderConversationEmpty();
assert.equal(conversation.children[0].children[0].textContent, "A2");
assert.equal(lastEmptyPrompt.alpha, 1);
assert.equal(lastEmptyPrompt.constructor, 0);
ready = true;
activeProjectTypeId = "alpha";
select.value = "constructor";
renderProjectTypes(types);
assert.equal(activeProjectTypeId, "alpha");
console.log("registry rendering passed");
'''
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("registry rendering passed", result.stdout)

    def test_source_questions_come_first_and_fixed_ones_fill_up(self):
        """Offer three source questions, or fill up with fixed ones."""

        source = (_ROOT / "kilin/gui/web/app.js").read_text(encoding="utf-8")
        script = r'''
const assert = require("node:assert/strict");
''' + extract_function(source, "chooseSuggestions") + r'''
const fixed = ["Fixed first", "Fixed two", "Fixed three"];
const questions = [
  {intent: "a", question: "A1"}, {intent: "a", question: "A2"},
  {intent: "b", question: "B1"}, {intent: "c", question: "C1"},
  {intent: "d", question: "D1"},
];
const intentOf = (text) =>
  questions.find((item) => item.question === text).intent;
for (let run = 0; run < 20; run += 1) {
  const picked = chooseSuggestions(questions, fixed);
  assert.equal(picked.length, 3);
  assert.equal(new Set(picked.map(intentOf)).size, 3);
}
const seen = new Set();
for (let run = 0; run < 200; run += 1) {
  seen.add(...chooseSuggestions(questions.slice(0, 2), []));
}
assert.deepEqual([...seen].sort(), ["A1", "A2"]);
const one = [{intent: "a", question: "A1"}];
assert.deepEqual(chooseSuggestions(one, fixed).slice(0, 2),
                 ["A1", "Fixed first"]);
assert.equal(chooseSuggestions(one, fixed).length, 3);
// Without source questions: the first two fixed ones, then one other.
const many = ["Help", "Add files", "X", "Y", "Z"];
const thirds = new Set();
for (let run = 0; run < 200; run += 1) {
  const shown = chooseSuggestions([], many);
  assert.deepEqual(shown.slice(0, 2), ["Help", "Add files"]);
  thirds.add(shown[2]);
}
assert.deepEqual([...thirds].sort(), ["X", "Y", "Z"]);
// The next empty chat shows another example than the last one.
let shown = chooseSuggestions([], many);
for (let run = 0; run < 50; run += 1) {
  const next = chooseSuggestions([], many, shown);
  assert.notEqual(next[2], shown[2]);
  shown = next;
}
assert.deepEqual(chooseSuggestions([], ["Only"]), ["Only"]);
assert.deepEqual(chooseSuggestions([], []), []);
console.log("suggestion choice passed");
'''
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("suggestion choice passed", result.stdout)
