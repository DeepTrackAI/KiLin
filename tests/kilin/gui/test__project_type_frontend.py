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

from tests.kilin.gui._frontend_support import run_node


_ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which("node"), "Node.js is not available")
class TestProjectTypeFrontend(unittest.TestCase):
    """Select arbitrary types and render source-independent welcome text."""

    def test_registry_order_selection_and_dynamic_prompt_state(self):
        """No code/literature assumptions are needed to drive the browser."""

        source = (_ROOT / "kilin/gui/web/app.js").read_text(encoding="utf-8")
        functions = []
        for name in ("renderProjectTypes", "renderConversationEmpty"):
            start = source.index(f"function {name}(")
            end = source.index("\n}\n", start) + 3
            functions.append(source[start:end])
        state = []
        for prefix in (
            "let activeProjectTypeId", "const lastEmptyPrompt",
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
