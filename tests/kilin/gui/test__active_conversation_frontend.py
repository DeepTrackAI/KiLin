"""Exercise the workspace tree's active-conversation marker without Qt.

Run the production ``markActiveConversationFile`` function in Node with a
minimal DOM adapter. The engine's trace path is the active conversation's
absolute JSON path; workspace rows carry project-relative paths, so the
row whose path ends that trace path is marked. Rendering is covered only
by the Qt/WebEngine smoke test and manual checks.

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
class TestActiveConversationFrontend(unittest.TestCase):
    """Mark the workspace row of the conversation file in use."""

    def test_active_conversation_file_is_marked(self):
        """Only the row whose path ends the active trace path is marked."""

        source = (_ROOT / "kilin/gui/web/app.js").read_text(encoding="utf-8")
        start = source.index("function markActiveConversationFile(")
        function = source[start : source.index("\n}\n", start) + 3]
        script = r"""
const assert = require("node:assert/strict");
function row(path) {
  const classes = new Set();
  return {
    dataset: path ? { path } : {},
    classList: {
      toggle(name, on) { on ? classes.add(name) : classes.delete(name); },
      contains: (name) => classes.has(name),
    },
  };
}
const rows = [
  row("logs/traces/conversation_a.json"),
  row("logs/traces/conversation_b.json"),
  row("sources/conversation_a.json"),
  row(null),
];
const byId = () => ({ querySelectorAll: () => rows });
let activeTracePath = "";
const marked = () => rows.map(
  (item) => item.classList.contains("active-conversation"));
"""
        script += function
        script += r"""
markActiveConversationFile();
assert.deepEqual(marked(), [false, false, false, false]);
activeTracePath = "C:\\work\\trial\\logs\\traces\\conversation_a.json";
markActiveConversationFile();
assert.deepEqual(marked(), [true, false, false, false]);
activeTracePath = "/work/trial/logs/traces/conversation_b.json";
markActiveConversationFile();
assert.deepEqual(marked(), [false, true, false, false]);
console.log("active conversation passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("active conversation passed", result.stdout)
