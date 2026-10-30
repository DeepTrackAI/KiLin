"""Exercise KíLín's own yes/no dialog.

Run the production ``askKilin`` function in Node with a stub dialog, and
check that the questions moved off the system pop-up use it. Rendering is
covered only by the Qt/WebEngine smoke test and manual checks.

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


class TestKilinDialogsFrontend(unittest.TestCase):
    """Ask in a KíLín dialog instead of the system pop-up."""

    def test_dialog_has_a_title_message_and_two_buttons(self):
        """One shared dialog; Cancel and the confirming action."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        start = page.index('<dialog id="kilin-ask"')
        dialog = page[start : page.index("</dialog>", start)]
        for part in (
            "kilin-ask-title",
            "kilin-ask-message",
            "kilin-ask-cancel",
            "kilin-ask-confirm",
        ):
            self.assertIn(f'id="{part}"', dialog)

    def test_delete_conversation_asks_in_the_kilin_dialog(self):
        """Deleting a chat no longer uses the system pop-up."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        render = _function(source, "renderConversations")
        self.assertNotIn("confirm(", render)
        ask = 'await askKilin(`Delete “${item.name}”?`, "", "Delete")'
        self.assertIn(ask, render)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_answer_follows_the_button_pressed(self):
        """Confirm gives true; Cancel and Escape give false."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        script = r"""
const assert = require("node:assert/strict");
const focused = [];
function element(id) {
  const classes = new Set();
  return {
    id, textContent: "", focus() { focused.push(id); },
    classList: {
      toggle(name, on) {
        if (on) classes.add(name); else classes.delete(name);
      },
      contains(name) { return classes.has(name); },
    },
  };
}
const dialog = {
  returnValue: "x", open: false, listeners: [],
  addEventListener(type, listener) { this.listeners.push(listener); },
  showModal() { this.open = true; },
  close(value) {
    this.open = false;
    if (value !== undefined) this.returnValue = value;
    const listeners = this.listeners;
    this.listeners = [];
    listeners.forEach((listener) => listener());
  },
};
const parts = {};
const byId = (id) => id === "kilin-ask" ? dialog
  : (parts[id] = parts[id] || element(id));
"""
        script += _function(source, "askKilin")
        script += r"""
(async () => {
  let answer = askKilin("Delete “Notes”?", "KíLín moves it.", "Delete");
  assert.equal(dialog.open, true);
  assert.equal(parts["kilin-ask-title"].textContent, "Delete “Notes”?");
  const message = parts["kilin-ask-message"];
  assert.equal(message.textContent, "KíLín moves it.");
  assert.equal(message.classList.contains("hidden"), false);
  assert.equal(parts["kilin-ask-confirm"].textContent, "Delete");
  // Cancel has focus first, so Enter does not confirm by accident.
  assert.deepEqual(focused, ["kilin-ask-cancel"]);
  dialog.close("confirm");
  assert.equal(await answer, true);
  // No message: only the question shows.
  answer = askKilin("Again?", "", "Delete");
  assert.equal(message.classList.contains("hidden"), true);
  dialog.close("cancel");
  assert.equal(await answer, false);
  // Escape closes without a value; a stale "confirm" must not count.
  dialog.returnValue = "confirm";
  answer = askKilin("Once more?", "KíLín asks once more.", "Delete");
  dialog.close();
  assert.equal(await answer, false);
  console.log("kilin dialog passed");
})().catch((error) => { console.error(error); process.exit(1); });
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kilin dialog passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
