"""Verify that deleting a conversation reports its actual archive location.

Execute the production event handler with lightweight Node page substitutes.
A static assertion remains available when Node.js is not installed.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_APP = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"


class TestConversationArchiveFrontend(unittest.TestCase):
    """Report logical deletion without misleading archive paths."""

    def test_delete_event_shows_hidden_archive_location(self) -> None:
        """Remove the conversation row and show the .deleted location."""

        source = _APP.read_text(encoding="utf-8")
        handler = extract_function(source, "onEvent")
        expected = "The archived trace remains in logs/traces/.deleted/."
        if shutil.which("node") is None:
            self.assertIn('data.event === "conversation_delete_started"',
                          handler)
            self.assertIn(expected, handler)
            return
        script = r"""
const assert = require("node:assert/strict");
const fields = {};
const byId = (name) => (fields[name] ??= {textContent: ""});
let removed = false;
const row = {remove() {removed = true;}};
const button = {
  dataset: {conversationId: "conversation_123"},
  closest(selector) {
    assert.equal(selector, ".conversation-row");
    return row;
  },
};
const document = {
  querySelectorAll(selector) {
    assert.equal(selector, ".conversation-item");
    return [button];
  },
};
""" + handler + r"""
onEvent(JSON.stringify({
  event: "conversation_delete_started", conversation_id: "conversation_123",
}));
assert.equal(removed, true);
assert.equal(byId("activity-detail").textContent,
  "The archived trace remains in logs/traces/.deleted/.");
assert.match(byId("activity-text").textContent, /Conversation removed/);
console.log("conversation archive location passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("conversation archive location passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
