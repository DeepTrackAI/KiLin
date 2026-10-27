"""Exercise how the side panels follow the user's show/hide choice.

Run the production ``chooseSidePanels`` function in Node with a stub, and
check the event handlers that used to close the panels on their own: a new
chat and an empty conversation now leave them as the user left them, and
opening a project shows them unless the user closed them. Rendering is
covered only by the Qt/WebEngine smoke test and manual checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import subprocess
import unittest

_ROOT = Path(__file__).resolve().parents[3]
_APP = _ROOT / "kilin/gui/web/app.js"


def _between(source: str, start: str, end: str) -> str:
    """Return the source text from ``start`` up to the next ``end``."""

    first = source.index(start)
    return source[first : source.index(end, first)]


class TestSidePanelChoiceFrontend(unittest.TestCase):
    """Keep the side panels as the user left them."""

    def test_automatic_paths_follow_the_user(self):
        """New chats and empty conversations no longer close the panels."""

        source = _APP.read_text(encoding="utf-8")
        new_chat = _between(source, 'byId("new-chat")', "});")
        self.assertNotIn("showWorkspace", new_chat)
        conversation = _between(
            source, 'data.event === "conversation_opened"', "} else if"
        )
        self.assertNotIn("showWorkspace(false)", conversation)
        opened = _between(source, 'data.event === "opened"', "} else if")
        self.assertIn("showWorkspace(sidePanelChoice ?? true)", opened)
        toggle = _between(source, 'byId("toggle-workspace")', "});")
        self.assertIn("chooseSidePanels(", toggle)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_choice_is_remembered(self):
        """chooseSidePanels records the choice and applies it."""

        source = _APP.read_text(encoding="utf-8")
        start = source.index("let sidePanelChoice")
        state = source[start : source.index(";", start) + 1]
        start = source.index("function chooseSidePanels(")
        function = source[start : source.index("\n}\n", start) + 3]
        script = r"""
const assert = require("node:assert/strict");
const shown = [];
const showWorkspace = (visible) => shown.push(visible);
"""
        script += "\n".join([state, function])
        script += r"""
assert.equal(sidePanelChoice, null);
chooseSidePanels(false);
assert.equal(sidePanelChoice, false);
chooseSidePanels(true);
assert.equal(sidePanelChoice, true);
assert.deepEqual(shown, [false, true]);
console.log("side panel choice passed");
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("side panel choice passed", result.stdout)
