"""Exercise the divider between the Workspace and References panels.

Run the production ``setWorkspaceShare`` function in Node with a minimal
DOM adapter, and check that the divider sits between the two panels. The
drag itself is covered only by the Qt/WebEngine smoke test and manual
checks.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import run_node

_ROOT = Path(__file__).resolve().parents[3]
_WEB = _ROOT / "kilin/gui/web"


class TestSideDividerFrontend(unittest.TestCase):
    """Split the side column between Workspace and References."""

    def test_divider_sits_between_the_panels(self):
        """The divider follows Workspace and precedes References."""

        page = (_WEB / "index.html").read_text(encoding="utf-8")
        workspace = page.index('<aside class="workspace-panel"')
        divider = page.index('id="side-divider"')
        references = page.index('<aside class="references-panel"')
        self.assertLess(workspace, divider)
        self.assertLess(divider, references)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_workspace_share_stays_within_limits(self):
        """Dragged heights are rounded and kept between 15% and 85%."""

        source = (_WEB / "app.js").read_text(encoding="utf-8")
        start = source.index("const WORKSPACE_SHARE_LIMITS")
        state = source[start : source.index(";", start) + 1]
        start = source.index("function setWorkspaceShare(")
        function = source[start : source.index("\n}\n", start) + 3]
        script = r"""
const assert = require("node:assert/strict");
const properties = new Map();
const panels = {
  style: {
    setProperty: (name, value) => properties.set(name, value),
    removeProperty: (name) => properties.delete(name),
  },
};
const document = { querySelector: () => panels };
const share = () => properties.get("--workspace-share");
"""
        script += "\n".join([state, function])
        script += r"""
setWorkspaceShare(40.4);
assert.equal(share(), "40%");
setWorkspaceShare(2);
assert.equal(share(), "15%");
setWorkspaceShare(99);
assert.equal(share(), "85%");
setWorkspaceShare(null);
assert.equal(share(), undefined);
console.log("side divider passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("side divider passed", result.stdout)
