"""Verify backend-specific installation advice in the desktop interface.

Exercise the Models panel's production JavaScript prefix with Node.js, so
switching the selected backend cannot leave the previous repair command.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_APP = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"


class TestBackendInstallFrontend(unittest.TestCase):
    """Render the complete command for the currently selected backend."""

    def test_repair_command_changes_with_selected_backend(self) -> None:
        """Do not rewrite arbitrary installer strings using a regex."""

        source = _APP.read_text(encoding="utf-8")
        render = extract_function(source, "renderModelsPanel")
        self.assertIn("installCommands[activeBackend]", render)
        self.assertNotIn("installCommand.replace", render)
        handler = extract_function(source, "onEvent")
        self.assertIn("data.install_commands", handler)
        if shutil.which("node") is None:
            self.skipTest("Node.js is not available")
        # Run the real first portion of renderModelsPanel without the
        # unrelated model-menu controls that follow it.
        prefix = render.split('  const chat = document.createElement(', 1)[0]
        prefix = prefix.replace(
            "function renderModelsPanel() {",
            "function renderBackendNote() {",
            1,
        ) + "}\n"
        script = r"""
const assert = require("node:assert/strict");
const fields = {};
const byId = (name) => (fields[name] ??= {
  textContent: "",
  classList: {toggle() {}},
  replaceChildren() {},
});
const document = {activeElement: null};
let activeBackend = "mlx";
let installedBackends = {mlx: false, torch: false};
let installCommands = {
  mlx: "uv pip install --python /one 'kilin[app,mlx]'",
  torch: "uv pip install --python /one 'kilin[app,torch]'",
};
""" + prefix + r"""
renderBackendNote();
assert.match(byId("models-backend").textContent, /kilin\[app,mlx\]/);
assert.doesNotMatch(byId("models-backend").textContent, /app,torch/);
activeBackend = "torch";
renderBackendNote();
assert.match(byId("models-backend").textContent, /kilin\[app,torch\]/);
assert.doesNotMatch(byId("models-backend").textContent, /app,mlx/);
installedBackends.torch = true;
renderBackendNote();
assert.equal(byId("models-backend").textContent, "");
console.log("backend-specific commands passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("backend-specific commands passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
