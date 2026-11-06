"""Check JavaScript preview decisions without starting Qt or a browser.

The focused Node.js test executes the actual file-selection functions with
synthetic preview rows. It requires no models, workspace, or UI process.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

import json
from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import (
    extract_function,
    run_node,
)


_APP_JS = (
    Path(__file__).resolve().parents[3]
    / "kilin" / "gui" / "web" / "app.js"
)


class TestGUIFileSelection(unittest.TestCase):
    """Keep selectable-file policy independent of explanation wording."""

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_oversized_selection_uses_reason_code(self) -> None:
        """Exercise the real frontend checks with reworded explanations."""

        source = _APP_JS.read_text(encoding="utf-8")
        functions = "\n".join(
            extract_function(source, name)
            for name in (
                "isOversizedSkippedFile", "skippedReason",
                "selectedFileSizeLimit", "renderPreview",
            )
        )
        script = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(process.argv[1], "utf8");
const codes = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

"""
        script += "const functions = " + json.dumps(functions) + ";\n"
        script += r"""
// Every relevant decision must use the same code-only classifier.
assert.equal(
  (source.match(/isOversizedSkippedFile\(item\)/g) || []).length, 5
);
assert.ok(!source.includes('item.reason === "larger than'));
assert.ok(!source.includes('"file_too_large"'));
assert.ok(source.includes(
  'selectionReasonCodes = data.selection_reason_codes;'
));

const tree = { replaceChildren() {}, append() {} };
const count = { textContent: "" };
const scope = { classList: { toggle() {} } };
const document = {
  createElement() {
    return { addEventListener() {}, append() {} };
  }
};
const rows = [
  {
    path: "big.py", reason_code: codes.FILE_TOO_LARGE,
    reason: "The file exceeds your limit.", size_bytes: 50
  },
  {
    path: "secret.py", reason_code: codes.PROTECTED_FILE,
    reason: "larger than the file-size limit", size_bytes: 100
  }
];
const context = {
  selectionReasonCodes: codes,
  selectedSources: ["source"],
  previewRoots: new Map([["source", ["ordinary.py"]]]),
  previewSkipped: new Map([["source", rows]]),
  checkedFiles: new Map([[
    "source", new Set(["ordinary.py", "big.py", "secret.py"])
  ]]),
  byId(id) {
    if (id === "max-file-bytes") return { value: "10" };
    if (id === "file-tree") return tree;
    if (id === "scope-count") return count;
    if (id === "scope") return scope;
    throw new Error(id);
  },
  document,
  branch() { return {}; },
  fileTree() { return {}; }
};
vm.runInNewContext(functions, context);
assert.equal(vm.runInNewContext("selectedFileSizeLimit()", context), 50);
vm.runInNewContext("renderPreview()", context);
assert.equal(count.textContent, "2 of 3 files selected · 1 unavailable");
"""
        result = run_node(
            script,
            args=(
                str(_APP_JS),
                str(_APP_JS.parents[2] / "repository_reason_codes.json"),
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
