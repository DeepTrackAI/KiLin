"""Check JavaScript preview decisions without starting Qt or a browser.

The focused Node.js test executes the actual file-selection functions with
synthetic preview rows. It requires no models, workspace, or UI process.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import unittest


_APP_JS = (
    Path(__file__).resolve().parents[3]
    / "qachat" / "gui" / "web" / "app.js"
)


class TestGUIFileSelection(unittest.TestCase):
    """Keep selectable-file policy independent of explanation wording."""

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_oversized_selection_uses_reason_code(self) -> None:
        """Exercise the real frontend checks with reworded explanations."""

        script = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(process.argv[1], "utf8");

function extract(name) {
  const start = source.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `Missing frontend function: ${name}`);
  const end = source.indexOf("\n}", start);
  assert.notEqual(end, -1, `Incomplete frontend function: ${name}`);
  return source.slice(start, end + 2);
}

const names = [
  "isOversizedSkippedFile", "selectedFileSizeLimit", "renderPreview"
];
const functions = names.map(extract).join("\n");
// Every relevant decision must use the same code-only classifier.
assert.equal(
  (source.match(/isOversizedSkippedFile\(item\)/g) || []).length, 5
);
assert.ok(!source.includes('item.reason === "larger than'));

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
    path: "big.py", reason_code: "file_too_large",
    reason: "The file exceeds your limit.", size_bytes: 50
  },
  {
    path: "secret.py", reason_code: "protected_file",
    reason: "larger than the file-size limit", size_bytes: 100
  }
];
const context = {
  FILE_TOO_LARGE_REASON: "file_too_large",
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
        result = subprocess.run(
            [shutil.which("node"), "-e", script, str(_APP_JS)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
