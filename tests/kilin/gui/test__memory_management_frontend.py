"""Test memory-management page rendering and correlated GUI requests.

Node executes production handlers without Qt or browser globals. The native
WebEngine remains an additional platform-level check.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"
_HTML = _WEB.with_name("index.html")
_CSS = _WEB.with_name("styles.css")


class TestMemoryManagementFrontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _WEB.read_text(encoding="utf-8")

    def test_controls_events_and_safe_dom_contract(self):
        html = _HTML.read_text(encoding="utf-8")
        css = _CSS.read_text(encoding="utf-8")
        for identifier in (
            "memory-messages-status", "memory-messages-list",
            "memory-previous", "memory-next",
        ):
            self.assertIn(f'id="{identifier}"', html)
        for event in ("memory_messages", "memory_exclusion"):
            self.assertIn(f'data.event === "{event}"', self.source)
        for slot in ("requestMemoryMessages", "setMemoryExclusion"):
            self.assertIn(f"bridge.{slot}(", self.source)
        self.assertIn(".memory-messages-list", css)
        rendering = extract_function(self.source, "renderMemoryMessages")
        self.assertNotIn("innerHTML", rendering)
        self.assertIn("title.textContent", rendering)
        self.assertIn("preview.textContent", rendering)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_render_navigation_exclusion_acknowledgement_and_scope(self):
        script = r"""
const assert = require("node:assert/strict");
class Node {
  constructor() {
    this.children = []; this.textContent = ""; this.disabled = false;
    this.className = ""; this.title = ""; this.events = {};
    this.classList = {add: () => {}, remove: () => {}};
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = [...nodes]; }
  addEventListener(name, callback) { this.events[name] = callback; }
}
const elements = new Map();
const byId = (id) => {
  if (!elements.has(id)) elements.set(id, new Node());
  return elements.get(id);
};
const document = {createElement: () => new Node()};
const error = new Node();
let openProjectPath = "/workspace", ready = true, busy = false;
let memoryMessagePage = null, memoryMessagesRequestedOffset = 0;
let memoryExcludeSave = {nextId: 0, pending: null};
let feedbackData = {project: "/workspace", revision: "a".repeat(64)};
const requests = [];
let refreshes = 0;
function refreshFeedbackControls() { refreshes += 1; }
const bridge = {
  requestMemoryMessages: (project, offset) =>
    requests.push({kind: "list", project, offset}),
  setMemoryExclusion: (data) =>
    requests.push({kind: "write", ...JSON.parse(data)}),
  requestConversationFeedback: (project) =>
    requests.push({kind: "feedback", project}),
};
"""
        for name in (
            "requestMemoryMessages", "renderMemoryMessages",
            "receiveMemoryMessages", "setMemoryExclusion",
            "receiveMemoryExclusion",
        ):
            script += "\n" + extract_function(self.source, name)
        script += r"""
const item = {turn_uuid: "0123456789abcdef0123456789abcdef",
  role: "user", conversation_id: "conversation_1",
  conversation_name: "Research <script>alert(1)</script>",
  turn_number: 1, status: "eligible", excluded: false,
  preview: "An earlier question <b>not HTML</b>"};
const page = {project: "/workspace", offset: 0, total: 1,
  revision: "a".repeat(64), messages: [item]};
receiveMemoryMessages({...page, project: "/other"});
assert.equal(memoryMessagePage, null);
receiveMemoryMessages(page);
const list = byId("memory-messages-list");
assert.equal(list.children.length, 1);
const row = list.children[0];
assert.match(row.children[0].children[0].textContent, /<script>/);
assert.equal(row.children[0].children[1].textContent, item.preview);
assert.equal(row.children[1].textContent, "Exclude");
assert.equal(byId("memory-next").disabled, true);
row.children[1].events.click();
assert.equal(memoryExcludeSave.pending.requestId, 1);
const write = requests[0];
assert.equal(write.kind, "write");
assert.equal(write.excluded, true);
assert.equal(write.role, "user");
assert.equal(write.revision, "a".repeat(64));
assert.equal(list.children[0].children[1].disabled, true);
receiveMemoryExclusion({project: "/other", request_id: 1,
  turn_uuid: item.turn_uuid, role: "user"});
assert.notEqual(memoryExcludeSave.pending, null);
receiveMemoryExclusion({project: "/workspace", request_id: 1,
  turn_uuid: item.turn_uuid, role: "user", revision: "b".repeat(64),
  entries: {[item.turn_uuid]: {user: {excluded: true}}}});
assert.equal(memoryExcludeSave.pending, null);
assert.equal(feedbackData.revision, "b".repeat(64));
assert.equal(refreshes, 1);
assert.deepEqual(requests[1], {kind: "list", project: "/workspace",
  offset: 0});
receiveMemoryMessages({...page, revision: "b".repeat(64),
  messages: [{...item, excluded: true, status: "excluded"}]});
assert.equal(list.children[0].children[1].textContent, "Restore");
list.children[0].children[1].events.click();
assert.equal(requests[2].excluded, false);
receiveMemoryExclusion({project: "/workspace", request_id: 2,
  turn_uuid: item.turn_uuid, role: "user", error: "stale revision"});
assert.equal(memoryExcludeSave.pending, null);
assert.match(error.textContent, /stale revision/);
assert.deepEqual(requests[3], {kind: "feedback", project: "/workspace"});
openProjectPath = "/different";
requestMemoryMessages();
assert.equal(list.children.length, 0);
assert.deepEqual(requests[5], {kind: "list", project: "/different",
  offset: 0});
receiveMemoryMessages({project: "/workspace", offset: 0,
  messages: [item]});
assert.equal(memoryMessagePage, null);
console.log("memory management contract passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
