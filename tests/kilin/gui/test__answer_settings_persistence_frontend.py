"""Verify answer-settings save acknowledgements in the desktop page.

These Node.js tests execute production ``app.js`` functions and the actual
answer-event branches without Qt. They cover serialized save requests,
correlated replies, authoritative failure recovery, and submission gating.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import shutil
import unittest

from tests.kilin.gui._frontend_support import extract_function, run_node

_WEB = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"


class TestAnswerSettingsPersistenceFrontend(unittest.TestCase):
    """Keep pending UI edits separate from saved answer configuration."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the production JavaScript once for isolated execution."""

        cls.source = _WEB.read_text(encoding="utf-8")

    @classmethod
    def _save_harness(cls) -> str:
        """Build a minimal DOM and extract the real save/event handlers."""

        script = r"""
const assert = require("node:assert/strict");
const fields = {};
const byId = (id) => (fields[id] ||= {
  _value: "", checked: false, textContent: "",
  set value(raw) { this._value = String(raw); },
  get value() { return this._value; },
  classList: { add: () => {}, remove: () => {} },
});
const error = byId("error");
const calls = [];
let ready = true;
let busy = false;
let openProjectPath = "/saved/alpha";
let savedAnswerNumbers = { temperature: 0, max_new_tokens: 8192 };
let savedAnswerThinking = true;
const answerSettingsSave = {
  nextId: 0, pending: null, queued: null, uncertain: false,
};
const bridge = {
  saveAnswerSettings: (json) => calls.push(JSON.parse(json)),
};
let updates = 0;
const updateSendButton = () => { updates++; };
const renderEffortPicker = () => {};
const keepNewProjectValue = () => {};
byId("temperature").value = "0";
byId("max-new-tokens").value = "8192";
byId("thinking").checked = true;
byId("question").style = {};
byId("question").scrollHeight = 54;
"""
        for name in (
            "resizeQuestionInput", "setQuestionDraft",
            "confirmedAnswerSettings", "sameAnswerSettings",
            "displayAnswerSettings", "resetAnswerSettings",
            "queueAnswerSettings", "matchesAnswerSettingsSave",
            "answerSettingsSaved", "answerSettingsFailed",
            "saveAnswerNumbers", "onEvent",
        ):
            script += extract_function(cls.source, name)
        return script

    def test_correlated_saves_are_wired_through_the_real_page(self) -> None:
        """Protect the event handlers even when Node is unavailable."""

        self.assertIn(
            'answerSettingsSaved(data);', self.source
        )
        self.assertIn(
            'answerSettingsFailed(data);', self.source
        )
        self.assertIn(
            'queueAnswerSettings({', self.source
        )
        self.assertIn(
            '|| Boolean(answerSettingsSave.pending)', self.source
        )
        self.assertIn(
            'answerSettingsSave.uncertain', self.source
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_serialized_acks_and_failure_reconciliation(self) -> None:
        """Retain latest edits and ignore mismatched or obsolete events."""

        script = self._save_harness() + r"""
byId("temperature").value = "0.5";
saveAnswerNumbers();
assert.equal(calls.length, 1);
assert.deepEqual(calls[0], {
  request_id: 1, project: "/saved/alpha",
  settings: { temperature: 0.5, thinking: true, max_new_tokens: 8192 },
});
assert.equal(savedAnswerNumbers.temperature, 0);
assert.match(byId("model-download-status").textContent, /Saving/);

// More edits while one request is running replace the queued snapshot.
byId("max-new-tokens").value = "2048";
saveAnswerNumbers();
byId("thinking").checked = false;
saveAnswerNumbers();
assert.equal(calls.length, 1);
assert.equal(answerSettingsSave.queued.max_new_tokens, 2048);
assert.equal(answerSettingsSave.queued.thinking, false);

onEvent(JSON.stringify({
  event: "answer_settings_saved", request_id: 1,
  project: "/wrong", temperature: 0.9, thinking: false,
  max_new_tokens: 123,
}));
assert.equal(calls.length, 1);
assert.equal(savedAnswerNumbers.temperature, 0);

onEvent(JSON.stringify({
  event: "answer_settings_saved", request_id: 1,
  project: "/saved/alpha", temperature: 0.5, thinking: true,
  max_new_tokens: 8192,
}));
assert.equal(savedAnswerNumbers.temperature, 0.5);
assert.equal(calls.length, 2);
assert.deepEqual(calls[1], {
  request_id: 2, project: "/saved/alpha",
  settings: { temperature: 0.5, thinking: false, max_new_tokens: 2048 },
});

// A failed second save can report a post-write profile, not old memory.
onEvent(JSON.stringify({
  event: "answer_settings_failed", request_id: 2,
  project: "/saved/alpha", verified: true,
  message: "Could not finish save", temperature: 0.5,
  thinking: false, max_new_tokens: 2048,
}));
assert.equal(answerSettingsSave.pending, null);
assert.equal(answerSettingsSave.queued, null);
assert.equal(answerSettingsSave.uncertain, false);
assert.equal(savedAnswerThinking, false);
assert.equal(byId("max-new-tokens").value, "2048");
assert.match(error.textContent, /Could not finish/);

// If reconciliation fails, do not accept new edits or permit an answer.
byId("temperature").value = "0.8";
saveAnswerNumbers();
assert.equal(calls.length, 3);
onEvent(JSON.stringify({
  event: "answer_settings_failed", request_id: 3,
  project: "/saved/alpha", verified: false,
  message: "Cannot read profile",
}));
assert.equal(answerSettingsSave.uncertain, true);
assert.equal(byId("temperature").value, "0.5");
byId("thinking").checked = true;
saveAnswerNumbers();
assert.equal(calls.length, 3);
assert.match(error.textContent, /Reopen the project/);

// A different project resets the queue; late acknowledgements are ignored.
resetAnswerSettings({
  temperature: 0.2, thinking: true, max_new_tokens: 512,
});
openProjectPath = "/saved/beta";
byId("temperature").value = "0.3";
saveAnswerNumbers();
assert.equal(calls.length, 4);
onEvent(JSON.stringify({
  event: "answer_settings_saved", request_id: 3,
  project: "/saved/alpha", temperature: 1.0, thinking: false,
  max_new_tokens: 99,
}));
assert.equal(savedAnswerNumbers.temperature, 0.2);
onEvent(JSON.stringify({
  event: "answer_settings_saved", request_id: 4,
  project: "/saved/beta", temperature: 0.3, thinking: true,
  max_new_tokens: 512,
}));
assert.equal(savedAnswerNumbers.temperature, 0.3);
assert.equal(answerSettingsSave.pending, null);
console.log("answer settings queue passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("answer settings queue passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_submit_waits_for_persisted_settings(self) -> None:
        """Never clear a question or send it with unconfirmed settings."""

        script = self._save_harness()
        start = self.source.index(
            'byId("question-form").addEventListener("submit"'
        )
        end = self.source.index("\n});\n", start) + 4
        script += r"""
let questionSubmit = null;
const form = byId("question-form");
form.addEventListener = (name, handler) => { questionSubmit = handler; };
let sent = [];
const sendQuestion = (ask) => { sent.push(ask); };
const returnToAnswer = () => {};
const renderAttachments = () => {};
let messageAttachments = [];
let selectedModelLevel = 2;
let selectedDepthStep = 3;
"""
        script += self.source[start:end]
        script += r"""
const submit = () => questionSubmit({ preventDefault: () => {} });
byId("question").value = "What changed?";
byId("temperature").value = "0.8";
submit();
assert.equal(sent.length, 0);
assert.equal(byId("question").value, "What changed?");
assert.equal(calls.length, 1);
assert.equal(answerSettingsSave.pending.id, 1);

submit();
assert.equal(sent.length, 0);
onEvent(JSON.stringify({
  event: "answer_settings_saved", request_id: 1,
  project: "/saved/alpha", temperature: 0.8,
  max_new_tokens: 8192, thinking: true,
}));
submit();
assert.equal(sent.length, 1);
assert.deepEqual(sent[0].settings, {
  temperature: 0.8, max_new_tokens: 8192, thinking: true,
});
assert.equal(byId("question").value, "");

byId("question").value = "Another question";
answerSettingsSave.uncertain = true;
submit();
assert.equal(sent.length, 1);
assert.equal(byId("question").value, "Another question");
console.log("answer submit gating passed");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("answer submit gating passed", result.stdout)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not available")
    def test_delayed_attachment_question_is_restored(self) -> None:
        """A delayed attached question is not lost during a save."""

        script = self._save_harness()
        script += r"""
let messageAttachments = [];
const renderAttachments = () => {};
"""
        script += extract_function(self.source, "sendQuestion")
        script += r"""
byId("question").value = "";
answerSettingsSave.pending = {
  id: 1, project: "/saved/alpha", settings: confirmedAnswerSettings(),
};
sendQuestion({
  question: "Use this source", attachments: ["/tmp/a.txt"],
  files: [], leftOut: [], effort: "{}", settings: confirmedAnswerSettings(),
});
assert.equal(byId("question").value, "Use this source");
assert.deepEqual(messageAttachments, ["/tmp/a.txt"]);
assert.match(error.textContent, /Confirm answer settings/);
console.log("attached answer preserved");
"""
        result = run_node(script)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("attached answer preserved", result.stdout)


if __name__ == "__main__":
    unittest.main()
