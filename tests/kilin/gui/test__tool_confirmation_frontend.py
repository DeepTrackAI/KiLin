"""Exercise tool-confirmation controls using the production JavaScript.

A small DOM adapter runs the actual event handler and prompt functions in
Node. Tests cover option rendering, model and network plans, single-use
choices, progress timing, and nonfatal rejection of unrelated commands.
No network requests, models, Qt installation, or browser downloads run.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from __future__ import annotations

from pathlib import Path
import re
import shutil
import unittest

from tests.kilin.gui._frontend_support import (
    extract_function as _function,
    run_node,
)

_APP = Path(__file__).resolve().parents[3] / "kilin/gui/web/app.js"


_HARNESS = r"""
const assert = require('node:assert/strict');
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.events = {};
    this.attrs = {}; this.parent = null; this.disabled = false;
    this.text = ''; this.className = '';
    const names = new Set();
    this.classList = {
      add: (...items) => items.forEach(x => names.add(x)),
      remove: (...items) => items.forEach(x => names.delete(x)),
      contains: x => names.has(x),
      toggle: (x, value) => {
        if (value ?? !names.has(x)) names.add(x); else names.delete(x);
      },
    };
  }
  set textContent(text) { this.text = text; this.children = []; }
  get textContent() {
    return this.text + this.children.map(x => x.textContent).join('');
  }
  append(...nodes) {
    for (const node of nodes) {
      node.remove(); node.parent = this; this.children.push(node);
    }
  }
  remove() {
    if (this.parent) {
      this.parent.children = this.parent.children.filter(x => x !== this);
      this.parent = null;
    }
  }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(name, callback) { this.events[name] = callback; }
  querySelectorAll(tag) {
    return this.children.flatMap(x => [
      ...(x.tag === tag ? [x] : []), ...x.querySelectorAll(tag),
    ]);
  }
  querySelector(tag) { return this.querySelectorAll(tag)[0] || null; }
  focus() { document.activeElement = this; }
}
const nodes = new Map();
const byId = id => {
  if (!nodes.has(id)) nodes.set(id, new Element('div'));
  return nodes.get(id);
};
const document = {createElement: tag => new Element(tag)};
const activity = new Element('div');
const error = new Element('div');
const conversationScroll = {scrollTop: 0, scrollHeight: 100};
let busy = true;
let activeTracePath = '';
const errors = [];
function showError(message) {
  errors.push(message); pendingTurn = null; busy = false;
}
function paintPendingTurn() {}
let clock = 0;
const performance = {now: () => ++clock};
let timerId = 0;
const timers = new Map();
const setTimeout = callback => {
  timers.set(++timerId, callback); return timerId;
};
const clearTimeout = id => timers.delete(id);
function flushTimers() {
  while (timers.size) {
    const [id, callback] = timers.entries().next().value;
    timers.delete(id); callback();
  }
}
let answerStageTimer = null;
let answerStageKey = null;
let answerStageQueue = [];
const commands = [];
const bridge = {choosePlan: value => commands.push(value)};
let pendingTurn = null;
// No attached files are being added to sources in these tests.
let attachedAsk = null;
// No indexing settings are being saved in these tests.
let applyingIndexing = false;
function begin() {
  resetAnswerStages(); commands.length = 0; errors.length = 0;
  error.classList.add('hidden'); busy = true;
  pendingTurn = {
    answer: new Element('div'), detail: new Element('div'), timeline: [],
  };
}
function event(payload) { onEvent(JSON.stringify(payload)); }
function choice() {
  event({event: 'answer_phase', phase: 'processing', choice: {
    question: "Which paper's references should I check?",
    options: [
      'mijalkov17braph.pdf (best match for the question)',
      'tiunn26braph2.pdf (matches the question)',
      'mijalkov25computational.pdf (in this project)',
      'patil25deep.pdf (in this project)',
    ],
  }});
}
function plan(operation = 'model') {
  event({event: 'answer_phase', phase: 'processing', plan: {
    operation, summary: 'Check the missing references.', scope: 'Project',
    batches_to_run: 12, quick_items: 5, estimated_seconds: 30,
    quick_seconds: 10, details: [],
  }});
}
"""


@unittest.skipUnless(shutil.which("node"), "Node.js is not available")
class TestToolConfirmationFrontend(unittest.TestCase):
    """Keep the waiting status paired with working confirmation buttons."""

    def _run(self, checks: str) -> None:
        source = _APP.read_text(encoding="utf-8")
        constants = "\n".join(re.findall(
            r"^const (?:ANSWER_PHASE_|PLAN_CHOICE_|ENGINE_PHASE_|"
            r"ANSWER_STAGE_)[^\n]+", source, flags=re.MULTILINE,
        ))
        functions = "\n".join(_function(source, name) for name in (
            "resetAnswerStages", "advanceAnswerStage", "queueAnswerStage",
            "markTimelineStep", "showTurnPhase", "formatDuration",
            "showToolPrompt", "submitToolPrompt", "showToolPlan",
            "showToolChoice", "onEvent",
        ))
        result = run_node("\n".join((
                _HARNESS, constants, functions, checks,
            )))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_recorded_document_choice_renders_all_options(self) -> None:
        self._run(r"""
begin();
choice();
const buttons = pendingTurn.answer.querySelectorAll('button');
assert.equal(buttons.length, 5);
assert.equal(buttons[4].textContent, 'Cancel');
assert.equal(buttons[1].textContent,
  'tiunn26braph2.pdf (matches the question)');
assert.equal(document.activeElement, buttons[0]);
assert.equal(pendingTurn.detail.textContent, 'Waiting for your choice');
assert.equal(pendingTurn.prompt.attrs.role, 'group');
assert.equal(commands.length, 0); // Rendering is not approval.
buttons[1].events.click();
assert.deepEqual(commands, ['option:1']);
assert.equal(pendingTurn.answer.querySelectorAll('button').length, 0);
assert.equal(pendingTurn.detail.textContent, 'Using tiunn26braph2.pdf');
""")

    def test_document_choice_cancel_and_empty_options(self) -> None:
        self._run(r"""
begin(); choice();
pendingTurn.prompt.querySelectorAll('button').at(-1).events.click();
assert.deepEqual(commands, ['cancel']);
assert.equal(pendingTurn.detail.textContent, 'Cancelling');
begin();
showToolChoice({question: 'No options', options: []});
const buttons = pendingTurn.prompt.querySelectorAll('button');
assert.equal(buttons.length, 1);
assert.equal(buttons[0].textContent, 'Cancel');
buttons[0].events.click();
assert.deepEqual(commands, ['cancel']);
""")

    def test_repeat_choice_has_no_cancel_button(self) -> None:
        """Routing choices require one of the two explicit options."""

        self._run(r"""
begin();
showToolChoice({
  question: 'How should I answer?',
  options: ['Same way again', 'Try another way'],
  allow_cancel: false,
});
const buttons = pendingTurn.prompt.querySelectorAll('button');
assert.equal(buttons.length, 2);
assert.equal(buttons[0].textContent, 'Same way again');
assert.equal(buttons[1].textContent, 'Try another way');
buttons[1].events.click();
assert.deepEqual(commands, ['option:1']);
""")

    def test_model_and_network_plans_require_an_explicit_choice(self):
        self._run(r"""
for (const operation of ['model', 'network']) {
  for (const [index, value] of ['continue', 'quick', 'cancel'].entries()) {
    begin(); plan(operation);
    const buttons = pendingTurn.prompt.querySelectorAll('button');
    assert.equal(buttons.length, 3);
    assert.equal(commands.length, 0);
    assert.match(buttons[0].textContent, operation === 'network'
      ? /Authorize lookup/ : /Read everything/);
    assert.equal(buttons[2].textContent, operation === 'network'
      ? 'Stay offline' : 'Cancel');
    buttons[index].events.click();
    assert.deepEqual(commands, [value]);
  }
}
""")

    def test_old_and_double_clicks_cannot_submit_another_answer(self):
        self._run(r"""
begin(); choice();
const old = pendingTurn.prompt.querySelector('button');
plan();
assert.equal(pendingTurn.answer.children.length, 1);
old.events.click();
assert.deepEqual(commands, []);
const current = pendingTurn.prompt.querySelector('button');
current.events.click(); current.events.click(); old.events.click();
assert.deepEqual(commands, ['continue']);
assert.equal(current.disabled, true);
""")

    def test_waiting_and_submission_clear_stale_progress(self) -> None:
        self._run(r"""
begin();
queueAnswerStage('earlier', 'Earlier processing');
queueAnswerStage('queued', 'Unrelated queued text');
choice(); flushTimers();
assert.equal(pendingTurn.detail.textContent, 'Waiting for your choice');
assert.equal(byId('activity-text').textContent, 'Waiting for your choice');
pendingTurn.prompt.querySelector('button').events.click();
assert.equal(pendingTurn.detail.textContent, 'Using mijalkov17braph.pdf');
flushTimers();
assert.equal(pendingTurn.detail.textContent, 'Using mijalkov17braph.pdf');
assert.equal(byId('activity-text').textContent, 'Using mijalkov17braph.pdf');
""")

    def test_rejected_command_preserves_the_pending_question(self):
        self._run(r"""
begin(); choice();
const card = pendingTurn.prompt;
event({event: 'error', message: 'Choose first', pending_choice: true});
assert.equal(pendingTurn.prompt, card);
assert.equal(card.querySelectorAll('button').length, 5);
assert.equal(busy, true);
assert.deepEqual(errors, []);
assert.equal(error.textContent, 'Choose first');
card.querySelector('button').events.click();
assert.deepEqual(commands, ['option:0']);
assert.equal(error.classList.contains('hidden'), true);
// A rejection already in transit can arrive after the user's click.
event({event: 'error', message: 'Choose first', pending_choice: true});
assert.ok(pendingTurn);
assert.equal(busy, true);
assert.equal(pendingTurn.prompt, null);
assert.deepEqual(errors, []);
""")

    def test_terminal_errors_still_abort_and_late_choices_do_nothing(self):
        self._run(r"""
begin(); choice();
const detached = pendingTurn.prompt.querySelector('button');
event({event: 'error', message: 'Answer failed'});
assert.deepEqual(errors, ['Answer failed']);
assert.equal(pendingTurn, null);
assert.equal(busy, false);
detached.events.click();
choice(); plan();
assert.equal(pendingTurn, null);
assert.deepEqual(commands, []);
""")


if __name__ == "__main__":
    unittest.main()
