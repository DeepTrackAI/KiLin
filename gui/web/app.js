"use strict";

const byId = (id) => document.getElementById(id);
const setup = byId("setup");
const chat = byId("chat");
const conversation = byId("conversation");
const activity = byId("activity");
const error = byId("error");
let bridge = null;
let busy = false;
let ready = false;

function showError(message) {
  error.textContent = message;
  error.classList.remove("hidden");
  setBusy(false);
}

function setBusy(value, message = "") {
  busy = value;
  byId("open").disabled = value || !bridge;
  byId("send").disabled = value || !ready;
  byId("new-chat").disabled = value || !ready;
  byId("stop").disabled = !value;
  activity.classList.toggle("hidden", !value);
  byId("activity-text").textContent = message;
  if (value) {
    byId("activity-percent").textContent = "Working…";
    byId("activity-detail").textContent = "";
  } else {
    byId("activity-fill").style.width = "0%";
  }
}

function seconds(value) {
  const rounded = Math.max(0, Math.round(value));
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded % 60;
  return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

function showProgress(data) {
  setBusy(true, data.message);
  const percent = Math.min(100, Math.max(0, Number(data.percent)));
  byId("activity-percent").textContent = `${percent.toFixed(1)}% overall`;
  byId("activity-fill").style.width = `${percent}%`;
  const details = [];
  if (data.current != null && data.total != null) {
    details.push(`${data.current.toLocaleString()} / ${data.total.toLocaleString()}`);
  }
  if (data.remaining != null) {
    details.push(`${data.remaining.toLocaleString()} remaining`);
  }
  if (data.elapsed_seconds != null) {
    details.push(`${seconds(data.elapsed_seconds)} elapsed`);
  }
  if (data.eta_seconds != null) {
    details.push(`about ${seconds(data.eta_seconds)} left`);
  }
  byId("activity-detail").textContent = details.join(" · ");
}

function addTurn(turn) {
  const item = document.createElement("article");
  item.className = "turn";
  const question = document.createElement("div");
  question.className = "user-message";
  question.textContent = turn.question;
  const answer = document.createElement("div");
  answer.className = "answer-message";
  answer.textContent = turn.answer;
  item.append(question, answer);

  if (turn.evidence?.length) {
    const evidence = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `Evidence · ${turn.evidence.length} sources`;
    evidence.append(summary);
    for (const source of turn.evidence) {
      const block = document.createElement("div");
      block.className = "evidence-item";
      const label = document.createElement("strong");
      label.textContent = `[${source.citation}] ${source.path}:${source.start_line}-${source.end_line}`;
      const content = document.createElement("pre");
      content.textContent = source.text;
      block.append(label, content);
      evidence.append(block);
    }
    item.append(evidence);
  }
  conversation.append(item);
  conversation.scrollTop = conversation.scrollHeight;
}

function onEvent(raw) {
  const data = JSON.parse(raw);
  if (data.event === "status") {
    setBusy(true, data.message);
  } else if (data.event === "progress") {
    showProgress(data);
  } else if (data.event === "opened") {
    ready = true;
    setup.classList.add("hidden");
    chat.classList.remove("hidden");
    conversation.replaceChildren();
    data.turns.forEach(addTurn);
    byId("project-name").textContent = data.source.split(/[\\/]/).pop();
    byId("project-detail").textContent = `${data.kind} · index ${data.index_status} · ${data.turns.length} saved turns`;
    byId("connection").textContent = "Local · ready";
    setBusy(false);
  } else if (data.event === "answered") {
    addTurn(data.turn);
    byId("question").value = "";
    setBusy(false);
  } else if (data.event === "reset") {
    conversation.replaceChildren();
    byId("project-detail").textContent = "New conversation · index retained";
    setBusy(false);
  } else if (data.event === "stopped") {
    ready = false;
    chat.classList.add("hidden");
    setup.classList.remove("hidden");
    byId("connection").textContent = "Stopped · reopen your project";
    setBusy(false);
  } else if (data.event === "backend_failed") {
    ready = false;
    chat.classList.add("hidden");
    setup.classList.remove("hidden");
    showError(data.message);
  } else if (data.event === "error") {
    showError(data.message);
  }
}

byId("browse").addEventListener("click", () => bridge?.chooseFolder());
byId("stop").addEventListener("click", () => {
  if (!busy || !bridge) return;
  byId("stop").disabled = true;
  byId("activity-text").textContent = "Stopping…";
  bridge.stopOperation();
});
byId("project-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || !bridge) return;
  error.classList.add("hidden");
  setBusy(true, "Opening project…");
  bridge.openProject(byId("source").value.trim(), byId("kind").value);
});
byId("question-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = byId("question").value.trim();
  if (!question || busy || !ready) return;
  error.classList.add("hidden");
  setBusy(true, "Finding evidence and answering…");
  bridge.askQuestion(question);
});
byId("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    byId("question-form").requestSubmit();
  }
});
byId("new-chat").addEventListener("click", () => {
  if (busy || !confirm("Archive this conversation and start a new one?")) return;
  setBusy(true, "Starting a new conversation…");
  bridge.resetConversation();
});
byId("change-project").addEventListener("click", () => {
  chat.classList.add("hidden");
  setup.classList.remove("hidden");
});

new QWebChannel(qt.webChannelTransport, (channel) => {
  bridge = channel.objects.qachat;
  bridge.event.connect(onEvent);
  bridge.folderSelected.connect((path) => { byId("source").value = path; });
  byId("connection").textContent = "Local · choose a project";
  setBusy(false);
});
