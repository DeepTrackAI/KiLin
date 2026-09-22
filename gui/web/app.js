"use strict";

const byId = (id) => document.getElementById(id);
const setup = byId("setup");
const chat = byId("chat");
const conversation = byId("conversation");
const activity = byId("activity");
const error = byId("error");
const shell = document.querySelector(".app-shell");
let bridge = null;
let busy = false;
let ready = false;
const selectedSources = [];
const previewRoots = new Map();
const previewSkipped = new Map();
const checkedFiles = new Map();
const expandedFolders = new Set();
let profile = null;
let cachedModels = new Set();
let modelChoices = [];
let downloadingModel = null;
let selectedTurn = null;
let activeReference = null;
let installedBackends = {};
let supportedExtensions = new Set();
let automaticBackend = "mlx";
let activeBackend = "mlx";
let resumeRefreshTimer = null;

function hideResume() {
  byId("recent-project").classList.add("hidden");
  byId("resume-recent").dataset.project = "";
}

function refreshSavedProject() {
  if (!bridge || busy || ready) return;
  bridge.findSavedProject(byId("kind").value, byId("workspace-path").value.trim());
}

function queueSavedProjectRefresh() {
  hideResume();
  clearTimeout(resumeRefreshTimer);
  resumeRefreshTimer = setTimeout(refreshSavedProject, 250);
}

function showWorkspace(visible) {
  // Keep the workspace out of the layout until a project is actually open.
  shell.classList.toggle("sidebar-open", visible);
  byId("toggle-workspace").classList.toggle("hidden", !ready);
  byId("toggle-workspace").textContent = visible ? "Hide panels" : "Show panels";
}

function addSources(paths) {
  for (const path of paths) {
    const clean = String(path).trim();
    if (clean && !selectedSources.includes(clean)) selectedSources.push(clean);
  }
  previewSources();
}

function fileTypes() {
  return byId("extensions").value.split(",").map((value) => value.trim().toLowerCase()).filter(Boolean);
}

function previewSources() {
  if (bridge && selectedSources.length) {
    setBusy(true, "Scanning selected sources…");
    bridge.previewSources(
      JSON.stringify(selectedSources), JSON.stringify(fileTypes()),
      Number(byId("max-file-bytes").value), byId("workspace-path").value.trim(),
    );
  }
}

function branch(name, children, paths, selection, depth = 0, enabled = null, reasons = {}) {
  const details = document.createElement("details");
  details.className = "tree-folder";
  const key = `${name}|${paths[0] || ""}`;
  if (depth < 1 || expandedFolders.has(key)) details.open = true;
  details.addEventListener("toggle", () => {
    if (!details.isConnected) return;
    details.open ? expandedFolders.add(key) : expandedFolders.delete(key);
  });
  const summary = document.createElement("summary");
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  const available = enabled ? paths.filter((path) => enabled.has(path)) : paths;
  checkbox.checked = available.length > 0 && available.every((path) => selection.has(path));
  checkbox.indeterminate = available.some((path) => selection.has(path)) && !checkbox.checked;
  checkbox.disabled = available.length === 0;
  checkbox.addEventListener("click", (event) => event.stopPropagation());
  checkbox.addEventListener("change", () => {
    available.forEach((path) => checkbox.checked ? selection.add(path) : selection.delete(path));
    renderPreview();
  });
  summary.append(checkbox, document.createTextNode(name));
  details.append(summary);
  const content = document.createElement("div");
  content.className = "tree-children";
  for (const [folder, node] of [...children.directories].sort()) {
    content.append(branch(folder, node, allFiles(node), selection, depth + 1, enabled, reasons));
  }
  for (const file of children.files.sort()) {
    const row = document.createElement("label");
    row.className = "tree-file";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = selection.has(file);
    check.disabled = enabled !== null && !enabled.has(file);
    check.addEventListener("change", () => {
      check.checked ? selection.add(file) : selection.delete(file);
      renderPreview();
    });
    row.append(check, document.createTextNode(file.split("/").pop()));
    if (reasons[file]) {
      const reason = document.createElement("small");
      reason.className = "skip-reason";
      reason.textContent = reasons[file];
      row.append(reason);
    }
    content.append(row);
  }
  details.append(content);
  return details;
}

function allFiles(node) {
  return [...node.files, ...[...node.directories.values()].flatMap(allFiles)];
}

function fileTree(files) {
  const root = { directories: new Map(), files: [] };
  for (const file of files) {
    const parts = file.split("/");
    let node = root;
    for (const folder of parts.slice(0, -1)) {
      if (!node.directories.has(folder)) node.directories.set(folder, { directories: new Map(), files: [] });
      node = node.directories.get(folder);
    }
    node.files.push(file);
  }
  return root;
}

function renderPreview() {
  const tree = byId("file-tree");
  tree.replaceChildren();
  let total = 0;
  let selected = 0;
  let unavailable = 0;
  for (const rootPath of selectedSources) {
    const files = previewRoots.get(rootPath) || [];
    const skipped = previewSkipped.get(rootPath) || [];
    const oversized = skipped.filter((item) => item.reason === "larger than the file-size limit" && Number(item.size_bytes) > 0);
    const offered = new Set([...files, ...oversized.map((item) => item.path)]);
    const chosen = checkedFiles.get(rootPath) || new Set();
    total += files.length + skipped.length;
    unavailable += skipped.length - oversized.length;
    selected += [...chosen].filter((file) => offered.has(file)).length;
    const holder = document.createElement("div");
    holder.className = "source-root";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "source-remove";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      selectedSources.splice(selectedSources.indexOf(rootPath), 1);
      previewRoots.delete(rootPath);
      previewSkipped.delete(rootPath);
      checkedFiles.delete(rootPath);
      renderPreview();
    });
    const reasons = Object.fromEntries(skipped.map((item) => [
      item.path,
      item.reason === "larger than the file-size limit"
        ? "Above current limit · check to include"
        : item.reason,
    ]));
    const all = [...files, ...skipped.map((item) => item.path)];
    holder.append(branch(rootPath, fileTree(all), all, chosen, 0, offered, reasons), remove);
    tree.append(holder);
  }
  byId("scope-count").textContent = `${selected} of ${total} files selected`
    + (unavailable ? ` · ${unavailable} unavailable` : "");
  byId("scope").classList.toggle("hidden", selectedSources.length === 0);
  updateFileSizeHint();
}

function saveModelFields() {
  if (!profile) return;
  profile[`chat_${activeBackend}`] = byId("chat-model").value.trim();
  profile[`embedding_${activeBackend}`] = byId("embedding-model").value.trim();
}

function compatibleChoices(role) {
  // Chat and embedding models need not share a family. Each must support its
  // own role, the chosen backend, and (for embedding) the query format.
  return modelChoices.filter((choice) => choice.role === role
    && choice.backend === activeBackend
    && (!choice.query_format || choice.query_format === byId("query-format").value));
}

function modelStatus(id, role) {
  const choice = modelChoices.find((item) => item.model_id === id);
  if (choice && !compatibleChoices(role).includes(choice)) {
    return "Not offered for this role, backend, or query format";
  }
  if (choice && cachedModels.has(id)) return "Cached locally · suggested for this role";
  if (choice) return "Not cached · open Models to download it";
  if (cachedModels.has(id)) return "Cached locally · compatibility not verified";
  if (id.startsWith("/") || id.startsWith(".") || /^[A-Za-z]:[\\/]/.test(id)) {
    return "Local path · checked when preparing";
  }
  return "Custom model · checked when preparing";
}

function refreshModelStatus() {
  for (const [field, role] of [["chat-model", "chat"], ["embedding-model", "embedding"]]) {
    byId(`${field}-status`).textContent = modelStatus(byId(field).value.trim(), role);
  }
  const hasBoth = ["chat", "embedding"].every((role) =>
    compatibleChoices(role).some((choice) => cachedModels.has(choice.model_id)));
  byId("models-hint").classList.toggle("hidden", hasBoth);
  byId("open-models").classList.toggle("models-needed", !hasBoth);
}

function renderModelsPanel() {
  const host = byId("model-list");
  host.replaceChildren();
  const installed = installedBackends[activeBackend];
  byId("models-backend").textContent = installed
    ? `Showing suggested ${activeBackend.toUpperCase()} models. Change the backend in Advanced settings to see its choices.`
    : `${activeBackend.toUpperCase()} packages are not installed. Run uv sync --extra gui --extra ${activeBackend} from the QAChat folder first.`;
  for (const [role, title] of [["chat", "Language models"], ["embedding", "Embedding models"]]) {
    const group = document.createElement("section");
    group.className = "model-group";
    const heading = document.createElement("h3");
    heading.textContent = title;
    group.append(heading);
    for (const choice of compatibleChoices(role)) {
      const row = document.createElement("div");
      row.className = "model-row";
      const description = document.createElement("div");
      const name = document.createElement("div");
      name.textContent = choice.model_id;
      const state = document.createElement("small");
      state.textContent = cachedModels.has(choice.model_id)
        ? "Cached locally · checked when the project opens"
        : "Not cached · download only if you choose to";
      description.append(name, state);
      const action = document.createElement("button");
      action.className = cachedModels.has(choice.model_id) ? "secondary" : "primary";
      action.type = "button";
      action.textContent = cachedModels.has(choice.model_id)
        ? (ready ? "Ready" : "Use") : "Download";
      action.disabled = busy || !installed || (ready && cachedModels.has(choice.model_id));
      action.addEventListener("click", () => {
        if (cachedModels.has(choice.model_id)) {
          const field = role === "chat" ? "chat-model" : "embedding-model";
          byId(field).value = choice.model_id;
          byId(field).dispatchEvent(new Event("input"));
          saveModelFields();
          byId("model-download-status").textContent = `${choice.model_id} selected for this project.`;
        } else {
          downloadingModel = choice.model_id;
          byId("model-download-status").textContent = `Downloading ${choice.model_id}…`;
          byId("cancel-model-download").classList.remove("hidden");
          setBusy(true, `Downloading ${choice.model_id}…`);
          bridge.downloadModel(choice.model_id);
        }
      });
      row.append(description, action);
      group.append(row);
    }
    if (group.children.length === 1) {
      const empty = document.createElement("p");
      empty.className = "hint";
      empty.textContent = "No suggested models for this query format. Choose Qwen3 or enter a custom model in Advanced settings.";
      group.append(empty);
    }
    host.append(group);
  }
  const other = [...cachedModels].filter((id) =>
    !modelChoices.some((choice) => choice.model_id === id)).sort();
  byId("other-model-list").replaceChildren(...other.map((id) => {
    const item = document.createElement("div");
    item.className = "other-model-id";
    item.textContent = id;
    return item;
  }));
  if (!other.length) byId("other-model-list").textContent = "No other cached models.";
}

function openModels() {
  renderModelsPanel();
  byId("models-panel").classList.remove("hidden");
  byId("close-models").focus();
}

function closeModelMenus() {
  for (const id of ["chat-model", "embedding-model"]) {
    byId(`${id}-menu`).classList.add("hidden");
    byId(`${id}-options`).setAttribute("aria-expanded", "false");
  }
}

function toggleModelMenu(id) {
  const menu = byId(`${id}-menu`);
  const shouldOpen = menu.classList.contains("hidden");
  closeModelMenus();
  if (!shouldOpen) return;

  const role = id === "chat-model" ? "chat" : "embedding";
  const options = compatibleChoices(role).filter((choice) =>
    cachedModels.has(choice.model_id)).map((choice) => {
    const option = document.createElement("button");
    option.type = "button";
    option.setAttribute("role", "option");
    option.textContent = choice.model_id;
    option.addEventListener("click", () => {
      byId(id).value = choice.model_id;
      byId(id).dispatchEvent(new Event("input"));
      closeModelMenus();
    });
    return option;
  });
  if (!options.length) {
    const message = document.createElement("small");
    message.className = "hint";
    message.textContent = "No suggested model cached. Open Models to download one.";
    options.push(message);
  }
  menu.replaceChildren(...options);
  menu.classList.remove("hidden");
  byId(`${id}-options`).setAttribute("aria-expanded", "true");
}

function updateFileSizeHint() {
  const bytes = Number(byId("max-file-bytes").value);
  if (!Number.isFinite(bytes) || bytes < 1) {
    byId("file-size-hint").textContent = "Enter a positive file-size limit.";
    return;
  }
  const mib = (bytes / (1024 * 1024)).toLocaleString(undefined, { maximumFractionDigits: 1 });
  const selectedBytes = selectedFileSizeLimit();
  const raised = selectedBytes > bytes
    ? ` Checked larger files will raise the saved limit to ${(selectedBytes / (1024 * 1024)).toFixed(1)} MiB.`
    : " Check a larger file to include it, or change this limit.";
  byId("file-size-hint").textContent = `Current limit: ${mib} MiB per file.${raised}`;
}

function selectedFileSizeLimit() {
  let limit = Number(byId("max-file-bytes").value);
  for (const rootPath of selectedSources) {
    const chosen = checkedFiles.get(rootPath) || new Set();
    for (const item of previewSkipped.get(rootPath) || []) {
      if (chosen.has(item.path) && item.reason === "larger than the file-size limit") {
        limit = Math.max(limit, Number(item.size_bytes));
      }
    }
  }
  return limit;
}

function showModelFields() {
  if (!profile) return;
  activeBackend = byId("backend").value === "auto" ? automaticBackend : byId("backend").value;
  byId("chat-model").value = profile[`chat_${activeBackend}`];
  byId("embedding-model").value = profile[`embedding_${activeBackend}`];
  refreshModelStatus();
  if (!byId("models-panel").classList.contains("hidden")) renderModelsPanel();
}

function showSettings(data) {
  profile = data.profile;
  automaticBackend = data.backend;
  cachedModels = new Set(data.cached_models);
  modelChoices = data.model_choices || [];
  installedBackends = data.installed_backends || {};
  supportedExtensions = new Set(data.supported_extensions);
  closeModelMenus();
  byId("backend").value = profile.backend;
  byId("query-format").value = profile.query_format;
  byId("extensions").value = profile.extensions.filter((extension) => !supportedExtensions.has(extension)).join(", ");
  for (const key of ["max_file_bytes", "target_tokens", "max_tokens", "overlap_tokens", "temperature", "max_new_tokens"]) {
    byId(key.replaceAll("_", "-")).value = profile[key] ?? "";
  }
  // A new GUI project does not silently download model weights.
  byId("allow-download").checked = Boolean(data.workspace && profile.allow_download);
  byId("thinking").checked = profile.thinking;
  updateFileSizeHint();
  showModelFields();
  if (data.workspace) {
    byId("workspace-location").textContent = data.workspace;
    renderWorkspace(data.workspace_files);
  }
  if (selectedSources.length) previewSources();
}

function collectSettings() {
  // Preparation settings may require a model reload or index rebuild.
  // Answer-time controls are sent only with the question below.
  saveModelFields();
  return {
    backend: byId("backend").value,
    chat_mlx: profile.chat_mlx, chat_torch: profile.chat_torch,
    embedding_mlx: profile.embedding_mlx, embedding_torch: profile.embedding_torch,
    query_format: byId("query-format").value,
    extensions: [...new Set([...profile.extensions, ...fileTypes()])],
    max_file_bytes: selectedFileSizeLimit(),
    target_tokens: Number(byId("target-tokens").value),
    max_tokens: Number(byId("max-tokens").value),
    overlap_tokens: Number(byId("overlap-tokens").value),
    allow_download: byId("allow-download").checked,
  };
}

function renderWorkspace(paths) {
  const host = byId("workspace-tree");
  host.replaceChildren();
  const root = fileTree(paths);
  const selected = new Set(paths);
  for (const [name, node] of [...root.directories].sort()) {
    const item = branch(name, node, allFiles(node), selected);
    item.querySelectorAll("input").forEach((input) => input.remove());
    host.append(item);
  }
  for (const file of root.files) {
    const line = document.createElement("div");
    line.className = "tree-file";
    line.textContent = file;
    host.append(line);
  }
}

function showError(message, logPath = "") {
  error.textContent = logPath ? `${message} Details: ${logPath}` : message;
  error.classList.remove("hidden");
  if (downloadingModel) {
    byId("model-download-status").textContent = error.textContent;
    byId("cancel-model-download").classList.add("hidden");
    downloadingModel = null;
  }
  setBusy(false);
  requestAnimationFrame(() => error.scrollIntoView({ block: "nearest" }));
}

function setBusy(value, message = "") {
  busy = value;
  byId("open").disabled = value || !bridge;
  byId("send").disabled = value || !ready;
  byId("new-chat").disabled = value || !ready;
  byId("change-project").disabled = value;
  byId("stop").disabled = !value;
  activity.classList.toggle("hidden", !value);
  byId("activity-text").textContent = message;
  byId("connection").textContent = value
    ? (message.startsWith("Scanning") ? "Scanning sources…"
      : chat.classList.contains("hidden") ? "Preparing project…" : "Answering…")
    : (ready ? "Ready" : "Choose a project");
  if (value) {
    byId("activity-percent").textContent = "Working…";
    byId("activity-detail").textContent = "";
  }
  if (!byId("models-panel").classList.contains("hidden")) renderModelsPanel();
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

function clearSourcePreview() {
  activeReference = null;
  byId("source-preview").classList.add("hidden");
  byId("source-preview-lines").replaceChildren();
}

function clearReferences() {
  selectedTurn?.classList.remove("selected");
  selectedTurn = null;
  byId("references-for").textContent = "Select an answer to see its sources.";
  byId("references-list").replaceChildren();
  clearSourcePreview();
}

function openReference(item, source) {
  byId("references-section").open = true;
  const turnIndex = Number(item.dataset.turnIndex);
  activeReference = { turnIndex, citation: Number(source.citation) };
  const preview = byId("source-preview");
  preview.classList.remove("hidden");
  byId("source-preview-title").textContent = source.path;
  byId("source-preview-note").textContent = "Loading nearby lines…";
  byId("source-preview-lines").replaceChildren();
  bridge?.previewReference(turnIndex, activeReference.citation);
  preview.scrollIntoView({ block: "nearest" });
}

function selectTurn(item, turn, citation = null) {
  selectedTurn?.classList.remove("selected");
  selectedTurn = item;
  item.classList.add("selected");
  clearSourcePreview();
  byId("references-for").textContent = turn.evidence?.length
    ? `Sources for: ${turn.question}`
    : "No sources saved for this answer.";
  const list = byId("references-list");
  list.replaceChildren();
  for (const source of turn.evidence || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "reference-item";
    button.dataset.citation = String(source.citation);
    const number = document.createElement("span");
    number.className = "reference-number";
    number.textContent = String(source.citation);
    const path = document.createElement("span");
    path.className = "reference-path";
    path.textContent = `${source.path}:${source.start_line}-${source.end_line}`;
    button.append(number, path);
    button.addEventListener("click", () => openReference(item, source));
    list.append(button);
  }
  if (citation !== null) {
    const source = (turn.evidence || []).find((entry) => Number(entry.citation) === citation);
    if (source) openReference(item, source);
  }
}
function renderSourcePreview(data) {
  if (!activeReference || data.turn_index !== activeReference.turnIndex
      || data.citation !== activeReference.citation) return;
  byId("source-preview-title").textContent = data.path;
  const location = data.page
    ? `Page ${data.page} · lines ${data.start_line}–${data.end_line}`
    : `Lines ${data.start_line}–${data.end_line}`;
  byId("source-preview-note").textContent = data.note
    ? `${location} · ${data.note}` : location;
  const host = byId("source-preview-lines");
  host.replaceChildren();
  for (const line of data.lines) {
    const row = document.createElement("span");
    row.className = line.highlighted ? "source-line highlighted" : "source-line";
    const number = document.createElement("span");
    number.className = "source-line-number";
    number.textContent = String(line.number);
    const content = document.createElement("span");
    content.className = "source-line-text";
    content.textContent = line.text || " ";
    row.append(number, content);
    host.append(row);
  }
  byId("source-preview").scrollIntoView({ block: "nearest" });
}


function renderAnswer(answer, turn, item) {
  // Keep the model's text as text nodes. Only evidence numbers become links.
  const citations = new Set((turn.evidence || []).map((source) => Number(source.citation)));
  const source = String(turn.answer || "");
  const markers = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  let start = 0;
  for (const match of source.matchAll(markers)) {
    const numbers = match[1].split(",").map((value) => Number(value.trim()));
    if (!numbers.every((number) => citations.has(number))) continue;
    answer.append(document.createTextNode(source.slice(start, match.index).replace(/[ \t]+$/, "")));
    const superscript = document.createElement("sup");
    superscript.className = "answer-citation";
    numbers.forEach((number, index) => {
      if (index) superscript.append(document.createTextNode(","));
      const link = document.createElement("button");
      link.type = "button";
      link.textContent = String(number);
      link.setAttribute("aria-label", `Show reference ${number}`);
      link.addEventListener("click", (event) => {
        event.stopPropagation();
        showWorkspace(true);
        selectTurn(item, turn, number);
      });
      superscript.append(link);
    });
    answer.append(superscript);
    start = match.index + match[0].length;
  }
  answer.append(document.createTextNode(source.slice(start)));
}

function addTurn(turn) {
  const item = document.createElement("article");
  item.className = "turn";
  item.dataset.turnIndex = String(conversation.children.length);
  const question = document.createElement("div");
  question.className = "user-message";
  question.textContent = turn.question;
  const answer = document.createElement("div");
  answer.className = "answer-message";
  answer.tabIndex = 0;
  answer.setAttribute("aria-label", "Select this answer to view its references");
  answer.addEventListener("click", () => {
    showWorkspace(true);
    selectTurn(item, turn);
  });
  answer.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      showWorkspace(true);
      selectTurn(item, turn);
    }
  });
  renderAnswer(answer, turn, item);
  item.append(question, answer);
  conversation.append(item);
  selectTurn(item, turn);
  conversation.scrollTop = conversation.scrollHeight;
}

function onEvent(raw) {
  const data = JSON.parse(raw);
  if (data.event === "settings") {
    showSettings(data);
  } else if (data.event === "saved_project") {
    // Ignore a reply for a selection that the user has already changed.
    if (data.kind !== byId("kind").value
        || data.workspace !== byId("workspace-path").value.trim()) return;
    const resume = byId("resume-recent");
    resume.dataset.project = data.project || "";
    if (data.project) {
      resume.textContent = `Resume ${data.project.split(/[\\/]/).pop()}`;
    }
    byId("recent-project").classList.toggle("hidden", !data.project);
  } else if (data.event === "preview") {
    data.roots.forEach((root, index) => {
      const original = selectedSources[index];
      const previous = checkedFiles.get(original) || checkedFiles.get(root.path);
      selectedSources[index] = root.path;
      if (original !== root.path) {
        previewRoots.delete(original);
        checkedFiles.delete(original);
      }
      previewRoots.set(root.path, root.files);
      previewSkipped.set(root.path, root.skipped || []);
      const offered = [
        ...root.files,
        ...(root.skipped || []).filter((item) => item.reason === "larger than the file-size limit").map((item) => item.path),
      ];
      checkedFiles.set(
        root.path,
        previous ? new Set(offered.filter((file) => previous.has(file))) : new Set(root.files),
      );
    });
    renderPreview();
    setBusy(false);
  } else if (data.event === "status") {
    setBusy(true, data.message);
  } else if (data.event === "model_download_started") {
    byId("model-download-status").textContent = `Downloading ${data.model_id}…`;
  } else if (data.event === "model_downloaded") {
    cachedModels.add(data.model_id);
    downloadingModel = null;
    byId("cancel-model-download").classList.add("hidden");
    byId("model-download-status").textContent = `${data.model_id} is cached locally. Choose Use to select it.`;
    setBusy(false);
    refreshModelStatus();
  } else if (data.event === "progress") {
    showProgress(data);
  } else if (data.event === "opened") {
    ready = true;
    selectedSources.length = 0;
    previewRoots.clear();
    previewSkipped.clear();
    checkedFiles.clear();
    renderPreview();
    setup.classList.add("hidden");
    chat.classList.remove("hidden");
    conversation.replaceChildren();
    clearReferences();
    data.turns.forEach(addTurn);
    byId("workspace-path").value = data.project;
    byId("workspace-location").textContent = data.project;
    renderWorkspace(data.workspace_files);
    showWorkspace(true);
    byId("retrieval-count").value = data.retrieval_count;
    byId("temperature").value = data.temperature;
    byId("max-new-tokens").value = data.max_new_tokens ?? "";
    byId("thinking").checked = data.thinking;
    byId("project-name").textContent = data.project.split(/[\\/]/).pop();
    byId("project-detail").textContent = `${data.kind} · ${data.project} · index ${data.index_status} · ${data.turns.length} saved turns`;
    setBusy(false);
  } else if (data.event === "reference_preview") {
    renderSourcePreview(data);
  } else if (data.event === "reference_error") {
    if (activeReference && data.turn_index === activeReference.turnIndex
        && data.citation === activeReference.citation) {
      byId("source-preview-note").textContent = data.message;
    }
  } else if (data.event === "answered") {
    addTurn(data.turn);
    byId("question").value = "";
    setBusy(false);
  } else if (data.event === "reset") {
    conversation.replaceChildren();
    clearReferences();
    byId("project-detail").textContent = "New conversation · index retained";
    setBusy(false);
  } else if (data.event === "stopped") {
    if (downloadingModel) {
      byId("model-download-status").textContent = "Download stopped. Reopen Models to check what is cached.";
      downloadingModel = null;
      byId("cancel-model-download").classList.add("hidden");
    }
    ready = false;
    chat.classList.add("hidden");
    setup.classList.remove("hidden");
    showWorkspace(false);
    clearReferences();
    byId("connection").textContent = "Stopped · reopen your project";
    setBusy(false);
    bridge.requestSettings(byId("kind").value, byId("workspace-path").value.trim());
  } else if (data.event === "backend_failed") {
    ready = false;
    chat.classList.add("hidden");
    setup.classList.remove("hidden");
    showWorkspace(false);
    clearReferences();
    showError(data.message, data.log_path);
  } else if (data.event === "error") {
    showError(data.message, data.log_path);
  }
}
byId("close-source-preview").addEventListener("click", clearSourcePreview);

byId("browse-workspace").addEventListener("click", () => bridge?.chooseWorkspace());
const dropZone = byId("drop-zone");
dropZone.addEventListener("click", () => bridge?.chooseSources());
dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  const paths = Array.from(event.dataTransfer.files, (file) => file.path).filter(Boolean);
  addSources(paths);
});
byId("stop").addEventListener("click", () => {
  if (!busy || !bridge) return;
  byId("stop").disabled = true;
  byId("activity-text").textContent = "Stopping…";
  bridge.stopOperation();
});
byId("open-models").addEventListener("click", openModels);
byId("resume-recent").addEventListener("click", () => {
  const project = byId("resume-recent").dataset.project;
  if (!project || busy) return;
  setBusy(true, "Resuming your last project…");
  bridge.resumeProject(project);
});
byId("models-hint-button").addEventListener("click", openModels);
byId("close-models").addEventListener("click", () => byId("models-panel").classList.add("hidden"));
byId("cancel-model-download").addEventListener("click", () => {
  if (downloadingModel) bridge?.stopOperation();
});
byId("project-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || !bridge) return;
  if (!profile) return showError("Settings are still loading. Try again in a moment.");
  if (!installedBackends[activeBackend]) {
    openModels();
    byId("model-download-status").textContent = `Install the ${activeBackend.toUpperCase()} packages before preparing this project.`;
    return;
  }
  for (const [field, role] of [["chat-model", "chat"], ["embedding-model", "embedding"]]) {
    const chosen = modelChoices.find((choice) => choice.model_id === byId(field).value.trim());
    if (chosen && !compatibleChoices(role).includes(chosen)) {
      byId("advanced").open = true;
      return showError(`The selected ${role} model does not match the backend or embedding query format.`);
    }
  }
  const localModel = (id) => cachedModels.has(id)
    || id.startsWith("/") || id.startsWith(".") || /^[A-Za-z]:[\\/]/.test(id);
  if (!byId("allow-download").checked
      && ![byId("chat-model").value.trim(), byId("embedding-model").value.trim()].every(localModel)) {
    openModels();
    byId("model-download-status").textContent = "Download both selected models before preparing, or explicitly allow downloads in Advanced settings.";
    return;
  }
  const selected = Object.fromEntries(
    selectedSources.map((path) => [path, [...(checkedFiles.get(path) || [])]]),
  );
  const included = selectedSources.filter((path) => selected[path].length);
  if (selectedSources.length && !included.length) return showError("Select at least one file.");
  error.classList.add("hidden");
  byId("advanced").open = false;
  setBusy(true, "Opening project…");
  requestAnimationFrame(() => activity.scrollIntoView({ block: "nearest" }));
  bridge.openProject(
    JSON.stringify(included), byId("kind").value, byId("workspace-path").value.trim(),
    JSON.stringify(selected), JSON.stringify(collectSettings()),
  );
});
byId("question-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = byId("question").value.trim();
  if (!question || busy || !ready) return;
  error.classList.add("hidden");
  const count = Number(byId("retrieval-count").value);
  if (!Number.isInteger(count) || count < 1) return showError("Choose a positive evidence count.");
  const temperature = Number(byId("temperature").value);
  if (!byId("temperature").value.trim() || !Number.isFinite(temperature) || temperature < 0) {
    return showError("Choose a temperature of zero or higher.");
  }
  const answerTokens = byId("max-new-tokens").value.trim();
  const maxNewTokens = answerTokens ? Number(answerTokens) : null;
  if (answerTokens && (!Number.isInteger(maxNewTokens) || maxNewTokens < 1)) {
    return showError("Choose a positive answer-token limit or leave it blank.");
  }
  setBusy(true, "Finding evidence and answering…");
  bridge.askQuestion(question, count, JSON.stringify({
    temperature, thinking: byId("thinking").checked, max_new_tokens: maxNewTokens,
  }));
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
  if (busy) return;
  bridge.closeProject();
  ready = false;
  chat.classList.add("hidden");
  setup.classList.remove("hidden");
  byId("workspace-path").value = "";
  selectedSources.length = 0;
  previewRoots.clear();
  previewSkipped.clear();
  checkedFiles.clear();
  renderPreview();
  showWorkspace(false);
  clearReferences();
  bridge.requestSettings(byId("kind").value, "");
  setBusy(false);
  refreshSavedProject();
});
byId("toggle-workspace").addEventListener("click", () => {
  showWorkspace(!shell.classList.contains("sidebar-open"));
});
byId("kind").addEventListener("change", () => {
  hideResume();
  bridge?.requestSettings(byId("kind").value, byId("workspace-path").value.trim());
  refreshSavedProject();
});
byId("workspace-path").addEventListener("input", queueSavedProjectRefresh);
byId("workspace-path").addEventListener("change", () => {
  bridge?.requestSettings(byId("kind").value, byId("workspace-path").value.trim());
  refreshSavedProject();
});
byId("backend").addEventListener("change", () => { saveModelFields(); showModelFields(); });
byId("query-format").addEventListener("change", () => {
  refreshModelStatus();
  if (!byId("models-panel").classList.contains("hidden")) renderModelsPanel();
});
for (const id of ["chat-model", "embedding-model"]) {
  byId(`${id}-options`).addEventListener("click", () => toggleModelMenu(id));
  byId(id).addEventListener("input", () => {
    refreshModelStatus();
  });
}
document.addEventListener("click", (event) => {
  if (!event.target.closest(".model-picker")) closeModelMenus();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeModelMenus();
});
byId("max-file-bytes").addEventListener("input", updateFileSizeHint);
for (const id of ["extensions", "max-file-bytes"]) byId(id).addEventListener("change", previewSources);

new QWebChannel(qt.webChannelTransport, (channel) => {
  bridge = channel.objects.qachat;
  bridge.event.connect(onEvent);
  bridge.pathsSelected.connect(addSources);
  bridge.workspaceSelected.connect((path) => {
    byId("workspace-path").value = path;
    hideResume();
    bridge.requestSettings(byId("kind").value, path);
    refreshSavedProject();
  });
  bridge.requestSettings(byId("kind").value, byId("workspace-path").value.trim());
  showWorkspace(false);
  setBusy(false);
  refreshSavedProject();
});

window.addEventListener("focus", refreshSavedProject);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshSavedProject();
});
// Also catch files removed while the project chooser stays in the foreground.
window.setInterval(refreshSavedProject, 3000);
