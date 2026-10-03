/**
 * Present the local QAChat interface inside Qt WebEngine.
 *
 * The page owns transient UI state and delegates filesystem/model operations
 * through QWebChannel. Engine events update project, conversation, evidence,
 * and progress displays. Model answers and source text are inserted as text,
 * not interpreted as HTML. Source selection is a preview, not a filesystem
 * security boundary. Stop terminates the engine rather than pausing inference.
 *
 * Completed answers arrive as complete strings. The character-by-character
 * reveal is a presentation timer, not streamed model output. Conversation
 * persistence and tracing remain backend responsibilities.
 */

"use strict";

// Resolve fixed DOM identifiers after the deferred script can see the page.
const byId = (id) => document.getElementById(id);
const setup = byId("setup");
const chat = byId("chat");
const conversation = byId("conversation");
const conversationScroll = byId("conversation-scroll");
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
let requiredModels = [];
let readinessEnvironmentReady = false;
let supportedExtensions = new Set();
let automaticBackend = "mlx";
let activeBackend = "mlx";
let resumeRefreshTimer = null;
let activeTracePath = "";
let activeConversationId = "";
let pendingTurn = null;
let phaseTimer = null;
let answerTimer = null;
let selectedEffort = "lagom";
let resumeAfterStop = false;
let activeProjectKind = "code";
let lastEmptyPrompt = { code: -1, literature: -1 };
let messageAttachments = [];
let currentTurnCount = 0;
const effortLevels = [
  { key: "light", name: "Light", detail: "0.6B model · 5 references" },
  { key: "instant", name: "Instant", detail: "0.6B model · 20 references" },
  { key: "lagom", name: "Lagom", detail: "14B model · 20 references" },
  { key: "thorough", name: "Thorough", detail: "14B model · 50 references" },
  { key: "deep", name: "Deep", detail: "27B model · 50 references" },
];
const emptyChatContent = {
  code: {
    prompts: [
      "What would you like to understand about this codebase?",
      "Which part of your project should we explore?",
      "What do you want to trace through the code?",
      "What behavior would you like to investigate?",
      "Where should we start in this repository?",
    ],
    suggestions: [
      "Explain a concept", "Find where a feature is implemented",
      "Trace a function through the code", "Compare two approaches",
      "Find evidence across my files", "Explain how modules connect",
      "Identify code related to an error", "Summarize an architecture decision",
      "Find tests for a behavior",
    ],
  },
  literature: {
    prompts: [
      "What would you like to understand from your literature?",
      "Which concept should we explore across these sources?",
      "What evidence are you looking for?",
      "What would you like to compare in the literature?",
      "Which finding should we investigate?",
    ],
    suggestions: [
      "Explain a concept", "Find evidence across my files",
      "Compare two approaches", "Summarize findings on a topic",
      "Find supporting and conflicting evidence", "Trace a claim to its sources",
      "Compare methods across papers", "Identify gaps in the evidence",
      "Find how a term is defined",
    ],
  },
};

function turnCountLabel(count) {
  return `${count} ${count === 1 ? "turn" : "turns"}`;
}

function updateProjectSummary(count = currentTurnCount) {
  currentTurnCount = Number(count) || 0;
  const projectName = byId("project-name").textContent.trim();
  byId("topbar-project-name").textContent = projectName;
  byId("topbar-turn-count").textContent = turnCountLabel(currentTurnCount);
  byId("project-detail").textContent = `${currentTurnCount} saved ${currentTurnCount === 1 ? "turn" : "turns"}`;
}

function updateFloatingHeader() {
  const scrolled = ready && conversationScroll.scrollTop > 96;
  document.querySelector(".topbar").classList.toggle("floating", scrolled);
  byId("topbar-project").classList.toggle("hidden", !scrolled);
}

/**
 * Hide and clear the previously offered saved-project shortcut.
 */
function hideResume() { byId("saved-projects").classList.add("hidden"); }

/**
 * Ask the engine for a saved project only on an idle setup screen.
 */
function refreshSavedProject() {
  if (!bridge || busy || ready) return;
  bridge.listSavedProjects(byId("workspace-path").value.trim());
}

/**
 * Invalidate the resume shortcut and debounce the next lookup.
 */
function queueSavedProjectRefresh() {
  hideResume();
  clearTimeout(resumeRefreshTimer);
  resumeRefreshTimer = setTimeout(refreshSavedProject, 250);
}

/**
 * Toggle the workspace panels and their control label.
 *
 * @param visible Whether the sidebar layout should be shown.
 */
function showWorkspace(visible) {
  // Keep the workspace out of the layout until a project is actually open.
  shell.classList.toggle("sidebar-open", visible);
  byId("toggle-workspace").classList.toggle("hidden", !ready);
  const control = byId("toggle-workspace");
  control.setAttribute("aria-label", visible ? "Hide side panels" : "Show side panels");
  control.title = control.getAttribute("aria-label");
  control.classList.toggle("panels-hidden", !visible);
}

function showConversations(visible) {
  shell.classList.toggle("conversations-hidden", !visible);
  const control = byId("toggle-conversations");
  control.classList.toggle("hidden", !ready);
  control.classList.toggle("panels-hidden", !visible);
  control.setAttribute("aria-label", visible ? "Hide conversations" : "Show conversations");
  control.title = control.getAttribute("aria-label");
}

function applyAppearance(value) {
  const selected = ["system", "light", "dark"].includes(value) ? value : "system";
  document.documentElement.dataset.appearance = selected;
  localStorage.setItem("qachat-appearance", selected);
  if (byId("appearance")) byId("appearance").value = selected;
}

function renderSavedProjects(projects) {
  const host = byId("saved-projects");
  host.replaceChildren();
  const groups = [
    ["code", "Resume previous code projects"],
    ["literature", "Resume previous literature projects"],
  ];
  for (const [kind, title] of groups) {
    const matches = projects.filter((project) => project.kind === kind);
    if (!matches.length) continue;
    const group = document.createElement("div");
    group.className = "saved-project-group";
    const heading = document.createElement("strong");
    heading.textContent = title;
    group.append(heading);
    for (const project of matches) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `Resume ${project.name}`;
      button.title = `Resume ${project.kind} project from ${project.project}`;
      button.addEventListener("click", () => {
        if (busy) return;
        setBusy(true, "Resuming project…");
        bridge.resumeProject(project.project);
      });
      group.append(button);
    }
    host.append(group);
  }
  host.classList.toggle("hidden", !host.children.length);
}

/**
 * Normalize and deduplicate source paths, then request a preview.
 *
 * @param paths Local paths provided by a native dialog or drag-and-drop.
 */
function addSources(paths) {
  for (const path of paths) {
    const clean = String(path).trim();
    if (clean && !selectedSources.includes(clean)) selectedSources.push(clean);
  }
  previewSources();
}

/**
 * Return nonempty, lowercase custom suffixes from the settings field.
 * @returns Nonempty custom suffix strings.
 */
function fileTypes() {
  return byId("extensions").value.split(",").map((value) => value.trim().toLowerCase()).filter(Boolean);
}

/**
 * Request a backend metadata scan for the current source selection.
 */
function previewSources() {
  if (bridge && selectedSources.length) {
    setBusy(true, "Scanning selected sources…");
    bridge.previewSources(
      JSON.stringify(selectedSources), JSON.stringify(fileTypes()),
      Number(byId("max-file-bytes").value), byId("workspace-path").value.trim(),
    );
  }
}

/**
 * Build a nested selection tree with tri-state folder checkboxes.
 *
 * @param name Displayed folder label.
 * @param children Tree node with directory children and file paths.
 * @param paths Descendant paths used to compute folder selection.
 * @param selection Mutable Set of checked paths.
 * @param depth Nesting depth; the top-level branch starts expanded.
 * @param enabled Allowed paths, or null to enable every path.
 * @param reasons Path-to-explanation mapping for skipped entries.
 * @returns New details element containing the branch.
 */
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
    row.dataset.path = file;
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

/**
 * Flatten a tree node into its descendant file paths.
 *
 * @param node Node containing files and a Map of directories.
 * @returns Flat array of descendant paths.
 */
function allFiles(node) {
  return [...node.files, ...[...node.directories.values()].flatMap(allFiles)];
}

/**
 * Group forward-slash-separated file paths into directory nodes.
 *
 * @param files Relative file paths to group; the input is not modified.
 * @returns Root tree node.
 */
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

/**
 * Rebuild source trees, counts, and removal controls from preview state.
 */
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
}

/**
 * Retain the current backend model fields in the local profile object.
 */
function saveModelFields() {
  if (!profile) return;
  profile[`chat_${activeBackend}`] = byId("chat-model").value.trim();
  profile[`embedding_${activeBackend}`] = byId("embedding-model").value.trim();
}

/**
 * Filter presets by role, backend, and any required query format.
 *
 * @param role Model role to show: chat or embedding.
 * @returns Matching model-preset records.
 */
function compatibleChoices(role) {
  // Chat and embedding models need not share a family. Each must support its
  // own role, the chosen backend, and (for embedding) the query format.
  return modelChoices.filter((choice) => choice.role === role
    && choice.backend === activeBackend
    && (!choice.query_format || choice.query_format === byId("query-format").value));
}

/**
 * Show whether a compatible cached chat-and-embedding pair is known.
 */
function refreshModelStatus() {
  const hasBoth = ["chat", "embedding"].every((role) =>
    compatibleChoices(role).some((choice) => cachedModels.has(choice.model_id)));
  byId("models-hint").classList.add("hidden");
  byId("open-models").classList.toggle("models-needed", !hasBoth);
}

/**
 * Build suggested-model controls and the unclassified cache listing.
 */
function renderModelsPanel() {
  const host = byId("model-list");
  host.replaceChildren();
  const installed = installedBackends[activeBackend];
  byId("models-backend").textContent = installed
    ? `Showing suggested ${activeBackend.toUpperCase()} models. Change the backend in Settings to see its choices.`
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
      empty.textContent = "No suggested models for this query format. Choose Qwen3 or enter a custom model in Settings.";
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

/**
 * Render and open the models panel, then focus its close control.
 */
function openModels() {
  renderModelsPanel();
  byId("models-panel").classList.remove("hidden");
  byId("close-models").focus();
}

/**
 * Hide both model suggestion menus and reset their accessibility state.
 */
function closeModelMenus() {
  for (const id of ["chat-model", "embedding-model"]) {
    byId(`${id}-menu`).classList.add("hidden");
    byId(`${id}-options`).setAttribute("aria-expanded", "false");
  }
}

/**
 * Toggle cached choices for a model field without loading a model.
 *
 * @param id Model input identifier: chat-model or embedding-model.
 */
function toggleModelMenu(id) {
  const menu = byId(`${id}-menu`);
  const shouldOpen = menu.classList.contains("hidden");
  closeModelMenus();
  if (!shouldOpen) return;

  const role = id === "chat-model" ? "chat" : "embedding";
  const modelIds = role === "chat"
    ? [...cachedModels].sort((left, right) => left.localeCompare(right))
    : compatibleChoices(role)
      .filter((choice) => cachedModels.has(choice.model_id))
      .map((choice) => choice.model_id)
      .sort((left, right) => left.localeCompare(right));
  const options = modelIds.map((modelId) => {
    const option = document.createElement("button");
    option.type = "button";
    option.setAttribute("role", "option");
    option.textContent = modelId;
    option.addEventListener("click", () => {
      byId(id).value = modelId;
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

/**
 * Return a size limit large enough for explicitly selected oversized files.
 * @returns Requested maximum file size in bytes.
 */
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

/**
 * Resolve the selected backend and restore its profile model fields.
 */
function showModelFields() {
  if (!profile) return;
  activeBackend = byId("backend").value === "auto" ? automaticBackend : byId("backend").value;
  byId("chat-model").value = profile[`chat_${activeBackend}`];
  byId("embedding-model").value = profile[`embedding_${activeBackend}`];
  refreshModelStatus();
  if (!byId("models-panel").classList.contains("hidden")) renderModelsPanel();
}

/**
 * Apply a settings response and refresh model and workspace controls.
 *
 * @param data Settings event with profile, catalog, cache, and backend
 *   metadata.
 */
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
  showModelFields();
  if (data.workspace) {
    byId("workspace-location").textContent = data.workspace;
    renderWorkspace(data.workspace_files);
  }
  if (selectedSources.length) previewSources();
}

/**
 * Collect preparation settings and selected file-type/size adjustments.
 * @returns Preparation settings object for the engine.
 */
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

/**
 * Render the saved workspace file tree without selection checkboxes.
 *
 * @param paths Workspace-relative file paths reported by the backend.
 */
function renderWorkspace(paths) {
  const host = byId("workspace-tree");
  host.replaceChildren();
  const root = fileTree(paths);
  const selected = new Set(paths);
  for (const [name, node] of [...root.directories].sort()) {
    const item = branch(name, node, allFiles(node), selected);
    item.querySelectorAll("input").forEach((input) => input.remove());
    item.querySelectorAll(".tree-file").forEach((line) => {
      line.tabIndex = 0;
      line.setAttribute("role", "button");
      line.addEventListener("click", () => openWorkspaceFile(line.dataset.path));
      line.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openWorkspaceFile(line.dataset.path);
        }
      });
    });
    host.append(item);
  }
  for (const file of root.files) {
    const line = document.createElement("div");
    line.className = "tree-file";
    line.textContent = file;
    line.tabIndex = 0;
    line.setAttribute("role", "button");
    line.addEventListener("click", () => openWorkspaceFile(file));
    host.append(line);
  }
}

function openWorkspaceFile(path) {
  if (!path || !bridge || !confirm(`Open ${path} with its default application?`)) return;
  bridge.openWorkspaceFile(path);
}

function showError(message, tracePath = "") {
  stopPhaseAnimation();
  if (pendingTurn) {
    pendingTurn.answer.textContent = "Could not complete this answer.";
    pendingTurn.item.classList.add("failed");
    pendingTurn = null;
  }
  error.textContent = tracePath ? `${message} Trace: ${tracePath}` : message;
  error.classList.remove("hidden");
  if (downloadingModel) {
    byId("model-download-status").textContent = error.textContent;
    byId("cancel-model-download").classList.add("hidden");
    downloadingModel = null;
  }
  setBusy(false);
  if (!byId("first-run-setup").classList.contains("hidden")) {
    byId("install-required").disabled = !readinessEnvironmentReady;
  }
  byId("connection").textContent = "Error";
  requestAnimationFrame(() => error.scrollIntoView({ block: "nearest" }));
}

/**
 * Gate operation controls and update the visible activity state.
 *
 * @param value Whether a backend operation is active.
 * @param message Activity label; the default empty string clears the old
 *   text.
 */
function setBusy(value, message = "") {
  busy = value;
  if (!value) activity.classList.remove("answer-progress");
  byId("open").disabled = value || !bridge;
  byId("send").disabled = !ready;
  byId("send").classList.toggle("stopping", value && ready);
  byId("send").querySelector("span").textContent = value && ready ? "■" : "↑";
  byId("send").setAttribute(
    "aria-label", value && ready ? "Stop generation" : "Send message",
  );
  byId("send").title = byId("send").getAttribute("aria-label");
  byId("new-chat").disabled = value || !ready;
  byId("branch-chat").disabled = value || !ready;
  document.querySelectorAll(".conversation-item, .conversation-delete").forEach((item) => { item.disabled = value || !ready; });
  byId("change-project").disabled = value;
  byId("stop").disabled = !value;
  activity.classList.toggle("hidden", !value);
  byId("activity-text").textContent = message;
  byId("connection").textContent = value ? "Busy" : (ready || bridge ? "Ready" : "Init");
  if (value) {
    byId("activity-percent").textContent = "Working…";
    byId("activity-detail").textContent = "";
  }
  if (!byId("models-panel").classList.contains("hidden")) renderModelsPanel();
}

/**
 * Format a numeric duration for display in seconds or minutes.
 *
 * @param value Elapsed or estimated seconds.
 * @returns Formatted duration text.
 */
function seconds(value) {
  const rounded = Math.max(0, Math.round(value));
  const minutes = Math.floor(rounded / 60);
  const remainder = rounded % 60;
  return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

/**
 * Render backend progress, elapsed time, and optional remaining time.
 *
 * @param data Progress event; percentages are presentation values, not
 *   guarantees.
 */
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

/**
 * Remove the active reference preview and reset its selection state.
 */
function clearSourcePreview() {
  activeReference?.button?.classList.remove("active");
  activeReference?.holder?.classList.remove("expanded");
  activeReference?.preview?.remove();
  activeReference = null;
}

/**
 * Clear selected-turn and reference presentation state.
 */
function clearReferences() {
  selectedTurn?.classList.remove("selected");
  selectedTurn = null;
  byId("references-for").textContent = "Select an answer to see its sources.";
  byId("references-list").replaceChildren();
  clearSourcePreview();
}

/**
 * Create a source-preview container and return its DOM references.
 * @returns Object containing the preview DOM references.
 */
function makeSourcePreview() {
  const preview = document.createElement("section");
  preview.className = "source-preview";
  preview.setAttribute("aria-label", "Cited source preview");
  const heading = document.createElement("header");
  heading.className = "source-preview-heading";
  const note = document.createElement("p");
  note.className = "hint";
  const close = document.createElement("button");
  close.type = "button";
  close.className = "sidebar-toggle";
  close.textContent = "Close";
  const lines = document.createElement("pre");
  lines.className = "source-preview-lines";
  heading.append(note, close);
  preview.append(heading, lines);
  return { preview, close, note, lines };
}

/**
 * Toggle a citation preview or request its source lines from the engine.
 *
 * @param item Conversation article containing the answer.
 * @param source Evidence record carrying a citation number and provenance.
 * @param button Reference button whose active state is updated.
 * @param holder Container in which to place the preview.
 */
function openReference(item, source, button, holder) {
  byId("references-section").open = true;
  const turnIndex = Number(item.dataset.turnIndex);
  const citation = Number(source.citation);
  if (activeReference?.turnIndex === turnIndex && activeReference?.citation === citation) {
    clearSourcePreview();
    return;
  }
  clearSourcePreview();
  const nodes = makeSourcePreview();
  nodes.note.textContent = "Loading nearby lines…";
  nodes.close.addEventListener("click", clearSourcePreview);
  holder.append(nodes.preview);
  holder.classList.add("expanded");
  button.classList.add("active");
  activeReference = { turnIndex, citation, button, holder, ...nodes };
  bridge?.previewReference(turnIndex, activeReference.citation);
  nodes.preview.scrollIntoView({ block: "nearest" });
}

/**
 * Select an answer and render its evidence list, optionally opening a
 * citation.
 *
 * @param item Conversation article to mark as selected.
 * @param turn Saved or newly completed question/answer/evidence record.
 * @param citation Optional evidence number to open immediately.
 */
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
    const holder = document.createElement("div");
    holder.className = "reference-entry";
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
    button.addEventListener("click", () => openReference(item, source, button, holder));
    holder.append(button);
    list.append(holder);
  }
  if (citation !== null) {
    const source = (turn.evidence || []).find((entry) => Number(entry.citation) === citation);
    const button = list.querySelector(`[data-citation="${citation}"]`);
    if (source && button) openReference(item, source, button, button.parentElement);
  }
}
/**
 * Render numbered source lines only for the currently active reference.
 *
 * @param data Reference response with matching turn/citation identifiers
 *   and lines.
 */
function renderSourcePreview(data) {
  if (!activeReference || data.turn_index !== activeReference.turnIndex
      || data.citation !== activeReference.citation) return;
  const location = data.page
    ? `Page ${data.page} · lines ${data.start_line}–${data.end_line}`
    : `Lines ${data.start_line}–${data.end_line}`;
  activeReference.note.textContent = data.note
    ? `${location} · ${data.note}` : location;
  const host = activeReference.lines;
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
  activeReference.preview.scrollIntoView({ block: "nearest" });
}


/**
 * Render answer text with links only for recognized numeric citations.
 *
 * @param answer Destination DOM element.
 * @param turn Answer text and its evidence records.
 * @param item Conversation article used when selecting a linked citation.
 */
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

/**
 * Append a completed turn, enable answer selection, and show its
 * references.
 *
 * @param turn Saved question, answer, and evidence data.
 */
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
  conversationScroll.scrollTop = conversationScroll.scrollHeight;
}

/**
 * Replace the conversation picker using the backend registry.
 *
 * @param data Event containing conversation records and the active
 *   identifier.
 */
function renderConversations(data) {
  activeConversationId = data.active_conversation_id || activeConversationId;
  const list = byId("conversation-list");
  list.replaceChildren();
  let activeName = "";
  let activeEmpty = false;
  for (const item of data.conversations || []) {
    if (item.conversation_id === activeConversationId) {
      activeName = item.name;
      activeEmpty = Boolean(item.empty);
    }
    if (item.empty) continue;
    const row = document.createElement("div");
    row.className = "conversation-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `conversation-item${item.conversation_id === activeConversationId ? " active" : ""}`;
    button.dataset.conversationId = item.conversation_id;
    button.textContent = item.name;
    button.title = item.name;
    button.addEventListener("click", () => {
      if (busy || item.conversation_id === activeConversationId) return;
      setBusy(true, "Opening conversation…");
      bridge.selectConversation(item.conversation_id);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "conversation-delete";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Delete ${item.name}`);
    remove.title = "Delete conversation";
    remove.addEventListener("click", () => {
      if (busy || !confirm(`Delete “${item.name}”?`)) return;
      setBusy(true, "Deleting conversation…");
      bridge.deleteConversation(item.conversation_id);
    });
    row.append(button, remove);
    list.append(row);
  }
  byId("conversation-name").textContent = activeEmpty ? "" : activeName;
  byId("conversation-name").classList.toggle("hidden", activeEmpty);
}

function renderConversationEmpty() {
  conversation.replaceChildren();
  const content = emptyChatContent[activeProjectKind] || emptyChatContent.code;
  let promptIndex = Math.floor(Math.random() * content.prompts.length);
  if (content.prompts.length > 1 && promptIndex === lastEmptyPrompt[activeProjectKind]) {
    promptIndex = (promptIndex + 1) % content.prompts.length;
  }
  lastEmptyPrompt[activeProjectKind] = promptIndex;
  const landing = document.createElement("section");
  landing.className = "conversation-empty";
  const prompt = document.createElement("h2");
  prompt.textContent = content.prompts[promptIndex];
  const suggestions = document.createElement("div");
  suggestions.className = "prompt-suggestions";
  const choices = [...content.suggestions].sort(() => Math.random() - 0.5).slice(0, 3);
  for (const text of choices) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.addEventListener("click", () => { byId("question").value = text; byId("question").focus(); });
    suggestions.append(button);
  }
  landing.append(prompt, suggestions);
  conversation.append(landing);
  conversationScroll.scrollTop = 0;
  updateFloatingHeader();
}

function renderAttachments() {
  const host = byId("attachment-list");
  host.replaceChildren();
  for (const path of messageAttachments) {
    const chip = document.createElement("span");
    chip.className = "attachment-chip";
    chip.textContent = path.split(/[\\/]/).pop() || path;
    chip.title = path;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${chip.textContent}`);
    remove.addEventListener("click", () => {
      messageAttachments = messageAttachments.filter((item) => item !== path);
      renderAttachments();
    });
    chip.append(remove);
    host.append(chip);
  }
  host.classList.toggle("hidden", !messageAttachments.length);
}

function updateEffort(index) {
  const level = effortLevels[Math.max(0, Math.min(effortLevels.length - 1, Number(index)))];
  selectedEffort = level.key;
  byId("effort-button").textContent = level.name;
  byId("effort-name").textContent = level.name;
  byId("effort-detail").textContent = level.detail;
  byId("effort-range").value = String(effortLevels.indexOf(level));
  byId("effort-range").style.setProperty(
    "--effort-percent",
    `${(effortLevels.indexOf(level) / (effortLevels.length - 1)) * 100}%`,
  );
  [...document.querySelectorAll(".effort-points i")].forEach((point, pointIndex) => {
    point.classList.toggle("selected", pointIndex === effortLevels.indexOf(level));
  });
}

/**
 * Clear the interval used by the pending-operation label.
 */
function stopPhaseAnimation() {
  clearInterval(phaseTimer);
  phaseTimer = null;
}

/**
 * Animate a pending answer phase while no completed answer is available.
 *
 * @param label Current backend phase label.
 * @param detail Optional supporting status text.
 */
function showTurnPhase(label, detail = "") {
  if (!pendingTurn) return;
  stopPhaseAnimation();
  pendingTurn.detail.textContent = detail;
  pendingTurn.detail.classList.toggle("hidden", !detail);
  let dots = 1;
  const paint = () => {
    const suffix = ".".repeat(dots);
    pendingTurn.label.textContent = `${label}${suffix}`;
  };
  paint();
  phaseTimer = setInterval(() => {
    dots += 1;
    paint();
    if (dots === 3) stopPhaseAnimation();
  }, 400);
}

/**
 * Append a pending question and begin its retrieval-phase indicator.
 *
 * @param question Validated question text submitted by the user.
 */
function addPendingTurn(question) {
  if (conversation.querySelector(".conversation-empty")) conversation.replaceChildren();
  const item = document.createElement("article");
  item.className = "turn pending";
  item.dataset.turnIndex = String(conversation.children.length);
  const questionNode = document.createElement("div");
  questionNode.className = "user-message";
  questionNode.textContent = question;
  const answer = document.createElement("div");
  answer.className = "answer-message answer-phase";
  const phaseLabel = document.createElement("div");
  phaseLabel.className = "answer-phase-label";
  const phaseDetail = document.createElement("div");
  phaseDetail.className = "answer-phase-detail";
  answer.append(phaseLabel, phaseDetail);
  item.append(questionNode, answer);
  conversation.append(item);
  pendingTurn = { item, answer, label: phaseLabel, detail: phaseDetail };
  showTurnPhase("Generating", "Starting retrieval");
  conversationScroll.scrollTop = conversationScroll.scrollHeight;
}

/**
 * Enable pointer and keyboard selection of a completed answer.
 *
 * @param answer Focusable answer element.
 * @param item Parent conversation article.
 * @param turn Completed turn whose references will be displayed.
 */
function bindAnswerSelection(answer, item, turn) {
  answer.tabIndex = 0;
  answer.setAttribute("aria-label", "Select this answer to view its references");
  const choose = () => {
    showWorkspace(true);
    selectTurn(item, turn);
  };
  answer.addEventListener("click", choose);
  answer.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      choose();
    }
  });
}

/**
 * Reveal an already-completed answer through a local typing animation.
 *
 * @param turn Completed backend response; the animation is not token
 *   streaming.
 */
function completePendingTurn(turn) {
  stopPhaseAnimation();
  updateProjectSummary(currentTurnCount + 1);
  const current = pendingTurn;
  if (!current) {
    addTurn(turn);
    setBusy(false);
    return;
  }
  pendingTurn = null;
  current.item.classList.remove("pending");
  current.answer.classList.remove("answer-phase");
  current.answer.classList.add("typing");
  current.answer.textContent = "";
  const text = String(turn.answer || "");
  let position = 0;
  clearInterval(answerTimer);
  answerTimer = setInterval(() => {
    position += 1;
    current.answer.textContent = text.slice(0, position);
    conversationScroll.scrollTop = conversationScroll.scrollHeight;
    if (position < text.length) return;
    clearInterval(answerTimer);
    answerTimer = null;
    current.answer.classList.remove("typing");
    current.answer.replaceChildren();
    bindAnswerSelection(current.answer, current.item, turn);
    renderAnswer(current.answer, turn, current.item);
    selectTurn(current.item, turn);
    setBusy(false);
  }, 25);
}

/**
 * Dispatch a JSON event from the Qt bridge into interface state changes.
 *
 * @param raw JSON string using the engine event protocol; parsing errors
 *   propagate.
 */
function onEvent(raw) {
  const data = JSON.parse(raw);
  if (data.event === "trace_ready") {
    activeTracePath = data.trace_path || "";
  } else if (data.event === "readiness") {
    requiredModels = data.required_models || [];
    readinessEnvironmentReady = Boolean(data.environment_ready);
    cachedModels = new Set(data.cached_models || []);
    const card = byId("first-run-setup");
    const steps = byId("readiness-steps");
    steps.replaceChildren();
    const runtime = document.createElement("li");
    runtime.textContent = data.environment_ready
      ? `${String(data.backend).toUpperCase()} runtime is ready`
      : `${String(data.backend).toUpperCase()} runtime is missing; restart through uv run qachat --gui`;
    runtime.classList.toggle("complete", data.environment_ready);
    steps.append(runtime);
    for (const modelId of requiredModels) {
      const item = document.createElement("li");
      const present = cachedModels.has(modelId);
      item.textContent = `${present ? "Ready" : "Missing"}: ${modelId}`;
      item.classList.toggle("complete", present);
      steps.append(item);
    }
    card.classList.toggle("hidden", data.ready);
    byId("readiness-summary").textContent = data.ready
      ? "QAChat is ready."
      : `${data.missing_models.length} required model${data.missing_models.length === 1 ? "" : "s"} need to be downloaded.`;
    byId("install-required").disabled = !data.environment_ready || !data.missing_models.length;
    refreshModelStatus();
    if (!data.ready) {
      byId("models-panel").classList.remove("hidden");
      byId("close-models").focus();
    }
  } else if (data.event === "installation_started") {
    setBusy(true, `Preparing ${data.total} required models…`);
    byId("first-run-setup").classList.remove("hidden");
    byId("install-required").disabled = true;
    byId("readiness-time").textContent = "Starting download. Remaining time depends on model size and network speed.";
  } else if (data.event === "installation_progress") {
    const elapsed = seconds(data.elapsed_seconds || 0);
    byId("readiness-summary").textContent = `Downloading ${data.index} of ${data.total}`;
    byId("readiness-time").textContent = `${elapsed} elapsed · downloading ${data.model_id}`;
    byId("activity-percent").textContent = `Model ${data.index} of ${data.total}`;
    byId("activity-text").textContent = `Downloading ${data.model_id}`;
    byId("activity-detail").textContent = `${elapsed} elapsed`;
  } else if (data.event === "installation_model_complete") {
    cachedModels.add(data.model_id);
    byId("readiness-time").textContent = `${data.model_id} completed in ${seconds(data.model_seconds)} · ${seconds(data.elapsed_seconds)} total`;
  } else if (data.event === "installation_complete") {
    byId("readiness-summary").textContent = "Required models are installed.";
    byId("readiness-time").textContent = `Setup completed in ${seconds(data.elapsed_seconds)}.`;
    setBusy(false);
  } else if (data.event === "settings") {
    showSettings(data);
  } else if (data.event === "projects") {
    renderSavedProjects(data.projects || []);
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
  } else if (data.event === "answer_phase") {
    activity.classList.add("answer-progress");
    if (data.phase === "retrieving") {
      showTurnPhase("Generating", data.message || "Searching project evidence");
      byId("activity-text").textContent = data.message || "Searching project evidence";
      byId("activity-detail").textContent = "";
    } else if (data.phase === "evidence_ready") {
      showTurnPhase(
        "Generating",
        data.evidence_count == null
          ? "Preparing retrieved evidence"
          : `Preparing ${data.evidence_count} retrieved references`,
      );
      byId("activity-text").textContent = data.evidence_count == null
        ? "Preparing retrieved evidence for the model"
        : `Preparing ${data.evidence_count} retrieved references for the model`;
      byId("activity-detail").textContent = "";
    } else if (data.phase === "generating") {
      showTurnPhase(
        "Generating",
        data.evidence_count == null
          ? "Writing the answer"
          : `Writing with ${data.evidence_count} retrieved references`,
      );
      byId("activity-text").textContent = data.evidence_count == null
        ? "Writing the answer"
        : `Writing the answer using ${data.evidence_count} retrieved references`;
      byId("activity-detail").textContent = "";
    } else if (data.phase === "saving") {
      showTurnPhase("Generating", data.message || "Saving the answer");
      byId("activity-text").textContent = data.message || "Saving this turn to the conversation";
      byId("activity-detail").textContent = "";
    } else if (data.phase === "completed") {
      stopPhaseAnimation();
      byId("activity-text").textContent = data.message || "Displaying the completed response";
      byId("activity-detail").textContent = "";
    }
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
    activeProjectKind = data.kind;
    messageAttachments = [];
    renderAttachments();
    selectedSources.length = 0;
    previewRoots.clear();
    previewSkipped.clear();
    checkedFiles.clear();
    renderPreview();
    setup.classList.add("hidden");
    chat.classList.remove("hidden");
    shell.classList.add("chat-open");
    byId("conversations-panel").classList.remove("hidden");
    showConversations(true);
    conversation.replaceChildren();
    clearReferences();
    renderConversations(data);
    data.turns.forEach(addTurn);
    if (!data.turns.length) renderConversationEmpty();
    byId("workspace-path").value = data.project;
    byId("workspace-location").textContent = data.project;
    renderWorkspace(data.workspace_files);
    showWorkspace(Boolean(data.turns.length));
    byId("temperature").value = data.temperature;
    byId("max-new-tokens").value = data.max_new_tokens ?? "";
    byId("thinking").checked = data.thinking;
    byId("project-name").textContent = data.project.split(/[\\/]/).pop();
    updateProjectSummary(data.turns.length);
    updateFloatingHeader();
    setBusy(false);
  } else if (data.event === "conversation_opened") {
    stopPhaseAnimation();
    clearInterval(answerTimer);
    answerTimer = null;
    pendingTurn = null;
    messageAttachments = [];
    renderAttachments();
    conversation.replaceChildren();
    clearReferences();
    renderConversations(data);
    data.turns.forEach(addTurn);
    if (!data.turns.length) {
      renderConversationEmpty();
      showWorkspace(false);
    }
    byId("temperature").value = data.temperature;
    byId("max-new-tokens").value = data.max_new_tokens ?? "";
    byId("thinking").checked = data.thinking;
    updateProjectSummary(data.turns.length);
    updateFloatingHeader();
    setBusy(false);
  } else if (data.event === "conversations") {
    renderConversations(data);
    setBusy(false);
  } else if (data.event === "reference_preview") {
    renderSourcePreview(data);
  } else if (data.event === "reference_error") {
    if (activeReference && data.turn_index === activeReference.turnIndex
        && data.citation === activeReference.citation) {
      activeReference.note.textContent = data.message;
    }
  } else if (data.event === "answered") {
    activity.classList.remove("answer-progress");
    completePendingTurn(data.turn);
  } else if (data.event === "reset") {
    renderConversationEmpty();
    clearReferences();
    updateProjectSummary(0);
    updateFloatingHeader();
    setBusy(false);
  } else if (data.event === "stopped") {
    if (downloadingModel) {
      byId("model-download-status").textContent = "Download stopped. Reopen Models to check what is cached.";
      downloadingModel = null;
      byId("cancel-model-download").classList.add("hidden");
    }
    if (resumeAfterStop) {
      resumeAfterStop = false;
      stopPhaseAnimation();
      pendingTurn?.item.remove();
      pendingTurn = null;
      ready = false;
      setBusy(true, "Reopening project after interruption…");
      bridge.resumeProjectConversation(
        byId("workspace-path").value,
        activeConversationId,
      );
      return;
    }
    ready = false;
    updateFloatingHeader();
    chat.classList.add("hidden");
    shell.classList.remove("chat-open");
    byId("conversations-panel").classList.add("hidden");
    byId("toggle-conversations").classList.add("hidden");
    setup.classList.remove("hidden");
    showWorkspace(false);
    clearReferences();
    byId("connection").textContent = "Stopped · reopen your project";
    setBusy(false);
    bridge.requestSettings(byId("kind").value, byId("workspace-path").value.trim());
  } else if (data.event === "backend_failed") {
    ready = false;
    updateFloatingHeader();
    chat.classList.add("hidden");
    shell.classList.remove("chat-open");
    byId("conversations-panel").classList.add("hidden");
    byId("toggle-conversations").classList.add("hidden");
    setup.classList.remove("hidden");
    showWorkspace(false);
    clearReferences();
    showError(data.message, data.trace_path || activeTracePath);
  } else if (data.event === "error") {
    showError(data.message, data.trace_path || activeTracePath);
  }
}

// UI handlers validate user input before issuing bridge commands.
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
byId("send").addEventListener("click", (event) => {
  if (!busy || !ready || !bridge) return;
  event.preventDefault();
  resumeAfterStop = true;
  byId("send").disabled = true;
  byId("activity-text").textContent = "Stopping generation…";
  bridge.stopOperation();
});
byId("open-models").addEventListener("click", openModels);
byId("models-hint-button").addEventListener("click", openModels);
byId("close-models").addEventListener("click", () => byId("models-panel").classList.add("hidden"));
byId("install-required").addEventListener("click", () => {
  if (busy || !bridge) return;
  setBusy(true, "Checking required models…");
  bridge.installRequiredModels();
});
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
      openModels();
      return showError(`The selected ${role} model does not match the backend or embedding query format.`);
    }
  }
  const localModel = (id) => cachedModels.has(id)
    || id.startsWith("/") || id.startsWith(".") || /^[A-Za-z]:[\\/]/.test(id);
  if (!byId("allow-download").checked
      && ![byId("chat-model").value.trim(), byId("embedding-model").value.trim()].every(localModel)) {
    openModels();
    byId("model-download-status").textContent = "Download both selected models before preparing, or allow downloads in Settings.";
    return;
  }
  const selected = Object.fromEntries(
    selectedSources.map((path) => [path, [...(checkedFiles.get(path) || [])]]),
  );
  const included = selectedSources.filter((path) => selected[path].length);
  if (selectedSources.length && !included.length) return showError("Select at least one file.");
  error.classList.add("hidden");
  const usingExample = included.length === 0;
  byId("example-status").classList.toggle("hidden", !usingExample);
  setBusy(true, usingExample ? "Loading example repository…" : "Opening project…");
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
  const temperature = Number(byId("temperature").value);
  if (!byId("temperature").value.trim() || !Number.isFinite(temperature) || temperature < 0) {
    return showError("Choose a temperature of zero or higher.");
  }
  const answerTokens = byId("max-new-tokens").value.trim();
  const maxNewTokens = answerTokens ? Number(answerTokens) : null;
  if (answerTokens && (!Number.isInteger(maxNewTokens) || maxNewTokens < 1)) {
    return showError("Choose a positive answer-token limit or leave it blank.");
  }
  byId("question").value = "";
  addPendingTurn(question);
  setBusy(true, "Finding evidence and answering…");
  const attachedContext = messageAttachments.length
    ? `\n\nAttached paths to consider:\n${messageAttachments.map((path) => `- ${path}`).join("\n")}`
    : "";
  messageAttachments = [];
  renderAttachments();
  bridge.askQuestion(`${question}${attachedContext}`, selectedEffort, JSON.stringify({
    temperature, thinking: byId("thinking").checked, max_new_tokens: maxNewTokens,
    _title_question: question,
  }));
});
byId("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    byId("question-form").requestSubmit();
  }
});
conversationScroll.addEventListener("scroll", updateFloatingHeader, { passive: true });
byId("project-name").addEventListener("input", () => {
  byId("topbar-project-name").textContent = byId("project-name").textContent.trim();
});
byId("new-chat").addEventListener("click", () => {
  if (busy) return;
  showWorkspace(false);
  setBusy(true, "Starting a new conversation…");
  bridge.newConversation();
});
byId("branch-chat").addEventListener("click", () => {
  if (busy) return;
  setBusy(true, "Branching this conversation…");
  bridge.branchConversation();
});
function commitConversationName() {
  const name = byId("conversation-name").textContent.trim();
  if (!busy && activeConversationId && name) bridge.renameConversation(activeConversationId, name);
}
function commitProjectName() {
  const name = byId("project-name").textContent.trim();
  const current = byId("workspace-path").value.split(/[\\/]/).pop();
  if (!busy && ready && name && name !== current) {
    setBusy(true, "Renaming project…");
    bridge.renameProject(name);
  }
}
for (const [id, commit] of [["conversation-name", commitConversationName], ["project-name", commitProjectName]]) {
  byId(id).addEventListener("blur", commit);
  byId(id).addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); event.target.blur(); }
  });
}
byId("change-project").addEventListener("click", () => {
  if (busy) return;
  bridge.closeProject();
  ready = false;
  updateFloatingHeader();
  chat.classList.add("hidden");
  shell.classList.remove("chat-open");
  byId("conversations-panel").classList.add("hidden");
  byId("toggle-conversations").classList.add("hidden");
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
byId("toggle-conversations").addEventListener("click", () => {
  showConversations(shell.classList.contains("conversations-hidden"));
});
byId("effort-button").addEventListener("click", (event) => {
  event.stopPropagation();
  const menu = byId("effort-menu");
  menu.classList.toggle("hidden");
  byId("effort-button").setAttribute("aria-expanded", String(!menu.classList.contains("hidden")));
});
byId("effort-range").addEventListener("input", (event) => {
  updateEffort(event.target.value);
});
byId("attach-button").addEventListener("click", (event) => {
  event.stopPropagation();
  const menu = byId("attach-menu");
  menu.classList.toggle("hidden");
  byId("attach-button").setAttribute("aria-expanded", String(!menu.classList.contains("hidden")));
});
byId("attach-files").addEventListener("click", () => {
  byId("attach-menu").classList.add("hidden");
  bridge?.chooseAttachmentFiles();
});
byId("attach-folder").addEventListener("click", () => {
  byId("attach-menu").classList.add("hidden");
  bridge?.chooseAttachmentFolder();
});
byId("appearance").addEventListener("change", (event) => applyAppearance(event.target.value));
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
  if (!event.target.closest(".effort-picker")) byId("effort-menu").classList.add("hidden");
  if (!event.target.closest(".attach-picker")) byId("attach-menu").classList.add("hidden");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeModelMenus();
});
for (const id of ["extensions", "max-file-bytes"]) byId(id).addEventListener("change", previewSources);

// Attach the bridge only after Qt exposes the channel; then request state.
new QWebChannel(qt.webChannelTransport, (channel) => {
  bridge = channel.objects.qachat;
  bridge.event.connect(onEvent);
  bridge.pathsSelected.connect(addSources);
  bridge.attachmentsSelected.connect((paths) => {
    for (const path of paths) {
      if (path && !messageAttachments.includes(path)) messageAttachments.push(path);
    }
    renderAttachments();
  });
  bridge.workspaceSelected.connect((path) => {
    byId("workspace-path").value = path;
    bridge.requestSettings(byId("kind").value, path);
    refreshSavedProject();
  });
  bridge.requestSettings(byId("kind").value, byId("workspace-path").value.trim());
  bridge.requestReadiness();
  showWorkspace(false);
  updateEffort(2);
  setBusy(false);
  refreshSavedProject();
});

// Refresh the setup shortcut when external changes may invalidate a project.
window.addEventListener("focus", refreshSavedProject);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshSavedProject();
});
// Also catch files removed while the project chooser stays in the foreground.
window.setInterval(refreshSavedProject, 3000);
applyAppearance(localStorage.getItem("qachat-appearance") || "system");
