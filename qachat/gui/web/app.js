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
// Answer phases sent by RAGSession (qachat.rag.ANSWER_PHASE_*); the strings
// are the core's wire values and must stay identical to it.
const ANSWER_PHASE_ROUTING = "routing";
const ANSWER_PHASE_ROUTED = "routed";
const ANSWER_PHASE_RETRIEVING = "retrieving";
const ANSWER_PHASE_EVIDENCE_READY = "evidence_ready";
const ANSWER_PHASE_GENERATING = "generating";
const ANSWER_PHASE_SAVING = "saving";
const ANSWER_PHASE_COMPLETED = "completed";
const ANSWER_PHASE_PROCESSING = "processing";
// Mirrors qachat.tools.planning.PlanChoice.
const PLAN_CHOICE_CONTINUE = "continue";
const PLAN_CHOICE_QUICK = "quick";
const PLAN_CHOICE_CANCEL = "cancel";
// Phase added by the GUI engine (qachat.gui.engine.ENGINE_PHASE_*).
const ENGINE_PHASE_LOADING_MODEL = "loading_model";
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
// Workspace folders that start open but the user has folded.
const collapsedWorkspaceFolders = new Set();
let profile = null;
let cachedModels = new Set();
let modelChoices = [];
// Effort levels per backend: saved for the project, built-in defaults, and
// the Settings panel's unsaved copy for the active backend.
let savedLevels = {};
let defaultLevels = {};
let draftLevels = [];
// Intent router model per backend: saved, built-in, and unsaved choice.
let savedRouters = {};
let defaultRouters = {};
let draftRouter = "";
let modelRoles = {};
let modelsBaseline = null;
let applyingModels = false;
let closeAfterModelsApply = false;
let downloadingModel = null;
let selectedTurn = null;
let activeReference = null;
let installedBackends = {};
let requiredModels = [];
let readinessEnvironmentReady = false;
let setupCompletionVisible = false;
let supportedExtensions = new Set();
let automaticBackend = "mlx";
let activeBackend = "mlx";
let resumeRefreshTimer = null;
let activeTracePath = "";
let activeConversationId = "";
// Last conversation registry, and the empty chat whose first question is
// being answered; that chat is listed before it has saved turns.
let lastConversationData = null;
let askingConversation = null;
let pendingTurn = null;
let phaseTimer = null;
let answerTimer = null;
let answerStageTimer = null;
let answerStageKey = null;
let answerStageQueue = [];
let selectedModelLevel = 2;
let selectedRetrievalLevel = 2;
let resumeAfterStop = false;
let activeProjectKind = "code";
let lastEmptyPrompt = { code: -1, literature: -1 };
let messageAttachments = [];
let currentTurnCount = 0;
// Project types from qachat/project_types.json, keyed by ID.
let projectTypes = {};
// Used only if a type lists no welcome prompts or suggestions.
const genericEmptyChat = {
  welcome_prompts: ["What would you like to explore in these sources?"],
  suggestions: ["Explain a concept", "Find evidence across my files", "Compare two approaches"],
};

/**
 * Rebuild the project type dropdown, keeping the current choice.
 *
 * @param types Project types sent by the engine.
 */
function renderProjectTypes(types) {
  if (!Array.isArray(types) || !types.length) return;
  projectTypes = Object.fromEntries(types.map((item) => [item.kind, item]));
  const select = byId("kind");
  const current = select.value;
  select.replaceChildren(...types.map((item) => {
    const option = document.createElement("option");
    option.value = item.kind;
    option.textContent = item.label;
    return option;
  }));
  // The first type in the file is the default.
  select.value = projectTypes[current] ? current : types[0].kind;
  activeProjectKind = ready ? activeProjectKind : select.value;
}

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
  bridge?.setAppearance(selected);
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
  if (busy || !paths.length) return;
  if (ready) {
    const names = paths.map((path) => String(path).split(/[\\/]/).pop()).join("\n");
    if (!confirm(`Add these sources to this project?\n\n${names}\n\nQAChat will copy supported files into sources and update the index, reusing unchanged chunks when possible. This may take a while. Existing sources and saved chats will be kept. Re-added files may replace their existing copies. Continue?`)) return;
    error.classList.add("hidden");
    setBusy(true, "Adding sources and updating the index…");
    bridge.addProjectSources(JSON.stringify(paths));
    return;
  }
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
  // Root-relative trees (the workspace) can recover the folder's own path.
  if (paths[0]) details.dataset.path = paths[0].split("/").slice(0, depth + 1).join("/");
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

const DEFAULT_LEVEL = 2;
const MAX_RETRIEVAL = 200;

const cloneLevels = (levels) => (levels || []).map((level) => ({ ...level }));

/**
 * Return the saved effort levels for the active backend.
 *
 * @returns Five {name, model, retrieval} records, or none before settings load.
 */
function currentLevels() {
  return savedLevels[activeBackend] || defaultLevels[activeBackend] || [];
}

/**
 * Return the saved intent router model for the active backend.
 */
function currentRouter() {
  return savedRouters[activeBackend] || defaultRouters[activeBackend] || "";
}

/**
 * Report a cached model's role: chat, embedding, or other.
 *
 * @param id Model ID.
 * @returns Role from the cache scan, the preset catalog, or the name.
 */
function modelRole(id) {
  if (modelRoles[id]) return modelRoles[id];
  const preset = modelChoices.find((choice) => choice.model_id === id);
  if (preset) return preset.role;
  return /embed/i.test(id) ? "embedding" : "other";
}

/**
 * List dropdown choices: cached models of a role plus the current value.
 *
 * @param role chat or embedding.
 * @param current Model already chosen, kept even when not downloaded.
 * @returns Sorted model IDs.
 */
function roleModelIds(role, current) {
  const ids = [...cachedModels].filter((id) => modelRole(id) === role);
  if (current && !ids.includes(current)) ids.push(current);
  return ids.sort((left, right) => left.localeCompare(right));
}

/**
 * Build a model dropdown; models that are not downloaded are marked.
 */
function modelSelect(ids, value, label) {
  const select = document.createElement("select");
  select.className = "model-select";
  select.setAttribute("aria-label", label);
  for (const id of ids) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = cachedModels.has(id) ? id : `${id} (not downloaded)`;
    select.append(option);
  }
  select.value = value;
  select.disabled = busy;
  return select;
}

/**
 * Show a Download button under a row whose model is not cached.
 */
function downloadNotice(id, installed) {
  if (!id || cachedModels.has(id) || id.startsWith("/") || id.startsWith(".")) return null;
  const notice = document.createElement("div");
  notice.className = "model-missing";
  const text = document.createElement("small");
  text.textContent = "Not downloaded";
  const download = document.createElement("button");
  download.type = "button";
  download.className = "primary";
  download.textContent = "Download";
  download.disabled = busy || !installed;
  // An explicit Download click authorizes this download.
  download.addEventListener("click", () => applyModelEdits(false, id));
  notice.append(text, download);
  return notice;
}

/**
 * Build a model group heading with its Restore defaults button.
 */
function modelGroupHeader(role, title) {
  const header = document.createElement("div");
  header.className = "model-group-header";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const restore = document.createElement("button");
  restore.type = "button";
  restore.className = "secondary restore-defaults";
  restore.textContent = "Restore defaults";
  restore.setAttribute("aria-label", `Restore default ${title.toLowerCase()}`);
  restore.dataset.role = role;
  restore.disabled = busy || !hasModelChanges(role);
  restore.addEventListener("click", () => restoreDefaultModels(role));
  header.append(heading, restore);
  return header;
}

/**
 * Build the effort-level rows, the embedding choice, and the cache listing.
 */
function renderModelsPanel() {
  const host = byId("model-list");
  const focused = document.activeElement?.closest?.("[data-focus-key]")?.dataset.focusKey;
  host.replaceChildren();
  const installed = installedBackends[activeBackend];
  const mlx = activeBackend === "mlx";
  // Only a missing backend needs a note above the levels.
  byId("models-backend").textContent = installed ? ""
    : `${activeBackend.toUpperCase()} packages are not installed. Run uv sync --extra gui --extra ${activeBackend} from the QAChat folder first.`;
  byId("models-backend").classList.toggle("hidden", Boolean(installed));

  const chat = document.createElement("section");
  chat.className = "model-group";
  chat.append(modelGroupHeader("chat", "Effort levels"));
  if (!mlx) {
    // PyTorch cannot switch models per question.
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "PyTorch answers every level with one model; levels still set the number of references.";
    chat.append(hint);
  }
  const columns = document.createElement("div");
  columns.className = "level-row level-columns";
  columns.setAttribute("aria-hidden", "true");
  for (const label of ["Level", "Language model", "References"]) {
    const cell = document.createElement("span");
    cell.textContent = label;
    columns.append(cell);
  }
  chat.append(columns);
  draftLevels.forEach((level, index) => {
    const row = document.createElement("div");
    row.className = "level-row";
    const name = document.createElement("input");
    name.type = "text";
    name.className = "level-name";
    name.maxLength = 24;
    name.value = level.name;
    name.disabled = busy;
    name.dataset.focusKey = `name-${index}`;
    name.setAttribute("aria-label", `Name of effort level ${index + 1}`);
    name.addEventListener("input", () => {
      level.name = name.value;
      refreshRestoreButtons();
    });
    const select = modelSelect(roleModelIds("chat", level.model), level.model,
      `Language model for ${level.name || `level ${index + 1}`}`);
    select.dataset.focusKey = `model-${index}`;
    select.addEventListener("change", () => {
      // Other backends cannot switch models per level: one model for all.
      for (const target of mlx ? [level] : draftLevels) target.model = select.value;
      byId("chat-model").value = draftLevels[DEFAULT_LEVEL].model;
      renderModelsPanel();
    });
    const count = document.createElement("input");
    count.type = "number";
    count.className = "level-retrieval";
    count.min = "1";
    count.max = String(MAX_RETRIEVAL);
    count.step = "1";
    count.value = String(level.retrieval);
    count.disabled = busy;
    count.dataset.focusKey = `refs-${index}`;
    count.setAttribute("aria-label", `References for ${level.name || `level ${index + 1}`}`);
    count.addEventListener("input", () => {
      level.retrieval = Number(count.value);
      refreshRestoreButtons();
    });
    row.append(name, select, count);
    chat.append(row);
    const notice = downloadNotice(level.model, installed);
    if (notice) chat.append(notice);
  });
  host.append(chat);

  const embedding = document.createElement("section");
  embedding.className = "model-group";
  embedding.append(modelGroupHeader("embedding", "Embedding model"));
  const current = byId("embedding-model").value.trim();
  const row = document.createElement("div");
  row.className = "model-row";
  const select = modelSelect(roleModelIds("embedding", current), current, "Embedding model");
  select.dataset.focusKey = "embedding";
  select.addEventListener("change", () => {
    byId("embedding-model").value = select.value;
    renderModelsPanel();
  });
  row.append(select);
  embedding.append(row);
  const notice = downloadNotice(current, installed);
  if (notice) embedding.append(notice);
  host.append(embedding);

  const router = document.createElement("section");
  router.className = "model-group";
  router.append(modelGroupHeader("router", "Intent model"));
  const routerRow = document.createElement("div");
  routerRow.className = "model-row";
  const routerSelect = modelSelect(roleModelIds("chat", draftRouter), draftRouter, "Intent model");
  routerSelect.dataset.focusKey = "router";
  routerSelect.addEventListener("change", () => {
    draftRouter = routerSelect.value;
    renderModelsPanel();
  });
  routerRow.append(routerSelect);
  router.append(routerRow);
  const routerNotice = downloadNotice(draftRouter, installed);
  if (routerNotice) router.append(routerNotice);
  host.append(router);
  if (focused) host.querySelector(`[data-focus-key="${focused}"]`)?.focus();
}

/**
 * Enable each Restore defaults button only when it would change something.
 */
function refreshRestoreButtons() {
  for (const button of byId("model-list").querySelectorAll(".restore-defaults")) {
    button.disabled = busy || !hasModelChanges(button.dataset.role);
  }
}

/**
 * Return the built-in embedding model for the active backend.
 */
function defaultEmbeddingModel() {
  return compatibleChoices("embedding")[0]?.model_id
    || backendChoices("embedding")[0]?.model_id || "";
}

/**
 * Report whether a group differs from its built-in defaults.
 *
 * @param role chat (effort levels) or embedding.
 * @returns True when restoring defaults would change something.
 */
function hasModelChanges(role) {
  if (role === "chat") {
    return JSON.stringify(draftLevels) !== JSON.stringify(defaultLevels[activeBackend] || []);
  }
  if (role === "router") return draftRouter !== (defaultRouters[activeBackend] || "");
  return byId("embedding-model").value.trim() !== defaultEmbeddingModel()
    || byId("query-format").value !== "qwen3";
}

/**
 * Return a role's presets for the active backend, ignoring query format.
 *
 * @param role Model role: chat or embedding.
 * @returns Matching model-preset records.
 */
function backendChoices(role) {
  return modelChoices.filter((choice) =>
    choice.role === role && choice.backend === activeBackend);
}

/**
 * Reset a group to its built-in values; Close applies the change.
 *
 * @param role chat (effort levels) or embedding.
 */
function restoreDefaultModels(role) {
  if (busy) return;
  if (role === "chat") {
    draftLevels = cloneLevels(defaultLevels[activeBackend]);
    if (draftLevels.length) byId("chat-model").value = draftLevels[DEFAULT_LEVEL].model;
  } else if (role === "router") {
    draftRouter = defaultRouters[activeBackend] || draftRouter;
  } else {
    byId("query-format").value = "qwen3";
    byId("embedding-model").value = defaultEmbeddingModel();
  }
  refreshModelStatus();
  renderModelsPanel();
}

/**
 * Render and open the models panel, then focus its close control.
 */
function openModels() {
  if (byId("models-panel").classList.contains("hidden")) {
    draftLevels = cloneLevels(currentLevels());
    draftRouter = currentRouter();
    modelsBaseline = currentModelSettings();
  }
  renderModelsPanel();
  byId("models-panel").classList.remove("hidden");
  byId("close-models").focus();
}

function currentModelSettings() {
  return {
    chat_model: draftLevels[DEFAULT_LEVEL]?.model || byId("chat-model").value.trim(),
    embedding_model: byId("embedding-model").value.trim(),
    query_format: byId("query-format").value,
    effort_levels: cloneLevels(draftLevels),
    router_model: draftRouter,
  };
}
function hideModels() {
  setupCompletionVisible = false;
  byId("models-panel").classList.add("hidden");
  syncSourceDropTarget();
}

/**
 * Explain the first invalid effort level, or return an empty string.
 */
function levelProblem(levels) {
  for (const [index, level] of levels.entries()) {
    if (!level.name.trim()) return `Give effort level ${index + 1} a name.`;
    if (!Number.isInteger(level.retrieval) || level.retrieval < 1 || level.retrieval > MAX_RETRIEVAL) {
      return `References for ${level.name.trim()} must be a whole number from 1 to ${MAX_RETRIEVAL}.`;
    }
  }
  return "";
}

function applyModelEdits(closeAfter = false, explicitDownload = null) {
  if (busy || !bridge) return;
  const settings = currentModelSettings();
  const before = modelsBaseline || settings;
  const changed = JSON.stringify(settings.effort_levels) !== JSON.stringify(before.effort_levels)
    || settings.embedding_model !== before.embedding_model || settings.query_format !== before.query_format
    || settings.router_model !== before.router_model;
  if (!changed && !explicitDownload) { if (closeAfter) hideModels(); return; }
  const problem = levelProblem(settings.effort_levels);
  if (problem) {
    byId("model-download-status").textContent = problem;
    return;
  }
  for (const level of settings.effort_levels) level.name = level.name.trim();
  if (!settings.chat_model || !settings.embedding_model) {
    byId("model-download-status").textContent = "Choose language and embedding models.";
    return;
  }
  const requested = [...new Set([settings.embedding_model,
    ...settings.effort_levels.map((level) => level.model),
    ...(settings.router_model ? [settings.router_model] : [])])];
  const missing = requested.filter((id) => !cachedModels.has(id)
    && !id.startsWith("/") && !id.startsWith("."));
  const reindex = ready && (settings.embedding_model !== before.embedding_model
    || settings.query_format !== before.query_format);
  const warnings = [];
  if (missing.some((id) => id !== explicitDownload)) warnings.push(`These models need to download, which may take a while:\n${missing.join("\n")}`);
  if (reindex) warnings.push("Changing the embedding model or query format will rebuild this project's index. Your saved chats will be kept.");
  if (warnings.length && !confirm(`${warnings.join("\n\n")}\n\nProceed?`)) return;
  applyingModels = true;
  closeAfterModelsApply = closeAfter;
  error.classList.add("hidden");
  byId("model-download-status").textContent = reindex ? "Applying models and rebuilding the index…" : "Applying model settings…";
  setBusy(true, byId("model-download-status").textContent);
  bridge.applyModelSettings(JSON.stringify({ ...settings, download: missing.length > 0 }));
}
function closeModels() {
  if (busy) return;
  if (document.activeElement?.matches(".model-id-input")) document.activeElement.blur();
  applyModelEdits(true);
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
  draftLevels = cloneLevels(currentLevels());
  draftRouter = currentRouter();
  renderEffortPicker();
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
  savedLevels = data.effort_levels || {};
  defaultLevels = data.default_effort_levels || defaultLevels;
  savedRouters = data.router_models || {};
  defaultRouters = data.default_router_models || defaultRouters;
  renderProjectTypes(data.project_types);
  modelRoles = data.model_roles || modelRoles;
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
    if (ready && ["temperature", "max_new_tokens"].includes(key)) continue;
    byId(key.replaceAll("_", "-")).value = profile[key] ?? "";
  }
  if (!ready) byId("thinking").checked = profile.thinking;
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
    _gui_effort_levels: { [activeBackend]: cloneLevels(currentLevels()) },
    chat_mlx: profile.chat_mlx, chat_torch: profile.chat_torch,
    embedding_mlx: profile.embedding_mlx, embedding_torch: profile.embedding_torch,
    query_format: byId("query-format").value,
    extensions: [...new Set([...profile.extensions, ...fileTypes()])],
    max_file_bytes: selectedFileSizeLimit(),
    target_tokens: Number(byId("target-tokens").value),
    max_tokens: Number(byId("max-tokens").value),
    overlap_tokens: Number(byId("overlap-tokens").value),
    // Models are downloaded from Settings, never silently while preparing.
    allow_download: false,
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
    const files = allFiles(node);
    const item = branch(name, node, files, selected);
    // Sources and target start open; indexes and logs stay folded.
    const key = `${name}|${files[0] || ""}`;
    const startsOpen = name === "sources" || name === "target";
    item.open = startsOpen ? !collapsedWorkspaceFolders.has(key) : expandedFolders.has(key);
    item.addEventListener("toggle", () => {
      if (!item.isConnected || !startsOpen) return;
      item.open ? collapsedWorkspaceFolders.delete(key) : collapsedWorkspaceFolders.add(key);
    });
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
    if (name === "sources") {
      item.id = "workspace-source-drop";
      item.classList.add("source-drop-target");
      const hint = document.createElement("p");
      hint.className = "source-drop-hint";
      hint.textContent = "Drop files or folders here to add sources";
      item.querySelector("summary").append(hint);
      item.addEventListener("dragover", (event) => {
        if (busy || !ready) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        showNativeSourceDrag(true);
      });
      item.addEventListener("dragleave", (event) => {
        if (!item.contains(event.relatedTarget)) showNativeSourceDrag(false);
      });
      item.addEventListener("drop", (event) => {
        event.preventDefault();
        showNativeSourceDrag(false);
        addSources(Array.from(event.dataTransfer.files, (file) => file.path).filter(Boolean));
      });
      item.addEventListener("contextmenu", (event) => {
        const row = event.target.closest(".tree-file, .tree-folder > summary");
        const path = row?.classList.contains("tree-file")
          ? row.dataset.path : row?.parentElement.dataset.path;
        // The sources folder itself is not deletable; only its contents are.
        if (!path || !path.startsWith("sources/")) return;
        event.preventDefault();
        showSourceMenu(event.clientX, event.clientY, path, row.classList.contains("tree-file"));
      });
    }
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

// Send the visible drop rectangle to Qt; all other workspace rows reject drops.
let sourceDragActive = false;
const sourceDropSpotlight = document.createElement("div");
sourceDropSpotlight.className = "source-drop-spotlight hidden";
sourceDropSpotlight.setAttribute("aria-hidden", "true");
document.body.append(sourceDropSpotlight);
let lastSourceDropBounds = "";
function syncSourceDropTarget() {
  if (!bridge?.setSourceDropTarget) return;
  const target = ready ? byId("workspace-source-drop") : byId("drop-zone");
  const rect = target?.getBoundingClientRect();
  const visible = !busy && rect?.width && rect?.height && target.getClientRects().length
    && !target.closest(".hidden") && byId("models-panel").classList.contains("hidden");
  if (!visible) return sendSourceDropBounds("{}");
  let left = Math.max(0, rect.left), top = Math.max(0, rect.top);
  let right = Math.min(innerWidth, rect.right), bottom = Math.min(innerHeight, rect.bottom);
  for (let parent = target.parentElement; parent; parent = parent.parentElement) {
    const style = getComputedStyle(parent);
    const bounds = parent.getBoundingClientRect();
    if (/(auto|scroll|hidden)/.test(style.overflowY)) {
      top = Math.max(top, bounds.top); bottom = Math.min(bottom, bounds.bottom);
    }
    if (/(auto|scroll|hidden)/.test(style.overflowX)) {
      left = Math.max(left, bounds.left); right = Math.min(right, bounds.right);
    }
  }
  sendSourceDropBounds(JSON.stringify({x: left, y: top, width: Math.max(0, right-left), height: Math.max(0, bottom-top)}));
}
function sendSourceDropBounds(bounds) {
  const region = JSON.parse(bounds);
  const visible = sourceDragActive && region.width > 0 && region.height > 0;
  sourceDropSpotlight.classList.toggle("hidden", !visible);
  if (visible) {
    Object.assign(sourceDropSpotlight.style, {
      left: `${region.x}px`, top: `${region.y}px`,
      width: `${region.width}px`, height: `${region.height}px`,
    });
  }
  if (bounds === lastSourceDropBounds) return;
  lastSourceDropBounds = bounds;
  bridge.setSourceDropTarget(bounds);
}
function showNativeSourceDrag(active) {
  sourceDragActive = Boolean(active && !busy);
  const target = ready ? byId("workspace-source-drop") : byId("drop-zone");
  target?.classList.toggle("drag-over", sourceDragActive);
  syncSourceDropTarget();
}
window.addEventListener("resize", syncSourceDropTarget);
document.addEventListener("scroll", syncSourceDropTarget, true);
document.addEventListener("toggle", syncSourceDropTarget, true);
new ResizeObserver(syncSourceDropTarget).observe(shell);
new MutationObserver(() => requestAnimationFrame(syncSourceDropTarget)).observe(shell, {
  subtree: true, childList: true, attributes: true, attributeFilter: ["class", "open"],
});
// Never let a file drop navigate the page or write to indexes/profile areas.
document.addEventListener("dragover", (event) => {
  if (!event.target.closest("#workspace-source-drop, #drop-zone")) {
    event.preventDefault(); event.dataTransfer.dropEffect = "none";
  }
});
document.addEventListener("drop", (event) => {
  event.preventDefault();
  showNativeSourceDrag(false);
});
window.addEventListener("blur", () => showNativeSourceDrag(false));

// One shared right-click menu for items inside the workspace sources folder.
const sourceMenu = document.createElement("div");
sourceMenu.className = "context-menu hidden";
sourceMenu.setAttribute("role", "menu");
document.body.append(sourceMenu);

/**
 * Offer deletion for one workspace source file or folder at the pointer.
 *
 * @param x Viewport x coordinate of the right-click.
 * @param y Viewport y coordinate of the right-click.
 * @param path Workspace-relative path beginning with sources/.
 * @param isFile Whether the path is a file rather than a folder.
 */
function showSourceMenu(x, y, path, isFile) {
  if (busy || !ready) return;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.setAttribute("role", "menuitem");
  remove.textContent = isFile ? "Delete file from project…" : "Delete folder from project…";
  remove.addEventListener("click", () => {
    hideSourceMenu();
    removeSource(path, isFile);
  });
  sourceMenu.replaceChildren(remove);
  sourceMenu.classList.remove("hidden");
  const { width, height } = sourceMenu.getBoundingClientRect();
  sourceMenu.style.left = `${Math.min(x, window.innerWidth - width - 8)}px`;
  sourceMenu.style.top = `${Math.min(y, window.innerHeight - height - 8)}px`;
  remove.focus();
}

function hideSourceMenu() {
  sourceMenu.classList.add("hidden");
}

/**
 * Confirm, then delete a workspace source copy and update the index.
 *
 * @param path Workspace-relative path beginning with sources/.
 * @param isFile Whether the path is a file rather than a folder.
 */
function removeSource(path, isFile) {
  if (busy || !bridge) return;
  const name = path.split("/").pop();
  const what = isFile ? `the file "${name}"` : `the folder "${name}" and everything in it`;
  if (!confirm(`Delete ${what} from this project's sources?\n\n${path}\n\nThe copy in the workspace is deleted and the index is updated. Your original files outside the workspace are not changed. Saved chats are kept. Continue?`)) return;
  error.classList.add("hidden");
  setBusy(true, "Removing sources and updating the index…");
  bridge.removeProjectSources(JSON.stringify([path]));
}

document.addEventListener("click", (event) => {
  if (!sourceMenu.contains(event.target)) hideSourceMenu();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideSourceMenu();
});
window.addEventListener("blur", hideSourceMenu);
document.addEventListener("scroll", hideSourceMenu, true);

function openWorkspaceFile(path) {
  if (!path || !bridge || !confirm(`Open ${path} with its default application?`)) return;
  bridge.openWorkspaceFile(path);
}

function showError(message, tracePath = "") {
  clearAskingConversation();
  applyingModels = false;
  closeAfterModelsApply = false;
  byId("model-download-status").textContent = message;
  if (ready) {
    byId("project-name").textContent = byId("workspace-path").value.split(/[\\/]/).pop();
    updateProjectSummary();
  }
  stopPhaseAnimation();
  resetAnswerStages();
  if (pendingTurn) {
    pendingTurn.answer.textContent = "Could not complete this answer.";
    pendingTurn.item.classList.add("failed");
    pendingTurn = null;
  }
  error.textContent = tracePath ? `${message} Trace: ${tracePath}` : message;
  error.classList.remove("hidden");
  setupCompletionVisible = false;
  byId("readiness-meter").classList.add("hidden");
  byId("readiness-meter").classList.remove("indeterminate");
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
  byId("project-name").contentEditable = String(!value);
  byId("topbar-project-name").contentEditable = String(!value);
  byId("backend").disabled = ready || value;
  for (const id of ["chat-model", "embedding-model", "query-format"]) byId(id).disabled = value;
  byId("close-models").disabled = value;
  byId("conversation-name").contentEditable = String(!value);
  syncSourceDropTarget();
  if (!value) activity.classList.remove("answer-progress");
  byId("open").disabled = value || !bridge;
  const answering = value && ready && activity.classList.contains("answer-progress");
  byId("question-form").classList.toggle("composer-disabled", value && !answering);
  byId("question-form").setAttribute("aria-busy", String(value));
  for (const id of ["question", "attach-button", "attach-files", "attach-folder", "effort-button", "effort-model-range", "effort-retrieval-range"]) {
    byId(id).disabled = value || !ready;
  }
  if (value) {
    showNativeSourceDrag(false);
    byId("attach-menu").classList.add("hidden");
    byId("effort-menu").classList.add("hidden");
  }
  byId("send").disabled = !ready || (value && !answering);
  byId("send").classList.toggle("stopping", answering);
  byId("send").querySelector("span").textContent = answering ? "■" : "↑";
  byId("send").setAttribute("aria-label", answering ? "Stop generation" : "Send message");
  byId("send").title = byId("send").getAttribute("aria-label");
  byId("stop").classList.toggle("hidden", !value || answering);
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

/** Format a byte count using compact binary units. */
function bytes(value) {
  const amount = Math.max(0, Number(value) || 0);
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = amount;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  const digits = unit === 0 || size >= 100 ? 0 : (size >= 10 ? 1 : 2);
  return `${size.toFixed(digits)} ${units[unit]}`;
}

/** Return presentation details for a backend model-download heartbeat. */
function downloadProgress(data) {
  const downloaded = Math.max(0, Number(data.completed_bytes) || 0);
  const total = Math.max(0, Number(data.total_bytes) || 0);
  const rate = Math.max(0, Number(data.bytes_per_second) || 0);
  const elapsed = Math.max(0, Number(data.elapsed_seconds) || 0);
  const percent = total ? Math.min(100, (downloaded / total) * 100) : null;
  const parts = [`${seconds(elapsed)} elapsed`];
  if (total) parts.push(`${bytes(downloaded)} of ${bytes(total)}`);
  if (rate) parts.push(`${bytes(rate)}/s`);
  if (data.eta_seconds != null && Number.isFinite(Number(data.eta_seconds))) {
    parts.push(Number(data.eta_seconds) <= 0 ? "finishing…" : `about ${seconds(data.eta_seconds)} left`);
  } else {
    parts.push("estimating time remaining…");
  }
  return { percent, text: parts.join(" · ") };
}

/** Update the setup card's determinate or animated download meter. */
function updateDownloadMeter(percent) {
  const meter = byId("readiness-meter");
  meter.classList.remove("hidden");
  meter.classList.toggle("indeterminate", percent == null);
  if (percent == null) {
    meter.removeAttribute("aria-valuenow");
    meter.querySelector("span").style.width = "";
  } else {
    const value = Math.min(100, Math.max(0, percent));
    meter.setAttribute("aria-valuenow", value.toFixed(1));
    meter.querySelector("span").style.width = `${value}%`;
  }
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
function renderAnswer(answer, turn, item, thinkingMs = null) {
  const citations = new Set((turn.evidence || []).map((source) => Number(source.citation)));
  const parts = splitThinking(turn.answer);
  if (parts.thinking) {
    answer.append(thoughtsBlock(parts.thinking,
      thinkingMs == null ? "Show thinking" : `Thought for ${formatDuration(thinkingMs)}`));
  }
  // Markdown and citations are both recognized; only evidence numbers that
  // belong to this answer become links.
  const body = document.createElement("div");
  body.className = "answer-body";
  renderMarkdown(body, parts.answer, (marker) => {
    const numbers = marker.slice(1, -1).split(",").map((value) => Number(value.trim()));
    if (!numbers.every((number) => citations.has(number))) return null;
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
    return superscript;
  });
  answer.append(body);
  if (turn.parameters?.finish_reason === "length") {
    const notice = document.createElement("p");
    notice.className = "answer-limit-notice";
    notice.setAttribute("role", "status");
    notice.textContent = "This answer reached the token limit and may be incomplete. Ask for a shorter answer, or raise Maximum answer tokens in Settings and retry.";
    answer.append(notice);
  }
}

// Markdown blocks the renderer recognizes at the start of a line.
const MARKDOWN_FENCE = /^\s*```/;
const MARKDOWN_HEADING = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
const MARKDOWN_RULE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const MARKDOWN_QUOTE = /^\s{0,3}>\s?/;
const MARKDOWN_BULLET = /^\s{0,3}[-*+]\s+/;
const MARKDOWN_NUMBER = /^\s{0,3}\d{1,3}[.)]\s+/;

function isMarkdownBlockStart(line) {
  return [MARKDOWN_FENCE, MARKDOWN_HEADING, MARKDOWN_RULE, MARKDOWN_QUOTE, MARKDOWN_BULLET, MARKDOWN_NUMBER]
    .some((pattern) => pattern.test(line));
}

/**
 * Render model text as Markdown using DOM nodes only, never HTML parsing.
 *
 * Supports headings, paragraphs, bullet and numbered lists, block quotes,
 * fenced code, rules, and inline bold, italics, and code. Plain text simply
 * becomes paragraphs.
 *
 * @param host Element that receives the blocks.
 * @param text Model output without thinking text.
 * @param linkCitation Turns a "[1, 2]" marker into a node, or null to keep
 *   it as text; omit while streaming.
 */
function renderMarkdown(host, text, linkCitation = null) {
  const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }
    if (MARKDOWN_FENCE.test(line)) {
      const code = [];
      index += 1;
      while (index < lines.length && !MARKDOWN_FENCE.test(lines[index])) code.push(lines[index++]);
      index += 1;
      const pre = document.createElement("pre");
      const element = document.createElement("code");
      element.textContent = code.join("\n");
      pre.append(element);
      host.append(pre);
      continue;
    }
    const heading = line.match(MARKDOWN_HEADING);
    if (heading) {
      // Chat answers sit inside a page, so # starts at a small heading.
      const element = document.createElement(`h${Math.min(6, heading[1].length + 2)}`);
      appendInlineMarkdown(element, heading[2], linkCitation);
      host.append(element);
      index += 1;
      continue;
    }
    if (MARKDOWN_RULE.test(line)) {
      host.append(document.createElement("hr"));
      index += 1;
      continue;
    }
    if (MARKDOWN_QUOTE.test(line)) {
      const quoted = [];
      while (index < lines.length && MARKDOWN_QUOTE.test(lines[index])) {
        quoted.push(lines[index++].replace(MARKDOWN_QUOTE, ""));
      }
      const quote = document.createElement("blockquote");
      renderMarkdown(quote, quoted.join("\n"), linkCitation);
      host.append(quote);
      continue;
    }
    const pattern = MARKDOWN_NUMBER.test(line) ? MARKDOWN_NUMBER
      : MARKDOWN_BULLET.test(line) ? MARKDOWN_BULLET : null;
    if (pattern) {
      const list = document.createElement(pattern === MARKDOWN_NUMBER ? "ol" : "ul");
      const start = line.match(/\d+/);
      if (pattern === MARKDOWN_NUMBER && start && start[0] !== "1") list.start = Number(start[0]);
      while (index < lines.length && pattern.test(lines[index])) {
        let content = lines[index++].replace(pattern, "");
        // Indented lines continue the item.
        while (index < lines.length && /^\s{2,}\S/.test(lines[index]) && !isMarkdownBlockStart(lines[index])) {
          content += `\n${lines[index++].trim()}`;
        }
        const item = document.createElement("li");
        appendInlineMarkdown(item, content, linkCitation);
        list.append(item);
      }
      host.append(list);
      continue;
    }
    const paragraph = [];
    while (index < lines.length && lines[index].trim() && !isMarkdownBlockStart(lines[index])) {
      paragraph.push(lines[index++]);
    }
    const element = document.createElement("p");
    appendInlineMarkdown(element, paragraph.join("\n"), linkCitation);
    host.append(element);
  }
}

// Inline code, bold, italics (not inside words such as snake_case), and
// citation markers such as [1] or [2, 3].
const MARKDOWN_INLINE = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*|__[^_\n]+__)|((?<![\w*])\*[^*\s][^*\n]*\*(?![\w*])|(?<![\w_])_[^_\s][^_\n]*_(?![\w_]))|(\[\d+(?:\s*,\s*\d+)*\])/g;

function appendInlineMarkdown(element, text, linkCitation) {
  let last = 0;
  for (const match of text.matchAll(MARKDOWN_INLINE)) {
    let before = text.slice(last, match.index);
    last = match.index + match[0].length;
    if (match[1]) {
      element.append(before);
      const code = document.createElement("code");
      code.textContent = match[1].slice(1, -1);
      element.append(code);
    } else if (match[2] || match[3]) {
      element.append(before);
      const node = document.createElement(match[2] ? "strong" : "em");
      appendInlineMarkdown(node, match[0].slice(match[2] ? 2 : 1, match[2] ? -2 : -1), linkCitation);
      element.append(node);
    } else {
      const link = linkCitation ? linkCitation(match[4]) : null;
      // A linked citation sits right after its claim.
      if (link) before = before.replace(/[ \t]+$/, "");
      element.append(before, link || match[4]);
    }
  }
  element.append(text.slice(last));
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
  lastConversationData = data;
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
    const asking = askingConversation?.id === item.conversation_id;
    if (item.empty && !asking) continue;
    // Until the generated title arrives, label the chat with its question.
    const provisional = asking && /^Conversation \d+$/.test(item.name);
    const name = provisional ? provisionalTitle(askingConversation.question) : item.name;
    const row = document.createElement("div");
    row.className = "conversation-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `conversation-item${item.conversation_id === activeConversationId ? " active" : ""}${provisional ? " provisional" : ""}`;
    button.dataset.conversationId = item.conversation_id;
    button.textContent = name;
    button.title = name;
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

/**
 * Shorten a first question into a temporary sidebar label.
 *
 * @param question The question as typed.
 * @returns Up to six words, without trailing punctuation.
 */
function provisionalTitle(question) {
  const words = question.trim().replace(/[?.!]+$/, "").split(/\s+/);
  return words.slice(0, 6).join(" ").slice(0, 80) + (words.length > 6 ? "…" : "");
}

/**
 * List the active empty chat as soon as its first question is sent.
 *
 * @param question The question as typed.
 */
function showAskingConversation(question) {
  const active = lastConversationData?.conversations?.find(
    (item) => item.conversation_id === activeConversationId);
  if (!active?.empty) return;
  askingConversation = { id: activeConversationId, question };
  renderConversations(lastConversationData);
}

/**
 * Stop listing an unsaved chat, for example after a failed first answer.
 */
function clearAskingConversation() {
  if (!askingConversation) return;
  askingConversation = null;
  if (lastConversationData) renderConversations(lastConversationData);
}

function renderConversationEmpty() {
  conversation.replaceChildren();
  const type = projectTypes[activeProjectKind] || {};
  const content = {
    prompts: type.welcome_prompts?.length ? type.welcome_prompts : genericEmptyChat.welcome_prompts,
    suggestions: type.suggestions?.length ? type.suggestions : genericEmptyChat.suggestions,
  };
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
  // The type's first suggestion is always shown; two others vary.
  const [first, ...rest] = content.suggestions;
  const choices = [first, ...rest.sort(() => Math.random() - 0.5).slice(0, 2)].filter(Boolean);
  for (const text of choices) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.addEventListener("click", () => {
      const question = byId("question");
      question.value = text;
      question.focus();
      // Select the "…" so typing the missing part replaces it.
      const gap = text.indexOf("…");
      if (gap >= 0) question.setSelectionRange(gap, gap + 1);
    });
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

/**
 * Choose the model level or the reference level for the next answers.
 *
 * @param kind "model" or "retrieval".
 * @param index Zero-based effort level.
 */
function setEffortLevel(kind, index) {
  const level = Math.max(0, Math.min(4, Number(index)));
  if (kind === "model") selectedModelLevel = level;
  else selectedRetrievalLevel = level;
  renderEffortPicker();
}

/**
 * Show the chosen levels on both sliders and on the composer button.
 */
function renderEffortPicker() {
  const levels = currentLevels();
  if (levels.length !== 5) return;
  // The effort is named by the model level; references are just a count.
  byId("effort-name").textContent = levels[selectedModelLevel].name;
  for (const [kind, index] of [["model", selectedModelLevel], ["retrieval", selectedRetrievalLevel]]) {
    const level = levels[index];
    byId(`effort-${kind}-detail`).textContent = kind === "model"
      ? level.model.split("/").pop()
      : `${level.retrieval} references`;
    const range = byId(`effort-${kind}-range`);
    range.value = String(index);
    range.style.setProperty("--effort-percent", `${(index / (levels.length - 1)) * 100}%`);
    range.parentElement.querySelectorAll(".effort-points i").forEach((point, pointIndex) => {
      point.classList.toggle("selected", pointIndex === index);
    });
  }
  byId("effort-button").textContent =
    `${levels[selectedModelLevel].name} · ${levels[selectedRetrievalLevel].retrieval}`;
  byId("effort-button").title =
    `${levels[selectedModelLevel].name}: ${levels[selectedModelLevel].model.split("/").pop()} · ${levels[selectedRetrievalLevel].retrieval} references`;
}

/**
 * Clear the interval used by the pending-operation label.
 */
function stopPhaseAnimation() {
  clearInterval(phaseTimer);
  phaseTimer = null;
}

// Each stage stays readable briefly. Routing and retrieval report several
// steps at once; they queue (oldest dropped past the limit) and finish well
// within the time the model spends reading before it writes.
const ANSWER_STAGE_DURATION_MS = 450;
const ANSWER_STAGE_QUEUE_LIMIT = 8;

function resetAnswerStages() {
  clearTimeout(answerStageTimer);
  answerStageTimer = null;
  answerStageKey = null;
  answerStageQueue = [];
}

function advanceAnswerStage() {
  if (answerStageTimer) return;
  const stage = answerStageQueue.shift();
  if (!stage) {
    answerStageKey = null;
    return;
  }
  answerStageKey = stage.key;
  showTurnPhase("Generating", stage.detail);
  answerStageTimer = setTimeout(() => {
    answerStageTimer = null;
    answerStageKey = null;
    advanceAnswerStage();
  }, ANSWER_STAGE_DURATION_MS);
}

function queueAnswerStage(key, detail) {
  if (!pendingTurn) return;
  markTimelineStep(detail);
  if (answerStageKey === key) return;
  if (answerStageQueue.some((stage) => stage.key === key)) return;
  answerStageQueue.push({ key, detail });
  answerStageQueue = answerStageQueue.slice(-ANSWER_STAGE_QUEUE_LIMIT);
  advanceAnswerStage();
}

/**
 * Record the moment a step started, for the expandable timeline.
 *
 * @param label Step description, as shown in the shimmering line.
 */
function markTimelineStep(label) {
  const timeline = pendingTurn?.timeline;
  if (!timeline || timeline.at(-1)?.label === label) return;
  timeline.push({ label, at: performance.now() });
}

/**
 * Build the collapsed "Took Ns" line; expanding it lists each step and how
 * long it took.
 *
 * @param current The pending turn being completed.
 * @param turn Saved turn with its routing parameters.
 * @param totalMs Time from Ask to the complete answer.
 */
function renderTimeline(current, turn, totalMs) {
  const details = document.createElement("details");
  details.className = "answer-duration";
  const summary = document.createElement("summary");
  summary.textContent = `Took ${formatDuration(totalMs)}`;
  const list = document.createElement("ol");
  list.className = "answer-timeline";
  const end = current.started + totalMs;
  const steps = (current.timeline || []).filter((step) => step.label !== "Starting");
  const query = turn.parameters?.retrieval_query;
  steps.forEach((step, index) => {
    const next = steps[index + 1]?.at ?? end;
    const row = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = step.label;
    if (query && /semantic search/i.test(step.label)) label.textContent += ` for “${query}”`;
    const time = document.createElement("span");
    time.className = "timeline-time";
    const ms = Math.max(0, next - step.at);
    time.textContent = ms < 100 ? "<0.1s" : `${(ms / 1000).toFixed(1)}s`;
    row.append(label, time);
    list.append(row);
  });
  details.append(summary, list);
  return details;
}

function completeAfterAnswerStages(turn) {
  // Freeze the timer when the answer arrives, not after the stage animation.
  if (pendingTurn && pendingTurn.elapsedMs === null) {
    pendingTurn.elapsedMs = performance.now() - pendingTurn.started;
  }
  // Show the answer as soon as it arrives; queued stage labels are dropped.
  resetAnswerStages();
  completePendingTurn(turn);
}

/**
 * Show the current backend phase under the pending answer's label.
 *
 * @param label Current backend phase label.
 * @param detail Optional supporting status text.
 */
function showTurnPhase(label, detail = "") {
  if (!pendingTurn) return;
  pendingTurn.phase = label;
  pendingTurn.detail.textContent = detail;
  pendingTurn.detail.classList.toggle("hidden", !detail);
  paintPendingTurn();
}

/**
 * Format a duration compactly: 8s, 42s, 1m 05s.
 *
 * @param ms Duration in milliseconds.
 * @returns Short human-readable duration.
 */
function formatDuration(ms) {
  const seconds = Math.max(0, Math.round(ms / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}

/**
 * Repaint the pending label (cycling dots) and its elapsed-time counter.
 */
function paintPendingTurn() {
  if (!pendingTurn) return;
  const dots = Math.floor(performance.now() / 400) % 4;
  pendingTurn.label.textContent = `${pendingTurn.phase}${".".repeat(dots)}`;
  pendingTurn.timer.textContent = formatDuration(
    pendingTurn.elapsedMs ?? performance.now() - pendingTurn.started);
  if (pendingTurn.durationLine) pendingTurn.durationLine.textContent = pendingTurn.timer.textContent;
}

/**
 * Append a pending question and begin its retrieval-phase indicator.
 *
 * @param question Validated question text submitted by the user.
 */
function addPendingTurn(question) {
  resetAnswerStages();
  activity.classList.add("answer-progress");
  if (conversation.querySelector(".conversation-empty")) conversation.replaceChildren();
  const item = document.createElement("article");
  item.className = "turn pending";
  item.dataset.turnIndex = String(conversation.children.length);
  const questionNode = document.createElement("div");
  questionNode.className = "user-message";
  questionNode.textContent = question;
  const answer = document.createElement("div");
  answer.className = "answer-message answer-phase";
  const phaseHeader = document.createElement("div");
  phaseHeader.className = "answer-phase-header";
  const phaseLabel = document.createElement("span");
  phaseLabel.className = "answer-phase-label";
  const phaseTimerLabel = document.createElement("span");
  phaseTimerLabel.className = "answer-phase-timer";
  phaseHeader.append(phaseLabel, phaseTimerLabel);
  const phaseDetail = document.createElement("div");
  phaseDetail.className = "answer-phase-detail";
  answer.append(phaseHeader, phaseDetail);
  item.append(questionNode, answer);
  conversation.append(item);
  pendingTurn = {
    item, answer, label: phaseLabel, timer: phaseTimerLabel, detail: phaseDetail,
    phase: "Generating", started: performance.now(), elapsedMs: null,
    timeline: [],
  };
  // One ticker drives both the cycling dots and the elapsed time.
  stopPhaseAnimation();
  paintPendingTurn();
  phaseTimer = setInterval(paintPendingTurn, 200);
  queueAnswerStage("starting", "Starting");
  conversationScroll.scrollTop = conversationScroll.scrollHeight;
}

/**
 * Show a tool plan with Continue, Quick, and Cancel under the pending
 * answer. The backend waits for the choice; one click answers it.
 *
 * @param plan Plan fields from qachat.tools.planning.ToolPlan.
 */
function showToolPlan(plan) {
  if (!pendingTurn) return;
  const card = document.createElement("div");
  card.className = "tool-plan";
  const summary = document.createElement("p");
  summary.className = "tool-plan-summary";
  summary.textContent = plan.summary;
  const facts = document.createElement("ul");
  const calls = plan.batches_to_run === 1 ? "1 model call" : `${plan.batches_to_run} model calls`;
  for (const line of [plan.scope, `${calls}, about ${formatDuration(plan.estimated_seconds * 1000)}`, ...(plan.details || [])]) {
    const item = document.createElement("li");
    item.textContent = line;
    facts.append(item);
  }
  const actions = document.createElement("div");
  actions.className = "tool-plan-actions";
  const choices = [
    [PLAN_CHOICE_CONTINUE, "primary", "Continue"],
    [PLAN_CHOICE_QUICK, "secondary",
      `Quick sample (${plan.quick_items}, about ${formatDuration(plan.quick_seconds * 1000)})`],
    [PLAN_CHOICE_CANCEL, "secondary", "Cancel"],
  ];
  for (const [choice, style, label] of choices) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = style;
    button.textContent = label;
    button.addEventListener("click", () => {
      card.remove();
      queueAnswerStage(`choice:${choice}`, choice === PLAN_CHOICE_CANCEL ? "Cancelling" : "Starting");
      bridge.choosePlan(choice);
    });
    actions.append(button);
  }
  card.append(summary, facts, actions);
  pendingTurn.answer.append(card);
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
  resetAnswerStages();
  updateProjectSummary(currentTurnCount + 1);
  const current = pendingTurn;
  if (!current) {
    addTurn(turn);
    setBusy(false);
    return;
  }
  pendingTurn = null;
  current.item.classList.remove("pending");
  // "Took Ns" only; expanding it shows the routing steps and their times.
  const timeline = renderTimeline(
    current, turn, current.elapsedMs ?? performance.now() - current.started);
  if (current.durationLine) current.durationLine.replaceWith(timeline);
  else current.item.insertBefore(timeline, current.answer);
  current.durationLine = timeline;
  // The text was shown live as it was generated; now add citation links.
  current.answer.classList.remove("answer-phase", "streaming");
  current.answer.replaceChildren();
  bindAnswerSelection(current.answer, current.item, turn);
  renderAnswer(current.answer, turn, current.item, current.thinkingMs);
  selectTurn(current.item, turn);
  conversationScroll.scrollTop = conversationScroll.scrollHeight;
  setBusy(false);
}

/**
 * Split model output into its thinking and its answer.
 *
 * Thinking appears inside <think>…</think>; a chat template may open the
 * block in the prompt, leaving only the closing tag in the output.
 *
 * @param raw Model output so far.
 * @returns Thinking text, answer text, and whether thinking is still open.
 */
function splitThinking(raw) {
  let thinking = "";
  let text = String(raw || "").replace(/<think>([\s\S]*?)<\/think>/g, (_, inner) => {
    thinking += inner;
    return "";
  });
  const close = text.indexOf("</think>");
  if (close !== -1) {
    thinking = text.slice(0, close) + thinking;
    text = text.slice(close + "</think>".length);
  }
  const open = text.indexOf("<think>");
  if (open !== -1) {
    thinking += text.slice(open + "<think>".length);
    text = text.slice(0, open);
  }
  return { thinking: thinking.trim(), answer: text.replace(/^\s+/, ""), thinkingOpen: open !== -1 };
}

/**
 * Build the collapsible box that shows the model's thinking.
 *
 * @param text Thinking text.
 * @param label Summary line, such as "Thought for 12s".
 * @param open Whether the box starts expanded.
 */
function thoughtsBlock(text, label, open = false) {
  const details = document.createElement("details");
  details.className = "answer-thoughts";
  details.open = open;
  const summary = document.createElement("summary");
  summary.textContent = label;
  // Expanding the thoughts should not also select the answer.
  summary.addEventListener("click", (event) => event.stopPropagation());
  const body = document.createElement("div");
  body.className = "answer-thoughts-text";
  body.textContent = text;
  details.append(summary, body);
  return details;
}

/**
 * Name an intent for display, using the project type's list.
 */
function intentName(intentId) {
  const names = projectTypes[activeProjectKind]?.intents || {};
  return names[intentId] || String(intentId || "").replaceAll("_", " ");
}

/**
 * Describe which earlier turns the answer sees, such as "Recent 2 + turn 3".
 */
function describeHistory(data) {
  const ids = data.history_turn_ids || [];
  if (!ids.length) return "No earlier conversation needed";
  const related = data.related_turn_ids || [];
  const recent = ids.filter((id) => !related.includes(id));
  const parts = [];
  if (recent.length) parts.push(`the last ${recent.length === 1 ? "turn" : `${recent.length} turns`}`);
  if (related.length) parts.push(`turn${related.length === 1 ? "" : "s"} ${related.join(", ")}`);
  return `Including ${parts.join(" and ")} from the conversation`;
}

/**
 * Show generated text live in the pending answer.
 *
 * @param text Newly generated text.
 */
function appendAnswerText(text) {
  const current = pendingTurn;
  if (!current || !text) return;
  current.raw = (current.raw || "") + text;
  const parts = splitThinking(current.raw);
  const thinkingStarted = parts.thinking || parts.thinkingOpen;
  if (!current.streaming && !thinkingStarted && !parts.answer) return;
  if (!current.streaming) {
    current.streaming = true;
    resetAnswerStages();
    markTimelineStep("Writing the answer");
    // Live elapsed time above the answer; it becomes "Took …" at the end.
    current.durationLine = document.createElement("div");
    current.durationLine.className = "answer-duration";
    current.item.insertBefore(current.durationLine, current.answer);
    current.answer.classList.remove("answer-phase");
    current.answer.classList.add("streaming");
    current.answer.replaceChildren();
    current.body = document.createElement("div");
    current.body.className = "answer-body";
    current.answer.append(current.body);
    paintPendingTurn();
  }
  if (thinkingStarted) {
    if (!current.thoughts) {
      current.thinkingStarted = performance.now();
      current.thoughts = thoughtsBlock("", "Thinking…", true);
      current.answer.prepend(current.thoughts);
    }
    const body = current.thoughts.querySelector(".answer-thoughts-text");
    body.textContent = parts.thinking;
    body.scrollTop = body.scrollHeight;
    if (!parts.thinkingOpen && current.thinkingMs == null) {
      current.thinkingMs = performance.now() - current.thinkingStarted;
      current.thoughts.open = false;
      current.thoughts.querySelector("summary").textContent =
        `Thought for ${formatDuration(current.thinkingMs)}`;
    }
  }
  current.body.replaceChildren();
  renderMarkdown(current.body, parts.answer);
  const nearBottom = conversationScroll.scrollHeight - conversationScroll.scrollTop
    - conversationScroll.clientHeight < 120;
  if (nearBottom) conversationScroll.scrollTop = conversationScroll.scrollHeight;
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
    card.classList.toggle("hidden", data.ready && !setupCompletionVisible);
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
    setupCompletionVisible = false;
    setBusy(true, `Preparing ${data.total} required models…`);
    byId("first-run-setup").classList.remove("hidden");
    byId("install-required").disabled = true;
    byId("readiness-time").textContent = "Starting download. Remaining time depends on model size and network speed.";
    updateDownloadMeter(null);
  } else if (data.event === "installation_progress") {
    const progress = downloadProgress(data);
    const percent = progress.percent == null ? "" : ` · ${progress.percent.toFixed(1)}%`;
    byId("readiness-summary").textContent = `Downloading ${data.index} of ${data.total}${percent}`;
    byId("readiness-time").textContent = `${progress.text} · ${data.model_id}`;
    byId("activity-percent").textContent = progress.percent == null
      ? `Model ${data.index} of ${data.total}`
      : `${progress.percent.toFixed(1)}% · model ${data.index} of ${data.total}`;
    byId("activity-text").textContent = `Downloading ${data.model_id}`;
    byId("activity-detail").textContent = progress.text;
    updateDownloadMeter(progress.percent);
  } else if (data.event === "installation_model_complete") {
    cachedModels.add(data.model_id);
    byId("readiness-time").textContent = `${data.model_id} completed in ${seconds(data.model_seconds)} · ${seconds(data.elapsed_seconds)} total`;
  } else if (data.event === "installation_complete") {
    setupCompletionVisible = true;
    byId("readiness-summary").textContent = "Required models are installed.";
    byId("readiness-time").textContent = `Setup completed in ${seconds(data.elapsed_seconds)}.`;
    updateDownloadMeter(100);
    setBusy(false);
  } else if (data.event === "settings") {
    showSettings(data);
  } else if (data.event === "projects") {
    renderSavedProjects(data.projects || []);
  } else if (data.event === "conversation_delete_started") {
    const button = [...document.querySelectorAll(".conversation-item")]
      .find((item) => item.dataset.conversationId === data.conversation_id);
    button?.closest(".conversation-row")?.remove();
    byId("activity-text").textContent = "Conversation removed. Preserving its trace and opening the next chat…";
    byId("activity-detail").textContent = "The archived trace remains in logs/traces/deleted.";
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
  } else if (data.event === "answer_delta") {
    appendAnswerText(data.text || "");
  } else if (data.event === "answer_phase") {
    activity.classList.add("answer-progress");
    if (data.phase === ANSWER_PHASE_ROUTING) {
      queueAnswerStage("intent", "Identifying intent");
      byId("activity-text").textContent = "Identifying what the question asks for";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_ROUTED) {
      queueAnswerStage("intent-done", `${intentName(data.intent)} identified`);
      queueAnswerStage("history", describeHistory(data));
      // Retrieval steps follow from their own phases; "none" shows none.
    } else if (data.phase === ENGINE_PHASE_LOADING_MODEL) {
      const name = String(data.model_id || "the model").split("/").pop();
      queueAnswerStage("loading", `Loading ${name}`);
      byId("activity-text").textContent = `Loading ${name}`;
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_RETRIEVING) {
      // Show what retrieval is doing (semantic search, lexical search, …);
      // generic and result-count messages are skipped.
      const message = data.message || "Searching project evidence";
      if (/^(Embedding|Running|Searching|Checking|Combining|Expanding|Finding|Scanning)\b/.test(message)) {
        queueAnswerStage(`retrieval:${message}`, message.replace(/^Embedding the query and running /, "Running "));
      }
      byId("activity-text").textContent = data.message || "Searching project evidence";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_PROCESSING && data.plan) {
      // An expensive tool run waits for Continue, Quick, or Cancel.
      queueAnswerStage("plan", "Waiting for your choice");
      showToolPlan(data.plan);
      byId("activity-text").textContent = "Waiting for your choice";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_PROCESSING) {
      queueAnswerStage(`processing:${data.current ?? ""}`, data.message || "Working");
      byId("activity-text").textContent = data.message || "Working";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_EVIDENCE_READY && data.skipped) {
      // No retrieval needed: no step to show.
    } else if (
      data.phase === ANSWER_PHASE_EVIDENCE_READY &&
      (data.document !== undefined || data.tool !== undefined)
    ) {
      // Document reading or a tool describes its own result.
      queueAnswerStage("preparing", data.message || "Reading the document");
      byId("activity-text").textContent = data.message || "Reading the document";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_EVIDENCE_READY && data.reused) {
      queueAnswerStage("preparing", `Reusing ${data.evidence_count} references from earlier turns`);
      byId("activity-text").textContent = "Follow-up: reusing the previous references";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_EVIDENCE_READY) {
      queueAnswerStage(
        "preparing",
        data.evidence_count == null
          ? "Preparing retrieved evidence"
          : data.related_count
            ? `Found ${data.evidence_count - data.related_count} references + ${data.related_count} from earlier turns`
            : `Found ${data.evidence_count} references`,
      );
      byId("activity-text").textContent = data.evidence_count == null
        ? "Preparing retrieved evidence for the model"
        : `Preparing ${data.evidence_count} retrieved references for the model`;
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_GENERATING) {
      // The model reads its whole prompt before the first word appears;
      // streamed text replaces this label.
      queueAnswerStage(
        "reading",
        data.evidence_count
          ? `Reading ${data.evidence_count} references`
          : "Reading the conversation",
      );
      byId("activity-text").textContent = data.evidence_count == null
        ? "The model is reading the prompt"
        : `The model is reading ${data.evidence_count} references before writing`;
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_SAVING) {
      queueAnswerStage("saving", data.message || "Saving the answer");
      byId("activity-text").textContent = data.message || "Saving this turn to the conversation";
      byId("activity-detail").textContent = "";
    } else if (data.phase === ANSWER_PHASE_COMPLETED) {
      stopPhaseAnimation();
      byId("activity-text").textContent = data.message || "Displaying the completed response";
      byId("activity-detail").textContent = "";
    }
  } else if (data.event === "model_download_started") {
    byId("model-download-status").textContent = `Downloading ${data.model_id}…`;
  } else if (data.event === "model_download_progress") {
    const progress = downloadProgress(data);
    byId("model-download-status").textContent = `${data.model_id} · ${progress.text}`;
    byId("activity-percent").textContent = progress.percent == null
      ? "Connecting…"
      : `${progress.percent.toFixed(1)}% downloaded`;
    byId("activity-text").textContent = `Downloading ${data.model_id}`;
    byId("activity-detail").textContent = progress.text;
  } else if (data.event === "models_applied") {
    applyingModels = false;
    cachedModels = new Set(data.cached_models || [...cachedModels]);
    modelRoles = data.model_roles || modelRoles;
    // Without an open project, keep the levels for project creation.
    if (data.effort_levels) savedLevels = data.effort_levels;
    else savedLevels = { ...savedLevels, [activeBackend]: cloneLevels(draftLevels) };
    if (data.router_models) savedRouters = data.router_models;
    else savedRouters = { ...savedRouters, [activeBackend]: draftRouter };
    if (data.profile) { profile = data.profile; byId("query-format").value = profile.query_format; showModelFields(); }
    else saveModelFields();
    modelsBaseline = currentModelSettings();
    byId("model-download-status").textContent = "Model settings applied.";
    setBusy(false);
    renderModelsPanel();
    if (closeAfterModelsApply) hideModels();
    closeAfterModelsApply = false;
  } else if (data.event === "model_downloaded") {
    cachedModels.add(data.model_id);
    downloadingModel = null;
    byId("cancel-model-download").classList.add("hidden");
    byId("model-download-status").textContent = `${data.model_id} downloaded in ${seconds(data.elapsed_seconds || 0)}.`;
    if (!applyingModels) setBusy(false);
    refreshModelStatus();
    if (!byId("models-panel").classList.contains("hidden")) renderModelsPanel();
  } else if (data.event === "progress") {
    showProgress(data);
  } else if (data.event === "opened") {
    ready = true;
    if (data.profile) {
      profile = data.profile;
      byId("query-format").value = profile.query_format;
      savedLevels = data.effort_levels || savedLevels;
      savedRouters = data.router_models || savedRouters;
      showModelFields();
    }
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
    resetAnswerStages();
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
    completeAfterAnswerStages(data.turn);
    if (data.workspace_files) renderWorkspace(data.workspace_files);
  } else if (data.event === "reset") {
    renderConversationEmpty();
    clearReferences();
    updateProjectSummary(0);
    updateFloatingHeader();
    setBusy(false);
  } else if (data.event === "stopped") {
    setupCompletionVisible = false;
    byId("readiness-meter").classList.add("hidden");
    byId("readiness-meter").classList.remove("indeterminate");
    if (downloadingModel) {
      byId("model-download-status").textContent = "Download stopped. Reopen Models to check what is cached.";
      downloadingModel = null;
      byId("cancel-model-download").classList.add("hidden");
    }
    if (resumeAfterStop) {
      resumeAfterStop = false;
      stopPhaseAnimation();
      resetAnswerStages();
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
  } else if (data.event === "backend_restarted") {
    const reopenProject = ready
      && byId("workspace-path").value.trim()
      && activeConversationId;
    error.classList.add("hidden");
    setupCompletionVisible = true;
    byId("readiness-time").textContent = data.message;
    byId("model-download-status").textContent = data.message;
    downloadingModel = null;
    byId("cancel-model-download").classList.add("hidden");
    ready = false;
    setBusy(false);
    byId("connection").textContent = "Ready";
    bridge.requestSettings(byId("kind").value, byId("workspace-path").value.trim());
    bridge.requestReadiness();
    if (reopenProject) {
      setBusy(true, "Reopening the project after backend recovery…");
      bridge.resumeProjectConversation(
        byId("workspace-path").value.trim(),
        activeConversationId,
      );
    }
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
  if (!busy || !ready || !bridge || !activity.classList.contains("answer-progress")) return;
  event.preventDefault();
  resumeAfterStop = true;
  byId("send").disabled = true;
  byId("activity-text").textContent = "Stopping generation…";
  bridge.stopOperation();
});
byId("open-models").addEventListener("click", openModels);
byId("models-hint-button").addEventListener("click", openModels);
byId("close-models").addEventListener("click", closeModels);
byId("models-panel").addEventListener("click", (event) => {
  if (event.target === byId("models-panel")) closeModels();
});
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
  if (![byId("chat-model").value.trim(), byId("embedding-model").value.trim()].every(localModel)) {
    openModels();
    byId("model-download-status").textContent = "Download the selected models in Settings before preparing the project.";
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
  showAskingConversation(question);
  setBusy(true, "Finding evidence and answering…");
  const attachedContext = messageAttachments.length
    ? `\n\nAttached paths to consider:\n${messageAttachments.map((path) => `- ${path}`).join("\n")}`
    : "";
  messageAttachments = [];
  renderAttachments();
  const effort = JSON.stringify({ model: selectedModelLevel, retrieval: selectedRetrievalLevel });
  bridge.askQuestion(`${question}${attachedContext}`, effort, JSON.stringify({
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
byId("topbar-project-name").addEventListener("input", () => {
  byId("project-name").textContent = byId("topbar-project-name").textContent;
});
for (const [id, commit] of [["conversation-name", commitConversationName], ["project-name", commitProjectName], ["topbar-project-name", commitProjectName]]) {
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
byId("effort-model-range").addEventListener("input", (event) => {
  setEffortLevel("model", event.target.value);
});
byId("effort-retrieval-range").addEventListener("input", (event) => {
  setEffortLevel("retrieval", event.target.value);
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
  if (event.key === "Escape") {
    closeModelMenus();
    if (!byId("models-panel").classList.contains("hidden")) closeModels();
  }
});
for (const id of ["extensions", "max-file-bytes"]) byId(id).addEventListener("change", previewSources);

// Attach the bridge only after Qt exposes the channel; then request state.
new QWebChannel(qt.webChannelTransport, (channel) => {
  bridge = channel.objects.qachat;
  bridge.setAppearance(document.documentElement.dataset.appearance || "system");
  bridge.event.connect(onEvent);
  bridge.pathsSelected.connect(addSources);
  bridge.sourceDragChanged.connect(showNativeSourceDrag);
  syncSourceDropTarget();
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
  renderEffortPicker();
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

// Draggable borders beside the conversation list and the side panels.
const PANEL_LIMITS = { left: [180, 420], right: [260, 640] };
const MIDDLE_MIN_WIDTH = 420;
const panelResizers = {};

/**
 * Apply a remembered or dragged panel width, within its limits.
 *
 * @param side "left" (conversations) or "right" (workspace and references).
 * @param width Requested width in pixels; null restores the default.
 */
function setPanelWidth(side, width) {
  const property = side === "left" ? "--left-width" : "--right-width";
  if (width === null) {
    shell.style.removeProperty(property);
    return;
  }
  const [min, max] = PANEL_LIMITS[side];
  shell.style.setProperty(property, `${Math.round(Math.min(max, Math.max(min, width)))}px`);
}

/**
 * Place each handle on its visible border, or hide it.
 */
function positionPanelResizers() {
  const grid = getComputedStyle(shell).display === "grid" && shell.classList.contains("chat-open");
  const left = byId("conversations-panel");
  const right = document.querySelector(".side-panels");
  for (const [side, panel] of [["left", left], ["right", right]]) {
    const handle = panelResizers[side];
    const box = panel.getBoundingClientRect();
    const visible = grid && box.width > 0 && getComputedStyle(panel).display !== "none";
    handle.classList.toggle("hidden", !visible);
    if (visible) handle.style.left = `${(side === "left" ? box.right : box.left) - 3}px`;
  }
}

for (const side of ["left", "right"]) {
  const handle = document.createElement("div");
  handle.className = "panel-resizer hidden";
  handle.setAttribute("role", "separator");
  handle.setAttribute("aria-orientation", "vertical");
  handle.title = "Drag to resize; double-click to reset";
  let lastPress = 0;
  const reset = () => {
    setPanelWidth(side, null);
    try { localStorage.removeItem(`qachat-${side}-width`); } catch {}
    positionPanelResizers();
    syncSourceDropTarget();
  };
  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    // Detect double-clicks here: pointer capture can suppress dblclick.
    if (event.timeStamp - lastPress < 400) {
      lastPress = 0;
      reset();
      return;
    }
    lastPress = event.timeStamp;
    handle.setPointerCapture(event.pointerId);
    handle.classList.add("dragging");
    document.body.classList.add("resizing-panels");
  });
  handle.addEventListener("pointermove", (event) => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    lastPress = 0;  // A drag is not the first half of a double-click.
    const other = side === "left"
      ? document.querySelector(".side-panels").getBoundingClientRect().width
      : byId("conversations-panel").getBoundingClientRect().width;
    const requested = side === "left" ? event.clientX : window.innerWidth - event.clientX;
    // Keep the conversation column usable.
    setPanelWidth(side, Math.min(requested, window.innerWidth - other - MIDDLE_MIN_WIDTH));
    positionPanelResizers();
  });
  const finish = (event) => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    handle.releasePointerCapture(event.pointerId);
    handle.classList.remove("dragging");
    document.body.classList.remove("resizing-panels");
    try {
      localStorage.setItem(`qachat-${side}-width`, shell.style.getPropertyValue(`--${side}-width`));
    } catch {}
    syncSourceDropTarget();
  };
  handle.addEventListener("pointerup", finish);
  handle.addEventListener("pointercancel", finish);
  document.body.append(handle);
  panelResizers[side] = handle;
  try {
    const saved = parseFloat(localStorage.getItem(`qachat-${side}-width`));
    if (Number.isFinite(saved)) setPanelWidth(side, saved);
  } catch {}
}
// Panels appear, hide, and change size on many events; follow them all.
const panelObserver = new ResizeObserver(positionPanelResizers);
panelObserver.observe(shell);
panelObserver.observe(byId("conversations-panel"));
panelObserver.observe(document.querySelector(".side-panels"));
window.addEventListener("resize", positionPanelResizers);
