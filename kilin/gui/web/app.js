/**
 * Present the local KíLín interface inside Qt WebEngine.
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
// Answer phases sent by RAGSession (kilin.rag.ANSWER_PHASE_*); the strings
// are the core's wire values and must stay identical to it.
const ANSWER_PHASE_ROUTING = "routing";
const ANSWER_PHASE_ROUTED = "routed";
const ANSWER_PHASE_RETRIEVING = "retrieving";
const ANSWER_PHASE_EVIDENCE_READY = "evidence_ready";
const ANSWER_PHASE_GENERATING = "generating";
const ANSWER_PHASE_SAVING = "saving";
const ANSWER_PHASE_COMPLETED = "completed";
const ANSWER_PHASE_PROCESSING = "processing";
// Mirrors kilin.tools.planning.PlanChoice.
const PLAN_CHOICE_CONTINUE = "continue";
const PLAN_CHOICE_QUICK = "quick";
const PLAN_CHOICE_CANCEL = "cancel";
// Phase added by the GUI engine (kilin.gui.engine.ENGINE_PHASE_*).
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
// The Settings status text that showError last wrote, cleared with it.
let shownErrorStatus = "";
const selectedSources = [];
const previewRoots = new Map();
const previewSkipped = new Map();
// Codes arrive from the backend's validated, packaged JSON resource.
let selectionReasonCodes = null;
const checkedFiles = new Map();
const expandedFolders = new Set();
// Workspace folders that start open but the user has folded.
const collapsedWorkspaceFolders = new Set();
// Paths shown in the workspace tree, to warn before replacing files.
let workspacePaths = new Set();
// The user's last show/hide choice for the side panels; null before any.
let sidePanelChoice = null;
// Workspace folders the user edits directly; neither is indexed.
const EDITABLE_FOLDERS = ["outputs", "templates"];
// Drag data for a workspace file dragged within the panel.
const WORKSPACE_DRAG_TYPE = "application/x-kilin-workspace-path";
let profile = null;
let cachedModels = new Set();
let modelChoices = [];
// Effort levels per backend: saved for the project, built-in defaults, and
// the Settings panel's unsaved copy for the active backend.
let savedLevels = {};
let defaultLevels = {};
let draftLevels = [];
let defaultEmbeddings = {};
// Search depth: passages retrieved at each of five steps (project-wide).
let savedDepth = null;
let defaultDepth = [5, 8, 10, 15, 20];
let draftDepth = [];
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
// A stopped answer's partial text, shown again after the engine restarts.
let stoppedTurn = null;
let phaseTimer = null;
let answerTimer = null;
// Tokens used by the open chat and the project, from the engine; the
// running answer's tokens are added live while it is generated.
let usageTotals = null;
let answerStageTimer = null;
let answerStageKey = null;
let answerStageQueue = [];
let selectedModelLevel = 2;
let selectedDepthStep = 2;
let resumeAfterStop = false;
let activeProjectTypeId = "";
const lastEmptyPrompt = Object.create(null);
let messageAttachments = [];
let currentTurnCount = 0;
// Project types sent by the engine from kilin/project_types/*.json.
let projectTypes = Object.create(null);
// Saved projects last listed, shown again once the types' names arrive.
let savedProjects = [];
// Used only if a type lists no welcome prompts or suggestions.
const genericEmptyChat = {
  welcome_prompts: ["What would you like to work on?"],
  suggestions: ["Explain a concept", "Develop an idea", "Compare two approaches"],
};

/**
 * Rebuild the project type dropdown, keeping the current choice.
 *
 * @param types Project types sent by the engine.
 */
function renderProjectTypes(types) {
  if (!Array.isArray(types) || !types.length) return;
  projectTypes = Object.fromEntries(types.map((item) => [item.type_id, item]));
  if (savedProjects.length) renderSavedProjects(savedProjects);
  const select = byId("type-id");
  const current = select.value;
  select.replaceChildren(...types.map((item) => {
    const option = document.createElement("option");
    option.value = item.type_id;
    option.textContent = item.label;
    return option;
  }));
  // The first type in registry filename order is the initial default.
  select.value = projectTypes[current] ? current : types[0].type_id;
  activeProjectTypeId = ready ? activeProjectTypeId : select.value;
}

/**
 * Format the number of conversation turns for display.
 *
 * @param count Number of conversation turns.
 * @returns Singular or plural turn-count label.
 */
function turnCountLabel(count) {
  return `${count} ${count === 1 ? "turn" : "turns"}`;
}

/**
 * Refresh the header and project summary from the saved turn count.
 *
 * @param count Latest saved count; defaults to the current count.
 */
function updateProjectSummary(count = currentTurnCount) {
  currentTurnCount = Number(count) || 0;
  const projectName = byId("project-name").textContent.trim();
  byId("topbar-project-name").textContent = projectName;
  byId("topbar-turn-count").textContent = turnCountLabel(currentTurnCount);
  byId("project-detail").textContent = `${currentTurnCount} saved ${currentTurnCount === 1 ? "turn" : "turns"}`;
}

/**
 * Keep the compact project header visible after scrolling the conversation.
 */
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

/**
 * Show or hide the side panels because the user asked, and remember it.
 *
 * Automatic layout changes (opening a project, a new chat) then follow
 * this choice instead of overriding it.
 *
 * @param visible Whether the user wants the side panels shown.
 */
function chooseSidePanels(visible) {
  sidePanelChoice = visible;
  showWorkspace(visible);
}

/**
 * Show or hide the conversations panel and synchronize its toggle.
 *
 * @param visible Whether the conversations panel should be visible.
 */
function showConversations(visible) {
  shell.classList.toggle("conversations-hidden", !visible);
  const control = byId("toggle-conversations");
  control.classList.toggle("hidden", !ready);
  control.classList.toggle("panels-hidden", !visible);
  control.setAttribute("aria-label", visible ? "Hide conversations" : "Show conversations");
  control.title = control.getAttribute("aria-label");
}

/**
 * Apply and persist the page appearance preference.
 *
 * @param value Requested theme: system, light, or dark.
 */
function applyAppearance(value) {
  const selected = ["system", "light", "dark"].includes(value) ? value : "system";
  document.documentElement.dataset.appearance = selected;
  localStorage.setItem("kilin-appearance", selected);
  if (byId("appearance")) byId("appearance").value = selected;
  bridge?.setAppearance(selected);
}

/**
 * Render the resumable projects provided by the backend.
 *
 * @param projects Saved projects ordered by recent use.
 */
function renderSavedProjects(projects) {
  savedProjects = projects;
  const host = byId("saved-projects");
  host.replaceChildren();
  if (projects.length) {
    // One list of every project type, most recently used first (engine order).
    const group = document.createElement("div");
    group.className = "saved-project-group";
    const heading = document.createElement("strong");
    heading.textContent = "Resume a previous project";
    group.append(heading);
    for (const project of projects) {
      const button = document.createElement("button");
      button.type = "button";
      const name = document.createElement("span");
      name.className = "saved-project-name";
      name.textContent = `Resume ${project.name}`;
      const typeLabel = document.createElement("span");
      typeLabel.className = "saved-project-type";
      // The type's own name, or its ID until the types have arrived.
      typeLabel.textContent = projectTypes[project.type_id]?.label || project.type_id;
      button.append(name, typeLabel);
      button.title = `Resume ${project.type_id} project from ${project.project}`;
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
    if (!confirm(`Add these sources?\n\n${names}\n\nKíLín copies them into sources and updates the index; this may take a while.`)) return;
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
 * Add "(1 folder, 25 files)" after a workspace folder's name: what the
 * folder directly contains, not its nested levels.
 *
 * @param folder A tree-folder element built by branch().
 */
function showFolderCount(folder) {
  const summary = folder.querySelector(":scope > summary");
  const name = [...summary.childNodes].find((node) => node.nodeType === Node.TEXT_NODE);
  if (!name) return;
  const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;
  const folders = Number(folder.dataset.folders || 0);
  const files = Number(folder.dataset.files || 0);
  const parts = [
    folders ? plural(folders, "folder") : "",
    files ? plural(files, "file") : "",
  ].filter(Boolean);
  const count = document.createElement("span");
  count.className = "tree-count";
  count.textContent = ` (${parts.join(", ") || "empty"})`;
  name.after(count);
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
 * @param selectable True gives each row a checkbox (the source preview).
 *   False gives plain rows (the workspace tree), so clicking a file's
 *   name does not also click its delete button.
 * @returns New details element containing the branch.
 */
function branch(name, children, paths, selection, depth = 0, enabled = null, reasons = {}, selectable = true) {
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
  // Direct contents only (not nested levels), shown by the workspace tree.
  details.dataset.folders = String(children.directories.size);
  details.dataset.files = String(children.files.length);
  const summary = document.createElement("summary");
  if (selectable) {
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
    summary.append(checkbox);
  }
  summary.append(document.createTextNode(name));
  details.append(summary);
  const content = document.createElement("div");
  content.className = "tree-children";
  for (const [folder, node] of [...children.directories].sort()) {
    content.append(branch(folder, node, allFiles(node), selection, depth + 1, enabled, reasons, selectable));
  }
  for (const file of children.files.sort()) {
    // A label forwards clicks to its first control, so plain rows are divs.
    const row = document.createElement(selectable ? "label" : "div");
    row.className = "tree-file";
    row.dataset.path = file;
    if (selectable) {
      const check = document.createElement("input");
      check.type = "checkbox";
      check.checked = selection.has(file);
      check.disabled = enabled !== null && !enabled.has(file);
      check.addEventListener("change", () => {
        check.checked ? selection.add(file) : selection.delete(file);
        renderPreview();
      });
      row.append(check);
    }
    row.append(document.createTextNode(file.split("/").pop()));
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
    // Trailing slashes represent real empty or nonempty folders, not files.
    const directory = file.endsWith("/");
    const parts = file.split("/").filter(Boolean);
    let node = root;
    const folders = directory ? parts : parts.slice(0, -1);
    for (const folder of folders) {
      if (!node.directories.has(folder)) node.directories.set(folder, { directories: new Map(), files: [] });
      node = node.directories.get(folder);
    }
    if (!directory) node.files.push(file);
  }
  return root;
}

/**
 * Identify files that may be included by explicitly raising the size limit.
 *
 * @param item Skipped-file preview row with a reason code.
 * @returns Whether explicit selection can override the size limit.
 */
function isOversizedSkippedFile(item) {
  return selectionReasonCodes !== null
    && item.reason_code === selectionReasonCodes.FILE_TOO_LARGE;
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
    const oversized = skipped.filter((item) => isOversizedSkippedFile(item) && Number(item.size_bytes) > 0);
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
      isOversizedSkippedFile(item)
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
const MAX_SEARCH_DEPTH = 200;

const cloneLevels = (levels) => (levels || []).map((level) => ({ ...level }));

/**
 * Return the saved effort levels for the active backend.
 *
 * @returns Five {name, model} records, or none before settings load.
 */
function currentLevels() {
  return savedLevels[activeBackend] || defaultLevels[activeBackend] || [];
}

/**
 * Return the saved search-depth steps (passages per step).
 */
function currentDepth() {
  return savedDepth || defaultDepth;
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
 * @param ids Model IDs offered in the dropdown.
 * @param value Currently chosen model ID.
 * @param label Accessible label of the model selector.
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
 * @param id Model identifier shown by this row.
 * @param installed Whether the chosen backend is installed.
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
 * @param role Model role whose defaults can be restored.
 * @param title Visible model-group heading.
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
    : `${activeBackend.toUpperCase()} packages are not installed. Run uv sync --extra gui --extra ${activeBackend} from the KíLín folder first.`;
  byId("models-backend").classList.toggle("hidden", Boolean(installed));

  const chat = document.createElement("section");
  chat.className = "model-group";
  chat.append(modelGroupHeader("chat", "Effort levels"));
  if (!mlx) {
    // PyTorch cannot switch models per question.
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "PyTorch answers every level with one model.";
    chat.append(hint);
  }
  const columns = document.createElement("div");
  columns.className = "level-row level-columns";
  columns.setAttribute("aria-hidden", "true");
  for (const label of ["Level", "Language model"]) {
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
    row.append(name, select);
    chat.append(row);
    const notice = downloadNotice(level.model, installed);
    if (notice) chat.append(notice);
  });
  host.append(chat);

  // Search depth: how many passages each step of the composer's second
  // slider retrieves.
  const depth = document.createElement("section");
  depth.className = "model-group";
  depth.append(modelGroupHeader("depth", "Search depth"));
  const steps = document.createElement("div");
  steps.className = "depth-row";
  draftDepth.forEach((value, index) => {
    const cell = document.createElement("label");
    cell.className = "depth-step";
    const caption = document.createElement("span");
    caption.textContent = `Step ${index + 1}`;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "1";
    input.max = String(MAX_SEARCH_DEPTH);
    input.step = "1";
    input.value = String(value);
    input.disabled = busy;
    input.dataset.focusKey = `depth-${index}`;
    input.setAttribute("aria-label", `Search depth step ${index + 1}, in passages`);
    input.addEventListener("input", () => {
      draftDepth[index] = Number(input.value);
      refreshRestoreButtons();
    });
    cell.append(caption, input);
    steps.append(cell);
  });
  depth.append(steps);
  host.append(depth);

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
  // From defaults.json (sent by the engine); the catalog is a fallback.
  return defaultEmbeddings[activeBackend]
    || compatibleChoices("embedding")[0]?.model_id
    || backendChoices("embedding")[0]?.model_id || "";
}

/**
 * Report whether a group differs from its built-in defaults.
 *
 * @param role chat (effort levels), depth (search depth), router, or
 *   embedding.
 * @returns True when restoring defaults would change something.
 */
function hasModelChanges(role) {
  if (role === "chat") {
    return JSON.stringify(draftLevels) !== JSON.stringify(defaultLevels[activeBackend] || []);
  }
  if (role === "depth") return JSON.stringify(draftDepth) !== JSON.stringify(defaultDepth);
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
  } else if (role === "depth") {
    draftDepth = [...defaultDepth];
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
    draftDepth = [...currentDepth()];
    draftRouter = currentRouter();
    modelsBaseline = currentModelSettings();
  }
  renderModelsPanel();
  byId("models-panel").classList.remove("hidden");
  byId("close-models").focus();
}

/**
 * Collect the unsaved model, router, and retrieval choices.
 *
 * @returns Current settings object to send to the engine.
 */
function currentModelSettings() {
  return {
    chat_model: draftLevels[DEFAULT_LEVEL]?.model || byId("chat-model").value.trim(),
    embedding_model: byId("embedding-model").value.trim(),
    query_format: byId("query-format").value,
    backend: activeBackend,
    effort_levels: cloneLevels(draftLevels),
    search_depth: [...draftDepth],
    router_model: draftRouter,
  };
}

/**
 * Report whether Settings hold edits that are not saved yet.
 */
function modelEditsPending() {
  const settings = currentModelSettings();
  const before = modelsBaseline || settings;
  return JSON.stringify(settings.effort_levels) !== JSON.stringify(before.effort_levels)
    || JSON.stringify(settings.search_depth) !== JSON.stringify(before.search_depth)
    || settings.embedding_model !== before.embedding_model || settings.query_format !== before.query_format
    || settings.router_model !== before.router_model;
}
/**
 * Close the model settings panel and refresh source-drop bounds.
 */
function hideModels() {
  setupCompletionVisible = false;
  byId("models-panel").classList.add("hidden");
  syncSourceDropTarget();
}

/**
 * Explain the first invalid effort level, or return an empty string.
 * @param levels Current draft effort-level settings.
 * @param depth Current draft retrieval-depth steps.
 */
function levelProblem(levels, depth) {
  for (const [index, level] of levels.entries()) {
    if (!level.name.trim()) return `Give effort level ${index + 1} a name.`;
  }
  for (const [index, value] of depth.entries()) {
    if (!Number.isInteger(value) || value < 1 || value > MAX_SEARCH_DEPTH) {
      return `Search depth step ${index + 1} must be a whole number from 1 to ${MAX_SEARCH_DEPTH}.`;
    }
  }
  return "";
}

/**
 * Validate pending model changes and request that the engine apply them.
 *
 * @param closeAfter Whether to close Settings after applying the changes.
 * @param explicitDownload Model ID to download, or null for ordinary edits.
 */
function applyModelEdits(closeAfter = false, explicitDownload = null) {
  if (busy || !bridge) return;
  const settings = currentModelSettings();
  const before = modelsBaseline || settings;
  const changed = modelEditsPending();
  if (!changed && !explicitDownload) { if (closeAfter) hideModels(); return; }
  const problem = levelProblem(settings.effort_levels, settings.search_depth);
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
/**
 * Close Settings if possible, applying confirmed edits when idle.
 */
function closeModels() {
  if (document.activeElement?.matches(".model-id-input")) document.activeElement.blur();
  if (busy) {
    // Nothing to apply: just close. Edits wait until KíLín is idle.
    if (!modelEditsPending()) hideModels();
    else byId("model-download-status").textContent = "KíLín is busy. Close Settings again when it finishes to apply your changes.";
    return;
  }
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
      if (chosen.has(item.path) && isOversizedSkippedFile(item)) {
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
  draftDepth = [...currentDepth()];
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
  if (!data.selection_reason_codes
      || typeof data.selection_reason_codes.FILE_TOO_LARGE !== "string") {
    throw new Error("Missing file-selection codes in GUI settings.");
  }
  selectionReasonCodes = data.selection_reason_codes;
  profile = data.profile;
  savedLevels = data.effort_levels || {};
  defaultLevels = data.default_effort_levels || defaultLevels;
  savedRouters = data.router_models || {};
  defaultRouters = data.default_router_models || defaultRouters;
  defaultEmbeddings = data.default_embedding_models || defaultEmbeddings;
  savedDepth = data.search_depth || null;
  defaultDepth = data.default_search_depth || defaultDepth;
  // The appearance is an app-wide preference; a choice from before
  // preferences existed (kept by the page) is carried over once.
  const remembered = localStorage.getItem("kilin-appearance");
  if (data.appearance && data.appearance !== "system") applyAppearance(data.appearance);
  else if (remembered && remembered !== "system") applyAppearance(remembered);
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
    _gui_search_depth: [...currentDepth()],
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
 * Tell whether the workspace tree shows a file.
 *
 * The core keeps a lock file next to each open conversation, so that only
 * one window writes it. The lock is not the user's file, so it is hidden.
 *
 * @param path Workspace-relative file path reported by the backend.
 * @returns False for a conversation lock under logs/, true otherwise.
 */
function shownInWorkspace(path) {
  return !(path.startsWith("logs/") && path.endsWith(".json.lock"));
}

/**
 * Render the saved workspace file tree without selection checkboxes.
 *
 * @param paths Workspace-relative file paths reported by the backend.
 */
function renderWorkspace(paths) {
  workspacePaths = new Set(paths);
  const host = byId("workspace-tree");
  host.replaceChildren();
  // The core lists every file; the tree leaves out conversation locks.
  const root = fileTree(paths.filter(shownInWorkspace));
  // Every folder comes from the core's actual workspace inventory.
  for (const [name, node] of [...root.directories].sort()) {
    const files = allFiles(node);
    // Plain rows: the delete button added below must not act as a label's
    // control, or clicking a file's name would also ask to delete it.
    const item = branch(name, node, files, new Set(), 0, null, {}, false);
    // Sources and outputs start open; the other folders stay folded.
    const key = `${name}|${files[0] || ""}`;
    const startsOpen = name === "sources" || name === "outputs";
    item.open = startsOpen ? !collapsedWorkspaceFolders.has(key) : expandedFolders.has(key);
    item.addEventListener("toggle", () => {
      if (!item.isConnected || !startsOpen) return;
      item.open ? collapsedWorkspaceFolders.delete(key) : collapsedWorkspaceFolders.add(key);
    });
    [item, ...item.querySelectorAll(".tree-folder")].forEach(showFolderCount);
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
        // A file dragged from outputs or templates in this panel.
        const inside = event.dataTransfer.getData(WORKSPACE_DRAG_TYPE);
        if (inside) return copyToSources(inside);
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
    if (EDITABLE_FOLDERS.includes(name)) {
      item.dataset.path = name;
      item.querySelectorAll(".tree-file").forEach((line) => {
        const path = line.dataset.path;
        // Drag a file onto sources to add it there (and index it).
        line.draggable = true;
        line.addEventListener("dragstart", (event) => {
          event.dataTransfer.setData(WORKSPACE_DRAG_TYPE, path);
          event.dataTransfer.effectAllowed = "copy";
        });
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "tree-delete";
        remove.textContent = "×";
        remove.title = `Delete ${path.split("/").pop()}`;
        remove.setAttribute("aria-label", remove.title);
        remove.addEventListener("click", (event) => {
          event.stopPropagation();
          deleteFromFolder(name, path, true);
        });
        line.append(remove);
      });
      const hint = document.createElement("p");
      hint.className = "source-drop-hint";
      hint.textContent = "Right-click to add, replace, or delete; never indexed";
      item.querySelector("summary").append(hint);
      item.addEventListener("contextmenu", (event) => {
        const row = event.target.closest(".tree-file, .tree-folder > summary");
        const isFile = Boolean(row?.classList.contains("tree-file"));
        const path = isFile ? row.dataset.path : row?.parentElement.dataset.path;
        if (!path || !(path === name || path.startsWith(`${name}/`))) return;
        event.preventDefault();
        showFolderMenu(event.clientX, event.clientY, name, path, isFile);
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
  markActiveConversationFile();
}

/**
 * Bold the workspace entry of the conversation file now in use.
 *
 * The engine's trace path is the active conversation's JSON file, an
 * absolute path; workspace rows carry project-relative paths.
 */
function markActiveConversationFile() {
  const active = activeTracePath.replaceAll("\\", "/");
  byId("workspace-tree").querySelectorAll(".tree-file").forEach((row) => {
    const path = row.dataset.path;
    row.classList.toggle(
      "active-conversation",
      Boolean(active && path && active.endsWith(`/${path}`)),
    );
  });
}

// Send the visible drop rectangle to Qt; all other workspace rows reject drops.
let sourceDragActive = false;
const sourceDropSpotlight = document.createElement("div");
sourceDropSpotlight.className = "source-drop-spotlight hidden";
sourceDropSpotlight.setAttribute("aria-hidden", "true");
document.body.append(sourceDropSpotlight);
let lastSourceDropBounds = "";
/**
 * The visible part of an element, clipped by scrolling parents, or null.
 * @param target DOM element whose visible rectangle is needed.
 */
function visibleRect(target) {
  const rect = target?.getBoundingClientRect();
  const visible = !busy && rect?.width && rect?.height && target.getClientRects().length
    && !target.closest(".hidden") && byId("models-panel").classList.contains("hidden");
  if (!visible) return null;
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
  return {x: left, y: top, width: Math.max(0, right - left), height: Math.max(0, bottom - top)};
}

/**
 * Tell Qt where files may be dropped: sources, and outputs or templates.
 */
function syncSourceDropTarget() {
  if (!bridge?.setSourceDropTarget) return;
  const sources = visibleRect(ready ? byId("workspace-source-drop") : byId("drop-zone"));
  const folders = ready ? EDITABLE_FOLDERS.flatMap((name) => {
    const rect = visibleRect(document.querySelector(`#workspace-tree details[data-path="${name}"]`));
    return rect ? [{ name, ...rect }] : [];
  }) : [];
  sendSourceDropBounds(JSON.stringify({ ...(sources || {}), folders }));
}
/**
 * Update the native source-drop overlay and send its bounds to Qt.
 *
 * @param bounds JSON string containing the visible target rectangle.
 */
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
/**
 * Reflect native dragging over the editable source locations.
 *
 * @param active Whether a native file drag is currently in progress.
 */
function showNativeSourceDrag(active) {
  sourceDragActive = Boolean(active && !busy);
  const target = ready ? byId("workspace-source-drop") : byId("drop-zone");
  target?.classList.toggle("drag-over", sourceDragActive);
  for (const name of EDITABLE_FOLDERS) {
    document.querySelector(`#workspace-tree details[data-path="${name}"]`)
      ?.classList.toggle("drag-over", sourceDragActive);
  }
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
  showWorkspaceMenu(x, y, [[
    isFile ? "Delete file from project…" : "Delete folder from project…",
    () => removeSource(path, isFile),
  ]]);
}

/**
 * Show the shared workspace menu with labelled actions at the pointer.
 *
 * @param x Viewport x coordinate of the right-click.
 * @param y Viewport y coordinate of the right-click.
 * @param items Pairs of menu label and action.
 */
function showWorkspaceMenu(x, y, items) {
  if (busy || !ready) return;
  const buttons = items.map(([label, action]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "menuitem");
    button.textContent = label;
    button.addEventListener("click", () => {
      hideSourceMenu();
      action();
    });
    return button;
  });
  sourceMenu.replaceChildren(...buttons);
  sourceMenu.classList.remove("hidden");
  const { width, height } = sourceMenu.getBoundingClientRect();
  sourceMenu.style.left = `${Math.min(x, window.innerWidth - width - 8)}px`;
  sourceMenu.style.top = `${Math.min(y, window.innerHeight - height - 8)}px`;
  buttons[0]?.focus();
}

/**
 * Offer edits in an editable folder: add files, replace a file, or delete.
 *
 * @param x Viewport x coordinate of the right-click.
 * @param y Viewport y coordinate of the right-click.
 * @param folder Editable folder: outputs or templates.
 * @param path Workspace-relative path: the folder or inside it.
 * @param isFile Whether the path is a file rather than a folder.
 */
function showFolderMenu(x, y, folder, path, isFile) {
  const items = [[`Add files to ${folder}…`, () => bridge.chooseFolderFiles(folder, "add")]];
  if (isFile) {
    items.push(["Replace with another file…", () => bridge.chooseFolderFiles(folder, path)]);
    items.push(["Copy to sources (and index it)…", () => copyToSources(path)]);
  }
  if (path !== folder) {
    items.push([isFile ? `Delete file from ${folder}…` : `Delete folder from ${folder}…`,
      () => deleteFromFolder(folder, path, isFile)]);
  }
  showWorkspaceMenu(x, y, items);
}

/**
 * Confirm, then delete a file or folder in outputs or templates.
 * @param folder Editable workspace folder name.
 * @param path Workspace-relative target path.
 * @param isFile Whether the target is a file.
 */
function deleteFromFolder(folder, path, isFile) {
  const name = path.split("/").pop();
  if (!confirm(`Delete "${name}" from ${folder}?\n\n${path}\n\nKíLín deletes it permanently; it does not go to the Trash.`)) return;
  editFolder({ folder, operation: "remove", paths: [path] }, `Deleting from ${folder}…`);
}

/**
 * Copy a workspace file (from outputs or templates) into the sources.
 *
 * @param path Workspace-relative path, such as outputs/summary.md.
 */
function copyToSources(path) {
  const root = byId("workspace-path").value.trim().replace(/[\\/]+$/, "");
  if (!root || !EDITABLE_FOLDERS.some((name) => path.startsWith(`${name}/`))) return;
  addSources([`${root}/${path}`]);
}

/**
 * Send a folder edit; the engine only refreshes the workspace tree.
 * @param request Folder-edit request sent to the backend.
 * @param message Progress message displayed during the edit.
 */
function editFolder(request, message) {
  if (busy || !bridge) return;
  error.classList.add("hidden");
  setBusy(true, message);
  bridge.editFolder(JSON.stringify(request));
}

/**
 * Copy chosen files into an editable folder, or replace one file there.
 *
 * @param folder Editable folder: outputs or templates.
 * @param purpose "add", or the workspace path to replace.
 * @param paths Local files chosen in the native dialog.
 */
function folderFilesChosen(folder, purpose, paths) {
  if (purpose !== "add") {
    const name = purpose.split("/").pop();
    if (!confirm(`Replace "${name}" with the content of ${paths[0].split(/[\\/]/).pop()}?\n\nThe name stays ${name}.`)) return;
    editFolder({ folder, operation: "replace", path: purpose, source: paths[0] }, `Replacing a file in ${folder}…`);
    return;
  }
  const names = paths.map((path) => String(path).split(/[\\/]/).pop());
  const taken = names.filter((name) => workspacePaths.has(`${folder}/${name}`));
  let replace = false;
  if (taken.length) {
    replace = confirm(`These names are already in ${folder}:\n\n${taken.join("\n")}\n\nOK replaces them; Cancel keeps both (the new copies get a number).`);
  }
  editFolder({ folder, operation: "add", paths, replace }, `Adding to ${folder}…`);
}

/**
 * Hide the context menu for workspace source files.
 */
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
  if (!confirm(`Delete ${what} from sources?\n\n${path}\n\nKíLín deletes only the project's copy; your original stays.`)) return;
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

/**
 * Confirm opening a project file with its default application.
 *
 * @param path Workspace-relative path to the selected file.
 */
function openWorkspaceFile(path) {
  if (!path || !bridge || !confirm(`Open ${path} with its default application?`)) return;
  bridge.openWorkspaceFile(path);
}

/**
 * Restore the interface after an engine failure and display the error.
 *
 * @param message Error message for the user.
 * @param tracePath Path to the saved trace, if available.
 */
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
  shownErrorStatus = byId("model-download-status").textContent;
  setBusy(false);
  if (!byId("first-run-setup").classList.contains("hidden")) {
    byId("install-required").disabled = !readinessEnvironmentReady;
  }
  byId("connection").textContent = "Error";
  requestAnimationFrame(() => error.scrollIntoView({ block: "nearest" }));
}

/**
 * Hide an error that no longer applies, such as after a project opens.
 *
 * The Settings status line is cleared only while it still shows that
 * error's text, so later messages there (such as download progress) stay.
 */
function clearError() {
  const status = byId("model-download-status");
  if (shownErrorStatus && status.textContent === shownErrorStatus) {
    status.textContent = "";
  }
  shownErrorStatus = "";
  error.classList.add("hidden");
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
  // Settings can close while KíLín works; only unsaved edits must wait.
  byId("close-models").disabled = false;
  byId("conversation-name").contentEditable = String(!value);
  syncSourceDropTarget();
  if (!value) activity.classList.remove("answer-progress");
  byId("open").disabled = value || !bridge;
  const answering = value && ready && activity.classList.contains("answer-progress");
  byId("question-form").classList.toggle("composer-disabled", value && !answering);
  byId("question-form").setAttribute("aria-busy", String(value));
  for (const id of ["attach-button", "attach-files", "attach-folder", "effort-button", "effort-model-range", "effort-depth-range"]) {
    byId(id).disabled = value || !ready;
  }
  // The next question can be typed while an answer is generated; sending
  // waits until it is done (the submit handler checks busy).
  byId("question").disabled = !ready || (value && !answering);
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
  if (!value) focusComposer();
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
 * Format a byte count using compact binary units.
 *
 * @param value Raw byte count.
 * @returns Human-readable binary-unit size.
 */
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

/**
 * Describe a model-download progress update.
 *
 * @param data Model-download heartbeat data.
 * @returns Elapsed time, progress, and estimated time details.
 */
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

/**
 * Update the setup card's download progress meter.
 *
 * @param percent Current completion percentage, or null.
 */
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
 * @param thinkingMs Time spent on thinking, or null if absent.
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
        chooseSidePanels(true);
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

/**
 * Detect Markdown lines that begin a block-level structure.
 *
 * @param line Candidate Markdown source line.
 * @returns Whether the line starts a supported Markdown block.
 */
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

/**
 * Render escaped inline Markdown and optionally link evidence citations.
 *
 * @param element DOM element receiving the parsed content.
 * @param text Inline Markdown text to render.
 * @param linkCitation Callback constructing a citation link when valid.
 */
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
    chooseSidePanels(true);
    selectTurn(item, turn);
  });
  answer.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      chooseSidePanels(true);
      selectTurn(item, turn);
    }
  });
  renderAnswer(answer, turn, item);
  // A saved answer shows the timing and tokens it showed when it arrived.
  const timeline = renderTimeline(turn.timeline || null, turn);
  item.append(...[question, timeline, answer].filter(Boolean));
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
    let opening = null;
    button.addEventListener("click", () => {
      if (busy || item.conversation_id === activeConversationId) return;
      // Wait briefly so a double-click renames instead of opening.
      clearTimeout(opening);
      opening = setTimeout(() => {
        if (busy) return;
        setBusy(true, "Opening conversation…");
        bridge.selectConversation(item.conversation_id);
      }, RENAME_CLICK_DELAY_MS);
    });
    button.addEventListener("dblclick", (event) => {
      event.preventDefault();
      clearTimeout(opening);
      if (!busy && !provisional) renameInSidebar(row, button, item);
    });
    button.title = `${name}\nDouble-click to rename`;
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
 * Keep the text an answer had generated when the user pressed Stop.
 *
 * Stop restarts the engine, so the partial answer is never saved; it stays
 * on screen, marked as stopped, until the user opens another chat.
 *
 * @param current The pending turn that was stopped.
 */
function keepStoppedTurn(current) {
  stoppedTurn = null;
  if (!current) return;
  if (!current.streaming || !current.body?.textContent.trim()) {
    current.item.remove();
    return;
  }
  const elapsed = current.elapsedMs ?? performance.now() - current.started;
  current.item.classList.remove("pending");
  current.item.classList.add("stopped");
  current.answer.classList.remove("answer-phase", "streaming");
  current.durationLine.textContent = `Stopped after ${formatDuration(elapsed)} · not saved`;
  stoppedTurn = { item: current.item, conversationId: activeConversationId };
}

/**
 * Show the stopped answer again after the restarted engine redraws its chat.
 */
function restoreStoppedTurn() {
  const kept = stoppedTurn;
  stoppedTurn = null;
  if (kept && kept.conversationId === activeConversationId) conversation.append(kept.item);
}

/**
 * Rename a conversation in place in the sidebar: Enter or clicking away
 * saves, Escape cancels.
 *
 * @param row The conversation's sidebar row.
 * @param button The row's conversation button, replaced while editing.
 * @param item The conversation record from the backend.
 */
function renameInSidebar(row, button, item) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "conversation-rename";
  input.value = item.name;
  input.setAttribute("aria-label", `Rename ${item.name}`);
  let done = false;
  const finish = (save) => {
    if (done) return;
    done = true;
    const name = input.value.trim();
    if (save && name && name !== item.name && !busy) {
      bridge.renameConversation(item.conversation_id, name);
    }
    // The backend's conversation list redraws the sidebar after a rename.
    input.replaceWith(button);
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); finish(true); }
    if (event.key === "Escape") { event.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
  button.replaceWith(input);
  input.focus();
  input.select();
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

/**
 * Render the project-type welcome screen for an empty conversation.
 */
function renderConversationEmpty() {
  conversation.replaceChildren();
  const type = projectTypes[activeProjectTypeId] || {};
  const content = {
    prompts: type.welcome_prompts?.length ? type.welcome_prompts : genericEmptyChat.welcome_prompts,
    suggestions: type.suggestions?.length ? type.suggestions : genericEmptyChat.suggestions,
  };
  let promptIndex = Math.floor(Math.random() * content.prompts.length);
  if (content.prompts.length > 1 && promptIndex === lastEmptyPrompt[activeProjectTypeId]) {
    promptIndex = (promptIndex + 1) % content.prompts.length;
  }
  lastEmptyPrompt[activeProjectTypeId] = promptIndex;
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

/**
 * Render removable chips for source files attached to the next prompt.
 */
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
 * Choose the effort level or the search-depth step for the next answers.
 *
 * @param kind "model" or "depth".
 * @param index Zero-based level or step.
 */
function setEffortLevel(kind, index) {
  const level = Math.max(0, Math.min(4, Number(index)));
  if (kind === "model") selectedModelLevel = level;
  else selectedDepthStep = level;
  renderEffortPicker();
}

/**
 * Show the chosen levels on both sliders and on the composer button.
 */
function renderEffortPicker() {
  const levels = currentLevels();
  const depth = currentDepth();
  if (levels.length !== 5 || depth.length !== 5) return;
  // The effort is named by the model level; search depth is a count.
  byId("effort-name").textContent = levels[selectedModelLevel].name;
  for (const [kind, index] of [["model", selectedModelLevel], ["depth", selectedDepthStep]]) {
    byId(`effort-${kind}-detail`).textContent = kind === "model"
      ? levels[index].model.split("/").pop()
      : `${depth[index]} passages`;
    const range = byId(`effort-${kind}-range`);
    range.value = String(index);
    range.style.setProperty("--effort-percent", `${(index / (levels.length - 1)) * 100}%`);
    range.parentElement.querySelectorAll(".effort-points i").forEach((point, pointIndex) => {
      point.classList.toggle("selected", pointIndex === index);
    });
  }
  byId("effort-button").textContent =
    `${levels[selectedModelLevel].name} · ${depth[selectedDepthStep]}`;
  byId("effort-button").title =
    `${levels[selectedModelLevel].name}: ${levels[selectedModelLevel].model.split("/").pop()} · search depth ${depth[selectedDepthStep]} passages`;
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
// A click on another chat waits this long, so a double-click renames it.
const RENAME_CLICK_DELAY_MS = 250;
const ANSWER_STAGE_QUEUE_LIMIT = 8;

/**
 * Clear any queued answer-stage labels and the running stage timer.
 */
function resetAnswerStages() {
  clearTimeout(answerStageTimer);
  answerStageTimer = null;
  answerStageKey = null;
  answerStageQueue = [];
}

/**
 * Show the next queued stage and schedule the following stage.
 */
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

/**
 * Queue a distinct stage label while a question is being answered.
 *
 * @param key Stable identifier used to deduplicate stage labels.
 * @param detail Human-readable status shown by the activity display.
 */
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
 * Record how long a completed answer took and how long each step took.
 *
 * The record is saved with the project, so a resumed conversation shows the
 * same "Took Ns" line and step list as when the answer arrived.
 *
 * @param current The pending turn being completed.
 * @param turn Saved turn with its routing parameters.
 * @param totalMs Time from Ask to the complete answer.
 * @returns The total time and each step's label and duration.
 */
function timelineRecord(current, turn, totalMs) {
  const end = current.started + totalMs;
  const steps = (current.timeline || []).filter((step) => step.label !== "Starting");
  const query = turn.parameters?.retrieval_query;
  return {
    elapsed_ms: Math.round(totalMs),
    steps: steps.map((step, index) => ({
      label: query && /semantic search/i.test(step.label)
        ? `${step.label} for “${query}”`
        : step.label,
      ms: Math.round(Math.max(0, (steps[index + 1]?.at ?? end) - step.at)),
    })),
  };
}

/**
 * Build the collapsed "Took Ns · Xk tokens" line; expanding it lists each
 * step and how long it took.
 *
 * @param record Saved timing from timelineRecord, or null for an answer
 *   saved before timings were kept (then only its tokens are shown).
 * @param turn Saved turn with its token usage.
 * @returns The line, or null when there is neither a timing nor tokens.
 */
function renderTimeline(record, turn) {
  const usage = turn.parameters?.usage;
  const tokens = usageSum(usage);
  if (!record && !tokens) return null;
  const details = document.createElement("details");
  details.className = "answer-duration";
  const summary = document.createElement("summary");
  summary.textContent = [
    record ? `Took ${formatDuration(record.elapsed_ms)}` : "",
    tokens ? `${formatTokens(tokens)} tokens` : "",
  ].filter(Boolean).join(" · ");
  if (tokens) {
    summary.title = `${usage.input_tokens.toLocaleString()} tokens read, ${usage.output_tokens.toLocaleString()} written`;
  }
  const list = document.createElement("ol");
  list.className = "answer-timeline";
  for (const step of record?.steps || []) {
    const row = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = step.label;
    const time = document.createElement("span");
    time.className = "timeline-time";
    time.textContent = step.ms < 100 ? "<0.1s" : `${(step.ms / 1000).toFixed(1)}s`;
    row.append(label, time);
    list.append(row);
  }
  if (tokens) {
    const row = document.createElement("li");
    row.className = "timeline-usage";
    const label = document.createElement("span");
    label.textContent = `Tokens: ${usage.input_tokens.toLocaleString()} read, ${usage.output_tokens.toLocaleString()} written`;
    row.append(label);
    list.append(row);
  }
  details.append(summary, list);
  return details;
}

/**
 * Complete an answer immediately and discard pending stage animations.
 *
 * @param turn Completed answer and saved conversation metadata.
 */
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
 * Format a token count compactly: 850, 3.5k, 42k, 1.2M.
 *
 * @param count Number of tokens.
 * @returns Short human-readable count.
 */
function formatTokens(count) {
  const n = Math.max(0, Number(count) || 0);
  const short = (value, unit) => `${value >= 10 ? Math.round(value) : value.toFixed(1).replace(/\.0$/, "")}${unit}`;
  if (n < 1000) return String(Math.round(n));
  if (n < 1e6) return short(n / 1e3, "k");
  if (n < 1e9) return short(n / 1e6, "M");
  return short(n / 1e9, "B");
}

/**
 * Total tokens of a usage record (read plus written).
 * @param usage Input and output token-count record.
 */
function usageSum(usage) {
  return (Number(usage?.input_tokens) || 0) + (Number(usage?.output_tokens) || 0);
}

/**
 * Show the composer's token pill and its menu, including a running answer.
 */
function renderUsage() {
  const live = pendingTurn?.usage || null;
  const plus = (base) => ({
    input_tokens: (Number(base?.input_tokens) || 0) + (Number(live?.input_tokens) || 0),
    output_tokens: (Number(base?.output_tokens) || 0) + (Number(live?.output_tokens) || 0),
    answers: Number(base?.answers) || 0,
  });
  const show = (key, usage) => {
    const total = usageSum(usage);
    byId(`usage-${key}-total`).textContent = `${formatTokens(total)} tokens`;
    byId(`usage-${key}-answers`).textContent = usage.answers === 1 ? "1 answer" : `${usage.answers} answers`;
    // The bar is full: read (green-grey) then written (orange).
    const read = total ? (100 * usage.input_tokens) / total : 0;
    byId(`usage-${key}-read`).style.width = `${read.toFixed(1)}%`;
    byId(`usage-${key}-written`).style.width = total ? `${(100 - read).toFixed(1)}%` : "0";
    byId(`usage-${key}-read-text`).textContent = `${usage.input_tokens.toLocaleString()} read`;
    byId(`usage-${key}-written-text`).textContent = `${usage.output_tokens.toLocaleString()} written`;
    return total;
  };
  const total = show("project", plus(usageTotals?.project));
  show("chat", plus(usageTotals?.chat));
  byId("usage-button").title = `${formatTokens(total)} tokens used in this project`;
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
  const tokens = usageSum(pendingTurn.usage);
  pendingTurn.timer.textContent = formatDuration(
    pendingTurn.elapsedMs ?? performance.now() - pendingTurn.started)
    + (tokens ? ` · ${formatTokens(tokens)} tokens` : "");
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
 * @param plan Plan fields from kilin.tools.planning.ToolPlan.
 */
function showToolPlan(plan) {
  if (!pendingTurn) return;
  const card = document.createElement("div");
  card.className = "tool-plan";
  const summary = document.createElement("p");
  summary.className = "tool-plan-summary";
  summary.textContent = plan.summary;
  const facts = document.createElement("p");
  facts.className = "tool-plan-facts";
  facts.textContent = [plan.scope, ...(plan.details || [])].join(" · ");
  const duration = (seconds) => (seconds < 1 ? "instant" : `about ${formatDuration(seconds * 1000)}`);
  const calls = plan.batches_to_run === 1 ? "1 model call" : `${plan.batches_to_run} model calls`;
  const actions = document.createElement("div");
  actions.className = "tool-plan-actions";
  const choices = [
    [PLAN_CHOICE_CONTINUE, "primary", "Read everything", `${calls} · ${duration(plan.estimated_seconds)}`],
    [PLAN_CHOICE_QUICK, "secondary", plan.quick_label || "Quick sample",
      `${plan.quick_note || `A sample of ${plan.quick_items}`} · ${duration(plan.quick_seconds)}`],
  ];
  const choose = (choice) => {
    card.remove();
    queueAnswerStage(`choice:${choice}`, choice === PLAN_CHOICE_CANCEL ? "Cancelling" : "Starting");
    bridge.choosePlan(choice);
  };
  for (const [choice, style, label, note] of choices) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `tool-plan-option ${style}`;
    const title = document.createElement("span");
    title.className = "tool-plan-option-title";
    title.textContent = label;
    const detail = document.createElement("span");
    detail.className = "tool-plan-option-note";
    detail.textContent = note;
    button.append(title, detail);
    button.addEventListener("click", () => choose(choice));
    actions.append(button);
  }
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "tool-plan-cancel";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => choose(PLAN_CHOICE_CANCEL));
  actions.append(cancel);
  card.append(summary, facts, actions);
  pendingTurn.answer.append(card);
  conversationScroll.scrollTop = conversationScroll.scrollHeight;
}

/**
 * Show a tool's question with one button per option, plus Cancel.
 *
 * @param choice Question and option labels from the tool.
 */
function showToolChoice(choice) {
  if (!pendingTurn) return;
  const card = document.createElement("div");
  card.className = "tool-plan tool-choice";
  const question = document.createElement("p");
  question.className = "tool-plan-summary";
  question.textContent = choice.question;
  const actions = document.createElement("div");
  actions.className = "tool-choice-options";
  const answer = (value, label) => {
    card.remove();
    queueAnswerStage(`choice:${value}`, value === "cancel" ? "Cancelling" : `Using ${label}`);
    bridge.choosePlan(value);
  };
  (choice.options || []).forEach((label, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = label;
    button.addEventListener("click", () => answer(`option:${index}`, label.replace(/ \(.*\)$/, "")));
    actions.append(button);
  });
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "secondary tool-choice-cancel";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", () => answer("cancel", ""));
  actions.append(cancel);
  card.append(question, actions);
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
    chooseSidePanels(true);
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
  const record = timelineRecord(
    current, turn, current.elapsedMs ?? performance.now() - current.started);
  const timeline = renderTimeline(record, turn);
  if (current.durationLine) current.durationLine.replaceWith(timeline);
  else current.item.insertBefore(timeline, current.answer);
  current.durationLine = timeline;
  // Keep the timing with the project so a resumed chat shows it too.
  if (bridge && turn.timestamp) bridge.saveAnswerTimeline(turn.timestamp, JSON.stringify(record));
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
 * Put the cursor back in the message box when KíLín is free again, unless
 * the user is typing somewhere else (a rename, a field) or Settings is open.
 */
function focusComposer() {
  const question = byId("question");
  const active = document.activeElement;
  const elsewhere = active && active !== question && active !== document.body
    && (active.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName));
  const settingsOpen = !byId("models-panel").classList.contains("hidden");
  if (!ready || question.disabled || elsewhere || settingsOpen) return;
  question.focus();
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
 * @param intentId Configured project-type intent ID.
 */
function intentName(intentId) {
  const names = projectTypes[activeProjectTypeId]?.intents || {};
  return names[intentId] || String(intentId || "").replaceAll("_", " ");
}

/**
 * Describe which earlier turns the answer sees, such as "Recent 2 + turn 3".
 * @param data Turn parameters containing history metadata.
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
    markTimelineStep("Generating the answer");
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
    markActiveConversationFile();
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
      : `${String(data.backend).toUpperCase()} runtime is missing; restart through uv run kilin --gui`;
    runtime.classList.toggle("complete", data.environment_ready);
    steps.append(runtime);
    const replacements = data.replacements || {};
    for (const modelId of requiredModels) {
      const item = document.createElement("li");
      const present = cachedModels.has(modelId);
      item.textContent = present
        ? `Ready: ${modelId}`
        : replacements[modelId]
          ? `Missing: ${modelId} (installed alternative: ${replacements[modelId]})`
          : `Missing: ${modelId}`;
      item.classList.toggle("complete", present);
      steps.append(item);
    }
    // Always available; it opens by itself when something is missing.
    card.classList.remove("hidden");
    if (!data.ready) card.open = true;
    const missing = data.missing_models.length + (data.environment_ready ? 0 : 1);
    byId("readiness-status").textContent = data.ready ? "Ready" : `${missing} missing`;
    byId("readiness-status").classList.toggle("ready", Boolean(data.ready));
    byId("readiness-summary").textContent = data.ready
      ? "KíLín is ready."
      : data.missing_models.length === 1
        ? "1 required model needs to be downloaded."
        : `${data.missing_models.length} required models need to be downloaded.`;
    byId("install-required").disabled = !data.environment_ready || !data.missing_models.length;
    byId("install-required").classList.toggle("hidden", !data.missing_models.length);
    // Compatible models already on this computer can be used instead.
    byId("use-installed").classList.toggle("hidden", !data.can_use_installed);
    byId("use-installed").disabled = !data.environment_ready;
    refreshModelStatus();
    if (!data.ready) {
      byId("models-panel").classList.remove("hidden");
      byId("close-models").focus();
    }
  } else if (data.event === "installed_models_used") {
    savedLevels = data.effort_levels || savedLevels;
    savedRouters = data.router_models || savedRouters;
    savedDepth = data.search_depth || savedDepth;
    if (profile && data.embedding_models) {
      for (const [backend, model] of Object.entries(data.embedding_models)) profile[`embedding_${backend}`] = model;
    }
    showModelFields();
    setBusy(false);
    const used = Object.entries(data.replacements || {}).map(([from, to]) => `${to} for ${from}`);
    byId("readiness-time").textContent = used.length ? `Using ${used.join("; ")}.` : "";
  } else if (data.event === "installation_started") {
    setupCompletionVisible = false;
    setBusy(true, `Preparing ${data.total} required models…`);
    byId("first-run-setup").classList.remove("hidden");
    byId("first-run-setup").open = true;
    byId("readiness-status").textContent = "Downloading";
    byId("readiness-status").classList.remove("ready");
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
    byId("readiness-status").textContent = "Ready";
    byId("readiness-status").classList.add("ready");
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
        ...(root.skipped || []).filter((item) => isOversizedSkippedFile(item)).map((item) => item.path),
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
    } else if (data.phase === ANSWER_PHASE_PROCESSING && data.choice) {
      // A tool asks which option the user means, such as which paper.
      queueAnswerStage("choice", "Waiting for your choice");
      showToolChoice(data.choice);
      byId("activity-text").textContent = "Waiting for your choice";
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
    savedDepth = data.search_depth || [...draftDepth];
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
  } else if (data.event === "usage_progress") {
    if (pendingTurn) {
      pendingTurn.usage = { input_tokens: data.input_tokens, output_tokens: data.output_tokens };
      paintPendingTurn();
      renderUsage();
    }
  } else if (data.event === "progress") {
    showProgress(data);
  } else if (data.event === "opened") {
    // An error from an earlier attempt no longer applies to this project.
    clearError();
    usageTotals = data.usage || usageTotals;
    renderUsage();
    ready = true;
    if (data.profile) {
      profile = data.profile;
      byId("query-format").value = profile.query_format;
      savedLevels = data.effort_levels || savedLevels;
      savedRouters = data.router_models || savedRouters;
      savedDepth = data.search_depth || savedDepth;
      showModelFields();
    }
    activeProjectTypeId = data.type_id;
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
    restoreStoppedTurn();
    if (!data.turns.length) renderConversationEmpty();
    byId("workspace-path").value = data.project;
    byId("workspace-location").textContent = data.project;
    renderWorkspace(data.workspace_files);
    // Open with references only when there are answers, unless the user chose.
    showWorkspace(sidePanelChoice ?? Boolean(data.turns.length));
    byId("temperature").value = data.temperature;
    byId("max-new-tokens").value = data.max_new_tokens ?? "";
    byId("thinking").checked = data.thinking;
    byId("project-name").textContent = data.project.split(/[\\/]/).pop();
    updateProjectSummary(data.turns.length);
    updateFloatingHeader();
    setBusy(false);
  } else if (data.event === "conversation_opened") {
    usageTotals = data.usage || usageTotals;
    renderUsage();
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
    restoreStoppedTurn();
    if (!data.turns.length) renderConversationEmpty();
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
  } else if (data.event === "folder_changed") {
    renderWorkspace(data.workspace_files);
    setBusy(false);
  } else if (data.event === "answered") {
    if (data.usage) usageTotals = data.usage;
    completeAfterAnswerStages(data.turn);
    renderUsage();
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
      keepStoppedTurn(pendingTurn);
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
    bridge.requestSettings(byId("type-id").value, byId("workspace-path").value.trim());
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
    bridge.requestSettings(byId("type-id").value, byId("workspace-path").value.trim());
    bridge.requestReadiness();
    if (reopenProject) {
      setBusy(true, "Reopening the project after backend recovery…");
      bridge.resumeProjectConversation(
        byId("workspace-path").value.trim(),
        activeConversationId,
      );
    }
  } else if (data.event === "workspace_conflict") {
    // Creation never reuses a project; opening one is a separate decision.
    setBusy(false);
    error.textContent = data.message;
    error.classList.remove("hidden");
    const resume = data.existing_project && confirm(
      `A project already exists at ${data.workspace}.\n\nOpen that project instead? Selected sources and new-project settings will not be applied.\n\nCancel to choose another destination.`,
    );
    if (resume) {
      setBusy(true, "Opening the existing project...");
      bridge.resumeProject(data.workspace);
    } else {
      openModels();
      byId("workspace-path").focus();
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
byId("use-installed").addEventListener("click", () => {
  if (busy || !bridge) return;
  setBusy(true, "Switching to installed models…");
  bridge.useInstalledModels();
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
  if (!included.length && !byId("workspace-path").value.trim()) {
    openModels();
    byId("workspace-path").focus();
    return showError("Choose a workspace folder to create a project without sources.");
  }
  setBusy(true, "Opening project…");
  requestAnimationFrame(() => activity.scrollIntoView({ block: "nearest" }));
  bridge.openProject(
    JSON.stringify(included), byId("type-id").value, byId("workspace-path").value.trim(),
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
  const effort = JSON.stringify({ model: selectedModelLevel, search_depth: selectedDepthStep });
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
  setBusy(true, "Starting a new conversation…");
  bridge.newConversation();
});
byId("branch-chat").addEventListener("click", () => {
  if (busy) return;
  setBusy(true, "Branching this conversation…");
  bridge.branchConversation();
});
/**
 * Send a valid edited conversation title to the backend.
 */
function commitConversationName() {
  const name = byId("conversation-name").textContent.trim();
  if (!busy && activeConversationId && name) bridge.renameConversation(activeConversationId, name);
}
/**
 * Rename the current project when the edited title differs.
 */
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
  bridge.requestSettings(byId("type-id").value, "");
  setBusy(false);
  refreshSavedProject();
});
byId("toggle-workspace").addEventListener("click", () => {
  chooseSidePanels(!shell.classList.contains("sidebar-open"));
});
byId("toggle-conversations").addEventListener("click", () => {
  showConversations(shell.classList.contains("conversations-hidden"));
});
byId("usage-button").addEventListener("click", (event) => {
  event.stopPropagation();
  const menu = byId("usage-menu");
  renderUsage();
  byId("effort-menu").classList.add("hidden");
  byId("effort-button").setAttribute("aria-expanded", "false");
  menu.classList.toggle("hidden");
  byId("usage-button").setAttribute("aria-expanded", String(!menu.classList.contains("hidden")));
});
byId("usage-menu").addEventListener("click", (event) => event.stopPropagation());
byId("effort-button").addEventListener("click", (event) => {
  event.stopPropagation();
  const menu = byId("effort-menu");
  // One composer menu at a time.
  byId("usage-menu").classList.add("hidden");
  byId("usage-button").setAttribute("aria-expanded", "false");
  menu.classList.toggle("hidden");
  byId("effort-button").setAttribute("aria-expanded", String(!menu.classList.contains("hidden")));
});
byId("effort-model-range").addEventListener("input", (event) => {
  setEffortLevel("model", event.target.value);
});
byId("effort-depth-range").addEventListener("input", (event) => {
  setEffortLevel("depth", event.target.value);
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
byId("type-id").addEventListener("change", () => {
  hideResume();
  bridge?.requestSettings(byId("type-id").value, byId("workspace-path").value.trim());
  refreshSavedProject();
});
byId("workspace-path").addEventListener("input", queueSavedProjectRefresh);
byId("workspace-path").addEventListener("change", () => {
  bridge?.requestSettings(byId("type-id").value, byId("workspace-path").value.trim());
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
  if (!event.target.closest(".usage-picker")) byId("usage-menu").classList.add("hidden");
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
  bridge = channel.objects.kilin;
  bridge.setAppearance(document.documentElement.dataset.appearance || "system");
  bridge.event.connect(onEvent);
  bridge.pathsSelected.connect(addSources);
  bridge.folderFilesSelected.connect(folderFilesChosen);
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
    bridge.requestSettings(byId("type-id").value, path);
    refreshSavedProject();
  });
  bridge.requestSettings(byId("type-id").value, byId("workspace-path").value.trim());
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
applyAppearance(localStorage.getItem("kilin-appearance") || "system");

// Draggable borders beside the conversation list and the side panels.
const PANEL_LIMITS = { left: [180, 420], right: [260, 640] };
const MIDDLE_MIN_WIDTH = 420;
// Dragging a panel below this share of its minimum width hides it.
const COLLAPSE_SHARE = 0.55;
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
  let collapse = false;
  let startWidth = "";
  const panelClass = `collapsing-${side}`;
  const reset = () => {
    setPanelWidth(side, null);
    try { localStorage.removeItem(`kilin-${side}-width`); } catch {}
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
    collapse = false;
    startWidth = shell.style.getPropertyValue(`--${side}-width`);
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
    // Far below the minimum: preview hiding the panel instead of resizing.
    collapse = requested < PANEL_LIMITS[side][0] * COLLAPSE_SHARE;
    shell.classList.toggle(panelClass, collapse);
    if (collapse) return;
    // Keep the conversation column usable.
    setPanelWidth(side, Math.min(requested, window.innerWidth - other - MIDDLE_MIN_WIDTH));
    positionPanelResizers();
  });
  const finish = (event) => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    handle.releasePointerCapture(event.pointerId);
    handle.classList.remove("dragging");
    document.body.classList.remove("resizing-panels");
    shell.classList.remove(panelClass);
    if (collapse) {
      // Hide the panel as its toggle does; it reopens at its last width.
      collapse = false;
      if (startWidth) shell.style.setProperty(`--${side}-width`, startWidth);
      else shell.style.removeProperty(`--${side}-width`);
      if (side === "left") showConversations(false);
      else chooseSidePanels(false);
      positionPanelResizers();
      syncSourceDropTarget();
      return;
    }
    try {
      localStorage.setItem(`kilin-${side}-width`, shell.style.getPropertyValue(`--${side}-width`));
    } catch {}
    syncSourceDropTarget();
  };
  handle.addEventListener("pointerup", finish);
  handle.addEventListener("pointercancel", finish);
  document.body.append(handle);
  panelResizers[side] = handle;
  try {
    const saved = parseFloat(localStorage.getItem(`kilin-${side}-width`));
    if (Number.isFinite(saved)) setPanelWidth(side, saved);
  } catch {}
}
// Panels appear, hide, and change size on many events; follow them all.
const panelObserver = new ResizeObserver(positionPanelResizers);
panelObserver.observe(shell);
panelObserver.observe(byId("conversations-panel"));
panelObserver.observe(document.querySelector(".side-panels"));
window.addEventListener("resize", positionPanelResizers);

// Draggable border between the Workspace and References panels.
const WORKSPACE_SHARE_LIMITS = [15, 85];
const sideDivider = byId("side-divider");

/**
 * Apply a remembered or dragged Workspace height, within its limits.
 *
 * @param share Requested percentage of the side column for Workspace;
 *   null restores the default even split.
 */
function setWorkspaceShare(share) {
  const panels = document.querySelector(".side-panels");
  if (share === null) {
    panels.style.removeProperty("--workspace-share");
    return;
  }
  const [min, max] = WORKSPACE_SHARE_LIMITS;
  panels.style.setProperty("--workspace-share", `${Math.round(Math.min(max, Math.max(min, share)))}%`);
}

let sideDividerPress = 0;
sideDivider.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  // Detect double-clicks here: pointer capture can suppress dblclick.
  if (event.timeStamp - sideDividerPress < 400) {
    sideDividerPress = 0;
    setWorkspaceShare(null);
    try { localStorage.removeItem("kilin-workspace-share"); } catch {}
    return;
  }
  sideDividerPress = event.timeStamp;
  sideDivider.setPointerCapture(event.pointerId);
  sideDivider.classList.add("dragging");
  document.body.classList.add("resizing-panels", "resizing-rows");
});
sideDivider.addEventListener("pointermove", (event) => {
  if (!sideDivider.hasPointerCapture(event.pointerId)) return;
  sideDividerPress = 0;  // A drag is not the first half of a double-click.
  const box = document.querySelector(".side-panels").getBoundingClientRect();
  setWorkspaceShare(((event.clientY - box.top) / box.height) * 100);
});
const finishSideDivider = (event) => {
  if (!sideDivider.hasPointerCapture(event.pointerId)) return;
  sideDivider.releasePointerCapture(event.pointerId);
  sideDivider.classList.remove("dragging");
  document.body.classList.remove("resizing-panels", "resizing-rows");
  try {
    localStorage.setItem("kilin-workspace-share", document.querySelector(".side-panels").style.getPropertyValue("--workspace-share"));
  } catch {}
  syncSourceDropTarget();
};
sideDivider.addEventListener("pointerup", finishSideDivider);
sideDivider.addEventListener("pointercancel", finishSideDivider);
try {
  const saved = parseFloat(localStorage.getItem("kilin-workspace-share"));
  if (Number.isFinite(saved)) setWorkspaceShare(saved);
} catch {}
