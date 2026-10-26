# Code example: grading worksheet

Indexed revision: 1a3580723 (HEAD)

## markdown_sections (location, evaluation)

**Question.** How does KíLín split a Markdown file into searchable parts, and what happens to the opening lines that come before any heading?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín splits a Markdown file into searchable parts by identifying ATX-style headings (lines starting with `#`) and separating the text before the first heading as a distinct "preamble" section. The code in `sources_md.py` uses `_markdown_headings` to locate these headings [1], then `_markdown_units` divides the file into the preamble (lines 1 to the first heading) and sections between headings [5]. The preamble is preserved as a standalone `SourceUnit` with the kind `"markdown_preamble"` [5]. Lines after the last heading are also captured as a section. If no headings exist, the entire file is treated as preamble. This approach ensures searchable chunks are derived from structured sections while retaining unstructured pre-heading text.

**Cited passages:**

- [1] kilin/sources/extensions/sources_md.py `_markdown_headings` (retrieved by: lexical, semantic)
- [5] kilin/sources/extensions/sources_md.py `_markdown_units` (retrieved by: semantic)

**Gold functions:** `_markdown_units`, `_markdown_headings`

- in evidence: ['_markdown_units', '_markdown_headings']
- cited: ['_markdown_units', '_markdown_headings']

**Facts (tick those the answer states correctly):**

- [ ] Lines starting with one to six # characters (ATX headings) start a new flat section that runs until the next heading.
- [ ] Headings inside backtick or tilde code fences are ignored.
- [ ] The text before the first heading becomes a separate preamble unit.
- [ ] Each section keeps its heading title and original line range; nested structure and underlined (setext) headings are not recognized.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín splits a Markdown file into searchable parts by identifying headings (e.g., `#`, `##`, etc.) and treating each heading as the start of a new section. The content between headings is grouped into sections, and each section is indexed for search. 

**Opening lines before the first heading** are included in the first section. If there are no headings in the file, the entire content is treated as a single section. This ensures that even introductory text or pre-heading content is part of the searchable index. 

This behavior allows KíLín to handle both structured and unstructured Markdown files, ensuring all content is indexed unless explicitly excluded by the user.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## matlab_calls (location, evaluation)

**Question.** How does KíLín work out which functions a MATLAB file calls, and how does this differ from the way it handles Python?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín identifies MATLAB function calls using regex patterns to parse source code without execution. For MATLAB, it matches function definitions with `_FUNCTION` regex (e.g., `function ...`) and calls with `_CALL` regex (e.g., `function_name(...`), while excluding control statements like `if` or `for` [4][9]. It resolves function names using context from the code graph and known entities [8]. This approach relies on structural text patterns, not semantic analysis. 

For Python, the evidence does not explicitly describe how KíLín handles function calls. The provided code focuses on MATLAB-specific regex and graph-building logic, with no direct comparison to Python’s handling. However, the general framework (e.g., `CodeGraph`, `extract_code_relations`) suggests Python might use similar regex-based parsing, though the exact methods are not detailed in the supplied evidence. The MATLAB approach is explicitly tied to structural text matching, while Python’s method (if similar) would likely follow the same pattern but adapted to Python syntax. 

The key difference lies in MATLAB’s explicit regex rules for function and call detection, whereas Python’s handling is not fully described here. [4][8][9]

**Cited passages:**

- [4] kilin/retrievals/code_graph/extensions/code_graph_m.py `code_graph_m` (retrieved by: semantic)
- [8] kilin/retrievals/code_graph/extensions/code_graph_m.py `extract_code_relations` (retrieved by: lexical)
- [9] kilin/retrievals/code_graph/extensions/code_graph_m.py `extract_code_relations` (retrieved by: context_expansion)

**Gold functions:** `extract_code_relations`

- in evidence: ['extract_code_relations']
- cited: ['extract_code_relations']

**Facts (tick those the answer states correctly):**

- [ ] MATLAB relations come from regular expressions over the text, not from a parser; the code is never run.
- [ ] A classdef makes the class the parent of the file's functions and records inheritance from its base class.
- [ ] Each function's body runs until the next function declaration; every name followed by an opening parenthesis in it is recorded as a call, except control words such as if, for, while, and switch.
- [ ] Names are resolved against known entities with the shared name resolution, as for Python.
- [ ] Python files are parsed into syntax trees instead; MATLAB matches can miss or misread comments, strings, and nested constructs.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín determines which functions a MATLAB file calls through **static analysis** of the code. It parses the MATLAB script or function file, examining the syntax to identify all explicit function calls (e.g., `plot(x, y)`, `sin(x)`, or calls to other MATLAB functions). Since MATLAB has a structured syntax and is compiled, this approach is feasible without executing the code.

For Python, KíLín uses a **combination of static analysis and dynamic execution**. Python's dynamic nature (e.g., dynamic imports, conditional execution, or runtime function resolution) means static analysis alone might miss some dependencies. KíLín may run the Python code in a sandboxed environment to observe actual function calls, ensuring it captures all dependencies, even those that are conditionally executed or imported at runtime.

**Key difference**: MATLAB's static structure allows KíLín to rely solely on syntax parsing, while Python's flexibility requires dynamic execution to fully resolve function calls.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## long_document (location, evaluation)

**Question.** When a question asks KíLín to read a whole file that does not fit in the evidence budget, how are the parts the model sees chosen?

**Intent:** code_explanation

**Answer with retrieval:**

When a whole file exceeds the evidence budget, KíLín selects parts based on relevance and token estimation. The `_fit` function in `document_reading.py` [1] prioritizes the most relevant chunks from ranked search results, adding them first in order of relevance. If the total token count still exceeds the budget, less relevant chunks are appended last. The process uses estimated token counts to ensure each part fits within the budget [6]. For large files, the `plan_parts` function splits the document into segments, reading them sequentially [7]. This ensures the model sees the most relevant passages first, even if the entire file isn't included. The selection logic avoids loading entire files into memory, relying on pre-indexed chunks and their estimated token sizes [10].

**Cited passages:**

- [1] kilin/retrievals/document_reading.py `_fit` (retrieved by: lexical, semantic)
- [6] kilin/tools/batch_review/tool.py `_fit` (retrieved by: lexical)
- [7] kilin/tools/batch_review/tool.py `run` (retrieved by: lexical)
- [10] kilin/retrievals/document_reading.py `_fit` (retrieved by: context_expansion)

**Gold functions:** `read_document`, `_fit`

- in evidence: ['read_document', '_fit']
- cited: ['_fit']

**Facts (tick those the answer states correctly):**

- [ ] The file is the one the caller chose, a named function's or file's, or otherwise the file dominating the top search results.
- [ ] If the whole file fits the budget (estimated tokens), it is supplied whole.
- [ ] Otherwise its chunks are taken in the order of the fused search ranking, unranked chunks last; a chunk that does not fit is skipped while smaller later ones may still fit.
- [ ] The selected chunks are given to the model in source order, all from one file.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package is designed to handle large files by processing them in a streaming or chunked manner, but the exact method for selecting parts of the file when it exceeds the evidence budget depends on the specific implementation details of the package. However, based on general principles of such tools:

1. **Chunking**: KíLín might divide the file into fixed-size chunks and process them sequentially, ensuring only a manageable portion of the file is in memory at any time. This approach avoids loading the entire file into memory.

2. **Sampling**: If the file is too large, the package might use sampling techniques (e.g., random sampling or stratified sampling) to select representative parts of the file for analysis, especially if the query does not require full coverage.

3. **Query-Driven Selection**: If the query specifies a particular region or pattern, KíLín might focus on relevant sections of the file, prioritizing areas that match the query's criteria.

4. **Configuration Parameters**: The behavior could be controlled by parameters such as `chunk_size`, `sample_rate`, or `max_tokens`, which users can adjust to balance between computational efficiency and the completeness of the analysis.

For precise details, you would need to consult the KíLín documentation or source code, as the exact selection strategy may vary based on the package's design. If you have access to the documentation, it would provide the most accurate answer.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## exact_mentions (location, evaluation)

**Question.** How does KíLín find every place where an exact name appears in the project, and how does it avoid matching the name inside a longer word?

**Intent:** mention_listing

**Answer with retrieval:**

Found **1 mention** of “exact name” or “longer word” in **1 of 316 files** (case-sensitive, whole words).

- “exact name”: 0
- “longer word”: 1

**Mentions per file**

- retrieval_code.py: 1

**Where**

1. retrieval_code.py, line 399, in _contains_whole_words: …heck would let a short entity name match inside an unrelated **longer word** that merely starts the same way -- for example the module en… [1]

Scanned all indexed text in 316 files.

**Cited passages:**


**Gold functions:** `scan_mentions`, `term_pattern`, `_units`

- in evidence: None
- cited: None

**Facts (tick those the answer states correctly):**

- [ ] The scan runs over the indexed text, reassembling each unit's lines once from its chunks so overlapping chunks are not counted twice.
- [ ] Each term becomes a regular expression whose parts may be joined by spaces, underscores, or hyphens.
- [ ] With whole words, the term may not continue into adjacent letters or digits (MRI does not match MRIs).
- [ ] Case sensitivity is an option (the code project's mention listing matches case); each occurrence keeps its file, line, and a few context lines.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín uses a combination of **regular expressions with word boundaries** and **AST (Abstract Syntax Tree) parsing** to identify exact names in a project while avoiding partial matches within longer words. Here's how it works:

1. **Word Boundaries with Regex**:  
   KíLín employs regular expressions that include word boundaries (`\b`) to ensure the exact name is matched as a standalone identifier. For example, searching for `name` would match `name` but not `username` or `names` because the regex ensures the name is surrounded by non-word characters (like spaces, punctuation, or the start/end of a line).

2. **AST Parsing for Contextual Accuracy**:  
   By parsing the code into an AST, KíLín can analyze identifiers in their syntactic context. This allows it to distinguish between standalone names (e.g., variable names, function names) and parts of longer words. The AST ensures that matches occur only where the name is explicitly used as an identifier, not as a substring within other constructs (e.g., `username` vs. `user`).

3. **Language-Specific Rules**:  
   KíLín adheres to the syntax and semantics of the target language (e.g., Python) to refine matches. For instance, it avoids matching names within strings, comments, or other non-identifier contexts, further reducing false positives.

This approach ensures precise detection of exact names while avoiding unintended matches within longer words or phrases.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## answer_streaming (location, evaluation)

**Question.** How does the desktop app show an answer while the model is still writing it?

**Intent:** code_explanation

**Answer with retrieval:**

The desktop app streams partial answers incrementally using the `_AnswerStream` class [6]. This class buffers generated text and emits it as `"answer_delta"` events to the frontend at regular intervals (default 0.08 seconds). The `add` method accumulates text fragments, and `flush` sends them as updates when the interval threshold is met. This allows the UI to display the answer progressively while the model is still generating it, rather than waiting for the full response. The `send` callback in `_AnswerStream` connects to the frontend, ensuring real-time updates without blocking the user interface.

**Cited passages:**

- [6] kilin/gui/engine.py `_AnswerStream` (retrieved by: semantic)

**Gold functions:** `ChatEngine.ask`, `_AnswerStream.add`, `_AnswerStream.flush`

- in evidence: ['ChatEngine.ask']
- cited: []

**Facts (tick those the answer states correctly):**

- [ ] ChatEngine.ask passes a token callback (an _AnswerStream's add) to the session, which the chat model calls with each decoded piece of text.
- [ ] The stream buffers pieces and sends them to the window as answer_delta events at most every 0.08 seconds.
- [ ] After the answer, flush sends whatever is left.
- [ ] With MLX the text arrives piece by piece; PyTorch generation is not streamed, so it arrives at once.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The KíLín Python package's desktop app displays answers incrementally while the model is generating them by using a combination of background processing and real-time updates. Here's how it works:

1. **Background Thread/Process**: The model's inference (e.g., text generation) runs in a separate thread or process to prevent the UI from freezing. This ensures the desktop app remains responsive during generation.

2. **Streaming Output**: The model generates text in a step-by-step manner (token by token). KíLín captures these tokens as they are produced, often using a streaming API or custom implementation (e.g., via `generate` methods with `stream=True`).

3. **UI Updates**: The generated tokens are sent to the main thread (which handles the UI) via a queue, event loop, or signal-slot mechanism (e.g., PyQt/PySide's `QThread` or `QTimer`). The UI widget (e.g., `QTextEdit`) is updated incrementally with each new token, creating the effect of the answer being written in real time.

4. **Efficient Rendering**: The app updates the UI in small chunks to avoid lag, ensuring smooth real-time feedback without overwhelming the system.

This approach allows users to see the model's output as it is being generated, rather than waiting for the entire response. For specific implementation details, refer to KíLín's documentation or source code for how it integrates with LLMs and GUI frameworks.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## superscript_citations (location, evaluation)

**Question.** How does KíLín recognize citation numbers printed as small raised numbers in a PDF, and how does it avoid mistaking other raised numbers for citations?

**Intent:** code_explanation

**Answer with retrieval:**

KíLín recognizes citation numbers as small raised digits in PDFs by extracting superscript runs using `superscript_runs` [1], which identifies sequences of small, raised digits, commas, and dashes. It leverages PDFMiner to parse text and filter out lowered subscripts, ensuring only true superscripts are considered. Regular expressions like `_RUN_CHARACTER` [2] match these runs, while patterns like `_POWER_OF_TEN` and `_EXPONENT` exclude exponents (e.g., "cm²") and powers of ten (e.g., "10^3") from being mistaken for citations. Contextual checks, such as requiring the digits to follow the same characters on the line, further confirm their role as citations [6]. To avoid errors, KíLín validates numbers against the reference list’s range via `cited_numbers` [10], rejecting outliers or invalid formats. This combination of glyph analysis, regex filtering, and contextual validation ensures citations are distinguished from other raised numbers like exponents or unit markers.

**Cited passages:**

- [1] kilin/tools/reference_check/_citations.py `superscript_runs` (retrieved by: lexical, semantic)
- [2] kilin/tools/reference_check/_citations.py `_citations` (retrieved by: semantic)
- [6] kilin/tools/reference_check/_citations.py `_citations` (retrieved by: semantic)
- [10] kilin/tools/reference_check/_citations.py `cited_numbers` (retrieved by: context_expansion)

**Gold functions:** `superscript_runs`, `_superscripts`, `_confirmed`

- in evidence: ['superscript_runs']
- cited: ['superscript_runs']

**Facts (tick those the answer states correctly):**

- [ ] pdfminer.six reads each line's characters with their size and position; digits, commas, and dashes smaller than a set share of the line's median letter size and raised above its baseline form a superscript run.
- [ ] Each run is stored with the text printed just before it, as a signature.
- [ ] A number found in a sentence counts as a superscript citation only when the PDF confirms a matching raised run there.
- [ ] Superscripts in author lists, affiliations, and equations are skipped; a written-out 'Ref. 50' needs no confirmation.
- [ ] If the PDF cannot be read this way, no superscript is confirmed.

**Errors or unsupported statements:** 

**Answer without retrieval:**

KíLín, a Python package for extracting citations from PDFs, recognizes citation numbers printed as small raised numbers (superscripts) by analyzing **layout, font attributes, and contextual patterns**. Here's how it works:

1. **Superscript Detection**:  
   KíLín identifies superscripted text by checking for **smaller font sizes**, **vertical alignment** (e.g., raised above the baseline), and **font styling** (e.g., italic or specific font weights). These characteristics are common in citation numbers but differ from other raised elements like mathematical exponents or footnotes.

2. **Contextual Positioning**:  
   Citation numbers are typically placed **at the end of sentences or phrases**. KíLín uses **layout analysis** to detect such patterns, distinguishing them from isolated superscripts (e.g., in equations or footnotes).

3. **Citation Style Matching**:  
   The tool may apply **rules based on citation styles** (e.g., APA, MLA, Chicago) to confirm the number is a citation. For example, MLA-style citations often use superscript numbers at the end of sentences, while APA uses parentheses.

4. **Avoiding False Positives**:  
   KíLín avoids mistaking other raised numbers by:  
   - **Checking proximity to text**: Superscripts in equations or footnotes are often surrounded by mathematical or formatting symbols, whereas citations are embedded in prose.  
   - **Analyzing surrounding text**: Citation numbers are usually followed by a reference list or a corresponding entry in the document.  
   - **Font and style consistency**: Citation numbers often use a specific font or size, which may differ from other superscripts (e.g., exponents in equations).

For precise details, refer to KíLín's documentation or source code, as the exact implementation may involve heuristics or machine learning models trained on citation patterns.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## claim_verdict (mechanism, evaluation)

**Question.** Explain how verify_claim reaches a verdict on a cited claim, and how the quote the model gives is checked.

**Intent:** code_explanation

**Answer with retrieval:**

The `verify_claim` function uses the model to generate a verdict by first creating prompts based on the claim and relevant passages [1]. These prompts guide the model to assess whether the cited work supports the claim. The model's response is then parsed by `parse_verdict` [12], which extracts the verdict, quote, and comment. The quote is checked against the provided passages to ensure it matches exactly or near-exactly (with adjustments for minor differences like punctuation or spacing). If no matching passage is found, the verdict may be adjusted (e.g., "contradicted" becomes "not supported"), and the comment notes this discrepancy. The final verdict, along with the validated quote and comment, is returned as part of a `Verdict` object [3]. This process ensures the model's answer is both evaluated against the text and refined for accuracy.

**Cited passages:**

- [1] kilin/tools/_shared/claims.py `verify_claim` (retrieved by: code, lexical, semantic)
- [3] kilin/tools/_shared/claims.py `Verdict` (retrieved by: lexical, semantic)
- [12] kilin/tools/_shared/claims.py `parse_verdict` (retrieved by: context_expansion)

**Gold functions:** `verify_claim`, `_judge`, `parse_verdict`

- in evidence: ['verify_claim', '_judge', 'parse_verdict']
- cited: ['verify_claim', 'parse_verdict']

**Facts (tick those the answer states correctly):**

- [ ] Without passages from the cited file the verdict is 'no text' and the model is not called; a failing model call gives 'failed'.
- [ ] Otherwise the model sees the claim and the cited passages and replies with a verdict, a quote, and a comment.
- [ ] An unrecognized verdict becomes 'unclear', with the reply as comment.
- [ ] The quote is kept only if it is found in a passage word for word, or for most of its words in order; otherwise it is dropped and the comment says so.
- [ ] 'contradicted' without a confirmed quote becomes 'not supported'.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `verify_claim` function in the KíLín Python package evaluates whether a cited claim is supported by the provided evidence. Here's how it works:

1. **Claim and Evidence Analysis**:  
   The function takes a claim and associated evidence (e.g., a text or document) as input. It uses natural language processing (NLP) to analyze the claim and identify relevant parts of the evidence that might support or contradict it.

2. **Model-Based Verification**:  
   A pre-trained model (likely a language model) is used to determine if the claim is **supported**, **refuted**, or **unsupported** by the evidence. The model assesses the logical consistency between the claim and the evidence, checking for factual alignment, relevance, and coherence.

3. **Quote Generation and Validation**:  
   - The model generates a **quote** (or summary) from the evidence that directly supports the claim. This quote is intended to be a concise, accurate representation of the evidence's relevant portion.  
   - The quote is then **cross-checked** against the original evidence to ensure it is verbatim or paraphrased correctly. This step ensures the quote does not introduce errors, misinterpretations, or fabrications.  
   - If the quote aligns with the evidence, the claim is marked as supported. If not, the verdict may shift to "refuted" or "unsupported."

4. **Verdict Determination**:  
   Based on the model's analysis of the claim and the validated quote, a verdict is reached. For example:  
   - **Supported**: The claim is directly confirmed by the evidence.  
   - **Refuted**: The evidence contradicts the claim.  
   - **Unsupported**: The evidence neither supports nor contradicts the claim.

### Key Considerations:
- The quote is critical for transparency, as it allows users to trace the reasoning behind the verdict.  
- The validation of the quote ensures the model does not hallucinate or misrepresent the evidence.  
- The exact implementation details (e.g., model architecture, validation rules) may vary depending on the specific version of KíLín.  

If you need further details about the specific model or validation rules, you may need to consult the KíLín documentation or source code.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## output_saving (mechanism, evaluation)

**Question.** Explain how save_output_file writes a tool's output into the project without overwriting an existing file.

**Intent:** code_explanation

**Answer with retrieval:**

save_output_file writes a tool's output into the project without overwriting existing files by using a two-step process: staging and publishing [1]. First, it creates a temporary staging file in the project's outputs directory using `staged_output_file` [8], ensuring exclusive access to prevent conflicts. The content is written to this staging file. Next, `publish_output_file` [7] attempts to publish the content by creating a hard link to the staging file with the desired name. If the name already exists, it appends a number (e.g., `-2`, `-3`) to generate a unique filename, avoiding overwrites. This process ensures existing files remain untouched, and concurrent saves receive distinct filenames [1]. The function also validates the output name to prevent invalid characters or reserved names [11], and checks for collisions by iterating through numbered alternatives [13]. The final published file is returned as a Path object, ensuring the output is safely stored without replacing existing files.

**Cited passages:**

- [1] kilin/tools/_shared/outputs.py `save_output_file` (retrieved by: code, lexical, semantic)
- [7] kilin/tools/_shared/outputs.py `publish_output_file` (retrieved by: semantic)
- [8] kilin/tools/_shared/outputs.py `staged_output_file` (retrieved by: code)
- [11] kilin/tools/_shared/outputs.py `_output_name` (retrieved by: context_expansion)
- [13] kilin/tools/_shared/outputs.py `_output_candidates` (retrieved by: context_expansion)

**Gold functions:** `save_output_file`, `publish_output_file`, `staged_output_file`

- in evidence: ['save_output_file', 'publish_output_file', 'staged_output_file']
- cited: ['save_output_file', 'publish_output_file', 'staged_output_file']

**Facts (tick those the answer states correctly):**

- [ ] Files go to the project's outputs folder, which may not be a link.
- [ ] The text is first written completely to a private temporary staging file in that folder.
- [ ] It is then published under the requested name with a hard link, which fails if the name exists; the next candidates are name-2, name-3, and so on.
- [ ] If hard links are not supported, the staged file is copied into a newly and exclusively created file, which also cannot overwrite.
- [ ] The staging file is removed afterwards; concurrent saves get different names.

**Errors or unsupported statements:** 

**Answer without retrieval:**

The `save_output_file` function in the KíLín Python package is designed to save a tool's output to a specified file path without overwriting existing files. While the exact implementation details may vary, the general approach typically involves the following steps:

1. **Check for Existing File**: The function first checks whether the target file already exists in the specified directory. This is often done using file system operations like `os.path.exists()`.

2. **Generate a Unique Filename**: If the file exists, the function generates a unique filename to avoid overwriting. This could involve appending a timestamp, a counter, or a unique identifier (e.g., `output_20231010_1234.txt` or `output_v2.txt`).

3. **Save the Output**: The tool's output is then written to the new filename, ensuring the original file remains untouched.

4. **Return or Log the Result**: The function may return the path to the saved file or log the action for user confirmation.

This behavior ensures that existing files are preserved while allowing new outputs to be saved without conflicts. For precise details, refer to the KíLín package's documentation or source code. If the function is not part of KíLín, additional context would be needed.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## newest_listed_callers (dependency, evaluation)

**Question.** Explain which functions call newest_listed, and what each of them does with the earlier turn it returns.

**Intent:** code_explanation

**Answer with retrieval:**

The `newest_listed` function is called by two functions: `run` in `kilin/tools/show_source/tool.py` [3] and `run` in `kilin/tools/filter_results/tool.py` [9].  

1. **`show_source` tool's `run` function**:  
   - Calls `newest_listed(request.earlier)` to retrieve the most recent earlier turn with evidence.  
   - If a valid turn is found, it uses the turn's `answer`, `paths`, and `texts` to extract relevant sentences, construct a summary, and generate an explanation via a language model. The turn’s evidence is formatted into citations and included in the response [3].  

2. **`filter_results` tool's `run` function**:  
   - Calls `newest_listed(request.earlier)` to identify the latest turn with evidence.  
   - If found, it processes the turn’s `texts` and `paths` to create a list of items, batches them for processing, and generates a filtered list of matching items. The turn’s evidence is used to build citations and evidence entries for the response [9].  

Both functions rely on `newest_listed` to access the most recent turn with evidence, then use its metadata (e.g., `texts`, `paths`) to tailor their specific output, such as explanations or filtered results.

**Cited passages:**

- [3] kilin/tools/show_source/tool.py `run` (retrieved by: code)
- [9] kilin/tools/filter_results/tool.py `run` (retrieved by: context_expansion)

**Gold functions:** `run`, `run`

- in evidence: ['run', 'run']
- cited: ['run', 'run']

**Facts (tick those the answer states correctly):**

- [ ] newest_listed returns the newest earlier turn with evidence that the router linked to the question, otherwise the newest turn with evidence.
- [ ] filter_results' run judges that turn's listed items against the question in batches with the model and keeps the matching ones.
- [ ] show_source's run picks the cited passages of that turn that the question and the earlier answer point to, and shows their text word for word with nearby sentences.
- [ ] Without such a turn, both reply that there is no earlier answer to work from.

**Errors or unsupported statements:** 

**Answer without retrieval:**

I currently do not have specific information about a Python package named "KíLín" or a function called `newest_listed` within it. If this refers to a specific library, framework, or tool, additional context or documentation would be needed to provide an accurate answer. 

If you can clarify the package's purpose, domain, or provide a link to its documentation, I’d be happy to help further!

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---

## workspace_path_callers (dependency, evaluation)

**Question.** Explain which functions call default_workspace_path, and when each of them uses the folder it suggests.

**Intent:** code_explanation

**Answer with retrieval:**

The function `prepare_sources_workspace` calls `default_workspace_path` when `workspace_directory` is not provided, using `workspace_root` to generate a default path for the workspace. This occurs during workspace creation, ensuring a deterministic, non-colliding name within the specified parent directory [6]. 

The function `_workspace_path` (defined in `kilin/gui/engine.py`) calls `default_workspace_path` when an explicit `chosen` workspace path is empty, deriving a default path from the first source. This is used to suggest a preview location without reserving or reusing a workspace [1]. 

Both functions rely on `default_workspace_path` to compute a deterministic path based on source metadata and a parent directory, ensuring uniqueness and avoiding collisions. The folder suggested by `default_workspace_path` is used in these contexts to allocate workspace directories, but neither function directly uses the folder itself—only the path it generates.

**Cited passages:**

- [1] kilin/gui/engine.py `_workspace_path` (retrieved by: code, lexical, semantic)
- [6] kilin/workspace.py `prepare_sources_workspace` (retrieved by: code)

**Gold functions:** `_workspace_path`, `prepare_sources_workspace`

- in evidence: ['_workspace_path', 'prepare_sources_workspace']
- cited: ['_workspace_path', 'prepare_sources_workspace']

**Facts (tick those the answer states correctly):**

- [ ] default_workspace_path suggests <first source name>-<8-character hash of its path> inside a parent folder, without creating or reserving it.
- [ ] The GUI's _workspace_path uses it when the user has not chosen a workspace folder; a chosen folder is used as given.
- [ ] prepare_sources_workspace uses it when given a parent folder instead of a workspace folder.
- [ ] Creation reserves the folder exclusively and may pick a numbered sibling if the suggested folder is taken.

**Errors or unsupported statements:** 

**Answer without retrieval:**

I don't have information about a Python package named "KíLín" or its functions. It's possible there may be a typo or confusion with another package name. If you meant a different package (e.g., "Kiln" or another name), please clarify, and I'll do my best to assist. Otherwise, I recommend checking the package's documentation or source code for details about `default_workspace_path` and its usage.

**Without retrieval, the model:** [ ] says it does not know  [ ] invents details  [ ] other: 

---
