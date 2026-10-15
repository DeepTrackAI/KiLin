# Trace-based performance analysis

The benchmark examines existing QAChat behavior without changing the
answering pipeline or adding production instrumentation. The main tool is
[`assets/benchmark_traces.ipynb`](../assets/benchmark_traces.ipynb), supported
by a small, read-only parser in
[`assets/trace_benchmark.py`](../assets/trace_benchmark.py).

## Scope and method

The notebook supports two workflows:

1. **Analyze saved traces (default):** point `EXISTING_TRACES` to explicit
   conversation JSON files and run the notebook. No models or projects are
   opened. Readable spans are extracted from the existing numbered session,
   turn, stage, and operation hierarchy.
2. **Repeat standardized questions (opt-in):** create a separate QAChat project
   under `sandbox/`, configure its models in `profile.ini`, disable model
   downloads, and set `RUN_BENCHMARK = True`. Each repetition creates a new
   conversation in this benchmark project and asks the same ordered questions.
   The notebook measures external wall time around session initialization and
   each `RAGSession.ask()` call. Existing QAChat traces provide stage timings.

The included `code` and `literature` question sets each contain three fixed
question roles: overview, mechanism, and limitations. They are intended for
**comparative timing**, not for evaluating answer correctness. Record the
project snapshot, operating system, backend, model identifiers, token settings,
retrieval count, corpus size, question set, trace state, and repetition count
when interpreting results. Do not alter the question text/order across runs.

By default, the benchmark chooses the project's configured **default intent**
(`USE_ROUTER = False`) to avoid introducing router or tool-selection
variability. Changing this option changes the workload, so record it as part
of the benchmark configuration. The sequence shares conversation history;
subsequent questions are not independent of earlier answers.

### Preparation for a local model run

1. Prepare a dedicated QAChat project under
   `QAChat/sandbox/performance_workspace/`, using normal QAChat project
   creation. A regular project is not suitable for benchmark writes.
2. Add a fixed, small source corpus and verify the intended profile, retrieval
   settings, and locally available models. Set `[models] allow_download = false`
   in its `profile.ini`. The notebook refuses to run if this is not set.
3. Open `assets/benchmark_traces.ipynb` with the QAChat Python environment.
   Set `QUESTION_SET`, `REPEATS` (for example, `2`), and `RUN_BENCHMARK`.
4. Execute the notebook in order. For a coarse tracing on/off comparison, set
   `TRACE_SETTINGS = (True, False)`; otherwise keep `(True,)` for stage detail.
5. Examine the per-question wall-clock medians and the trace-derived
   per-turn operations. The notebook displays a plot if matplotlib is already
   installed; it does not install packages.

Each repetition creates and closes a fresh RAG session. It does not reset
operating-system caches, so the result is **not** a controlled hardware cold
start. The notebook sets commonly supported Hugging Face offline flags before
loading QAChat; other third-party backends may not honor them. The actual
`profile.ini` `allow_download = false` setting is the primary QAChat safeguard.

## Measurements and meanings

| Metric | Source | Interpretation |
|---|---|---|
| `startup_ms` | External `perf_counter()` | Creating `RAGSession`, including project, index, and model preparation |
| `wall_ms` | External `perf_counter()` | Full `RAGSession.ask()` call, regardless of diagnostic tracing |
| `turn_ms` | `turn` span | Inclusive traced turn; includes nested work and persistence |
| `routing_ms` | `intent.classification` span | Router classification when enabled; absent when bypassed |
| `semantic_ms` | `retrieval.semantic` span | Semantic search operation |
| `lexical_ms` | `retrieval.lexical` span | Lexical search operation |
| `code_ms` | `retrieval.code` span | Code-graph search operation, when applicable |
| `generation_ms` | `llm.call.output` | Sum of recorded generation durations for a turn |
| `output_tokens_per_second` | `llm.call.output` | Recorded output tokens divided by recorded generation time |
| `median_ms`, `min_ms`, `max_ms` | External per-question repeats | Median and observed range, separated by trace state |

The notebook also produces an operation-level listing for initialization and
other recorded spans, including `session.ready` and model-load spans. This
allows inspection of startup components without introducing new counters.

The `turn_ms` measurement is **inclusive**: never add routing, retrieval or
generation durations to it. Similarly, individual retrieval operations can
be nested under other retrieval spans; they cannot be added indiscriminately.
The tool intentionally does not invent an aggregate retrieval metric.
Missing stages mean **not measured/not applicable**, not zero milliseconds.
Incomplete or failed spans are not considered completed measurements.

The output-token count is trace-dependent: it can be unavailable or derived
from decoded text rather than backend token IDs. A turn generated through
other tools may contain several calls or no LLM generation at all. The
analysis labels these cases rather than assuming one generation per question.

Time to first generated token, GUI-rendering latency, and actual hardware
cold-start latency are not available from the current trace. They must not
be inferred from the reported durations.

## Small tracing-overhead check

The notebook has two optional and deliberately lightweight comparisons:

- Set `TRACE_SETTINGS = (True, False)` to compare whole-answer wall time with
  diagnostic tracing enabled and disabled, using the same question sequence.
  With tracing disabled, **conversation checkpoints still persist**, but
  operation-level spans cannot be recovered.
- Set `TRACE_WRITE_CHECK = True` for a single batch of 16 synthetic events
  with an enabled logger and a null logger. This is a rough sanity check, not
  a statistically reliable estimate of disk-write overhead or an end-to-end
  performance benchmark. It does not benchmark mandatory checkpoints.

Neither result should be interpreted as a precise measurement of tracing's
cost without repetitions, matched workloads, and control of run ordering.
These checks do not modify trace persistence or establish that an
optimization is necessary.

## Data location and privacy

Enabled QAChat traces record diagnostic content, including prompts, answers,
retrieved passages, token IDs, local file paths, and exceptions. **Do not
share raw traces as benchmark output.** The parser deliberately returns only
safe, enumerated timing fields, status, counts, and a user-assigned run label.
It does not copy, redact, or rewrite source trace files. The original files
remain sensitive and must be handled according to
[`trace_recording.md`](trace_recording.md).

If `EXPORT_CSV = True`, timing-only summaries are stored under
`sandbox/performance_pr4/`, which is ignored by Git and omitted from the
source ZIP. A live run writes new conversations and traces only into the
explicitly selected dedicated benchmark workspace. The default read-only
notebook run creates no files. Review run labels before sharing even the
sanitized summaries.

## Tests and reproducibility

Deterministic tests in `tests/test__trace_benchmark.py` check trace parsing,
per-turn separation, missing/failed spans, privacy, summary statistics,
notebook syntax, and safe opt-in behavior. They do not load real models or
measure hardware performance.

For a baseline comparison, record all individual samples rather than just
medians. Use the same fixed project snapshot and questions, repeat each
configuration, and report the observed spread. A small number of repeats is
sufficient for exploratory diagnostics, **not** statistical significance.
The user's native macOS/Python 3.12 execution is needed for representative
MLX measurements. The test suite cannot substitute for hardware benchmarking.
