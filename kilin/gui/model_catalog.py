"""Describe the model presets displayed by the desktop interface.

``MODEL_CHOICES`` combines core-owned model defaults with GUI presentation
records for model role, backend, and query format. These records are selection
hints, not downloaded model files or runtime compatibility checks.
``installed_backends`` consults installed package metadata; a listed backend
is not thereby proven usable on the current device.

Key Features
------------

- **Catalog records**

    Represent a model preset using immutable ``ModelChoice`` fields.

- **Catalog availability**

    Return all presets with cache flags; the web interface filters choices by
    role and backend. A lightweight snapshot check excludes interrupted
    downloads without loading a model or reading its weight files.

- **Installation hints**

    Expose distribution-presence checks without loading an inference model.

- **Thinking switch**

    Tell, for each cached model whose chat template renders, whether it
    reacts to ``enable_thinking``, by rendering it with it on and off.

Module Structure
----------------

- ``MODEL_CHOICES``

    Tuple of GUI model presets, not a runtime availability check.

- ``ModelChoice``

    One KíLín starting model and the role it is offered for.

- ``installed_backends``

    Check whether the selected computation packages are installed.

- ``model_choices``

    Mark the suggested models that are present in the local cache.

- ``downloaded_models``

    List cached models with a plausible complete local snapshot.

- ``thinking_switches``

    Tell which cached models' chat templates switch thinking on and off.

Examples
--------
>>> import kilin.gui.model_catalog
>>> backends = kilin.gui.model_catalog.installed_backends()
>>> presets = kilin.gui.model_catalog.MODEL_CHOICES

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
from typing import Final, TypeAlias

from kilin.huggingface import available_models
from kilin.model_defaults import (
    DEFAULT_CHAT_MODEL_MLX,
    DEFAULT_CHAT_MODEL_TORCH,
    DEFAULT_EMBEDDING_MODEL_MLX,
    DEFAULT_EMBEDDING_MODEL_TORCH,
    MODEL_ROLE_CHAT,
    MODEL_ROLE_EMBEDDING,
    ModelRole,
)
from kilin.query_formats import QUERY_FORMAT_QWEN3, QueryFormat
from kilin.runtime import BACKEND_MLX, BACKEND_TORCH, SelectedBackend


__all__ = [
    "DEEP_CHAT_MODEL_MLX",
    "LIGHT_CHAT_MODEL_MLX",
    "cached_model_roles",
    "downloaded_models",
    "GUI_REQUIRED_MODEL_IDS",
    "MODEL_CHOICES",
    "ModelChoice",
    "installed_backends",
    "model_choices",
    "thinking_switches",
]

ModelBackend: TypeAlias = SelectedBackend

# Cached models that are neither chat nor embedding models. Only the GUI's
# cache classification uses it; kilin.model_defaults owns the model roles.
MODEL_ROLE_OTHER: Final[str] = "other"

# The GUI's smallest and largest effort models on MLX.
LIGHT_CHAT_MODEL_MLX = "Qwen/Qwen3-0.6B-MLX-4bit"
DEEP_CHAT_MODEL_MLX = "mlx-community/Qwen3-14B-4bit"


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """One KíLín starting model and the role it is offered for.

    Parameters
    ----------
    model_id: str
        Hugging Face repository identifier for this application preset.
    role: ModelRole
        Model role offered by the UI: chat or embedding.
    backend: SelectedBackend
        Backend associated with this preset: mlx or torch.
    query_format: QueryFormat | None, optional
        Embedding query format label when applicable, otherwise None. Defaults
        to ``None``.

    Attributes
    ----------
    model_id: str
        Hugging Face repository identifier for this application preset.
    role: ModelRole
        Model role offered by the UI: chat or embedding.
    backend: SelectedBackend
        Backend associated with this preset: mlx or torch.
    query_format: QueryFormat | None
        Embedding query format label when applicable, otherwise None.

    Notes
    -----
    A catalog record is an application preset, not a runtime load test. It does
    not verify a Hub repository, local cache completeness, installed backend,
    or available hardware.

    """

    model_id: str
    role: ModelRole
    backend: SelectedBackend
    query_format: QueryFormat | None = None


MODEL_CHOICES = (
    ModelChoice(LIGHT_CHAT_MODEL_MLX, MODEL_ROLE_CHAT, BACKEND_MLX),
    ModelChoice(
        DEEP_CHAT_MODEL_MLX,
        MODEL_ROLE_CHAT,
        BACKEND_MLX,
    ),
    ModelChoice(DEFAULT_CHAT_MODEL_MLX, MODEL_ROLE_CHAT, BACKEND_MLX),
    ModelChoice(
        DEFAULT_EMBEDDING_MODEL_MLX,
        MODEL_ROLE_EMBEDDING,
        BACKEND_MLX,
        QUERY_FORMAT_QWEN3,
    ),
    ModelChoice(DEFAULT_CHAT_MODEL_TORCH, MODEL_ROLE_CHAT, BACKEND_TORCH),
    ModelChoice(
        DEFAULT_EMBEDDING_MODEL_TORCH,
        MODEL_ROLE_EMBEDDING,
        BACKEND_TORCH,
        QUERY_FORMAT_QWEN3,
    ),
)

GUI_REQUIRED_MODEL_IDS = {
    BACKEND_MLX: (
        LIGHT_CHAT_MODEL_MLX,
        DEFAULT_CHAT_MODEL_MLX,
        DEEP_CHAT_MODEL_MLX,
        DEFAULT_EMBEDDING_MODEL_MLX,
    ),
    BACKEND_TORCH: (
        DEFAULT_CHAT_MODEL_TORCH,
        DEFAULT_EMBEDDING_MODEL_TORCH,
    ),
}


def installed_backends() -> dict[str, bool]:
    """Check whether the selected computation packages are installed.

    This checks packages, not model weights or device compatibility.

    Returns
    -------
    dict[str, bool]
        Installation status for the MLX and PyTorch backends.

    """

    required = {
        BACKEND_MLX: ("mlx-embeddings", "mlx-lm"),
        BACKEND_TORCH: ("torch", "transformers"),
    }
    result: dict[str, bool] = {}
    for backend, packages in required.items():
        try:
            for package in packages:
                version(package)
        except PackageNotFoundError:
            result[backend] = False
        else:
            result[backend] = True
    return result


def model_choices(cached_models: set[str]) -> list[dict[str, object]]:
    """Mark the suggested models that are present in the local cache.

    Cache presence is not a load test. The actual model is checked when KíLín
    opens the project.

    Parameters
    ----------
    cached_models: set[str]
        Model IDs returned by the local Hugging Face cache scan.

    Returns
    -------
    list[dict[str, object]]
        Suggested choices with role, backend, and local-cache status.

    Notes
    -----
    Returns every preset with a cached flag derived from the supplied
    identifier set. Filtering by role/backend/query format is performed by the
    frontend, not this function.

    """

    return [
        {**asdict(choice), "cached": choice.model_id in cached_models}
        for choice in MODEL_CHOICES
    ]


def cached_model_roles(
    cached_models: list[dict[str, object]],
) -> dict[str, str]:
    """Classify cached models as chat, embedding, or other.

    Presets keep their catalog role. Otherwise a name containing "embed" or
    a sentence-transformers configuration marks an embedding model, and a
    causal or conditional-generation architecture marks a chat model.
    Embedding models converted from chat architectures report a chat
    architecture, so names are checked first.

    Parameters
    ----------
    cached_models: list[dict[str, object]]
        Entries from ``kilin.huggingface.available_models``.

    Returns
    -------
    dict[str, str]
        Role for each cached model ID.

    """

    presets = {choice.model_id: choice.role for choice in MODEL_CHOICES}
    roles: dict[str, str] = {}
    for entry in cached_models:
        model_id = str(entry["model_id"])
        snapshots = Path(str(entry.get("path", ""))) / "snapshots"
        if model_id in presets:
            roles[model_id] = presets[model_id]
            continue
        if "embed" in model_id.lower() or any(
            snapshots.glob("*/config_sentence_transformers.json")
        ):
            roles[model_id] = MODEL_ROLE_EMBEDDING
            continue
        role = MODEL_ROLE_OTHER
        for config in snapshots.glob("*/config.json"):
            try:
                value = json.loads(config.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            architectures = value.get("architectures") or []
            if any(
                str(name).endswith(
                    ("ForCausalLM", "ForConditionalGeneration")
                )
                for name in architectures
            ):
                role = MODEL_ROLE_CHAT
            break
        roles[model_id] = role
    return roles


def thinking_switches(
    cached_models: list[dict[str, object]],
) -> dict[str, bool]:
    """Tell which cached models' chat templates switch thinking.

    KíLín turns thinking on or off by passing ``enable_thinking`` to the
    model's chat template. Each cached template is rendered for one short
    question with it on and off, in a sandbox, the way the Hugging Face
    tokenizer renders it: different prompts mean the switch works, equal
    ones mean the template ignores it. Nothing is guessed from a model's
    name, so a new model is checked the same way once it is downloaded. A
    missing, unreadable or failing template leaves the model out: unknown.

    Parameters
    ----------
    cached_models: list[dict[str, object]]
        Entries from ``kilin.huggingface.available_models``.

    Returns
    -------
    dict[str, bool]
        For each checked model ID, whether its template switches thinking;
        models that could not be checked are not included.

    """

    try:
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        return {}

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    # The helpers the Hugging Face tokenizer gives templates; the date is
    # fixed so both renders differ only by the switch.
    environment.filters["tojson"] = lambda value, **_: json.dumps(value)
    environment.globals["raise_exception"] = _raise_template_error
    environment.globals["strftime_now"] = lambda _format: "1 January 2026"
    found: dict[str, bool] = {}
    for entry in cached_models:
        for snapshot in sorted(
            (Path(str(entry.get("path", ""))) / "snapshots").glob("*")
        ):
            template = _chat_template(snapshot)
            if not template:
                continue
            try:
                compiled = environment.from_string(template)
                prompts = {
                    compiled.render(
                        messages=[{"role": "user", "content": "Hi"}],
                        add_generation_prompt=True,
                        enable_thinking=switch,
                    )
                    for switch in (True, False)
                }
            except Exception:  # pylint: disable=broad-exception-caught
                continue  # A template that fails to render is not verified.
            found[str(entry["model_id"])] = len(prompts) == 2
            break
    return dict(sorted(found.items()))


def _raise_template_error(message: str) -> None:
    """Stop rendering, as a template's ``raise_exception`` does.

    Parameters
    ----------
    message: str
        The template's own error message.

    Raises
    ------
    ValueError
        Always, so the model is not marked.

    """

    raise ValueError(message)


def _chat_template(snapshot: Path) -> str:
    """Return a cached model's chat template, or "" when it has none.

    The same files the Hugging Face tokenizer reads, in its order:
    ``chat_template.jinja``, ``chat_template.json``, then the
    ``chat_template`` in ``tokenizer_config.json`` (the "default" one when
    several are named).

    Parameters
    ----------
    snapshot: pathlib.Path
        One snapshot folder of a cached model.

    Returns
    -------
    str
        The template text, or "" when none can be read.

    """

    try:
        jinja = snapshot / "chat_template.jinja"
        if jinja.is_file():
            return jinja.read_text(encoding="utf-8")
        for name in ("chat_template.json", "tokenizer_config.json"):
            path = snapshot / name
            if not path.is_file():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            template = value.get("chat_template")
            if isinstance(template, list):
                template = next(
                    (
                        item.get("template")
                        for item in template
                        if isinstance(item, dict)
                        and item.get("name") == "default"
                    ),
                    None,
                )
            if isinstance(template, str):
                return template
    except (OSError, ValueError, AttributeError):
        return ""
    return ""


# Recognize only weight and tokenizer layouts supported by the cache
# inventory. This is a file-presence check, not a model compatibility test.
_WEIGHT_PATTERNS: Final[tuple[str, ...]] = (
    "*.safetensors", "*.bin", "*.npz", "*.gguf",
)
_WEIGHT_INDEX_PATTERNS: Final[tuple[str, ...]] = (
    "*.safetensors.index.json", "*.bin.index.json",
)
_TOKENIZER_FILES: Final[tuple[str, ...]] = (
    "tokenizer.json", "tokenizer.model", "spiece.model",
    "sentencepiece.bpe.model", "tokenizer.tiktoken", "vocab.txt",
)
_SHARD_FILENAME: Final[re.Pattern[str]] = re.compile(
    r".+-[0-9]{5}-of-[0-9]{5}\.(?:safetensors|bin)$"
)


def downloaded_models() -> list[dict[str, object]]:
    """List cached models with a plausible complete local snapshot.

    A Hub cache entry can represent an interrupted download. This lightweight,
    offline check requires valid configuration metadata, nonempty tokenizer
    assets, and a supported weight layout with all indexed shards present.
    It never reads weight or tokenizer contents or loads a model. A reported
    model can still fail to load due to incompatible or corrupt files.

    Returns
    -------
    list[dict[str, object]]
        The ``available_models`` entries with a locally complete snapshot.

    """

    return [entry for entry in available_models() if _download_finished(entry)]


def _nonempty_file(path: Path) -> bool:
    """Check one cache file without reading its contents.

    Parameters
    ----------
    path: pathlib.Path
        File to check, including a possible Hugging Face snapshot symlink.

    Returns
    -------
    bool
        Whether the path resolves to a nonempty regular file.

    """

    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        # A missing target or a concurrent cache change is not readiness.
        return False


def _json_object(path: Path) -> dict[str, object] | None:
    """Read a small configuration or weight-index mapping if valid.

    Parameters
    ----------
    path: pathlib.Path
        JSON metadata file; never a tokenizer or model-weight file.

    Returns
    -------
    dict[str, object] | None
        The JSON object, or ``None`` if missing or malformed.

    """

    if not _nonempty_file(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _tokenizer_present(snapshot: Path) -> bool:
    """Check standard tokenizer files without parsing vocabularies.

    Parameters
    ----------
    snapshot: pathlib.Path
        One local Hugging Face snapshot directory.

    Returns
    -------
    bool
        Whether a recognized nonempty tokenizer representation exists.

    """

    return (
        any(_nonempty_file(snapshot / name) for name in _TOKENIZER_FILES)
        or all(
            _nonempty_file(snapshot / name)
            for name in ("vocab.json", "merges.txt")
        )
    )


def _index_complete(index: Path, snapshot: Path) -> bool:
    """Check that a weight index references only present, nonempty shards.

    Parameters
    ----------
    index: pathlib.Path
        Safetensors or PyTorch weight-index metadata.
    snapshot: pathlib.Path
        Snapshot root to which indexed filenames must be relative.

    Returns
    -------
    bool
        Whether the index lists at least one valid weight file and every
        referenced shard is present and nonempty.

    """

    metadata = _json_object(index)
    if metadata is None:
        return False
    weight_map = metadata.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    filenames = weight_map.values()
    if not all(isinstance(name, str) and name for name in filenames):
        return False
    for name in set(filenames):
        relative = Path(name)
        if (
            relative.is_absolute() or ".." in relative.parts
            or relative.suffix not in (".safetensors", ".bin")
            or not _nonempty_file(snapshot / relative)
        ):
            return False
    return True


def _weights_present(snapshot: Path) -> bool:
    """Check a complete indexed checkpoint or one unsharded weight file.

    Parameters
    ----------
    snapshot: pathlib.Path
        One local Hugging Face snapshot directory.

    Returns
    -------
    bool
        Whether a recognized, nonempty weight representation is present.

    """

    indexes = [
        path for pattern in _WEIGHT_INDEX_PATTERNS
        for path in snapshot.glob(pattern)
    ]
    if indexes:
        # A broken index must not be hidden by an unrelated weight file.
        return all(_index_complete(index, snapshot) for index in indexes)
    for pattern in _WEIGHT_PATTERNS:
        for path in snapshot.glob(pattern):
            # Sharded downloads without their index cannot be verified.
            if _SHARD_FILENAME.fullmatch(path.name):
                continue
            if _nonempty_file(path):
                return True
    return False


def _download_finished(entry: dict[str, object]) -> bool:
    """Check whether at least one cached revision is locally complete.

    Parameters
    ----------
    entry: dict[str, object]
        One repository from ``kilin.huggingface.available_models``.

    Returns
    -------
    bool
        Whether one snapshot has plausible metadata, tokenizer and weights.

    Notes
    -----
    This is deliberately not a loader check: no model inference, tokenizer
    parsing, remote request, or large-file hashing is performed.

    """

    for snapshot in sorted(
        (Path(str(entry.get("path", ""))) / "snapshots").glob("*")
    ):
        if (
            _json_object(snapshot / "config.json") is not None
            and _tokenizer_present(snapshot)
            and _weights_present(snapshot)
        ):
            return True
    return False
