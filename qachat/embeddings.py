"""Load one embedding model and turn text into comparable vectors.

Classes
-------
EmbeddingSession
    Load compatible MLX or PyTorch embedding weights, count tokens, and embed
    document or query text supplied by the caller.

Chat models and embedding models are independent. A later RAG layer can use
any supported chat model with any supported embedding model.

"""

from __future__ import annotations

import gc
from hashlib import sha256
from importlib import import_module
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from huggingface_hub import snapshot_download

from qachat.huggingface import cached_model_snapshot
from qachat.runtime import (
    Backend,
    SelectedBackend,
    choose_backend,
    torch_device,
)
from qachat.trace_logging import NullTraceLogger, TraceSink, trace_text_preview

__all__ = [
    "EmbeddingSession",
]


class EmbeddingSession:
    """One loaded embedding model and tokenizer.

    Parameters
    ----------
    model_id: str
        Hugging Face model ID or local model snapshot.
    backend: "auto" | "mlx" | "torch", optional
        ``auto`` selects MLX on Apple Silicon and PyTorch elsewhere.
    allow_download: bool, optional
        Whether Hugging Face may download missing files to its normal cache.
    revision: str | None, optional
        Optional Hugging Face branch, tag, or commit.
    trace: TraceSink | None, optional
        Structured execution trace shared with the calling session.

    Attributes
    ----------
    model_id: str
        Selected Hugging Face model ID or local directory.
    backend: "mlx" | "torch"
        Selected calculation backend.
    device: str
        Device used for embedding calculations.

    """

    model_id: str
    backend: SelectedBackend
    device: str
    _model: Any | None
    _tokenizer: Any | None

    def __init__(
        self: EmbeddingSession,
        model_id: str,
        *,
        backend: Backend = "auto",
        allow_download: bool = True,
        revision: str | None = None,
        trace: TraceSink | None = None,
    ) -> None:
        """Check settings, find the snapshot, and load one backend."""

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(
                "Enter a Hugging Face embedding model ID or local path."
            )

        if not isinstance(allow_download, bool):
            raise ValueError("`allow_download` must be True or False.")

        self.trace = trace if trace is not None else NullTraceLogger()
        self.model_id = model_id.strip()
        self.backend = choose_backend(backend)
        self.device = ""
        self._model = None
        self._tokenizer = None

        with self.trace.span(
            "model.embedding.load",
            component="embeddings",
            data={
                "model_id": self.model_id,
                "backend": self.backend,
                "allow_download": allow_download,
                "requested_revision": revision,
            },
        ):
            source = _model_source(
                self.model_id,
                allow_download=allow_download,
                revision=revision,
            )
            self.model_source = source
            if self.backend == "mlx":
                self._load_mlx(source)
            else:
                self._load_torch(source)
            self.trace.emit(
                "model.embedding.resolved",
                component="embeddings",
                data={
                    "model_id": self.model_id,
                    "model_source": source,
                    "requested_revision": revision,
                    "backend": self.backend,
                    "device": self.device,
                },
            )

    def count_tokens(self: EmbeddingSession, text: str) -> int:
        """Count tokens with the selected embedding tokenizer.

        Parameters
        ----------
        text: str
            Text to count.

        Returns
        -------
        int
            Number of tokens including the model's special tokens.

        Raises
        ------
        RuntimeError
            If the session is closed.
        ValueError
            If text is not a string.

        """

        if self._tokenizer is None:
            raise RuntimeError("This embedding session is closed.")

        if not isinstance(text, str):
            raise ValueError("Text to count must be a string.")

        tokens = self._tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        return len(tokens)

    def embed_documents(
        self: EmbeddingSession,
        texts: list[str],
        *,
        trace_turn_id: str | None = None,
        purpose: str = "documents",
    ) -> list[list[float]]:
        """Embed repository text without changing it.

        Parameters
        ----------
        texts: list[str]
            Non-empty repository text values.

        Returns
        -------
        list[list[float]]
            One normalized vector for each input.

        Raises
        ------
        RuntimeError
            If the session is closed or returns invalid vectors.
        ValueError
            If an input is empty or invalid.

        """

        checked = _check_texts(texts)
        if not checked:
            return []

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("This embedding session is closed.")

        # Index construction already has one enclosing index.semantic.build
        # span. Recording every document batch, token ID, mask, and vector
        # digest makes that trace enormous without adding another decision.
        record_details = purpose != "documents"
        operation_id = self.trace.new_id("embedding")
        if record_details:
            input_data: dict[str, object] = {
                "operation_id": operation_id,
                "purpose": purpose,
                "model_id": self.model_id,
                "backend": self.backend,
                "device": self.device,
                "text_count": len(checked),
                "text_sha256": [
                    sha256(text.encode("utf-8")).hexdigest()
                    for text in checked
                ],
            }
            if self.trace.include_content:
                input_data["text_previews"] = [
                    trace_text_preview(text) for text in checked
                ]
            self.trace.emit(
                "embedding.input.prepared",
                component="embeddings",
                data=input_data,
                turn_id=trace_turn_id,
            )

        self._last_embedding_input_token_counts = None
        calculation_started_at = perf_counter()

        def calculate() -> list[list[float]]:
            """Run one backend call and validate its vectors."""

            if self.backend == "mlx":
                if not self.trace.enabled:
                    vectors = self._embed_mlx(checked)
                else:
                    vectors = self._embed_mlx(
                        checked,
                        operation_id=(
                            operation_id if record_details else None
                        ),
                        trace_turn_id=trace_turn_id,
                        record_trace=record_details,
                    )
            else:
                if not self.trace.enabled:
                    vectors = self._embed_torch(checked)
                else:
                    vectors = self._embed_torch(
                        checked,
                        operation_id=(
                            operation_id if record_details else None
                        ),
                        trace_turn_id=trace_turn_id,
                        record_trace=record_details,
                    )
            return _check_vectors(vectors, len(checked))

        if record_details:
            with self.trace.span(
                "embedding.calculate",
                component="embeddings",
                data={
                    "operation_id": operation_id,
                    "purpose": purpose,
                    "text_count": len(checked),
                },
                turn_id=trace_turn_id,
            ):
                checked_vectors = calculate()
        else:
            checked_vectors = calculate()
        calculation_duration_ms = (
            perf_counter() - calculation_started_at
        ) * 1000

        if not record_details:
            return checked_vectors

        result_data: dict[str, object] = {
            "operation_id": operation_id,
            "purpose": purpose,
            "vector_count": len(checked_vectors),
            "dimensions": (len(checked_vectors[0]) if checked_vectors else 0),
            "input_token_counts": self._last_embedding_input_token_counts,
            "input_token_count": (
                sum(self._last_embedding_input_token_counts)
                if self._last_embedding_input_token_counts is not None
                else None
            ),
            "calculation_duration_ms": round(
                calculation_duration_ms,
                3,
            ),
        }
        if self.trace.include_embedding_vectors:
            result_data["vectors_artifact"] = self.trace.artifact(
                f"{operation_id}-vectors",
                checked_vectors,
            )
        else:
            encoded = json.dumps(
                checked_vectors,
                separators=(",", ":"),
            ).encode("utf-8")
            result_data["vectors_sha256"] = sha256(encoded).hexdigest()
        self.trace.emit(
            "embedding.output.completed",
            component="embeddings",
            data=result_data,
            turn_id=trace_turn_id,
            duration_ms=calculation_duration_ms,
        )
        return checked_vectors

    def embed_query(
        self: EmbeddingSession,
        query_text: str,
        *,
        trace_turn_id: str | None = None,
    ) -> list[float]:
        """Embed one query exactly as provided.

        Parameters
        ----------
        query_text: str
            Text formatted for the selected embedding model.

        Returns
        -------
        list[float]
            Normalized query vector.

        Raises
        ------
        ValueError
            If the query text is empty.

        """

        checked = _check_texts([query_text])[0]
        return self.embed_documents(
            [checked],
            trace_turn_id=trace_turn_id,
            purpose="query",
        )[0]

    def close(self: EmbeddingSession) -> None:
        """Release model references and backend caches."""

        self._model = None
        self._tokenizer = None
        gc.collect()
        self._clear_backend_cache()

    def _clear_backend_cache(self: EmbeddingSession) -> None:
        """Release the backend's cached allocator memory.

        ``close()`` calls this once, when the session ends. A caller
        embedding many batches in one un-closed session (for example
        ``build_semantic_index`` batching over a large chunk list) can
        call it after each batch too -- MLX in particular keeps every
        batch's freed buffers cached for reuse rather than returning
        them to the OS, so the cache otherwise grows across batches even
        when no single batch is large.
        """

        if self.backend == "mlx":
            mx = import_module("mlx.core")
            mx.clear_cache()
        else:
            torch = import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def _load_mlx(self: EmbeddingSession, source: str) -> None:
        """Load an MLX embedding model and tokenizer.

        Parameters
        ----------
        source: str
            Local Hugging Face snapshot.

        Raises
        ------
        RuntimeError
            If Metal is unavailable.

        """

        mx = import_module("mlx.core")
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is not available.")

        transformers = import_module("transformers")
        utilities = import_module("mlx_embeddings.utils")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            padding_side="right",
        )
        self._model = utilities.load_model(Path(source))
        self.device = "metal"

    def _load_torch(self: EmbeddingSession, source: str) -> None:
        """Load a PyTorch embedding model and tokenizer.

        Parameters
        ----------
        source: str
            Local Hugging Face snapshot.

        """

        torch = import_module("torch")
        transformers = import_module("transformers")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            padding_side="right",
        )
        self._model = transformers.AutoModel.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            dtype="auto",
        )
        self.device = torch_device(
            torch.cuda.is_available(),
            torch.backends.mps.is_available(),
        )
        self._model.to(self.device)
        self._model.eval()

    def _embed_mlx(
        self: EmbeddingSession,
        texts: list[str],
        *,
        operation_id: str | None = None,
        trace_turn_id: str | None = None,
        record_trace: bool = True,
    ) -> list[list[float]]:
        """Calculate normalized MLX vectors.

        Parameters
        ----------
        texts: list[str]
            Checked input text.

        Returns
        -------
        list[list[float]]
            Model vectors.

        """

        mx = import_module("mlx.core")
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="np",
        )
        if self.trace.enabled and record_trace:
            token_counts = inputs["attention_mask"].sum(axis=1).tolist()
            self._last_embedding_input_token_counts = token_counts
            token_data: dict[str, object] = {
                "operation_id": operation_id,
                "input_token_counts": token_counts,
                "input_token_count": sum(token_counts),
            }
            if self.trace.include_token_ids:
                token_data.update(
                    {
                        "input_ids_artifact": self.trace.artifact(
                            f"{operation_id}-input-token-ids",
                            inputs["input_ids"].tolist(),
                        ),
                        "attention_mask_artifact": self.trace.artifact(
                            f"{operation_id}-attention-mask",
                            inputs["attention_mask"].tolist(),
                        ),
                    }
                )
            self.trace.emit(
                "embedding.tokens.prepared",
                component="embeddings",
                data=token_data,
                turn_id=trace_turn_id,
            )
        outputs = self._model(
            mx.array(inputs["input_ids"]),
            attention_mask=mx.array(inputs["attention_mask"]),
        )
        vectors = outputs.text_embeds.astype(mx.float32)
        lengths = mx.linalg.norm(vectors, axis=-1, keepdims=True)
        vectors = vectors / lengths
        mx.eval(vectors)
        return vectors.tolist()

    def _embed_torch(
        self: EmbeddingSession,
        texts: list[str],
        *,
        operation_id: str | None = None,
        trace_turn_id: str | None = None,
        record_trace: bool = True,
    ) -> list[list[float]]:
        """Calculate normalized PyTorch vectors.

        Parameters
        ----------
        texts: list[str]
            Checked input text.

        Returns
        -------
        list[list[float]]
            Model vectors.

        """

        torch = import_module("torch")
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        if self.trace.enabled and record_trace:
            token_counts = (
                inputs["attention_mask"].detach().cpu().sum(dim=1).tolist()
            )
            self._last_embedding_input_token_counts = token_counts
            token_data: dict[str, object] = {
                "operation_id": operation_id,
                "input_token_counts": token_counts,
                "input_token_count": sum(token_counts),
            }
            if self.trace.include_token_ids:
                token_data.update(
                    {
                        "input_ids_artifact": self.trace.artifact(
                            f"{operation_id}-input-token-ids",
                            inputs["input_ids"].detach().cpu().tolist(),
                        ),
                        "attention_mask_artifact": self.trace.artifact(
                            f"{operation_id}-attention-mask",
                            inputs["attention_mask"].detach().cpu().tolist(),
                        ),
                    }
                )
            self.trace.emit(
                "embedding.tokens.prepared",
                component="embeddings",
                data=token_data,
                turn_id=trace_turn_id,
            )
        with torch.inference_mode():
            hidden = self._model(**inputs).last_hidden_state
            mask = inputs["attention_mask"]
            positions = torch.arange(
                mask.shape[1],
                device=mask.device,
            ).expand_as(mask)
            final_positions = (
                positions.masked_fill(
                    mask == 0,
                    -1,
                )
                .max(dim=1)
                .values
            )
            vectors = hidden[
                torch.arange(hidden.shape[0], device=hidden.device),
                final_positions,
            ].float()
            vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)

        return vectors.cpu().tolist()


def _model_source(
    model_id: str,
    *,
    allow_download: bool,
    revision: str | None,
) -> str:
    """Resolve a local directory or Hugging Face snapshot.

    Parameters
    ----------
    model_id: str
        Hugging Face model ID or local directory.
    allow_download: bool
        Whether missing files may be downloaded.
    revision: str | None
        Optional model revision.

    Returns
    -------
    str
        Local model snapshot.

    """

    local = Path(model_id).expanduser()
    if local.is_dir():
        return str(local)

    if not allow_download:
        return str(cached_model_snapshot(model_id, revision=revision))

    return snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_files_only=False,
    )


def _check_texts(texts: list[str]) -> list[str]:
    """Check embedding inputs without modifying valid text.

    Parameters
    ----------
    texts: list[str]
        Values to check.

    Returns
    -------
    list[str]
        Original text values.

    Raises
    ------
    ValueError
        If the collection or one text value is invalid.

    """

    if not isinstance(texts, list):
        raise ValueError("Embedding input must be a list of strings.")

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding text cannot be empty.")

    return texts


def _check_vectors(
    vectors: list[list[float]],
    expected_count: int,
) -> list[list[float]]:
    """Check embedding count, dimensions, and finite values.

    Parameters
    ----------
    vectors: list[list[float]]
        Vectors returned by one backend.
    expected_count: int
        Number of embedded text values.

    Returns
    -------
    list[list[float]]
        Checked vectors.

    Raises
    ------
    RuntimeError
        If vectors are missing, inconsistent, or non-finite.

    """

    if len(vectors) != expected_count or not vectors:
        raise RuntimeError("The embedding model returned the wrong count.")

    dimension = len(vectors[0])
    invalid = (
        dimension == 0
        or any(len(vector) != dimension for vector in vectors)
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for vector in vectors
            for value in vector
        )
        or any(
            not any(float(value) != 0.0 for value in vector)
            for vector in vectors
        )
    )
    if invalid:
        raise RuntimeError("The embedding model returned invalid vectors.")

    return [[float(value) for value in vector] for vector in vectors]
