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
from importlib import import_module
import math
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from qachat.huggingface import cached_model_snapshot
from qachat.runtime import (
    Backend,
    SelectedBackend,
    choose_backend,
    torch_device,
)


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
    ) -> None:
        """Check settings, find the snapshot, and load one backend."""

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(
                "Enter a Hugging Face embedding model ID or local path."
            )

        if not isinstance(allow_download, bool):
            raise ValueError("`allow_download` must be True or False.")

        self.model_id = model_id.strip()
        self.backend = choose_backend(backend)
        self.device = ""
        self._model = None
        self._tokenizer = None

        source = _model_source(
            self.model_id,
            allow_download=allow_download,
            revision=revision,
        )
        if self.backend == "mlx":
            self._load_mlx(source)
        else:
            self._load_torch(source)

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

        if self.backend == "mlx":
            vectors = self._embed_mlx(checked)
        else:
            vectors = self._embed_torch(checked)

        return _check_vectors(vectors, len(checked))

    def embed_query(
        self: EmbeddingSession,
        query_text: str,
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
        return self.embed_documents([checked])[0]

    def close(self: EmbeddingSession) -> None:
        """Release model references and backend caches."""

        self._model = None
        self._tokenizer = None
        gc.collect()

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
