"""Load a catalogued local model and have a short chat. Never download here."""

import gc
from importlib import import_module
from pathlib import Path
import platform

from app_core.models import check_model, model_info, package_versions


def load_model(name: str, directory: str | Path | None = None, *, backend: str = "auto") -> "ChatSession":
    """Use the model's matching engine. Check all file contents before loading."""
    engine = model_info(name)["backend"]
    if backend not in {"auto", "mlx", "torch"}:
        raise ValueError("Choose auto, mlx, or torch.")
    if backend != "auto" and backend != engine:
        raise ValueError(f"{name} needs {engine} weights and engine, not {backend}. Choose a matching model.")
    if engine == "mlx" and (platform.system(), platform.machine().lower()) != ("Darwin", "arm64"):
        raise ValueError("MLX models need an Apple Silicon Mac. Choose a torch model on this computer.")
    missing = [package for package, installed in package_versions(name).items() if installed is None]
    if missing:
        raise RuntimeError(f"Missing packages: {', '.join(missing)}. Rerun setup with the {engine} extra.")

    report = check_model(name, directory, verify=True)  # Breakpoint 1: inspect the local files.
    if report["status"] != "verified":
        raise FileNotFoundError(
            f"Model files are {report['status']} at {report['path']}. "
            "Use notebooks/model_download.ipynb or scripts/download_model.py first."
        )
    path = report["path"]
    local_only = {"local_files_only": True, "trust_remote_code": False}
    if engine == "mlx":
        mx = import_module("mlx.core")
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is unavailable. Check setup before loading this model.")
        mlx_lm = import_module("mlx_lm")
        model, tokenizer = mlx_lm.load(path, tokenizer_config=local_only)  # Breakpoint 2: load weights.
        device = "metal"
    else:
        torch = import_module("torch")
        transformers = import_module("transformers")
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        tokenizer = transformers.AutoTokenizer.from_pretrained(path, **local_only)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            path, **local_only, use_safetensors=True,
            dtype=torch.float32 if device == "cpu" else "auto",
        )
        model.to(device)
        model.eval()
    return ChatSession(name, engine, device, model, tokenizer)


class ChatSession:
    """One loaded model and its conversation. History stays in memory, not on disk."""

    def __init__(self, name, backend, device, model, tokenizer):
        self.name = name
        self.backend = backend
        self.device = device
        self.messages = []
        self._model = model
        self._tokenizer = tokenizer

    def reply(self, message: str, *, max_new_tokens: int = 128) -> str:
        """Generate a reply; save the turn only after generation succeeds."""
        if self._model is None:
            raise RuntimeError("This chat is closed. Load a model again.")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Enter a non-empty message.")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be a whole number from 1 to 512.")
        messages = [*self.messages, {"role": "user", "content": message.strip()}]
        # The model's own template formats roles and turns into input tokens.
        tokens = self._tokenizer.apply_chat_template(  # Breakpoint 3: inspect the conversation.
            messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        if len(tokens) + max_new_tokens > 4096:
            raise ValueError("This small example allows 4096 tokens. Shorten the message or call chat.reset().")
        if self.backend == "mlx":
            mlx_lm = import_module("mlx_lm")
            sampling = import_module("mlx_lm.sample_utils")
            answer = mlx_lm.generate(  # Breakpoint 4: generate new text.
                self._model, self._tokenizer, prompt=tokens,
                max_tokens=max_new_tokens, sampler=sampling.make_sampler(temp=0.0), verbose=False,
            )
        else:
            torch = import_module("torch")
            inputs = torch.tensor([tokens], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                output = self._model.generate(
                    input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            # Generated output includes the input; decode only the new tokens.
            answer = self._tokenizer.decode(output[0, len(tokens):].tolist(), skip_special_tokens=True)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The model returned no text. Try another short message.")
        answer = answer.strip()
        self.messages = [*messages, {"role": "assistant", "content": answer}]
        return answer

    def reset(self):
        """Forget the conversation, but keep the model loaded."""
        self.messages.clear()

    def close(self):
        """Release this session's weights, history, and unused engine memory."""
        if self._model is None:
            return
        self._model = None  # Breakpoint 5: drop our references to the loaded model.
        self._tokenizer = None
        self.reset()
        gc.collect()
        if self.backend == "mlx":
            import_module("mlx.core").clear_cache()
        elif self.device == "cuda":
            import_module("torch").cuda.empty_cache()
        elif self.device == "mps":
            import_module("torch").mps.empty_cache()
