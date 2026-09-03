"""Chat checks use stand-ins and one tiny local PyTorch model, never downloads."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from app_core import chat, models


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """Small stand-ins let any computer check both engine branches."""
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = [1, 2, 3]
    tokenizer.eos_token_id = 9
    tokenizer.decode.return_value = " Hello Sam! "
    model = Mock()
    output = MagicMock()
    output.__getitem__.return_value.tolist.return_value = [7, 8]
    model.generate.return_value = output
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=Mock(return_value=False), empty_cache=Mock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=False))),
        mps=SimpleNamespace(empty_cache=Mock()), float32="float32", long="long",
        tensor=Mock(return_value=object()), ones_like=Mock(return_value=object()),
        inference_mode=Mock(side_effect=nullcontext),
    )
    modules = {
        "mlx.core": SimpleNamespace(metal=SimpleNamespace(is_available=Mock(return_value=True)), clear_cache=Mock()),
        "mlx_lm": SimpleNamespace(load=Mock(return_value=(model, tokenizer)), generate=Mock(return_value=" Hello Sam! ")),
        "mlx_lm.sample_utils": SimpleNamespace(make_sampler=Mock(return_value="greedy")),
        "torch": torch,
        "transformers": SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        ),
    }
    importer = Mock(side_effect=modules.__getitem__)
    monkeypatch.setattr(chat, "import_module", importer)
    monkeypatch.setattr(chat.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(chat.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(chat, "model_info", lambda name: {"backend": name})
    monkeypatch.setattr(chat, "package_versions", lambda name: {name: "installed"})
    check = Mock(return_value={"status": "verified", "path": str(tmp_path / "saved-model")})
    monkeypatch.setattr(chat, "check_model", check)
    monkeypatch.setattr(chat.gc, "collect", Mock())
    return SimpleNamespace(modules=modules, importer=importer, check=check,
                           model=model, tokenizer=tokenizer, torch=torch, output=output)


def test_import_does_not_load_engines():
    result = subprocess.run(
        [sys.executable, "-c", "from app_core import chat; import sys; "
         "assert not {'mlx.core', 'torch', 'transformers', 'huggingface_hub'} & sys.modules.keys()"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("engine", ["mlx", "torch"])
def test_load_checks_full_files_and_uses_explicit_folder(runtime, tmp_path, engine):
    session = chat.load_model(engine, directory=tmp_path, backend=engine)
    runtime.check.assert_called_once_with(engine, tmp_path, verify=True)
    assert (session.name, session.backend) == (engine, engine)
    assert session.messages == []
    if engine == "mlx":
        runtime.modules["mlx_lm"].load.assert_called_once_with(
            runtime.check.return_value["path"],
            tokenizer_config={"local_files_only": True, "trust_remote_code": False},
        )
        assert "torch" not in [call.args[0] for call in runtime.importer.call_args_list]
    else:
        loader = runtime.modules["transformers"].AutoModelForCausalLM.from_pretrained
        assert loader.call_args.kwargs == {
            "local_files_only": True, "trust_remote_code": False,
            "use_safetensors": True, "dtype": "float32",
        }
        assert runtime.modules["transformers"].AutoTokenizer.from_pretrained.call_args.kwargs == {
            "local_files_only": True, "trust_remote_code": False,
        }
        runtime.model.to.assert_called_once_with("cpu")
        runtime.model.eval.assert_called_once()
        assert "mlx.core" not in [call.args[0] for call in runtime.importer.call_args_list]


@pytest.mark.parametrize("backend", ["bad", "torch"])
def test_incorrect_engine_rejected_before_loading(runtime, backend):
    with pytest.raises(ValueError):
        chat.load_model("mlx", backend=backend)
    runtime.importer.assert_not_called()
    runtime.check.assert_not_called()


def test_unknown_model_rejected(runtime, monkeypatch):
    monkeypatch.setattr(chat, "model_info", models.model_info)
    with pytest.raises(ValueError, match="Unknown model"):
        chat.load_model("not-catalogued")
    runtime.importer.assert_not_called()


@pytest.mark.parametrize("system,machine", [("Windows", "AMD64"), ("Linux", "aarch64"), ("Darwin", "x86_64")])
def test_mlx_rejected_on_wrong_platform(runtime, monkeypatch, system, machine):
    monkeypatch.setattr(chat.platform, "system", lambda: system)
    monkeypatch.setattr(chat.platform, "machine", lambda: machine)
    with pytest.raises(ValueError, match="Apple Silicon"):
        chat.load_model("mlx")
    runtime.importer.assert_not_called()


def test_missing_package_has_setup_instructions(runtime, monkeypatch):
    monkeypatch.setattr(chat, "package_versions", lambda name: {"mlx-lm": None})
    with pytest.raises(RuntimeError, match="Missing packages: mlx-lm"):
        chat.load_model("mlx")
    runtime.importer.assert_not_called()


@pytest.mark.parametrize("status", ["missing", "incomplete", "corrupt"])
def test_bad_local_files_never_load_or_download(runtime, status):
    runtime.check.return_value["status"] = status
    with pytest.raises(FileNotFoundError, match="model_download.ipynb"):
        chat.load_model("mlx")
    runtime.importer.assert_not_called()


def test_missing_metal_does_not_fall_back(runtime):
    runtime.modules["mlx.core"].metal.is_available.return_value = False
    with pytest.raises(RuntimeError, match="Metal is unavailable"):
        chat.load_model("mlx")
    runtime.modules["mlx_lm"].load.assert_not_called()


@pytest.mark.parametrize("cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_torch_device_choice(runtime, cuda, mps, expected):
    runtime.torch.cuda.is_available.return_value = cuda
    runtime.torch.backends.mps.is_available.return_value = mps
    session = chat.load_model("torch")
    assert session.device == expected
    runtime.model.to.assert_called_once_with(expected)


@pytest.mark.parametrize("engine", ["mlx", "torch"])
def test_reply_uses_template_and_keeps_both_turns(runtime, engine):
    session = chat.load_model(engine)
    assert session.reply(" Hello! ", max_new_tokens=32) == "Hello Sam!"
    session.reply("What is my name?", max_new_tokens=16)
    expected = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello Sam!"},
        {"role": "user", "content": "What is my name?"},
    ]
    runtime.tokenizer.apply_chat_template.assert_called_with(
        expected, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert session.messages == expected + [{"role": "assistant", "content": "Hello Sam!"}]
    if engine == "mlx":
        runtime.modules["mlx_lm"].generate.assert_called_with(
            runtime.model, runtime.tokenizer, prompt=[1, 2, 3], max_tokens=16,
            sampler="greedy", verbose=False,
        )
    else:
        kwargs = runtime.model.generate.call_args.kwargs
        assert kwargs["max_new_tokens"] == 16
        assert kwargs["do_sample"] is False
        assert kwargs["attention_mask"] is runtime.torch.ones_like.return_value
        assert runtime.torch.inference_mode.call_count == 2
        runtime.output.__getitem__.assert_called_with((0, slice(3, None)))
        runtime.tokenizer.decode.assert_called_with([7, 8], skip_special_tokens=True)


@pytest.mark.parametrize("message", [None, 3, "", "  "])
def test_empty_or_invalid_message_rejected(runtime, message):
    session = chat.load_model("mlx")
    with pytest.raises(ValueError, match="non-empty"):
        session.reply(message)
    assert session.messages == []
    runtime.tokenizer.apply_chat_template.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, 513, 2.5, True, "32"])
def test_reply_limit_rejected(runtime, limit):
    with pytest.raises(ValueError, match="whole number"):
        chat.load_model("mlx").reply("Hello", max_new_tokens=limit)


def test_context_limit_includes_reply_budget(runtime):
    session = chat.load_model("mlx")
    runtime.tokenizer.apply_chat_template.return_value = [1] * 3968
    session.reply("Hello")  # Exactly 4096 including the 128-token reply budget.
    before = list(session.messages)
    runtime.tokenizer.apply_chat_template.return_value = [1] * 3969
    with pytest.raises(ValueError, match="4096"):
        session.reply("Too long")
    assert session.messages == before


@pytest.mark.parametrize("engine", ["mlx", "torch"])
@pytest.mark.parametrize("failure", [RuntimeError("generation failed"), KeyboardInterrupt()])
def test_failed_or_stopped_reply_does_not_add_partial_turn(runtime, engine, failure):
    session = chat.load_model(engine)
    session.reply("First")
    before = list(session.messages)
    generator = runtime.modules["mlx_lm"].generate if engine == "mlx" else runtime.model.generate
    generator.side_effect = failure
    with pytest.raises(type(failure)):
        session.reply("Second")
    assert session.messages == before


def test_empty_reply_not_saved(runtime):
    session = chat.load_model("mlx")
    runtime.modules["mlx_lm"].generate.return_value = "  "
    with pytest.raises(RuntimeError, match="no text"):
        session.reply("Hello")
    assert session.messages == []


def test_reset_keeps_weights(runtime):
    session = chat.load_model("mlx")
    session.reply("Hello")
    session.reset()
    assert session.messages == []
    assert session._model is runtime.model
    assert session.reply("A new chat") == "Hello Sam!"


@pytest.mark.parametrize("engine,device", [("mlx", "metal"), ("torch", "cpu"), ("torch", "mps"), ("torch", "cuda")])
def test_close_is_repeatable_and_releases_session(runtime, engine, device):
    session = chat.ChatSession(engine, engine, device, runtime.model, runtime.tokenizer)
    session.reply("Hello")
    session.close()
    session.close()
    assert session._model is None and session._tokenizer is None
    assert session.messages == []
    chat.gc.collect.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        session.reply("Hello")
    if engine == "mlx":
        runtime.modules["mlx.core"].clear_cache.assert_called_once()
    elif device == "cpu":
        runtime.torch.cuda.empty_cache.assert_not_called()
        runtime.torch.mps.empty_cache.assert_not_called()
    else:
        getattr(runtime.torch, device).empty_cache.assert_called_once()


def test_chat_notebook_is_clean_and_reuses_core():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/model_chat.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in cells)
    source = "\n".join("".join(cell["source"]) for cell in cells)
    assert "from app_core.chat import load_model" in source
    assert "load_model(MODEL, directory=MODEL_DIRECTORY, backend=BACKEND)" in source
    assert source.count("chat.reply(") == 2
    assert "chat.close()" in source
    assert not any(term in source for term in ("from_pretrained", "mlx_lm", "download_model("))


def test_real_torch_loader_and_generation_without_downloads(tmp_path, monkeypatch):
    """Tiny made-up weights test real APIs, not language quality or a GPU."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    model_class = transformers.Qwen3ForCausalLM
    from huggingface_hub import HfApi
    import huggingface_hub

    def no_network(*args, **kwargs):
        pytest.fail("The local chat check must not access the Hub")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_network)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_network)
    monkeypatch.setattr(HfApi, "model_info", no_network)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    path = tmp_path / "tiny" / ("a" * 40)
    tokenizer_core = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"hello": 0, "[UNK]": 1, "Sam": 2, "user": 3, "assistant": 4}, unk_token="[UNK]",
    ))
    tokenizer_core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer_core, unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}assistant"
    tokenizer.save_pretrained(path)
    config = transformers.Qwen3Config(
        vocab_size=5, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, bos_token_id=None, eos_token_id=None,
        pad_token_id=None, tie_word_embeddings=True,
    )
    # Zero weights make the output predictable: the first vocabulary item.
    model = model_class(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.save_pretrained(path)
    files = []
    for file in path.iterdir():
        if file.is_file():
            import hashlib
            files.append({"name": file.name, "size": file.stat().st_size,
                          "hash": hashlib.sha256(file.read_bytes()).hexdigest(), "hash_type": "sha256"})
    monkeypatch.setattr(models, "CATALOG", {"tiny": {
        "backend": "torch", "repo_id": "local/tiny", "revision": "a" * 40, "files": files,
    }})
    session = chat.load_model("tiny", directory=tmp_path)
    try:
        assert session.device == "cpu"
        assert session.reply("hello", max_new_tokens=3) == "hello hello hello"
        assert session.reply("Sam", max_new_tokens=3) == "hello hello hello"
        assert len(session.messages) == 4
    finally:
        session.close()
