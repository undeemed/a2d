from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tiny_gpt2() -> Callable[..., Any]:
    """Factory for a seeded tiny GPT-2 (eager, eval, no download). Same seed =>
    bit-identical weights, so two calls give an aligned base/patched pair."""

    def _make(seed: int = 0) -> Any:
        import torch
        from transformers import AutoModelForCausalLM, GPT2Config

        torch.manual_seed(seed)
        config = GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=16,
            n_layer=2,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        return AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()

    return _make


@pytest.fixture
def tiny_gqa() -> Callable[..., Any]:
    """Factory for a seeded tiny RoPE/GQA-family model (eager, eval, no download) for
    ``family in {"gemma", "llama", "qwen2"}``. Same ``(family, seed)`` => bit-identical
    weights, so two calls give an aligned base/patched pair. Gemma exercises the
    quirks that matter here: MQA (``num_key_value_heads=1``) and an independent
    ``head_dim`` (decoupled from ``hidden_size // num_attention_heads``); llama/qwen2
    exercise GQA (``num_key_value_heads=2``). All are full attention (no SWA)."""

    def _make(family: str = "gemma", seed: int = 0) -> Any:
        import torch
        from transformers import AutoModelForCausalLM

        torch.manual_seed(seed)
        common = {
            "vocab_size": 48,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
        }
        if family == "gemma":
            from transformers import GemmaConfig

            config: Any = GemmaConfig(
                num_key_value_heads=1, head_dim=8, hidden_act="gelu_pytorch_tanh", **common
            )
        elif family == "llama":
            from transformers import LlamaConfig

            config = LlamaConfig(num_key_value_heads=2, **common)
        elif family == "qwen2":
            from transformers import Qwen2Config

            config = Qwen2Config(num_key_value_heads=2, **common)
        else:
            raise ValueError(f"unknown gqa family {family!r}")
        return AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()

    return _make


@pytest.fixture
def tiny_gemma3() -> Callable[..., Any]:
    """Factory for a seeded tiny Gemma 3 (``gemma3_text``, eager, eval, no download).

    Same ``(kwargs, seed)`` => bit-identical weights, so two calls give an aligned
    base/patched pair. Exercises the Gemma 3 sliding-window seam: local (sliding) vs
    global (full) layers chosen by ``sliding_window_pattern``, plus the 270M-class
    quirks - MQA (``num_key_value_heads=1``), an independent ``head_dim``, query-key
    norm, and per-layer local/global RoPE theta. Defaults (4 layers, pattern 2) put
    both a local layer (0, 2) and a global layer (1, 3) in the stack; pass
    ``num_hidden_layers=1, sliding_window_pattern=6`` for an all-sliding model whose
    single-layer receptive field is exactly the window."""

    def _make(
        seed: int = 0,
        num_hidden_layers: int = 4,
        sliding_window: int = 2,
        sliding_window_pattern: int = 2,
    ) -> Any:
        import torch
        from transformers import AutoModelForCausalLM, Gemma3TextConfig

        torch.manual_seed(seed)
        config = Gemma3TextConfig(
            vocab_size=48,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rope_local_base_freq=10000.0,
            sliding_window=sliding_window,
            sliding_window_pattern=sliding_window_pattern,
            query_pre_attn_scalar=8,
        )
        return AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()

    return _make


@dataclass
class ConvertSetup:
    """A saved tiny model dir + corpus + a ConversionJob-JSON builder for the
    end-to-end worker pipeline (test_worker, test_smoke_convert)."""

    model_src: Path
    corpus: Path
    run_dir: Path
    build_job: Callable[..., str]


def _save_tiny_convertible_model(dir_path: Path) -> None:
    """Save a seeded tiny GPT-2 + a matching 64-token tokenizer (NO mask token, so the
    worker's grow step actually resizes 64->65) to ``dir_path``. Network-free."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast

    # 64-token vocab so tokenizer len == model vocab_size; grow appends one row (id 64).
    vocab = {"<eos>": 0}
    for i in range(1, 64):
        vocab[f"w{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<eos>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
    fast.save_pretrained(str(dir_path))

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval().save_pretrained(
        str(dir_path)
    )


def _save_tiny_convertible_gemma(dir_path: Path) -> None:
    """Save a seeded tiny Gemma 1 + a matching 64-token tokenizer (NO mask token, so the
    worker's grow step resizes the tied embeddings 64->65) to ``dir_path``. Exercises the
    RoPE/GQA seam end to end: MQA (num_key_value_heads=1) and an independent head_dim.
    Network-free."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, GemmaConfig, PreTrainedTokenizerFast

    vocab = {"<eos>": 0}
    for i in range(1, 64):
        vocab[f"w{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<eos>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
    fast.save_pretrained(str(dir_path))

    torch.manual_seed(0)
    config = GemmaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_act="gelu_pytorch_tanh",
    )
    AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval().save_pretrained(
        str(dir_path)
    )


def _save_tiny_convertible_gemma3(dir_path: Path) -> None:
    """Save a seeded tiny Gemma 3 (``gemma3_text``) + a matching 64-token tokenizer (NO
    mask token, so the worker's grow step resizes the tied embeddings 64->65) to
    ``dir_path``. Exercises the sliding-window seam end to end: 4 layers with
    ``sliding_window_pattern=2`` put both a local (sliding) and a global (full) layer in
    the stack, and ``sliding_window=4`` (< the job's seq_len 8) means the window is
    actually active. Network-free."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

    vocab = {"<eos>": 0}
    for i in range(1, 64):
        vocab[f"w{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<eos>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
    fast.save_pretrained(str(dir_path))

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_local_base_freq=10000.0,
        sliding_window=4,
        sliding_window_pattern=2,
        query_pre_attn_scalar=8,
    )
    AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval().save_pretrained(
        str(dir_path)
    )


def _build_convert_setup(tmp_path: Path, saver: Callable[[Path], None]) -> ConvertSetup:
    """Shared ConvertSetup: save a tiny model via ``saver``, write a matching corpus, and
    return a ConversionJob-JSON builder. The corpus/job knobs are model-agnostic, so GPT-2
    and Gemma share them verbatim (only the saved architecture differs)."""
    model_src = tmp_path / "src"
    saver(model_src)

    corpus = tmp_path / "corpus.jsonl"
    line = json.dumps({"text": " ".join(f"w{(i % 63) + 1}" for i in range(40))})
    corpus.write_text("\n".join(line for _ in range(8)), encoding="utf-8")

    run_dir = tmp_path / "run"

    def build_job(**overrides: object) -> str:
        cfg: dict[str, object] = {
            "objective": "mdlm",
            "data": str(corpus),
            "anneal_steps": 2,
            "anneal_schedule": "linear",
            "seq_len": 8,
            "per_device_batch_size": 2,
            "grad_accum": 1,
            "lr": 1e-3,
            "max_steps": 2,
            "mask_token": "grow",
            "keep_last": 3,
            "seed": 0,
            "device": "cpu",
            "dtype": "float32",
        }
        cfg.update(overrides)
        return json.dumps(
            {
                "schema_version": version("a2d-contracts"),
                "job_id": str(uuid.uuid4()),
                "model_path": str(model_src),
                "run_dir": str(run_dir),
                "conversion_config": cfg,
            }
        )

    return ConvertSetup(model_src=model_src, corpus=corpus, run_dir=run_dir, build_job=build_job)


@pytest.fixture
def convert_setup(tmp_path: Path) -> ConvertSetup:
    return _build_convert_setup(tmp_path, _save_tiny_convertible_model)


@pytest.fixture
def gemma_convert_setup(tmp_path: Path) -> ConvertSetup:
    return _build_convert_setup(tmp_path, _save_tiny_convertible_gemma)


@pytest.fixture
def gemma3_convert_setup(tmp_path: Path) -> ConvertSetup:
    return _build_convert_setup(tmp_path, _save_tiny_convertible_gemma3)
