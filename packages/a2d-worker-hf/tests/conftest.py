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


@pytest.fixture
def convert_setup(tmp_path: Path) -> ConvertSetup:
    model_src = tmp_path / "src"
    _save_tiny_convertible_model(model_src)

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
