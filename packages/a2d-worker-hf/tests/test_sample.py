"""Step 10: the ``a2d-sample`` entry denoises a saved model and prints text.

Builds a tiny GPT-2 + a network-free WordLevel tokenizer (with a grown mask token),
saves them, then drives ``sample_main.main`` as a subprocess over the stdin-JSON
``SampleRequest`` contract, exactly as the CLI would.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest


def _save_tiny_model(dir_path: Path) -> None:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, GPT2Config, PreTrainedTokenizerFast

    words = ["the", "history", "of", "world", "hello", "a", "b", "c", "d", "e"]
    vocab = {"<eos>": 0}
    for i, word in enumerate(words):
        vocab[word] = i + 1
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<eos>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
    fast.add_special_tokens({"mask_token": "<|mdlm_mask|>"})  # grown mask token (Decision 7)
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


def test_sample_main_prints_text(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _save_tiny_model(model_dir)

    request = json.dumps(
        {
            "schema_version": version("a2d-contracts"),
            "model_dir": str(model_dir),
            "prompt": "the history of",
            "canvas_len": 10,
            "num_steps": 5,
            "temperature": 1.0,
            "seed": 0,
            "device": "cpu",
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "a2d_core.sample_main"],
        input=request,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip(), "a2d-sample printed no text"


def test_sample_main_rejects_garbage() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "a2d_core.sample_main"],
        input="not json",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stdout.strip() == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
