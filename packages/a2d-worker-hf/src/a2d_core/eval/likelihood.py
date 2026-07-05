"""The MDLM likelihood bound (Decision 2) and the best-effort AR baseline (Decision 6).

The bound reuses ``objectives.mdlm.MDLM.corrupt`` for the t-schedule + masking (the
objective-defining part) and only aggregates the weighted NLL as a SUM per sequence
(the bound is a per-sequence NLL, not the per-masked-token mean the training loss uses).
The AR baseline loads the SOURCE model + tokenizer and reports teacher-forced perplexity,
degrading to ``available=false`` with a reason when the source is gone or its hash drifts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import torch
from a2d_contracts import ArBaseline, LikelihoodBound

from a2d_core.data import DATA


def _eval_chunks(
    data_path: str, tokenizer: Any, seq_len: int, max_eval_tokens: int
) -> torch.Tensor:
    """First ``max_eval_tokens // seq_len`` fixed-length chunks as one [n, seq_len] tensor."""
    fmt = Path(data_path).suffix.lower().lstrip(".")
    dataset = DATA.get(fmt)(data_path, tokenizer, seq_len)
    n_chunks = max(1, min(len(cast(Sized, dataset)), max_eval_tokens // seq_len))
    return torch.stack([dataset[i]["input_ids"] for i in range(n_chunks)])


def mdlm_bound(
    model: Any,
    tokenizer: Any,
    mask_token_id: int,
    *,
    data_path: str,
    seq_len: int,
    mc_samples: int,
    max_eval_tokens: int,
    seed: int,
    device: str,
) -> LikelihoodBound:
    """MDLM NLL upper bound in nats/token, Monte-Carlo over the diffusion time t.

    The model must be bidirectional (alpha=1); the caller installs the anneal patch.
    """
    from a2d_core.objectives.mdlm import MDLM

    dev = torch.device(device)
    model.to(dev).eval()
    chunks = _eval_chunks(data_path, tokenizer, seq_len, max_eval_tokens).to(dev)
    n_chunks = int(chunks.size(0))

    mdlm = MDLM(mask_token_id, seed=seed)
    batch = [{"input_ids": chunks[i]} for i in range(n_chunks)]
    per_seq_nats: list[float] = []
    for _ in range(max(1, mc_samples)):
        corrupted = mdlm.corrupt(batch)  # per-sequence t, Bernoulli(t) masking, 1/t weights
        noisy = corrupted["input_ids"].to(dev)
        clean = corrupted["clean"].to(dev)
        weight = corrupted["mask"].to(dev)  # (1/t) at masked positions, 0 elsewhere
        with torch.no_grad():
            logits = model(input_ids=noisy).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(-1, clean.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        # sum_masked (1/t)*nll per sequence = the MDLM sequence NLL estimate; /seq_len => per token.
        seq_nats = (weight * nll).sum(dim=-1) / seq_len
        per_seq_nats.extend(seq_nats.tolist())

    values = torch.tensor(per_seq_nats, dtype=torch.float64)
    nats = float(values.mean())
    # MC standard error over all (chunk, draw) estimates.
    std_error = (
        float(values.std(unbiased=True) / math.sqrt(values.numel())) if values.numel() > 1 else 0.0
    )
    return LikelihoodBound(
        nats_per_token=nats,
        bits_per_token=nats / math.log(2),
        std_error=std_error,
        mc_samples=mc_samples,
        n_tokens=n_chunks * seq_len,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_reason(source_model: str | None, source_hash: str | None) -> str | None:
    """Why the SOURCE AR model can't be trusted, or None if it's present and hash-matched.

    Shared by the AR baseline and the AR throughput measurement so the hash policy lives
    in one place (Decision 6). Primary-file only, matching a2d-run's source_hash.
    """
    if source_model is None:
        return "no source model recorded"
    weights = Path(source_model) / "model.safetensors"
    if not weights.is_file():
        # ponytail: primary-file only; extend to shard manifests if sharded sources
        # ever need an AR baseline.
        return f"source weights not found at {weights}"
    if source_hash is not None and _sha256(weights) != source_hash:
        return "source weights hash mismatch (source changed since conversion)"
    return None


def ar_baseline(
    source_model: str | None,
    source_hash: str | None,
    *,
    data_path: str,
    seq_len: int,
    max_eval_tokens: int,
    device: str,
) -> ArBaseline:
    """Teacher-forced perplexity of the SOURCE AR model; best-effort (Decision 6).

    Returns ``available=false`` with a reason when the source is absent or its primary
    safetensors hash no longer matches the manifest, rather than failing the whole eval.
    """
    reason = source_reason(source_model, source_hash)
    if reason is not None:
        return ArBaseline(available=False, reason=reason, perplexity=None, nats_per_token=None)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(device)
    model = (
        AutoModelForCausalLM.from_pretrained(source_model, attn_implementation="eager")
        .to(dev)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(source_model)
    chunks = _eval_chunks(data_path, tokenizer, seq_len, max_eval_tokens).to(dev)

    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(int(chunks.size(0))):
            ids = chunks[i : i + 1]
            # HF causal LM: labels=input_ids gives the shifted mean next-token CE.
            out = model(input_ids=ids, labels=ids)
            n = int(ids.size(1)) - 1  # shifted => seq_len-1 scored positions
            total_nll += float(out.loss) * n
            total_tokens += n

    nats = total_nll / total_tokens if total_tokens else float("nan")
    return ArBaseline(
        available=True,
        reason=None,
        perplexity=math.exp(nats),
        nats_per_token=nats,
    )
