"""Step 3: the MDLM likelihood bound and the best-effort AR baseline."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from a2d_core.eval.likelihood import ar_baseline, mdlm_bound
from conftest import ConvertSetup


def _converted(setup: ConvertSetup) -> tuple[Any, Any, int]:
    from a2d_core.transform.apply import load_model, resolve_mask_token
    from a2d_core.transform.attention import AnnealState, install_anneal_patch

    model, tokenizer = load_model(str(setup.model_src))
    mask_id = resolve_mask_token(model, tokenizer, "grow")
    install_anneal_patch(model, AnnealState(alpha=1.0))  # bound is on the bidirectional model
    return model, tokenizer, mask_id


def _bound(setup: ConvertSetup, mc_samples: int) -> Any:
    model, tokenizer, mask_id = _converted(setup)
    return mdlm_bound(
        model,
        tokenizer,
        mask_id,
        data_path=str(setup.corpus),
        seq_len=8,
        mc_samples=mc_samples,
        max_eval_tokens=256,
        seed=0,
        device="cpu",
    )


def test_mdlm_bound_sub_batch_matches_single_batch(convert_setup: ConvertSetup) -> None:
    """Splitting the forward into sub-batches must not change the bound (identical up to fp
    round-off): the corruption RNG runs over the full chunk set and rows don't attend to each
    other."""
    model, tokenizer, mask_id = _converted(convert_setup)
    kw: dict[str, Any] = dict(
        data_path=str(convert_setup.corpus),
        seq_len=8,
        mc_samples=3,
        max_eval_tokens=256,  # 32 chunks -> exercises real splitting
        seed=0,
        device="cpu",
    )
    single = mdlm_bound(model, tokenizer, mask_id, eval_batch_size=0, **kw)  # one big forward
    subbed = mdlm_bound(model, tokenizer, mask_id, eval_batch_size=4, **kw)  # 8 sub-batches
    tiny = mdlm_bound(model, tokenizer, mask_id, eval_batch_size=1, **kw)  # one seq per forward
    for other in (subbed, tiny):
        assert math.isclose(
            single.nats_per_token, other.nats_per_token, rel_tol=1e-9, abs_tol=1e-12
        )
        assert math.isclose(
            single.bits_per_token, other.bits_per_token, rel_tol=1e-9, abs_tol=1e-12
        )
        assert math.isclose(single.std_error, other.std_error, rel_tol=1e-9, abs_tol=1e-12)
    assert single.n_tokens == subbed.n_tokens == tiny.n_tokens
    assert single.mc_samples == subbed.mc_samples == tiny.mc_samples


def test_mdlm_bound_forward_is_sub_batched(convert_setup: ConvertSetup) -> None:
    """A large token budget must not run one monolithic forward: peak batch dim is capped by
    ``eval_batch_size`` and the model is called once per sub-batch per MC sample."""
    model, tokenizer, mask_id = _converted(convert_setup)
    batch_dims: list[int] = []
    orig_forward = model.forward

    def spy(*args: Any, **kwargs: Any) -> Any:
        batch_dims.append(int(kwargs["input_ids"].size(0)))
        return orig_forward(*args, **kwargs)

    model.forward = spy
    result = mdlm_bound(
        model,
        tokenizer,
        mask_id,
        data_path=str(convert_setup.corpus),
        seq_len=8,
        mc_samples=2,
        max_eval_tokens=256,  # 32 chunks; a single forward here is what OOMs at real scale
        seed=0,
        device="cpu",
        eval_batch_size=4,
    )
    assert result.n_tokens == 32 * 8  # all chunks still scored
    assert max(batch_dims) <= 4  # never a giant forward
    # 32 chunks / 4 per sub-batch = 8 sub-batches, x2 MC samples = 16 forwards (not 2).
    assert len(batch_dims) == 16


def test_mdlm_bound_finite_positive_deterministic(convert_setup: ConvertSetup) -> None:
    a = _bound(convert_setup, 4)
    b = _bound(convert_setup, 4)
    assert math.isfinite(a.nats_per_token) and a.nats_per_token > 0
    assert math.isclose(a.bits_per_token, a.nats_per_token / math.log(2))
    assert a.n_tokens > 0 and a.mc_samples == 4
    assert a.nats_per_token == b.nats_per_token  # seeded => deterministic


def test_std_error_shrinks_with_more_mc_samples(convert_setup: ConvertSetup) -> None:
    few = _bound(convert_setup, 1)
    many = _bound(convert_setup, 16)
    assert many.std_error < few.std_error  # more Monte-Carlo draws => tighter estimate


def test_ar_baseline_unavailable_without_source() -> None:
    result = ar_baseline(
        None, None, data_path="unused.jsonl", seq_len=8, max_eval_tokens=64, device="cpu"
    )
    assert result.available is False
    assert result.reason and result.perplexity is None and result.nats_per_token is None


def test_ar_baseline_available_for_real_source(convert_setup: ConvertSetup) -> None:
    weights = convert_setup.model_src / "model.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    result = ar_baseline(
        str(convert_setup.model_src),
        digest,
        data_path=str(convert_setup.corpus),
        seq_len=8,
        max_eval_tokens=256,
        device="cpu",
    )
    assert result.available is True and result.reason is None
    assert (
        result.perplexity is not None
        and math.isfinite(result.perplexity)
        and result.perplexity > 1.0
    )


def test_ar_baseline_unavailable_on_hash_mismatch(convert_setup: ConvertSetup) -> None:
    result = ar_baseline(
        str(convert_setup.model_src),
        "0" * 64,
        data_path=str(convert_setup.corpus),
        seq_len=8,
        max_eval_tokens=256,
        device="cpu",
    )
    assert result.available is False and "hash mismatch" in (result.reason or "")
