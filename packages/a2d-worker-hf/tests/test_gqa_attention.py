"""GQA+RoPE-family (Gemma/Qwen2/Llama) bidirectionalization: the D13 identity gate
at ``alpha=0`` (bit-identical to base) and the ``alpha=1`` bidirectional-behavior
guard, plus the worker's structural handler dispatch.

Mirrors ``test_identity``/``test_bidir`` (GPT-2) for the RoPE family, whose eager
seam is the 4D mask ``_update_causal_mask`` builds, not GPT-2's ``self.bias``. Also
covers the single-mask windowed flavor (Mistral v0.1), whose sliding window is folded
into that same model-level mask: the shared reveal must open its far-past window
alongside the strictly-future cells, so a converted Mistral is never silently still
windowed at ``alpha=1``. All hermetic: tiny random-weight configs on CPU float32, no
network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from a2d_core.transform.apply import apply_transforms, grow_embeddings, resolve_capabilities
from a2d_core.transform.attention import AnnealState
from a2d_core.transform.gqa_attention import install_gqa_anneal_patch
from a2d_core.transform.identity import IDENTITY_TOLERANCE, check_identity

FAMILIES = ["gemma", "llama", "qwen2"]


def _shift(
    model: Any, state: AnnealState, ids: torch.Tensor, q: int, k: int, alpha: float
) -> float:
    """Max |Δ| of position ``q``'s logits when token ``k`` is perturbed, at ``alpha``."""
    state.alpha = alpha
    perturbed = ids.clone()
    vocab = int(model.config.vocab_size)
    perturbed[:, k] = (perturbed[:, k] + 1) % vocab
    with torch.no_grad():
        base = model(ids).logits[:, q, :]
        after = model(perturbed).logits[:, q, :]
    return float((base - after).abs().max().item())


@pytest.mark.parametrize("family", FAMILIES)
def test_gqa_patched_at_alpha0_is_bit_identical_to_base(
    tiny_gqa: Callable[..., Any], family: str
) -> None:
    """HEADLINE gate (D13): patched@alpha=0 logits (sliced to base_vocab) equal base to
    0.0 for each RoPE family - including the grown mask-token column (Risk 3)."""
    base = tiny_gqa(family, 0)
    patched = tiny_gqa(family, 0)
    patched.load_state_dict(base.state_dict())  # guarantee identical weights
    base_vocab = int(base.config.vocab_size)
    grow_embeddings(patched, base_vocab + 1)  # exercise the grown-column slice (Decision 7)
    grown = patched.get_input_embeddings().weight
    assert torch.allclose(grown[base_vocab], grown[:base_vocab].mean(dim=0))  # mean-init
    state = AnnealState()
    install_gqa_anneal_patch(patched, state)

    probe = torch.randint(0, base_vocab, (2, 8))
    result = check_identity(base, patched, state, probe, base_vocab)

    assert result.passed
    assert result.tolerance == IDENTITY_TOLERANCE
    assert result.max_abs_diff <= IDENTITY_TOLERANCE
    assert result.max_abs_diff == 0.0  # eager + fp32 is exact, not merely within tolerance


@pytest.mark.parametrize("family", FAMILIES)
def test_gqa_future_token_reaches_earlier_positions_only_when_bidirectional(
    tiny_gqa: Callable[..., Any], family: str
) -> None:
    """GUARD (Decision 2): the only proof the patch actually reaches the family's
    causality. Perturbing a strictly-future token must MOVE an earlier position's
    logits at alpha=1 (bidirectional) and must NOT move them at alpha=0 (causal).
    The alpha=0 identity gate passes even for a no-op seam, so this is the real
    regression guard."""
    model = tiny_gqa(family, 0)
    state = AnnealState()
    install_gqa_anneal_patch(model, state)

    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (1, 8))
    perturbed = ids.clone()
    future_pos, earlier_pos = 6, 2
    perturbed[:, future_pos] = (perturbed[:, future_pos] + 1) % vocab

    # alpha=0: causal, so an earlier position cannot see the future token.
    state.alpha = 0.0
    with torch.no_grad():
        causal_earlier = model(ids).logits[:, earlier_pos, :]
        causal_earlier_perturbed = model(perturbed).logits[:, earlier_pos, :]
    assert torch.equal(causal_earlier, causal_earlier_perturbed)

    # alpha=1: bidirectional, so an earlier position attends the perturbed future token.
    state.alpha = 1.0
    with torch.no_grad():
        bidir_earlier = model(ids).logits[:, earlier_pos, :]
        bidir_earlier_perturbed = model(perturbed).logits[:, earlier_pos, :]
    assert not torch.equal(bidir_earlier, bidir_earlier_perturbed)
    assert float((bidir_earlier - bidir_earlier_perturbed).abs().max().item()) > 1e-6


@pytest.mark.parametrize("family", FAMILIES)
def test_gqa_intermediate_alpha_is_monotone_reveal(
    tiny_gqa: Callable[..., Any], family: str
) -> None:
    """Between causal and bidirectional the future token's influence grows with alpha:
    the earlier position's response to a perturbed future token is strictly larger at
    alpha=1 than at a small intermediate alpha (a smooth reveal, not a step)."""
    model = tiny_gqa(family, 0)
    state = AnnealState()
    install_gqa_anneal_patch(model, state)
    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (1, 8))
    perturbed = ids.clone()
    perturbed[:, 6] = (perturbed[:, 6] + 1) % vocab

    def earlier_shift(alpha: float) -> float:
        state.alpha = alpha
        with torch.no_grad():
            a = model(ids).logits[:, 2, :]
            b = model(perturbed).logits[:, 2, :]
        return float((a - b).abs().max().item())

    assert earlier_shift(0.1) < earlier_shift(1.0)


def test_resolve_capabilities_picks_gqa_for_rope_family(tiny_gqa: Callable[..., Any]) -> None:
    """Dispatch: each RoPE family resolves to the attn.gqa handler (its mask seam)."""
    for family in FAMILIES:
        assert resolve_capabilities(tiny_gqa(family, 0)) == ["attn.gqa"]


def test_resolve_capabilities_keeps_gpt2_on_full(tiny_gpt2: Callable[..., Any]) -> None:
    """GPT-2 must keep selecting its own attn.full self.bias seam, unchanged."""
    assert resolve_capabilities(tiny_gpt2(0)) == ["attn.full"]


@pytest.mark.parametrize("family", FAMILIES)
def test_apply_transforms_via_resolved_caps_installs_gqa_patch(
    tiny_gqa: Callable[..., Any], family: str
) -> None:
    """The exact worker dispatch path: resolve_capabilities -> apply_transforms installs
    the mask-seam patch, so alpha=1 attention is bidirectional through the registry."""
    model = tiny_gqa(family, 0)
    state = AnnealState()
    apply_transforms(model, resolve_capabilities(model), state)
    assert model.config.use_cache is False  # patch forces cache off (ARCHITECTURE.md §7)

    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (1, 8))
    perturbed = ids.clone()
    perturbed[:, 6] = (perturbed[:, 6] + 1) % vocab
    state.alpha = 1.0
    with torch.no_grad():
        base_earlier = model(ids).logits[:, 2, :]
        pert_earlier = model(perturbed).logits[:, 2, :]
    assert not torch.equal(base_earlier, pert_earlier)


@pytest.mark.parametrize("family", FAMILIES)
def test_gqa_padding_stays_masked_and_identity_holds_with_padding(
    tiny_gqa: Callable[..., Any], family: str
) -> None:
    """Padding preservation (module contract): with a right-padded 2D attention_mask
    the padded keys are strictly-future for the real queries, so alpha=1 must NOT
    reveal them (perturbing a padded token leaves every real position's logits
    untouched) while a perturbed real future token still reaches earlier positions;
    and at alpha=0 the padded forward stays bit-identical to base."""
    base = tiny_gqa(family, 0)
    patched = tiny_gqa(family, 0)
    patched.load_state_dict(base.state_dict())
    state = AnnealState()
    install_gqa_anneal_patch(patched, state)

    vocab = int(base.config.vocab_size)
    ids = torch.randint(0, vocab, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    attention_mask[0, 5:] = 0  # row 0 right-padded: positions 5..7 are padding

    state.alpha = 0.0
    with torch.no_grad():
        base_logits = base(input_ids=ids, attention_mask=attention_mask).logits
        patched_logits = patched(input_ids=ids, attention_mask=attention_mask).logits
    assert torch.equal(base_logits, patched_logits)

    state.alpha = 1.0
    pad_perturbed = ids.clone()
    pad_perturbed[0, 6] = (pad_perturbed[0, 6] + 1) % vocab
    with torch.no_grad():
        real = patched(input_ids=ids, attention_mask=attention_mask).logits[0, :5, :]
        real_after_pad = patched(input_ids=pad_perturbed, attention_mask=attention_mask).logits[
            0, :5, :
        ]
    assert torch.equal(real, real_after_pad)

    # Guard against a vacuous pass: a real future token (position 4) must still be
    # revealed to earlier real positions at alpha=1 under the same padded mask.
    future_perturbed = ids.clone()
    future_perturbed[0, 4] = (future_perturbed[0, 4] + 1) % vocab
    with torch.no_grad():
        real_after_future = patched(
            input_ids=future_perturbed, attention_mask=attention_mask
        ).logits[0, :4, :]
    assert not torch.equal(real[:4, :], real_after_future)


@pytest.mark.parametrize("family", FAMILIES)
def test_gqa_reinstall_with_fresh_state_takes_effect(
    tiny_gqa: Callable[..., Any], family: str
) -> None:
    """Re-installing on an already-patched model must swap in the NEW state (parity
    with the GPT-2 seam's per-install re-tag) without double-wrapping: after a second
    install at alpha=1, an earlier position sees a perturbed future token."""
    model = tiny_gqa(family, 0)
    install_gqa_anneal_patch(model, AnnealState(alpha=0.0))
    install_gqa_anneal_patch(model, AnnealState(alpha=1.0))

    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (1, 8))
    perturbed = ids.clone()
    perturbed[:, 6] = (perturbed[:, 6] + 1) % vocab
    with torch.no_grad():
        a = model(ids).logits[:, 2, :]
        b = model(perturbed).logits[:, 2, :]
    assert not torch.equal(a, b)


@pytest.mark.parametrize("family", FAMILIES)
def test_gqa_patch_is_idempotent(tiny_gqa: Callable[..., Any], family: str) -> None:
    """Installing twice must not double-anneal: a second install is a no-op, so the
    alpha=0 causal behavior is preserved (an earlier position ignores a future token)."""
    model = tiny_gqa(family, 0)
    state = AnnealState()
    install_gqa_anneal_patch(model, state)
    install_gqa_anneal_patch(model, state)  # second call must be inert

    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (1, 8))
    perturbed = ids.clone()
    perturbed[:, 6] = (perturbed[:, 6] + 1) % vocab
    state.alpha = 0.0
    with torch.no_grad():
        a = model(ids).logits[:, 2, :]
        b = model(perturbed).logits[:, 2, :]
    assert torch.equal(a, b)


def test_mistral_windowed_patched_at_alpha0_is_bit_identical_to_base(
    tiny_mistral: Callable[..., Any],
) -> None:
    """D13 gate for the single-mask windowed flavor: with an ACTIVE sliding window,
    patched@alpha=0 logits equal base to 0.0 (the generalized reveal must leave every
    base-masked cell - future AND far-past window - at exactly finfo.min)."""
    base = tiny_mistral()
    patched = tiny_mistral()
    patched.load_state_dict(base.state_dict())  # guarantee identical weights
    # Sanity: no per-layer sliding seam, so dispatch stays on the shared mask seam.
    assert not any(getattr(m, "is_sliding", False) for m in patched.modules())
    assert resolve_capabilities(patched) == ["attn.gqa"]
    state = AnnealState()
    install_gqa_anneal_patch(patched, state)

    base_vocab = int(base.config.vocab_size)
    probe = torch.randint(0, base_vocab, (2, 8))
    result = check_identity(base, patched, state, probe, base_vocab)

    assert result.passed
    assert result.max_abs_diff == 0.0  # eager + fp32 is exact, not merely within tolerance


def test_mistral_window_and_future_open_at_alpha1_not_alpha0(
    tiny_mistral: Callable[..., Any],
) -> None:
    """GUARD (review: swa-mistral-silent-window): a converted Mistral must not stay
    silently windowed. With window 2, query 3 sees only keys {2, 3} at alpha=0: an
    in-window key moves its logits even at alpha=0 (non-vacuity), while a far-past
    out-of-window key AND a strictly-future key move them at alpha=1 only."""
    model = tiny_mistral(sliding_window=2)
    state = AnnealState()
    install_gqa_anneal_patch(model, state)
    ids = torch.randint(0, int(model.config.vocab_size), (1, 8))

    # In-window past token (k=2) reaches q=3 already at alpha=0: guards against vacuity.
    assert _shift(model, state, ids, q=3, k=2, alpha=0.0) > 1e-6

    # Far-past token outside the window (k=0): windowed out at alpha=0, revealed at alpha=1.
    assert _shift(model, state, ids, q=3, k=0, alpha=0.0) == 0.0
    assert _shift(model, state, ids, q=3, k=0, alpha=1.0) > 1e-6

    # Strictly-future token (k=6): causally masked at alpha=0, revealed at alpha=1.
    assert _shift(model, state, ids, q=3, k=6, alpha=0.0) == 0.0
    assert _shift(model, state, ids, q=3, k=6, alpha=1.0) > 1e-6


def test_mistral_padding_stays_masked_at_alpha1(tiny_mistral: Callable[..., Any]) -> None:
    """The generalized reveal opens ONLY real keys: with a right-padded 2D mask a
    perturbed padded token leaves every real position's logits untouched at alpha=1,
    while a real out-of-window token still reaches the query."""
    model = tiny_mistral(sliding_window=2)
    state = AnnealState()
    install_gqa_anneal_patch(model, state)

    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    attention_mask[0, 5:] = 0  # row 0 right-padded: positions 5..7 are padding

    state.alpha = 1.0
    pad_perturbed = ids.clone()
    pad_perturbed[0, 6] = (pad_perturbed[0, 6] + 1) % vocab
    with torch.no_grad():
        real = model(input_ids=ids, attention_mask=attention_mask).logits[0, :5, :]
        real_after_pad = model(input_ids=pad_perturbed, attention_mask=attention_mask).logits[
            0, :5, :
        ]
    assert torch.equal(real, real_after_pad)

    # Guard against a vacuous pass: a real out-of-window token (position 0) must still
    # be revealed to a later real position at alpha=1 under the same padded mask.
    window_perturbed = ids.clone()
    window_perturbed[0, 0] = (window_perturbed[0, 0] + 1) % vocab
    with torch.no_grad():
        real_after = model(input_ids=window_perturbed, attention_mask=attention_mask).logits[
            0, 4, :
        ]
    assert not torch.equal(real[4, :], real_after)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
