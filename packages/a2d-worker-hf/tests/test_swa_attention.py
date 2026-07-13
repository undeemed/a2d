"""Gemma 2/3 sliding-window (``attn.swa``) bidirectionalization: the D13 identity gate
at ``alpha=0`` (bit-identical to base, with BOTH a local and a global layer present)
and the ``alpha=1`` behavior guard - proving BOTH the future-reveal (causality opens)
and the window-open (the sliding window's far-past opens) - plus the worker's
structural handler dispatch.

The Gemma 2/3 eager seam is the single 4D ``_update_causal_mask`` (like Gemma 1) PLUS
a per-layer far-past window re-mask on the local (sliding) decoder layers. The window
test uses an all-sliding single-layer model so a query's receptive field is exactly
its window; the identity/future tests use a mixed 4-layer stack (local layers 0, 2 and
global layers 1, 3). Gemma 2 - whose decoder layer takes a single
``position_embeddings`` pair where Gemma 3 takes a global/local pair - guards the
wrapper's signature-agnostic re-bind. All hermetic: tiny random-weight configs on CPU
float32, no network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from a2d_core.transform.apply import apply_transforms, grow_embeddings, resolve_capabilities
from a2d_core.transform.attention import AnnealState
from a2d_core.transform.identity import IDENTITY_TOLERANCE, check_identity
from a2d_core.transform.swa_attention import install_swa_anneal_patch


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


def test_swa_patched_at_alpha0_is_bit_identical_to_base(tiny_gemma3: Callable[..., Any]) -> None:
    """HEADLINE gate (D13): patched@alpha=0 logits (sliced to base_vocab) equal base to
    0.0, with BOTH a local (sliding) and a global (full) layer exercised - including the
    grown mask-token column (Risk 3)."""
    base = tiny_gemma3()
    patched = tiny_gemma3()
    patched.load_state_dict(base.state_dict())  # guarantee identical weights
    # Sanity: the default stack really does mix local and global layers.
    kinds = [layer.is_sliding for layer in patched.model.layers]
    assert any(kinds) and not all(kinds), f"need both local and global layers, got {kinds}"

    base_vocab = int(base.config.vocab_size)
    grow_embeddings(patched, base_vocab + 1)  # exercise the grown-column slice (Decision 7)
    grown = patched.get_input_embeddings().weight
    assert torch.allclose(grown[base_vocab], grown[:base_vocab].mean(dim=0))  # mean-init
    state = AnnealState()
    install_swa_anneal_patch(patched, state)

    probe = torch.randint(0, base_vocab, (2, 8))
    result = check_identity(base, patched, state, probe, base_vocab)

    assert result.passed
    assert result.tolerance == IDENTITY_TOLERANCE
    assert result.max_abs_diff <= IDENTITY_TOLERANCE
    assert result.max_abs_diff == 0.0  # eager + fp32 is exact, not merely within tolerance


def test_swa_reveals_future_at_alpha1_not_alpha0(tiny_gemma3: Callable[..., Any]) -> None:
    """Future-reveal on the mixed stack: perturbing a strictly-future token MOVES an
    earlier position's logits at alpha=1 (bidirectional) and must NOT at alpha=0 (causal),
    across both the local and global layers."""
    model = tiny_gemma3()
    state = AnnealState()
    install_swa_anneal_patch(model, state)
    ids = torch.randint(0, int(model.config.vocab_size), (1, 8))

    assert _shift(model, state, ids, q=2, k=6, alpha=0.0) == 0.0
    assert _shift(model, state, ids, q=2, k=6, alpha=1.0) > 1e-6


def test_swa_opens_window_at_alpha1_not_alpha0(tiny_gemma3: Callable[..., Any]) -> None:
    """Window-open (the load-bearing SWA guard): on an all-sliding single-layer model a
    query attends only its window at alpha=0. Perturbing a token OUTSIDE that window -
    both a strictly-future token AND a far-past token - must MOVE the query's logits at
    alpha=1 but NOT at alpha=0, while an IN-window token moves them even at alpha=0 (so
    the test is not vacuous)."""
    # 1 sliding layer (pattern 6 => layer 0 local), window 2: query 3 sees only cols {2, 3}.
    model = tiny_gemma3(num_hidden_layers=1, sliding_window=2, sliding_window_pattern=6)
    assert all(layer.is_sliding for layer in model.model.layers)
    state = AnnealState()
    install_swa_anneal_patch(model, state)
    ids = torch.randint(0, int(model.config.vocab_size), (1, 8))

    # In-window past token (k=2) reaches q=3 already at alpha=0: guards against vacuity.
    assert _shift(model, state, ids, q=3, k=2, alpha=0.0) > 1e-6

    # Far-past token outside the window (k=0): windowed out at alpha=0, revealed at alpha=1.
    assert _shift(model, state, ids, q=3, k=0, alpha=0.0) == 0.0
    assert _shift(model, state, ids, q=3, k=0, alpha=1.0) > 1e-6

    # Strictly-future token (k=6): causally masked at alpha=0, revealed at alpha=1.
    assert _shift(model, state, ids, q=3, k=6, alpha=0.0) == 0.0
    assert _shift(model, state, ids, q=3, k=6, alpha=1.0) > 1e-6


def test_swa_intermediate_alpha_is_monotone_reveal(tiny_gemma3: Callable[..., Any]) -> None:
    """Between causal+windowed and fully open the far-past token's influence grows with
    alpha: q's response to a perturbed out-of-window token is strictly larger at alpha=1
    than at a small intermediate alpha (a smooth reveal, not a step)."""
    model = tiny_gemma3(num_hidden_layers=1, sliding_window=2, sliding_window_pattern=6)
    state = AnnealState()
    install_swa_anneal_patch(model, state)
    ids = torch.randint(0, int(model.config.vocab_size), (1, 8))
    small = _shift(model, state, ids, q=3, k=0, alpha=0.1)
    full = _shift(model, state, ids, q=3, k=0, alpha=1.0)
    assert small < full


def test_resolve_capabilities_routes_gemma3_to_swa(
    tiny_gemma3: Callable[..., Any],
    tiny_gqa: Callable[..., Any],
    tiny_gpt2: Callable[..., Any],
) -> None:
    """Dispatch: a sliding-window Gemma 3 resolves to attn.swa (checked before the shared
    attn.gqa mask seam it also owns); the full-attention RoPE family and GPT-2 are
    unchanged."""
    assert resolve_capabilities(tiny_gemma3()) == ["attn.swa"]
    for family in ("gemma", "llama", "qwen2"):
        assert resolve_capabilities(tiny_gqa(family, 0)) == ["attn.gqa"]
    assert resolve_capabilities(tiny_gpt2(0)) == ["attn.full"]


def test_apply_transforms_via_resolved_caps_installs_swa_patch(
    tiny_gemma3: Callable[..., Any],
) -> None:
    """The exact worker/sample/eval dispatch path: resolve_capabilities -> apply_transforms
    installs the SWA patch, so alpha=1 attention opens both future and window through the
    registry."""
    model = tiny_gemma3(num_hidden_layers=1, sliding_window=2, sliding_window_pattern=6)
    state = AnnealState()
    apply_transforms(model, resolve_capabilities(model), state)
    assert model.config.use_cache is False  # patch forces cache off (ARCHITECTURE.md §7)

    ids = torch.randint(0, int(model.config.vocab_size), (1, 8))
    assert _shift(model, state, ids, q=3, k=0, alpha=1.0) > 1e-6


def test_swa_padding_stays_masked_and_identity_holds_with_padding(
    tiny_gemma3: Callable[..., Any],
) -> None:
    """Padding preservation: with a right-padded 2D attention_mask the padded keys must
    stay masked at alpha=1 (perturbing a padded token leaves every real position's logits
    untouched) while a perturbed real out-of-window token still reaches the query; and at
    alpha=0 the padded forward is bit-identical to base."""
    base = tiny_gemma3(num_hidden_layers=1, sliding_window=2, sliding_window_pattern=6)
    patched = tiny_gemma3(num_hidden_layers=1, sliding_window=2, sliding_window_pattern=6)
    patched.load_state_dict(base.state_dict())
    state = AnnealState()
    install_swa_anneal_patch(patched, state)

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

    # Guard against a vacuous pass: a real out-of-window token (position 0) must still be
    # revealed to a later real position at alpha=1 under the same padded mask.
    future_perturbed = ids.clone()
    future_perturbed[0, 0] = (future_perturbed[0, 0] + 1) % vocab
    with torch.no_grad():
        real_after = patched(input_ids=future_perturbed, attention_mask=attention_mask).logits[
            0, 4, :
        ]
    assert not torch.equal(real[4, :], real_after)


def test_swa_gemma2_patched_at_alpha0_is_bit_identical_to_base(
    tiny_gemma2: Callable[..., Any],
) -> None:
    """Regression (review: swa-gemma2-signature-crash): Gemma 2's decoder layer takes a
    single ``position_embeddings`` pair, so the wrapper must survive its keyword call
    path without crashing, and patched@alpha=0 logits must equal base to 0.0 with both
    a local and a global layer exercised."""
    base = tiny_gemma2()
    patched = tiny_gemma2()
    patched.load_state_dict(base.state_dict())  # guarantee identical weights
    kinds = [layer.is_sliding for layer in patched.model.layers]
    assert any(kinds) and not all(kinds), f"need both local and global layers, got {kinds}"
    assert resolve_capabilities(patched) == ["attn.swa"]
    state = AnnealState()
    install_swa_anneal_patch(patched, state)

    base_vocab = int(base.config.vocab_size)
    probe = torch.randint(0, base_vocab, (2, 8))
    result = check_identity(base, patched, state, probe, base_vocab)

    assert result.passed
    assert result.max_abs_diff == 0.0  # eager + fp32 is exact, not merely within tolerance


def test_swa_gemma2_opens_window_and_future_at_alpha1_not_alpha0(
    tiny_gemma2: Callable[..., Any],
) -> None:
    """Gemma 2 end-to-end reveal through the signature-agnostic wrapper: on an
    all-sliding single-layer model an in-window key moves the query's logits even at
    alpha=0 (non-vacuity), while a far-past out-of-window key AND a strictly-future key
    move them at alpha=1 only."""
    model = tiny_gemma2(num_hidden_layers=1, sliding_window=2)
    assert all(layer.is_sliding for layer in model.model.layers)
    state = AnnealState()
    install_swa_anneal_patch(model, state)
    ids = torch.randint(0, int(model.config.vocab_size), (1, 8))

    # In-window past token (k=2) reaches q=3 already at alpha=0: guards against vacuity.
    assert _shift(model, state, ids, q=3, k=2, alpha=0.0) > 1e-6

    # Far-past token outside the window (k=0): windowed out at alpha=0, revealed at alpha=1.
    assert _shift(model, state, ids, q=3, k=0, alpha=0.0) == 0.0
    assert _shift(model, state, ids, q=3, k=0, alpha=1.0) > 1e-6

    # Strictly-future token (k=6): causally masked at alpha=0, revealed at alpha=1.
    assert _shift(model, state, ids, q=3, k=6, alpha=0.0) == 0.0
    assert _shift(model, state, ids, q=3, k=6, alpha=1.0) > 1e-6


def test_swa_wrapped_layer_survives_gradient_checkpointing(
    tiny_gemma2: Callable[..., Any], tiny_gemma3: Callable[..., Any]
) -> None:
    """The positional call path: under gradient checkpointing the model invokes the
    decoder layer with every argument positional, which the wrapper must re-bind
    against the original signature for both the Gemma 2 and Gemma 3 shapes."""
    for model in (tiny_gemma2(), tiny_gemma3()):
        state = AnnealState(alpha=0.5)
        install_swa_anneal_patch(model, state)
        model.gradient_checkpointing_enable()
        model.train()
        ids = torch.randint(0, int(model.config.vocab_size), (1, 8))
        out = model(ids, labels=ids)
        out.loss.backward()


def test_swa_install_on_non_sliding_model_raises_and_leaves_model_unpatched(
    tiny_gqa: Callable[..., Any],
) -> None:
    """Atomic install: on a model with no sliding decoder layers the install must raise
    BEFORE the shared mask patch runs, leaving use_cache and _update_causal_mask
    untouched."""
    model = tiny_gqa("gemma", 0)
    with pytest.raises(ValueError, match="no sliding-window decoder layers"):
        install_swa_anneal_patch(model, AnnealState())

    assert model.config.use_cache is True
    assert "_update_causal_mask" not in vars(model.model)
    assert not hasattr(model.model, "_a2d_gqa_orig_update_causal_mask")


def test_swa_reinstall_with_fresh_state_takes_effect(tiny_gemma3: Callable[..., Any]) -> None:
    """Re-installing on an already-patched model must swap in the NEW state without
    double-wrapping: after a second install at alpha=1, an out-of-window token reaches the
    query."""
    model = tiny_gemma3(num_hidden_layers=1, sliding_window=2, sliding_window_pattern=6)
    install_swa_anneal_patch(model, AnnealState(alpha=0.0))
    fresh = AnnealState(alpha=1.0)
    install_swa_anneal_patch(model, fresh)

    ids = torch.randint(0, int(model.config.vocab_size), (1, 8))
    assert _shift(model, fresh, ids, q=3, k=0, alpha=1.0) > 1e-6


def test_swa_patch_is_idempotent(tiny_gemma3: Callable[..., Any]) -> None:
    """Installing twice must not double-anneal: a second install is a no-op, so the
    alpha=0 causal+windowed behavior is preserved (q ignores an out-of-window token)."""
    model = tiny_gemma3(num_hidden_layers=1, sliding_window=2, sliding_window_pattern=6)
    state = AnnealState()
    install_swa_anneal_patch(model, state)
    install_swa_anneal_patch(model, state)  # second call must be inert

    ids = torch.randint(0, int(model.config.vocab_size), (1, 8))
    assert _shift(model, state, ids, q=3, k=0, alpha=0.0) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
