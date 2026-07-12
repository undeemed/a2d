"""MDLM iterative parallel confidence-reveal denoiser (Decision 9).

The canvas is a fixed ``prompt_ids`` prefix followed by ``M`` mask tokens. For each
of ``K`` steps we run ONE forward over the whole canvas at ``alpha=1`` (fully
bidirectional, ``use_cache=False``), take each position's max-softmax confidence and
argmax id, and reveal the most-confident still-masked positions on a linear top-k
schedule (cumulative ``round((step+1)/K * M)`` revealed) until no mask remains.

The remask policy (confidence-ordered reveal) is folded inline here; ``schedulers.py``
is a P5 add only when a second policy exists (no premature scaffolding).
"""

from __future__ import annotations

from typing import Any

import torch

from a2d_core.sample import register
from a2d_core.transform.apply import apply_transforms, resolve_capabilities
from a2d_core.transform.attention import AnnealState


@register("mdlm")
def denoise(
    model: Any,
    *,
    prompt_ids: list[int],
    mask_token_id: int,
    canvas_len: int,
    num_steps: int,
    temperature: float,
    device: str = "cpu",
) -> list[int]:
    """Return the fully-revealed canvas ids (``len == canvas_len``).

    ``prompt_ids`` is the untouched prefix; the ``canvas_len - len(prompt_ids)``
    suffix positions start masked and are filled by iterative confidence reveal.
    Installs the model's resolved attention transform (GPT-2 -> ``attn.full``, RoPE
    family -> ``attn.gqa``) at ``alpha=1`` so attention is bidirectional.
    """
    prompt_len = len(prompt_ids)
    if canvas_len < prompt_len:
        raise ValueError(f"canvas_len {canvas_len} < prompt length {prompt_len}")
    num_masked = canvas_len - prompt_len

    # Bidirectional (Decision 9), via the same capability dispatch as the worker.
    apply_transforms(model, resolve_capabilities(model), AnnealState(alpha=1.0))
    dev = torch.device(device)
    model.to(dev).eval()

    canvas = torch.tensor(
        prompt_ids + [mask_token_id] * num_masked, dtype=torch.long, device=dev
    ).unsqueeze(0)
    masked = torch.zeros(canvas_len, dtype=torch.bool, device=dev)
    masked[prompt_len:] = True

    steps = max(1, num_steps)
    # ponytail: temperature only scales confidence ranking (argmax id is temp-invariant);
    # clamp so temperature=0 does not divide by zero.
    temp = max(temperature, 1e-6)
    for i in range(steps):
        if not bool(masked.any()):
            break
        with torch.no_grad():
            logits = model(input_ids=canvas).logits[0]  # [canvas_len, vocab]
        conf, ids = torch.softmax(logits / temp, dim=-1).max(dim=-1)
        remaining = int(masked.sum())
        # Linear top-k schedule; the final step reveals whatever is left.
        target = num_masked if i == steps - 1 else round((i + 1) / steps * num_masked)
        k = min(max(target - (num_masked - remaining), 0), remaining)
        if k == 0:
            continue
        top = torch.topk(conf.masked_fill(~masked, float("-inf")), k).indices
        canvas[0, top] = ids[top]
        masked[top] = False

    return canvas[0].tolist()
