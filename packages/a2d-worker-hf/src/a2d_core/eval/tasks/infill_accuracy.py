"""infill_accuracy: exact-match recovery of masked tokens (Decision 3).

Mask a seed-determined ~15% of each held-out sequence, run ONE bidirectional forward
(alpha=1), argmax at the masked positions, and score exact-match against the originals.
Single-step (not iterative denoise) so the metric is cheap and deterministic; it is
the model's core infilling competency and needs no labels.
"""

from __future__ import annotations

import torch

from a2d_core.eval.tasks import register
from a2d_core.eval.tasks.base import TaskContext, TaskScore, load_jsonl, task_data_path

_MASK_FRACTION = 0.15


@register("infill_accuracy")
def infill_accuracy(ctx: TaskContext) -> TaskScore:
    rows = load_jsonl(task_data_path(ctx, "infill_accuracy", "infill.jsonl"), ctx.max_examples)
    dev = torch.device(ctx.device)
    ctx.model.to(dev).eval()

    correct = 0
    total = 0
    n = 0
    for i, row in enumerate(rows):
        ids = ctx.tokenizer(str(row["text"]))["input_ids"]
        if len(ids) < 2:
            continue
        seq = torch.tensor(ids, dtype=torch.long, device=dev)
        # Deterministic per-example mask choice: at least one position, never the last-only.
        gen = torch.Generator().manual_seed(ctx.seed + i)
        k = max(1, int(round(len(ids) * _MASK_FRACTION)))
        pos = torch.randperm(len(ids), generator=gen)[:k].to(dev)
        noisy = seq.clone()
        noisy[pos] = ctx.mask_token_id
        with torch.no_grad():
            logits = ctx.model(input_ids=noisy.unsqueeze(0)).logits[0]  # [seq, vocab]
        pred = logits[pos].argmax(dim=-1)
        correct += int((pred == seq[pos]).sum())
        total += int(pos.numel())
        n += 1

    accuracy = correct / total if total else 0.0
    return TaskScore(name="infill_accuracy", metric="accuracy", value=accuracy, n=n)
