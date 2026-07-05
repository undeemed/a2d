"""cloze_likelihood: multiple-choice accuracy scored by the model's own infilling.

For each ``{context, choices, answer}`` example, place ``context ++ [mask]*len(choice)``,
run ONE bidirectional forward (alpha=1), and sum the log-prob of the true choice tokens
at their masked positions. Pick the highest-scoring choice; accuracy is ``pick == answer``.
This reuses the diffusion model's infilling to score candidates, the way AR models are
scored by perplexity (Decision 3), and needs a single forward per choice.
"""

from __future__ import annotations

import torch

from a2d_core.eval.tasks import register
from a2d_core.eval.tasks.base import TaskContext, TaskScore, load_jsonl, task_data_path


def _choice_logprob(
    ctx: TaskContext, dev: torch.device, ctx_ids: list[int], choice_ids: list[int]
) -> float:
    """Sum log P(choice token) over the masked choice region, in one forward."""
    canvas = torch.tensor(
        ctx_ids + [ctx.mask_token_id] * len(choice_ids), dtype=torch.long, device=dev
    ).unsqueeze(0)
    with torch.no_grad():
        logits = ctx.model(input_ids=canvas).logits[0]  # [len, vocab]
    logp = torch.log_softmax(logits.float(), dim=-1)
    start = len(ctx_ids)
    targets = torch.tensor(choice_ids, dtype=torch.long, device=dev)
    picked = logp[start : start + len(choice_ids)].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(picked.sum())


@register("cloze_likelihood")
def cloze_likelihood(ctx: TaskContext) -> TaskScore:
    rows = load_jsonl(task_data_path(ctx, "cloze_likelihood", "cloze.jsonl"), ctx.max_examples)
    dev = torch.device(ctx.device)
    ctx.model.to(dev).eval()

    correct = 0
    n = 0
    for row in rows:
        ctx_ids = ctx.tokenizer(str(row["context"]))["input_ids"]
        choices = [ctx.tokenizer(str(c))["input_ids"] for c in row["choices"]]
        if not choices or any(len(c) == 0 for c in choices):
            continue
        scores = [_choice_logprob(ctx, dev, ctx_ids, c) for c in choices]
        pick = int(torch.tensor(scores).argmax())
        correct += int(pick == int(row["answer"]))
        n += 1

    accuracy = correct / n if n else 0.0
    return TaskScore(name="cloze_likelihood", metric="accuracy", value=accuracy, n=n)
