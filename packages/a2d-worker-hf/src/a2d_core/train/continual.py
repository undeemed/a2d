"""HF ``Trainer`` wiring for MDLM continual pretraining (Decision 8).

Three small hooks bend ``Trainer`` to MDLM without hand-rolling the fiddly
optimizer/RNG/checkpoint/resume machinery: ``data_collator = objective.corrupt``
noises each batch; a ``compute_loss`` override scores the objective loss on raw
logits with NO label shift (Decision 3); and ``AnnealCallback`` drives ``alpha`` and
mirrors the loop onto the event stream. Resume is implicit - if ``output_dir``
already holds a ``checkpoint-*`` dir, ``Trainer`` restores optimizer/scheduler/RNG/step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from a2d_contracts import ConversionConfig
from transformers import Trainer, TrainingArguments

from a2d_core.objectives import OBJECTIVES
from a2d_core.train.callbacks import AnnealCallback, Emit
from a2d_core.transform.attention import AnnealState


def _has_checkpoint(output_dir: Path) -> bool:
    return output_dir.is_dir() and any(output_dir.glob("checkpoint-*"))


def train(
    *,
    model: Any,
    dataset: Any,
    cfg: ConversionConfig,
    mask_token_id: int,
    state: AnnealState,
    emit: Emit,
    output_dir: str | Path,
    save_steps: int = 500,
) -> Any:
    """Continual-pretrain ``model`` on ``dataset``; return the fitted ``Trainer``.

    ``output_dir`` is ``run_dir/checkpoints`` (Trainer writes ``checkpoint-*`` there);
    an existing checkpoint there signals resume. ``cfg.max_steps`` must already be
    resolved (config-build maps ``max_tokens`` -> ``max_steps``).
    """
    if cfg.max_steps is None:
        raise ValueError(
            "max_steps must be resolved before training (config-build maps max_tokens)"
        )

    objective_cls: Any = OBJECTIVES.get(cfg.objective)
    objective = objective_cls(mask_token_id, seed=cfg.seed)

    class _MDLMTrainer(Trainer):  # type: ignore[misc]
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> Any:
            clean = inputs.pop("clean")
            mask = inputs.pop("mask")
            outputs = model(input_ids=inputs["input_ids"])  # no labels, no shift (Decision 3)
            loss = objective.loss(outputs.logits, clean, mask)
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=cfg.lr,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=cfg.keep_last,
        seed=cfg.seed,
        logging_steps=1,  # one TrainStep event per optimizer step
        report_to=[],  # no wandb/tensorboard side channel; the event stream is the log
        remove_unused_columns=False,  # the collator adds clean/mask columns Trainer must keep
        disable_tqdm=True,
        use_cpu=cfg.device == "cpu",  # force CPU when asked (this Mac defaults to mps)
        bf16=cfg.dtype == "bfloat16",
    )

    tokens_per_step = cfg.seq_len * cfg.per_device_batch_size * cfg.grad_accum
    callback = AnnealCallback(state, cfg.anneal_steps, cfg.anneal_schedule, tokens_per_step, emit)

    trainer = _MDLMTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=objective.corrupt,
        callbacks=[callback],
    )
    trainer.train(resume_from_checkpoint=_has_checkpoint(Path(output_dir)))
    return trainer
