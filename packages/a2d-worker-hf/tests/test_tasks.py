"""Step 2: the two downstream eval tasks are registered, in-range, and deterministic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from a2d_core.eval.tasks import EVAL_TASKS
from a2d_core.eval.tasks.base import TaskContext
from conftest import ConvertSetup


def _ctx(setup: ConvertSetup, overrides: dict[str, str], seed: int = 0) -> TaskContext:
    from a2d_core.transform.apply import load_model, resolve_mask_token
    from a2d_core.transform.attention import AnnealState, install_anneal_patch

    model, tokenizer = load_model(str(setup.model_src))
    mask_id = resolve_mask_token(model, tokenizer, "grow")
    install_anneal_patch(model, AnnealState(alpha=1.0))  # bidirectional for scoring
    return TaskContext(
        model=model,
        tokenizer=tokenizer,
        mask_token_id=mask_id,
        device="cpu",
        seed=seed,
        max_examples=50,
        data_overrides=overrides,
    )


def _write(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def test_registry_lists_both_tasks() -> None:
    assert EVAL_TASKS.names() == ["cloze_likelihood", "infill_accuracy"]


def test_infill_accuracy_in_range_and_deterministic(
    convert_setup: ConvertSetup, tmp_path: Path
) -> None:
    infill = _write(
        tmp_path / "infill.jsonl",
        [{"text": " ".join(f"w{(i % 63) + 1}" for i in range(8))} for _ in range(5)],
    )
    overrides = {"infill_accuracy": infill}
    a = EVAL_TASKS.get("infill_accuracy")(_ctx(convert_setup, overrides))
    b = EVAL_TASKS.get("infill_accuracy")(_ctx(convert_setup, overrides))

    assert a.name == "infill_accuracy" and a.metric == "accuracy"
    assert 0.0 <= a.value <= 1.0
    assert a.n == 5
    assert a.value == b.value  # deterministic given the seed


def test_cloze_likelihood_in_range_and_deterministic(
    convert_setup: ConvertSetup, tmp_path: Path
) -> None:
    cloze = _write(
        tmp_path / "cloze.jsonl",
        [
            {"context": "w1 w2 w3", "choices": ["w4", "w5 w6", "w7"], "answer": 0},
            {"context": "w8 w9", "choices": ["w10", "w11"], "answer": 1},
            {"context": "w12 w13 w14", "choices": ["w15", "w16", "w17"], "answer": 2},
        ],
    )
    overrides = {"cloze_likelihood": cloze}
    a = EVAL_TASKS.get("cloze_likelihood")(_ctx(convert_setup, overrides))
    b = EVAL_TASKS.get("cloze_likelihood")(_ctx(convert_setup, overrides))

    assert a.name == "cloze_likelihood" and a.metric == "accuracy"
    assert 0.0 <= a.value <= 1.0
    assert a.n == 3
    assert a.value == b.value  # deterministic (argmax over per-choice log-prob)
