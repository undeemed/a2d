"""Step 9: continual training over a tiny corpus produces checkpoints + events.

Exercises the real data reader (jsonl) and the real ``Trainer`` wiring on a seeded
tiny GPT-2, CPU only, no download. A tiny fake tokenizer keeps the corpus in the
64-token vocab and the whole thing network-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from a2d_contracts import ConversionConfig
from a2d_core.data import DATA
from a2d_core.train import continual
from a2d_core.transform.attention import AnnealState

MASK_ID = 63  # distinct from eos (0) and every data token (1..62)


class _FakeTokenizer:
    """Char-level tokenizer into ids 1..62 (eos=0), enough for the data reader."""

    eos_token_id = 0

    def __call__(self, text: str) -> dict[str, list[int]]:
        return {"input_ids": [(ord(c) % 62) + 1 for c in text]}


def _cfg(data: str) -> ConversionConfig:
    return ConversionConfig(
        objective="mdlm",
        data=data,
        anneal_steps=2,
        anneal_schedule="linear",
        seq_len=8,
        per_device_batch_size=2,
        grad_accum=1,
        lr=1e-3,
        max_steps=2,
        max_tokens=None,
        mask_token="grow",
        keep_last=3,
        seed=0,
        device="cpu",
        dtype="float32",
    )


def test_continual_train_writes_checkpoints_and_events(tmp_path: Path, tiny_gpt2: Any) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            '{"text": "the quick brown fox jumps over the lazy dog again"}' for _ in range(4)
        ),
        encoding="utf-8",
    )
    dataset = DATA.get("jsonl")(str(corpus), _FakeTokenizer(), 8)

    cfg = _cfg(str(corpus))
    state = AnnealState()
    events: list[dict[str, Any]] = []
    output_dir = tmp_path / "checkpoints"

    continual.train(
        model=tiny_gpt2(),
        dataset=dataset,
        cfg=cfg,
        mask_token_id=MASK_ID,
        state=state,
        emit=events.append,
        output_dir=output_dir,
        save_steps=1,  # checkpoint every step so a 2-step run writes some
    )

    # checkpoints/ written on disk
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    assert checkpoints, "no checkpoint-* dir written"

    # TrainStep events: one per optimizer step, steps 1..2, monotone anneal
    train_steps = [e for e in events if e["type"] == "train_step"]
    assert [e["step"] for e in train_steps] == [1, 2]
    assert all(isinstance(e["loss"], float) for e in train_steps)
    assert train_steps[0]["anneal"] <= train_steps[1]["anneal"]
    assert train_steps[1]["tokens"] == 2 * (8 * 2 * 1)  # step * seq_len*bs*accum

    # Checkpoint events captured
    checkpoint_events = [e for e in events if e["type"] == "checkpoint"]
    assert checkpoint_events, "no checkpoint event emitted"
    assert Path(checkpoint_events[-1]["path"]).name.startswith("checkpoint-")
