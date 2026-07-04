"""``a2d-sample`` worker: MDLM denoise a saved ``model/`` and print text (Decision 9).

Reads ONE ``SampleRequest`` JSON document from stdin, validates the generated
pydantic (exit 2 on contract violation), loads the model + tokenizer, runs the MDLM
denoiser, and prints the decoded text to stdout (exit 0). This is a spot-check, so
it touches neither the manifest nor the event stream. Reuses the ``worker.py``
stdin-JSON + pydantic-validate pattern rather than inventing argv parsing.
"""

from __future__ import annotations

import sys

from a2d_contracts import SampleRequest
from pydantic import ValidationError

from a2d_core.worker import SCHEMA_VERSION


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = SampleRequest.model_validate_json(raw)
    except ValidationError as exc:
        print(f"a2d-sample: invalid SampleRequest: {exc}", file=sys.stderr)
        return 2

    if req.schema_version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        print(
            "a2d-sample: schema_version major mismatch: "
            f"request={req.schema_version!r} worker={SCHEMA_VERSION!r}",
            file=sys.stderr,
        )
        return 2

    import torch

    from a2d_core.device import select_device
    from a2d_core.sample import SAMPLERS
    from a2d_core.transform.apply import load_model

    torch.manual_seed(req.seed)
    model, tokenizer = load_model(req.model_dir)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        print(f"a2d-sample: {req.model_dir} tokenizer has no mask token", file=sys.stderr)
        return 2

    prompt_ids = tokenizer(req.prompt)["input_ids"]
    # ponytail: P2 is MDLM-only; read the sampler name from the model's a2d block when
    # P5 adds a second sampler (BD3LM) behind this same registry.
    ids = SAMPLERS.get("mdlm")(
        model,
        prompt_ids=prompt_ids,
        mask_token_id=int(mask_token_id),
        canvas_len=int(req.canvas_len),
        num_steps=int(req.num_steps),
        temperature=req.temperature,
        device=select_device(req.device),
    )
    print(tokenizer.decode(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
