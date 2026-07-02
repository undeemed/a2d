# fixtures/golden

Placeholder holding this directory (git cannot track empty dirs).

Phase 2 fills it with pinned base-model logits for the identity test (D13): every transform handler must
leave `anneal=0` behavior identical to the base model. Local runs compute logits live; CI compares against
these goldens. See `docs/SPEC-HANDOFF.md` section 7.
