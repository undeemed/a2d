"""Extension point: transform handlers (SPEC-HANDOFF 3.3).

New attention variant => one handler module here, plus a capability tag and a
conformance test. Handlers self-register via a ``Registry`` and must leave
``anneal=0`` behavior identical to the base model (identity / golden-logits test).
"""
