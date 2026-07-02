"""Extension point: training objectives (SPEC-HANDOFF 3.3).

New objective (post-MDLM / BD3LM research) => one module here implementing
``corrupt`` and ``loss``. Objectives self-register via a ``Registry``; there is
no central ``match`` to edit.
"""
