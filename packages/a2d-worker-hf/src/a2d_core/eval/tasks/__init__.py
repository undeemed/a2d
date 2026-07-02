"""Extension point: eval tasks (SPEC-HANDOFF 3.3).

New eval task => one task module here. Tasks self-register via a ``Registry`` and
the gate suite is parameterized over the registry, so a new task joins the matrix
automatically.
"""
