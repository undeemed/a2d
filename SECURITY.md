# Security Policy

a2d executes third-party model code in its Python worker.
`trust_remote_code` is off by default and only ever enabled per-run by explicit flag; anything that weakens that boundary is a security bug.

## Reporting a vulnerability

Please do not open a public issue for security problems.
Use [GitHub private vulnerability reporting](https://github.com/undeemed/a2d/security/advisories/new) instead.
You should get a response within a week.
