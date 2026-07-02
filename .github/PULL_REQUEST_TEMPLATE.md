## What

<!-- One or two sentences: what does this PR do, and why? Link the issue if one exists. -->

## Checklist

- [ ] All local gates pass (see [CONTRIBUTING.md](../CONTRIBUTING.md)): build, tests, lints, codegen.
- [ ] If I changed boundary types in `crates/a2d-contracts`, I ran `bash scripts/codegen.sh` and committed `schema/` and `packages/a2d-contracts` in the same commit.
- [ ] If I added support for a model, attention variant, objective, format, or eval task, I **added** files at the matching extension point rather than editing existing modules ([SPEC-HANDOFF section 3.3](../docs/SPEC-HANDOFF.md)).
- [ ] New non-trivial logic has a test.
