## What Changed
Brief description of changes.

## Why
Motivation and context.

## How to Test
Steps to verify the changes work.

## Checklist
- [ ] Tests pass
- [ ] Code follows project style guidelines
- [ ] Documentation updated (if applicable)
- [ ] Conventional commit messages used

## UI / UX changes
_Only if this PR touches the frontend. Full rationale:
[UX & Design Principles](../docs/design/ux-design-principles.md) → "before you ship a view"._

- [ ] **Design tokens only** — no hardcoded hex/rgba that duplicates a token
- [ ] **AA** contrast on all text/controls; **AAA** on body text & primary actions
- [ ] Visible `:focus-visible`, keyboard operable, ≥44px primary targets
- [ ] Status is **not** signaled by color alone
- [ ] **Reduced-motion safe**; animated transforms contained (no spurious scrollbars)
- [ ] Scrolls internally (no whole-page push-down); `minmax(0, …)` grid tracks
- [ ] Nothing hardcoded that should come from runtime config (provider/model/labels)
- [ ] Feels like the **command deck** (glass HUD, energy accents, mono telemetry)
- [ ] `npm run build:ng` is clean
