# CORE — UX & Design Principles

_Shared learnings for building CORE's interface. Read this before adding or
restyling a view. It captures the **mindset**, not just the rules — the rules
change, the judgment shouldn't._

---

## 1. The ethos

- **Solarpunk × LCARS, "optimistic futuristic."** Cyan/mint energy, warm amber
  accents, deep-navy space, HUD/scan-line chrome, glassmorphism. A JARVIS-grade
  **command surface**, not a chatbot skin.
- **CORE is a neutral cognitive kernel, not a persona.** The UI is a command
  deck the human *operates*, surfacing CORE's cognition (Comprehension →
  Orchestration → Reasoning → Evaluation), live telemetry, and council
  deliberation — not a face with a personality.
- **The command deck (`landing-page`) is the reference implementation.** A new
  view should feel like the same machine: glass HUD panels, energy accents,
  monospace telemetry, choreographed entrances.

---

## 2. Standards are a floor, not a ceiling

This is the core mindset: **accessibility standards are a floor that frees bold
design — not a checklist that flattens it.** Going for blanket WCAG AAA isn't
realistic (W3C says so itself); dogmatically chasing it would grey-out the whole
aesthetic. So:

- **AA everywhere is the hard floor — non-negotiable.** 4.5:1 text, 3:1 large
  text / UI components, visible focus, keyboard operable, reduced-motion safe.
- **AAA where it serves comprehension & interaction.** 7:1 body text, ≥44px
  primary targets, enhanced focus. Worth it for reading-heavy surfaces and
  primary controls.
- **Don't chase AAA where it fights the vision.** The cyan/amber **energy is
  expressive decoration** — accents, glows, status rails, scan-lines. Forcing
  7:1 on every neon flourish would flatten the look. Keep neon as *emphasis*;
  keep *meaning* in tokened text that clears contrast.
- **Never signal by color alone.** Pair a status color with a shape, icon, or
  label (a red dot is also "✕ Error"; a pulsing ring also says "RUNNING").

> The floor is settled so the ceiling can be wild. Do what's necessary for the
> standard, then think outside the box.

---

## 3. Tokens are the single source of truth

- All color / spacing / radius / type / motion come from
  [`src/solarpunk-theme.scss`](../../ui/core-ui/src/solarpunk-theme.scss):
  `--core-*`, `--spacing-*`, `--radius-*`, `--font-*`, `--transition-*`.
- **Never hardcode an `rgba()`/hex that duplicates a token.** Derive variants
  with `color-mix(in srgb, var(--core-energy) N%, transparent)`.
- **Fix accessibility at the token, not the component.** Raising one text token
  lifts every muted label in the app at once. Per-component overrides drift.

### Contrast reference (≈, measured on the deep-navy panels `#0a1628`)

| Token | Hex | ≈ ratio | Verdict | Use for |
|---|---|---|---|---|
| `--core-text-primary` | `#e8e8e8` | ~15:1 | AAA | body & headings |
| `--core-text-secondary` | `#9aa6c6` | ~7:1 | AAA | muted labels, meta, captions |
| `--core-text-dim` | `#6b7793` | ~AA | AA-ish | **decorative / disabled / placeholder only — never body text** |
| `--core-energy` (neon) | `#00ffc8` | high | — | accents, focus, status — *emphasis, not paragraphs* |

_Ratios are hand-computed estimates — confirm with axe / Lighthouse before
claiming a level. The lesson that produced this table: a label that is both
small **and** on `--core-text-dim` was failing even AA._

---

## 4. Motion

- **Every animation must be `@media (prefers-reduced-motion: reduce)` safe.**
- **Motion carries meaning** — entrance choreography, status pulses, hover
  micro-interactions, count-ups — not decoration for its own sake.
- **Contain animated transforms** (`overflow: clip` + `overflow-clip-margin`) so
  expanding glows/waves never spawn a scrollbar.

---

## 5. Angular Material, tamed

- **Scope Material-internal overrides with `:host ::ng-deep`** so they apply
  *and* don't leak. Bare `.mat-mdc-*` selectors under Emulated encapsulation
  silently no-op — a frequent "why isn't my style applying?" trap.
- **Material 19 is MDC-based** (`.mat-mdc-*`, `.mdc-*`). Legacy class names
  (`.mat-list-item`, `.mat-line`) won't match. When Material fights the theme,
  prefer a **small custom component** over class-name combat (the conversation
  list is a good example).
- **Kill the default white seams explicitly** — notched outline, line-ripple,
  subscript row — and restyle the wrapper as dark glass with a cyan focus ring.

---

## 6. Layout & scroll

- Routed pages live inside an `overflow-y: auto` shell. A view that grows
  unbounded scrolls the **whole page**, dragging fixed UI (sidebars, lists) out
  of view. **Own the viewport height and let each pane scroll internally.**
- Use **`minmax(0, …)`** grid/flex tracks so long content (UUIDs, `<pre>`) wraps
  instead of forcing horizontal overflow.

---

## 7. Agnostic by default

- The UI must not bake in one machine's config. Provider/model lists, labels,
  and capabilities are **resolved at runtime** from the backend
  (e.g. `/local-llm/provider`, `/local-llm/models`) so the same build serves an
  Ollama desktop and an LM Studio laptop. See
  [Local LLM Providers](../deployment/local-llm-providers.md).

---

## 8. Before you ship a view — checklist

- [ ] **Tokens only** — no duplicated hex/rgba
- [ ] **AA** contrast on all text & controls; **AAA** on body text & primary actions
- [ ] Visible `:focus-visible`; keyboard operable; ≥44px primary targets
- [ ] Status is **not** color-only
- [ ] **Reduced-motion safe**; animated transforms contained
- [ ] Scrolls internally (no whole-page push-down); `minmax(0, …)` tracks
- [ ] Nothing hardcoded that should come from config (provider/model/labels)
- [ ] Feels like the **command deck** (glass HUD, energy accents, mono telemetry)
- [ ] `npm run build:ng` is clean

---

## 9. Think outside the box

CORE is a place to **push human–AI interaction**, not replicate a SaaS dashboard.
Make cognition visible (reactor/vitals metaphors, the live pipeline, council
deliberation as a chamber); invent new surfaces for reasoning, memory, and
emergence. Just keep the meaning legible and the motion kind — that's what the
floor buys you: the freedom to be bold without leaving anyone behind.
