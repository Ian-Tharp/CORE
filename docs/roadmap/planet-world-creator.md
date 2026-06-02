# Planet World Creator — Tiered Zoom + Procedural Sphere Studio

**Date:** 2026-06-02
**Status:** Vision + self-prompts (not yet implemented). Successor to the removed 2D paint palette.

## Vision

Replace the (now-removed) flat tile-paint palette with a **3D sphere world creator**, reached by **zooming through altitude tiers**. Each command-center orb is a world; diving into one turns it into a **planet-scale procedural sphere** you configure with parameters + biome selection. Those parameters are the world's "palette" — and they feed the existing AI art/lore generation (biome already grounds `worldSubject()`).

### Zoom altitudes (LOD tiers)

| Tier | What you see | What you do |
|------|--------------|-------------|
| **GALAXY** (today) | the spiral galaxy of world-orbs | navigate / select / Next-Back stepper |
| **ORBIT** | the selected orb swells into a rotating planet with atmosphere + clouds | confirm "enter this world", quick glance |
| **PLANET (creator)** | a full procedural sphere — terrain, biome colours, water, atmosphere | the **creation studio**: param sliders, biome palette, re-roll, regenerate art/lore |
| **SURFACE** (later) | close-up of a region | generate/inspect surface art; localized lore |

`Esc` steps back up an altitude; selecting steps down. This mirrors Elite Dangerous' galaxy→system→approach→surface flow and our existing `engine.focusOn()/returnToOverview()` camera tweens.

## Technical pillars (from the research)

- **Sphere mesh** — an **icosphere** or **cube-sphere** (6 subdividable faces) displaced along its normals by layered noise. Cube-sphere gives clean LOD per face; icosphere gives uniform vertices. (Sebastian Lague; Andreas Kähler.)
- **Procedural height** — fractal simplex/Perlin: `octaves`, `persistence`, `lacunarity`, `noiseStrength`; ridged noise for mountains; a `seaLevel` threshold. Seeded for reproducibility.
- **Biome colouring (Whittaker)** — per-vertex/fragment **temperature** (from latitude + elevation) × **moisture** (a second noise field) → Whittaker lookup → colour ramp (triplanar blend of e.g. sand/grass/rock/snow). One **archetype biome** per world plus local variation, à la No Man's Sky.
- **Water + atmosphere as post-process** — a translucent sea sphere + a Lague-style fresnel/scatter atmosphere shell (we already do orb fresnel glow). Clouds = a second alpha sphere.
- **LOD** — subdivide / swap mesh resolution by camera distance, so the same world is a cheap orb at GALAXY and a detailed sphere at PLANET.

### `WorldGenParams` (the new world "palette")
```
seed: string
radius, octaves, persistence, lacunarity, noiseStrength: number
seaLevel, mountainHeight: number
temperatureBias, moistureBias: number   // shift the Whittaker lookup
biomeArchetype: 'temperate'|'desert'|'tundra'|'oceanic'|'volcanic'|'jungle'|'astral'|…
hasWater, hasAtmosphere, hasClouds: boolean
palette: { water, sand, grass, rock, snow }   // colour ramp
```
Persisted per world (backend) and reloadable; `biomeArchetype` is the bridge to the existing art-theme + lore grounding.

## Integration with CORE

- **Engine** — reuse `engine.service` `focusOn/animateCameraTo/returnToOverview/onBeforeRender/projectPoint`; add altitude state.
- **Tile-grid** — the orb is the GALAXY-tier LOD of the same world; swap to the detailed sphere at PLANET tier.
- **Persistence** — store `WorldGenParams` alongside the world (e.g. `world_metadata` snapshot or a dedicated column); reload on dive.
- **AI** — `biomeArchetype` + params seed the existing image/lore prompts (and the lore-agents work in flight). A "Regenerate art from this planet" button closes the loop.
- **Replaces** — the deleted paint palette; the detail panel's biome selector (the coordinated follow-up) becomes part of this creator.

## Self-prompts (engage with these in order)

Each is a self-contained task — runnable directly, as an Explore/Plan agent, or as a Workflow phase. Keep the sphere renderer **decoupled** so it can be built and demoed in isolation before wiring into the galaxy.

### Prompt 1 — Anchor (read-only)
> Map how to slot a planet-scale view into the command center without disturbing the galaxy. Read `engine.service.ts` (camera tweens `focusOn/animateCameraTo/returnToOverview`, `onBeforeRender`, `projectPoint`, dispose), `tile-grid.service.ts` (orb meshes, `getOrbScale`, selection/focus, per-frame hooks), `world-detail-panel.component.*` (where world config + biome live today), and the worlds persistence (`world_metadata`, world config, `worlds.py` routes). Output: the exact seams to add an altitude state machine, where to mount a planet `THREE.Group`, and where `WorldGenParams` should be stored + loaded. Flag anything owned by the parallel worker to avoid collisions.

### Prompt 2 — Standalone procedural planet module
> Build a self-contained `engine/planet/` Three.js module that renders one procedural sphere from a `WorldGenParams` object: cube-sphere or icosphere displaced by seeded fractal noise (octaves/persistence/lacunarity/noiseStrength, ridged mountains, seaLevel), Whittaker biome colouring (temperature from latitude+elevation × moisture noise → colour ramp, snow caps), and optional translucent water sphere + fresnel atmosphere shell. Expose `buildPlanet(params): THREE.Group`, `updatePlanet(params)` (re-mesh on change), and `disposePlanet()`. Include a tiny standalone demo route/harness so it can be tuned in isolation, reduced-motion aware, token-coloured where UI shows. Reference open-source ports (prolearner/procedural-planet, fqhd/ProceduralPlanets, SebLague/Procedural-Planets) for the noise+shader approach; do not copy wholesale.

### Prompt 3 — Zoom-altitude state machine + camera tiers
> Add a GALAXY → ORBIT → PLANET (→ SURFACE) altitude state to the command center. On selecting a world, animate `focusOn` into ORBIT (orb swells, rotates); a "Descend"/double-click transitions to PLANET, mounting the Prompt-2 planet group at the orb's position and swapping the orb LOD for the detailed sphere; `Esc` ascends. Drive transitions through the existing camera tweens, keep it reduced-motion safe (collapse swoops to quick fades), and ensure label/stepper/detail-panel view-state never mutates mid-CD (NG0100 discipline). No painting; selection only.

### Prompt 4 — Planet creator UI + params + persistence
> Build the planet creator panel shown at PLANET altitude: sliders/controls for the `WorldGenParams` (seed + re-roll, terrain detail/roughness, sea level, mountain height, temperature, moisture, biome archetype, water/atmosphere/clouds toggles, colour ramp), live-updating the sphere via `updatePlanet`. Define a `WorldGenParams` model shared FE/BE, persist it per world (backend route + storage) and reload on dive. Wire `biomeArchetype` + params into the existing art/lore prompt builders, and add a "Regenerate art/lore from this planet" action. Token-only solarpunk styling, reduced-motion safe, a11y (labelled controls, focus-visible). Coordinate with the parallel worker on `world-detail-panel`.

### Prompt 5 — Loop closure + polish
> Make the galaxy reflect each world's params: orb colour/atmosphere derived from `biomeArchetype`/palette (so the universe reads at a glance), a subtle "ignite" pulse when a world is configured, and a thumbnail/preview captured from the planet view for the world-select slots. Add LOD/perf guards (subdivision by distance, dispose off-altitude planets, cap concurrent spheres), and verify the whole galaxy↔planet loop stays within the reduced-motion/perf floor.

## References
- [SebLague/Procedural-Planets](https://github.com/SebLague/Procedural-Planets) — the canonical sphere + noise + atmosphere approach
- [fqhd/ProceduralPlanets](https://github.com/fqhd/ProceduralPlanets) — WebGL2 realtime planet (Lague water, Kähler sphere)
- [prolearner/procedural-planet](https://github.com/prolearner/procedural-planet) — Three.js planet with sky/water/atmosphere shaders + triplanar sand/grass/stone/snow
- [IceCreamYou/THREE.Terrain](https://github.com/IceCreamYou/THREE.Terrain) — elevation/slope/biome auto-texturing in Three.js
- [Whittaker Diagram (PCG wiki)](http://pcg.wikidot.com/pcg-algorithm:whittaker-diagram) — temperature×moisture → biome
- [Biomes generation & rendering (Azgaar)](https://azgaar.wordpress.com/2017/06/30/biomes-generation-and-rendering/) — practical Whittaker implementation
- [World Orogen](https://www.orogen.studio/) — browser planet generator UX (detail/continents/temperature/precipitation sliders)
- [No Man's Sky procedural generation](https://nomanssky-archive.fandom.com/wiki/Procedural_generation) — two-level gen, biome archetypes, tagged props
- [Elite Dangerous System Map](https://elite-dangerous.fandom.com/wiki/System_Map) — galaxy→system→approach→surface zoom tiers
