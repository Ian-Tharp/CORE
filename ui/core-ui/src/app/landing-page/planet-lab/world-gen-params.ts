// world-gen-params.ts — single source of truth for the planet "palette".
// Pure; no THREE import (types only where needed elsewhere).

export type BiomeArchetype =
  | 'temperate' | 'desert' | 'tundra' | 'oceanic' | 'volcanic' | 'jungle' | 'astral';

export interface PlanetPalette {
  water: string; sand: string; grass: string; rock: string; snow: string; // sRGB hex
}

export interface WorldGenParams {
  seed: string;
  radius: number;          // world units; geometry is built at radius 1 and scaled via displacement
  octaves: number;         // 1..8
  persistence: number;     // amplitude falloff per octave (~0.5)
  lacunarity: number;      // frequency growth per octave (~2.0)
  noiseStrength: number;   // continental displacement, fraction of radius (0..1)
  seaLevel: number;        // normalized threshold on the [0,1] continental field
  mountainHeight: number;  // ridged-mountain displacement multiplier (0..1)
  temperatureBias: number; // [-1..1] shifts biome banding warm/cold
  moistureBias: number;    // [-1..1] shifts wet/dry
  biomeArchetype: BiomeArchetype;
  hasWater: boolean;
  hasAtmosphere: boolean;
  hasClouds: boolean;
  palette: PlanetPalette;
}

/** Perf tier -> IcosahedronGeometry(1, detail). Detail is DERIVED, never a raw param. */
export type LodTier = 'preview' | 'standard' | 'high' | 'ultra';
/** INDEXED icosphere vertex count = 10*4^detail + 2. */
export const LOD_DETAIL: Record<LodTier, number> = {
  preview: 4,  //  ~2,562 verts  (galaxy embed)
  standard: 5, // ~10,242 verts  (demo default)
  high: 6,     // ~40,962 verts  (close inspect)
  ultra: 7    // ~163,842 verts (demo toggle only)
};
export const MAX_DETAIL = 7;
export interface PlanetOptions { tier?: LodTier; }
export function detailFor(tier: LodTier): number { return Math.min(MAX_DETAIL, LOD_DETAIL[tier]); }
/** INDEXED icosphere vertex count. */
export function vertexCount(detail: number): number { return 10 * 4 ** detail + 2; }

export const DEFAULT_PLANET_PARAMS: WorldGenParams = {
  seed: 'core-0001',
  radius: 1, octaves: 5, persistence: 0.5, lacunarity: 2.0,
  noiseStrength: 0.18, seaLevel: 0.5, mountainHeight: 0.35,
  temperatureBias: 0, moistureBias: 0,
  biomeArchetype: 'temperate',
  hasWater: true, hasAtmosphere: true, hasClouds: true,
  palette: { water: '#1d6e7a', sand: '#c9b287', grass: '#4f9d5d', rock: '#6b6256', snow: '#eef3f2' }
};

/** Preset = DEFAULTS overlay (caller values still win for fields the caller explicitly sets). */
type Preset = Partial<Omit<WorldGenParams, 'seed' | 'biomeArchetype'>>;
export const BIOME_PRESETS: Record<BiomeArchetype, Preset> = {
  temperate: {},
  desert:   { seaLevel: 0.30, mountainHeight: 0.28, temperatureBias: 0.30, moistureBias: -0.45,
    palette: { water: '#2d7d9a', sand: '#e3c887', grass: '#bca85c', rock: '#9c6f43', snow: '#efe6cf' } },
  tundra:   { seaLevel: 0.45, mountainHeight: 0.22, temperatureBias: -0.45, moistureBias: 0.05,
    palette: { water: '#5b8aa6', sand: '#c9cdd2', grass: '#8fae8c', rock: '#7d8389', snow: '#f7fbff' } },
  oceanic:  { seaLevel: 0.68, mountainHeight: 0.30, temperatureBias: 0.05, moistureBias: 0.30,
    palette: { water: '#1c5f86', sand: '#cdb98a', grass: '#3f8f5a', rock: '#5f6b6e', snow: '#eef6fb' } },
  volcanic: { seaLevel: 0.35, mountainHeight: 0.55, temperatureBias: 0.45, moistureBias: -0.30,
    palette: { water: '#3a2b2b', sand: '#5a4036', grass: '#b5532a', rock: '#2c2624', snow: '#ffcaa0' } },
  jungle:   { seaLevel: 0.52, mountainHeight: 0.32, temperatureBias: 0.30, moistureBias: 0.45,
    palette: { water: '#15716b', sand: '#cdbf7e', grass: '#2f7d33', rock: '#5d5a44', snow: '#eafff2' } },
  astral:   { seaLevel: 0.40, mountainHeight: 0.40, temperatureBias: 0, moistureBias: 0.10,
    hasAtmosphere: true,
    palette: { water: '#142a4d', sand: '#6a5a8f', grass: '#2fd9c8', rock: '#3a3358', snow: '#b98cff' } }
};

const NUM_CLAMP: Record<string, [number, number]> = {
  radius: [0.05, 50], octaves: [1, 8], persistence: [0.1, 0.95], lacunarity: [1.2, 4],
  noiseStrength: [0, 1], seaLevel: [0, 1], mountainHeight: [0, 1],
  temperatureBias: [-1, 1], moistureBias: [-1, 1]
};

export function clampParams(p: WorldGenParams): WorldGenParams {
  const out: WorldGenParams = { ...p, palette: { ...p.palette } };
  for (const k of Object.keys(NUM_CLAMP)) {
    const [lo, hi] = NUM_CLAMP[k];
    (out as any)[k] = Math.min(hi, Math.max(lo, Number((p as any)[k])));
  }
  out.octaves = Math.round(out.octaves);
  return out;
}

/** Preset-as-DEFAULT precedence: DEFAULTS < preset < caller. Palette merged field-wise. */
export function resolveParams(input: Partial<WorldGenParams> & { seed: string }): WorldGenParams {
  const archetype = input.biomeArchetype ?? DEFAULT_PLANET_PARAMS.biomeArchetype;
  const preset = BIOME_PRESETS[archetype];
  const merged: WorldGenParams = {
    ...DEFAULT_PLANET_PARAMS, ...preset, ...input,
    biomeArchetype: archetype,
    palette: { ...DEFAULT_PLANET_PARAMS.palette, ...preset.palette, ...input.palette }
  };
  return clampParams(merged);
}

/** Topology identity: ONLY radius + tier change triangulation. */
export function topoSig(p: WorldGenParams, opts: PlanetOptions = {}): string {
  return `${opts.tier ?? 'standard'}|${p.radius.toFixed(4)}`;
}

/** Full params hash for no-op detection. */
export function hashParams(p: WorldGenParams): string {
  const o = clampParams(p);
  return JSON.stringify([
    o.seed, o.radius, o.octaves, o.persistence, o.lacunarity, o.noiseStrength,
    o.seaLevel, o.mountainHeight, o.temperatureBias, o.moistureBias,
    o.biomeArchetype, o.hasWater, o.hasAtmosphere, o.hasClouds,
    o.palette.water, o.palette.sand, o.palette.grass, o.palette.rock, o.palette.snow
  ]);
}
export function paramsEqual(a: WorldGenParams, b: WorldGenParams): boolean {
  return hashParams(a) === hashParams(b);
}
