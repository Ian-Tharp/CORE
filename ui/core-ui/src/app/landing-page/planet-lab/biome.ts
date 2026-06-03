// biome.ts — Whittaker classifier -> LINEAR THREE.Color (allocation-light).
import * as THREE from 'three';
import type { WorldGenParams } from './world-gen-params';

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export interface PaletteLinear {
  water: THREE.Color; sand: THREE.Color; grass: THREE.Color; rock: THREE.Color; snow: THREE.Color;
}
export function linearPalette(p: WorldGenParams): PaletteLinear {
  const c = (hex: string) => new THREE.Color().setStyle(hex).convertSRGBToLinear();
  return {
    water: c(p.palette.water), sand: c(p.palette.sand), grass: c(p.palette.grass),
    rock: c(p.palette.rock), snow: c(p.palette.snow)
  };
}

const _scratch = new THREE.Color();
/**
 * heightAboveSea: signed world units. slope01: 0 flat..1 cliff (mesher supplies).
 * Writes into `out` (already-linear colours), returns it.
 */
export function biomeColor(
  heightAboveSea: number, temperature: number, moisture: number, slope01: number,
  pal: PaletteLinear, p: WorldGenParams, out: THREE.Color = new THREE.Color()
): THREE.Color {
  // 1) Water (deep -> shallow). Body colour; the water shell overlays it.
  if (heightAboveSea < 0) {
    const depth = clamp01(-heightAboveSea / (p.noiseStrength * p.radius + 1e-4));
    return out.copy(pal.water).lerp(_scratch.copy(pal.water).multiplyScalar(0.45), depth);
  }
  // 2) Shore/beach band, wider when arid
  const shoreWidth = THREE.MathUtils.lerp(0.012, 0.05, 1 - moisture) * (p.noiseStrength + 0.2) * p.radius;
  if (heightAboveSea < shoreWidth) { return out.copy(pal.sand); }
  // 3) Snow: cold OR high; snowLine rises with elevation
  const elevN = clamp01(heightAboveSea / (p.mountainHeight * p.radius + 1e-4));
  const snowLine = 0.32 + 0.45 * elevN;
  if (temperature < snowLine) {
    const t = clamp01((snowLine - temperature) / 0.25);
    return out.copy(pal.rock).lerp(pal.snow, t);
  }
  // 4) Grass body: dry->sand (savanna), wet->deep green; then blend rock by slope+altitude
  const aridity = 1 - moisture;
  const rockiness = clamp01(slope01 * 1.4 + elevN * 0.5);
  out.copy(pal.grass).lerp(pal.sand, aridity * 0.6).lerp(pal.rock, clamp01(rockiness - 0.45));
  // 5) Archetype flavour tints
  if (p.biomeArchetype === 'astral') { out.offsetHSL(0.0, 0.25, 0.04); } // neon pop
  return out;
}
