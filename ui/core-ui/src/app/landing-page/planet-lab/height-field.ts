// height-field.ts — pure fields. `unit` MUST be normalized (length 1).
import type { Vector3 } from 'three';
import { WorldFields, makeFields } from './noise';
import type { WorldGenParams } from './world-gen-params';

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
function smoothstep(a: number, b: number, x: number): number {
  const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t);
}

export interface VertexSample {
  elevation01: number;    // raw [0,1] continental field
  heightAboveSea: number; // signed displacement in WORLD UNITS, relative to radius
  temp: number;           // [0,1]
  moist: number;          // [0,1]
}

/** ONE evaluation pass per vertex — avoids the redundant-fbm risk. */
export function sampleAll(unit: Vector3, f: WorldFields, p: WorldGenParams): VertexSample {
  // continental fbm, domain-scaled ~1.5 for planet-sized features
  const c = f.terrain.fbm(unit.x * 1.5, unit.y * 1.5, unit.z * 1.5, p.octaves, p.persistence, p.lacunarity);
  const elevation01 = c * 0.5 + 0.5; // [0,1]
  const sea = p.seaLevel;
  const land = smoothstep(sea, sea + 0.06, elevation01);
  // base continental displacement; ocean floor gentler so it doesn't waste vertices under water
  const base = (elevation01 >= sea)
    ? (elevation01 - sea) * p.noiseStrength
    : (elevation01 - sea) * p.noiseStrength * 0.25;
  // ridged mountains masked to land, weighted by how high the land is
  const r = f.terrain.ridged(unit.x * 2.3 + 11.7, unit.y * 2.3 - 5.1, unit.z * 2.3 + 3.9,
    Math.min(8, p.octaves + 1), p.persistence, p.lacunarity);
  const mountains = r * land * Math.max(0, elevation01 - sea) * p.mountainHeight;
  const heightAboveSea = (base + mountains) * p.radius; // world units
  // temperature: latitude warmth - altitude cooling + bias
  const lat01 = 1 - Math.abs(unit.y);
  const temp = clamp01(smoothstep(0.0, 0.85, lat01) - Math.max(0, heightAboveSea / p.radius) * 1.6 + p.temperatureBias);
  // moisture: decorrelated low-freq field + mild equatorial wetness + bias
  const mn = f.moisture.fbm(unit.x * 1.1 + 100, unit.y * 1.1 - 100, unit.z * 1.1 + 50,
    Math.max(3, p.octaves - 1), p.persistence, p.lacunarity);
  const moist = clamp01((mn * 0.5 + 0.5) + 0.15 * smoothstep(0.2, 0.7, lat01) + p.moistureBias);
  return { elevation01, heightAboveSea, temp, moist };
}

/** Signed world-unit displacement. pos = unit * (radius + height(unit,params)). */
export function height(unit: Vector3, params: WorldGenParams): number {
  return sampleAll(unit, makeFields(params.seed), params).heightAboveSea;
}
