// planet-descriptor.ts — turn WorldGenParams into free-text the existing AI
// prompt fields can consume (lore `context`, image `prompt`). Pure, no THREE.
import { WorldGenParams } from './world-gen-params';

const qual = (v: number, lo: string, mid: string, hi: string) =>
  (v <= -0.25 ? lo : v >= 0.25 ? hi : mid);
const band = (v: number, lo: string, mid: string, hi: string) =>
  (v < 0.4 ? lo : v > 0.6 ? hi : mid);

/** One-line FACTUAL grounding string for lore generation (precise + qualitative). */
export function planetLoreContext(p: WorldGenParams): string {
  const climate = qual(p.temperatureBias, 'cold', 'temperate', 'hot');
  const wet = qual(p.moistureBias, 'arid', 'balanced', 'humid');
  const sea = band(p.seaLevel, 'low seas', 'moderate seas', 'high seas');
  const mts = band(p.mountainHeight, 'gentle relief', 'rolling mountains', 'towering mountains');
  const features = [
    p.hasWater && 'liquid water',
    p.hasAtmosphere && 'an atmosphere',
    p.hasClouds && 'cloud cover'
  ].filter(Boolean).join(', ');
  return [
    `biome archetype: ${p.biomeArchetype}`,
    `${climate} climate`,
    `${wet} moisture`,
    sea,
    mts,
    features && `features: ${features}`,
    `palette — water ${p.palette.water}, lowlands ${p.palette.grass}, highlands ${p.palette.rock}, peaks ${p.palette.snow}`
  ].filter(Boolean).join('; ');
}

/** Painterly subject+palette clause for IMAGE prompts (caller appends the quality suffix). */
export function planetArtClause(p: WorldGenParams): string {
  const climate = qual(p.temperatureBias, 'frozen', 'temperate', 'scorching');
  const wet = qual(p.moistureBias, 'parched', '', 'lush');
  const tone = [climate, wet].filter(Boolean).join(', ');
  return `A ${p.biomeArchetype} world — ${tone}. `
    + `Color palette: ocean ${p.palette.water}, lowlands ${p.palette.grass}, `
    + `highlands ${p.palette.rock}, peaks ${p.palette.snow}.`;
}
