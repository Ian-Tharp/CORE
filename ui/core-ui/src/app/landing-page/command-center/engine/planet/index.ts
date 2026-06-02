// Public surface of the procedural planet module.
export { buildPlanet, updatePlanet, disposePlanet, attachAutoRotate } from './planet-factory';
export {
  DEFAULT_PLANET_PARAMS, BIOME_PRESETS, resolveParams, clampParams, detailFor, vertexCount,
} from './world-gen-params';
export type {
  WorldGenParams, BiomeArchetype, PlanetPalette, LodTier, PlanetOptions,
} from './world-gen-params';
