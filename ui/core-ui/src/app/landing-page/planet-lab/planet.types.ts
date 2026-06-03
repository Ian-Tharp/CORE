// planet.types.ts — internal userData stamped on the planet THREE.Group.
export interface PlanetGroupUserData {
  kind: 'core-planet';
  paramsHash: string;
  topoSig: string;
  tier: string;
  vertexCount: number;
  /** Pristine unit-sphere positions (== base normals); nulled on dispose. */
  basePositions: Float32Array | null;
  seed: string;
}
