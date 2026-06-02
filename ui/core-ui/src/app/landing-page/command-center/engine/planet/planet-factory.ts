// planet-factory.ts — the public API: buildPlanet / updatePlanet / disposePlanet
// + attachAutoRotate. Camera/renderer/scene-agnostic: produces a THREE.Group you
// drop into any scene.
import * as THREE from 'three';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import {
  resolveParams, hashParams, topoSig, detailFor,
  type WorldGenParams, type PlanetOptions,
} from './world-gen-params';
import { makeFields, evictFields } from './noise';
import { sampleAll } from './height-field';
import { linearPalette, biomeColor } from './biome';
import {
  makeSurfaceMaterial, makeWaterMaterial, makeAtmosphereMaterial, makeCloudMaterial,
} from './planet-materials';
import type { PlanetGroupUserData } from './planet.types';

const SHELL_NAMES = ['planet-water', 'planet-atmosphere', 'planet-clouds'] as const;

/** Build a planet THREE.Group from params. Synchronous (CPU noise runs once). */
export function buildPlanet(input: WorldGenParams, opts: PlanetOptions = {}): THREE.Group {
  const p = resolveParams(input);
  const tier = opts.tier ?? 'standard';
  const detail = detailFor(tier);

  const group = new THREE.Group();
  group.name = 'planet';

  // IcosahedronGeometry is NON-indexed with per-face flat normals; strip normals/uv
  // and mergeVertices to get a shared-vertex (indexed) sphere → ~10*4^detail+2 verts
  // and smooth normals after computeVertexNormals().
  const raw = new THREE.IcosahedronGeometry(1, detail);
  raw.deleteAttribute('normal');
  raw.deleteAttribute('uv');
  const geo = mergeVertices(raw);
  raw.dispose();

  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  const basePositions = new Float32Array(pos.array as Float32Array); // pristine unit corners
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3));

  const surface = new THREE.Mesh(geo, makeSurfaceMaterial(p));
  surface.name = 'planet-surface';
  surface.castShadow = surface.receiveShadow = true;
  group.add(surface);

  const ud: PlanetGroupUserData = {
    kind: 'core-planet', paramsHash: hashParams(p), topoSig: topoSig(p, opts), tier,
    vertexCount: pos.count, basePositions, seed: p.seed,
  };
  group.userData = ud;

  displaceSurface(group, p);
  rebuildShells(group, p, opts);
  return group;
}

/** Re-mesh a planet for new params. No-op if unchanged; rebuild only if topology changed. */
export function updatePlanet(group: THREE.Group, params: WorldGenParams, opts: PlanetOptions = {}): void {
  const ud = group.userData as PlanetGroupUserData | undefined;
  if (ud?.kind !== 'core-planet') { return; }
  const p = resolveParams(params);
  const newHash = hashParams(p);
  if (ud.paramsHash === newHash) { return; }

  if (ud.topoSig !== topoSig(p, opts)) {
    // radius/tier changed → full rebuild, preserving the group's parent + transform
    const parent = group.parent;
    const fresh = buildPlanet(params, opts);
    disposeChildren(group);
    group.clear();
    while (fresh.children.length) { group.add(fresh.children[0]); }
    group.userData = fresh.userData;
    if (parent && !group.parent) { parent.add(group); }
    return;
  }

  displaceSurface(group, p);  // in place, from cached basePositions
  rebuildShells(group, p, opts);
  ud.paramsHash = newHash;
}

/** Tear down a planet group: dispose all geometry/materials, evict the noise seed. */
export function disposePlanet(group: THREE.Group): void {
  const ud = group.userData as PlanetGroupUserData | undefined;
  disposeChildren(group);
  group.removeFromParent();
  group.clear();
  if (ud) {
    ud.basePositions = null;
    if (ud.seed) { evictFields(ud.seed); }
  }
}

/**
 * Opt-in, a11y-gated auto-rotation. `host.onBeforeRender(cb)` must return a disposer.
 * Returns a disposer that unregisters BOTH the frame callback and the media listener.
 */
export function attachAutoRotate(
  group: THREE.Group,
  host: { onBeforeRender(cb: () => void): () => void },
  radPerSec = 0.15,
): () => void {
  const mq = matchMedia('(prefers-reduced-motion: reduce)');
  let remove: (() => void) | null = null;
  let last = performance.now();
  const apply = () => {
    if (mq.matches) { remove?.(); remove = null; }
    else if (!remove) {
      last = performance.now();
      remove = host.onBeforeRender(() => {
        const now = performance.now();
        group.rotation.y += radPerSec * Math.min(0.1, (now - last) / 1000);
        last = now;
      });
    }
  };
  apply();
  mq.addEventListener('change', apply);
  return () => { remove?.(); remove = null; mq.removeEventListener('change', apply); };
}

// ── internals ────────────────────────────────────────────────────────────────

function displaceSurface(group: THREE.Group, p: WorldGenParams): void {
  const surface = group.getObjectByName('planet-surface') as THREE.Mesh;
  const geo = surface.geometry as THREE.BufferGeometry;
  const ud = group.userData as PlanetGroupUserData;
  const base = ud.basePositions!;
  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  const col = geo.getAttribute('color') as THREE.BufferAttribute;
  const fields = makeFields(p.seed);
  const pal = linearPalette(p);
  const dir = new THREE.Vector3();
  const c = new THREE.Color();
  const outArr = pos.array as Float32Array;
  const colArr = col.array as Float32Array;

  for (let i = 0; i < base.length; i += 3) {
    dir.set(base[i], base[i + 1], base[i + 2]); // already unit length
    const s = sampleAll(dir, fields, p);
    const r = p.radius + s.heightAboveSea;
    outArr[i] = dir.x * r; outArr[i + 1] = dir.y * r; outArr[i + 2] = dir.z * r;
    // slope01 = 0 for v1 (single pass); biome by elevation/temperature/moisture.
    biomeColor(s.heightAboveSea, s.temp, s.moist, 0, pal, p, c);
    colArr[i] = c.r; colArr[i + 1] = c.g; colArr[i + 2] = c.b;
  }
  pos.needsUpdate = true;
  col.needsUpdate = true;
  geo.computeVertexNormals();   // smooth normals after displacement
  geo.computeBoundingSphere();
}

/** Remove existing shells, re-add per has* flags (keeps colours/radii in sync). */
function rebuildShells(group: THREE.Group, p: WorldGenParams, opts: PlanetOptions): void {
  const ud = group.userData as PlanetGroupUserData;
  for (const name of SHELL_NAMES) {
    const ex = group.getObjectByName(name) as THREE.Mesh | undefined;
    if (ex) {
      group.remove(ex);
      (ex.geometry as THREE.BufferGeometry).dispose();
      disposeMat(ex.material as THREE.Material);
    }
  }
  const detail = detailFor((opts.tier ?? ud.tier) as any);
  if (p.hasWater) { group.add(buildWater(p, detail)); }
  if (p.hasAtmosphere) { group.add(buildAtmosphere(p)); }
  if (p.hasClouds) { group.add(buildClouds(p)); }
}

function buildWater(p: WorldGenParams, detail: number): THREE.Mesh {
  const geo = new THREE.IcosahedronGeometry(p.radius, Math.max(1, detail - 1));
  const mesh = new THREE.Mesh(geo, makeWaterMaterial(p));
  mesh.name = 'planet-water';
  mesh.renderOrder = 1;
  return mesh;
}

function buildAtmosphere(p: WorldGenParams): THREE.Mesh {
  const geo = new THREE.IcosahedronGeometry(p.radius * 1.16, 4);
  const mesh = new THREE.Mesh(geo, makeAtmosphereMaterial(p));
  mesh.name = 'planet-atmosphere';
  mesh.renderOrder = 2;
  return mesh;
}

function buildClouds(p: WorldGenParams): THREE.Mesh {
  const r = p.radius * 1.04;
  const geo = new THREE.IcosahedronGeometry(r, 4);
  const pos = geo.getAttribute('position') as THREE.BufferAttribute;
  const aCloud = new Float32Array(pos.count);
  const fields = makeFields(p.seed);
  const inv = 1 / r;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i) * inv, y = pos.getY(i) * inv, z = pos.getZ(i) * inv;
    const n = fields.moisture.fbm(x * 2.2 + 7, y * 2.2 - 3, z * 2.2 + 1, 4, 0.5, 2.2) * 0.5 + 0.5;
    aCloud[i] = Math.max(0, n - 0.55) / 0.45; // sparse puffs
  }
  geo.setAttribute('aCloud', new THREE.BufferAttribute(aCloud, 1));
  const mesh = new THREE.Mesh(geo, makeCloudMaterial(p));
  mesh.name = 'planet-clouds';
  mesh.renderOrder = 2;
  return mesh;
}

function disposeChildren(group: THREE.Group): void {
  group.traverse((o: THREE.Object3D) => {
    const m = o as THREE.Mesh;
    if (m.geometry) { m.geometry.dispose(); }
    const mat = m.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(mat)) { mat.forEach(disposeMat); }
    else if (mat) { disposeMat(mat); }
  });
}

function disposeMat(m: THREE.Material): void {
  const uniforms = (m as THREE.ShaderMaterial).uniforms;
  if (uniforms) {
    for (const u of Object.values(uniforms)) {
      const t = (u as { value?: { isTexture?: boolean; dispose?: () => void } })?.value;
      if (t?.isTexture) { t.dispose?.(); }
    }
  }
  m.dispose();
}
