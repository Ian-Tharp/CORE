// noise.ts — vendored seeded 3D simplex, ZERO npm deps. Allocation-free hot path.
export function hashStringToInt(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const GRAD3 = new Float32Array([
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0, 1, 0, 1, -1, 0, 1,
  1, 0, -1, -1, 0, -1, 0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
]);
const F3 = 1 / 3, G3 = 1 / 6;

export class SimplexNoise3 {
  private readonly perm = new Uint8Array(512);
  private readonly permMod12 = new Uint8Array(512);
  constructor(seed: string) {
    const rng = mulberry32(hashStringToInt(seed));
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) { p[i] = i; }
    for (let i = 255; i > 0; i--) { const j = (rng() * (i + 1)) | 0; const t = p[i]; p[i] = p[j]; p[j] = t; }
    for (let i = 0; i < 512; i++) { this.perm[i] = p[i & 255]; this.permMod12[i] = this.perm[i] % 12; }
  }
  noise3(xin: number, yin: number, zin: number): number {
    const perm = this.perm, pm12 = this.permMod12;
    const s = (xin + yin + zin) * F3;
    const i = Math.floor(xin + s), j = Math.floor(yin + s), k = Math.floor(zin + s);
    const t = (i + j + k) * G3;
    const x0 = xin - (i - t), y0 = yin - (j - t), z0 = zin - (k - t);
    let i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
      if (y0 >= z0)      { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
      else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
      else               { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
    } else {
      if (y0 < z0)       { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
      else if (x0 < z0)  { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
      else               { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    }
    const x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
    const x2 = x0 - i2 + 2 * G3, y2 = y0 - j2 + 2 * G3, z2 = z0 - k2 + 2 * G3;
    const x3 = x0 - 1 + 3 * G3, y3 = y0 - 1 + 3 * G3, z3 = z0 - 1 + 3 * G3;
    const ii = i & 255, jj = j & 255, kk = k & 255;
    let n = 0;
    n += this.corner(pm12[ii + perm[jj + perm[kk]]], x0, y0, z0);
    n += this.corner(pm12[ii + i1 + perm[jj + j1 + perm[kk + k1]]], x1, y1, z1);
    n += this.corner(pm12[ii + i2 + perm[jj + j2 + perm[kk + k2]]], x2, y2, z2);
    n += this.corner(pm12[ii + 1 + perm[jj + 1 + perm[kk + 1]]], x3, y3, z3);
    return 32 * n; // ~[-1,1]
  }
  private corner(gi: number, x: number, y: number, z: number): number {
    let t = 0.6 - x * x - y * y - z * z; if (t < 0) { return 0; }
    const g = gi * 3; t *= t;
    return t * t * (GRAD3[g] * x + GRAD3[g + 1] * y + GRAD3[g + 2] * z);
  }
  fbm(x: number, y: number, z: number, octaves: number, persistence: number, lacunarity: number): number {
    let freq = 1, amp = 1, sum = 0, norm = 0;
    const oct = Math.max(1, Math.min(8, octaves | 0));
    for (let o = 0; o < oct; o++) { sum += amp * this.noise3(x * freq, y * freq, z * freq); norm += amp; amp *= persistence; freq *= lacunarity; }
    return sum / norm; // ~[-1,1]
  }
  ridged(x: number, y: number, z: number, octaves: number, persistence: number, lacunarity: number): number {
    let freq = 1, amp = 1, sum = 0, norm = 0, prev = 1;
    const oct = Math.max(1, Math.min(8, octaves | 0));
    for (let o = 0; o < oct; o++) {
      let nn = 1 - Math.abs(this.noise3(x * freq, y * freq, z * freq)); nn *= nn; nn *= prev; prev = nn;
      sum += nn * amp; norm += amp; amp *= persistence; freq *= lacunarity;
    }
    return sum / norm; // [0,1]
  }
}

export interface WorldFields { terrain: SimplexNoise3; moisture: SimplexNoise3; seed: string; }
// LRU-capped seed cache (fixes unbounded-growth risk during live seed scrubbing).
const _cache = new Map<string, WorldFields>();
const _CACHE_MAX = 8;
export function makeFields(seed: string): WorldFields {
  let f = _cache.get(seed);
  if (f) { _cache.delete(seed); _cache.set(seed, f); return f; } // LRU touch
  f = { terrain: new SimplexNoise3(seed), moisture: new SimplexNoise3(seed + '~moisture'), seed };
  _cache.set(seed, f);
  if (_cache.size > _CACHE_MAX) { _cache.delete(_cache.keys().next().value as string); }
  return f;
}
export function evictFields(seed: string): void { _cache.delete(seed); }
