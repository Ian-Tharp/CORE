import { Injectable } from '@angular/core';
import * as THREE from 'three';
import { Subject } from 'rxjs';
import type { EngineService } from './engine.service';
import type { HexWorldSnapshot } from './project.service';

export interface HexWorldConfig {
  radius: number; // hex radius in world units
  gridWidth: number;
  gridHeight: number;
  elevation: number; // tile height
}

export type TerrainState = 'plain' | 'water' | 'mountain';
export type BiomeState = 'none' | 'forest' | 'desert' | 'tundra';
export type ResourceState = 'none' | 'node';

@Injectable({ providedIn: 'root' })
export class HexWorldService {
  private engine?: EngineService;
  private instancedMesh?: THREE.InstancedMesh;
  private colorAttribute?: THREE.InstancedBufferAttribute;
  private hexes: Array<{ q: number; r: number; x: number; z: number }> = [];
  private terrainStates: Map<number, TerrainState> = new Map();
  private biomeStates: Map<number, BiomeState> = new Map();
  private resourceStates: Map<number, ResourceState> = new Map();
  private currentConfig?: HexWorldConfig;
  private activeLayer: 'terrain' | 'biome' | 'resources' = 'terrain';
  private terrainTool: TerrainState = 'plain';
  private biomeTool: BiomeState = 'forest';
  private resourceTool: ResourceState | 'erase' = 'node';
  private terrainVisible = true;
  private biomeVisible = true;
  private resourcesVisible = true;
  private brushRadius = 1;
  private painting = false;
  private hoveredIndex: number = -1;
  private hovered$ = new Subject<{ index: number; q: number; r: number; s: number; x: number; y: number; z: number; terrain: TerrainState; biome: BiomeState; resource: ResourceState } | null>();
  private contextHandler?: (ctx: { index: number; q: number; r: number; x: number; z: number; screen: { x: number; y: number } }) => void;
  private hoverOutline?: THREE.Line;
  private selectedOutline?: THREE.Line;
  private selectedIndex: number = -1;
  private selected$ = new Subject<{ index: number; q: number; r: number; s: number; x: number; y: number; z: number; terrain: TerrainState; biome: BiomeState; resource: ResourceState } | null>();
  private editMode = true;
  private outlineMesh?: THREE.InstancedMesh;
  private outlinesVisible = false;
  private rng?: () => number;

  constructor() {}

  initialize(engine: EngineService): void {
    this.engine = engine;
    this.engine.onHover((hits) => this.onHover(hits));
    this.engine.onPointerDown((hits, button) => {
      if (button === 0) { // left
        if (this.editMode) {
          this.painting = true;
          this.applyPaintFromHits(hits);
        } else {
          this.selectFromHits(hits);
        }
      }
    });
    this.engine.onPointerUp((_hits, _button) => {
      this.painting = false;
    });
  }

  /**
   * Set a deterministic random seed for procedural generation. Pass null to clear and use Math.random.
   */
  setRandomSeed(seed: number | string | null): void {
    if (seed === null || seed === undefined || seed === '') {
      this.rng = undefined;
      return;
    }
    const n = typeof seed === 'number' ? seed : this.hashStringToInt(seed);
    this.rng = this.mulberry32(n >>> 0);
  }

  private mulberry32(a: number): () => number {
    return function() {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  private hashStringToInt(str: string): number {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  createHexPlane(config: HexWorldConfig): void {
    if (!this.engine) throw new Error('Engine not initialized');

    // Dispose existing
    if (this.instancedMesh) {
      this.engine.remove(this.instancedMesh);
      this.instancedMesh.geometry.dispose();
      (this.instancedMesh.material as THREE.Material).dispose();
      this.instancedMesh = undefined;
    }

    const size = config.radius;
    const hexGrid: Array<{ q: number; r: number; x: number; z: number }> = [];
    const width = config.gridWidth;
    const height = config.gridHeight;
    const radiusCells = Math.floor(Math.min(width, height) / 2);

    const hexToWorld = (q: number, r: number): { x: number; z: number } => {
      const x = size * 1.5 * q;
      const z = size * Math.sqrt(3) * (r + q / 2);
      return { x, z };
    };

    // Build a hexagon-shaped grid (axial coordinates) instead of a parallelogram
    for (let q = -radiusCells; q <= radiusCells; q++) {
      const r1 = Math.max(-radiusCells, -q - radiusCells);
      const r2 = Math.min(radiusCells, -q + radiusCells);
      for (let r = r1; r <= r2; r++) {
        const { x, z } = hexToWorld(q, r);
        hexGrid.push({ q, r, x, z });
      }
    }
    this.hexes = hexGrid;
    this.currentConfig = { ...config };

    const geometry = new THREE.CylinderGeometry(config.radius, config.radius, config.elevation, 6, 1, false);
    geometry.translate(0, -config.elevation / 2, 0);
    const material = new THREE.MeshStandardMaterial({
      color: 0x49c0bb,
      metalness: 0.05,
      roughness: 0.8,
      emissive: 0x0a1216,
      emissiveIntensity: 0.35,
      flatShading: true,
      vertexColors: true
    });

    const count = hexGrid.length;
    const mesh = new THREE.InstancedMesh(geometry, material, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // Defensive: avoid incorrect culling on some drivers/sizes
    mesh.frustumCulled = false;

    const colorArray = new Float32Array(count * 3);
    this.colorAttribute = new THREE.InstancedBufferAttribute(colorArray, 3);
    // Attach per-instance colors (supported in three@0.165 via mesh.instanceColor)
    mesh.instanceColor = this.colorAttribute as unknown as THREE.InstancedBufferAttribute;

    const tempObj = new THREE.Object3D();

    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const hex = hexGrid[i];
      tempObj.position.set(hex.x, 0, hex.z);
      tempObj.updateMatrix();
      mesh.setMatrixAt(i, tempObj.matrix);
      // initialize layer states
      this.terrainStates.set(i, 'plain');
      this.biomeStates.set(i, 'none');
      this.resourceStates.set(i, 'none');
      this.updateDisplayColor(i);
      if (hex.x < minX) minX = hex.x;
      if (hex.x > maxX) maxX = hex.x;
      if (hex.z < minZ) minZ = hex.z;
      if (hex.z > maxZ) maxZ = hex.z;
    }
    mesh.instanceMatrix.needsUpdate = true;
    (mesh.instanceColor as any).needsUpdate = true;

    this.instancedMesh = mesh;
    this.engine.add(mesh, true);
    this.engine.onClick((hits) => this.onClick(hits));
    this.engine.onContextClick((hits) => this.onContext(hits));

    // Hover and Selected hexagonal outlines (LineLoop)
    const makeHexLine = (r: number, color: number, alpha = 1.0): THREE.Line => {
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i < 6; i++) {
        const angle = Math.PI / 3 * i; // flat-top
        const x = r * Math.cos(angle + Math.PI / 6);
        const z = r * Math.sin(angle + Math.PI / 6);
        pts.push(new THREE.Vector3(x, 0.025, z));
      }
      pts.push(pts[0].clone());
      const geom = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({ color, transparent: alpha < 1, opacity: alpha });
      const line = new THREE.Line(geom, mat);
      line.visible = false;
      return line;
    };
    this.hoverOutline = makeHexLine(size * 1.03, 0x98fff0, 0.9);
    this.selectedOutline = makeHexLine(size * 1.06, 0x00ffd5, 1.0);
    this.engine.add(this.hoverOutline);
    this.engine.add(this.selectedOutline);

    // Optional per-cell thin outline grid — disabled by default to avoid visual noise
    if (this.outlinesVisible) {
      const outlineGeom = new THREE.RingGeometry(size * 0.995, size * 1.01, 6, 1);
      outlineGeom.rotateX(-Math.PI / 2);
      const outlineMat = new THREE.MeshBasicMaterial({ color: 0x0e1b21, transparent: true, opacity: 0.12, side: THREE.DoubleSide });
      const outline = new THREE.InstancedMesh(outlineGeom, outlineMat, count);
      outline.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      for (let i = 0; i < count; i++) {
        const hex = hexGrid[i];
        tempObj.position.set(hex.x, 0.021, hex.z);
        tempObj.updateMatrix();
        outline.setMatrixAt(i, tempObj.matrix);
      }
      outline.renderOrder = 2;
      outline.frustumCulled = false;
      this.outlineMesh = outline;
      this.engine.add(outline);
    }

    // Center and fit camera to grid bounds
    const centerX = (minX + maxX) * 0.5;
    const centerZ = (minZ + maxZ) * 0.5;
    const worldWidth = (maxX - minX) + size * 2;
    const worldHeight = (maxZ - minZ) + size * 2;
    this.engine.recenterTo(new THREE.Vector3(centerX, 0, centerZ));
    this.engine.fitToBounds(worldWidth, worldHeight, 1.15);
  }

  private updateDisplayColor(index: number): void {
    if (!this.colorAttribute) return;
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';

    const terrainColor = new THREE.Color(
      terrain === 'water' ? 0x3a8cf0 : terrain === 'mountain' ? 0x99a9b3 : 0x3aa3a0
    );
    let c = terrainColor;
    if (this.biomeVisible && biome !== 'none') {
      const tint = new THREE.Color(
        biome === 'forest' ? 0x25c07a : biome === 'desert' ? 0xd8b14a : 0xa0c8ff
      );
      c = c.clone().lerp(tint, 0.35);
    }
    if (this.resourcesVisible && resource === 'node') {
      const res = new THREE.Color(0x9ab3ff);
      c = c.clone().lerp(res, 0.45);
    }
    if (!this.terrainVisible) {
      // If terrain hidden, keep a faint neutral grid for orientation
      const base = new THREE.Color(0x0b141a);
      c = base.clone().lerp(c, 0.35);
    }
    this.colorAttribute.setXYZ(index, c.r, c.g, c.b);
  }

  private onClick(hits: THREE.Intersection[]): void {
    if (!this.instancedMesh || hits.length === 0) return;
    this.applyPaintFromHits(hits);
  }

  private onContext(hits: THREE.Intersection[]): void {
    if (this.editMode) return; // context only in non-edit mode
    if (!this.instancedMesh || hits.length === 0) return;
    const hit = hits.find((h) => h.object === this.instancedMesh);
    if (!hit) return;
    const index = (hit.instanceId ?? -1) as number;
    if (index < 0) return;
    const hex = this.hexes[index];
    const screen = this.engine?.worldToCanvas(hit.point.clone()) ?? null;
    if (screen && this.contextHandler) {
      this.contextHandler({ index, q: hex.q, r: hex.r, x: hex.x, z: hex.z, screen });
    }
  }

  private onHover(hits: THREE.Intersection[]): void {
    // hover highlight
    if (!this.instancedMesh) return;
    const hit = hits.find((h) => h.object === this.instancedMesh);
    const newIndex = hit && (hit.instanceId ?? -1) >= 0 ? (hit.instanceId as number) : -1;
    if (newIndex !== this.hoveredIndex) {
      // restore previous
      if (this.hoveredIndex >= 0) {
        this.updateDisplayColor(this.hoveredIndex);
      }
      this.hoveredIndex = newIndex;
      // set highlight
      if (this.hoveredIndex >= 0) {
        const highlight = new THREE.Color(0x98fff0);
        this.colorAttribute?.setXYZ(this.hoveredIndex, highlight.r, highlight.g, highlight.b);
        const h = this.hexes[this.hoveredIndex];
        const s = - (h.q + h.r);
        this.hovered$.next({
          index: this.hoveredIndex,
          q: h.q,
          r: h.r,
          s,
          x: h.x,
          y: 0,
          z: h.z,
          terrain: this.terrainStates.get(this.hoveredIndex) ?? 'plain',
          biome: this.biomeStates.get(this.hoveredIndex) ?? 'none',
          resource: this.resourceStates.get(this.hoveredIndex) ?? 'none'
        });
        if (this.hoverOutline) {
          this.hoverOutline.position.set(h.x, 0.0, h.z);
          this.hoverOutline.visible = true;
        }
      } else {
        this.hovered$.next(null);
        if (this.hoverOutline) this.hoverOutline.visible = false;
      }
      (this.instancedMesh.instanceColor as any).needsUpdate = true;
    }
    // paint if dragging
    if (this.painting) {
      this.applyPaintFromHits(hits);
    }
  }

  private axialDistance(a: { q: number; r: number }, b: { q: number; r: number }): number {
    const dq = a.q - b.q;
    const dr = a.r - b.r;
    const ds = (-a.q - a.r) - (-b.q - b.r);
    return Math.max(Math.abs(dq), Math.abs(dr), Math.abs(ds));
  }

  private applyPaintFromHits(hits: THREE.Intersection[]): void {
    if (!this.instancedMesh) return;
    const hit = hits.find((h) => h.object === this.instancedMesh);
    if (!hit || (hit.instanceId ?? -1) < 0) return;
    const centerIndex = hit.instanceId as number;
    const center = this.hexes[centerIndex];
    for (let i = 0; i < this.hexes.length; i++) {
      if (this.axialDistance(this.hexes[i], center) <= this.brushRadius) {
        if (this.activeLayer === 'terrain') {
          this.terrainStates.set(i, this.terrainTool);
        } else if (this.activeLayer === 'biome') {
          this.biomeStates.set(i, this.biomeTool);
        } else {
          if (this.resourceTool === 'erase') this.resourceStates.set(i, 'none');
          else this.resourceStates.set(i, this.resourceTool);
        }
        this.updateDisplayColor(i);
      }
    }
    (this.instancedMesh.instanceColor as any).needsUpdate = true;
    // Update hover payload so panel reflects the tile's new values while painting
    if (this.hoveredIndex >= 0) {
      const h = this.hexes[this.hoveredIndex];
      const s = - (h.q + h.r);
      this.hovered$.next({
        index: this.hoveredIndex,
        q: h.q,
        r: h.r,
        s,
        x: h.x,
        y: 0,
        z: h.z,
        terrain: this.terrainStates.get(this.hoveredIndex) ?? 'plain',
        biome: this.biomeStates.get(this.hoveredIndex) ?? 'none',
        resource: this.resourceStates.get(this.hoveredIndex) ?? 'none'
      });
    }
  }

  setActiveLayer(layer: 'terrain' | 'biome' | 'resources'): void { this.activeLayer = layer; }
  setTerrainTool(tool: TerrainState): void { this.terrainTool = tool; }
  setBiomeTool(tool: BiomeState): void { this.biomeTool = tool; }
  setResourceTool(tool: ResourceState | 'erase'): void { this.resourceTool = tool; }
  setLayerVisibility(layer: 'terrain' | 'biome' | 'resources', visible: boolean): void {
    if (layer === 'terrain') this.terrainVisible = visible; else if (layer === 'biome') this.biomeVisible = visible; else this.resourcesVisible = visible;
    for (let i = 0; i < this.hexes.length; i++) this.updateDisplayColor(i);
    if (this.instancedMesh) (this.instancedMesh.instanceColor as any).needsUpdate = true;
  }

  setOutlinesVisible(visible: boolean): void {
    this.outlinesVisible = visible;
    if (this.outlineMesh) this.outlineMesh.visible = visible;
  }

  setBrushRadius(radius: number): void {
    this.brushRadius = Math.max(0, Math.floor(radius));
  }

  onHoverChanged() {
    return this.hovered$.asObservable();
  }

  onSelectedChanged() {
    return this.selected$.asObservable();
  }

  onTileContext(handler: (ctx: { index: number; q: number; r: number; x: number; z: number; screen: { x: number; y: number } }) => void): void {
    this.contextHandler = handler;
  }

  setEditMode(isEdit: boolean): void { this.editMode = isEdit; }

  snapshot(name: string): Omit<HexWorldSnapshot, 'id' | 'createdAt'> {
    const terrain: Array<{ index: number; state: TerrainState }> = [];
    const biome: Array<{ index: number; state: BiomeState }> = [];
    const resources: Array<{ index: number; state: ResourceState }> = [];
    for (let i = 0; i < this.hexes.length; i++) {
      const t = this.terrainStates.get(i) ?? 'plain';
      const b = this.biomeStates.get(i) ?? 'none';
      const r = this.resourceStates.get(i) ?? 'none';
      if (t !== 'plain') terrain.push({ index: i, state: t });
      if (b !== 'none') biome.push({ index: i, state: b });
      if (r !== 'none') resources.push({ index: i, state: r });
    }
    return { name, config: this.currentConfig!, layers: { terrain, biome, resources } } as any;
  }

  restore(name: string, payload: { config: HexWorldConfig; tiles?: Array<{ index: number; state: any }>; layers?: { terrain: Array<{ index: number; state: TerrainState }>; biome: Array<{ index: number; state: BiomeState }>; resources: Array<{ index: number; state: ResourceState }> } }): void {
    this.createHexPlane(payload.config);
    if (payload.layers) {
      for (const t of payload.layers.terrain) this.terrainStates.set(t.index, t.state);
      for (const b of payload.layers.biome) this.biomeStates.set(b.index, b.state);
      for (const r of payload.layers.resources) this.resourceStates.set(r.index, r.state);
      for (let i = 0; i < this.hexes.length; i++) this.updateDisplayColor(i);
    } else if (payload.tiles) {
      // backward compatibility: map legacy life/resource states to new layers
      for (const t of payload.tiles) {
        if (t.state === 'life') this.biomeStates.set(t.index, 'forest');
        else if (t.state === 'resource') this.resourceStates.set(t.index, 'node');
      }
      for (let i = 0; i < this.hexes.length; i++) this.updateDisplayColor(i);
    }
    if (this.instancedMesh) (this.instancedMesh.instanceColor as any).needsUpdate = true;
  }

  randomize(): void {
    if (!this.instancedMesh) return;
    const rand = this.rng ?? Math.random;
    for (let i = 0; i < this.hexes.length; i++) {
      // terrain
      const rT = rand();
      const t: TerrainState = rT < 0.08 ? 'water' : rT < 0.14 ? 'mountain' : 'plain';
      this.terrainStates.set(i, t);
      // biome
      const rB = rand();
      const b: BiomeState = rB < 0.1 ? 'forest' : rB < 0.14 ? 'tundra' : rB < 0.18 ? 'desert' : 'none';
      this.biomeStates.set(i, b);
      // resources
      const rR = rand();
      const res: ResourceState = rR < 0.05 ? 'node' : 'none';
      this.resourceStates.set(i, res);
      this.updateDisplayColor(i);
    }
    (this.instancedMesh.instanceColor as any).needsUpdate = true;
  }

  getCurrentConfig(): HexWorldConfig | undefined { return this.currentConfig ? { ...this.currentConfig } : undefined; }

  clear(): void {
    if (!this.instancedMesh) return;
    for (let i = 0; i < this.hexes.length; i++) {
      this.terrainStates.set(i, 'plain');
      this.biomeStates.set(i, 'none');
      this.resourceStates.set(i, 'none');
      this.updateDisplayColor(i);
    }
    (this.instancedMesh.instanceColor as any).needsUpdate = true;
  }

  private selectFromHits(hits: THREE.Intersection[]): void {
    if (!this.instancedMesh) return;
    const hit = hits.find((h) => h.object === this.instancedMesh);
    if (!hit || (hit.instanceId ?? -1) < 0) { this.selected$.next(null); return; }
    const index = hit.instanceId as number;
    const tile = this.hexes[index];
    const s = -(tile.q + tile.r);
    const payload = {
      index,
      q: tile.q,
      r: tile.r,
      s,
      x: tile.x,
      y: 0,
      z: tile.z,
      terrain: this.terrainStates.get(index) ?? 'plain',
      biome: this.biomeStates.get(index) ?? 'none',
      resource: this.resourceStates.get(index) ?? 'none',
    };
    this.selected$.next(payload);
    // Persist selected outline
    this.selectedIndex = index;
    if (this.selectedOutline) {
      this.selectedOutline.position.set(tile.x, 0.0, tile.z);
      this.selectedOutline.visible = true;
    }
  }
}


