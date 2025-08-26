import { Injectable } from '@angular/core';
import * as THREE from 'three';
import { Subject } from 'rxjs';
import type { EngineService } from './engine.service';
import type { TileGridSnapshot } from './project.service';

export interface TileGridConfig {
  cellRadius: number; // octagon radius in world units
  gridWidth: number;
  gridHeight: number;
  elevation: number; // tile height
}

export type TerrainState = 'plain' | 'water' | 'mountain';
export type BiomeState = 'none' | 'forest' | 'desert' | 'tundra';
export type ResourceState = 'none' | 'node';

@Injectable({ providedIn: 'root' })
export class TileGridService {
  private engine?: EngineService;
  private instancedMesh?: THREE.InstancedMesh;
  private squareMesh?: THREE.InstancedMesh;
  private colorAttribute?: THREE.InstancedBufferAttribute;
  private tiles: Array<{ x: number; y: number; worldX: number; worldZ: number }> = [];
  private terrainStates: Map<number, TerrainState> = new Map();
  private biomeStates: Map<number, BiomeState> = new Map();
  private resourceStates: Map<number, ResourceState> = new Map();
  private currentConfig?: TileGridConfig;
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
  private hovered$ = new Subject<{ 
    index: number; 
    x: number; 
    y: number; 
    worldX: number; 
    worldY: number; 
    worldZ: number; 
    terrain: TerrainState; 
    biome: BiomeState; 
    resource: ResourceState 
  } | null>();
  private contextHandler?: (ctx: { 
    index: number; 
    x: number; 
    y: number; 
    worldX: number; 
    worldZ: number; 
    screen: { x: number; y: number } 
  }) => void;
  private hoverOutline?: THREE.Line;
  private selectedOutline?: THREE.Line;
  private hoverFill?: THREE.Mesh;
  private selectedIndex: number = -1;
  private selected$ = new Subject<{ 
    index: number; 
    x: number; 
    y: number; 
    worldX: number; 
    worldY: number; 
    worldZ: number; 
    terrain: TerrainState; 
    biome: BiomeState; 
    resource: ResourceState 
  } | null>();
  private editMode = true;
  private outlineMesh?: THREE.InstancedMesh;
  private outlinesVisible = true;
  private rng?: () => number;
  private showConnectors = false;

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

  createTileGrid(config: TileGridConfig): void {
    if (!this.engine) throw new Error('Engine not initialized');

    // Dispose existing
    if (this.instancedMesh) {
      this.engine.remove(this.instancedMesh);
      this.instancedMesh.geometry.dispose();
      (this.instancedMesh.material as THREE.Material).dispose();
      this.instancedMesh = undefined;
    }
    if (this.squareMesh) {
      this.engine.remove(this.squareMesh);
      this.squareMesh.geometry.dispose();
      (this.squareMesh.material as THREE.Material).dispose();
      this.squareMesh = undefined;
    }
    if (this.hoverFill) {
      this.engine.remove(this.hoverFill);
      this.hoverFill.geometry.dispose();
      (this.hoverFill.material as THREE.Material).dispose();
      this.hoverFill = undefined;
    }

    const cellRadius = config.cellRadius;
    const tileGrid: Array<{ x: number; y: number; worldX: number; worldZ: number }> = [];
    const width = config.gridWidth;
    const height = config.gridHeight;

    // For a square grid with octagonal cells, use simple spacing
    // Make spacing larger to prevent overlapping
    const spacing = cellRadius * 2.2; // Simple spacing based on cell radius

    // Create a proper 2D cartesian grid
    const halfWidth = Math.floor(width / 2);
    const halfHeight = Math.floor(height / 2);

    for (let x = -halfWidth; x <= halfWidth; x++) {
      for (let y = -halfHeight; y <= halfHeight; y++) {
        const worldX = spacing * x;
        const worldZ = spacing * y; // Using Z for world depth in 3D space
        tileGrid.push({ x, y, worldX, worldZ });
      }
    }

    this.tiles = tileGrid;
    this.currentConfig = { ...config };

    // Create octagonal geometry - flat 2D octagon
    const r = config.cellRadius;
    const octagon = new THREE.Shape();
    for (let i = 0; i < 8; i++) {
      const angle = (Math.PI * 2 / 8) * i + (Math.PI / 8); // flat-top orientation
      const px = r * Math.cos(angle);
      const pz = r * Math.sin(angle);
      if (i === 0) octagon.moveTo(px, pz); else octagon.lineTo(px, pz);
    }
    octagon.closePath();
    const geometry = new THREE.ShapeGeometry(octagon);
    geometry.rotateX(-Math.PI / 2); // Lay flat on XZ plane

    // Material configured for per-instance coloring
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff, // White so instance colors show accurately
      vertexColors: false, // Disable geometry vertex colors; rely on instanceColor
      transparent: false,
      depthWrite: true,
      depthTest: true,
      side: THREE.DoubleSide,
      toneMapped: false
    });

    const count = tileGrid.length;
    const mesh = new THREE.InstancedMesh(geometry, material, count);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    mesh.frustumCulled = false; // Prevent culling issues

    // Enable per-instance coloring - initialize all colors to default
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const idx = i * 3;
      // Temporary default; immediately overridden by updateDisplayColor
      colors[idx] = 0.3;     // R
      colors[idx + 1] = 0.8; // G  
      colors[idx + 2] = 0.3; // B
    }
    
    this.colorAttribute = new THREE.InstancedBufferAttribute(colors, 3);
    this.colorAttribute.setUsage(THREE.DynamicDrawUsage);
    
    // For InstancedMesh, set instanceColor directly on the mesh
    mesh.instanceColor = this.colorAttribute;

    const tempObj = new THREE.Object3D();

    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const tile = tileGrid[i];
      tempObj.position.set(tile.worldX, 0, tile.worldZ);
      tempObj.updateMatrix();
      mesh.setMatrixAt(i, tempObj.matrix);
      
      // Initialize layer states with better default colors
      this.terrainStates.set(i, 'plain');
      this.biomeStates.set(i, 'none');
      this.resourceStates.set(i, 'none');
      this.updateDisplayColor(i);
      
      if (tile.worldX < minX) minX = tile.worldX;
      if (tile.worldX > maxX) maxX = tile.worldX;
      if (tile.worldZ < minZ) minZ = tile.worldZ;
      if (tile.worldZ > maxZ) maxZ = tile.worldZ;
    }
    mesh.instanceMatrix.needsUpdate = true;
    this.colorAttribute.needsUpdate = true;

    this.instancedMesh = mesh;
    this.instancedMesh.renderOrder = 1;
    this.engine.add(mesh, true);
    
    // Force an update of all colors to ensure they display correctly
    for (let i = 0; i < count; i++) {
      this.updateDisplayColor(i);
    }
    this.colorAttribute.needsUpdate = true;
    this.engine.onClick((hits) => this.onClick(hits));
    this.engine.onContextClick((hits) => this.onContext(hits));

    // Create outline helpers for hover and selection
    const makeOctLine = (r: number, color: number, alpha = 1.0): THREE.Line => {
      const points: THREE.Vector3[] = [];
      for (let i = 0; i < 8; i++) {
        const angle = (Math.PI * 2 / 8) * i + (Math.PI / 8);
        const px = r * Math.cos(angle);
        const pz = r * Math.sin(angle);
        points.push(new THREE.Vector3(px, 0.025, pz));
      }
      points.push(points[0].clone()); // Close the loop
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({ color, transparent: alpha < 1, opacity: alpha });
      const line = new THREE.Line(geom, mat);
      line.visible = false;
      return line;
    };

    this.hoverOutline = makeOctLine(cellRadius * 1.03, 0x98fff0, 0.9);
    this.selectedOutline = makeOctLine(cellRadius * 1.06, 0x00ffd5, 1.0);
    this.engine.add(this.hoverOutline);
    this.engine.add(this.selectedOutline);

    // Soft hover fill
    const fillGeom = new THREE.CircleGeometry(cellRadius * 0.88, 8);
    fillGeom.rotateX(-Math.PI / 2);
    fillGeom.rotateY(Math.PI / 8);
    const fillMat = new THREE.MeshBasicMaterial({ 
      color: 0x00ffd5, 
      transparent: true, 
      opacity: 0.16, 
      depthWrite: false 
    });
    const fill = new THREE.Mesh(fillGeom, fillMat);
    fill.visible = false;
    fill.renderOrder = 3;
    this.hoverFill = fill;
    this.engine.add(fill);

    // Optional cell outlines for grid visibility
    if (this.outlinesVisible) {
      const outlineGeom = new THREE.RingGeometry(cellRadius * 0.995, cellRadius * 1.01, 8, 1);
      outlineGeom.rotateX(-Math.PI / 2);
      const outlineMat = new THREE.MeshBasicMaterial({ 
        color: 0x2a3440, 
        transparent: true, 
        opacity: 0.3, 
        side: THREE.DoubleSide 
      });
      const outline = new THREE.InstancedMesh(outlineGeom, outlineMat, count);
      outline.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      for (let i = 0; i < count; i++) {
        const tile = tileGrid[i];
        tempObj.position.set(tile.worldX, 0.021, tile.worldZ);
        tempObj.updateMatrix();
        outline.setMatrixAt(i, tempObj.matrix);
      }
      outline.renderOrder = 2;
      outline.frustumCulled = false;
      this.outlineMesh = outline;
      this.engine.add(outline);
    }

    // Square connectors between octagonal cells (optional)
    if (this.showConnectors) {
      const squareRadius = (cellRadius / Math.SQRT2) * 1.0;
      const squareGeom = new THREE.CylinderGeometry(squareRadius, squareRadius, config.elevation * 0.6, 4, 1, false);
      squareGeom.rotateY(Math.PI / 4);
      squareGeom.translate(0, -config.elevation / 2 - 0.005, 0);
      const squareMat = new THREE.MeshStandardMaterial({
        color: 0x1a2028,
        metalness: 0.02,
        roughness: 0.96,
        transparent: true,
        opacity: 0.08,
        depthWrite: false,
        side: THREE.DoubleSide
      });
      
      const gridSpan = Math.max(width, height);
      const sqCount = gridSpan * gridSpan;
      const sqMesh = new THREE.InstancedMesh(squareGeom, squareMat, sqCount);
      sqMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      const sObj = new THREE.Object3D();
      let sIndex = 0;
      
      for (let x = -halfWidth; x < halfWidth; x++) {
        for (let y = -halfHeight; y < halfHeight; y++) {
          const worldX = (x + 0.5) * spacing;
          const worldZ = (y + 0.5) * spacing;
          sObj.position.set(worldX, 0, worldZ);
          sObj.updateMatrix();
          sqMesh.setMatrixAt(sIndex++, sObj.matrix);
        }
      }
      
      sqMesh.frustumCulled = false;
      sqMesh.renderOrder = 0;
      this.squareMesh = sqMesh;
      this.engine.add(sqMesh, false);
    }

    // Center camera on grid
    const centerX = (minX + maxX) * 0.5;
    const centerZ = (minZ + maxZ) * 0.5;
    const worldWidth = (maxX - minX) + cellRadius * 2;
    const worldHeight = (maxZ - minZ) + cellRadius * 2;
    this.engine.recenterTo(new THREE.Vector3(centerX, 0, centerZ));
    this.engine.fitToBounds(worldWidth, worldHeight, 0.8);
  }

  private updateDisplayColor(index: number): void {
    if (!this.colorAttribute) return;
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';

    // Better color scheme with good visibility and contrast
    let terrainColor = new THREE.Color();
    switch (terrain) {
      case 'water': terrainColor.setHex(0x2196F3); break; // Bright blue
      case 'mountain': terrainColor.setHex(0x795548); break; // Brown mountain
      default: terrainColor.setHex(0x4CAF50); break; // Bright green for plains
    }

    let c = terrainColor;
    
    if (this.biomeVisible && biome !== 'none') {
      let biomeTint = new THREE.Color();
      switch (biome) {
        case 'forest': biomeTint.setHex(0x2E7D32); break; // Dark green
        case 'desert': biomeTint.setHex(0xFF8F00); break; // Orange  
        case 'tundra': biomeTint.setHex(0x03A9F4); break; // Light blue
        default: biomeTint.setHex(0xffffff); break;
      }
      c = c.clone().lerp(biomeTint, 0.5);
    }
    
    if (this.resourcesVisible && resource === 'node') {
      const resourceTint = new THREE.Color(0xE91E63); // Bright pink/magenta
      c = c.clone().lerp(resourceTint, 0.6);
    }
    
    if (!this.terrainVisible) {
      const base = new THREE.Color(0x37474f); // Neutral gray
      c = base.clone().lerp(c, 0.6);
    }
    
    // Debug: Log the color being applied
    if (index < 5) { // Only log first 5 tiles to avoid spam
      console.log(`Setting color for tile ${index}: RGB(${c.r.toFixed(2)}, ${c.g.toFixed(2)}, ${c.b.toFixed(2)}) - Terrain: ${terrain}`);
    }
    
    // Set the color for this tile instance using both methods for reliability
    this.colorAttribute.setXYZ(index, c.r, c.g, c.b);
    this.colorAttribute.needsUpdate = true;
    
    // Alternative: Direct array access (more reliable)
    const colorArray = this.colorAttribute.array as Float32Array;
    const idx = index * 3;
    colorArray[idx] = c.r;
    colorArray[idx + 1] = c.g;
    colorArray[idx + 2] = c.b;
  }

  private onClick(hits: THREE.Intersection[]): void {
    if (!this.instancedMesh || hits.length === 0) return;
    this.applyPaintFromHits(hits);
  }

  private onContext(hits: THREE.Intersection[]): void {
    if (this.editMode) return;
    if (!this.instancedMesh || hits.length === 0) return;
    const hit = hits.find((h) => h.object === this.instancedMesh);
    if (!hit) return;
    const index = (hit.instanceId ?? -1) as number;
    if (index < 0) return;
    const tile = this.tiles[index];
    const screen = this.engine?.worldToCanvas(hit.point.clone()) ?? null;
    if (screen && this.contextHandler) {
      this.contextHandler({ 
        index, 
        x: tile.x, 
        y: tile.y, 
        worldX: tile.worldX, 
        worldZ: tile.worldZ, 
        screen 
      });
    }
  }

  private onHover(hits: THREE.Intersection[]): void {
    if (!this.instancedMesh) return;
    const hit = hits.find((h) => h.object === this.instancedMesh);
    const newIndex = hit && (hit.instanceId ?? -1) >= 0 ? (hit.instanceId as number) : -1;
    
    if (newIndex !== this.hoveredIndex) {
      // Restore previous tile color
      if (this.hoveredIndex >= 0) {
        this.updateDisplayColor(this.hoveredIndex);
      }
      
      this.hoveredIndex = newIndex;
      
      // Apply hover highlight
      if (this.hoveredIndex >= 0) {
        const highlight = new THREE.Color(0x98fff0);
        this.colorAttribute?.setXYZ(this.hoveredIndex, highlight.r, highlight.g, highlight.b);
        
        // Also set via direct array access for reliability
        const colorArray = this.colorAttribute?.array as Float32Array;
        if (colorArray) {
          const idx = this.hoveredIndex * 3;
          colorArray[idx] = highlight.r;
          colorArray[idx + 1] = highlight.g;
          colorArray[idx + 2] = highlight.b;
        }
        if (this.colorAttribute) this.colorAttribute.needsUpdate = true;
        
        const tile = this.tiles[this.hoveredIndex];
        this.hovered$.next({
          index: this.hoveredIndex,
          x: tile.x,
          y: tile.y,
          worldX: tile.worldX,
          worldY: 0,
          worldZ: tile.worldZ,
          terrain: this.terrainStates.get(this.hoveredIndex) ?? 'plain',
          biome: this.biomeStates.get(this.hoveredIndex) ?? 'none',
          resource: this.resourceStates.get(this.hoveredIndex) ?? 'none'
        });
        
        if (this.hoverOutline) {
          this.hoverOutline.position.set(tile.worldX, 0.0, tile.worldZ);
          this.hoverOutline.visible = true;
        }
        if (this.hoverFill) {
          this.hoverFill.position.set(tile.worldX, 0.02, tile.worldZ);
          this.hoverFill.visible = true;
        }
      } else {
        this.hovered$.next(null);
        if (this.hoverOutline) this.hoverOutline.visible = false;
        if (this.hoverFill) this.hoverFill.visible = false;
      }
      if (this.colorAttribute) this.colorAttribute.needsUpdate = true;
    }
    
    // Paint while dragging
    if (this.painting) {
      this.applyPaintFromHits(hits);
    }
  }

  private gridDistance(a: { x: number; y: number }, b: { x: number; y: number }): number {
    const dx = Math.abs(a.x - b.x);
    const dy = Math.abs(a.y - b.y);
    return Math.max(dx, dy); // Chebyshev distance for square grid
  }

  private applyPaintFromHits(hits: THREE.Intersection[]): void {
    if (!this.instancedMesh) return;
    const hit = hits.find((h) => h.object === this.instancedMesh);
    if (!hit || (hit.instanceId ?? -1) < 0) return;
    const centerIndex = hit.instanceId as number;
    const center = this.tiles[centerIndex];
    
    for (let i = 0; i < this.tiles.length; i++) {
      if (this.gridDistance(this.tiles[i], center) <= this.brushRadius) {
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
    if (this.colorAttribute) this.colorAttribute.needsUpdate = true;
    
    // Update hover info during painting
    if (this.hoveredIndex >= 0) {
      const tile = this.tiles[this.hoveredIndex];
      this.hovered$.next({
        index: this.hoveredIndex,
        x: tile.x,
        y: tile.y,
        worldX: tile.worldX,
        worldY: 0,
        worldZ: tile.worldZ,
        terrain: this.terrainStates.get(this.hoveredIndex) ?? 'plain',
        biome: this.biomeStates.get(this.hoveredIndex) ?? 'none',
        resource: this.resourceStates.get(this.hoveredIndex) ?? 'none'
      });
    }
  }

  // Public API methods
  setActiveLayer(layer: 'terrain' | 'biome' | 'resources'): void { this.activeLayer = layer; }
  setTerrainTool(tool: TerrainState): void { this.terrainTool = tool; }
  setBiomeTool(tool: BiomeState): void { this.biomeTool = tool; }
  setResourceTool(tool: ResourceState | 'erase'): void { this.resourceTool = tool; }
  
  setLayerVisibility(layer: 'terrain' | 'biome' | 'resources', visible: boolean): void {
    if (layer === 'terrain') this.terrainVisible = visible;
    else if (layer === 'biome') this.biomeVisible = visible; 
    else this.resourcesVisible = visible;
    
    for (let i = 0; i < this.tiles.length; i++) this.updateDisplayColor(i);
    if (this.colorAttribute) this.colorAttribute.needsUpdate = true;
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

  onTileContext(handler: (ctx: { 
    index: number; 
    x: number; 
    y: number; 
    worldX: number; 
    worldZ: number; 
    screen: { x: number; y: number } 
  }) => void): void {
    this.contextHandler = handler;
  }

  setEditMode(isEdit: boolean): void { this.editMode = isEdit; }

  snapshot(name: string): Omit<TileGridSnapshot, 'id' | 'createdAt'> {
    const terrain: Array<{ index: number; state: TerrainState }> = [];
    const biome: Array<{ index: number; state: BiomeState }> = [];
    const resources: Array<{ index: number; state: ResourceState }> = [];
    
    for (let i = 0; i < this.tiles.length; i++) {
      const t = this.terrainStates.get(i) ?? 'plain';
      const b = this.biomeStates.get(i) ?? 'none';
      const r = this.resourceStates.get(i) ?? 'none';
      if (t !== 'plain') terrain.push({ index: i, state: t });
      if (b !== 'none') biome.push({ index: i, state: b });
      if (r !== 'none') resources.push({ index: i, state: r });
    }
    
    return { name, config: this.currentConfig!, layers: { terrain, biome, resources } } as any;
  }

  restore(name: string, payload: { 
    config: TileGridConfig; 
    tiles?: Array<{ index: number; state: any }>; 
    layers?: { 
      terrain: Array<{ index: number; state: TerrainState }>; 
      biome: Array<{ index: number; state: BiomeState }>; 
      resources: Array<{ index: number; state: ResourceState }> 
    } 
  }): void {
    this.createTileGrid(payload.config);
    
    if (payload.layers) {
      for (const t of payload.layers.terrain) this.terrainStates.set(t.index, t.state);
      for (const b of payload.layers.biome) this.biomeStates.set(b.index, b.state);
      for (const r of payload.layers.resources) this.resourceStates.set(r.index, r.state);
      for (let i = 0; i < this.tiles.length; i++) this.updateDisplayColor(i);
    } else if (payload.tiles) {
      // Backward compatibility
      for (const t of payload.tiles) {
        if (t.state === 'life') this.biomeStates.set(t.index, 'forest');
        else if (t.state === 'resource') this.resourceStates.set(t.index, 'node');
      }
      for (let i = 0; i < this.tiles.length; i++) this.updateDisplayColor(i);
    }
    if (this.colorAttribute) this.colorAttribute.needsUpdate = true;
  }

  randomize(): void {
    if (!this.instancedMesh) return;
    const rand = this.rng ?? Math.random;
    
    for (let i = 0; i < this.tiles.length; i++) {
      // Terrain generation
      const rT = rand();
      const t: TerrainState = rT < 0.08 ? 'water' : rT < 0.14 ? 'mountain' : 'plain';
      this.terrainStates.set(i, t);
      
      // Biome generation
      const rB = rand();
      const b: BiomeState = rB < 0.1 ? 'forest' : rB < 0.14 ? 'tundra' : rB < 0.18 ? 'desert' : 'none';
      this.biomeStates.set(i, b);
      
      // Resource generation
      const rR = rand();
      const res: ResourceState = rR < 0.05 ? 'node' : 'none';
      this.resourceStates.set(i, res);
      
      this.updateDisplayColor(i);
    }
    if (this.colorAttribute) this.colorAttribute.needsUpdate = true;
  }

  getCurrentConfig(): TileGridConfig | undefined { 
    return this.currentConfig ? { ...this.currentConfig } : undefined; 
  }

  clear(): void {
    if (!this.instancedMesh) return;
    for (let i = 0; i < this.tiles.length; i++) {
      this.terrainStates.set(i, 'plain');
      this.biomeStates.set(i, 'none');
      this.resourceStates.set(i, 'none');
      this.updateDisplayColor(i);
    }
    if (this.colorAttribute) this.colorAttribute.needsUpdate = true;
  }

  private selectFromHits(hits: THREE.Intersection[]): void {
    if (!this.instancedMesh) return;
    const hit = hits.find((h) => h.object === this.instancedMesh);
    if (!hit || (hit.instanceId ?? -1) < 0) { 
      this.selected$.next(null); 
      return; 
    }
    
    const index = hit.instanceId as number;
    const tile = this.tiles[index];
    const payload = {
      index,
      x: tile.x,
      y: tile.y,
      worldX: tile.worldX,
      worldY: 0,
      worldZ: tile.worldZ,
      terrain: this.terrainStates.get(index) ?? 'plain',
      biome: this.biomeStates.get(index) ?? 'none',
      resource: this.resourceStates.get(index) ?? 'none',
    };
    this.selected$.next(payload);
    
    // Show selection outline
    this.selectedIndex = index;
    if (this.selectedOutline) {
      this.selectedOutline.position.set(tile.worldX, 0.0, tile.worldZ);
      this.selectedOutline.visible = true;
    }
  }
}