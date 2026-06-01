import { Injectable } from '@angular/core';
import * as THREE from 'three';
import { Subject } from 'rxjs';
import type { EngineService } from './engine.service';
import type { TileGridSnapshot } from './project.service';
import { WorldConnection, CONNECTION_STYLES, ConnectionType } from './tile-metadata.model';

export interface TileGridConfig {
  cellRadius: number; // octagon radius in world units
  gridWidth: number;
  gridHeight: number;
  elevation: number; // tile height
}

export type TerrainState = 'plain' | 'water' | 'mountain';
export type BiomeState = 'none' | 'forest' | 'desert' | 'tundra';
export type ResourceState = 'none' | 'node';

interface TileTransformOptions {
  scale?: number;
  yOffset?: number;
}

interface WorldTile {
  x: number;
  y: number;
  worldX: number;
  worldY: number;
  worldZ: number;
}

interface BrushPulse {
  mesh: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
  startMs: number;
  durationMs: number;
  maxRadius: number;
}

interface ConnectionPacket {
  mesh: THREE.Mesh<THREE.SphereGeometry, THREE.MeshBasicMaterial>;
  curve: THREE.QuadraticBezierCurve3;
  offset: number;
  speed: number;
}

@Injectable({ providedIn: 'root' })
export class TileGridService {
  private engine?: EngineService;
  private instancedMesh?: THREE.InstancedMesh;
  private squareMesh?: THREE.InstancedMesh;
  private tileShadowMesh?: THREE.InstancedMesh;
  private tileEdgeMesh?: THREE.InstancedMesh;
  private resourceCoreMesh?: THREE.InstancedMesh;
  private worldDomeMesh?: THREE.InstancedMesh;
  private landmarkMesh?: THREE.InstancedMesh;
  private atmosphereRingMesh?: THREE.InstancedMesh;
  private worldBeaconMesh?: THREE.InstancedMesh;
  private galaxyStars?: THREE.Points;
  private galaxyCore?: THREE.Mesh;
  private colorAttribute?: THREE.InstancedBufferAttribute;
  private domeColorAttribute?: THREE.InstancedBufferAttribute;
  private landmarkColorAttribute?: THREE.InstancedBufferAttribute;
  private atmosphereColorAttribute?: THREE.InstancedBufferAttribute;
  private beaconColorAttribute?: THREE.InstancedBufferAttribute;
  private tiles: WorldTile[] = [];
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
  private animationTimeMs = 0;
  private gridBirthMs = 0;
  private gridBootDurationMs = 1300;
  private seedBloomMs: number | null = null;
  private randomizeRippleMs: number | null = null;
  private motionReduced = false;
  private removeAnimationUpdater?: () => void;
  private readonly animationObject = new THREE.Object3D();
  private readonly brushPulses: BrushPulse[] = [];
  private lastBrushPulseMs = -Infinity;
  private lastColorAnimationMs = -Infinity;

  // Connection visualization
  private connectionLinesGroup?: THREE.Group;
  private connectionPacketsGroup?: THREE.Group;
  private readonly connectionPackets: ConnectionPacket[] = [];
  private connectionsVisible = true;

  // Worlds that carry authored content (name / wiki / knowledge). These are
  // visually emphasised and become the scope the Next/Back stepper walks.
  private documentedWorlds = new Set<number>();

  constructor() {}

  initialize(engine: EngineService): void {
    this.engine = engine;
    this.motionReduced = typeof window !== 'undefined' &&
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
    this.removeAnimationUpdater?.();
    this.removeAnimationUpdater = this.engine.onBeforeRender((deltaMs, elapsedMs) => {
      this.updateAnimations(deltaMs, elapsedMs);
    });
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

  createTileGrid(rawConfig: TileGridConfig): void {
    if (!this.engine) {throw new Error('Engine not initialized');}

    // Normalize config — guards against legacy API responses that use `radius` instead of
    // `cellRadius`, and against null/NaN/0 values that produce NaN positions in ShapeGeometry.
    const legacyRadius = (rawConfig as any).radius;
    const resolvedRadius =
      typeof rawConfig.cellRadius === 'number' && isFinite(rawConfig.cellRadius) && rawConfig.cellRadius > 0
        ? rawConfig.cellRadius
        : typeof legacyRadius === 'number' && isFinite(legacyRadius) && legacyRadius > 0
          ? legacyRadius
          : 1;
    const config: TileGridConfig = {
      cellRadius: resolvedRadius,
      gridWidth: rawConfig.gridWidth || 50,
      gridHeight: rawConfig.gridHeight || 50,
      elevation: typeof rawConfig.elevation === 'number' && isFinite(rawConfig.elevation) ? rawConfig.elevation : 0.1
    };

    // Dispose existing render objects before rebuilding the grid.
    this.removeAndDispose(this.instancedMesh);
    this.removeAndDispose(this.squareMesh);
    this.removeAndDispose(this.tileShadowMesh);
    this.removeAndDispose(this.tileEdgeMesh);
    this.removeAndDispose(this.resourceCoreMesh);
    this.removeAndDispose(this.worldDomeMesh);
    this.removeAndDispose(this.landmarkMesh);
    this.removeAndDispose(this.atmosphereRingMesh);
    this.removeAndDispose(this.worldBeaconMesh);
    this.removeAndDispose(this.galaxyStars);
    this.removeAndDispose(this.galaxyCore);
    this.removeAndDispose(this.hoverOutline);
    this.removeAndDispose(this.selectedOutline);
    this.removeAndDispose(this.hoverFill);
    this.removeAndDispose(this.outlineMesh);
    this.removeAndDispose(this.connectionLinesGroup);
    this.removeAndDispose(this.connectionPacketsGroup);
    this.instancedMesh = undefined;
    this.squareMesh = undefined;
    this.tileShadowMesh = undefined;
    this.tileEdgeMesh = undefined;
    this.resourceCoreMesh = undefined;
    this.worldDomeMesh = undefined;
    this.landmarkMesh = undefined;
    this.atmosphereRingMesh = undefined;
    this.worldBeaconMesh = undefined;
    this.galaxyStars = undefined;
    this.galaxyCore = undefined;
    this.domeColorAttribute = undefined;
    this.landmarkColorAttribute = undefined;
    this.atmosphereColorAttribute = undefined;
    this.beaconColorAttribute = undefined;
    this.hoverOutline = undefined;
    this.selectedOutline = undefined;
    this.hoverFill = undefined;
    this.outlineMesh = undefined;
    this.connectionLinesGroup = undefined;
    this.connectionPacketsGroup = undefined;
    this.connectionPackets.length = 0;
    this.clearBrushPulses();

    const cellRadius = config.cellRadius;
    const tileGrid: WorldTile[] = [];
    const width = config.gridWidth;
    const height = config.gridHeight;

    // Create a proper 2D cartesian grid
    const halfWidth = Math.floor(width / 2);
    const halfHeight = Math.floor(height / 2);
    const logicalTiles: Array<{ x: number; y: number }> = [];

    for (let x = -halfWidth; x <= halfWidth; x++) {
      for (let y = -halfHeight; y <= halfHeight; y++) {
        logicalTiles.push({ x, y });
      }
    }

    const galaxyRadius = Math.max(width, height, Math.sqrt(logicalTiles.length)) * cellRadius * 1.85;
    const armCount = 5;
    const maxPerArm = Math.max(1, Math.ceil(logicalTiles.length / armCount));
    const hashNoise = (index: number, salt: number): number => {
      const n = Math.sin((index + 1) * 12.9898 + salt * 78.233) * 43758.5453;
      return n - Math.floor(n);
    };
    for (let i = 0; i < logicalTiles.length; i++) {
      const logical = logicalTiles[i];
      const arm = i % armCount;
      const rank = Math.floor(i / armCount);
      const t = maxPerArm <= 1 ? 0 : rank / (maxPerArm - 1);
      const radialJitter = (hashNoise(i, 1) - 0.5) * galaxyRadius * 0.11;
      const angularJitter = (hashNoise(i, 2) - 0.5) * 0.42;
      const verticalJitter = (hashNoise(i, 3) - 0.5) * galaxyRadius * 0.22;
      const radius = galaxyRadius * (0.09 + Math.pow(t, 0.72) * 0.91) + radialJitter;
      const theta = (arm / armCount) * Math.PI * 2 + t * Math.PI * 3.25 + angularJitter;
      tileGrid.push({
        x: logical.x,
        y: logical.y,
        worldX: Math.cos(theta) * radius,
        worldY: verticalJitter,
        worldZ: Math.sin(theta) * radius
      });
    }

    this.tiles = tileGrid;
    this.currentConfig = { ...config };
    this.terrainStates.clear();
    this.biomeStates.clear();
    this.resourceStates.clear();
    this.createGalaxyBackdrop(galaxyRadius, logicalTiles.length);

    // Primary interactive world-orb geometry.
    const geometry = new THREE.SphereGeometry(cellRadius * 0.78, 24, 16);

    // Material configured for per-instance coloring
    const material = new THREE.MeshStandardMaterial({
      color: 0xffffff, // White so instance colors show accurately
      transparent: false,
      depthWrite: true,
      depthTest: true,
      toneMapped: false,
      roughness: 0.58,
      metalness: 0.06
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

    const shadowGeometry = geometry.clone();
    const shadowMaterial = new THREE.MeshBasicMaterial({
      color: 0x01090c,
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false
    });
    const shadowMesh = new THREE.InstancedMesh(shadowGeometry, shadowMaterial, count);
    shadowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    shadowMesh.frustumCulled = false;
    shadowMesh.renderOrder = 0;
    this.tileShadowMesh = shadowMesh;

    const edgeGeometry = new THREE.RingGeometry(cellRadius * 0.82, cellRadius * 1.02, 8, 1);
    edgeGeometry.rotateX(-Math.PI / 2);
    edgeGeometry.rotateY(Math.PI / 8);
    const edgeMaterial = new THREE.MeshBasicMaterial({
      color: 0x032c35,
      transparent: true,
      opacity: 0.66,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false
    });
    const edgeMesh = new THREE.InstancedMesh(edgeGeometry, edgeMaterial, count);
    edgeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    edgeMesh.frustumCulled = false;
    edgeMesh.renderOrder = 2;
    this.tileEdgeMesh = edgeMesh;

    const coreGeometry = new THREE.CircleGeometry(cellRadius * 0.26, 16);
    coreGeometry.rotateX(-Math.PI / 2);
    const coreMaterial = new THREE.MeshBasicMaterial({
      color: 0xff4fd8,
      transparent: true,
      opacity: 0.86,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false
    });
    const resourceCoreMesh = new THREE.InstancedMesh(coreGeometry, coreMaterial, count);
    resourceCoreMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    resourceCoreMesh.frustumCulled = false;
    resourceCoreMesh.renderOrder = 4;
    this.instancedMesh = mesh;
    this.resourceCoreMesh = resourceCoreMesh;

    const domeGeometry = new THREE.SphereGeometry(cellRadius * 0.86, 22, 12, 0, Math.PI * 2, 0, Math.PI / 2);
    const domeMaterial = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 1,
      depthWrite: true,
      side: THREE.DoubleSide,
      toneMapped: false,
      roughness: 0.62,
      metalness: 0.04
    });
    const domeColors = new Float32Array(count * 3);
    this.domeColorAttribute = new THREE.InstancedBufferAttribute(domeColors, 3);
    this.domeColorAttribute.setUsage(THREE.DynamicDrawUsage);
    const worldDomeMesh = new THREE.InstancedMesh(domeGeometry, domeMaterial, count);
    worldDomeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    worldDomeMesh.instanceColor = this.domeColorAttribute;
    worldDomeMesh.frustumCulled = false;
    worldDomeMesh.renderOrder = 3;
    this.worldDomeMesh = worldDomeMesh;

    const atmosphereGeometry = new THREE.TorusGeometry(cellRadius * 0.88, cellRadius * 0.035, 8, 36);
    atmosphereGeometry.rotateX(Math.PI / 2);
    const atmosphereMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.78,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false
    });
    const atmosphereColors = new Float32Array(count * 3);
    this.atmosphereColorAttribute = new THREE.InstancedBufferAttribute(atmosphereColors, 3);
    this.atmosphereColorAttribute.setUsage(THREE.DynamicDrawUsage);
    const atmosphereRingMesh = new THREE.InstancedMesh(atmosphereGeometry, atmosphereMaterial, count);
    atmosphereRingMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    atmosphereRingMesh.instanceColor = this.atmosphereColorAttribute;
    atmosphereRingMesh.frustumCulled = false;
    atmosphereRingMesh.renderOrder = 6;
    this.atmosphereRingMesh = atmosphereRingMesh;

    const landmarkGeometry = new THREE.ConeGeometry(cellRadius * 0.18, cellRadius * 0.62, 6);
    const landmarkMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.9,
      depthWrite: true,
      side: THREE.DoubleSide,
      toneMapped: false
    });
    const landmarkColors = new Float32Array(count * 3);
    this.landmarkColorAttribute = new THREE.InstancedBufferAttribute(landmarkColors, 3);
    this.landmarkColorAttribute.setUsage(THREE.DynamicDrawUsage);
    const landmarkMesh = new THREE.InstancedMesh(landmarkGeometry, landmarkMaterial, count);
    landmarkMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    landmarkMesh.instanceColor = this.landmarkColorAttribute;
    landmarkMesh.frustumCulled = false;
    landmarkMesh.renderOrder = 5;
    this.landmarkMesh = landmarkMesh;

    const beaconGeometry = new THREE.CylinderGeometry(cellRadius * 0.055, cellRadius * 0.15, cellRadius * 1.35, 12);
    const beaconMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      toneMapped: false
    });
    const beaconColors = new Float32Array(count * 3);
    this.beaconColorAttribute = new THREE.InstancedBufferAttribute(beaconColors, 3);
    this.beaconColorAttribute.setUsage(THREE.DynamicDrawUsage);
    const worldBeaconMesh = new THREE.InstancedMesh(beaconGeometry, beaconMaterial, count);
    worldBeaconMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    worldBeaconMesh.instanceColor = this.beaconColorAttribute;
    worldBeaconMesh.frustumCulled = false;
    worldBeaconMesh.renderOrder = 8;
    this.worldBeaconMesh = worldBeaconMesh;

    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < count; i++) {
      const tile = tileGrid[i];
      
      // Initialize layer states with better default colors
      this.terrainStates.set(i, 'plain');
      this.biomeStates.set(i, 'none');
      this.resourceStates.set(i, 'none');
      this.setTileTransform(i, { scale: 1 });
      this.updateDisplayColor(i);
      
      if (tile.worldX < minX) {minX = tile.worldX;}
      if (tile.worldX > maxX) {maxX = tile.worldX;}
      if (tile.worldZ < minZ) {minZ = tile.worldZ;}
      if (tile.worldZ > maxZ) {maxZ = tile.worldZ;}
    }
    mesh.instanceMatrix.needsUpdate = true;
    shadowMesh.instanceMatrix.needsUpdate = true;
    edgeMesh.instanceMatrix.needsUpdate = true;
    resourceCoreMesh.instanceMatrix.needsUpdate = true;
    worldDomeMesh.instanceMatrix.needsUpdate = true;
    atmosphereRingMesh.instanceMatrix.needsUpdate = true;
    landmarkMesh.instanceMatrix.needsUpdate = true;
    worldBeaconMesh.instanceMatrix.needsUpdate = true;
    this.colorAttribute.needsUpdate = true;
    this.domeColorAttribute.needsUpdate = true;
    this.atmosphereColorAttribute.needsUpdate = true;
    this.landmarkColorAttribute.needsUpdate = true;
    this.beaconColorAttribute.needsUpdate = true;

    this.instancedMesh = mesh;
    this.instancedMesh.renderOrder = 1;
    this.engine.add(shadowMesh);
    this.engine.add(mesh, true);
    this.engine.add(edgeMesh);
    this.engine.add(worldDomeMesh);
    this.engine.add(atmosphereRingMesh);
    this.engine.add(resourceCoreMesh);
    this.engine.add(landmarkMesh);
    this.engine.add(worldBeaconMesh);
    
    // Force an update of all colors to ensure they display correctly
    for (let i = 0; i < count; i++) {
      this.updateDisplayColor(i);
    }
    this.colorAttribute.needsUpdate = true;
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
      const connectorSpacing = cellRadius * 2.2;
      let sIndex = 0;
      
      for (let x = -halfWidth; x < halfWidth; x++) {
        for (let y = -halfHeight; y < halfHeight; y++) {
          const worldX = (x + 0.5) * connectorSpacing;
          const worldZ = (y + 0.5) * connectorSpacing;
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
    const maxDistance = tileGrid.reduce(
      (distance, tile) => Math.max(distance, Math.hypot(tile.worldX - centerX, tile.worldZ - centerZ)),
      0
    );
    this.gridBootDurationMs = maxDistance * 11 + 620;
    this.engine.recenterTo(new THREE.Vector3(centerX, 0, centerZ));
    this.engine.fitToBounds(worldWidth, worldHeight, 1.08);
    this.gridBirthMs = this.animationTimeMs;
  }

  private updateDisplayColor(index: number): void {
    if (!this.colorAttribute) {return;}
    const c = this.getDisplayColor(index);
    const domeColor = this.getDomeColor(index, c);
    const landmarkColor = this.getLandmarkColor(index);
    const atmosphereColor = this.getAtmosphereColor(index, domeColor);
    const beaconColor = this.getBeaconColor(index);
    this.colorAttribute.setXYZ(index, c.r, c.g, c.b);
    this.domeColorAttribute?.setXYZ(index, domeColor.r, domeColor.g, domeColor.b);
    this.landmarkColorAttribute?.setXYZ(index, landmarkColor.r, landmarkColor.g, landmarkColor.b);
    this.atmosphereColorAttribute?.setXYZ(index, atmosphereColor.r, atmosphereColor.g, atmosphereColor.b);
    this.beaconColorAttribute?.setXYZ(index, beaconColor.r, beaconColor.g, beaconColor.b);
    this.colorAttribute.needsUpdate = true;
    if (this.domeColorAttribute) {this.domeColorAttribute.needsUpdate = true;}
    if (this.landmarkColorAttribute) {this.landmarkColorAttribute.needsUpdate = true;}
    if (this.atmosphereColorAttribute) {this.atmosphereColorAttribute.needsUpdate = true;}
    if (this.beaconColorAttribute) {this.beaconColorAttribute.needsUpdate = true;}
    
    // Alternative: direct array access keeps instance colors reliable across renderer updates.
    const colorArray = this.colorAttribute.array as Float32Array;
    const idx = index * 3;
    colorArray[idx] = c.r;
    colorArray[idx + 1] = c.g;
    colorArray[idx + 2] = c.b;

    const domeArray = this.domeColorAttribute?.array as Float32Array | undefined;
    if (domeArray) {
      domeArray[idx] = domeColor.r;
      domeArray[idx + 1] = domeColor.g;
      domeArray[idx + 2] = domeColor.b;
    }

    const landmarkArray = this.landmarkColorAttribute?.array as Float32Array | undefined;
    if (landmarkArray) {
      landmarkArray[idx] = landmarkColor.r;
      landmarkArray[idx + 1] = landmarkColor.g;
      landmarkArray[idx + 2] = landmarkColor.b;
    }

    const atmosphereArray = this.atmosphereColorAttribute?.array as Float32Array | undefined;
    if (atmosphereArray) {
      atmosphereArray[idx] = atmosphereColor.r;
      atmosphereArray[idx + 1] = atmosphereColor.g;
      atmosphereArray[idx + 2] = atmosphereColor.b;
    }

    const beaconArray = this.beaconColorAttribute?.array as Float32Array | undefined;
    if (beaconArray) {
      beaconArray[idx] = beaconColor.r;
      beaconArray[idx + 1] = beaconColor.g;
      beaconArray[idx + 2] = beaconColor.b;
    }
  }

  private createGalaxyBackdrop(galaxyRadius: number, worldCount: number): void {
    if (!this.engine) {return;}

    const starCount = Math.min(1800, Math.max(600, worldCount * 3));
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);
    const arms = 5;
    for (let i = 0; i < starCount; i++) {
      const noiseA = this.hashNoise(i, 17);
      const noiseB = this.hashNoise(i, 23);
      const noiseC = this.hashNoise(i, 31);
      const arm = i % arms;
      const t = Math.pow(noiseA, 0.62);
      const radius = galaxyRadius * (0.05 + t * 1.08);
      const theta = (arm / arms) * Math.PI * 2 + t * Math.PI * 3.35 + (noiseB - 0.5) * 0.68;
      const idx = i * 3;
      positions[idx] = Math.cos(theta) * radius + (noiseB - 0.5) * galaxyRadius * 0.12;
      positions[idx + 1] = (noiseC - 0.5) * galaxyRadius * 0.34;
      positions[idx + 2] = Math.sin(theta) * radius + (noiseA - 0.5) * galaxyRadius * 0.12;

      const cool = new THREE.Color(0x9fffea);
      const warm = new THREE.Color(0xffd580);
      const color = cool.lerp(warm, this.hashNoise(i, 41) * 0.45);
      colors[idx] = color.r;
      colors[idx + 1] = color.g;
      colors[idx + 2] = color.b;
    }

    const starGeometry = new THREE.BufferGeometry();
    starGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    starGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const starMaterial = new THREE.PointsMaterial({
      size: 0.16,
      vertexColors: true,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false
    });
    this.galaxyStars = new THREE.Points(starGeometry, starMaterial);
    this.galaxyStars.renderOrder = -2;
    this.engine.add(this.galaxyStars);

    const coreGeometry = new THREE.SphereGeometry(Math.max(1.4, galaxyRadius * 0.055), 32, 16);
    const coreMaterial = new THREE.MeshBasicMaterial({
      color: 0x00ffc8,
      transparent: true,
      opacity: 0.32,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false
    });
    this.galaxyCore = new THREE.Mesh(coreGeometry, coreMaterial);
    this.galaxyCore.renderOrder = -1;
    this.engine.add(this.galaxyCore);
  }

  private setTileTransform(index: number, options: TileTransformOptions = {}): void {
    const tile = this.tiles[index];
    if (!tile) {return;}

    const scale = (options.scale ?? 1) * this.getOrbScale(index);
    const y = this.getTileElevation(index) + (options.yOffset ?? 0);

    this.animationObject.position.set(tile.worldX, y, tile.worldZ);
    this.animationObject.position.y += tile.worldY;
    this.animationObject.scale.setScalar(scale);
    this.animationObject.updateMatrix();
    this.instancedMesh?.setMatrixAt(index, this.animationObject.matrix);

    this.animationObject.position.set(tile.worldX + 0.04, -0.045, tile.worldZ + 0.06);
    this.animationObject.position.y += tile.worldY;
    this.animationObject.scale.setScalar(scale * 1.06);
    this.animationObject.updateMatrix();
    this.tileShadowMesh?.setMatrixAt(index, this.animationObject.matrix);

    this.animationObject.position.set(tile.worldX, y + 0.032, tile.worldZ);
    this.animationObject.position.y += tile.worldY;
    this.animationObject.scale.setScalar(scale);
    this.animationObject.updateMatrix();
    this.tileEdgeMesh?.setMatrixAt(index, this.animationObject.matrix);

    const domeBaseScale = this.getDomeScale(index);
    const domeScale = {
      x: domeBaseScale.x * scale,
      y: domeBaseScale.y * scale,
      z: domeBaseScale.z * scale
    };
    this.animationObject.position.set(tile.worldX, y + 0.04, tile.worldZ);
    this.animationObject.position.y += tile.worldY;
    this.animationObject.scale.set(domeScale.x, domeScale.y, domeScale.z);
    this.animationObject.updateMatrix();
    this.worldDomeMesh?.setMatrixAt(index, this.animationObject.matrix);

    const documentedHalo = this.documentedWorlds.has(index) ? 1.32 : 1;
    const atmosphereScale = (index === this.hoveredIndex || index === this.selectedIndex ? 1.12 : 1) * documentedHalo * scale;
    this.animationObject.position.set(tile.worldX, y + 0.16 + domeScale.y * 0.28, tile.worldZ);
    this.animationObject.position.y += tile.worldY;
    this.animationObject.scale.set(atmosphereScale, atmosphereScale, atmosphereScale);
    this.animationObject.updateMatrix();
    this.atmosphereRingMesh?.setMatrixAt(index, this.animationObject.matrix);

    const hasResource = this.resourcesVisible && (this.resourceStates.get(index) ?? 'none') === 'node';
    const coreScale = hasResource ? Math.max(0.55, scale * 0.92) : 0.001;
    this.animationObject.position.set(tile.worldX, y + 0.22 + domeScale.y * 0.18, tile.worldZ);
    this.animationObject.position.y += tile.worldY;
    this.animationObject.scale.setScalar(coreScale);
    this.animationObject.updateMatrix();
    this.resourceCoreMesh?.setMatrixAt(index, this.animationObject.matrix);

    const landmarkScale = this.getLandmarkScale(index) * scale;
    const landmarkOffset = this.getLandmarkOffset(index);
    this.animationObject.position.set(
      tile.worldX + landmarkOffset.x,
      tile.worldY + y + 0.34 + domeScale.y * 0.32,
      tile.worldZ + landmarkOffset.z
    );
    this.animationObject.rotation.set(0, this.tileNoise(index) * Math.PI * 2, 0);
    this.animationObject.scale.setScalar(landmarkScale);
    this.animationObject.updateMatrix();
    this.landmarkMesh?.setMatrixAt(index, this.animationObject.matrix);
    this.animationObject.rotation.set(0, 0, 0);

    const beaconHeight = this.getBeaconHeight(index);
    const beaconScale = (index === this.hoveredIndex || index === this.selectedIndex ? 1.35 : 1) * scale;
    this.animationObject.position.set(
      tile.worldX,
      tile.worldY + y + 0.42 + domeScale.y * 0.58 + beaconHeight * 0.38,
      tile.worldZ
    );
    this.animationObject.scale.set(beaconScale, beaconHeight, beaconScale);
    this.animationObject.updateMatrix();
    this.worldBeaconMesh?.setMatrixAt(index, this.animationObject.matrix);
  }

  private markTileTransformsDirty(): void {
    if (this.instancedMesh) {this.instancedMesh.instanceMatrix.needsUpdate = true;}
    if (this.tileShadowMesh) {this.tileShadowMesh.instanceMatrix.needsUpdate = true;}
    if (this.tileEdgeMesh) {this.tileEdgeMesh.instanceMatrix.needsUpdate = true;}
    if (this.resourceCoreMesh) {this.resourceCoreMesh.instanceMatrix.needsUpdate = true;}
    if (this.worldDomeMesh) {this.worldDomeMesh.instanceMatrix.needsUpdate = true;}
    if (this.landmarkMesh) {this.landmarkMesh.instanceMatrix.needsUpdate = true;}
    if (this.atmosphereRingMesh) {this.atmosphereRingMesh.instanceMatrix.needsUpdate = true;}
    if (this.worldBeaconMesh) {this.worldBeaconMesh.instanceMatrix.needsUpdate = true;}
  }

  private getTileElevation(index: number): number {
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';

    let elevation = 0;
    if (terrain === 'water') {
      elevation -= 0.035;
    } else if (terrain === 'mountain') {
      elevation += 0.13;
    }
    if (biome === 'forest') {
      elevation += 0.025;
    } else if (biome === 'tundra') {
      elevation += 0.012;
    }
    if (resource === 'node') {
      elevation += 0.045;
    }
    return elevation;
  }

  private getDomeScale(index: number): { x: number; y: number; z: number } {
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';
    const variation = this.tileNoise(index);

    let height = 0.62 + variation * 0.18;
    let width = 1.04;
    if (terrain === 'water') {
      height = 0.34 + variation * 0.08;
      width = 1.08;
    } else if (terrain === 'mountain') {
      height = 1.05 + variation * 0.3;
      width = 0.82;
    }
    if (biome === 'forest') {
      height += 0.12;
      width *= 0.92;
    } else if (biome === 'desert') {
      height *= 0.82;
      width *= 1.05;
    } else if (biome === 'tundra') {
      height += 0.04;
    }
    if (resource === 'node') {
      height += 0.08;
    }
    return { x: width, y: height, z: width };
  }

  private getOrbScale(index: number): number {
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const resource = this.resourceStates.get(index) ?? 'none';
    let scale = 0.78 + this.tileNoise(index) * 0.42;
    if (terrain === 'mountain') {scale += 0.16;}
    if (terrain === 'water') {scale -= 0.06;}
    if (resource === 'node') {scale += 0.12;}
    if (this.documentedWorlds.has(index)) {scale += 0.22;} // authored worlds loom larger
    return scale;
  }

  private getLandmarkScale(index: number): number {
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';

    if (terrain === 'mountain') {return 1.08;}
    if (resource === 'node') {return 0.74;}
    if (biome === 'forest') {return 0.62;}
    if (biome === 'desert' || biome === 'tundra') {return 0.4;}
    return 0.26;
  }

  private getLandmarkOffset(index: number): { x: number; z: number } {
    const noise = this.tileNoise(index);
    const angle = noise * Math.PI * 2;
    const radius = (this.currentConfig?.cellRadius ?? 1) * (0.12 + noise * 0.22);
    return { x: Math.cos(angle) * radius, z: Math.sin(angle) * radius };
  }

  private getBeaconHeight(index: number): number {
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';

    let h = 0.42;
    if (resource === 'node') {h = 1.25;}
    else if (terrain === 'mountain') {h = 1.05;}
    else if (terrain === 'water') {h = 0.52;}
    else if (biome !== 'none') {h = 0.78;}
    // Authored worlds raise a tall light pillar to stand out across the galaxy.
    if (this.documentedWorlds.has(index)) {h = Math.max(h, 1.0) * 1.6;}
    return h;
  }

  private tileNoise(index: number): number {
    const tile = this.tiles[index];
    const x = tile?.x ?? index;
    const y = tile?.y ?? index * 7;
    return this.hashNoise(x * 31 + y * 17 + index, 5);
  }

  private hashNoise(index: number, salt: number): number {
    const n = Math.sin((index + 1) * 12.9898 + salt * 78.233) * 43758.5453;
    return n - Math.floor(n);
  }

  private getDisplayColor(index: number): THREE.Color {
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';

    const tile = this.tiles[index];
    const variation = this.tileNoise(index);
    const terrainColor = new THREE.Color();
    switch (terrain) {
      case 'water':
        terrainColor.setHex(0x1976d2);
        terrainColor.lerp(new THREE.Color(0x38d8ff), 0.18 + variation * 0.18);
        break;
      case 'mountain':
        terrainColor.setHex(0x7a5a3d);
        terrainColor.lerp(new THREE.Color(0xffb347), 0.1 + variation * 0.18);
        break;
      default:
        terrainColor.setHex(0x39b85a);
        terrainColor.lerp(new THREE.Color(0x9be86f), 0.12 + variation * 0.2);
        break;
    }

    let c = terrainColor;
    const radialShade = tile ? Math.min(0.18, Math.hypot(tile.x, tile.y) * 0.004) : 0;
    c = c.clone().lerp(new THREE.Color(0x041a18), radialShade);
    
    if (this.biomeVisible && biome !== 'none') {
      const biomeTint = new THREE.Color();
      switch (biome) {
        case 'forest': biomeTint.setHex(0x175f33); break;
        case 'desert': biomeTint.setHex(0xd88a1d); break;
        case 'tundra': biomeTint.setHex(0x70dfff); break;
        default: biomeTint.setHex(0xffffff); break;
      }
      c = c.clone().lerp(biomeTint, 0.42);
    }
    
    if (this.resourcesVisible && resource === 'node') {
      const resourceTint = new THREE.Color(0xe91e63);
      c = c.clone().lerp(resourceTint, 0.45);
    }
    
    if (!this.terrainVisible) {
      const base = new THREE.Color(0x37474f); // Neutral gray
      c = base.clone().lerp(c, 0.6);
    }

    if (index === this.hoveredIndex) {
      c = c.clone().lerp(new THREE.Color(0x98fff0), 0.42);
    }
    if (index === this.selectedIndex) {
      c = c.clone().lerp(new THREE.Color(0x00ffd5), 0.34);
    }
    if (this.documentedWorlds.has(index)) {
      c = c.clone().lerp(new THREE.Color(0xd8fff4), 0.16); // authored worlds glow
    }

    if (!this.motionReduced) {
      const seconds = this.animationTimeMs / 1000;
      if (terrain === 'water') {
        const shimmer = (Math.sin(seconds * 2.4 + index * 0.37) + 1) * 0.06;
        c = c.clone().lerp(new THREE.Color(0x6ee7ff), shimmer);
      }
      if (this.biomeVisible && biome !== 'none') {
        const breath = (Math.sin(seconds * 0.9 + index * 0.09) + 1) * 0.04;
        const breathColor = biome === 'desert' ? 0xffd580 : biome === 'tundra' ? 0xb9f6ff : 0x8dffb0;
        c = c.clone().lerp(new THREE.Color(breathColor), breath);
      }
      if (this.resourcesVisible && resource === 'node') {
        const sparkle = Math.pow((Math.sin(seconds * 3.8 + index * 1.7) + 1) * 0.5, 5) * 0.35;
        c = c.clone().lerp(new THREE.Color(0xfff0ff), sparkle);
      }
      c = this.applyWaveColor(index, c);
    }

    return c;
  }

  private getDomeColor(index: number, baseColor: THREE.Color): THREE.Color {
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';
    let color = baseColor.clone();

    if (terrain === 'water') {
      color = color.lerp(new THREE.Color(0x9cf7ff), 0.34);
    } else if (terrain === 'mountain') {
      color = color.lerp(new THREE.Color(0xffd580), 0.18);
    } else {
      color = color.lerp(new THREE.Color(0xb8ff91), 0.16);
    }

    if (biome === 'forest') {
      color = color.lerp(new THREE.Color(0x0f7f46), 0.24);
    } else if (biome === 'desert') {
      color = color.lerp(new THREE.Color(0xffca74), 0.28);
    } else if (biome === 'tundra') {
      color = color.lerp(new THREE.Color(0xd8fbff), 0.28);
    }

    if (resource === 'node') {
      color = color.lerp(new THREE.Color(0xff70d7), 0.22);
    }
    if (index === this.hoveredIndex || index === this.selectedIndex) {
      color = color.lerp(new THREE.Color(0xe9fff8), 0.22);
    }
    return color;
  }

  private getAtmosphereColor(index: number, domeColor: THREE.Color): THREE.Color {
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';

    if (this.documentedWorlds.has(index)) {return new THREE.Color(0xb9fff2);} // bright halo
    if (resource === 'node') {return new THREE.Color(0xff4fd8);}
    if (terrain === 'water') {return new THREE.Color(0x66f1ff);}
    if (terrain === 'mountain') {return new THREE.Color(0xffd580);}
    if (biome === 'forest') {return new THREE.Color(0x80ffb8);}
    if (biome === 'desert') {return new THREE.Color(0xffc36b);}
    if (biome === 'tundra') {return new THREE.Color(0xc9fbff);}
    return domeColor.clone().lerp(new THREE.Color(0x00ffc8), 0.62);
  }

  private getBeaconColor(index: number): THREE.Color {
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';

    if (this.documentedWorlds.has(index)) {return new THREE.Color(0xeafffb);}
    if (resource === 'node') {return new THREE.Color(0xff8aea);}
    if (terrain === 'water') {return new THREE.Color(0x73eaff);}
    if (terrain === 'mountain') {return new THREE.Color(0xffd580);}
    if (biome === 'forest') {return new THREE.Color(0x9affbd);}
    if (biome === 'desert') {return new THREE.Color(0xffc870);}
    if (biome === 'tundra') {return new THREE.Color(0xe0fbff);}
    return new THREE.Color(0x00ffc8);
  }

  private getLandmarkColor(index: number): THREE.Color {
    const terrain = this.terrainStates.get(index) ?? 'plain';
    const biome = this.biomeStates.get(index) ?? 'none';
    const resource = this.resourceStates.get(index) ?? 'none';

    if (resource === 'node') {return new THREE.Color(0xff4fd8);}
    if (terrain === 'mountain') {return new THREE.Color(0xffd580);}
    if (biome === 'forest') {return new THREE.Color(0x8dffb0);}
    if (biome === 'desert') {return new THREE.Color(0xffb347);}
    if (biome === 'tundra') {return new THREE.Color(0xc7f8ff);}
    return new THREE.Color(0x00ffc8);
  }

  private applyWaveColor(index: number, baseColor: THREE.Color): THREE.Color {
    const tile = this.tiles[index];
    if (!tile) {return baseColor;}

    const distance = Math.hypot(tile.worldX, tile.worldZ);
    let waveStrength = 0;
    let waveColor = new THREE.Color(0x00ffd5);

    if (this.seedBloomMs !== null) {
      const ageMs = this.animationTimeMs - this.seedBloomMs;
      if (ageMs > 1400) {
        this.seedBloomMs = null;
      } else if (ageMs >= 0) {
        const waveCenter = ageMs * 0.04;
        const seedWave = Math.max(0, 1 - Math.abs(distance - waveCenter) / 5) * (1 - ageMs / 1400);
        waveStrength = Math.max(waveStrength, seedWave);
        waveColor = new THREE.Color(0xffd580);
      }
    }

    if (this.randomizeRippleMs !== null) {
      const ageMs = this.animationTimeMs - this.randomizeRippleMs;
      if (ageMs > 1200) {
        this.randomizeRippleMs = null;
      } else if (ageMs >= 0) {
        const waveCenter = ageMs * 0.05;
        const ripple = Math.max(0, 1 - Math.abs(distance - waveCenter) / 4.5) * (1 - ageMs / 1200);
        if (ripple > waveStrength) {
          waveStrength = ripple;
          waveColor = new THREE.Color(0x00d2ff);
        }
      }
    }

    return waveStrength > 0 ? baseColor.clone().lerp(waveColor, Math.min(0.55, waveStrength * 0.55)) : baseColor;
  }

  private updateAnimations(_deltaMs: number, elapsedMs: number): void {
    this.animationTimeMs = elapsedMs;
    if (this.motionReduced) {return;}

    const transformsUpdated = this.updateTileTransforms();
    const shouldUpdateColors = transformsUpdated ||
      (this.hasActiveColorAnimation() && this.animationTimeMs - this.lastColorAnimationMs > 66);
    this.updateHoverAndSelectionEffects();
    this.updateResourceCoreEffect();
    this.updateAtmosphereEffect();
    this.updateBrushPulses();
    this.updateConnectionPackets();

    if (shouldUpdateColors && this.colorAttribute) {
      this.lastColorAnimationMs = this.animationTimeMs;
      for (let i = 0; i < this.tiles.length; i++) {
        const c = this.getDisplayColor(i);
        const domeColor = this.getDomeColor(i, c);
        const landmarkColor = this.getLandmarkColor(i);
        const atmosphereColor = this.getAtmosphereColor(i, domeColor);
        const beaconColor = this.getBeaconColor(i);
        this.colorAttribute.setXYZ(i, c.r, c.g, c.b);
        this.domeColorAttribute?.setXYZ(i, domeColor.r, domeColor.g, domeColor.b);
        this.landmarkColorAttribute?.setXYZ(i, landmarkColor.r, landmarkColor.g, landmarkColor.b);
        this.atmosphereColorAttribute?.setXYZ(i, atmosphereColor.r, atmosphereColor.g, atmosphereColor.b);
        this.beaconColorAttribute?.setXYZ(i, beaconColor.r, beaconColor.g, beaconColor.b);
      }
      this.colorAttribute.needsUpdate = true;
      if (this.domeColorAttribute) {this.domeColorAttribute.needsUpdate = true;}
      if (this.landmarkColorAttribute) {this.landmarkColorAttribute.needsUpdate = true;}
      if (this.atmosphereColorAttribute) {this.atmosphereColorAttribute.needsUpdate = true;}
      if (this.beaconColorAttribute) {this.beaconColorAttribute.needsUpdate = true;}
    }
  }

  private updateTileTransforms(): boolean {
    if (!this.instancedMesh || !this.currentConfig) {return false;}

    const ageMs = this.animationTimeMs - this.gridBirthMs;
    const bootActive = ageMs >= 0 && ageMs <= this.gridBootDurationMs;
    const interactionActive = this.hoveredIndex >= 0 || this.selectedIndex >= 0 ||
      (this.resourcesVisible && this.hasResourceNodes());
    if (!bootActive && !interactionActive) {return false;}

    const centerX = this.tiles.reduce((sum, tile) => sum + tile.worldX, 0) / Math.max(1, this.tiles.length);
    const centerZ = this.tiles.reduce((sum, tile) => sum + tile.worldZ, 0) / Math.max(1, this.tiles.length);
    const seconds = this.animationTimeMs / 1000;

    for (let i = 0; i < this.tiles.length; i++) {
      const tile = this.tiles[i];
      let scale = 1;
      let yOffset = 0;

      if (bootActive) {
        const distance = Math.hypot(tile.worldX - centerX, tile.worldZ - centerZ);
        const delayMs = distance * 11;
        const progress = this.clamp01((ageMs - delayMs) / 520);
        const eased = 1 - Math.pow(1 - progress, 3);
        scale = 0.28 + eased * 0.72;
        yOffset += (1 - eased) * 0.5;
      }

      if (i === this.hoveredIndex) {
        scale += 0.08;
        yOffset += 0.12;
      }
      if (i === this.selectedIndex) {
        scale += 0.1 + (Math.sin(seconds * 3.5) + 1) * 0.02;
        yOffset += 0.18;
      }
      if ((this.resourceStates.get(i) ?? 'none') === 'node') {
        yOffset += (Math.sin(seconds * 4.2 + i) + 1) * 0.015;
      }

      this.setTileTransform(i, { scale, yOffset });
    }
    this.markTileTransformsDirty();
    return true;
  }

  private hasActiveColorAnimation(): boolean {
    return this.seedBloomMs !== null ||
      this.randomizeRippleMs !== null ||
      Array.from(this.terrainStates.values()).some((terrain) => terrain === 'water') ||
      Array.from(this.resourceStates.values()).some((resource) => resource === 'node') ||
      Array.from(this.biomeStates.values()).some((biome) => biome !== 'none');
  }

  private hasResourceNodes(): boolean {
    return Array.from(this.resourceStates.values()).some((resource) => resource === 'node');
  }

  private updateHoverAndSelectionEffects(): void {
    const seconds = this.animationTimeMs / 1000;

    if (this.hoverOutline?.visible) {
      const hoverScale = 1 + Math.sin(seconds * 4) * 0.035;
      this.hoverOutline.scale.setScalar(hoverScale);
    }

    if (this.hoverFill?.visible) {
      const material = this.hoverFill.material as THREE.MeshBasicMaterial;
      material.opacity = 0.12 + (Math.sin(seconds * 3.2) + 1) * 0.04;
    }

    if (this.selectedOutline?.visible) {
      const selectedScale = 1.02 + (Math.sin(seconds * 3) + 1) * 0.035;
      this.selectedOutline.scale.setScalar(selectedScale);
      this.selectedOutline.rotation.y = seconds * 0.65;
      const material = this.selectedOutline.material as THREE.LineBasicMaterial;
      material.opacity = 0.78 + (Math.sin(seconds * 2.4) + 1) * 0.11;
    }
  }

  private updateResourceCoreEffect(): void {
    if (!this.resourceCoreMesh) {return;}
    const material = this.resourceCoreMesh.material as THREE.MeshBasicMaterial;
    material.opacity = 0.68 + (Math.sin(this.animationTimeMs / 260) + 1) * 0.12;
  }

  private updateAtmosphereEffect(): void {
    if (!this.atmosphereRingMesh) {return;}
    const material = this.atmosphereRingMesh.material as THREE.MeshBasicMaterial;
    material.opacity = 0.62 + (Math.sin(this.animationTimeMs / 480) + 1) * 0.14;
  }

  private updateBrushPulses(): void {
    if (!this.engine) {return;}
    for (let i = this.brushPulses.length - 1; i >= 0; i--) {
      const pulse = this.brushPulses[i];
      const progress = this.clamp01((this.animationTimeMs - pulse.startMs) / pulse.durationMs);
      pulse.mesh.scale.setScalar(0.2 + progress * pulse.maxRadius);
      pulse.mesh.material.opacity = (1 - progress) * 0.52;
      if (progress >= 1) {
        this.engine.remove(pulse.mesh);
        pulse.mesh.geometry.dispose();
        pulse.mesh.material.dispose();
        this.brushPulses.splice(i, 1);
      }
    }
  }

  private updateConnectionPackets(): void {
    if (!this.connectionPacketsGroup || !this.connectionsVisible) {return;}
    for (const packet of this.connectionPackets) {
      const t = ((this.animationTimeMs * packet.speed) / 1000 + packet.offset) % 1;
      const point = packet.curve.getPoint(t);
      packet.mesh.position.copy(point);
      // Swell + brighten mid-arc so the packet reads as a travelling comet of light.
      const along = Math.sin(t * Math.PI);
      packet.mesh.scale.setScalar(0.65 + along * 0.7);
      packet.mesh.material.opacity = 0.4 + along * 0.45;
    }
  }

  private createBrushPulse(center: { worldX: number; worldZ: number }): void {
    if (!this.engine || !this.currentConfig || this.motionReduced) {return;}
    if (this.animationTimeMs - this.lastBrushPulseMs < 90) {return;}
    this.lastBrushPulseMs = this.animationTimeMs;

    const color = this.getActiveLayerColor();
    const geometry = new THREE.RingGeometry(1, 1.08, 48);
    geometry.rotateX(-Math.PI / 2);
    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.52,
      depthWrite: false,
      side: THREE.DoubleSide
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(center.worldX, 0.075, center.worldZ);
    mesh.renderOrder = 6;
    this.engine.add(mesh);
    this.brushPulses.push({
      mesh,
      startMs: this.animationTimeMs,
      durationMs: 620,
      maxRadius: Math.max(1.4, (this.brushRadius + 1) * this.currentConfig.cellRadius * 1.8)
    });
  }

  private getActiveLayerColor(): number {
    if (this.activeLayer === 'biome') {return 0xffd580;}
    if (this.activeLayer === 'resources') {return 0xe91e63;}
    if (this.terrainTool === 'water') {return 0x00d2ff;}
    if (this.terrainTool === 'mountain') {return 0xffb347;}
    return 0x00ffc8;
  }

  private clearBrushPulses(): void {
    if (!this.engine) {
      this.brushPulses.length = 0;
      return;
    }
    for (const pulse of this.brushPulses) {
      this.engine.remove(pulse.mesh);
      pulse.mesh.geometry.dispose();
      pulse.mesh.material.dispose();
    }
    this.brushPulses.length = 0;
  }

  private removeAndDispose(object?: THREE.Object3D): void {
    if (!object || !this.engine) {return;}
    this.engine.remove(object);
    object.traverse((child) => {
      const disposable = child as THREE.Object3D & {
        geometry?: THREE.BufferGeometry;
        material?: THREE.Material | THREE.Material[];
      };
      disposable.geometry?.dispose();
      if (Array.isArray(disposable.material)) {
        for (const material of disposable.material) {
          material.dispose();
        }
      } else {
        disposable.material?.dispose();
      }
    });
  }

  private clamp01(value: number): number {
    return Math.max(0, Math.min(1, value));
  }

  private onContext(hits: THREE.Intersection[]): void {
    if (this.editMode) {return;}
    if (!this.instancedMesh || hits.length === 0) {return;}
    const hit = hits.find((h) => h.object === this.instancedMesh);
    if (!hit) {return;}
    const index = (hit.instanceId ?? -1) as number;
    if (index < 0) {return;}
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
    if (!this.instancedMesh) {return;}
    const hit = hits.find((h) => h.object === this.instancedMesh);
    const newIndex = hit && (hit.instanceId ?? -1) >= 0 ? (hit.instanceId as number) : -1;
    
    if (newIndex !== this.hoveredIndex) {
      const previousIndex = this.hoveredIndex;
      // Restore previous tile color
      if (previousIndex >= 0) {
        this.updateDisplayColor(previousIndex);
        this.setTileTransform(previousIndex);
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
        if (this.colorAttribute) {this.colorAttribute.needsUpdate = true;}
        
        const tile = this.tiles[this.hoveredIndex];
        this.hovered$.next({
          index: this.hoveredIndex,
          x: tile.x,
          y: tile.y,
          worldX: tile.worldX,
          worldY: tile.worldY,
          worldZ: tile.worldZ,
          terrain: this.terrainStates.get(this.hoveredIndex) ?? 'plain',
          biome: this.biomeStates.get(this.hoveredIndex) ?? 'none',
          resource: this.resourceStates.get(this.hoveredIndex) ?? 'none'
        });
        
        if (this.hoverOutline) {
          this.hoverOutline.position.set(
            tile.worldX,
            tile.worldY + this.getTileElevation(this.hoveredIndex) + 1.4,
            tile.worldZ
          );
          this.hoverOutline.visible = true;
        }
        if (this.hoverFill) {
          this.hoverFill.position.set(
            tile.worldX,
            tile.worldY + this.getTileElevation(this.hoveredIndex) + 1.35,
            tile.worldZ
          );
          this.hoverFill.visible = true;
        }
      } else {
        this.hovered$.next(null);
        if (this.hoverOutline) {this.hoverOutline.visible = false;}
        if (this.hoverFill) {this.hoverFill.visible = false;}
      }
      if (this.hoveredIndex >= 0) {
        this.setTileTransform(this.hoveredIndex, { scale: 1.08, yOffset: 0.12 });
      }
      this.markTileTransformsDirty();
      if (this.colorAttribute) {this.colorAttribute.needsUpdate = true;}
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
    if (!this.instancedMesh) {return;}
    const hit = hits.find((h) => h.object === this.instancedMesh);
    if (!hit || (hit.instanceId ?? -1) < 0) {return;}
    const centerIndex = hit.instanceId as number;
    const center = this.tiles[centerIndex];
    this.createBrushPulse(center);
    
    for (let i = 0; i < this.tiles.length; i++) {
      if (this.gridDistance(this.tiles[i], center) <= this.brushRadius) {
        if (this.activeLayer === 'terrain') {
          this.terrainStates.set(i, this.terrainTool);
        } else if (this.activeLayer === 'biome') {
          this.biomeStates.set(i, this.biomeTool);
        } else {
          if (this.resourceTool === 'erase') {this.resourceStates.set(i, 'none');} else {this.resourceStates.set(i, this.resourceTool);}
        }
        this.updateDisplayColor(i);
        this.setTileTransform(i);
      }
    }
    this.markTileTransformsDirty();
    if (this.colorAttribute) {this.colorAttribute.needsUpdate = true;}
    
    // Update hover info during painting
    if (this.hoveredIndex >= 0) {
      const tile = this.tiles[this.hoveredIndex];
      this.hovered$.next({
        index: this.hoveredIndex,
        x: tile.x,
        y: tile.y,
        worldX: tile.worldX,
        worldY: tile.worldY,
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
    if (layer === 'terrain') {this.terrainVisible = visible;} else if (layer === 'biome') {this.biomeVisible = visible;} else {this.resourcesVisible = visible;}
    
    for (let i = 0; i < this.tiles.length; i++) {
      this.updateDisplayColor(i);
      this.setTileTransform(i);
    }
    this.markTileTransformsDirty();
    if (this.colorAttribute) {this.colorAttribute.needsUpdate = true;}
  }

  setOutlinesVisible(visible: boolean): void {
    this.outlinesVisible = visible;
    if (this.outlineMesh) {this.outlineMesh.visible = visible;}
  }

  setBrushRadius(radius: number): void {
    this.brushRadius = Math.max(0, Math.floor(radius));
  }

  triggerSeedBloom(): void {
    this.seedBloomMs = this.animationTimeMs;
  }

  returnToOverview(): void {
    const previousIndex = this.selectedIndex;
    this.selectedIndex = -1;
    this.selected$.next(null);
    if (this.selectedOutline) {this.selectedOutline.visible = false;}
    if (previousIndex >= 0) {
      this.updateDisplayColor(previousIndex);
      this.setTileTransform(previousIndex);
      this.markTileTransformsDirty();
    }
    this.engine?.returnToOverview();
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
      if (t !== 'plain') {terrain.push({ index: i, state: t });}
      if (b !== 'none') {biome.push({ index: i, state: b });}
      if (r !== 'none') {resources.push({ index: i, state: r });}
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
      for (const t of payload.layers.terrain) {this.terrainStates.set(t.index, t.state);}
      for (const b of payload.layers.biome) {this.biomeStates.set(b.index, b.state);}
      for (const r of payload.layers.resources) {this.resourceStates.set(r.index, r.state);}
      for (let i = 0; i < this.tiles.length; i++) {
        this.updateDisplayColor(i);
        this.setTileTransform(i);
      }
    } else if (payload.tiles) {
      // Backward compatibility
      for (const t of payload.tiles) {
        if (t.state === 'life') {this.biomeStates.set(t.index, 'forest');} else if (t.state === 'resource') {this.resourceStates.set(t.index, 'node');}
      }
      for (let i = 0; i < this.tiles.length; i++) {
        this.updateDisplayColor(i);
        this.setTileTransform(i);
      }
    }
    this.markTileTransformsDirty();
    if (this.colorAttribute) {this.colorAttribute.needsUpdate = true;}
  }

  randomize(): void {
    if (!this.instancedMesh) {return;}
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
      this.setTileTransform(i);
    }
    this.markTileTransformsDirty();
    if (this.colorAttribute) {this.colorAttribute.needsUpdate = true;}
    this.randomizeRippleMs = this.animationTimeMs;
  }

  getCurrentConfig(): TileGridConfig | undefined { 
    return this.currentConfig ? { ...this.currentConfig } : undefined; 
  }

  clear(): void {
    if (!this.instancedMesh) {return;}
    for (let i = 0; i < this.tiles.length; i++) {
      this.terrainStates.set(i, 'plain');
      this.biomeStates.set(i, 'none');
      this.resourceStates.set(i, 'none');
      this.updateDisplayColor(i);
      this.setTileTransform(i);
    }
    this.markTileTransformsDirty();
    if (this.colorAttribute) {this.colorAttribute.needsUpdate = true;}
  }

  private selectFromHits(hits: THREE.Intersection[]): void {
    if (!this.instancedMesh) {return;}
    const hit = hits.find((h) => h.object === this.instancedMesh);
    if (!hit || (hit.instanceId ?? -1) < 0) {
      this.clearSelection();
      return;
    }
    this.applySelection(hit.instanceId as number);
  }

  /** Programmatically highlight a world by index (used by Next/Back stepping). */
  selectByIndex(index: number): void {
    if (index < 0 || index >= this.tiles.length) {return;}
    this.applySelection(index);
  }

  /** Mark which worlds carry authored content; emphasises them and scopes stepping. */
  setDocumentedWorlds(indices: number[]): void {
    this.documentedWorlds = new Set(indices.filter((i) => i >= 0 && i < this.tiles.length));
    for (let i = 0; i < this.tiles.length; i++) {
      this.updateDisplayColor(i);
      this.setTileTransform(i);
    }
    this.markTileTransformsDirty();
    if (this.colorAttribute) {this.colorAttribute.needsUpdate = true;}
  }

  /** Ordered list of worlds the stepper walks: authored worlds if any, else all. */
  private navScope(): number[] | null {
    return this.documentedWorlds.size > 0
      ? Array.from(this.documentedWorlds).sort((a, b) => a - b)
      : null;
  }

  /** Step the highlighted world forward (+1) or back (-1), wrapping around. */
  stepSelection(delta: number): void {
    const scope = this.navScope();
    if (scope) {
      const n = scope.length;
      const cur = this.selectedIndex >= 0 ? scope.indexOf(this.selectedIndex) : -1;
      const pos = cur < 0 ? (delta >= 0 ? 0 : n - 1) : (((cur + delta) % n) + n) % n;
      this.applySelection(scope[pos]);
      return;
    }
    const n = this.tiles.length;
    if (n === 0) {return;}
    const next = this.selectedIndex < 0
      ? (delta >= 0 ? 0 : n - 1)
      : (((this.selectedIndex + delta) % n) + n) % n;
    this.applySelection(next);
  }

  getSelectedIndex(): number { return this.selectedIndex; }
  getWorldCount(): number { return this.tiles.length; }

  /** Stepper readout: position within the walked scope + whether scoped to authored worlds. */
  getNavInfo(): { position: number; total: number; scoped: boolean } {
    const scope = this.navScope();
    if (scope) {
      return {
        position: this.selectedIndex >= 0 ? scope.indexOf(this.selectedIndex) : -1,
        total: scope.length,
        scoped: true
      };
    }
    return { position: this.selectedIndex, total: this.tiles.length, scoped: false };
  }

  private clearSelection(): void {
    const previousIndex = this.selectedIndex;
    this.selected$.next(null);
    this.selectedIndex = -1;
    if (this.selectedOutline) {this.selectedOutline.visible = false;}
    if (previousIndex >= 0) {
      this.updateDisplayColor(previousIndex);
      this.setTileTransform(previousIndex);
      this.markTileTransformsDirty();
    }
  }

  private applySelection(index: number): void {
    const tile = this.tiles[index];
    if (!tile) {return;}
    const previousIndex = this.selectedIndex;
    this.selected$.next({
      index,
      x: tile.x,
      y: tile.y,
      worldX: tile.worldX,
      worldY: tile.worldY,
      worldZ: tile.worldZ,
      terrain: this.terrainStates.get(index) ?? 'plain',
      biome: this.biomeStates.get(index) ?? 'none',
      resource: this.resourceStates.get(index) ?? 'none'
    });

    this.selectedIndex = index;
    if (previousIndex >= 0 && previousIndex !== index) {
      this.updateDisplayColor(previousIndex);
      this.setTileTransform(previousIndex);
    }
    this.updateDisplayColor(index);
    this.setTileTransform(index, { scale: 1.12, yOffset: 0.18 });
    this.markTileTransformsDirty();
    if (this.selectedOutline) {
      this.selectedOutline.position.set(tile.worldX, tile.worldY + this.getTileElevation(index) + 1.55, tile.worldZ);
      this.selectedOutline.visible = true;
    }
    // A bright "ping" ring marks the newly highlighted world.
    this.createSelectionPing(tile);
    this.engine?.focusOn(new THREE.Vector3(tile.worldX, tile.worldY, tile.worldZ), 4.6);
  }

  /** Quick expanding ring at a world to punctuate selection (reduced-motion safe). */
  private createSelectionPing(tile: WorldTile): void {
    if (!this.engine || this.motionReduced || !this.currentConfig) {return;}
    const cellRadius = this.currentConfig.cellRadius;
    const geometry = new THREE.RingGeometry(1, 1.12, 48);
    geometry.rotateX(-Math.PI / 2);
    const material = new THREE.MeshBasicMaterial({
      color: 0x00ffd5,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      toneMapped: false
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(tile.worldX, tile.worldY + this.getTileElevation(this.selectedIndex) + 0.1, tile.worldZ);
    mesh.renderOrder = 9;
    this.engine.add(mesh);
    this.brushPulses.push({
      mesh,
      startMs: this.animationTimeMs,
      durationMs: 720,
      maxRadius: Math.max(2, cellRadius * 2.4)
    });
  }

  // ─────────────────────────────────────────────────────────────
  // Connection Visualization
  // ─────────────────────────────────────────────────────────────

  setConnectionsVisible(visible: boolean): void {
    this.connectionsVisible = visible;
    if (this.connectionLinesGroup) {
      this.connectionLinesGroup.visible = visible;
    }
    if (this.connectionPacketsGroup) {
      this.connectionPacketsGroup.visible = visible;
    }
  }

  updateConnections(connections: WorldConnection[]): void {
    if (!this.engine) {return;}

    // Remove existing connection lines
    if (this.connectionLinesGroup) {
      this.engine.remove(this.connectionLinesGroup);
      this.connectionLinesGroup.traverse((obj) => {
        if (obj instanceof THREE.Line) {
          obj.geometry.dispose();
          (obj.material as THREE.Material).dispose();
        }
      });
      this.connectionLinesGroup = undefined;
    }
    this.removeAndDispose(this.connectionPacketsGroup);
    this.connectionPacketsGroup = undefined;
    this.connectionPackets.length = 0;

    if (connections.length === 0) {return;}

    // Create new group for connection lines
    this.connectionLinesGroup = new THREE.Group();
    this.connectionLinesGroup.renderOrder = 5;
    this.connectionPacketsGroup = new THREE.Group();
    this.connectionPacketsGroup.renderOrder = 7;

    for (const conn of connections) {
      const fromTile = this.tiles[conn.fromTileIndex];
      const toTile = this.tiles[conn.toTileIndex];
      if (!fromTile || !toTile) {continue;}

      const line = this.createConnectionLine(
        fromTile.worldX, this.getConnectionEndpointY(conn.fromTileIndex), fromTile.worldZ,
        toTile.worldX, this.getConnectionEndpointY(conn.toTileIndex), toTile.worldZ,
        conn.type,
        conn.bidirectional
      );
      this.connectionLinesGroup.add(line);
      const packet = this.createConnectionPacket(line, this.connectionPackets.length * 0.22);
      if (packet) {
        this.connectionPacketsGroup.add(packet.mesh);
        this.connectionPackets.push(packet);
      }
    }

    this.connectionLinesGroup.visible = this.connectionsVisible;
    this.connectionPacketsGroup.visible = this.connectionsVisible;
    this.engine.add(this.connectionLinesGroup);
    if (this.connectionPackets.length > 0) {
      this.engine.add(this.connectionPacketsGroup);
    }
  }

  /** Y at which a connection meets a world-orb (its centre lifted just above). */
  private getConnectionEndpointY(index: number): number {
    const tile = this.tiles[index];
    if (!tile) {return 0;}
    const cellRadius = this.currentConfig?.cellRadius ?? 1;
    return tile.worldY + this.getTileElevation(index) + cellRadius * 0.3;
  }

  private createConnectionLine(
    x1: number, y1: number, z1: number,
    x2: number, y2: number, z2: number,
    type: ConnectionType,
    _bidirectional: boolean
  ): THREE.Line {
    const style = CONNECTION_STYLES[type];
    const color = new THREE.Color(style.color);
    const cellRadius = this.currentConfig?.cellRadius ?? 1;

    // Bow the route outward in the XZ plane and upward through space so it reads
    // as an arc between two floating worlds rather than a flat line on the ground.
    const dx = x2 - x1;
    const dz = z2 - z1;
    const len = Math.max(0.001, Math.sqrt(dx * dx + dz * dz));
    const curveAmount = len * 0.14;
    const perpX = -dz / len * curveAmount;
    const perpZ = dx / len * curveAmount;
    const midX = (x1 + x2) / 2 + perpX;
    const midZ = (z1 + z2) / 2 + perpZ;
    const midY = Math.max(y1, y2) + len * 0.16 + cellRadius * 0.9;

    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(x1, y1, z1),
      new THREE.Vector3(midX, midY, midZ),
      new THREE.Vector3(x2, y2, z2)
    );

    const points = curve.getPoints(32);
    const geometry = new THREE.BufferGeometry().setFromPoints(points);

    let material: THREE.LineBasicMaterial | THREE.LineDashedMaterial;

    if (style.dashPattern) {
      material = new THREE.LineDashedMaterial({
        color,
        dashSize: style.dashPattern[0] * 0.05,
        gapSize: style.dashPattern[1] * 0.05,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        toneMapped: false
      });
    } else {
      material = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        toneMapped: false
      });
    }

    const line = new THREE.Line(geometry, material);
    line.userData['curve'] = curve;
    line.userData['color'] = color.getHex();

    if (style.dashPattern) {
      line.computeLineDistances();
    }

    return line;
  }

  private createConnectionPacket(line: THREE.Line, offset: number): ConnectionPacket | null {
    if (this.motionReduced) {return null;}
    const curve = line.userData['curve'] as THREE.QuadraticBezierCurve3 | undefined;
    const color = line.userData['color'] as number | undefined;
    if (!curve || color === undefined) {return null;}

    const geometry = new THREE.SphereGeometry(0.28, 12, 8);
    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.75,
      depthWrite: false
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.renderOrder = 8;
    return {
      mesh,
      curve,
      offset: offset % 1,
      speed: 0.16
    };
  }

  getTileWorldPosition(index: number): { x: number; z: number } | null {
    const tile = this.tiles[index];
    if (!tile) {return null;}
    return { x: tile.worldX, z: tile.worldZ };
  }

  /** World-space anchor for a floating label, lifted above the orb + its beacon. */
  getWorldLabelAnchor(index: number): { x: number; y: number; z: number } | null {
    const tile = this.tiles[index];
    if (!tile) {return null;}
    const cellRadius = this.currentConfig?.cellRadius ?? 1;
    const y = tile.worldY + this.getTileElevation(index)
      + cellRadius * (1.1 + this.getBeaconHeight(index) * 0.55);
    return { x: tile.worldX, y, z: tile.worldZ };
  }
}