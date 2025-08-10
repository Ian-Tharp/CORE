import { AfterViewInit, Component, DestroyRef, ElementRef, HostListener, OnDestroy, ViewChild, inject } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { EngineService } from './engine/engine.service';
import { HexWorldService, TerrainState, BiomeState, ResourceState } from './engine/hex-world.service';
import { ProjectService } from './engine/project.service';
import { WorldsService } from '../../services/worlds/worlds.service';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { SaveWorldDialogComponent } from './save-world-dialog/save-world-dialog.component';
import { WorldsDialogComponent } from './worlds-dialog/worlds-dialog.component';
import { UiNotifyService } from '../../shared/services/ui-notify.service';

@Component({
  selector: 'app-command-center',
  imports: [CommonModule, FormsModule, MatDialogModule],
  templateUrl: './command-center.component.html',
  styleUrl: './command-center.component.scss'
})
export class CommandCenterComponent implements AfterViewInit, OnDestroy {
  @ViewChild('canvas', { static: true }) private readonly canvasRef!: ElementRef<HTMLCanvasElement>;

  private readonly destroyRef = inject(DestroyRef);
  private readonly engine = inject(EngineService);
  private readonly hexWorld = inject(HexWorldService);
  private readonly projects = inject(ProjectService);
  private readonly worlds = inject(WorldsService);
  private readonly dialog = inject(MatDialog);
  private readonly route = inject(ActivatedRoute);
  private readonly ui = inject(UiNotifyService);

  isInitialized = false;
  projectName = 'My World';
  public worldConfig: { radius: number; gridWidth: number; gridHeight: number; elevation: number } = {
    radius: 1,
    gridWidth: 50,
    gridHeight: 50,
    elevation: 0.1
  };
  public seed: string = '';
  // Layers & tools
  activeLayer: 'terrain' | 'biome' | 'resources' = 'terrain';
  terrainTool: TerrainState = 'plain';
  biomeTool: BiomeState = 'forest';
  resourceTool: ResourceState | 'erase' = 'node';
  layerVisibility: { terrain: boolean; biome: boolean; resources: boolean } = { terrain: true, biome: true, resources: true };
  brush = 1;
  outlinesVisible = false;
  hoveredInfo: { index: number; q: number; r: number; s: number; x: number; y: number; z: number; terrain: string; biome: string; resource: string } | null = null;
  contextMenu: { visible: boolean; x: number; y: number; index: number | null; q?: number; r?: number } = { visible: false, x: 0, y: 0, index: null };
  isEditMode = true;
  selectedInfo: { index: number; q: number; r: number; s: number; x: number; y: number; z: number; terrain: string; biome: string; resource: string } | null = null;

  ngAfterViewInit(): void {
    const canvas = this.canvasRef.nativeElement;
    this.engine.initialize(canvas);
    this.hexWorld.initialize(this.engine);
    this.hexWorld.onHoverChanged().subscribe((h) => { this.hoveredInfo = h; });
    this.hexWorld.onSelectedChanged().subscribe((s) => { this.selectedInfo = s; });
    this.hexWorld.onTileContext((ctx) => {
      this.contextMenu = { visible: true, x: ctx.screen.x, y: ctx.screen.y, index: ctx.index, q: ctx.q, r: ctx.r };
    });
    this.hexWorld.createHexPlane(this.worldConfig);
    this.engine.start();
    this.isInitialized = true;
    // defaults
    this._syncWorldStateToService();

    this.destroyRef.onDestroy(() => this.ngOnDestroy());

    // Load snapshot if provided via query param (local or remote)
    const params = this.route.snapshot.queryParamMap;
    const pid = params.get('projectId');
    if (pid) {
      const snap = this.projects.load(pid);
      if (snap) {
        this.projectName = snap.name;
        this.hexWorld.restore(snap.name, { config: snap.config, tiles: snap.tiles, layers: snap.layers });
      }
    } else {
      const worldId = params.get('worldId');
      if (worldId) {
        this.worlds.getLatestSnapshot(worldId).subscribe({
          next: (snap) => {
            this.projectName = 'World';
            this.hexWorld.restore('World', { config: snap.config, tiles: snap.tiles, layers: snap.layers });
          }
        });
      } else {
        const name = params.get('name');
        const seed = params.get('seed');
        if (name) this.projectName = name;
        if (seed) { this.seed = seed; this.onApplySeed(); }
      }
    }
  }

  ngOnDestroy(): void {
    this.engine.dispose();
  }

  onSave(): void {
    const snap = this.hexWorld.snapshot(this.projectName);
    this.projects.save({ name: snap.name, config: snap.config, layers: (snap as any).layers });
    // Fire-and-forget remote persistence (prototype)
    this._capturePreview((preview) => {
      this.worlds.saveFromHexSnapshot(snap.name, { config: snap.config as any, layers: (snap as any).layers, preview }).subscribe({
      next: () => {},
      error: () => {}
      });
    });
  }

  onLoadLatest(): void {
    // Try remote first; fallback to local
    this.worlds.listWorlds(1, 0).subscribe({
      next: (list) => {
        if (list && list.length > 0) {
          const world = list[0];
          this.worlds.getLatestSnapshot(world.id).subscribe({
            next: (snap) => {
              this.projectName = world.name;
              this.hexWorld.restore(world.name, { config: snap.config, tiles: snap.tiles, layers: snap.layers });
            },
            error: () => this._loadLatestLocal()
          });
        } else {
          this._loadLatestLocal();
        }
      },
      error: () => this._loadLatestLocal()
    });
  }

  private _loadLatestLocal(): void {
    const all = this.projects.list().sort((a, b) => b.createdAt.localeCompare(a.createdAt));
    const latest = all[0];
    if (latest) {
      this.projectName = latest.name;
      this.hexWorld.restore(latest.name, { config: latest.config, tiles: latest.tiles, layers: latest.layers });
    }
  }

  public onSaveAs(): void {
    const snap = this.hexWorld.snapshot(this.projectName);
    const dialogRef = this.dialog.open(SaveWorldDialogComponent, { data: { defaultName: snap.name }, panelClass: 'glass-dialog' });
    dialogRef.afterClosed().subscribe((name?: string) => {
      if (!name) return;
      this.projectName = name;
      // local
      this.projects.save({ name, config: snap.config, layers: (snap as any).layers });
      // remote
      this._capturePreview((preview) => this.worlds.saveFromHexSnapshot(name, { config: snap.config as any, layers: (snap as any).layers, preview }).subscribe());
    });
  }

  private _capturePreview(cb: (dataUrl: string) => void): void {
    const canvas = this.canvasRef.nativeElement;
    const prev = this.outlinesVisible;
    this.hexWorld.setOutlinesVisible(true);
    requestAnimationFrame(() => {
      const dataUrl = canvas.toDataURL('image/png');
      this.hexWorld.setOutlinesVisible(prev);
      cb(dataUrl);
    });
  }

  public onOpenWorlds(): void {
    const ref = this.dialog.open(WorldsDialogComponent, { data: { limit: 50 }, panelClass: 'glass-dialog' });
    ref.afterClosed().subscribe((world?: { id: string; name: string }) => {
      if (!world) return;
      this.worlds.getLatestSnapshot(world.id).subscribe({
        next: (snap) => {
          this.projectName = world.name;
          this.hexWorld.restore(world.name, { config: snap.config, tiles: snap.tiles, layers: snap.layers });
        },
        error: (err) => {
          this.ui.showError('No snapshot found for that world yet. Try Quick Save from Command Center.');
        }
      });
    });
  }

  public onApplyWorldConfig(): void {
    this.hexWorld.createHexPlane(this.worldConfig);
    this._syncWorldStateToService();
  }

  public onApplySeed(): void {
    const trimmed = (this.seed ?? '').toString().trim();
    this.hexWorld.setRandomSeed(trimmed.length > 0 ? trimmed : null);
  }

  onRandomize(): void {
    this.hexWorld.randomize();
  }

  onClear(): void {
    this.hexWorld.clear();
  }

  onCloseContextMenu(): void {
    this.contextMenu.visible = false;
  }

  onEnterWorld(): void {
    this.contextMenu.visible = false;
    this.isLoading = true;
    setTimeout(() => { this.isLoading = false; }, 2000);
  }

  isLoading = false;

  onActiveLayerChange(next: 'terrain' | 'biome' | 'resources'): void {
    this.activeLayer = next;
    this.hexWorld.setActiveLayer(next);
  }

  onTerrainToolChange(next: TerrainState): void {
    this.terrainTool = next;
    this.hexWorld.setTerrainTool(next);
  }
  onBiomeToolChange(next: BiomeState): void {
    this.biomeTool = next;
    this.hexWorld.setBiomeTool(next);
  }
  onResourceToolChange(next: ResourceState | 'erase'): void {
    this.resourceTool = next;
    this.hexWorld.setResourceTool(next);
  }
  onToggleLayerVisibility(layer: 'terrain' | 'biome' | 'resources', value: boolean): void {
    this.layerVisibility[layer] = value;
    this.hexWorld.setLayerVisibility(layer, value);
  }

  onToggleEditMode(): void {
    this.isEditMode = !this.isEditMode;
    this.hexWorld.setEditMode(this.isEditMode);
    // Hide context menu when entering edit mode
    if (this.isEditMode) this.contextMenu.visible = false;
  }

  private _syncWorldStateToService(): void {
    this.hexWorld.setActiveLayer(this.activeLayer);
    this.hexWorld.setTerrainTool(this.terrainTool);
    this.hexWorld.setBiomeTool(this.biomeTool);
    this.hexWorld.setResourceTool(this.resourceTool);
    this.hexWorld.setEditMode(this.isEditMode);
    // explicitly sync visibility to service to avoid any initial template-driven toggle mismatch
    this.hexWorld.setLayerVisibility('terrain', this.layerVisibility.terrain);
    this.hexWorld.setLayerVisibility('biome', this.layerVisibility.biome);
    this.hexWorld.setLayerVisibility('resources', this.layerVisibility.resources);
    this.hexWorld.setOutlinesVisible(this.outlinesVisible);
  }

  @HostListener('window:keydown', ['$event'])
  onKeyDown(e: KeyboardEvent): void {
    if (e.key === 'h' || e.key === 'H') {
      this.hexWorld.setOutlinesVisible(!(this as any)._outlinesVisibleInternal);
      (this as any)._outlinesVisibleInternal = !(this as any)._outlinesVisibleInternal;
    } else if (e.key === '+' || e.key === '=') {
      this.brush = Math.min(6, this.brush + 1);
      this.onBrushChange(this.brush);
    } else if (e.key === '-' || e.key === '_') {
      this.brush = Math.max(0, this.brush - 1);
      this.onBrushChange(this.brush);
    } else if (e.key === '1') {
      this.onActiveLayerChange('terrain');
    } else if (e.key === '2') {
      this.onActiveLayerChange('biome');
    } else if (e.key === '3') {
      this.onActiveLayerChange('resources');
    }
  }

  onBrushChange(next: number): void {
    this.hexWorld.setBrushRadius(next);
  }

  onToggleOutlines(value: boolean): void {
    this.outlinesVisible = value;
    this.hexWorld.setOutlinesVisible(value);
  }
}
