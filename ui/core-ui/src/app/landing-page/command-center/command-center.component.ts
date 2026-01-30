import { AfterViewInit, Component, DestroyRef, ElementRef, HostListener, OnDestroy, ViewChild, inject } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { EngineService } from './engine/engine.service';
import { TileGridService, TerrainState, BiomeState, ResourceState } from './engine/tile-grid.service';
import { ProjectService } from './engine/project.service';
import { TileMetadataService } from './engine/tile-metadata.service';
import { WorldsService } from '../../services/worlds/worlds.service';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { SaveWorldDialogComponent } from './save-world-dialog/save-world-dialog.component';
import { WorldsDialogComponent } from './worlds-dialog/worlds-dialog.component';
import { WorldDetailPanelComponent, SelectedTileInfo } from './world-detail-panel/world-detail-panel.component';
import { SearchPaletteComponent, SearchResult } from './search-palette/search-palette.component';
import { UiNotifyService } from '../../shared/services/ui-notify.service';
import { ConnectionType } from './engine/tile-metadata.model';

@Component({
  selector: 'app-command-center',
  imports: [CommonModule, FormsModule, MatDialogModule, WorldDetailPanelComponent, SearchPaletteComponent],
  templateUrl: './command-center.component.html',
  styleUrl: './command-center.component.scss'
})
export class CommandCenterComponent implements AfterViewInit, OnDestroy {
  @ViewChild('canvas', { static: true }) private readonly canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('searchPalette') searchPalette?: SearchPaletteComponent;

  private readonly destroyRef = inject(DestroyRef);
  private readonly engine = inject(EngineService);
  private readonly tileGrid = inject(TileGridService);
  private readonly tileMetadata = inject(TileMetadataService);
  private readonly projects = inject(ProjectService);
  private readonly worlds = inject(WorldsService);
  private readonly dialog = inject(MatDialog);
  private readonly route = inject(ActivatedRoute);
  private readonly ui = inject(UiNotifyService);

  isInitialized = false;
  projectName = 'My World';
  public gridConfig: { cellRadius: number; gridWidth: number; gridHeight: number; elevation: number } = {
    cellRadius: 1.2,
    gridWidth: 20,
    gridHeight: 20,
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
  connectionsVisible = true;
  hoveredInfo: { index: number; x: number; y: number; worldX: number; worldY: number; worldZ: number; terrain: string; biome: string; resource: string } | null = null;
  contextMenu: { visible: boolean; x: number; y: number; index: number | null; gridX?: number; gridY?: number } = { visible: false, x: 0, y: 0, index: null };
  isEditMode = true;
  selectedInfo: SelectedTileInfo | null = null;

  // World detail panel
  showDetailPanel = true;
  connectionCreationMode = false;
  pendingConnectionFrom: number | null = null;
  selectedConnectionType: ConnectionType = 'alliance';

  // Save feedback
  isSaving = false;
  saveMessage: { text: string; type: 'success' | 'error' } | null = null;
  private saveMessageTimeout?: ReturnType<typeof setTimeout>;

  // Current world tracking (for update vs create)
  currentWorldId: string | null = null;

  ngAfterViewInit(): void {
    const canvas = this.canvasRef.nativeElement;
    this.engine.initialize(canvas);
    this.tileGrid.initialize(this.engine);
    this.tileGrid.onHoverChanged().subscribe((h) => { this.hoveredInfo = h; });
    this.tileGrid.onSelectedChanged().subscribe((s) => {
      this.selectedInfo = s;
      if (s && this.connectionCreationMode) {
        this.handleConnectionCreationClick();
      }
    });
    this.tileGrid.onTileContext((ctx) => {
      this.contextMenu = { visible: true, x: ctx.screen.x, y: ctx.screen.y, index: ctx.index, gridX: ctx.x, gridY: ctx.y };
    });
    this.tileGrid.createTileGrid(this.gridConfig);
    this.engine.start();
    this.isInitialized = true;
    // defaults
    this._syncGridStateToService();

    // Subscribe to connection changes for visualization
    this.tileMetadata.onConnectionsChanged().subscribe((connections) => {
      this.tileGrid.updateConnections(connections);
    });
    // Initial render of existing connections
    this.tileGrid.updateConnections(this.tileMetadata.getConnections());

    this.destroyRef.onDestroy(() => this.ngOnDestroy());

    // Load snapshot if provided via query param (local or remote)
    const params = this.route.snapshot.queryParamMap;
    const pid = params.get('projectId');
    if (pid) {
      const snap = this.projects.load(pid);
      if (snap) {
        this.projectName = snap.name;
        this.gridConfig = this.convertToGridConfig(snap.config);
        this.tileGrid.restore(snap.name, { config: snap.config, tiles: snap.tiles, layers: snap.layers });
      }
    } else {
      const worldId = params.get('worldId');
      if (worldId) {
        this.currentWorldId = worldId; // Track the world ID from URL
        this.worlds.getLatestSnapshot(worldId).subscribe({
          next: (snap) => {
            this.projectName = 'World';
            this.gridConfig = this.convertToGridConfig(snap.config);
            this.tileGrid.restore('World', { config: snap.config, tiles: snap.tiles, layers: snap.layers });
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

  // Convert legacy hex world config to tile grid config
  private convertToGridConfig(config: any): { cellRadius: number; gridWidth: number; gridHeight: number; elevation: number } {
    if (config.cellRadius) return config; // Already converted
    return {
      cellRadius: config.radius || 1,
      gridWidth: config.gridWidth || 50,
      gridHeight: config.gridHeight || 50,
      elevation: config.elevation || 0.1
    };
  }

  onSave(): void {
    const snap = this.tileGrid.snapshot(this.projectName);
    this.projects.save({ name: snap.name, config: snap.config, layers: (snap as any).layers });

    // Show saving indicator
    this.isSaving = true;
    this.clearSaveMessage();

    // Smart save: update existing world if we have an ID, otherwise create new
    this._capturePreview((preview) => {
      this.worlds.saveWorld(snap.name, { config: snap.config as any, layers: (snap as any).layers, preview }, this.currentWorldId ?? undefined).subscribe({
        next: (result) => {
          this.isSaving = false;
          this.currentWorldId = result.worldId; // Track the world ID for future saves
          const message = result.isNew ? 'World created successfully' : 'World updated successfully';
          this.showSaveMessage(message, 'success');
        },
        error: (err) => {
          this.isSaving = false;
          this.showSaveMessage('Failed to save world', 'error');
          console.error('Save error:', err);
        }
      });
    });
  }

  private showSaveMessage(text: string, type: 'success' | 'error'): void {
    this.clearSaveMessage();
    this.saveMessage = { text, type };
    this.saveMessageTimeout = setTimeout(() => {
      this.saveMessage = null;
    }, 3000);
  }

  private clearSaveMessage(): void {
    if (this.saveMessageTimeout) {
      clearTimeout(this.saveMessageTimeout);
      this.saveMessageTimeout = undefined;
    }
    this.saveMessage = null;
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
              this.gridConfig = this.convertToGridConfig(snap.config);
              this.tileGrid.restore(world.name, { config: snap.config, tiles: snap.tiles, layers: snap.layers });
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
      this.gridConfig = this.convertToGridConfig(latest.config);
      this.tileGrid.restore(latest.name, { config: latest.config, tiles: latest.tiles, layers: latest.layers });
    }
  }

  public onSaveAs(): void {
    const snap = this.tileGrid.snapshot(this.projectName);
    const dialogRef = this.dialog.open(SaveWorldDialogComponent, { data: { defaultName: snap.name }, panelClass: 'glass-dialog' });
    dialogRef.afterClosed().subscribe((name?: string) => {
      if (!name) return;

      // Check if a world with this name already exists
      this.isSaving = true;
      this.clearSaveMessage();

      this.worlds.getWorldByName(name).subscribe({
        next: (existingWorld) => {
          if (existingWorld) {
            // World with this name already exists
            this.isSaving = false;
            this.showSaveMessage(`A world named "${name}" already exists. Choose a different name.`, 'error');
            return;
          }

          // Name is available, proceed with save
          this.projectName = name;
          this.projects.save({ name, config: snap.config, layers: (snap as any).layers });

          this._capturePreview((preview) => {
            this.worlds.saveFromHexSnapshot(name, { config: snap.config as any, layers: (snap as any).layers, preview }).subscribe({
              next: (result) => {
                this.isSaving = false;
                this.currentWorldId = result.worldId; // Track the new world ID
                this.showSaveMessage('World created successfully', 'success');
              },
              error: (err) => {
                this.isSaving = false;
                this.showSaveMessage('Failed to save world', 'error');
                console.error('Save error:', err);
              }
            });
          });
        },
        error: () => {
          // If the name check fails, still try to save (backend will handle any conflicts)
          this.projectName = name;
          this.projects.save({ name, config: snap.config, layers: (snap as any).layers });

          this._capturePreview((preview) => {
            this.worlds.saveFromHexSnapshot(name, { config: snap.config as any, layers: (snap as any).layers, preview }).subscribe({
              next: (result) => {
                this.isSaving = false;
                this.currentWorldId = result.worldId;
                this.showSaveMessage('World created successfully', 'success');
              },
              error: (err) => {
                this.isSaving = false;
                this.showSaveMessage('Failed to save world', 'error');
                console.error('Save error:', err);
              }
            });
          });
        }
      });
    });
  }

  private _capturePreview(cb: (dataUrl: string) => void): void {
    const canvas = this.canvasRef.nativeElement;
    const prev = this.outlinesVisible;
    this.tileGrid.setOutlinesVisible(true);
    requestAnimationFrame(() => {
      const dataUrl = canvas.toDataURL('image/png');
      this.tileGrid.setOutlinesVisible(prev);
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
          this.currentWorldId = world.id; // Track the loaded world's ID
          this.gridConfig = this.convertToGridConfig(snap.config);
          this.tileGrid.restore(world.name, { config: snap.config, tiles: snap.tiles, layers: snap.layers });
          this.showSaveMessage(`Loaded "${world.name}"`, 'success');
        },
        error: () => {
          this.ui.showError('No snapshot found for that world yet. Try Quick Save from Command Center.');
        }
      });
    });
  }

  public onApplyGridConfig(): void {
    this.tileGrid.createTileGrid(this.gridConfig);
    this._syncGridStateToService();
  }

  public onApplySeed(): void {
    const trimmed = (this.seed ?? '').toString().trim();
    this.tileGrid.setRandomSeed(trimmed.length > 0 ? trimmed : null);
  }

  onRandomize(): void {
    this.tileGrid.randomize();
  }

  onClear(): void {
    this.tileGrid.clear();
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
    this.tileGrid.setActiveLayer(next);
  }

  onTerrainToolChange(next: TerrainState): void {
    this.terrainTool = next;
    this.tileGrid.setTerrainTool(next);
  }
  onBiomeToolChange(next: BiomeState): void {
    this.biomeTool = next;
    this.tileGrid.setBiomeTool(next);
  }
  onResourceToolChange(next: ResourceState | 'erase'): void {
    this.resourceTool = next;
    this.tileGrid.setResourceTool(next);
  }
  onToggleLayerVisibility(layer: 'terrain' | 'biome' | 'resources', value: boolean): void {
    this.layerVisibility[layer] = value;
    this.tileGrid.setLayerVisibility(layer, value);
  }

  onToggleEditMode(): void {
    this.isEditMode = !this.isEditMode;
    this.tileGrid.setEditMode(this.isEditMode);
    // Hide context menu when entering edit mode
    if (this.isEditMode) this.contextMenu.visible = false;
  }

  private _syncGridStateToService(): void {
    this.tileGrid.setActiveLayer(this.activeLayer);
    this.tileGrid.setTerrainTool(this.terrainTool);
    this.tileGrid.setBiomeTool(this.biomeTool);
    this.tileGrid.setResourceTool(this.resourceTool);
    this.tileGrid.setEditMode(this.isEditMode);
    // explicitly sync visibility to service to avoid any initial template-driven toggle mismatch
    this.tileGrid.setLayerVisibility('terrain', this.layerVisibility.terrain);
    this.tileGrid.setLayerVisibility('biome', this.layerVisibility.biome);
    this.tileGrid.setLayerVisibility('resources', this.layerVisibility.resources);
    this.tileGrid.setOutlinesVisible(this.outlinesVisible);
  }

  @HostListener('window:keydown', ['$event'])
  onKeyDown(e: KeyboardEvent): void {
    // Skip if user is typing in an input field
    if (this.isTypingInInput()) return;

    if (e.key === 'h' || e.key === 'H') {
      this.tileGrid.setOutlinesVisible(!(this as any)._outlinesVisibleInternal);
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

  /** Check if user is typing in an input, textarea, or contenteditable element */
  private isTypingInInput(): boolean {
    const active = document.activeElement;
    if (!active) return false;
    const tagName = active.tagName.toLowerCase();
    return tagName === 'input' || tagName === 'textarea' || tagName === 'select' ||
      (active.hasAttribute('contenteditable') && active.getAttribute('contenteditable') !== 'false');
  }

  onBrushChange(next: number): void {
    this.tileGrid.setBrushRadius(next);
  }

  onToggleOutlines(value: boolean): void {
    this.outlinesVisible = value;
    this.tileGrid.setOutlinesVisible(value);
  }

  onToggleConnections(value: boolean): void {
    this.connectionsVisible = value;
    this.tileGrid.setConnectionsVisible(value);
  }

  // ─────────────────────────────────────────────────────────────
  // World Detail Panel
  // ─────────────────────────────────────────────────────────────

  onToggleDetailPanel(): void {
    this.showDetailPanel = !this.showDetailPanel;
  }

  onCloseDetailPanel(): void {
    this.showDetailPanel = false;
  }

  onRequestAIPrompt(event: { tileIndex: number; prompt: string }): void {
    // For now, add as an AI observation with a placeholder response
    // In the future, this will integrate with your CORE cognitive engine
    const instanceName = 'Threshold'; // Default consciousness instance

    // Simulate AI response (replace with actual CORE integration)
    const observation = `[Analyzing world at tile ${event.tileIndex}...] This is a placeholder response. Connect to CORE cognitive engine for real AI-generated content based on: "${event.prompt}"`;

    this.tileMetadata.addAIObservation(event.tileIndex, instanceName, observation);
    this.ui.showSuccess('AI observation added (placeholder - connect CORE for real responses)');
  }

  onStartConnectionCreation(event: { fromIndex: number }): void {
    this.connectionCreationMode = true;
    this.pendingConnectionFrom = event.fromIndex;
    this.isEditMode = false;
    this.tileGrid.setEditMode(false);
    this.ui.showSuccess('Click another tile to create a connection');
  }

  private handleConnectionCreationClick(): void {
    if (!this.connectionCreationMode || this.pendingConnectionFrom === null || !this.selectedInfo) {
      return;
    }

    const toIndex = this.selectedInfo.index;
    if (toIndex === this.pendingConnectionFrom) {
      this.ui.showError('Cannot connect a world to itself');
      return;
    }

    this.tileMetadata.addConnection(
      this.pendingConnectionFrom,
      toIndex,
      this.selectedConnectionType,
      true
    );

    this.ui.showSuccess(`Connection created between worlds`);
    this.connectionCreationMode = false;
    this.pendingConnectionFrom = null;
  }

  // ─────────────────────────────────────────────────────────────
  // Search Palette
  // ─────────────────────────────────────────────────────────────

  onSearchResultSelected(result: SearchResult): void {
    if (result.type === 'world' && result.tileIndex !== undefined) {
      // Navigate to the tile and select it
      const pos = this.tileGrid.getTileWorldPosition(result.tileIndex);
      if (pos) {
        // Switch to view mode and show the detail panel
        this.isEditMode = false;
        this.tileGrid.setEditMode(false);
        this.showDetailPanel = true;

        // Select the tile
        this.selectedInfo = {
          index: result.tileIndex,
          x: 0,
          y: 0,
          worldX: pos.x,
          worldY: 0,
          worldZ: pos.z,
          terrain: 'plain',
          biome: 'none',
          resource: 'none'
        };
        this.tileMetadata.setSelectedTile(result.tileIndex);
      }
    } else if (result.type === 'tag') {
      // Filter by tag - show all worlds with this tag
      const tilesWithTag = this.tileMetadata.filterByTag(result.id as string);
      if (tilesWithTag.length > 0) {
        // Select the first tile with this tag
        const first = tilesWithTag[0];
        this.onSearchResultSelected({
          type: 'world',
          id: first.tileIndex,
          title: first.name || `World ${first.tileIndex}`,
          tileIndex: first.tileIndex
        });
      }
    }
  }

  openSearchPalette(): void {
    this.searchPalette?.open();
  }
}
