import { AfterViewInit, Component, DestroyRef, ElementRef, HostListener, OnDestroy, QueryList, ViewChild, ViewChildren, inject } from '@angular/core';
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
import { UniversePickerDialogComponent, UniversePickerResult } from './universe-picker-dialog/universe-picker-dialog.component';
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
  @ViewChildren('worldLabelEl') private worldLabelEls?: QueryList<ElementRef<HTMLElement>>;

  /** Named worlds to float labels over (positioned each frame from the 3D projection). */
  worldLabels: Array<{ index: number; name: string }> = [];
  private removeLabelUpdater?: () => void;

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
  layerVisibility: { terrain: boolean; biome: boolean; resources: boolean } =
    { terrain: true, biome: true, resources: true };
  brush = 1;
  outlinesVisible = false;
  connectionsVisible = true;
  hoveredInfo: {
    index: number; x: number; y: number; worldX: number; worldY: number;
    worldZ: number; terrain: string; biome: string; resource: string
  } | null = null;
  contextMenu: { visible: boolean; x: number; y: number; index: number | null; gridX?: number; gridY?: number } =
    { visible: false, x: 0, y: 0, index: null };
  isEditMode = false;
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
  public viewportScanActive = false;
  private viewportScanTimeout?: ReturnType<typeof setTimeout>;
  public worldStepLabel = '';
  private worldStepLabelTimeout?: ReturnType<typeof setTimeout>;

  // Current world tracking (for update vs create)
  currentWorldId: string | null = null;

  ngAfterViewInit(): void {
    const canvas = this.canvasRef.nativeElement;
    this.engine.initialize(canvas);
    this.tileGrid.initialize(this.engine);
    this.tileGrid.onHoverChanged().subscribe((h) => { this.hoveredInfo = h; });
    this.tileGrid.onSelectedChanged().subscribe((s) => {
      this.selectedInfo = s;
      this.queueWorldStepLabelUpdate();
      if (s && this.connectionCreationMode) {
        this.handleConnectionCreationClick();
      }
    });
    this.tileGrid.onTileContext((ctx) => {
      this.contextMenu = {
        visible: true, x: ctx.screen.x, y: ctx.screen.y, index: ctx.index, gridX: ctx.x, gridY: ctx.y
      };
    });
    this.tileGrid.createTileGrid(this.gridConfig);
    this.engine.start();
    this.isInitialized = true;
    // defaults
    this._syncGridStateToService();
    this.queueWorldStepLabelUpdate();

    // Subscribe to connection changes for visualization
    this.tileMetadata.onConnectionsChanged().subscribe((connections) => {
      this.tileGrid.updateConnections(connections);
    });
    // Initial render of existing connections
    this.tileGrid.updateConnections(this.tileMetadata.getConnections());

    // Emphasise authored worlds (name / wiki / tags) in the galaxy and scope the
    // Next/Back stepper to them.
    this.tileMetadata.onAllMetadataChanged().subscribe((map) => {
      const documented: number[] = [];
      const labels: Array<{ index: number; name: string }> = [];
      map.forEach((meta, idx) => {
        const name = meta.name?.trim();
        if (name || (meta.wikiPageIds && meta.wikiPageIds.length > 0) || (meta.tags && meta.tags.length > 0)) {
          documented.push(idx);
        }
        if (name) { labels.push({ index: idx, name }); }
      });
      // Defer to a microtask so neither the documented-world emphasis (which the
      // stepper label reads via getNavInfo) nor the labels mutate view-bound state
      // during the change-detection pass that triggered this emission (NG0100).
      Promise.resolve().then(() => {
        this.tileGrid.setDocumentedWorlds(documented);
        this.worldLabels = labels;
        this.queueWorldStepLabelUpdate();
      });
    });

    // Float named-world labels above their orbs, following the camera each frame.
    this.removeLabelUpdater?.();
    this.removeLabelUpdater = this.engine.onBeforeRender(() => this.positionWorldLabels());

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
        this.loadWorldById(worldId, 'World'); // deep-link load bypasses the picker
      } else {
        const name = params.get('name');
        const seed = params.get('seed');
        if (name) {this.projectName = name;}
        if (seed) { this.seed = seed; this.onApplySeed(); }
        if (!name && !seed && !sessionStorage.getItem('core.universePicker.seen')) {
          // No deep-link target, first entry this session → greet with the
          // universe picker. Deferred to a microtask so opening the dialog can't
          // mutate view-bound state during the first-render CD pass (NG0100).
          sessionStorage.setItem('core.universePicker.seen', '1');
          Promise.resolve().then(() => this.openUniversePicker());
        }
      }
    }
  }

  /** Restore a world (snapshot geometry + tile metadata) by id — the single
   *  load routine shared by the URL deep-link, the Worlds dialog, and the
   *  first-load universe picker. */
  private loadWorldById(id: string, name: string): void {
    this.currentWorldId = id;
    this.worlds.getLatestSnapshot(id).subscribe({
      next: (snap) => {
        this.projectName = name;
        this.gridConfig = this.convertToGridConfig(snap.config);
        this.tileGrid.restore(name, { config: snap.config, tiles: snap.tiles, layers: snap.layers });
        this._hydrateMetadata(id);
        this.showSaveMessage(`Loaded "${name}"`, 'success');
      },
      error: () => {
        this.ui.showError('No snapshot found for that world yet. Try Quick Save from Command Center.');
      }
    });
  }

  /** First-load gateway: chart a new universe or descend into a saved one. */
  private openUniversePicker(): void {
    const ref = this.dialog.open(UniversePickerDialogComponent, {
      data: { limit: 50 },
      panelClass: 'glass-dialog',
      disableClose: true
    });
    ref.afterClosed().subscribe((res?: UniversePickerResult) => {
      if (res?.action === 'load') {
        this.loadWorldById(res.world.id, res.world.name);
      } else {
        // Chart a New Universe → keep the fresh default grid; the next save
        // creates a brand-new world (currentWorldId stays null).
        this.currentWorldId = null;
        this._triggerViewportScan();
      }
    });
  }

  ngOnDestroy(): void {
    if (this.viewportScanTimeout) {
      clearTimeout(this.viewportScanTimeout);
    }
    if (this.worldStepLabelTimeout) {
      clearTimeout(this.worldStepLabelTimeout);
    }
    this.removeLabelUpdater?.();
    this.engine.dispose();
  }

  /**
   * Position the floating world labels each frame by projecting each world's 3D
   * anchor to screen. Runs outside Angular (engine render loop), so it mutates
   * the DOM directly to avoid per-frame change detection.
   */
  private positionWorldLabels(): void {
    const els = this.worldLabelEls?.toArray();
    if (!els || els.length === 0) { return; }
    const rect = this.canvasRef?.nativeElement?.getBoundingClientRect();
    for (let i = 0; i < els.length && i < this.worldLabels.length; i++) {
      const el = els[i].nativeElement;
      if (this.isEditMode) { el.style.opacity = '0'; continue; }
      const anchor = this.tileGrid.getWorldLabelAnchor(this.worldLabels[i].index);
      const screen = anchor ? this.engine.projectPoint(anchor.x, anchor.y, anchor.z) : null;
      // Strictly clip to the canvas so labels never spill over the sidebar/panel.
      const offscreen = !screen || (rect && (
        screen.x < rect.left || screen.x > rect.right ||
        screen.y < rect.top || screen.y > rect.bottom
      ));
      if (offscreen) { el.style.opacity = '0'; continue; }
      el.style.left = `${screen!.x}px`;
      el.style.top = `${screen!.y}px`;
      el.style.opacity = '1';
    }
  }

  // Convert legacy hex world config to tile grid config
  private convertToGridConfig(
    config: any
  ): { cellRadius: number; gridWidth: number; gridHeight: number; elevation: number } {
    if (config.cellRadius) {return config;} // Already converted
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
      this.worlds.saveWorld(
        snap.name,
        { config: snap.config as any, layers: (snap as any).layers, preview },
        this.currentWorldId ?? undefined
      ).subscribe({
        next: (result) => {
          this.isSaving = false;
          this.currentWorldId = result.worldId; // Track the world ID for future saves
          this._flushMetadata(result.worldId);
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
              this._hydrateMetadata(world.id);
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
      if (!name) {return;}

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
            this.worlds.saveFromHexSnapshot(
              name,
              { config: snap.config as any, layers: (snap as any).layers, preview }
            ).subscribe({
              next: (result) => {
                this.isSaving = false;
                this.currentWorldId = result.worldId; // Track the new world ID
                this._flushMetadata(result.worldId);
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
            this.worlds.saveFromHexSnapshot(
              name,
              { config: snap.config as any, layers: (snap as any).layers, preview }
            ).subscribe({
              next: (result) => {
                this.isSaving = false;
                this.currentWorldId = result.worldId;
                this._flushMetadata(result.worldId);
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

  /** Persist the world's tile-metadata snapshot (annotations + connections). */
  private _flushMetadata(worldId: string): void {
    const snapshot = this.tileMetadata.createSnapshot(worldId);
    this.worlds.saveMetadata(worldId, snapshot).subscribe({
      error: (err) => console.error('Failed to save world metadata:', err)
    });
  }

  /** Restore a world's tile-metadata snapshot and re-render its connections. */
  private _hydrateMetadata(worldId: string): void {
    this.worlds.getMetadata(worldId).subscribe({
      next: (snapshot) => {
        if (snapshot && (snapshot.metadata || snapshot.connections)) {
          this.tileMetadata.restoreSnapshot(snapshot);
          this.tileGrid.updateConnections(this.tileMetadata.getConnections());
        }
      },
      error: () => { /* world has no saved metadata yet */ }
    });
  }

  /** Step the highlighted world forward (+1) or back (-1) through the universe. */
  public stepWorld(delta: number): void {
    this.tileGrid.stepSelection(delta);
    this.updateWorldStepLabel();
  }

  private updateWorldStepLabel(): void {
    const { position, total, scoped } = this.tileGrid.getNavInfo();
    if (!total) {
      this.worldStepLabel = '';
      return;
    }
    const mark = scoped ? ' ✦' : '';
    this.worldStepLabel = position >= 0
      ? `World ${position + 1} / ${total}${mark}`
      : `${total} world${total === 1 ? '' : 's'}${mark}`;
  }

  private queueWorldStepLabelUpdate(): void {
    if (this.worldStepLabelTimeout) {
      clearTimeout(this.worldStepLabelTimeout);
    }
    this.worldStepLabelTimeout = setTimeout(() => this.updateWorldStepLabel(), 0);
  }

  public onOpenWorlds(): void {
    const ref = this.dialog.open(WorldsDialogComponent, { data: { limit: 50 }, panelClass: 'glass-dialog' });
    ref.afterClosed().subscribe((world?: { id: string; name: string }) => {
      if (!world) {return;}
      this.loadWorldById(world.id, world.name);
    });
  }

  public onApplyGridConfig(): void {
    this.tileGrid.createTileGrid(this.gridConfig);
    this._syncGridStateToService();
    this.queueWorldStepLabelUpdate();
    this._triggerViewportScan();
  }

  public onApplySeed(): void {
    const trimmed = (this.seed ?? '').toString().trim();
    this.tileGrid.setRandomSeed(trimmed.length > 0 ? trimmed : null);
    this.tileGrid.triggerSeedBloom();
    this._triggerViewportScan();
  }

  onRandomize(): void {
    this.tileGrid.randomize();
    this._triggerViewportScan();
  }

  onClear(): void {
    this.tileGrid.clear();
    this._triggerViewportScan();
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
    this._triggerViewportScan();
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
    this._triggerViewportScan();
  }

  onToggleEditMode(): void {
    this.isEditMode = !this.isEditMode;
    this.tileGrid.setEditMode(this.isEditMode);
    // Hide context menu when entering edit mode
    if (this.isEditMode) {this.contextMenu.visible = false;}
    this._triggerViewportScan();
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
    if (this.isTypingInInput()) {return;}

    if (e.key === 'h' || e.key === 'H') {
      this.onToggleOutlines(!this.outlinesVisible);
    } else if (e.key === 'Escape') {
      this.tileGrid.returnToOverview();
      this.queueWorldStepLabelUpdate();
      this.contextMenu.visible = false;
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
    if (!active) {return false;}
    const tagName = active.tagName.toLowerCase();
    return tagName === 'input' || tagName === 'textarea' || tagName === 'select' ||
      (active.hasAttribute('contenteditable') && active.getAttribute('contenteditable') !== 'false');
  }

  onBrushChange(next: number): void {
    this.tileGrid.setBrushRadius(next);
    this._triggerViewportScan();
  }

  onToggleOutlines(value: boolean): void {
    this.outlinesVisible = value;
    this.tileGrid.setOutlinesVisible(value);
    this._triggerViewportScan();
  }

  onToggleConnections(value: boolean): void {
    this.connectionsVisible = value;
    this.tileGrid.setConnectionsVisible(value);
    this._triggerViewportScan();
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

  private _triggerViewportScan(): void {
    if (this.viewportScanTimeout) {
      clearTimeout(this.viewportScanTimeout);
    }
    this.viewportScanActive = false;
    requestAnimationFrame(() => {
      this.viewportScanActive = true;
      this.viewportScanTimeout = setTimeout(() => {
        this.viewportScanActive = false;
      }, 900);
    });
  }
}
