import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject, takeUntil, switchMap, map } from 'rxjs';
import { TileMetadataService } from '../engine/tile-metadata.service';
import { TileWorldMetadata, ConnectionType, CONNECTION_STYLES, WorldConnection } from '../engine/tile-metadata.model';
import { CreativeDataService, Board } from '../../../creative-design-product/services/creative-data.service';
import { CreativeService, WikiPageDto, CharacterDto } from '../../../services/creative/creative.service';
import { WorldsService, WorldAsset } from '../../../services/worlds/worlds.service';

export interface SelectedTileInfo {
  index: number;
  x: number;
  y: number;
  worldX: number;
  worldY: number;
  worldZ: number;
  terrain: string;
  biome: string;
  resource: string;
}

@Component({
  selector: 'app-world-detail-panel',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './world-detail-panel.component.html',
  styleUrl: './world-detail-panel.component.scss'
})
export class WorldDetailPanelComponent implements OnInit, OnDestroy, OnChanges {
  @Input() selectedTile: SelectedTileInfo | null = null;
  /** The backend world this panel belongs to; wiki pages are scoped to it. */
  @Input() worldId: string | null = null;
  @Output() requestAIPrompt = new EventEmitter<{ tileIndex: number; prompt: string }>();
  @Output() createConnection = new EventEmitter<{ fromIndex: number }>();
  @Output() closePanel = new EventEmitter<void>();

  metadata: TileWorldMetadata | null = null;
  linkedWikiPages: WikiPageDto[] = [];
  linkedBoards: Board[] = [];

  // AI-generated world art (persisted in the world_assets table, per tile).
  isGeneratingArt = false;
  artError = '';
  worldArt: WorldAsset[] = [];

  // AI-generated inhabitants (persisted in the characters table, world-scoped).
  inhabitants: CharacterDto[] = [];
  isGeneratingInhabitant = false;
  inhabitantError = '';

  // World-scoped knowledge (ingested wiki lore).
  knowledgeDocs: Array<{ id: string; title: string; source: string }> = [];
  isIngesting = false;
  knowledgeQuery = '';
  knowledgeResults: Array<{ text: string; document_id: string; distance: number }> = [];
  isSearchingKnowledge = false;
  tileConnections: WorldConnection[] = [];

  // Edit states
  isEditingName = false;
  isEditingDescription = false;
  editName = '';
  editDescription = '';
  newTag = '';
  newNote = '';
  aiPromptText = '';

  // Quick capture
  quickCaptureUrl = '';
  quickCaptureTitle = '';

  // Connection creation
  isCreatingConnection = false;
  newConnectionType: ConnectionType = 'alliance';

  connectionStyles = CONNECTION_STYLES;
  connectionTypes: ConnectionType[] = ['trade', 'conflict', 'alliance', 'portal', 'influence', 'mystery'];

  private destroy$ = new Subject<void>();

  constructor(
    private metadataService: TileMetadataService,
    private creativeData: CreativeDataService,
    private creativeService: CreativeService,
    private worldsService: WorldsService
  ) {}

  ngOnInit(): void {
    this.metadataService.onSelectedMetadataChanged()
      .pipe(takeUntil(this.destroy$))
      .subscribe(meta => {
        this.metadata = meta;
        this.loadLinkedContent();
        this.loadConnections();
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['selectedTile']) {
      this.artError = '';
      if (this.selectedTile) {
        this.metadataService.setSelectedTile(this.selectedTile.index);
      } else {
        this.metadataService.setSelectedTile(null);
      }
      this.loadWorldArt();
    }
    if (changes['worldId']) {
      this.knowledgeResults = [];
      this.knowledgeQuery = '';
      this.loadKnowledge();
      this.loadInhabitants();
      this.loadWorldArt();
    }
  }

  // ─────────────────────────────────────────────────────────────
  // World Knowledge (ingested wiki lore + world-scoped RAG search)
  // ─────────────────────────────────────────────────────────────

  private loadKnowledge(): void {
    if (!this.worldId) { this.knowledgeDocs = []; return; }
    this.worldsService.listKnowledge(this.worldId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (docs) => { this.knowledgeDocs = docs ?? []; },
        error: () => { this.knowledgeDocs = []; }
      });
  }

  /** Ingest the world's wiki pages into the knowledgebase (world-scoped, RAG). */
  ingestKnowledge(): void {
    if (!this.worldId || this.isIngesting) { return; }
    this.isIngesting = true;
    this.worldsService.ingestWiki(this.worldId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => { this.isIngesting = false; this.loadKnowledge(); },
        error: (err) => { this.isIngesting = false; console.error('Wiki ingest failed:', err); }
      });
  }

  /** Semantic search scoped to this world's knowledge. */
  searchKnowledge(): void {
    const q = this.knowledgeQuery.trim();
    if (!this.worldId || !q) { this.knowledgeResults = []; return; }
    this.isSearchingKnowledge = true;
    this.worldsService.searchKnowledge(this.worldId, q)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (res) => { this.knowledgeResults = res?.results ?? []; this.isSearchingKnowledge = false; },
        error: () => { this.knowledgeResults = []; this.isSearchingKnowledge = false; }
      });
  }

  /** Resolve a chunk's source document title for display. */
  knowledgeDocTitle(documentId: string): string {
    return this.knowledgeDocs.find(d => d.id === documentId)?.title || 'Knowledge';
  }

  // ─────────────────────────────────────────────────────────────
  // AI world art — generate a portrait of this world to populate it
  // ─────────────────────────────────────────────────────────────

  private loadWorldArt(): void {
    if (!this.worldId || !this.selectedTile) { this.worldArt = []; return; }
    this.worldsService.listAssets(this.worldId, this.selectedTile.index)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (assets) => { this.worldArt = (assets ?? []).filter(a => a.kind === 'art'); },
        error: () => { this.worldArt = []; }
      });
  }

  generateWorldArt(): void {
    if (!this.selectedTile || !this.worldId || this.isGeneratingArt) { return; }
    const tileIndex = this.selectedTile.index;
    const name = this.metadata?.name || `World ${tileIndex}`;
    this.isGeneratingArt = true;
    this.artError = '';
    // Generate the image, then persist it immediately to the world_assets table.
    this.creativeService.generateImage(this.buildArtPrompt())
      .pipe(
        switchMap(({ b64 }) => this.worldsService.saveAsset(this.worldId!, {
          image_b64: b64, kind: 'art', title: `${name} — portrait`, tile_index: tileIndex
        })),
        takeUntil(this.destroy$)
      )
      .subscribe({
        next: () => { this.isGeneratingArt = false; this.loadWorldArt(); },
        error: (err) => {
          this.isGeneratingArt = false;
          this.artError = 'Art generation failed — needs an OpenAI key configured.';
          console.error('World art generation failed:', err);
        }
      });
  }

  artImageUrl(a: WorldAsset): string {
    return `data:image/png;base64,${a.image_b64}`;
  }

  removeArt(a: WorldAsset): void {
    if (!this.worldId) { return; }
    this.worldsService.deleteAsset(this.worldId, a.id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({ next: () => this.loadWorldArt(), error: () => this.loadWorldArt() });
  }

  /** Build an evocative image prompt from the world's known character + lore. */
  private buildArtPrompt(): string {
    const t = this.selectedTile!;
    const m = this.metadata;
    const name = m?.name || `World ${t.index}`;
    const biome = t.biome !== 'none' ? `${t.biome} ` : '';
    const resource = t.resource === 'node' ? ', rich in rare resources' : '';
    const desc = m?.description ? ` ${m.description}` : '';
    return `Cinematic concept art of a world named "${name}": a ${biome}${t.terrain} world${resource}.${desc} `
      + `Painterly science-fantasy, luminous solarpunk aesthetic, vast scale, dramatic lighting.`;
  }

  // ─────────────────────────────────────────────────────────────
  // AI inhabitants — generate characters + portraits, persisted in the DB
  // ─────────────────────────────────────────────────────────────

  private loadInhabitants(): void {
    if (!this.worldId) { this.inhabitants = []; return; }
    this.creativeService.listCharacters(this.worldId)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (chars) => { this.inhabitants = chars ?? []; },
        error: () => { this.inhabitants = []; }
      });
  }

  /** Invent an inhabitant (name + portrait), persist it to the world, reload. */
  generateInhabitant(): void {
    if (!this.worldId || this.isGeneratingInhabitant) { return; }
    this.isGeneratingInhabitant = true;
    this.inhabitantError = '';
    const name = this.inventInhabitantName();
    this.creativeService.createCharacter({ world_id: this.worldId, name, traits: { origin: 'ai' } })
      .pipe(
        switchMap(({ id }) =>
          this.creativeService.generateCharacterImage(id, this.buildInhabitantPrompt(name)).pipe(map(() => id))
        ),
        takeUntil(this.destroy$)
      )
      .subscribe({
        next: () => { this.isGeneratingInhabitant = false; this.loadInhabitants(); },
        error: (err) => {
          this.isGeneratingInhabitant = false;
          this.inhabitantError = 'Inhabitant generation failed — needs an OpenAI key.';
          console.error('Inhabitant generation failed:', err);
          this.loadInhabitants(); // the character row may still have been created
        }
      });
  }

  /** Data URL for an inhabitant portrait (base64 PNG from the DB). */
  inhabitantImageUrl(c: CharacterDto): string | null {
    return c.image_b64 ? `data:image/png;base64,${c.image_b64}` : null;
  }

  private inventInhabitantName(): string {
    const roots = ['Kael', 'Vyra', 'Orin', 'Sela', 'Thane', 'Nyx', 'Aeris', 'Dax', 'Lumi', 'Cassia', 'Veld', 'Iro', 'Sora', 'Bram'];
    const epithets = ['the Wandering', 'Tideborn', 'of the Ember Reach', 'Skywright', 'the Verdant', 'Starbound', 'of Hollow Vale', 'the Unbroken', 'Dawnseer', 'of the Drift'];
    const r = roots[Math.floor(Math.random() * roots.length)];
    const e = epithets[Math.floor(Math.random() * epithets.length)];
    return `${r} ${e}`;
  }

  private buildInhabitantPrompt(name: string): string {
    const t = this.selectedTile;
    const biome = t && t.biome !== 'none' ? `${t.biome} ` : '';
    const terrain = t ? `${t.terrain} ` : '';
    return `Character portrait of "${name}", an inhabitant of a ${biome}${terrain}world. `
      + `Head-and-shoulders, painterly science-fantasy, solarpunk aesthetic, dramatic rim lighting.`;
  }

  private loadLinkedContent(): void {
    if (!this.metadata) {
      this.linkedWikiPages = [];
      this.linkedBoards = [];
      return;
    }

    // Wiki pages live in the backend (world-scoped); load and filter to the ones
    // this tile links to.
    const wikiIds = this.metadata.wikiPageIds ?? [];
    if (wikiIds.length) {
      this.creativeService.listWiki(this.worldId ?? undefined)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (pages) => { this.linkedWikiPages = pages.filter(p => wikiIds.includes(p.id)); },
          error: () => { this.linkedWikiPages = []; }
        });
    } else {
      this.linkedWikiPages = [];
    }

    // Boards remain local for now (no backend board API yet).
    const allBoards = this.creativeData.listBoards();
    this.linkedBoards = allBoards.filter(b => this.metadata?.boardIds?.includes(b.id));
  }

  private loadConnections(): void {
    if (!this.metadata) {
      this.tileConnections = [];
      return;
    }
    this.tileConnections = this.metadataService.getConnectionsForTile(this.metadata.tileIndex);
  }

  // ─────────────────────────────────────────────────────────────
  // World Identity
  // ─────────────────────────────────────────────────────────────

  startEditName(): void {
    this.isEditingName = true;
    this.editName = this.metadata?.name || '';
  }

  saveName(): void {
    if (this.selectedTile && this.editName.trim()) {
      this.metadataService.setWorldName(this.selectedTile.index, this.editName.trim());
    }
    this.isEditingName = false;
  }

  cancelEditName(): void {
    this.isEditingName = false;
    this.editName = '';
  }

  startEditDescription(): void {
    this.isEditingDescription = true;
    this.editDescription = this.metadata?.description || '';
  }

  saveDescription(): void {
    if (this.selectedTile) {
      this.metadataService.setWorldDescription(this.selectedTile.index, this.editDescription.trim());
    }
    this.isEditingDescription = false;
  }

  cancelEditDescription(): void {
    this.isEditingDescription = false;
    this.editDescription = '';
  }

  // ─────────────────────────────────────────────────────────────
  // Tags
  // ─────────────────────────────────────────────────────────────

  addTag(): void {
    if (this.selectedTile && this.newTag.trim()) {
      this.metadataService.addTag(this.selectedTile.index, this.newTag.trim());
      this.newTag = '';
    }
  }

  removeTag(tag: string): void {
    if (this.selectedTile) {
      this.metadataService.removeTag(this.selectedTile.index, tag);
    }
  }

  // ─────────────────────────────────────────────────────────────
  // Quick Notes
  // ─────────────────────────────────────────────────────────────

  addNote(): void {
    if (this.selectedTile && this.newNote.trim()) {
      this.metadataService.addQuickNote(this.selectedTile.index, this.newNote.trim());
      this.newNote = '';
    }
  }

  removeNote(noteId: string): void {
    if (this.selectedTile) {
      this.metadataService.removeQuickNote(this.selectedTile.index, noteId);
    }
  }

  // ─────────────────────────────────────────────────────────────
  // Quick Capture (Images/Links)
  // ─────────────────────────────────────────────────────────────

  onPasteCapture(event: ClipboardEvent): void {
    if (!this.selectedTile) {return;}

    const items = event.clipboardData?.items;
    if (!items) {return;}

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) {
          this.captureImage(file);
        }
        event.preventDefault();
        return;
      }
    }

    // Check for URL text
    const text = event.clipboardData?.getData('text');
    if (text && this.isValidUrl(text)) {
      this.captureLink(text);
      event.preventDefault();
    }
  }

  private captureImage(file: File): void {
    if (!this.selectedTile) {return;}

    const reader = new FileReader();
    reader.onload = (e) => {
      const imageData = e.target?.result as string;
      this.metadataService.addPinnedImage(this.selectedTile!.index, imageData, file.name);
    };
    reader.readAsDataURL(file);
  }

  captureLink(url?: string): void {
    if (!this.selectedTile) {return;}
    const linkUrl = url || this.quickCaptureUrl.trim();
    if (!linkUrl) {return;}

    this.metadataService.addPinnedLink(
      this.selectedTile.index,
      linkUrl,
      this.quickCaptureTitle.trim() || undefined
    );
    this.quickCaptureUrl = '';
    this.quickCaptureTitle = '';
  }

  removePinnedItem(itemId: string): void {
    if (this.selectedTile) {
      this.metadataService.removePinnedItem(this.selectedTile.index, itemId);
    }
  }

  private isValidUrl(str: string): boolean {
    try {
      new URL(str);
      return true;
    } catch {
      return false;
    }
  }

  // ─────────────────────────────────────────────────────────────
  // AI Integration
  // ─────────────────────────────────────────────────────────────

  sendAIPrompt(): void {
    if (this.selectedTile && this.aiPromptText.trim()) {
      this.requestAIPrompt.emit({
        tileIndex: this.selectedTile.index,
        prompt: this.aiPromptText.trim()
      });
      this.aiPromptText = '';
    }
  }

  quickAIPrompt(action: string): void {
    if (!this.selectedTile) {return;}

    let prompt = '';
    switch (action) {
      case 'lore':
        prompt = `Generate lore for this world. Terrain: ${this.selectedTile.terrain}, Biome: ${this.selectedTile.biome}, Resources: ${this.selectedTile.resource}. ${this.metadata?.name ? `World name: ${this.metadata.name}.` : ''} ${this.metadata?.description ? `Description: ${this.metadata.description}` : ''}`;
        break;
      case 'connections':
        prompt = `Suggest interesting connections this world might have with neighboring worlds based on its characteristics. Terrain: ${this.selectedTile.terrain}, Biome: ${this.selectedTile.biome}.`;
        break;
      case 'inhabitants':
        prompt = `Describe the inhabitants or entities that might exist in this world. Terrain: ${this.selectedTile.terrain}, Biome: ${this.selectedTile.biome}, Resources: ${this.selectedTile.resource}.`;
        break;
      case 'history':
        prompt = `Create a brief history or timeline for this world. What major events shaped it? Terrain: ${this.selectedTile.terrain}, Biome: ${this.selectedTile.biome}.`;
        break;
    }

    this.requestAIPrompt.emit({ tileIndex: this.selectedTile.index, prompt });
  }

  removeAIObservation(obsId: string): void {
    if (this.selectedTile) {
      this.metadataService.removeAIObservation(this.selectedTile.index, obsId);
    }
  }

  // ─────────────────────────────────────────────────────────────
  // Connections
  // ─────────────────────────────────────────────────────────────

  startCreateConnection(): void {
    this.isCreatingConnection = true;
    this.createConnection.emit({ fromIndex: this.selectedTile!.index });
  }

  cancelCreateConnection(): void {
    this.isCreatingConnection = false;
  }

  removeConnection(connectionId: string): void {
    this.metadataService.removeConnection(connectionId);
    this.loadConnections();
  }

  getConnectionPartnerIndex(connection: WorldConnection): number {
    if (!this.selectedTile) {return -1;}
    return connection.fromTileIndex === this.selectedTile.index
      ? connection.toTileIndex
      : connection.fromTileIndex;
  }

  getPartnerWorldName(connection: WorldConnection): string {
    const partnerIndex = this.getConnectionPartnerIndex(connection);
    const partnerMeta = this.metadataService.getMetadata(partnerIndex);
    return partnerMeta?.name || `World (${partnerIndex})`;
  }

  // ─────────────────────────────────────────────────────────────
  // Wiki & Boards
  // ─────────────────────────────────────────────────────────────

  createWikiPage(): void {
    if (!this.selectedTile) {return;}
    const tileIndex = this.selectedTile.index;
    const title = `${this.metadata?.name || `World ${tileIndex}`} — Lore`;
    // Persist to the backend wiki, scoped to this world, seeded with what we know
    // about the tile so the page isn't empty.
    this.creativeService.createWiki({
      world_id: this.worldId ?? undefined,
      title,
      content: this.seedWikiContent(),
      metadata: { source: 'command-center', tileIndex }
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: ({ id }) => {
          this.metadataService.linkWikiPage(tileIndex, id);
          this.loadLinkedContent();
        },
        error: (err) => console.error('Failed to create wiki page:', err)
      });
  }

  /** Build an initial Markdown stub for a tile's wiki page from its known state. */
  private seedWikiContent(): string {
    const t = this.selectedTile;
    const m = this.metadata;
    const lines: string[] = [`# ${m?.name || `World ${t?.index ?? ''}`.trim()}`, ''];
    if (m?.description) { lines.push(m.description, ''); }
    if (t) {
      lines.push(
        `- **Terrain:** ${t.terrain}`,
        `- **Biome:** ${t.biome}`,
        `- **Resource:** ${t.resource}`
      );
    }
    if (m?.tags?.length) { lines.push('', `**Tags:** ${m.tags.join(', ')}`); }
    return lines.join('\n');
  }

  unlinkWikiPage(pageId: string): void {
    if (this.selectedTile) {
      this.metadataService.unlinkWikiPage(this.selectedTile.index, pageId);
      this.loadLinkedContent();
    }
  }

  createBoard(): void {
    if (!this.selectedTile) {return;}
    const title = this.metadata?.name || `World ${this.selectedTile.index}`;
    const board = this.creativeData.createBoard({ title: `${title} - Mood Board` });
    this.metadataService.linkBoard(this.selectedTile.index, board.id);
    this.loadLinkedContent();
  }

  unlinkBoard(boardId: string): void {
    if (this.selectedTile) {
      this.metadataService.unlinkBoard(this.selectedTile.index, boardId);
      this.loadLinkedContent();
    }
  }

  // ─────────────────────────────────────────────────────────────
  // Utilities
  // ─────────────────────────────────────────────────────────────

  get hasContent(): boolean {
    if (!this.metadata) {return false;}
    return !!(
      this.metadata.name ||
      this.metadata.description ||
      (this.metadata.tags && this.metadata.tags.length > 0) ||
      (this.metadata.quickNotes && this.metadata.quickNotes.length > 0) ||
      (this.metadata.pinnedItems && this.metadata.pinnedItems.length > 0) ||
      (this.metadata.aiObservations && this.metadata.aiObservations.length > 0)
    );
  }

  formatDate(dateStr: string): string {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }
}
