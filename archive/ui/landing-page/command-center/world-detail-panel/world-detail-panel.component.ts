import { Component, Input, Output, EventEmitter, OnInit, OnDestroy, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { Subject, takeUntil, switchMap, map } from 'rxjs';
import { TileMetadataService } from '../engine/tile-metadata.service';
import { TileWorldMetadata, ConnectionType, CONNECTION_STYLES, WorldConnection } from '../engine/tile-metadata.model';
import { Board, BoardCard, CreativeDataService } from '../../../creative-design-product/services/creative-data.service';
import { CreativeService, WikiPageDto, CharacterDto } from '../../../services/creative/creative.service';
import {
  WorldAgentAuditResult,
  WorldConnectionSuggestion,
  WorldsService,
  WorldAsset
} from '../../../services/worlds/worlds.service';
import { MatDialog } from '@angular/material/dialog';
import { ImageLightboxDialogComponent } from '../image-lightbox-dialog/image-lightbox-dialog.component';
import { SpawnTemplateDto, SpawnTemplatesService } from '../../../services/spawn-templates.service';

/** Shared art-direction / quality suffix appended to generated image prompts. */
const ART_QUALITY =
  'Ultra-detailed digital matte painting, volumetric lighting, atmospheric haze, '
  + 'sweeping vista with an epic sense of scale, rich layered colour, sharp focus, '
  + 'cinematic wide composition, concept-art quality.';

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
  artPrompt = ''; // optional custom prompt; blank = auto-generate from the world's lore
  isDirectingArtPrompt = false;
  artPromptStatus = '';

  // Floating hover preview (a small studio "loupe" over any generated image).
  hoverPreview: { url: string; label: string; x: number; y: number } | null = null;

  // AI-generated inhabitants (persisted in the characters table, world-scoped).
  inhabitants: CharacterDto[] = [];
  isGeneratingInhabitant = false;
  inhabitantError = '';

  // AI lore generation (schema-tagged wiki pages, persisted + linked to the tile).
  isGeneratingLore = false;
  loreError = '';
  loreStatus = '';
  loreAuditSummary = '';
  isAuditingLore = false;
  worldAgentTemplates: SpawnTemplateDto[] = [];
  selectedLoreAgentRole = 'world_lore_architect';
  loreDraft: {
    tileIndex: number;
    title: string;
    content: string;
    generatedBy: string;
    audit: WorldAgentAuditResult;
  } | null = null;
  isSavingLoreDraft = false;

  // World-scoped knowledge (ingested wiki lore).
  knowledgeDocs: Array<{ id: string; title: string; source: string }> = [];
  isIngesting = false;
  knowledgeQuery = '';
  knowledgeResults: Array<{ text: string; document_id: string; distance: number }> = [];
  isSearchingKnowledge = false;
  tileConnections: WorldConnection[] = [];
  connectionSuggestions: WorldConnectionSuggestion[] = [];
  isSuggestingConnections = false;
  connectionSuggestionError = '';
  selectedBoard: Board | null = null;
  moodBoardStatus = '';

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
    private worldsService: WorldsService,
    private spawnTemplates: SpawnTemplatesService,
    private dialog: MatDialog,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadWorldAgentTemplates();
    this.metadataService.onSelectedMetadataChanged()
      .pipe(takeUntil(this.destroy$))
      .subscribe(meta => {
        this.metadata = meta;
        this.loadLinkedContent();
        this.loadConnections();
      });
  }

  private loadWorldAgentTemplates(): void {
    this.spawnTemplates.listTemplates({ tag: 'procedural-worlds', builtin_only: true })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: ({ templates }) => {
          this.worldAgentTemplates = templates ?? [];
          const defaultTemplate = this.worldAgentTemplates.find(t => t.role === 'world_lore_architect');
          this.selectedLoreAgentRole = defaultTemplate?.role ?? this.selectedLoreAgentRole;
        },
        error: () => { this.worldAgentTemplates = []; }
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['selectedTile']) {
      this.artError = '';
      this.artPrompt = '';
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
        next: (assets) => {
          // Newest first so the hero plate features the most recent portrait.
          this.worldArt = (assets ?? [])
            .filter(a => a.kind === 'art')
            .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
        },
        error: () => { this.worldArt = []; }
      });
  }

  /** The most recent portrait — promoted to the large "World Plate" hero. */
  get featuredArt(): WorldAsset | null {
    return this.worldArt[0] ?? null;
  }

  generateWorldArt(): void {
    if (!this.selectedTile || !this.worldId || this.isGeneratingArt) { return; }
    const tileIndex = this.selectedTile.index;
    const name = this.metadata?.name || `World ${tileIndex}`;
    this.isGeneratingArt = true;
    this.artError = '';
    // Use the typed prompt if provided, otherwise auto-build one from the world.
    const prompt = this.artPrompt.trim() || this.buildArtPrompt();
    // Generate the image, then persist it immediately to the world_assets table.
    this.creativeService.generateImage(prompt)
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
          const detail = this.getErrorDetail(err);
          this.artError = detail
            ? `Art generation failed — ${detail}`
            : 'Art generation failed — check OpenAI image configuration.';
          console.error('World art generation failed:', err);
        }
      });
  }

  directArtPrompt(): void {
    if (!this.worldId || !this.selectedTile || this.isDirectingArtPrompt) {return;}
    this.isDirectingArtPrompt = true;
    this.artError = '';
    this.artPromptStatus = '';
    this.worldsService.generateImagePrompt(this.worldId, {
      tile_index: this.selectedTile.index,
      world_name: this.metadata?.name || `World ${this.selectedTile.index}`,
      user_context: this.loreContext()
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: ({ prompt, generated_by }) => {
          this.isDirectingArtPrompt = false;
          this.artPrompt = prompt;
          this.artPromptStatus = `Prompt drafted by ${generated_by}`;
        },
        error: (err) => {
          this.isDirectingArtPrompt = false;
          const detail = this.getErrorDetail(err);
          this.artError = detail ? `Prompt direction failed — ${detail}` : 'Prompt direction failed.';
        }
      });
  }

  private getErrorDetail(err: unknown): string {
    const maybeError = err as { error?: { detail?: unknown; message?: unknown }; message?: unknown };
    const detail = maybeError.error?.detail ?? maybeError.error?.message ?? maybeError.message;
    return typeof detail === 'string' ? detail : '';
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

  // ─────────────────────────────────────────────────────────────
  // Image presentation — hover "loupe" preview + focused lightbox
  // ─────────────────────────────────────────────────────────────

  /** Show a floating preview of a world-art image at the cursor. */
  showArtPreview(a: WorldAsset, ev: MouseEvent): void {
    this.positionPreview(this.artImageUrl(a), a.title || 'World art', ev);
  }

  /** Show a floating preview of an inhabitant portrait (no-op without one). */
  showInhabitantPreview(c: CharacterDto, ev: MouseEvent): void {
    const url = this.inhabitantImageUrl(c);
    if (!url) { this.hidePreview(); return; }
    this.positionPreview(url, c.name, ev);
  }

  hidePreview(): void {
    this.hoverPreview = null;
  }

  /** Position a fixed popover near the cursor, clamped to the viewport. */
  private positionPreview(url: string, label: string, ev: MouseEvent): void {
    const width = 256;   // .img-preview is 16rem wide
    const height = 220;  // approximate popover height
    const pad = 12;
    let x = ev.clientX + 18;
    let y = ev.clientY + 18;
    if (x + width + pad > window.innerWidth) { x = ev.clientX - width - 18; }
    if (y + height + pad > window.innerHeight) { y = window.innerHeight - height - pad; }
    this.hoverPreview = { url, label, x: Math.max(pad, x), y: Math.max(pad, y) };
  }

  /** Open the focused, full-scale lightbox view of an image. */
  openLightbox(url: string | null, title: string): void {
    if (!url) { return; }
    this.hidePreview();
    this.dialog.open(ImageLightboxDialogComponent, {
      data: { url, title },
      panelClass: 'glass-dialog',
      maxWidth: '92vw',
      maxHeight: '92vh'
    });
  }

  /** The subject clause describing this world (name + character + lore). */
  private worldSubject(): string {
    const t = this.selectedTile;
    const m = this.metadata;
    const name = m?.name || (t ? `World ${t.index}` : 'an uncharted world');
    const biome = t && t.biome !== 'none' ? `${t.biome} ` : '';
    const terrain = t ? `${t.terrain} ` : '';
    const resource = t && t.resource === 'node' ? ', rich in rare resources' : '';
    const desc = m?.description ? ` ${m.description}` : '';
    return `A ${biome}${terrain}world named "${name}"${resource}.${desc}`;
  }

  /** Default auto-prompt (used when no theme/custom text is supplied). */
  private buildArtPrompt(): string {
    return `${this.worldSubject()} Painterly science-fantasy with a luminous solarpunk aesthetic. ${ART_QUALITY}`;
  }

  /** Selectable theme presets — each composes the world subject with a rich style. */
  readonly artThemes: ReadonlyArray<{ id: string; label: string; style: string }> = [
    { id: 'solarpunk', label: '🌿 Solarpunk Utopia', style: 'A radiant solarpunk utopia — terraced gardens cascading over living glass-and-vine architecture, solar sails and waterfalls catching golden-hour light, airborne gardens drifting between spires; verdant, hopeful, harmonious.' },
    { id: 'volcanic', label: '🌋 Volcanic Forge', style: 'A volcanic forge-world — rivers of molten gold winding between obsidian crags and basalt spires, ember sparks drifting through ashen skies, distant eruptions glowing on the horizon; fierce, primal, dramatic.' },
    { id: 'tundra', label: '❄️ Crystalline Tundra', style: 'A frozen crystalline tundra — fields of glittering ice shards and towering crystal spires beneath a vast shimmering aurora, breath-fogged stillness; glacial blues and violets, serene and immense.' },
    { id: 'oceanic', label: '🌊 Oceanic Expanse', style: 'A boundless oceanic world — bioluminescent swells rolling beneath floating coral-cities and arching skybridges, reflected starlight and drifting mist; teal and cyan, tranquil and immense.' },
    { id: 'desert', label: '🏜️ Desert Frontier', style: 'A sci-fantasy desert frontier — rippling dunes and weathered sandstone monoliths under twin suns, caravans and half-buried ruins casting long shadows; warm amber, rust and gold, sun-scorched grandeur.' },
    { id: 'jungle', label: '🌴 Verdant Wilds', style: 'Overgrown verdant wilds — emerald canopy swallowing colossal ancient ruins, vines and cascading waterfalls, shafts of misty god-rays piercing the green gloom; lush, mysterious, teeming with life.' },
    { id: 'cyber', label: '🌃 Cyber Metropolis', style: 'A towering cyber-metropolis — endless holographic skyscrapers and neon signage above rain-slick streets, flying traffic threading the canyons; electric magenta and cyan, dense, luminous, rain-soaked.' },
    { id: 'astral', label: '✨ Astral Void', style: 'An astral void world adrift among swirling nebulae and dense starfields, ribbons of aurora and shattered moons hanging in the dark; deep cosmic blues, gold and violet, ethereal and dreamlike.' }
  ];

  /** Apply a theme preset to the prompt box (still editable before generating). */
  applyArtTheme(theme: { style: string }): void {
    this.artPrompt = `${this.worldSubject()} ${theme.style} ${ART_QUALITY}`;
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
    return `Character portrait of "${name}", an inhabitant of a ${biome}${terrain}world — `
      + `distinctive attire, gear and bearing shaped by their homeworld, an expressive characterful face. `
      + `Head-and-shoulders, painterly science-fantasy with a solarpunk aesthetic, dramatic rim lighting, `
      + `intricate detail, concept-art quality.`;
  }

  // ─────────────────────────────────────────────────────────────
  // AI lore — generate schema-tagged wiki pages grounded in the world
  // ─────────────────────────────────────────────────────────────

  /** Maps the AI Assistant actions to a wiki page kind + generation focus. */
  private readonly loreKinds: Record<string, { kind: string; focus: string }> = {
    lore: { kind: 'Overview', focus: 'an evocative encyclopedic overview — what this world is, its defining character and atmosphere, and what makes it singular.' },
    history: { kind: 'History', focus: 'a history and timeline — its major eras, defining events, rises and falls, and how it became what it is.' },
    inhabitants: { kind: 'Peoples & Culture', focus: 'its peoples and cultures — who lives here, their customs, beliefs, factions and ways of life.' }
  };

  /** Generate a wiki page of the given kind, persist it, and link it to this tile. */
  generateLore(action: string): void {
    const spec = this.loreKinds[action];
    if (!spec) { this.quickAIPrompt(action); return; } // non-lore actions keep old behaviour
    if (!this.worldId) { this.loreError = 'Save this world first to generate lore.'; return; }
    if (!this.selectedTile || this.isGeneratingLore) { return; }
    const tileIndex = this.selectedTile.index;
    this.isGeneratingLore = true;
    this.loreError = '';
    this.loreAuditSummary = '';
    this.loreStatus = `Writing ${spec.kind}…`;
    this.worldsService.generateAgentLore(this.worldId, {
      tile_index: tileIndex,
      kind: spec.kind,
      focus: spec.focus,
      world_name: this.metadata?.name || `World ${tileIndex}`,
      user_context: this.loreContext(),
      agent_id: this.selectedLoreAgentRole
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: ({ title, content, generated_by, audit }) => {
          this.isGeneratingLore = false;
          this.loreDraft = { tileIndex, title, content, generatedBy: generated_by, audit };
          this.loreStatus = `Drafted “${title}” via ${generated_by}`;
          const missing = audit.missing_details.length
            ? ` Missing: ${audit.missing_details.join(' ')}`
            : '';
          this.loreAuditSummary = `Audit ${Math.round(audit.confidence * 100)}% confidence — ${audit.approved ? 'approved' : 'review suggested'}.${missing}`;
        },
        error: (err) => {
          this.isGeneratingLore = false;
          this.loreStatus = '';
          const detail = this.getErrorDetail(err);
          this.loreError = detail
            ? `Lore generation failed — ${detail}`
            : 'Lore generation failed — check the model/API key.';
          console.error('Lore generation failed:', err);
        }
      });
  }

  approveLoreDraft(): void {
    if (!this.worldId || !this.loreDraft || this.isSavingLoreDraft) {return;}
    this.isSavingLoreDraft = true;
    this.loreError = '';
    this.worldsService.saveAgentLore(this.worldId, {
      tile_index: this.loreDraft.tileIndex,
      title: this.loreDraft.title,
      content: this.loreDraft.content,
      generated_by: this.loreDraft.generatedBy,
      audit: this.loreDraft.audit
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: ({ id, title }) => {
          this.isSavingLoreDraft = false;
          this.loreStatus = `Saved “${title}” to wiki`;
          this.metadataService.linkWikiPage(this.loreDraft!.tileIndex, id);
          this.loreDraft = null;
          this.loadLinkedContent();
        },
        error: (err) => {
          this.isSavingLoreDraft = false;
          const detail = this.getErrorDetail(err);
          this.loreError = detail ? `Lore save failed — ${detail}` : 'Lore save failed.';
        }
      });
  }

  discardLoreDraft(): void {
    this.loreDraft = null;
    this.loreAuditSummary = '';
    this.loreStatus = '';
  }

  auditCurrentLore(): void {
    if (!this.worldId || !this.selectedTile || this.isAuditingLore) {return;}
    const tileIndex = this.selectedTile.index;
    this.isAuditingLore = true;
    this.loreError = '';
    this.worldsService.auditWorldAgent(this.worldId, {
      tile_index: tileIndex,
      content: this.loreDraft?.content ?? '',
      world_name: this.metadata?.name || `World ${tileIndex}`,
      user_context: this.loreContext()
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: ({ audit, generated_by }) => {
          this.isAuditingLore = false;
          const missing = audit.missing_details.length
            ? ` Missing: ${audit.missing_details.join(' ')}`
            : '';
          this.loreAuditSummary = `${generated_by}: ${Math.round(audit.confidence * 100)}% confidence — ${audit.approved ? 'approved' : 'review suggested'}.${missing}`;
        },
        error: (err) => {
          this.isAuditingLore = false;
          const detail = this.getErrorDetail(err);
          this.loreError = detail ? `Audit failed — ${detail}` : 'Audit failed.';
        }
      });
  }

  /** Summarise what's known about the selected world to ground generation. */
  private loreContext(): string {
    const t = this.selectedTile;
    const m = this.metadata;
    const parts: string[] = [];
    if (t) { parts.push(`terrain: ${t.terrain}`, `biome: ${t.biome}`, `resource: ${t.resource}`); }
    if (m?.tags?.length) { parts.push(`tags: ${m.tags.join(', ')}`); }
    if (m?.description) { parts.push(`description: ${m.description}`); }
    return parts.join('; ');
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
    const firstLinkedBoardId = this.metadata?.boardIds?.[0];
    if (!this.selectedBoard && firstLinkedBoardId) {
      this.selectedBoard = this.linkedBoards.find(board => board.id === firstLinkedBoardId) ?? null;
    }
    if (this.selectedBoard && !this.linkedBoards.some(board => board.id === this.selectedBoard?.id)) {
      this.selectedBoard = null;
    }
  }

  private loadConnections(): void {
    if (!this.metadata) {
      this.tileConnections = [];
      return;
    }
    this.tileConnections = this.metadataService.getConnectionsForTile(this.metadata.tileIndex);
  }

  suggestConnections(): void {
    if (!this.worldId || !this.selectedTile || this.isSuggestingConnections) {return;}
    this.isSuggestingConnections = true;
    this.connectionSuggestionError = '';
    this.worldsService.suggestWorldConnections(this.worldId, {
      tile_index: this.selectedTile.index,
      max_suggestions: 4,
      world_name: this.metadata?.name || `World ${this.selectedTile.index}`,
      user_context: this.loreContext()
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: ({ suggestions }) => {
          this.isSuggestingConnections = false;
          this.connectionSuggestions = suggestions ?? [];
        },
        error: (err) => {
          this.isSuggestingConnections = false;
          const detail = this.getErrorDetail(err);
          this.connectionSuggestionError = detail
            ? `Connection suggestions failed — ${detail}`
            : 'Connection suggestions failed.';
        }
      });
  }

  acceptConnectionSuggestion(suggestion: WorldConnectionSuggestion): void {
    this.metadataService.addConnection(
      suggestion.from_tile_index,
      suggestion.to_tile_index,
      suggestion.type,
      true,
      suggestion.label
    );
    this.connectionSuggestions = this.connectionSuggestions.filter(item => item !== suggestion);
    this.loadConnections();
  }

  rejectConnectionSuggestion(suggestion: WorldConnectionSuggestion): void {
    this.connectionSuggestions = this.connectionSuggestions.filter(item => item !== suggestion);
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
    const board = this.creativeData.createBoard({
      title: `${title} - Mood Board`,
      worldId: this.worldId ?? undefined
    });
    this.metadataService.linkBoard(this.selectedTile.index, board.id);
    this.loadLinkedContent();
    this.selectedBoard = board;
    this.moodBoardStatus = `Created ${board.title}`;
  }

  autoCreateMoodBoard(): void {
    if (!this.selectedTile) {return;}
    const title = this.metadata?.name || `World ${this.selectedTile.index}`;
    try {
      const board = this.creativeData.createBoardWithCards(
        { title: `${title} - Mood Board`, worldId: this.worldId ?? undefined },
        this.buildMoodBoardCards(title)
      );
      this.metadataService.linkBoard(this.selectedTile.index, board.id);
      this.loadLinkedContent();
      if (!this.linkedBoards.some(item => item.id === board.id)) {
        this.linkedBoards = [...this.linkedBoards, board];
      }
      this.selectedBoard = board;
      this.moodBoardStatus = `Auto-created ${board.title} with ${board.cards.length} cards`;
    } catch (err) {
      const detail = this.getErrorDetail(err);
      this.moodBoardStatus = detail
        ? `Mood board creation failed — ${detail}`
        : 'Mood board creation failed — browser storage is full.';
    }
  }

  selectBoard(board: Board): void {
    this.selectedBoard = this.creativeData.getBoard(board.id) ?? board;
    this.persistSelectedBoard(board.id);
  }

  openBoard(board: Board): void {
    this.selectBoard(board);
    this.router.navigate(['/creative/boards'], {
      queryParams: {
        worldId: this.worldId ?? undefined,
        projectId: this.worldId ?? undefined,
        boardId: board.id
      }
    });
  }

  unlinkBoard(boardId: string): void {
    if (this.selectedTile) {
      this.metadataService.unlinkBoard(this.selectedTile.index, boardId);
      if (this.selectedBoard?.id === boardId) {
        this.selectedBoard = null;
      }
      this.loadLinkedContent();
    }
  }

  private buildMoodBoardCards(worldName: string): Array<Omit<BoardCard, 'id' | 'createdAt'>> {
    const cards: Array<Omit<BoardCard, 'id' | 'createdAt'>> = [
      {
        title: 'World Overview',
        notes: this.moodBoardOverview(worldName)
      },
      {
        title: 'Palette',
        notes: this.moodBoardPalette()
      }
    ];

    for (const art of this.worldArt) {
      cards.push({
        title: `World Art: ${art.title || worldName}`,
        imageSource: 'world_asset',
        sourceId: art.id,
        notes: `Source: world_asset\nKind: ${art.kind}\nTile: ${art.tile_index ?? this.selectedTile?.index ?? 'world'}`
      });
    }

    for (const inhabitant of this.inhabitants) {
      cards.push({
        title: `Inhabitant: ${inhabitant.name}`,
        imageSource: 'character',
        sourceId: inhabitant.id,
        notes: `Source: character\nName: ${inhabitant.name}`
      });
    }

    for (const page of this.linkedWikiPages.slice(0, 4)) {
      cards.push({
        title: `Lore: ${page.title}`,
        notes: `Source: wiki\n\n${page.content.slice(0, 600)}`
      });
    }

    if (this.artPrompt.trim()) {
      cards.push({
        title: 'Prompt Direction',
        notes: this.artPrompt.trim()
      });
    }

    return cards;
  }

  private persistSelectedBoard(boardId: string): void {
    if (!this.selectedTile) {return;}
    const current = this.metadata?.boardIds ?? [];
    this.metadataService.unlinkBoard(this.selectedTile.index, boardId);
    this.metadataService.linkBoard(this.selectedTile.index, boardId);
    for (const id of current) {
      if (id !== boardId) {
        this.metadataService.linkBoard(this.selectedTile.index, id);
      }
    }
  }

  moodBoardCardImage(card: BoardCard): string | null {
    if (card.imageUrl) {return card.imageUrl;}
    if (card.imageSource === 'world_asset' && card.sourceId) {
      const asset = this.worldArt.find(item => item.id === card.sourceId);
      return asset ? this.artImageUrl(asset) : null;
    }
    if (card.imageSource === 'character' && card.sourceId) {
      const character = this.inhabitants.find(item => item.id === card.sourceId);
      return character ? this.inhabitantImageUrl(character) : null;
    }
    return null;
  }

  private moodBoardOverview(worldName: string): string {
    const terrain = this.selectedTile?.terrain ?? 'unknown';
    const biome = this.selectedTile?.biome ?? 'none';
    const resource = this.selectedTile?.resource ?? 'none';
    const tags = this.metadata?.tags?.length ? this.metadata.tags.join(', ') : 'none';
    const description = this.metadata?.description || 'No description yet.';
    return [
      `World: ${worldName}`,
      `Terrain: ${terrain}`,
      `Biome: ${biome}`,
      `Resource: ${resource}`,
      `Tags: ${tags}`,
      '',
      description
    ].join('\n');
  }

  private moodBoardPalette(): string {
    const palette = ['cyan energy', 'deep navy atmosphere', 'warm solar amber'];
    if (this.selectedTile?.terrain === 'water') {palette.push('aquatic blue', 'bioluminescent teal');}
    if (this.selectedTile?.terrain === 'mountain') {palette.push('basalt brown', 'golden ridge light');}
    if (this.selectedTile?.biome === 'forest') {palette.push('verdant green', 'moss glow');}
    if (this.selectedTile?.biome === 'desert') {palette.push('sand gold', 'rust orange');}
    if (this.selectedTile?.biome === 'tundra') {palette.push('ice blue', 'aurora violet');}
    if (this.selectedTile?.resource === 'node') {palette.push('magenta resource glow');}
    return palette.map(item => `- ${item}`).join('\n');
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
