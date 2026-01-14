import { Component, Input, Output, EventEmitter, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subject, takeUntil } from 'rxjs';
import { TileMetadataService } from '../engine/tile-metadata.service';
import { TileWorldMetadata, QuickNote, PinnedItem, AIObservation, ConnectionType, CONNECTION_STYLES, WorldConnection } from '../engine/tile-metadata.model';
import { CreativeDataService, WikiPage, Board } from '../../../creative-design-product/services/creative-data.service';

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
export class WorldDetailPanelComponent implements OnInit, OnDestroy {
  @Input() selectedTile: SelectedTileInfo | null = null;
  @Output() requestAIPrompt = new EventEmitter<{ tileIndex: number; prompt: string }>();
  @Output() createConnection = new EventEmitter<{ fromIndex: number }>();
  @Output() closePanel = new EventEmitter<void>();

  metadata: TileWorldMetadata | null = null;
  linkedWikiPages: WikiPage[] = [];
  linkedBoards: Board[] = [];
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
    private creativeData: CreativeDataService
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

  ngOnChanges(): void {
    if (this.selectedTile) {
      this.metadataService.setSelectedTile(this.selectedTile.index);
    } else {
      this.metadataService.setSelectedTile(null);
    }
  }

  private loadLinkedContent(): void {
    if (!this.metadata) {
      this.linkedWikiPages = [];
      this.linkedBoards = [];
      return;
    }

    // Load linked wiki pages
    const allWiki = this.creativeData.listWiki();
    this.linkedWikiPages = allWiki.filter(p => this.metadata?.wikiPageIds?.includes(p.id));

    // Load linked boards
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
    if (!this.selectedTile) return;

    const items = event.clipboardData?.items;
    if (!items) return;

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
    if (!this.selectedTile) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const imageData = e.target?.result as string;
      this.metadataService.addPinnedImage(this.selectedTile!.index, imageData, file.name);
    };
    reader.readAsDataURL(file);
  }

  captureLink(url?: string): void {
    if (!this.selectedTile) return;
    const linkUrl = url || this.quickCaptureUrl.trim();
    if (!linkUrl) return;

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
    if (!this.selectedTile) return;

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
    if (!this.selectedTile) return -1;
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
    if (!this.selectedTile) return;
    const title = this.metadata?.name || `World ${this.selectedTile.index}`;
    const page = this.creativeData.createWiki(undefined, `${title} - Lore`);
    this.metadataService.linkWikiPage(this.selectedTile.index, page.id);
    this.loadLinkedContent();
  }

  unlinkWikiPage(pageId: string): void {
    if (this.selectedTile) {
      this.metadataService.unlinkWikiPage(this.selectedTile.index, pageId);
      this.loadLinkedContent();
    }
  }

  createBoard(): void {
    if (!this.selectedTile) return;
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
    if (!this.metadata) return false;
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
