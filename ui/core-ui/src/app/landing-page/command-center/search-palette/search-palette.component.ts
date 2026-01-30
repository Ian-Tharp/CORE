import { Component, EventEmitter, HostListener, OnInit, Output, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { TileMetadataService } from '../engine/tile-metadata.service';
import { TileGridService, TerrainState, BiomeState, ResourceState } from '../engine/tile-grid.service';
import { TileWorldMetadata, WorldConnection, CONNECTION_STYLES } from '../engine/tile-metadata.model';
import { CreativeDataService, WikiPage } from '../../../creative-design-product/services/creative-data.service';

export interface SearchResult {
  type: 'world' | 'wiki' | 'connection' | 'tag';
  id: string | number;
  title: string;
  subtitle?: string;
  icon?: string;
  tileIndex?: number;
}

@Component({
  selector: 'app-search-palette',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './search-palette.component.html',
  styleUrl: './search-palette.component.scss'
})
export class SearchPaletteComponent implements OnInit {
  @Output() close = new EventEmitter<void>();
  @Output() selectResult = new EventEmitter<SearchResult>();

  private readonly metadataService = inject(TileMetadataService);
  private readonly tileGrid = inject(TileGridService);
  private readonly creativeData = inject(CreativeDataService);

  isOpen = false;
  query = '';
  results: SearchResult[] = [];
  selectedIndex = 0;

  // Quick filters
  filters = {
    terrain: null as TerrainState | null,
    biome: null as BiomeState | null,
    hasContent: false,
    hasConnections: false
  };

  terrainOptions: TerrainState[] = ['plain', 'water', 'mountain'];
  biomeOptions: BiomeState[] = ['none', 'forest', 'desert', 'tundra'];

  allTags: string[] = [];

  ngOnInit(): void {
    this.loadAllTags();
  }

  @HostListener('document:keydown', ['$event'])
  handleKeydown(event: KeyboardEvent): void {
    // Cmd/Ctrl + K to toggle
    if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
      event.preventDefault();
      this.toggle();
      return;
    }

    if (!this.isOpen) return;

    // Escape to close
    if (event.key === 'Escape') {
      this.closePalette();
      return;
    }

    // Arrow navigation
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      this.selectedIndex = Math.min(this.selectedIndex + 1, this.results.length - 1);
      return;
    }

    if (event.key === 'ArrowUp') {
      event.preventDefault();
      this.selectedIndex = Math.max(this.selectedIndex - 1, 0);
      return;
    }

    // Enter to select
    if (event.key === 'Enter' && this.results.length > 0) {
      event.preventDefault();
      this.onSelectResult(this.results[this.selectedIndex]);
      return;
    }
  }

  toggle(): void {
    this.isOpen = !this.isOpen;
    if (this.isOpen) {
      this.query = '';
      this.results = [];
      this.selectedIndex = 0;
      this.loadAllTags();
      // Show initial results (worlds with content)
      this.showInitialResults();
    }
  }

  open(): void {
    this.isOpen = true;
    this.query = '';
    this.results = [];
    this.selectedIndex = 0;
    this.loadAllTags();
    this.showInitialResults();
  }

  closePalette(): void {
    this.isOpen = false;
    this.close.emit();
  }

  private loadAllTags(): void {
    this.allTags = this.metadataService.getAllTags();
  }

  private showInitialResults(): void {
    // Show worlds that have content
    const worldsWithContent = this.metadataService.getTilesWithContent();
    this.results = worldsWithContent.slice(0, 10).map(meta => this.worldToResult(meta));
  }

  onSearch(): void {
    this.selectedIndex = 0;

    if (!this.query.trim() && !this.hasActiveFilters()) {
      this.showInitialResults();
      return;
    }

    this.results = [];
    const q = this.query.toLowerCase().trim();

    // Search worlds
    const worldResults = this.searchWorlds(q);
    this.results.push(...worldResults);

    // Search wiki pages
    const wikiResults = this.searchWiki(q);
    this.results.push(...wikiResults);

    // Search tags
    if (q) {
      const tagResults = this.searchTags(q);
      this.results.push(...tagResults);
    }

    // Limit results
    this.results = this.results.slice(0, 20);
  }

  private searchWorlds(query: string): SearchResult[] {
    const results: SearchResult[] = [];
    const allMetadata = this.metadataService.getTilesWithContent();

    for (const meta of allMetadata) {
      // Apply filters
      if (!this.matchesFilters(meta)) continue;

      // Apply text query
      if (query) {
        const nameMatch = meta.name?.toLowerCase().includes(query);
        const descMatch = meta.description?.toLowerCase().includes(query);
        const tagMatch = meta.tags?.some(t => t.toLowerCase().includes(query));
        const noteMatch = meta.quickNotes?.some(n => n.content.toLowerCase().includes(query));

        if (!nameMatch && !descMatch && !tagMatch && !noteMatch) continue;
      }

      results.push(this.worldToResult(meta));
    }

    return results;
  }

  private worldToResult(meta: TileWorldMetadata): SearchResult {
    const noteCount = meta.quickNotes?.length || 0;
    const connCount = meta.connectionIds?.length || 0;
    let subtitle = `Tile (${meta.tileIndex})`;
    if (noteCount > 0) subtitle += ` • ${noteCount} notes`;
    if (connCount > 0) subtitle += ` • ${connCount} connections`;

    return {
      type: 'world',
      id: meta.tileIndex,
      title: meta.name || `World ${meta.tileIndex}`,
      subtitle,
      tileIndex: meta.tileIndex
    };
  }

  private searchWiki(query: string): SearchResult[] {
    if (!query) return [];

    const results: SearchResult[] = [];
    const allWiki = this.creativeData.listWiki();

    for (const page of allWiki) {
      const titleMatch = page.title.toLowerCase().includes(query);
      const contentMatch = page.content.toLowerCase().includes(query);

      if (titleMatch || contentMatch) {
        results.push({
          type: 'wiki',
          id: page.id,
          title: page.title,
          subtitle: page.metadata?.type || 'Wiki Page',
          icon: '📖'
        });
      }
    }

    return results;
  }

  private searchTags(query: string): SearchResult[] {
    const results: SearchResult[] = [];

    for (const tag of this.allTags) {
      if (tag.toLowerCase().includes(query)) {
        const tilesWithTag = this.metadataService.filterByTag(tag);
        results.push({
          type: 'tag',
          id: tag,
          title: `#${tag}`,
          subtitle: `${tilesWithTag.length} worlds`,
          icon: '🏷️'
        });
      }
    }

    return results;
  }

  private matchesFilters(meta: TileWorldMetadata): boolean {
    if (this.filters.hasContent) {
      const hasContent = meta.name || meta.description ||
        (meta.quickNotes && meta.quickNotes.length > 0) ||
        (meta.pinnedItems && meta.pinnedItems.length > 0);
      if (!hasContent) return false;
    }

    if (this.filters.hasConnections) {
      if (!meta.connectionIds || meta.connectionIds.length === 0) return false;
    }

    // Terrain and biome filters would require cross-referencing with TileGridService
    // For now, we skip those filters in the metadata search

    return true;
  }

  private hasActiveFilters(): boolean {
    return this.filters.hasContent || this.filters.hasConnections ||
      this.filters.terrain !== null || this.filters.biome !== null;
  }

  onSelectResult(result: SearchResult): void {
    this.selectResult.emit(result);
    this.closePalette();
  }

  onFilterChange(): void {
    this.onSearch();
  }

  clearFilters(): void {
    this.filters = {
      terrain: null,
      biome: null,
      hasContent: false,
      hasConnections: false
    };
    this.onSearch();
  }

  onBackdropClick(event: MouseEvent): void {
    if ((event.target as HTMLElement).classList.contains('palette-backdrop')) {
      this.closePalette();
    }
  }
}
