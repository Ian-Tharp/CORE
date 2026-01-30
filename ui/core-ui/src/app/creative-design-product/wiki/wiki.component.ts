import { Component, inject, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { CreativeDataService, WikiPage, MediaItem, WikiPageType } from '../services/creative-data.service';
import { CreativeService } from '../../services/creative/creative.service';

@Component({
  selector: 'app-wiki',
  imports: [CommonModule, FormsModule],
  templateUrl: './wiki.component.html',
  styleUrl: './wiki.component.scss'
})
export class WikiComponent {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;
  @ViewChild('contextMenu') contextMenu!: ElementRef<HTMLDivElement>;

  private readonly route = inject(ActivatedRoute);
  private readonly data = inject(CreativeDataService);
  private readonly api = inject(CreativeService);
  worldId?: string | null = null;
  pages: WikiPage[] = [];
  editing?: WikiPage;
  newTitle = '';
  section: 'All' | WikiPageType = 'All';

  // Enhanced UI state
  viewMode: 'list' | 'grid' | 'timeline' = 'list';
  searchQuery = '';
  selectedTags: string[] = [];
  showTemplateSelector = false;

  // Context menu state
  contextMenuVisible = false;
  contextMenuX = 0;
  contextMenuY = 0;
  contextPage?: WikiPage;
  editingTitle = false;
  tempTitle = '';

  // Rich content state
  showMediaUpload = false;
  dragOverEditor = false;

  // Templates
  templates = {
    'Character': {
      content: '# Character Name\n\n## Appearance\n\n## Background\n\n## Abilities\n\n## Relationships\n\n## Notes',
      metadata: { type: 'Lore' as WikiPageType, tags: ['character'], icon: '👤' }
    },
    'Location': {
      content: '# Location Name\n\n## Description\n\n## Geography\n\n## Notable Features\n\n## Inhabitants\n\n## History',
      metadata: { type: 'Biomes' as WikiPageType, tags: ['location'], icon: '🏛️' }
    },
    'Faction': {
      content: '# Faction Name\n\n## Overview\n\n## Leadership\n\n## Goals\n\n## Resources\n\n## Relationships\n\n## Territory',
      metadata: { type: 'Factions' as WikiPageType, tags: ['faction'], icon: '⚔️' }
    },
    'Item': {
      content: '# Item Name\n\n## Description\n\n## Properties\n\n## History\n\n## Current Location\n\n## Value',
      metadata: { type: 'Items' as WikiPageType, tags: ['item'], icon: '🗡️' }
    },
    'Event': {
      content: '# Event Name\n\n## When\n\n## Where\n\n## What Happened\n\n## Participants\n\n## Consequences',
      metadata: { type: 'Lore' as WikiPageType, tags: ['event'], icon: '📜' }
    }
  };

  // Character builder
  characterName = '';
  characterTraits = '';
  characterImageB64: string | null = null;

  ngOnInit(): void {
    this.worldId = this.route.snapshot.queryParamMap.get('projectId');
    this.refresh();
    document.addEventListener('click', () => this.hideContextMenu());
  }

  refresh(): void {
    this.api.listWiki(this.worldId || undefined).subscribe({
      next: (remote) => {
        if (remote && remote.length > 0) {
          this.pages = remote.map(r => ({
            id: r.id, worldId: r.world_id, title: r.title, content: r.content,
            createdAt: r.created_at, updatedAt: r.updated_at,
            metadata: (r as any).metadata || {},
            media: (r as any).media || []
          } as WikiPage));
          if (!this.editing && this.pages.length > 0) {
            this.editing = this.pages[0];
          }
        } else {
          this.pages = this.data.listWiki(this.worldId || undefined);
        }
      },
      error: () => { this.pages = this.data.listWiki(this.worldId || undefined); }
    });
  }

  // Enhanced page creation and management
  create(): void {
    if (!this.newTitle.trim()) return;
    const local = this.data.createWiki(this.worldId || undefined, this.newTitle.trim());
    local.metadata = {
      type: this.section === 'All' ? undefined : this.section as WikiPageType,
      tags: [],
      connections: []
    };
    this.api.createWiki({
      world_id: this.worldId || undefined,
      title: local.title,
      content: local.content,
      metadata: local.metadata
    }).subscribe();
    this.editing = local;
    this.newTitle = '';
    this.refresh();
  }

  createFromTemplate(templateName: string): void {
    const template = this.templates[templateName as keyof typeof this.templates];
    if (!template) return;

    const local = this.data.createWiki(this.worldId || undefined, templateName + ' Page');
    local.content = template.content;
    local.metadata = { ...template.metadata, tags: [...template.metadata.tags!], connections: [] };

    this.api.createWiki({
      world_id: this.worldId || undefined,
      title: local.title,
      content: local.content,
      metadata: local.metadata
    }).subscribe();

    this.editing = local;
    this.showTemplateSelector = false;
    this.refresh();
  }

  save(): void {
    if (!this.editing) return;
    this.editing.updatedAt = new Date().toISOString();
    if (!this.editing.metadata) this.editing.metadata = {};
    this.data.upsertWiki(this.editing);
    this.api.updateWiki(this.editing.id, {
      world_id: this.worldId || undefined,
      title: this.editing.title,
      content: this.editing.content,
      metadata: this.editing.metadata
    }).subscribe();
  }

  // Enhanced filtering and search
  filteredPages(): WikiPage[] {
    let filtered = this.pages;

    // Filter by section
    if (this.section !== 'All') {
      filtered = filtered.filter(p => p.metadata?.type === this.section);
    }

    // Filter by search query
    if (this.searchQuery.trim()) {
      const query = this.searchQuery.toLowerCase();
      filtered = filtered.filter(p =>
        p.title.toLowerCase().includes(query) ||
        p.content.toLowerCase().includes(query) ||
        p.metadata?.tags?.some(tag => tag.toLowerCase().includes(query))
      );
    }

    // Filter by selected tags
    if (this.selectedTags.length > 0) {
      filtered = filtered.filter(p =>
        this.selectedTags.every(tag => p.metadata?.tags?.includes(tag))
      );
    }

    return filtered;
  }

  getAllTags(): string[] {
    const allTags = new Set<string>();
    this.pages.forEach(page => {
      page.metadata?.tags?.forEach(tag => allTags.add(tag));
    });
    return Array.from(allTags).sort();
  }

  toggleTag(tag: string): void {
    const index = this.selectedTags.indexOf(tag);
    if (index > -1) {
      this.selectedTags.splice(index, 1);
    } else {
      this.selectedTags.push(tag);
    }
  }

  // Context menu functionality
  showContextMenu(event: MouseEvent, page: WikiPage): void {
    event.preventDefault();
    event.stopPropagation();
    this.contextPage = page;
    this.contextMenuX = event.clientX;
    this.contextMenuY = event.clientY;
    this.contextMenuVisible = true;
  }

  hideContextMenu(): void {
    this.contextMenuVisible = false;
    this.contextPage = undefined;
  }

  duplicatePage(): void {
    if (!this.contextPage) return;
    const newPage = this.data.createWiki(this.worldId || undefined, `${this.contextPage.title} (Copy)`);
    newPage.content = this.contextPage.content;
    newPage.metadata = { ...this.contextPage.metadata };
    this.api.createWiki({
      world_id: this.worldId || undefined,
      title: newPage.title,
      content: newPage.content,
      metadata: newPage.metadata
    }).subscribe();
    this.hideContextMenu();
    this.refresh();
  }

  editPageTitle(): void {
    if (!this.contextPage) return;
    this.tempTitle = this.contextPage.title;
    this.editingTitle = true;
    this.hideContextMenu();

    // Focus the input after the view updates
    setTimeout(() => {
      const input = document.querySelector('.title-edit-input') as HTMLInputElement;
      if (input) {
        input.focus();
        input.select();
      }
    }, 0);
  }

  confirmTitleEdit(): void {
    if (!this.contextPage || !this.tempTitle.trim()) {
      this.cancelTitleEdit();
      return;
    }

    // Find the page in the array and update it
    const pageIndex = this.pages.findIndex(p => p.id === this.contextPage!.id);
    if (pageIndex >= 0) {
      this.pages[pageIndex].title = this.tempTitle.trim();
      this.pages[pageIndex].updatedAt = new Date().toISOString();

      // Update local storage
      this.data.upsertWiki(this.pages[pageIndex]);

      // Update via API
      this.api.updateWiki(this.contextPage.id, {
        world_id: this.worldId || undefined,
        title: this.tempTitle.trim(),
        content: this.contextPage.content,
        metadata: this.contextPage.metadata
      }).subscribe();

      // Update the editing page if it's the same
      if (this.editing?.id === this.contextPage.id) {
        this.editing.title = this.tempTitle.trim();
        this.editing.updatedAt = new Date().toISOString();
      }
    }

    this.editingTitle = false;
    this.tempTitle = '';
  }

  cancelTitleEdit(): void {
    this.editingTitle = false;
    this.tempTitle = '';
  }

  deletePage(): void {
    if (!this.contextPage) return;
    if (confirm(`Delete "${this.contextPage.title}"?`)) {
      this.pages = this.pages.filter(p => p.id !== this.contextPage!.id);
      this.data.write('creative.wiki.v1', this.pages);
      if (this.editing?.id === this.contextPage.id) {
        this.editing = this.pages[0] || undefined;
      }
      this.hideContextMenu();
    }
  }

  // Media handling
  addTag(tag: string): void {
    if (!this.editing || !tag.trim()) return;
    if (!this.editing.metadata) this.editing.metadata = {};
    if (!this.editing.metadata.tags) this.editing.metadata.tags = [];
    if (!this.editing.metadata.tags.includes(tag.trim())) {
      this.editing.metadata.tags.push(tag.trim());
      this.save();
    }
  }

  removeTag(tag: string): void {
    if (!this.editing?.metadata?.tags) return;
    this.editing.metadata.tags = this.editing.metadata.tags.filter(t => t !== tag);
    this.save();
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      const file = input.files[0];
      this.uploadMedia(file);
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragOverEditor = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.dragOverEditor = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragOverEditor = false;

    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      Array.from(event.dataTransfer.files).forEach(file => {
        this.uploadMedia(file);
      });
    }
  }

  uploadMedia(file: File): void {
    if (!this.editing) return;

    const reader = new FileReader();
    reader.onload = () => {
      const mediaItem: MediaItem = {
        id: crypto.randomUUID(),
        url: reader.result as string,
        type: file.type.startsWith('image/') ? 'image' :
              file.type.startsWith('video/') ? 'video' : 'audio'
      };

      if (!this.editing!.media) this.editing!.media = [];
      this.editing!.media.push(mediaItem);
      this.save();
    };
    reader.readAsDataURL(file);
  }

  removeMedia(mediaId: string): void {
    if (!this.editing?.media) return;
    this.editing.media = this.editing.media.filter(m => m.id !== mediaId);
    this.save();
  }

  createOfType(type: WikiPageType): void {
    this.section = type;
    this.newTitle = `${type} Page`;
    this.create();
  }

  // AI-powered features
  getContentSuggestions(): string[] {
    if (!this.editing?.metadata?.type) return [];

    const suggestions: Record<WikiPageType, string[]> = {
      'Lore': [
        'Add historical timeline with key events',
        'Describe cultural traditions and customs',
        'Include mythology and creation stories',
        'Detail religious or spiritual beliefs',
        'Explain social hierarchy and governance'
      ],
      'Factions': [
        'Define faction goals and motivations',
        'List key leaders and their roles',
        'Describe territory and resources',
        'Outline relationships with other factions',
        'Detail military capabilities and tactics'
      ],
      'Biomes': [
        'Add climate and weather patterns',
        'List native flora and fauna',
        'Describe geological features',
        'Include natural resources available',
        'Detail environmental hazards or challenges'
      ],
      'Items': [
        'Specify magical or technological properties',
        'Add creation method or origin story',
        'Include current market value or rarity',
        'Detail any curses or beneficial effects',
        'Describe visual appearance and materials'
      ],
      'Technology': [
        'Explain how the technology works',
        'List required resources or components',
        'Describe societal impact and adoption',
        'Include potential risks or limitations',
        'Detail manufacturing or research process'
      ]
    };

    return suggestions[this.editing.metadata.type] || [];
  }

  getRecommendedTags(): string[] {
    if (!this.editing?.metadata?.type) return [];

    const typeBasedTags: Record<WikiPageType, string[]> = {
      'Lore': ['history', 'culture', 'mythology', 'tradition', 'ancient', 'legend'],
      'Factions': ['political', 'military', 'alliance', 'conflict', 'leadership', 'territory'],
      'Biomes': ['environment', 'climate', 'wildlife', 'resources', 'geography', 'ecosystem'],
      'Items': ['artifact', 'weapon', 'tool', 'magical', 'rare', 'valuable'],
      'Technology': ['innovation', 'science', 'automation', 'advancement', 'experimental', 'practical']
    };

    const baseTags = typeBasedTags[this.editing.metadata.type] || [];
    const existingTags = this.editing.metadata.tags || [];

    return baseTags.filter(tag => !existingTags.includes(tag));
  }

  getConnectionSuggestions(): WikiPage[] {
    if (!this.editing) return [];

    const currentContent = this.editing.content.toLowerCase();
    const currentTags = this.editing.metadata?.tags || [];

    return this.pages
      .filter(page => page.id !== this.editing!.id)
      .filter(page => {
        const hasCommonTags = page.metadata?.tags?.some(tag => currentTags.includes(tag));
        const hasContentOverlap = currentContent.includes(page.title.toLowerCase()) ||
                                  page.content.toLowerCase().includes(this.editing!.title.toLowerCase());
        return hasCommonTags || hasContentOverlap;
      })
      .slice(0, 5);
  }

  applySuggestion(suggestion: string): void {
    if (!this.editing) return;

    const currentContent = this.editing.content;
    const newSection = `\n\n## ${suggestion.replace(/^(Add|Include|Detail|Describe|List|Explain|Define|Specify|Outline) /, '')}\n\n`;

    this.editing.content = currentContent + newSection;
    this.save();
  }

  addRecommendedTag(tag: string): void {
    this.addTag(tag);
  }

  connectToPage(targetPage: WikiPage): void {
    if (!this.editing || !this.editing.metadata) return;

    if (!this.editing.metadata.connections) {
      this.editing.metadata.connections = [];
    }

    if (!this.editing.metadata.connections.includes(targetPage.id)) {
      this.editing.metadata.connections.push(targetPage.id);

      // Add bidirectional connection
      if (!targetPage.metadata) targetPage.metadata = {};
      if (!targetPage.metadata.connections) targetPage.metadata.connections = [];
      if (!targetPage.metadata.connections.includes(this.editing.id)) {
        targetPage.metadata.connections.push(this.editing.id);
        this.data.upsertWiki(targetPage);
      }

      this.save();
    }
  }

  getConnectedPages(): WikiPage[] {
    if (!this.editing?.metadata?.connections) return [];

    return this.pages.filter(page =>
      this.editing!.metadata!.connections!.includes(page.id)
    );
  }

  // Enhanced content generation
  generateContentOutline(): void {
    if (!this.editing?.metadata?.type) return;

    const outlines: Record<WikiPageType, string> = {
      'Lore': `# ${this.editing.title}\n\n## Overview\n\n## Historical Context\n\n## Cultural Significance\n\n## Key Figures\n\n## Impact and Legacy\n\n## Related Events`,
      'Factions': `# ${this.editing.title}\n\n## Overview\n\n## Leadership\n- Leader:\n- Structure:\n\n## Goals and Motivations\n\n## Territory and Resources\n\n## Military Capabilities\n\n## Allies and Enemies\n\n## Recent Activities`,
      'Biomes': `# ${this.editing.title}\n\n## Description\n\n## Geography and Climate\n\n## Flora and Fauna\n\n## Natural Resources\n\n## Inhabitants\n\n## Notable Locations\n\n## Environmental Challenges`,
      'Items': `# ${this.editing.title}\n\n## Description\n\n## Properties and Abilities\n\n## Origin and History\n\n## Current Location\n\n## Value and Rarity\n\n## Related Items\n\n## Lore and Legends`,
      'Technology': `# ${this.editing.title}\n\n## Overview\n\n## Technical Specifications\n\n## Development Process\n\n## Applications and Uses\n\n## Societal Impact\n\n## Limitations and Risks\n\n## Future Development`
    };

    if (outlines[this.editing.metadata.type]) {
      this.editing.content = outlines[this.editing.metadata.type];
      this.save();
    }
  }

  // Utility methods for template safety
  safeGetTags(page: WikiPage): string[] {
    return page.metadata?.tags || [];
  }

  safeGetMediaUrl(page: WikiPage, index: number): string | undefined {
    return page.media?.[index]?.url;
  }

  safeHasMedia(page: WikiPage): boolean {
    return !!(page.media && page.media.length > 0);
  }

  // Character builder
  createCharacter(): void {
    const name = this.characterName.trim();
    if (!name) return;
    let traits: any = {};
    try {
      traits = this.characterTraits ? JSON.parse(this.characterTraits) : {};
    } catch {
      traits = { description: this.characterTraits };
    }
    this.api.createCharacter({
      world_id: this.worldId || undefined,
      name,
      traits
    }).subscribe(({ id }) => {
      const prompt = `${name}, ${JSON.stringify(traits)}`;
      this.api.generateCharacterImage(id, prompt).subscribe();
    });
    this.characterName = '';
    this.characterTraits = '';
  }
}
