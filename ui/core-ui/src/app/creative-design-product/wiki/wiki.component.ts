import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute } from '@angular/router';
import { CreativeDataService, WikiPage } from '../services/creative-data.service';
import { CreativeService } from '../../services/creative/creative.service';

@Component({
  selector: 'app-wiki',
  imports: [CommonModule, FormsModule],
  templateUrl: './wiki.component.html',
  styleUrl: './wiki.component.scss'
})
export class WikiComponent {
  private readonly route = inject(ActivatedRoute);
  private readonly data = inject(CreativeDataService);
  private readonly api = inject(CreativeService);
  worldId?: string | null = null;
  pages: WikiPage[] = [];
  editing?: WikiPage;
  newTitle = '';
  section: 'All' | 'Lore' | 'Factions' | 'Biomes' | 'Items' | 'Technology' = 'All';
  // Character builder
  characterName = '';
  characterTraits = '';
  characterImageB64: string | null = null;

  ngOnInit(): void { this.worldId = this.route.snapshot.queryParamMap.get('projectId'); this.refresh(); }
  refresh(): void {
    // Prefer remote; fallback local
    this.api.listWiki(this.worldId || undefined).subscribe({
      next: (remote) => {
        if (remote && remote.length > 0) {
          this.pages = remote.map(r => ({ id: r.id, worldId: r.world_id, title: r.title, content: r.content, createdAt: r.created_at, updatedAt: r.updated_at, metadata: (r as any).metadata } as any));
          this.editing = this.pages[0];
        } else {
          this.pages = this.data.listWiki(this.worldId || undefined);
        }
      },
      error: () => { this.pages = this.data.listWiki(this.worldId || undefined); }
    });
  }
  create(): void {
    if (!this.newTitle.trim()) return;
    const local = this.data.createWiki(this.worldId || undefined, this.newTitle.trim());
    this.api.createWiki({ world_id: this.worldId || undefined, title: local.title, content: local.content, metadata: { type: this.section } }).subscribe();
    this.editing = local; this.newTitle=''; this.refresh();
  }
  save(): void {
    if (!this.editing) return;
    this.editing.updatedAt = new Date().toISOString();
    this.data.upsertWiki(this.editing);
    this.api.updateWiki(this.editing.id, { world_id: this.worldId || undefined, title: this.editing.title, content: this.editing.content }).subscribe();
    this.refresh();
  }

  filteredPages(): WikiPage[] {
    if (this.section === 'All') return this.pages;
    return this.pages.filter((p: any) => (p as any).metadata?.type === this.section);
  }

  createOfType(type: 'Lore' | 'Factions' | 'Biomes' | 'Items' | 'Technology'): void {
    this.section = type;
    this.newTitle = `${type} Page`;
    this.create();
  }

  // Character builder
  createCharacter(): void {
    const name = this.characterName.trim();
    if (!name) return;
    let traits: any = {};
    try { traits = this.characterTraits ? JSON.parse(this.characterTraits) : {}; } catch { traits = { description: this.characterTraits }; }
    this.api.createCharacter({ world_id: this.worldId || undefined, name, traits }).subscribe(({ id }) => {
      const prompt = `${name}, ${JSON.stringify(traits)}`;
      this.api.generateCharacterImage(id, prompt).subscribe();
    });
    this.characterName = ''; this.characterTraits = '';
  }
}
