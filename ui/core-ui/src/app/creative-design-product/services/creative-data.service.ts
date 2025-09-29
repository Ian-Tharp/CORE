import { Injectable } from '@angular/core';

export interface BoardCard { id: string; title: string; imageUrl?: string; notes?: string; createdAt: string; }
export interface Board { id: string; worldId?: string; title: string; cards: BoardCard[]; createdAt: string; }

export interface MediaItem { id: string; url: string; type: 'image' | 'video' | 'audio'; caption?: string; }
export type WikiPageType = 'Lore' | 'Factions' | 'Biomes' | 'Items' | 'Technology';
export interface WikiPageMetadata { type?: WikiPageType; tags?: string[]; template?: string; connections?: string[]; color?: string; icon?: string; }
export interface WikiPage { id: string; worldId?: string; title: string; content: string; richContent?: any; media?: MediaItem[]; metadata?: WikiPageMetadata; createdAt: string; updatedAt: string; }

@Injectable({ providedIn: 'root' })
export class CreativeDataService {
  private boardsKey = 'creative.boards.v1';
  private wikiKey = 'creative.wiki.v1';

  private read<T>(key: string): T[] { try { return JSON.parse(localStorage.getItem(key) || '[]'); } catch { return []; } }
  public write<T>(key: string, value: T[]): void { localStorage.setItem(key, JSON.stringify(value)); }

  listBoards(worldId?: string): Board[] {
    const all = this.read<Board>(this.boardsKey);
    return worldId ? all.filter(b => b.worldId === worldId) : all;
  }
  createBoard(partial: { title: string; worldId?: string }): Board {
    const board: Board = { id: crypto.randomUUID(), title: partial.title, worldId: partial.worldId, cards: [], createdAt: new Date().toISOString() };
    const all = this.listBoards(); all.push(board); this.write(this.boardsKey, all); return board;
  }

  listWiki(worldId?: string): WikiPage[] {
    const all = this.read<WikiPage>(this.wikiKey);
    return worldId ? all.filter(p => p.worldId === worldId) : all;
  }
  upsertWiki(page: WikiPage): void {
    const all = this.read<WikiPage>(this.wikiKey);
    const i = all.findIndex(p => p.id === page.id);
    if (i >= 0) all[i] = page; else all.push(page);
    this.write(this.wikiKey, all);
  }
  createWiki(worldId: string | undefined, title: string): WikiPage {
    const page: WikiPage = {
      id: crypto.randomUUID(),
      worldId,
      title,
      content: '',
      richContent: null,
      media: [],
      metadata: { tags: [], connections: [] },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    this.upsertWiki(page);
    return page;
  }
}


