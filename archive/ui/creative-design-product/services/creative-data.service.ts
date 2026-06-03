import { Injectable } from '@angular/core';

export interface BoardCard {
  id: string;
  title: string;
  imageUrl?: string;
  imageSource?: 'world_asset' | 'character' | 'manual';
  sourceId?: string;
  notes?: string;
  createdAt: string;
}
export interface Board { id: string; worldId?: string; title: string; cards: BoardCard[]; createdAt: string; }

export interface MediaItem { id: string; url: string; type: 'image' | 'video' | 'audio'; caption?: string; }
export type WikiPageType = 'Lore' | 'Factions' | 'Biomes' | 'Items' | 'Technology';
export interface WikiPageMetadata {
  type?: WikiPageType;
  tags?: string[];
  template?: string;
  connections?: string[];
  color?: string;
  icon?: string;
}
export interface WikiPage {
  id: string;
  worldId?: string;
  title: string;
  content: string;
  richContent?: any;
  media?: MediaItem[];
  metadata?: WikiPageMetadata;
  createdAt: string;
  updatedAt: string;
}

@Injectable({ providedIn: 'root' })
export class CreativeDataService {
  private boardsKey = 'creative.boards.v1';
  private wikiKey = 'creative.wiki.v1';

  private read<T>(key: string): T[] { try { return JSON.parse(localStorage.getItem(key) || '[]'); } catch { return []; } }
  public write<T>(key: string, value: T[]): void {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
      if (key !== this.boardsKey || !this.isQuotaError(err)) {throw err;}
      const compacted = this.compactBoards(value as Board[]) as T[];
      localStorage.setItem(key, JSON.stringify(compacted));
    }
  }

  private isQuotaError(err: unknown): boolean {
    return err instanceof DOMException && (
      err.name === 'QuotaExceededError' ||
      err.name === 'NS_ERROR_DOM_QUOTA_REACHED'
    );
  }

  private compactBoards(boards: Board[]): Board[] {
    return boards.map(board => ({
      ...board,
      cards: board.cards.map(card => ({
        ...card,
        imageUrl: card.imageUrl?.startsWith('data:image/') ? undefined : card.imageUrl,
        notes: card.notes
          ? `${card.notes}\n\n[Storage note: embedded image data was compacted; source reference preserved when available.]`
          : card.notes
      }))
    }));
  }

  listBoards(worldId?: string): Board[] {
    const all = this.read<Board>(this.boardsKey);
    return worldId ? all.filter(b => b.worldId === worldId) : all;
  }
  createBoard(partial: { title: string; worldId?: string }): Board {
    const board: Board = {
      id: crypto.randomUUID(),
      title: partial.title,
      worldId: partial.worldId,
      cards: [],
      createdAt: new Date().toISOString()
    };
    const all = this.listBoards(); all.push(board); this.write(this.boardsKey, all); return board;
  }

  getBoard(boardId: string): Board | null {
    return this.listBoards().find(board => board.id === boardId) ?? null;
  }

  deleteBoard(boardId: string): boolean {
    const all = this.listBoards();
    const next = all.filter(board => board.id !== boardId);
    if (next.length === all.length) {return false;}
    this.write(this.boardsKey, next);
    return true;
  }

  addCardToBoard(boardId: string, card: Omit<BoardCard, 'id' | 'createdAt'>): Board | null {
    const all = this.listBoards();
    const index = all.findIndex(board => board.id === boardId);
    if (index < 0) {return null;}
    const nextCard: BoardCard = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      ...card
    };
    all[index] = { ...all[index], cards: [...all[index].cards, nextCard] };
    this.write(this.boardsKey, all);
    return all[index];
  }

  createBoardWithCards(
    partial: { title: string; worldId?: string },
    cards: Array<Omit<BoardCard, 'id' | 'createdAt'>>
  ): Board {
    let board = this.createBoard(partial);
    for (const card of cards) {
      board = this.addCardToBoard(board.id, card) ?? board;
    }
    return board;
  }

  listWiki(worldId?: string): WikiPage[] {
    const all = this.read<WikiPage>(this.wikiKey);
    return worldId ? all.filter(p => p.worldId === worldId) : all;
  }
  upsertWiki(page: WikiPage): void {
    const all = this.read<WikiPage>(this.wikiKey);
    const i = all.findIndex(p => p.id === page.id);
    if (i >= 0) {all[i] = page;} else {all.push(page);}
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


