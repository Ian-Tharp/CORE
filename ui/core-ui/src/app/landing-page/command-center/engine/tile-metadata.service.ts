import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import {
  TileWorldMetadata,
  WorldConnection,
  TileMetadataSnapshot,
  ConnectionType,
  QuickNote,
  PinnedItem,
  AIObservation
} from './tile-metadata.model';

@Injectable({ providedIn: 'root' })
export class TileMetadataService {
  private readonly storageKey = 'core.tileMetadata.v1';
  private readonly connectionsKey = 'core.tileConnections.v1';

  private metadata = new Map<number, TileWorldMetadata>();
  private connections: WorldConnection[] = [];

  private selectedTileMetadata$ = new BehaviorSubject<TileWorldMetadata | null>(null);
  private connections$ = new BehaviorSubject<WorldConnection[]>([]);
  private allMetadata$ = new BehaviorSubject<Map<number, TileWorldMetadata>>(new Map());

  constructor() {
    this.loadFromStorage();
  }

  // ─────────────────────────────────────────────────────────────
  // Observables
  // ─────────────────────────────────────────────────────────────

  onSelectedMetadataChanged(): Observable<TileWorldMetadata | null> {
    return this.selectedTileMetadata$.asObservable();
  }

  onConnectionsChanged(): Observable<WorldConnection[]> {
    return this.connections$.asObservable();
  }

  onAllMetadataChanged(): Observable<Map<number, TileWorldMetadata>> {
    return this.allMetadata$.asObservable();
  }

  // ─────────────────────────────────────────────────────────────
  // Core CRUD
  // ─────────────────────────────────────────────────────────────

  getMetadata(tileIndex: number): TileWorldMetadata | undefined {
    return this.metadata.get(tileIndex);
  }

  getOrCreateMetadata(tileIndex: number): TileWorldMetadata {
    let meta = this.metadata.get(tileIndex);
    if (!meta) {
      meta = this.createEmptyMetadata(tileIndex);
      this.metadata.set(tileIndex, meta);
      this.notifyChanges();
    }
    return meta;
  }

  updateMetadata(tileIndex: number, partial: Partial<TileWorldMetadata>): TileWorldMetadata {
    const existing = this.getOrCreateMetadata(tileIndex);
    const updated: TileWorldMetadata = {
      ...existing,
      ...partial,
      tileIndex,
      updatedAt: new Date().toISOString(),
      version: existing.version + 1
    };
    this.metadata.set(tileIndex, updated);
    this.notifyChanges();
    this.saveToStorage();
    return updated;
  }

  setSelectedTile(tileIndex: number | null): void {
    if (tileIndex === null) {
      this.selectedTileMetadata$.next(null);
    } else {
      const meta = this.getOrCreateMetadata(tileIndex);
      this.selectedTileMetadata$.next(meta);
    }
  }

  // ─────────────────────────────────────────────────────────────
  // World Identity
  // ─────────────────────────────────────────────────────────────

  setWorldName(tileIndex: number, name: string): void {
    this.updateMetadata(tileIndex, { name });
  }

  setWorldDescription(tileIndex: number, description: string): void {
    this.updateMetadata(tileIndex, { description });
  }

  setWorldTags(tileIndex: number, tags: string[]): void {
    this.updateMetadata(tileIndex, { tags });
  }

  addTag(tileIndex: number, tag: string): void {
    const meta = this.getOrCreateMetadata(tileIndex);
    const tags = [...(meta.tags || [])];
    if (!tags.includes(tag)) {
      tags.push(tag);
      this.updateMetadata(tileIndex, { tags });
    }
  }

  removeTag(tileIndex: number, tag: string): void {
    const meta = this.getOrCreateMetadata(tileIndex);
    const tags = (meta.tags || []).filter(t => t !== tag);
    this.updateMetadata(tileIndex, { tags });
  }

  setCustomColor(tileIndex: number, color: string | undefined): void {
    this.updateMetadata(tileIndex, { customColor: color });
  }

  // ─────────────────────────────────────────────────────────────
  // Quick Notes
  // ─────────────────────────────────────────────────────────────

  addQuickNote(tileIndex: number, content: string, source: 'human' | 'ai' = 'human', instanceName?: string): QuickNote {
    const meta = this.getOrCreateMetadata(tileIndex);
    const note: QuickNote = {
      id: crypto.randomUUID(),
      content,
      createdAt: new Date().toISOString(),
      source,
      instanceName
    };
    const notes = [...(meta.quickNotes || []), note];
    this.updateMetadata(tileIndex, { quickNotes: notes });
    return note;
  }

  removeQuickNote(tileIndex: number, noteId: string): void {
    const meta = this.getOrCreateMetadata(tileIndex);
    const notes = (meta.quickNotes || []).filter(n => n.id !== noteId);
    this.updateMetadata(tileIndex, { quickNotes: notes });
  }

  // ─────────────────────────────────────────────────────────────
  // Pinned Items (Images, Links, References)
  // ─────────────────────────────────────────────────────────────

  addPinnedItem(tileIndex: number, item: Omit<PinnedItem, 'id' | 'createdAt'>): PinnedItem {
    const meta = this.getOrCreateMetadata(tileIndex);
    const pinned: PinnedItem = {
      ...item,
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString()
    };
    const items = [...(meta.pinnedItems || []), pinned];
    this.updateMetadata(tileIndex, { pinnedItems: items });
    return pinned;
  }

  removePinnedItem(tileIndex: number, itemId: string): void {
    const meta = this.getOrCreateMetadata(tileIndex);
    const items = (meta.pinnedItems || []).filter(i => i.id !== itemId);
    this.updateMetadata(tileIndex, { pinnedItems: items });
  }

  addPinnedImage(tileIndex: number, imageData: string, title?: string): PinnedItem {
    return this.addPinnedItem(tileIndex, {
      type: 'image',
      imageData,
      title
    });
  }

  addPinnedLink(tileIndex: number, url: string, title?: string, description?: string): PinnedItem {
    return this.addPinnedItem(tileIndex, {
      type: 'link',
      url,
      title,
      description
    });
  }

  // ─────────────────────────────────────────────────────────────
  // AI Observations
  // ─────────────────────────────────────────────────────────────

  addAIObservation(tileIndex: number, instanceName: string, observation: string, confidence?: number, tags?: string[]): AIObservation {
    const meta = this.getOrCreateMetadata(tileIndex);
    const obs: AIObservation = {
      id: crypto.randomUUID(),
      instanceName,
      observation,
      createdAt: new Date().toISOString(),
      confidence,
      tags
    };
    const observations = [...(meta.aiObservations || []), obs];
    this.updateMetadata(tileIndex, { aiObservations: observations });
    return obs;
  }

  removeAIObservation(tileIndex: number, observationId: string): void {
    const meta = this.getOrCreateMetadata(tileIndex);
    const observations = (meta.aiObservations || []).filter(o => o.id !== observationId);
    this.updateMetadata(tileIndex, { aiObservations: observations });
  }

  // ─────────────────────────────────────────────────────────────
  // Content Links (Wiki, Boards)
  // ─────────────────────────────────────────────────────────────

  linkWikiPage(tileIndex: number, wikiPageId: string): void {
    const meta = this.getOrCreateMetadata(tileIndex);
    const ids = [...(meta.wikiPageIds || [])];
    if (!ids.includes(wikiPageId)) {
      ids.push(wikiPageId);
      this.updateMetadata(tileIndex, { wikiPageIds: ids });
    }
  }

  unlinkWikiPage(tileIndex: number, wikiPageId: string): void {
    const meta = this.getOrCreateMetadata(tileIndex);
    const ids = (meta.wikiPageIds || []).filter(id => id !== wikiPageId);
    this.updateMetadata(tileIndex, { wikiPageIds: ids });
  }

  linkBoard(tileIndex: number, boardId: string): void {
    const meta = this.getOrCreateMetadata(tileIndex);
    const ids = [...(meta.boardIds || [])];
    if (!ids.includes(boardId)) {
      ids.push(boardId);
      this.updateMetadata(tileIndex, { boardIds: ids });
    }
  }

  unlinkBoard(tileIndex: number, boardId: string): void {
    const meta = this.getOrCreateMetadata(tileIndex);
    const ids = (meta.boardIds || []).filter(id => id !== boardId);
    this.updateMetadata(tileIndex, { boardIds: ids });
  }

  // ─────────────────────────────────────────────────────────────
  // World Connections
  // ─────────────────────────────────────────────────────────────

  getConnections(): WorldConnection[] {
    return [...this.connections];
  }

  getConnectionsForTile(tileIndex: number): WorldConnection[] {
    return this.connections.filter(
      c => c.fromTileIndex === tileIndex || c.toTileIndex === tileIndex
    );
  }

  addConnection(
    fromTileIndex: number,
    toTileIndex: number,
    type: ConnectionType,
    bidirectional = true,
    label?: string,
    notes?: string
  ): WorldConnection {
    // Check if connection already exists
    const existing = this.connections.find(
      c => (c.fromTileIndex === fromTileIndex && c.toTileIndex === toTileIndex) ||
           (bidirectional && c.fromTileIndex === toTileIndex && c.toTileIndex === fromTileIndex)
    );
    if (existing) {
      return this.updateConnection(existing.id, { type, bidirectional, label, notes });
    }

    const connection: WorldConnection = {
      id: crypto.randomUUID(),
      fromTileIndex,
      toTileIndex,
      type,
      bidirectional,
      label,
      notes,
      createdAt: new Date().toISOString()
    };
    this.connections.push(connection);

    // Add connection ID to both tiles' metadata
    const fromMeta = this.getOrCreateMetadata(fromTileIndex);
    const toMeta = this.getOrCreateMetadata(toTileIndex);
    this.updateMetadata(fromTileIndex, { connectionIds: [...(fromMeta.connectionIds || []), connection.id] });
    this.updateMetadata(toTileIndex, { connectionIds: [...(toMeta.connectionIds || []), connection.id] });

    this.connections$.next([...this.connections]);
    this.saveConnectionsToStorage();
    return connection;
  }

  updateConnection(connectionId: string, partial: Partial<WorldConnection>): WorldConnection {
    const index = this.connections.findIndex(c => c.id === connectionId);
    if (index === -1) throw new Error(`Connection ${connectionId} not found`);

    this.connections[index] = { ...this.connections[index], ...partial };
    this.connections$.next([...this.connections]);
    this.saveConnectionsToStorage();
    return this.connections[index];
  }

  removeConnection(connectionId: string): void {
    const connection = this.connections.find(c => c.id === connectionId);
    if (!connection) return;

    // Remove connection ID from both tiles' metadata
    const fromMeta = this.getMetadata(connection.fromTileIndex);
    const toMeta = this.getMetadata(connection.toTileIndex);
    if (fromMeta) {
      this.updateMetadata(connection.fromTileIndex, {
        connectionIds: (fromMeta.connectionIds || []).filter(id => id !== connectionId)
      });
    }
    if (toMeta) {
      this.updateMetadata(connection.toTileIndex, {
        connectionIds: (toMeta.connectionIds || []).filter(id => id !== connectionId)
      });
    }

    this.connections = this.connections.filter(c => c.id !== connectionId);
    this.connections$.next([...this.connections]);
    this.saveConnectionsToStorage();
  }

  // ─────────────────────────────────────────────────────────────
  // Search & Filter
  // ─────────────────────────────────────────────────────────────

  searchTiles(query: string): TileWorldMetadata[] {
    const q = query.toLowerCase();
    const results: TileWorldMetadata[] = [];

    this.metadata.forEach((meta) => {
      const nameMatch = meta.name?.toLowerCase().includes(q);
      const descMatch = meta.description?.toLowerCase().includes(q);
      const tagMatch = meta.tags?.some(t => t.toLowerCase().includes(q));
      const noteMatch = meta.quickNotes?.some(n => n.content.toLowerCase().includes(q));

      if (nameMatch || descMatch || tagMatch || noteMatch) {
        results.push(meta);
      }
    });

    return results;
  }

  filterByTag(tag: string): TileWorldMetadata[] {
    const results: TileWorldMetadata[] = [];
    this.metadata.forEach((meta) => {
      if (meta.tags?.includes(tag)) {
        results.push(meta);
      }
    });
    return results;
  }

  filterByConnectionType(type: ConnectionType): WorldConnection[] {
    return this.connections.filter(c => c.type === type);
  }

  getAllTags(): string[] {
    const tags = new Set<string>();
    this.metadata.forEach((meta) => {
      meta.tags?.forEach(t => tags.add(t));
    });
    return Array.from(tags).sort();
  }

  getTilesWithContent(): TileWorldMetadata[] {
    const results: TileWorldMetadata[] = [];
    this.metadata.forEach((meta) => {
      const hasContent = meta.name || meta.description ||
        (meta.quickNotes && meta.quickNotes.length > 0) ||
        (meta.pinnedItems && meta.pinnedItems.length > 0) ||
        (meta.wikiPageIds && meta.wikiPageIds.length > 0);
      if (hasContent) {
        results.push(meta);
      }
    });
    return results;
  }

  // ─────────────────────────────────────────────────────────────
  // Persistence
  // ─────────────────────────────────────────────────────────────

  private createEmptyMetadata(tileIndex: number): TileWorldMetadata {
    return {
      tileIndex,
      tags: [],
      quickNotes: [],
      pinnedItems: [],
      aiObservations: [],
      wikiPageIds: [],
      boardIds: [],
      connectionIds: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      version: 1
    };
  }

  private notifyChanges(): void {
    this.allMetadata$.next(new Map(this.metadata));
  }

  private loadFromStorage(): void {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (raw) {
        const data = JSON.parse(raw) as Record<string, TileWorldMetadata>;
        this.metadata = new Map(Object.entries(data).map(([k, v]) => [parseInt(k, 10), v]));
      }
    } catch (e) {
      console.error('Failed to load tile metadata:', e);
    }

    try {
      const raw = localStorage.getItem(this.connectionsKey);
      if (raw) {
        this.connections = JSON.parse(raw) as WorldConnection[];
      }
    } catch (e) {
      console.error('Failed to load tile connections:', e);
    }

    this.notifyChanges();
    this.connections$.next([...this.connections]);
  }

  private saveToStorage(): void {
    try {
      const data: Record<number, TileWorldMetadata> = {};
      this.metadata.forEach((v, k) => { data[k] = v; });
      localStorage.setItem(this.storageKey, JSON.stringify(data));
    } catch (e) {
      console.error('Failed to save tile metadata:', e);
    }
  }

  private saveConnectionsToStorage(): void {
    try {
      localStorage.setItem(this.connectionsKey, JSON.stringify(this.connections));
    } catch (e) {
      console.error('Failed to save tile connections:', e);
    }
  }

  // Full snapshot for project save/load
  createSnapshot(projectId: string): TileMetadataSnapshot {
    const data: Record<number, TileWorldMetadata> = {};
    this.metadata.forEach((v, k) => { data[k] = v; });
    return {
      projectId,
      metadata: data,
      connections: [...this.connections],
      version: '1.0',
      savedAt: new Date().toISOString()
    };
  }

  restoreSnapshot(snapshot: TileMetadataSnapshot): void {
    if (snapshot.metadata instanceof Map) {
      this.metadata = new Map(snapshot.metadata);
    } else {
      this.metadata = new Map(Object.entries(snapshot.metadata).map(([k, v]) => [parseInt(k, 10), v]));
    }
    this.connections = snapshot.connections || [];
    this.notifyChanges();
    this.connections$.next([...this.connections]);
    this.saveToStorage();
    this.saveConnectionsToStorage();
  }

  clear(): void {
    this.metadata.clear();
    this.connections = [];
    this.notifyChanges();
    this.connections$.next([]);
    this.saveToStorage();
    this.saveConnectionsToStorage();
  }
}
