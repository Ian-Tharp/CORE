import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map, of, switchMap, catchError } from 'rxjs';
import { WorldGenParams } from '../../landing-page/command-center/engine/planet';

export interface HexWorldConfigDto {
  radius: number;
  gridWidth: number;
  gridHeight: number;
  elevation: number;
}

export interface LayersDto {
  terrain: Array<{ index: number; state: string }>;
  biome: Array<{ index: number; state: string }>;
  resources: Array<{ index: number; state: string }>;
}

export interface WorldRecord {
  id: string;
  name: string;
  origin?: string;
  created_at?: string;
  updated_at?: string;
}

export interface WorldAsset {
  id: string;
  tile_index?: number | null;
  kind: string;
  title?: string | null;
  image_b64: string;
  created_at: string;
}

export interface WorldAgentAuditResult {
  approved: boolean;
  confidence: number;
  contradictions: string[];
  missing_details: string[];
  suggestions: string[];
}

export interface WorldAgentLoreResponse {
  title: string;
  content: string;
  generated_by: string;
  audit: WorldAgentAuditResult;
}

export interface WorldAgentLoreSaveResponse {
  id: string;
  title: string;
}

export interface WorldAgentAuditResponse {
  generated_by: string;
  audit: WorldAgentAuditResult;
}

export interface WorldConnectionSuggestion {
  from_tile_index: number;
  to_tile_index: number;
  type: 'trade' | 'conflict' | 'alliance' | 'portal' | 'influence' | 'mystery';
  label: string;
  rationale: string;
  confidence: number;
}

export interface WorldAgentConnectionsResponse {
  generated_by: string;
  suggestions: WorldConnectionSuggestion[];
}

export interface WorldAgentImagePromptResponse {
  generated_by: string;
  prompt: string;
  palette: string[];
  constraints: string[];
}

export interface WorldAssetSaveRequest {
  image_b64: string;
  kind?: string;
  title?: string;
  tile_index?: number;
}

export interface WorldKnowledgeSearchResponse {
  results: Array<{ text: string; document_id: string; distance: number }>;
  doc_ids: string[];
  world_id: string;
}

@Injectable({ providedIn: 'root' })
export class WorldsService {
  private readonly apiUrl = 'http://localhost:8001';

  constructor(private readonly http: HttpClient) {}

  createWorld(name: string, config: HexWorldConfigDto, origin: 'human' | 'ai' = 'human', tags: string[] = []): Observable<{ id: string; name: string }> {
    return this.http.post<{ id: string; name: string }>(`${this.apiUrl}/worlds`, { name, config, origin, tags });
  }

  createSnapshot(
    worldId: string,
    payload: {
      config: HexWorldConfigDto;
      layers?: LayersDto;
      tiles?: Array<{ index: number; state: string }>;
      preview?: string;
    }
  ): Observable<{ id: string; world_id: string }> {
    return this.http.post<{ id: string; world_id: string }>(`${this.apiUrl}/worlds/${worldId}/snapshots`, payload);
  }

  getLatestSnapshot(worldId: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/worlds/${worldId}/snapshots/latest`);
  }

  /** Save a world's tile-metadata snapshot (names, tags, notes, connections, links). */
  saveMetadata(worldId: string, snapshot: any): Observable<{ status: string }> {
    return this.http.put<{ status: string }>(`${this.apiUrl}/worlds/${worldId}/metadata`, { snapshot });
  }

  /** Fetch a world's tile-metadata snapshot (for restore). */
  getMetadata(worldId: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/worlds/${worldId}/metadata`);
  }

  /** Persist a generated image (world art) immediately to the DB. */
  saveAsset(worldId: string, payload: WorldAssetSaveRequest): Observable<{ id: string }> {
    return this.http.post<{ id: string }>(`${this.apiUrl}/worlds/${worldId}/assets`, payload);
  }

  /** List a world's assets (optionally one tile's) — reloads art from the DB. */
  listAssets(worldId: string, tileIndex?: number): Observable<WorldAsset[]> {
    const params: any = tileIndex != null ? { tile_index: tileIndex } : {};
    return this.http.get<WorldAsset[]>(`${this.apiUrl}/worlds/${worldId}/assets`, { params });
  }

  deleteAsset(worldId: string, assetId: string): Observable<{ status: string }> {
    return this.http.delete<{ status: string }>(`${this.apiUrl}/worlds/${worldId}/assets/${assetId}`);
  }

  /** Ingest this world's wiki pages into the knowledgebase (world-scoped, RAG). */
  ingestWiki(worldId: string): Observable<{ ingested: number; skipped: number; total: number }> {
    return this.http.post<{ ingested: number; skipped: number; total: number }>(
      `${this.apiUrl}/worlds/${worldId}/knowledge/ingest-wiki`, {}
    );
  }

  /** List the knowledge documents linked to this world. */
  listKnowledge(worldId: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/worlds/${worldId}/knowledge`);
  }

  /** Generate a schema-tagged wiki page (Overview/History/Peoples…) for a world. */
  generateLore(
    worldId: string,
    payload: { kind: string; focus: string; world_name?: string; context?: string }
  ): Observable<{ id: string; title: string; content: string }> {
    return this.http.post<{ id: string; title: string; content: string }>(
      `${this.apiUrl}/worlds/${worldId}/lore/generate`, payload
    );
  }

  /** Run modular world-agent lore workflow: generate wiki page plus audit result. */
  generateAgentLore(
    worldId: string,
    payload: {
      tile_index: number;
      kind: string;
      focus: string;
      world_name?: string;
      user_context?: string;
      agent_id?: string;
      model?: string;
    }
  ): Observable<WorldAgentLoreResponse> {
    return this.http.post<WorldAgentLoreResponse>(
      `${this.apiUrl}/worlds/${worldId}/agents/lore`,
      payload
    );
  }

  /** Persist a user-approved world-agent lore draft as a wiki page. */
  saveAgentLore(
    worldId: string,
    payload: {
      tile_index: number;
      title: string;
      content: string;
      generated_by?: string;
      audit?: WorldAgentAuditResult;
    }
  ): Observable<WorldAgentLoreSaveResponse> {
    return this.http.post<WorldAgentLoreSaveResponse>(
      `${this.apiUrl}/worlds/${worldId}/agents/lore/save`,
      payload
    );
  }

  /** Run canon audit over selected-world context or a provided draft. */
  auditWorldAgent(
    worldId: string,
    payload: {
      tile_index: number;
      content?: string;
      world_name?: string;
      user_context?: string;
    }
  ): Observable<WorldAgentAuditResponse> {
    return this.http.post<WorldAgentAuditResponse>(
      `${this.apiUrl}/worlds/${worldId}/agents/audit`,
      payload
    );
  }

  /** Suggest typed world connections for user approval. */
  suggestWorldConnections(
    worldId: string,
    payload: { tile_index: number; max_suggestions?: number; world_name?: string; user_context?: string }
  ): Observable<WorldAgentConnectionsResponse> {
    return this.http.post<WorldAgentConnectionsResponse>(
      `${this.apiUrl}/worlds/${worldId}/agents/connections`,
      payload
    );
  }

  /** Generate a bounded image prompt from selected-world context. */
  generateImagePrompt(
    worldId: string,
    payload: { tile_index: number; world_name?: string; user_context?: string; style?: string }
  ): Observable<WorldAgentImagePromptResponse> {
    return this.http.post<WorldAgentImagePromptResponse>(
      `${this.apiUrl}/worlds/${worldId}/agents/image-prompt`,
      payload
    );
  }

  /** Semantic search scoped to this world's knowledge (world-filtered RAG). */
  searchKnowledge(
    worldId: string, query: string, maxChunks = 5
  ): Observable<WorldKnowledgeSearchResponse> {
    return this.http.post<WorldKnowledgeSearchResponse>(
      `${this.apiUrl}/worlds/${worldId}/knowledge/search`, { query, max_chunks: maxChunks }
    );
  }

  listWorlds(limit = 10, offset = 0): Observable<Array<{ id: string; name: string; updated_at: string }>> {
    return this.http.get<Array<{ id: string; name: string; updated_at: string }>>(`${this.apiUrl}/worlds`, { params: { limit, offset } as any });
  }

  saveFromHexSnapshot(
    name: string,
    hexSnapshot: {
      config: HexWorldConfigDto | { cellRadius: number; gridWidth: number; gridHeight: number; elevation: number };
      layers?: LayersDto;
      preview?: string;
    }
  ): Observable<{ worldId: string; snapshotId: string }> {
    // Convert frontend TileGridConfig (cellRadius) to backend HexWorldConfigDto (radius)
    const config: HexWorldConfigDto = {
      radius: 'cellRadius' in hexSnapshot.config ? hexSnapshot.config.cellRadius : hexSnapshot.config.radius,
      gridWidth: hexSnapshot.config.gridWidth,
      gridHeight: hexSnapshot.config.gridHeight,
      elevation: hexSnapshot.config.elevation
    };
    return this.createWorld(name, config, 'human').pipe(
      switchMap((world) => this.createSnapshot(
        world.id,
        { config, layers: hexSnapshot.layers, preview: hexSnapshot.preview }
      ).pipe(
        map((snap) => ({ worldId: world.id, snapshotId: snap.id }))
      ))
    );
  }

  deleteWorld(worldId: string): Observable<{ status: string }> {
    return this.http.delete<{ status: string }>(`${this.apiUrl}/worlds/${worldId}`);
  }

  deleteSnapshot(worldId: string, snapshotId: string): Observable<{ status: string }> {
    return this.http.delete<{ status: string }>(`${this.apiUrl}/worlds/${worldId}/snapshots/${snapshotId}`);
  }

  /**
   * Find a world by its name (case-insensitive).
   * Returns null if no world with that name exists.
   */
  getWorldByName(name: string): Observable<WorldRecord | null> {
    return this.http.get<WorldRecord | null>(`${this.apiUrl}/worlds/by-name/${encodeURIComponent(name)}`).pipe(
      catchError(() => of(null))
    );
  }

  /**
   * Rename an existing world.
   */
  renameWorld(worldId: string, newName: string): Observable<{ status: string }> {
    return this.http.patch<{ status: string }>(`${this.apiUrl}/worlds/${worldId}`, { name: newName });
  }

  /**
   * Update an existing world by creating a new snapshot.
   * Optionally rename the world if the name has changed.
   */
  updateExistingWorld(
    worldId: string,
    name: string,
    hexSnapshot: {
      config: HexWorldConfigDto | { cellRadius: number; gridWidth: number; gridHeight: number; elevation: number };
      layers?: LayersDto;
      preview?: string;
    }
  ): Observable<{ worldId: string; snapshotId: string }> {
    const config: HexWorldConfigDto = {
      radius: 'cellRadius' in hexSnapshot.config ? hexSnapshot.config.cellRadius : hexSnapshot.config.radius,
      gridWidth: hexSnapshot.config.gridWidth,
      gridHeight: hexSnapshot.config.gridHeight,
      elevation: hexSnapshot.config.elevation
    };

    return this.createSnapshot(worldId, { config, layers: hexSnapshot.layers, preview: hexSnapshot.preview }).pipe(
      map((snap) => ({ worldId, snapshotId: snap.id }))
    );
  }

  /**
   * Smart save: Updates existing world if worldId is provided, otherwise creates new.
   * Returns the worldId and snapshotId.
   */
  saveWorld(
    name: string,
    hexSnapshot: {
      config: HexWorldConfigDto | { cellRadius: number; gridWidth: number; gridHeight: number; elevation: number };
      layers?: LayersDto;
      preview?: string;
    },
    existingWorldId?: string
  ): Observable<{ worldId: string; snapshotId: string; isNew: boolean }> {
    if (existingWorldId) {
      // Update existing world
      return this.updateExistingWorld(existingWorldId, name, hexSnapshot).pipe(
        map((result) => ({ ...result, isNew: false }))
      );
    } else {
      // Create new world
      return this.saveFromHexSnapshot(name, hexSnapshot).pipe(
        map((result) => ({ ...result, isNew: true }))
      );
    }
  }

  /** Authored procedural-planet params for one orb (null when none saved). */
  getWorldGenParams(worldId: string, tileIndex: number): Observable<WorldGenParams | null> {
    return this.http.get<Record<string, unknown>>(
      `${this.apiUrl}/worlds/${worldId}/gen-params`,
      { params: { tile_index: String(tileIndex) } as any }
    ).pipe(map((p) => (p && Object.keys(p).length ? (p as unknown as WorldGenParams) : null)));
  }

  /** Persist an orb's authored procedural-planet params. */
  saveWorldGenParams(worldId: string, tileIndex: number, params: WorldGenParams): Observable<{ status: string }> {
    return this.http.put<{ status: string }>(
      `${this.apiUrl}/worlds/${worldId}/gen-params`,
      { tile_index: tileIndex, params }
    );
  }
}


