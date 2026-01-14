import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map, of, switchMap, catchError } from 'rxjs';

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

@Injectable({ providedIn: 'root' })
export class WorldsService {
  private readonly apiUrl = 'http://localhost:8001';

  constructor(private readonly http: HttpClient) {}

  createWorld(name: string, config: HexWorldConfigDto, origin: 'human' | 'ai' = 'human', tags: string[] = []): Observable<{ id: string; name: string }> {
    return this.http.post<{ id: string; name: string }>(`${this.apiUrl}/worlds`, { name, config, origin, tags });
  }

  createSnapshot(worldId: string, payload: { config: HexWorldConfigDto; layers?: LayersDto; tiles?: Array<{ index: number; state: string }>; preview?: string }): Observable<{ id: string; world_id: string }> {
    return this.http.post<{ id: string; world_id: string }>(`${this.apiUrl}/worlds/${worldId}/snapshots`, payload);
  }

  getLatestSnapshot(worldId: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/worlds/${worldId}/snapshots/latest`);
  }

  listWorlds(limit = 10, offset = 0): Observable<Array<{ id: string; name: string; updated_at: string }>> {
    return this.http.get<Array<{ id: string; name: string; updated_at: string }>>(`${this.apiUrl}/worlds`, { params: { limit, offset } as any });
  }

  saveFromHexSnapshot(name: string, hexSnapshot: { config: HexWorldConfigDto | { cellRadius: number; gridWidth: number; gridHeight: number; elevation: number }; layers?: LayersDto; preview?: string }): Observable<{ worldId: string; snapshotId: string }> {
    // Convert frontend TileGridConfig (cellRadius) to backend HexWorldConfigDto (radius)
    const config: HexWorldConfigDto = {
      radius: 'cellRadius' in hexSnapshot.config ? hexSnapshot.config.cellRadius : hexSnapshot.config.radius,
      gridWidth: hexSnapshot.config.gridWidth,
      gridHeight: hexSnapshot.config.gridHeight,
      elevation: hexSnapshot.config.elevation
    };
    return this.createWorld(name, config, 'human').pipe(
      switchMap((world) => this.createSnapshot(world.id, { config, layers: hexSnapshot.layers, preview: hexSnapshot.preview }).pipe(
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
    hexSnapshot: { config: HexWorldConfigDto | { cellRadius: number; gridWidth: number; gridHeight: number; elevation: number }; layers?: LayersDto; preview?: string }
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
    hexSnapshot: { config: HexWorldConfigDto | { cellRadius: number; gridWidth: number; gridHeight: number; elevation: number }; layers?: LayersDto; preview?: string },
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
}


