import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map, switchMap } from 'rxjs';

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

  saveFromHexSnapshot(name: string, hexSnapshot: { config: HexWorldConfigDto; layers?: LayersDto; preview?: string }): Observable<{ worldId: string; snapshotId: string }> {
    return this.createWorld(name, hexSnapshot.config, 'human').pipe(
      switchMap((world) => this.createSnapshot(world.id, { config: hexSnapshot.config, layers: hexSnapshot.layers, preview: hexSnapshot.preview }).pipe(
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
}


