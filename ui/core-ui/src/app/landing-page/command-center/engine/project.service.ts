import { Injectable } from '@angular/core';
import type { HexWorldConfig, TerrainState, BiomeState, ResourceState } from './hex-world.service';

export interface HexWorldSnapshot {
  id: string;
  name: string;
  createdAt: string; // ISO date
  config: HexWorldConfig;
  // v2 schema
  layers?: {
    terrain: Array<{ index: number; state: TerrainState }>;
    biome: Array<{ index: number; state: BiomeState }>;
    resources: Array<{ index: number; state: ResourceState }>;
  };
  // legacy v1 schema
  tiles?: Array<{ index: number; state: 'empty' | 'life' | 'resource' }>;
}

@Injectable({ providedIn: 'root' })
export class ProjectService {
  private readonly storageKey = 'core.hexWorld.projects.v1';

  private isHexWorldSnapshot(value: unknown): value is HexWorldSnapshot {
    if (typeof value !== 'object' || value === null) return false;
    const v = value as Record<string, unknown>;
    return (
      typeof v['id'] === 'string' &&
      typeof v['name'] === 'string' &&
      typeof v['createdAt'] === 'string' &&
      typeof v['config'] === 'object' && v['config'] !== null &&
      (Array.isArray(v['tiles']) || typeof v['layers'] === 'object')
    );
  }

  save(snapshot: Omit<HexWorldSnapshot, 'id' | 'createdAt'>): HexWorldSnapshot {
    const record: HexWorldSnapshot = {
      ...snapshot,
      id: (globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2)),
      createdAt: new Date().toISOString(),
    };
    const all = this.list();
    all.push(record);
    localStorage.setItem(this.storageKey, JSON.stringify(all));
    return record;
  }

  list(): HexWorldSnapshot[] {
    const raw = localStorage.getItem(this.storageKey);
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((x) => this.isHexWorldSnapshot(x)) as HexWorldSnapshot[];
    } catch {
      return [];
    }
  }

  load(id: string): HexWorldSnapshot | undefined {
    return this.list().find((p) => p.id === id);
  }

  delete(id: string): void {
    const next = this.list().filter((p) => p.id !== id);
    localStorage.setItem(this.storageKey, JSON.stringify(next));
  }
}


