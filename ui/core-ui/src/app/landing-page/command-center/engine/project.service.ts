import { Injectable } from '@angular/core';
import type { TileGridConfig, TerrainState, BiomeState, ResourceState } from './tile-grid.service';

export interface TileGridSnapshot {
  id: string;
  name: string;
  createdAt: string; // ISO date
  config: TileGridConfig;
  // v2 schema
  layers?: {
    terrain: Array<{ index: number; state: TerrainState }>;
    biome: Array<{ index: number; state: BiomeState }>;
    resources: Array<{ index: number; state: ResourceState }>;
  };
  // legacy v1 schema - for backward compatibility with hex worlds
  tiles?: Array<{ index: number; state: 'empty' | 'life' | 'resource' }>;
}

// Keep HexWorld types for backward compatibility
export interface HexWorldConfig extends TileGridConfig {
  radius?: number; // Legacy field mapping to cellRadius
}

export interface HexWorldSnapshot extends Omit<TileGridSnapshot, 'config'> {
  config: HexWorldConfig;
}

@Injectable({ providedIn: 'root' })
export class ProjectService {
  private readonly storageKey = 'core.tileGrid.projects.v1';
  private readonly legacyStorageKey = 'core.hexWorld.projects.v1';

  private isTileGridSnapshot(value: unknown): value is TileGridSnapshot {
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

  // Convert legacy hex world config to tile grid config
  private convertLegacyConfig(config: any): TileGridConfig {
    if (config.cellRadius) return config; // Already converted
    return {
      cellRadius: config.radius || 1,
      gridWidth: config.gridWidth || 50,
      gridHeight: config.gridHeight || 50,
      elevation: config.elevation || 0.1
    };
  }

  save(snapshot: Omit<TileGridSnapshot, 'id' | 'createdAt'>): TileGridSnapshot {
    const record: TileGridSnapshot = {
      ...snapshot,
      id: (globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2)),
      createdAt: new Date().toISOString(),
      config: this.convertLegacyConfig(snapshot.config)
    };
    const all = this.list();
    all.push(record);
    localStorage.setItem(this.storageKey, JSON.stringify(all));
    return record;
  }

  list(): TileGridSnapshot[] {
    const newData = this.loadFromStorage(this.storageKey);
    const legacyData = this.loadFromStorage(this.legacyStorageKey);
    
    // Convert legacy hex world snapshots to tile grid format
    const convertedLegacy = legacyData.map(item => ({
      ...item,
      config: this.convertLegacyConfig(item.config)
    }));
    
    return [...newData, ...convertedLegacy];
  }

  private loadFromStorage(key: string): TileGridSnapshot[] {
    const raw = localStorage.getItem(key);
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((x) => this.isTileGridSnapshot(x)) as TileGridSnapshot[];
    } catch {
      return [];
    }
  }

  load(id: string): TileGridSnapshot | undefined {
    return this.list().find((p) => p.id === id);
  }

  delete(id: string): void {
    // Try to delete from both new and legacy storage
    const newData = this.loadFromStorage(this.storageKey);
    const filteredNew = newData.filter((p) => p.id !== id);
    localStorage.setItem(this.storageKey, JSON.stringify(filteredNew));

    const legacyData = this.loadFromStorage(this.legacyStorageKey);  
    const filteredLegacy = legacyData.filter((p) => p.id !== id);
    localStorage.setItem(this.legacyStorageKey, JSON.stringify(filteredLegacy));
  }
}


