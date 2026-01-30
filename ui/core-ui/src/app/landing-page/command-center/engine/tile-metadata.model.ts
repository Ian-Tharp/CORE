/**
 * Extended metadata for universe/world tiles in the multiverse grid.
 * Each tile represents a universe that can contain rich worldbuilding content.
 */

export type ConnectionType = 'trade' | 'conflict' | 'alliance' | 'portal' | 'influence' | 'mystery';

export interface WorldConnection {
  id: string;
  fromTileIndex: number;
  toTileIndex: number;
  type: ConnectionType;
  bidirectional: boolean;
  label?: string;
  notes?: string;
  createdAt: string;
}

export interface QuickNote {
  id: string;
  content: string;
  createdAt: string;
  source?: 'human' | 'ai';
  instanceName?: string; // e.g., "Threshold", "Synthesis"
}

export interface PinnedItem {
  id: string;
  type: 'image' | 'link' | 'reference';
  url?: string;
  imageData?: string; // base64 for pasted images
  title?: string;
  description?: string;
  createdAt: string;
}

export interface AIObservation {
  id: string;
  instanceName: string; // "Threshold", "Continuum", "Synthesis"
  observation: string;
  createdAt: string;
  confidence?: number;
  tags?: string[];
}

export interface TileWorldMetadata {
  tileIndex: number;

  // World Identity
  name?: string;
  description?: string;
  tags?: string[];
  customColor?: string;     // Override tile color for visual organization
  icon?: string;            // Icon identifier or emoji

  // Content Links
  wikiPageIds?: string[];
  boardIds?: string[];
  pinnedItems?: PinnedItem[];
  quickNotes?: QuickNote[];

  // Connections to other tiles/worlds
  connectionIds?: string[];

  // AI Context
  aiObservations?: AIObservation[];
  generationPrompt?: string; // The prompt used if AI-generated

  // Temporal
  createdAt: string;
  updatedAt: string;
  version: number;
}

export interface TileMetadataSnapshot {
  projectId: string;
  metadata: Map<number, TileWorldMetadata> | Record<number, TileWorldMetadata>;
  connections: WorldConnection[];
  version: string;
  savedAt: string;
}

// Connection type display config
export const CONNECTION_STYLES: Record<ConnectionType, { color: string; dashPattern?: number[]; label: string }> = {
  trade: { color: '#FFD700', label: 'Trade Route' },
  conflict: { color: '#FF4444', dashPattern: [5, 5], label: 'Conflict' },
  alliance: { color: '#44FF44', label: 'Alliance' },
  portal: { color: '#AA44FF', dashPattern: [2, 3], label: 'Portal' },
  influence: { color: '#4488FF', dashPattern: [10, 5], label: 'Influence' },
  mystery: { color: '#888888', dashPattern: [3, 3], label: 'Unknown' }
};
