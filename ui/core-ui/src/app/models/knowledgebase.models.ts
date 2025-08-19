export interface KnowledgeFile {
  id: string;
  filename: string;
  originalName: string;
  title?: string;
  chunkCount?: number;
  embeddingModel?: string;
  embeddingDimensions?: number;
  size: number;
  mimeType: string;
  uploadDate: Date;
  lastModified: Date;
  userId?: string;
  isGlobal: boolean;
  description?: string;
  source: FileSource;
  status: FileStatus;
  processingProgress?: number;
  error?: string;
}

export interface VectorEmbedding {
  id: string;
  fileId: string;
  chunkIndex: number;
  chunkText: string;
  embedding: number[];
  embeddingModel: string;
  dimensions: number;
  createdAt: Date;
  metadata?: Record<string, any>;
}

export interface EmbeddingStats {
  fileId: string;
  totalChunks: number;
  totalTokens: number;
  avgChunkSize: number;
  embeddingModel: string;
  dimensions: number;
  processingTime: number;
  lastProcessed: Date;
  similarity?: number;
}

export interface FileTag {
  id: string;
  name: string;
  color: string;
  category?: string;
  isSystemTag?: boolean;
}

export interface FileMetadata {
  fileId: string;
  tags: FileTag[];
  customFields: Record<string, any>;
  agentCompatibility?: string[];
  ragEnabled: boolean;
  searchableFields: string[];
  relatedFiles?: string[];
  annotations?: FileAnnotation[];
}

export interface FileAnnotation {
  id: string;
  text: string;
  createdBy: string;
  createdAt: Date;
  type: 'note' | 'highlight' | 'insight' | 'connection';
}

export enum FileSource {
  USER_UPLOAD = 'user_upload',
  CONVERSATION_HISTORY = 'conversation_history',
  WORKFLOW_ARTIFACT = 'workflow_artifact',
  AGENT_GENERATED = 'agent_generated',
  SYSTEM_IMPORT = 'system_import',
  WEB_SCRAPE = 'web_scrape'
}

export enum FileStatus {
  UPLOADING = 'uploading',
  PROCESSING = 'processing',
  READY = 'ready',
  ERROR = 'error',
  ARCHIVED = 'archived'
}

export interface KnowledgebaseFilter {
  searchQuery?: string;
  tags?: string[];
  fileTypes?: string[];
  sources?: FileSource[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  status?: FileStatus[];
  isGlobal?: boolean;
  userId?: string;
}

export interface KnowledgebaseStats {
  totalFiles: number;
  totalSize: number;
  filesByType: Record<string, number>;
  filesBySource: Record<FileSource, number>;
  totalEmbeddings: number;
  processingQueue: number;
  recentActivity: ActivityLog[];
}

export interface ActivityLog {
  id: string;
  action: 'upload' | 'delete' | 'process' | 'tag' | 'annotate' | 'share';
  fileId: string;
  fileName: string;
  userId: string;
  timestamp: Date;
  details?: string;
}

export interface FileUploadRequest {
  file: File;
  tags?: string[];
  description?: string;
  isGlobal?: boolean;
  metadata?: Record<string, any>;
  processImmediately?: boolean;
} 