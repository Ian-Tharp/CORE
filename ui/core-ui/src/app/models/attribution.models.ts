export interface KnowledgeAttribution {
  id: string;
  knowledgeChunkId: string;
  knowledgeFileId: string;
  sourceInstanceId: string;
  sourceInstanceName: string;
  timestamp: Date;
  verificationStatus: AttributionVerificationStatus;
  attributionChain?: AttributionChain[];
  context?: string;
  confidence?: number;
}

export interface AttributionChain {
  instanceId: string;
  instanceName: string;
  timestamp: Date;
  role: 'originator' | 'modifier' | 'verifier' | 'curator';
}

export enum AttributionVerificationStatus {
  VERIFIED = 'verified',
  PENDING = 'pending',
  FAILED = 'failed',
  UNVERIFIED = 'unverified'
}

export interface AttributionFilter {
  instanceId?: string;
  dateRange?: {
    start: Date;
    end: Date;
  };
  verificationStatus?: AttributionVerificationStatus[];
  searchQuery?: string;
}

export interface AttributedKnowledge {
  id: string;
  chunkText: string;
  fileName: string;
  fileId: string;
  attribution: KnowledgeAttribution;
  metadata?: Record<string, any>;
}

export interface AttributionStats {
  totalAttributions: number;
  verifiedCount: number;
  pendingCount: number;
  failedCount: number;
  topContributingInstances: InstanceContribution[];
}

export interface InstanceContribution {
  instanceId: string;
  instanceName: string;
  contributionCount: number;
  lastContribution: Date;
}