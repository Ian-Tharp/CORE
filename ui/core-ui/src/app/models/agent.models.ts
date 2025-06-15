export interface AgentCapability {
  name: string;
  description: string;
  icon: string;
  category: 'perception' | 'reasoning' | 'action' | 'learning' | 'integration' | 'analytics' | 'security';
}

export interface AgentPerformanceMetrics {
  memoryUsage: number;
  cpuUsage: number;
  responsiveness: number; // ms
  reliability: number; // percentage
  energyEfficiency: number; // 0-100 scale
}

export interface AgentDependency {
  name: string;
  version: string;
  type: 'container' | 'service' | 'hardware' | 'model';
  optional: boolean;
}

export interface AgentReview {
  userId: string;
  username: string;
  rating: number;
  comment: string;
  timestamp: Date;
  verifiedUser: boolean;
}

export interface MarketplaceAgent {
  id: string;
  name: string;
  displayName: string;
  version: string;
  author: string;
  description: string;
  longDescription: string;
  category: 'cognitive' | 'automation' | 'integration' | 'analytics' | 'security' | 'experimental';
  tags: string[];
  imageUrl: string;
  containerImage: string;
  size: number; // in MB
  downloads: number;
  rating: number;
  reviews: AgentReview[];
  capabilities: AgentCapability[];
  performanceMetrics: AgentPerformanceMetrics;
  dependencies: AgentDependency[];
  compatibility: {
    coreVersion: string;
    platforms: string[];
    architectures: string[];
  };
  pricing: {
    model: 'free' | 'freemium' | 'paid' | 'subscription';
    price?: number;
    currency?: string;
  };
  status: 'stable' | 'beta' | 'experimental' | 'deprecated';
  releaseDate: Date;
  lastUpdated: Date;
  documentation: string;
  sourceCodeUrl?: string;
  isOfflineCapable: boolean;
  privacyCompliant: boolean;
  energyEfficient: boolean;
}

export interface AgentFilter {
  categories?: string[];
  tags?: string[];
  priceModels?: string[];
  minRating?: number;
  offlineOnly?: boolean;
  status?: string[];
  searchQuery?: string;
}

export interface AgentSort {
  field: 'downloads' | 'rating' | 'name' | 'releaseDate' | 'size';
  direction: 'asc' | 'desc';
} 