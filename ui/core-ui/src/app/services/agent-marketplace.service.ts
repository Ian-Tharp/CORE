import { Injectable } from '@angular/core';
import { Observable, BehaviorSubject, of } from 'rxjs';
import { delay, map } from 'rxjs/operators';
import { MarketplaceAgent, AgentFilter, AgentSort } from '../models/agent.models';

@Injectable({
  providedIn: 'root'
})
export class AgentMarketplaceService {
  private agentsSubject = new BehaviorSubject<MarketplaceAgent[]>([]);
  public agents$ = this.agentsSubject.asObservable();

  constructor() {
    this.loadMockAgents();
  }

  private loadMockAgents(): void {
    const mockAgents: MarketplaceAgent[] = [
      {
        id: 'eve-001',
        name: 'eve-learning-engine',
        displayName: 'E.V.E. - Emergent Vessel for Evolution',
        version: '2.4.0',
        author: 'CORE Team',
        description: 'Advanced self-play learning agent with emergent behavior capabilities',
        longDescription: 'E.V.E. represents the pinnacle of autonomous learning systems within the CORE ecosystem. This agent employs advanced reinforcement learning techniques combined with self-play methodologies to continuously evolve its capabilities without external supervision.',
        category: 'cognitive',
        tags: ['machine-learning', 'self-play', 'reinforcement-learning', 'autonomous'],
        imageUrl: '/assets/agents/eve-icon.svg',
        containerImage: 'core-registry/eve:2.4.0',
        size: 2048,
        downloads: 15420,
        rating: 4.8,
        reviews: [],
        capabilities: [
          {
            name: 'Self-Play Learning',
            description: 'Continuously improves through adversarial self-play',
            icon: 'psychology',
            category: 'learning'
          },
          {
            name: 'Pattern Recognition',
            description: 'Advanced pattern detection across multiple data streams',
            icon: 'pattern',
            category: 'perception'
          },
          {
            name: 'Adaptive Reasoning',
            description: 'Dynamic reasoning that evolves with experience',
            icon: 'hub',
            category: 'reasoning'
          }
        ],
        performanceMetrics: {
          memoryUsage: 512,
          cpuUsage: 35,
          responsiveness: 50,
          reliability: 98.5,
          energyEfficiency: 85
        },
        dependencies: [
          {
            name: 'PyTorch',
            version: '2.0+',
            type: 'container',
            optional: false
          },
          {
            name: 'CUDA',
            version: '11.8+',
            type: 'hardware',
            optional: true
          }
        ],
        compatibility: {
          coreVersion: '1.5.0+',
          platforms: ['linux/amd64', 'linux/arm64'],
          architectures: ['x86_64', 'arm64']
        },
        pricing: {
          model: 'free'
        },
        status: 'stable',
        releaseDate: new Date('2024-01-15'),
        lastUpdated: new Date('2024-03-20'),
        documentation: 'https://docs.core-platform.io/agents/eve',
        sourceCodeUrl: 'https://github.com/core-platform/eve',
        isOfflineCapable: true,
        privacyCompliant: true,
        energyEfficient: true
      },
      {
        id: 'aegis-001',
        name: 'aegis-security-framework',
        displayName: 'AEGIS - Adaptive Governance & Security',
        version: '1.2.0',
        author: 'CORE Security Team',
        description: 'Comprehensive security and governance framework for agent orchestration',
        longDescription: 'AEGIS provides a robust security layer for the CORE ecosystem, implementing advanced threat detection, access control, and governance policies to ensure safe agent interactions.',
        category: 'security',
        tags: ['security', 'governance', 'threat-detection', 'access-control'],
        imageUrl: '/assets/agents/aegis-icon.svg',
        containerImage: 'core-registry/aegis:1.2.0',
        size: 768,
        downloads: 8932,
        rating: 4.9,
        reviews: [],
        capabilities: [
          {
            name: 'Threat Detection',
            description: 'Real-time anomaly detection and threat analysis',
            icon: 'security',
            category: 'perception'
          },
          {
            name: 'Access Control',
            description: 'Fine-grained permission management',
            icon: 'lock',
            category: 'action'
          },
          {
            name: 'Audit Logging',
            description: 'Comprehensive activity tracking and reporting',
            icon: 'receipt_long',
            category: 'integration'
          }
        ],
        performanceMetrics: {
          memoryUsage: 256,
          cpuUsage: 15,
          responsiveness: 10,
          reliability: 99.9,
          energyEfficiency: 92
        },
        dependencies: [
          {
            name: 'Redis',
            version: '7.0+',
            type: 'service',
            optional: false
          }
        ],
        compatibility: {
          coreVersion: '1.3.0+',
          platforms: ['linux/amd64', 'linux/arm64', 'darwin/amd64'],
          architectures: ['x86_64', 'arm64']
        },
        pricing: {
          model: 'free'
        },
        status: 'stable',
        releaseDate: new Date('2024-02-01'),
        lastUpdated: new Date('2024-03-15'),
        documentation: 'https://docs.core-platform.io/agents/aegis',
        isOfflineCapable: true,
        privacyCompliant: true,
        energyEfficient: true
      },
      {
        id: 'orbit-001',
        name: 'orbit-home-automation',
        displayName: 'ORBIT - Omnipresent Residential Integration',
        version: '3.1.0',
        author: 'CORE IoT Team',
        description: 'Smart home integration agent with Home Assistant connectivity',
        longDescription: 'ORBIT seamlessly connects your CORE system with Home Assistant and other smart home platforms, enabling intelligent automation and device orchestration through natural language commands.',
        category: 'integration',
        tags: ['home-automation', 'iot', 'home-assistant', 'smart-home'],
        imageUrl: '/assets/agents/orbit-icon.svg',
        containerImage: 'core-registry/orbit:3.1.0',
        size: 512,
        downloads: 12843,
        rating: 4.7,
        reviews: [],
        capabilities: [
          {
            name: 'Device Discovery',
            description: 'Automatic detection of smart home devices',
            icon: 'devices',
            category: 'perception'
          },
          {
            name: 'Scene Automation',
            description: 'Intelligent scene creation and management',
            icon: 'auto_awesome',
            category: 'action'
          },
          {
            name: 'Energy Optimization',
            description: 'Smart energy usage patterns and recommendations',
            icon: 'eco',
            category: 'analytics'
          }
        ],
        performanceMetrics: {
          memoryUsage: 384,
          cpuUsage: 20,
          responsiveness: 25,
          reliability: 97.5,
          energyEfficiency: 88
        },
        dependencies: [
          {
            name: 'Home Assistant',
            version: '2023.12+',
            type: 'service',
            optional: false
          },
          {
            name: 'MQTT Broker',
            version: '5.0+',
            type: 'service',
            optional: true
          }
        ],
        compatibility: {
          coreVersion: '1.4.0+',
          platforms: ['linux/amd64', 'linux/arm64'],
          architectures: ['x86_64', 'arm64']
        },
        pricing: {
          model: 'free'
        },
        status: 'stable',
        releaseDate: new Date('2024-01-20'),
        lastUpdated: new Date('2024-03-18'),
        documentation: 'https://docs.core-platform.io/agents/orbit',
        sourceCodeUrl: 'https://github.com/core-platform/orbit',
        isOfflineCapable: true,
        privacyCompliant: true,
        energyEfficient: true
      },
      {
        id: 'nexus-001',
        name: 'nexus-data-synthesizer',
        displayName: 'NEXUS - Neural Data Synthesis Engine',
        version: '1.0.0-beta',
        author: 'CORE Labs',
        description: 'Experimental data synthesis and augmentation agent',
        longDescription: 'NEXUS pushes the boundaries of synthetic data generation, creating realistic datasets for training other agents while maintaining privacy and data sovereignty.',
        category: 'experimental',
        tags: ['data-synthesis', 'privacy', 'experimental', 'generative-ai'],
        imageUrl: '/assets/agents/nexus-icon.svg',
        containerImage: 'core-registry/nexus:1.0.0-beta',
        size: 1536,
        downloads: 3421,
        rating: 4.3,
        reviews: [],
        capabilities: [
          {
            name: 'Synthetic Data Generation',
            description: 'Creates realistic synthetic datasets',
            icon: 'dataset',
            category: 'action'
          },
          {
            name: 'Privacy Preservation',
            description: 'Ensures generated data maintains privacy',
            icon: 'privacy_tip',
            category: 'security'
          }
        ],
        performanceMetrics: {
          memoryUsage: 1024,
          cpuUsage: 60,
          responsiveness: 100,
          reliability: 92,
          energyEfficiency: 70
        },
        dependencies: [
          {
            name: 'Stable Diffusion',
            version: '2.0+',
            type: 'model',
            optional: true
          }
        ],
        compatibility: {
          coreVersion: '1.5.0+',
          platforms: ['linux/amd64'],
          architectures: ['x86_64']
        },
        pricing: {
          model: 'free'
        },
        status: 'beta',
        releaseDate: new Date('2024-03-01'),
        lastUpdated: new Date('2024-03-22'),
        documentation: 'https://docs.core-platform.io/agents/nexus',
        isOfflineCapable: true,
        privacyCompliant: true,
        energyEfficient: false
      },
      {
        id: 'prometheus-001',
        name: 'prometheus-analytics',
        displayName: 'PROMETHEUS - Predictive Analytics Engine',
        version: '2.0.0',
        author: 'CORE Analytics Team',
        description: 'Advanced predictive analytics and forecasting agent',
        longDescription: 'PROMETHEUS brings powerful time-series analysis and predictive modeling capabilities to the CORE platform, enabling data-driven insights and automated decision-making.',
        category: 'analytics',
        tags: ['analytics', 'forecasting', 'time-series', 'machine-learning'],
        imageUrl: '/assets/agents/prometheus-icon.svg',
        containerImage: 'core-registry/prometheus:2.0.0',
        size: 896,
        downloads: 7654,
        rating: 4.6,
        reviews: [],
        capabilities: [
          {
            name: 'Time-Series Analysis',
            description: 'Advanced temporal pattern recognition',
            icon: 'timeline',
            category: 'analytics'
          },
          {
            name: 'Predictive Modeling',
            description: 'Future state prediction with confidence intervals',
            icon: 'trending_up',
            category: 'reasoning'
          },
          {
            name: 'Anomaly Detection',
            description: 'Real-time identification of unusual patterns',
            icon: 'warning',
            category: 'perception'
          }
        ],
        performanceMetrics: {
          memoryUsage: 768,
          cpuUsage: 45,
          responsiveness: 75,
          reliability: 96,
          energyEfficiency: 80
        },
        dependencies: [
          {
            name: 'InfluxDB',
            version: '2.0+',
            type: 'service',
            optional: true
          }
        ],
        compatibility: {
          coreVersion: '1.4.0+',
          platforms: ['linux/amd64', 'linux/arm64'],
          architectures: ['x86_64', 'arm64']
        },
        pricing: {
          model: 'freemium',
          price: 0
        },
        status: 'stable',
        releaseDate: new Date('2024-02-15'),
        lastUpdated: new Date('2024-03-10'),
        documentation: 'https://docs.core-platform.io/agents/prometheus',
        isOfflineCapable: true,
        privacyCompliant: true,
        energyEfficient: true
      }
    ];

    this.agentsSubject.next(mockAgents);
  }

  getAgents(filter?: AgentFilter, sort?: AgentSort): Observable<MarketplaceAgent[]> {
    return this.agents$.pipe(
      map(agents => {
        let filtered = [...agents];

        // Apply filters
        if (filter) {
          if (filter.categories?.length) {
            filtered = filtered.filter(a => filter.categories!.includes(a.category));
          }
          if (filter.tags?.length) {
            filtered = filtered.filter(a => 
              a.tags.some(tag => filter.tags!.includes(tag))
            );
          }
          if (filter.minRating) {
            filtered = filtered.filter(a => a.rating >= filter.minRating!);
          }
          if (filter.offlineOnly) {
            filtered = filtered.filter(a => a.isOfflineCapable);
          }
          if (filter.searchQuery) {
            const query = filter.searchQuery.toLowerCase();
            filtered = filtered.filter(a => 
              a.displayName.toLowerCase().includes(query) ||
              a.description.toLowerCase().includes(query) ||
              a.tags.some(tag => tag.toLowerCase().includes(query))
            );
          }
        }

        // Apply sorting
        if (sort) {
          filtered.sort((a, b) => {
            let comparison = 0;
            switch (sort.field) {
              case 'downloads':
                comparison = a.downloads - b.downloads;
                break;
              case 'rating':
                comparison = a.rating - b.rating;
                break;
              case 'name':
                comparison = a.displayName.localeCompare(b.displayName);
                break;
              case 'releaseDate':
                comparison = a.releaseDate.getTime() - b.releaseDate.getTime();
                break;
              case 'size':
                comparison = a.size - b.size;
                break;
            }
            return sort.direction === 'asc' ? comparison : -comparison;
          });
        }

        return filtered;
      }),
      delay(300) // Simulate network delay
    );
  }

  getAgentById(id: string): Observable<MarketplaceAgent | undefined> {
    return this.agents$.pipe(
      map(agents => agents.find(a => a.id === id))
    );
  }

  installAgent(agentId: string): Observable<{ success: boolean; message: string }> {
    // Simulate installation process
    return of({ 
      success: true, 
      message: 'Agent installation initiated. Check the system monitor for progress.' 
    }).pipe(delay(1000));
  }
} 