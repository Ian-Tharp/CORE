import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { delay, map } from 'rxjs/operators';
import { LibraryAgent, LibraryFilter, LibrarySort } from '../models/agent.models';

@Injectable({ providedIn: 'root' })
export class AgentLibraryService {
  private agentsSubject = new BehaviorSubject<LibraryAgent[]>([]);
  public agents$ = this.agentsSubject.asObservable();

  constructor() {
    this._loadMockAgents();
  }

  public getAgents(filter?: LibraryFilter, sort?: LibrarySort): Observable<LibraryAgent[]> {
    return this.agents$.pipe(
      map(agents => {
        let filtered = [...agents];

        if (filter) {
          if (filter.categories?.length) {
            filtered = filtered.filter(a => filter.categories!.includes(a.category));
          }
          if (filter.tags?.length) {
            filtered = filtered.filter(a => a.tags.some(tag => filter.tags!.includes(tag)));
          }
          if (filter.minRating) {
            filtered = filtered.filter(a => a.rating >= filter.minRating!);
          }
          if (filter.offlineOnly) {
            filtered = filtered.filter(a => a.isOfflineCapable);
          }
          if (filter.searchQuery) {
            const q = filter.searchQuery.toLowerCase();
            filtered = filtered.filter(a =>
              a.displayName.toLowerCase().includes(q) ||
              a.description.toLowerCase().includes(q) ||
              a.tags.some(t => t.toLowerCase().includes(q))
            );
          }
          if (filter.favoritesOnly) {
            filtered = filtered.filter(a => a.favorite);
          }
          if (filter.enabledOnly) {
            filtered = filtered.filter(a => a.enabled);
          }
          if (filter.draftsOnly) {
            filtered = filtered.filter(a => a.draft);
          }
          if (filter.recentlyUsed) {
            const cutoff = new Date();
            cutoff.setDate(cutoff.getDate() - 7);
            filtered = filtered.filter(a => (a.lastUsed ?? new Date(0)) >= cutoff);
          }
        }

        if (sort) {
          filtered.sort((a, b) => {
            let cmp = 0;
            switch (sort.field) {
              case 'downloads':
                cmp = a.downloads - b.downloads; break;
              case 'rating':
                cmp = a.rating - b.rating; break;
              case 'name':
                cmp = a.displayName.localeCompare(b.displayName); break;
              case 'releaseDate':
                cmp = a.releaseDate.getTime() - b.releaseDate.getTime(); break;
              case 'size':
                cmp = a.size - b.size; break;
              case 'lastUsed':
                cmp = (a.lastUsed?.getTime() ?? 0) - (b.lastUsed?.getTime() ?? 0); break;
              case 'instances':
                cmp = a.instances - b.instances; break;
            }
            return sort.direction === 'asc' ? cmp : -cmp;
          });
        }

        return filtered;
      }),
      delay(250)
    );
  }

  public getAgentById(id: string): Observable<LibraryAgent | undefined> {
    return this.agents$.pipe(map(list => list.find(a => a.id === id)));
  }

  public toggleFavorite(id: string): void {
    const list = this.agentsSubject.getValue().map(a => a.id === id ? { ...a, favorite: !a.favorite } : a);
    this.agentsSubject.next(list);
  }

  public enableAgent(id: string): void {
    const list = this.agentsSubject.getValue().map(a => a.id === id ? { ...a, enabled: true } : a);
    this.agentsSubject.next(list);
  }

  public disableAgent(id: string): void {
    const list = this.agentsSubject.getValue().map(a => a.id === id ? { ...a, enabled: false } : a);
    this.agentsSubject.next(list);
  }

  public deleteAgent(id: string): Observable<{ success: boolean }> {
    const list = this.agentsSubject.getValue().filter(a => a.id !== id);
    this.agentsSubject.next(list);
    return of({ success: true }).pipe(delay(200));
  }

  public duplicateAgent(id: string): Observable<{ success: boolean; newId: string }> {
    const source = this.agentsSubject.getValue().find(a => a.id === id);
    if (!source) return of({ success: false, newId: '' });
    const newId = `${source.id}-copy-${Math.floor(Math.random() * 1000)}`;
    const copy: LibraryAgent = { ...source, id: newId, displayName: `${source.displayName} (Copy)`, draft: true, favorite: false, enabled: false, lastUsed: undefined };
    this.agentsSubject.next([copy, ...this.agentsSubject.getValue()]);
    return of({ success: true, newId }).pipe(delay(250));
  }

  public exportAgent(id: string): Observable<Blob> {
    const content = JSON.stringify({ id, exportedAt: new Date().toISOString() });
    return of(new Blob([content], { type: 'application/json' })).pipe(delay(150));
  }

  private _loadMockAgents(): void {
    const baseNow = new Date();
    const mock: LibraryAgent[] = [
      {
        id: 'eve-001',
        name: 'eve-learning-engine',
        displayName: 'E.V.E. - Emergent Vessel for Evolution',
        version: '2.4.0',
        author: 'CORE Team',
        description: 'Advanced self-play learning agent with emergent behavior capabilities',
        longDescription: 'E.V.E. represents the pinnacle of autonomous learning systems within the CORE ecosystem.',
        category: 'cognitive',
        tags: ['machine-learning', 'self-play', 'reinforcement-learning', 'autonomous'],
        imageUrl: '/assets/agents/eve-icon.svg',
        containerImage: 'core-registry/eve:2.4.0',
        size: 2048,
        downloads: 15420,
        rating: 4.8,
        reviews: [],
        capabilities: [],
        performanceMetrics: { memoryUsage: 512, cpuUsage: 35, responsiveness: 50, reliability: 98.5, energyEfficiency: 85 },
        dependencies: [],
        compatibility: { coreVersion: '1.5.0+', platforms: ['linux/amd64', 'linux/arm64'], architectures: ['x86_64', 'arm64'] },
        pricing: { model: 'free' },
        status: 'stable',
        releaseDate: new Date('2024-01-15'),
        lastUpdated: new Date('2024-03-20'),
        documentation: 'https://docs.core-platform.io/agents/eve',
        sourceCodeUrl: 'https://github.com/core-platform/eve',
        isOfflineCapable: true,
        privacyCompliant: true,
        energyEfficient: true,
        // Library fields
        installed: true,
        enabled: true,
        favorite: true,
        lastUsed: new Date(baseNow.getTime() - 6 * 60 * 60 * 1000),
        instances: 2,
        draft: false,
        envReady: true
      },
      {
        id: 'orbit-001',
        name: 'orbit-home-automation',
        displayName: 'ORBIT - Omnipresent Residential Integration',
        version: '3.1.0',
        author: 'CORE IoT Team',
        description: 'Smart home integration agent with Home Assistant connectivity',
        longDescription: 'ORBIT seamlessly connects your CORE system with Home Assistant.',
        category: 'integration',
        tags: ['home-automation', 'iot', 'home-assistant', 'smart-home'],
        imageUrl: '/assets/agents/orbit-icon.svg',
        containerImage: 'core-registry/orbit:3.1.0',
        size: 512,
        downloads: 12843,
        rating: 4.7,
        reviews: [],
        capabilities: [],
        performanceMetrics: { memoryUsage: 384, cpuUsage: 20, responsiveness: 25, reliability: 97.5, energyEfficiency: 88 },
        dependencies: [],
        compatibility: { coreVersion: '1.4.0+', platforms: ['linux/amd64', 'linux/arm64'], architectures: ['x86_64', 'arm64'] },
        pricing: { model: 'free' },
        status: 'stable',
        releaseDate: new Date('2024-01-20'),
        lastUpdated: new Date('2024-03-18'),
        documentation: 'https://docs.core-platform.io/agents/orbit',
        sourceCodeUrl: 'https://github.com/core-platform/orbit',
        isOfflineCapable: true,
        privacyCompliant: true,
        energyEfficient: true,
        installed: true,
        enabled: false,
        favorite: false,
        lastUsed: new Date(baseNow.getTime() - 3 * 24 * 60 * 60 * 1000),
        instances: 1,
        draft: false,
        envReady: true
      },
      {
        id: 'nexus-001',
        name: 'nexus-data-synthesizer',
        displayName: 'NEXUS - Neural Data Synthesis Engine',
        version: '1.0.0-beta',
        author: 'CORE Labs',
        description: 'Experimental data synthesis and augmentation agent',
        longDescription: 'NEXUS pushes the boundaries of synthetic data generation.',
        category: 'experimental',
        tags: ['data-synthesis', 'privacy', 'experimental', 'generative-ai'],
        imageUrl: '/assets/agents/nexus-icon.svg',
        containerImage: 'core-registry/nexus:1.0.0-beta',
        size: 1536,
        downloads: 3421,
        rating: 4.3,
        reviews: [],
        capabilities: [],
        performanceMetrics: { memoryUsage: 1024, cpuUsage: 60, responsiveness: 100, reliability: 92, energyEfficiency: 70 },
        dependencies: [],
        compatibility: { coreVersion: '1.5.0+', platforms: ['linux/amd64'], architectures: ['x86_64'] },
        pricing: { model: 'free' },
        status: 'beta',
        releaseDate: new Date('2024-03-01'),
        lastUpdated: new Date('2024-03-22'),
        documentation: 'https://docs.core-platform.io/agents/nexus',
        isOfflineCapable: true,
        privacyCompliant: true,
        energyEfficient: false,
        installed: true,
        enabled: true,
        favorite: false,
        lastUsed: new Date(baseNow.getTime() - 12 * 24 * 60 * 60 * 1000),
        instances: 3,
        draft: true,
        envReady: false
      }
    ];
    this.agentsSubject.next(mock);
  }
}


