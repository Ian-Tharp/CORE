import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, of } from 'rxjs';
import { delay, map, tap } from 'rxjs/operators';
import { LibraryAgent, LibraryFilter, LibrarySort } from '../models/agent.models';
import { HttpClient, HttpParams } from '@angular/common/http';
import { AppConfigService } from './config/app-config.service';

@Injectable({ providedIn: 'root' })
export class AgentLibraryService {
  private agentsSubject = new BehaviorSubject<LibraryAgent[]>([]);
  public agents$ = this.agentsSubject.asObservable();
  private readonly apiUrl: string;

  constructor(private readonly http: HttpClient, private readonly config: AppConfigService) {
    this.apiUrl = `${this.config.apiBaseUrl}/agents`;
    // Initial load from backend
    this.refreshAgents().subscribe();
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
      delay(100)
    );
  }

  public getAgentById(id: string): Observable<LibraryAgent | undefined> {
    // Prefer local cache
    return this.agents$.pipe(map(list => list.find(a => a.id === id)));
  }

  public toggleFavorite(id: string): void {
    const list = this.agentsSubject.getValue().map(a => a.id === id ? { ...a, favorite: !a.favorite } : a);
    this.agentsSubject.next(list);
  }

  public enableAgent(id: string): void {
    this.http.post<{ message: string }>(`${this.apiUrl}/${id}/activate`, {}).subscribe({
      next: () => {
        const list = this.agentsSubject.getValue().map(a => a.id === id ? { ...a, enabled: true } : a);
        this.agentsSubject.next(list);
      }
    });
  }

  public disableAgent(id: string): void {
    this.http.post<{ message: string }>(`${this.apiUrl}/${id}/deactivate`, {}).subscribe({
      next: () => {
        const list = this.agentsSubject.getValue().map(a => a.id === id ? { ...a, enabled: false } : a);
        this.agentsSubject.next(list);
      }
    });
  }

  public deleteAgent(id: string): Observable<{ success: boolean }> {
    return this.http.delete<void>(`${this.apiUrl}/${id}`).pipe(
      tap(() => {
        const list = this.agentsSubject.getValue().filter(a => a.id !== id);
        this.agentsSubject.next(list);
      }),
      map(() => ({ success: true }))
    );
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

  private refreshAgents(): Observable<LibraryAgent[]> {
    const params = new HttpParams().set('page', 1).set('page_size', 200);
    return this.http.get<{ agents: any[] }>(`${this.apiUrl}`, { params }).pipe(
      map(res => (res.agents ?? []).map(a => this.mapBackendAgentToLibrary(a))),
      tap(mapped => this.agentsSubject.next(mapped))
    );
  }

  private mapBackendAgentToLibrary(a: any): LibraryAgent {
    // Minimal, safe mapping with sensible defaults for UI fields
    const displayName = a.display_name || a.agent_name || a.agent_id;
    const now = new Date();
    return {
      id: a.agent_id,
      name: a.agent_name || a.agent_id,
      displayName,
      version: a.version || '1.0.0',
      author: a.author || 'Unknown',
      description: a.description || '',
      longDescription: a.system_prompt || '',
      category: 'cognitive',
      tags: Array.isArray(a.interests) ? a.interests : [],
      imageUrl: a.avatar_url || '/assets/agents/default-agent.svg',
      containerImage: '',
      size: 0,
      downloads: 0,
      rating: 4.5,
      reviews: [],
      capabilities: [],
      performanceMetrics: { memoryUsage: 0, cpuUsage: 0, responsiveness: 0, reliability: 99, energyEfficiency: 80 },
      dependencies: [],
      compatibility: { coreVersion: '1.0.0+', platforms: [], architectures: [] },
      pricing: { model: 'free' },
      status: a.is_active ? 'stable' : 'deprecated',
      releaseDate: now,
      lastUpdated: now,
      documentation: '',
      sourceCodeUrl: '',
      isOfflineCapable: true,
      privacyCompliant: true,
      energyEfficient: true,
      installed: true,
      enabled: a.is_active === true,
      favorite: false,
      lastUsed: undefined,
      instances: 0,
      draft: false,
      envReady: true
    };
  }
}


