import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of, shareReplay } from 'rxjs';
import { AppConfigService } from './config/app-config.service';

export interface AgentToolDto {
  name: string;
  mcp_server_id?: string;
}

@Injectable({ providedIn: 'root' })
export class AgentToolsService {
  private cache = new Map<string, Observable<AgentToolDto[]>>();
  private readonly apiBase: string;

  constructor(private http: HttpClient, private config: AppConfigService) {
    this.apiBase = `${this.config.apiBaseUrl}/agents`;
  }

  getTools(agentId: string): Observable<AgentToolDto[]> {
    const cached = this.cache.get(agentId);
    if (cached) return cached;
    const req = this.http
      .get<{ tools: AgentToolDto[] }>(`${this.apiBase}/${agentId}/tools`)
      .pipe(shareReplay(1));
    const out = new Observable<AgentToolDto[]>(observer => {
      const sub = req.subscribe({
        next: res => { observer.next(res.tools || []); observer.complete(); },
        error: err => { observer.error(err); }
      });
      return () => sub.unsubscribe();
    }).pipe(shareReplay(1));
    this.cache.set(agentId, out);
    return out;
  }

  refreshTools(agentId: string): Observable<AgentToolDto[]> {
    this.cache.delete(agentId);
    return this.getTools(agentId);
  }
}


