import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface WikiPageDto { id: string; world_id?: string; title: string; content: string; metadata?: any; created_at: string; updated_at: string; }

@Injectable({ providedIn: 'root' })
export class CreativeService {
  private readonly apiUrl = 'http://localhost:8001';
  constructor(private readonly http: HttpClient) {}

  listWiki(worldId?: string): Observable<WikiPageDto[]> {
    const params: any = worldId ? { world_id: worldId } : {};
    return this.http.get<WikiPageDto[]>(`${this.apiUrl}/creative/wiki`, { params });
  }
  createWiki(payload: { world_id?: string; title: string; content: string; metadata?: any }): Observable<{ id: string }> {
    return this.http.post<{ id: string }>(`${this.apiUrl}/creative/wiki`, payload);
  }
  updateWiki(id: string, payload: { world_id?: string; title: string; content: string; metadata?: any }): Observable<{ status: string }> {
    return this.http.put<{ status: string }>(`${this.apiUrl}/creative/wiki/${id}`, payload);
  }

  createCharacter(payload: { world_id?: string; name: string; traits?: any }): Observable<{ id: string }> {
    return this.http.post<{ id: string }>(`${this.apiUrl}/creative/characters`, payload);
  }
  generateCharacterImage(characterId: string, prompt: string, size: string = '512x512'): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(`${this.apiUrl}/creative/characters/${characterId}/image`, { prompt, size });
  }
}


