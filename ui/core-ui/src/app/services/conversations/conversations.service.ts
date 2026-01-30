import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AppConfigService } from '../config/app-config.service';

export interface ConversationSummary { id: string; title: string; messages: number; }

@Injectable({ providedIn: 'root' })
export class ConversationsService {
  private readonly _http = inject(HttpClient);
  private readonly _cfg = inject(AppConfigService);

  list(page?: number, pageSize?: number): Observable<ConversationSummary[]> {
    let params = new HttpParams();
    if (page) params = params.set('page', String(page));
    if (pageSize) params = params.set('page_size', String(pageSize));
    return this._http.get<ConversationSummary[]>(`${this._cfg.conversationsUrl}/`, { params });
  }

  updateTitle(convId: string, title: string): Observable<void> {
    return this._http.patch<void>(`${this._cfg.conversationsUrl}/${convId}`, { title });
  }
}



