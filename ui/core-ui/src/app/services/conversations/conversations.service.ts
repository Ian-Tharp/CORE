import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { AppConfigService } from '../config/app-config.service';

export interface ConversationSummary { id: string; title: string; messages: number; }

/** Backend list response: a paginated envelope (or, defensively, a bare array). */
type ConversationListResponse =
  | ConversationSummary[]
  | { conversations: ConversationSummary[]; total: number; page: number; page_size: number };

@Injectable({ providedIn: 'root' })
export class ConversationsService {
  private readonly _http = inject(HttpClient);
  private readonly _cfg = inject(AppConfigService);

  list(page?: number, pageSize?: number): Observable<ConversationSummary[]> {
    let params = new HttpParams();
    if (page) {params = params.set('page', String(page));}
    if (pageSize) {params = params.set('page_size', String(pageSize));}
    // The backend returns { conversations, total, page, page_size }; unwrap to the
    // array the UI iterates (tolerant of a bare array in case the shape changes).
    return this._http
      .get<ConversationListResponse>(`${this._cfg.conversationsUrl}/`, { params })
      .pipe(map((res) => (Array.isArray(res) ? res : res?.conversations ?? [])));
  }

  updateTitle(convId: string, title: string): Observable<void> {
    return this._http.patch<void>(`${this._cfg.conversationsUrl}/${convId}`, { title });
  }
}



