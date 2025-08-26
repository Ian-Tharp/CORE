import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface StepResponse {
  step: string;
  text: string;
  routing_decision?: string;
  plan?: string[];
  evaluation?: string;
}

export type StepStreamEvent =
  | { type: 'start'; step: string }
  | { type: 'chunk'; text: string }
  | { type: 'metrics'; duration_ms: number; ttfb_ms: number; tokens: number }
  | { type: 'end' };

@Injectable({ providedIn: 'root' })
export class EngineService {
  private readonly api = 'http://localhost:8001/core';

  constructor(private readonly http: HttpClient) {}

  entry(payload: { message_id: string; user_input: string; model?: string }): Observable<unknown> {
    return this.http.post(`${this.api}`, payload);
  }

  comprehension(payload: { message_id: string; user_input: string; model?: string }): Observable<StepResponse> {
    return this.http.post<StepResponse>(`${this.api}/comprehension`, payload);
  }

  orchestration(payload: {
    message_id: string;
    user_input: string;
    model?: string;
    comprehension_text?: string;
    comprehension_route?: string;
  }): Observable<StepResponse> {
    return this.http.post<StepResponse>(`${this.api}/orchestration`, payload);
  }

  reasoning(payload: {
    message_id: string;
    user_input: string;
    model?: string;
    comprehension_text?: string;
    orchestration_text?: string;
    orchestration_plan?: string[];
  }): Observable<StepResponse> {
    return this.http.post<StepResponse>(`${this.api}/reasoning`, payload);
  }

  evaluation(payload: {
    message_id: string;
    user_input: string;
    model?: string;
    comprehension_text?: string;
    orchestration_text?: string;
    orchestration_plan?: string[];
    reasoning_text?: string;
  }): Observable<StepResponse> {
    return this.http.post<StepResponse>(`${this.api}/evaluation`, payload);
  }

  private sse<T extends StepStreamEvent>(url: string, payload: unknown): Observable<T> {
    return new Observable<T>((observer) => {
      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
        .then((response) => {
          if (!response.ok || !response.body) {
            observer.error(`HTTP ${response.status}: ${response.statusText}`);
            return;
          }
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          const readChunk = () => {
            reader.read().then(({ value, done }) => {
              if (done) { observer.complete(); return; }
              const text = decoder.decode(value, { stream: true });
              for (const part of text.split('\n\n')) {
                const trimmed = part.trim();
                if (!trimmed.startsWith('data:')) continue;
                const jsonStr = trimmed.replace(/^data:\s*/, '');
                if (!jsonStr || jsonStr === '[DONE]') continue;
                observer.next(JSON.parse(jsonStr));
              }
              readChunk();
            });
          };
          readChunk();
        })
        .catch((err) => observer.error(err));
    });
  }

  comprehensionStream(payload: { message_id: string; user_input: string; model?: string }): Observable<StepStreamEvent> {
    return this.sse<StepStreamEvent>(`${this.api}/comprehension/stream`, payload);
  }

  orchestrationStream(payload: {
    message_id: string; user_input: string; model?: string;
    comprehension_text?: string; comprehension_route?: string;
  }): Observable<StepStreamEvent> {
    return this.sse<StepStreamEvent>(`${this.api}/orchestration/stream`, payload);
  }

  reasoningStream(payload: {
    message_id: string; user_input: string; model?: string;
    comprehension_text?: string; orchestration_text?: string; orchestration_plan?: string[];
  }): Observable<StepStreamEvent> {
    return this.sse<StepStreamEvent>(`${this.api}/reasoning/stream`, payload);
  }

  evaluationStream(payload: {
    message_id: string; user_input: string; model?: string;
    comprehension_text?: string; orchestration_text?: string; orchestration_plan?: string[]; reasoning_text?: string;
  }): Observable<StepStreamEvent> {
    return this.sse<StepStreamEvent>(`${this.api}/evaluation/stream`, payload);
  }
}


