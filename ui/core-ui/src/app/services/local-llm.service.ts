import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { AppConfigService } from './config/app-config.service';

export interface LocalLlmModel {
  name: string;
}

export interface LocalLlmModelsResponse {
  provider: string;
  models: LocalLlmModel[];
}

/**
 * Consumes the backend local-LLM endpoints so the UI never bakes in a fixed
 * model list. Provider/model ids are resolved at runtime from the backend,
 * letting the same build serve an Ollama desktop or an LM Studio laptop.
 */
@Injectable({ providedIn: 'root' })
export class LocalLlmService {
  private http = inject(HttpClient);
  private cfg = inject(AppConfigService);

  private get base(): string {
    return this.cfg.apiBaseUrl;
  }

  /**
   * Fetch the live local-model catalogue. On failure, fall back to a single
   * neutral placeholder so dependent dropdowns still render something usable.
   */
  getModels(): Observable<LocalLlmModelsResponse> {
    return this.http
      .get<LocalLlmModelsResponse>(`${this.base}/local-llm/models`)
      .pipe(
        catchError(() =>
          of({ provider: 'lmstudio', models: [{ name: 'local-model' }] })
        )
      );
  }
}
