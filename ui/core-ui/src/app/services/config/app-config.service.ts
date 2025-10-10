import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class AppConfigService {
  // RSI TODO: Load from environment files and allow runtime override via window.__CORE_CONFIG__.
  private readonly _apiBaseUrl = 'http://localhost:8001';

  public get apiBaseUrl(): string { return this._apiBaseUrl; }
  public get knowledgebaseUrl(): string { return `${this._apiBaseUrl}/knowledgebase`; }
  public get conversationsUrl(): string { return `${this._apiBaseUrl}/conversations`; }
  public get chatStreamUrl(): string { return `${this._apiBaseUrl}/chat/stream`; }
}



