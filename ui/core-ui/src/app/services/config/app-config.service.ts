import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class AppConfigService {
  // RSI TODO: Load from environment files and allow runtime override via window.__CORE_CONFIG__.
  private readonly _apiBaseUrl = 'http://localhost:8001';
  private readonly _defaultApiKey = 'core_dev_key';

  public get apiBaseUrl(): string { return this._apiBaseUrl; }
  public get apiKey(): string {
    return this._readLocalStorageValue('core_api_key') ?? this._defaultApiKey;
  }
  public get knowledgebaseUrl(): string { return `${this._apiBaseUrl}/knowledgebase`; }
  public get conversationsUrl(): string { return `${this._apiBaseUrl}/conversations`; }
  public get chatStreamUrl(): string { return `${this._apiBaseUrl}/chat/stream`; }
  public get discordBridgeUrl(): string { return `${this._apiBaseUrl}/discord`; }
  public isCoreApiUrl(url: string): boolean { return url.startsWith(this._apiBaseUrl); }

  private _readLocalStorageValue(key: string): string | null {
    if (typeof window === 'undefined' || !window.localStorage) {
      return null;
    }

    const value = window.localStorage.getItem(key)?.trim();
    return value ? value : null;
  }
}



