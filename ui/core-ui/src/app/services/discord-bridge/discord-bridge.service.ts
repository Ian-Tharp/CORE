import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import {
  BehaviorSubject,
  Observable,
  Subscription,
  catchError,
  combineLatest,
  interval,
  map,
  of,
  shareReplay,
  startWith,
  switchMap,
  tap,
} from 'rxjs';
import { AppConfigService } from '../config/app-config.service';

export interface DiscordBridgeStatus {
  status: string;
  connected: boolean;
  connected_at: string | null;
  last_error: string | null;
  reconnect_attempts: number;
  bot_user: string | null;
  guilds: number;
  channel_mappings: number;
  bridged_core_channels: string[];
}

export interface DiscordChannelMapping {
  discord_channel_id: string;
  discord_channel_name?: string | null;
  discord_guild_id?: string | null;
  discord_guild_name?: string | null;
  core_channel_id: string;
  core_channel_name?: string | null;
  require_mention: boolean;
  enabled: boolean;
}

export interface DiscordChannelMappingsResponse {
  mappings: DiscordChannelMapping[];
  count: number;
}

export interface DiscordMessageLink {
  id?: number;
  core_message_id: string;
  core_channel_id: string;
  discord_message_id: string;
  discord_channel_id: string;
  discord_guild_id?: string | null;
  discord_author_id?: string | null;
  direction: 'discord_to_core' | 'core_to_discord';
  chunk_index: number;
  total_chunks: number;
  metadata?: Record<string, unknown> | null;
  created_at?: string;
}

export interface DiscordMessageLinksResponse {
  links: DiscordMessageLink[];
  count: number;
  filters: Record<string, unknown>;
}

export interface DiscordDeliveryEvent {
  id?: number;
  event_id: string;
  status: 'success' | 'failed' | 'skipped';
  direction: 'discord_to_core' | 'core_to_discord';
  core_message_id?: string | null;
  core_channel_id?: string | null;
  discord_message_id?: string | null;
  discord_channel_id?: string | null;
  discord_guild_id?: string | null;
  error?: string | null;
  metadata?: Record<string, unknown> | null;
  created_at?: string;
}

export interface DiscordDeliveryEventsResponse {
  events: DiscordDeliveryEvent[];
  count: number;
  filters: Record<string, unknown>;
}

export interface DiscordBridgeMetrics {
  status: DiscordBridgeStatus;
  mappings_count: number;
  message_links_count: number;
  message_links_by_direction: Record<string, number>;
  delivery_events_count: number;
  delivery_events_by_status: Record<string, number>;
  delivery_events_by_direction: Record<string, number>;
  recent_failures: DiscordDeliveryEvent[];
}

export type DiscordBridgeIndicatorTone = 'ready' | 'pending' | 'attention' | 'unknown';

export interface DiscordBridgeIndicator {
  label: string;
  tone: DiscordBridgeIndicatorTone;
  detail: string;
  tooltip: string;
  status: DiscordBridgeStatus | null;
}

@Injectable({ providedIn: 'root' })
export class DiscordBridgeService {
  private readonly apiUrl: string;
  private readonly _statusSubject = new BehaviorSubject<DiscordBridgeStatus | null>(null);
  private readonly _statusUnavailableSubject = new BehaviorSubject<boolean>(false);
  private _statusPollingSubscription: Subscription | null = null;

  public readonly status$ = this._statusSubject.asObservable();
  public readonly indicator$ = combineLatest([
    this.status$,
    this._statusUnavailableSubject.asObservable(),
  ]).pipe(
    map(([status, unavailable]) => this._buildIndicator(status, unavailable)),
    shareReplay({ bufferSize: 1, refCount: true })
  );

  constructor(
    private readonly http: HttpClient,
    private readonly config: AppConfigService
  ) {
    this.apiUrl = this.config.discordBridgeUrl;
  }

  public getStatus(): Observable<DiscordBridgeStatus> {
    return this.http.get<DiscordBridgeStatus>(`${this.apiUrl}/status`);
  }

  public startStatusPolling(intervalMs: number = 30000): void {
    if (this._statusPollingSubscription) {
      return;
    }

    this._statusPollingSubscription = interval(intervalMs)
      .pipe(
        startWith(0),
        switchMap(() => this.getStatus().pipe(
          tap((status) => {
            this._statusSubject.next(status);
            this._statusUnavailableSubject.next(false);
          }),
          catchError((error) => {
            console.warn('Failed to refresh Discord gateway status', error);
            this._statusSubject.next(null);
            this._statusUnavailableSubject.next(true);
            return of(null);
          })
        ))
      )
      .subscribe();
  }

  public stopStatusPolling(): void {
    this._statusPollingSubscription?.unsubscribe();
    this._statusPollingSubscription = null;
  }

  public getMetrics(recentFailuresLimit: number = 10): Observable<DiscordBridgeMetrics> {
    const params = new HttpParams().set('recent_failures_limit', recentFailuresLimit);
    return this.http.get<DiscordBridgeMetrics>(`${this.apiUrl}/metrics`, { params });
  }

  public getMappings(): Observable<DiscordChannelMappingsResponse> {
    return this.http.get<DiscordChannelMappingsResponse>(`${this.apiUrl}/channels`);
  }

  public getMessageLinks(filters: {
    limit?: number;
    core_message_id?: string;
    core_channel_id?: string;
    discord_channel_id?: string;
    direction?: 'discord_to_core' | 'core_to_discord';
  } = {}): Observable<DiscordMessageLinksResponse> {
    let params = new HttpParams().set('limit', String(filters.limit ?? 25));

    if (filters.core_message_id) {
      params = params.set('core_message_id', filters.core_message_id);
    }
    if (filters.core_channel_id) {
      params = params.set('core_channel_id', filters.core_channel_id);
    }
    if (filters.discord_channel_id) {
      params = params.set('discord_channel_id', filters.discord_channel_id);
    }
    if (filters.direction) {
      params = params.set('direction', filters.direction);
    }

    return this.http.get<DiscordMessageLinksResponse>(`${this.apiUrl}/message-links`, { params });
  }

  public getDeliveries(filters: {
    limit?: number;
    status?: 'success' | 'failed' | 'skipped';
    direction?: 'discord_to_core' | 'core_to_discord';
    core_channel_id?: string;
    discord_channel_id?: string;
    core_message_id?: string;
  } = {}): Observable<DiscordDeliveryEventsResponse> {
    let params = new HttpParams().set('limit', String(filters.limit ?? 25));

    if (filters.status) {
      params = params.set('status', filters.status);
    }
    if (filters.direction) {
      params = params.set('direction', filters.direction);
    }
    if (filters.core_channel_id) {
      params = params.set('core_channel_id', filters.core_channel_id);
    }
    if (filters.discord_channel_id) {
      params = params.set('discord_channel_id', filters.discord_channel_id);
    }
    if (filters.core_message_id) {
      params = params.set('core_message_id', filters.core_message_id);
    }

    return this.http.get<DiscordDeliveryEventsResponse>(`${this.apiUrl}/deliveries`, { params });
  }

  private _buildIndicator(
    status: DiscordBridgeStatus | null,
    unavailable: boolean
  ): DiscordBridgeIndicator {
    if (unavailable || !status) {
      return {
        label: 'Unknown',
        tone: 'unknown',
        detail: 'Live gateway status is unavailable.',
        tooltip: 'Discord Gateway status is currently unavailable.',
        status,
      };
    }

    if (status.connected) {
      return {
        label: 'Connected',
        tone: 'ready',
        detail: `${status.channel_mappings} mapped channel(s) ready`,
        tooltip: `Discord Gateway connected as ${status.bot_user ?? 'unknown bot'} across ${status.guilds} guild(s).`,
        status,
      };
    }

    if (status.status.toLowerCase() === 'connecting' || status.reconnect_attempts > 0) {
      return {
        label: 'Reconnecting',
        tone: 'pending',
        detail: `Reconnect attempts: ${status.reconnect_attempts}`,
        tooltip: status.last_error
          ? `Discord Gateway is reconnecting. Last error: ${status.last_error}`
          : 'Discord Gateway is reconnecting.',
        status,
      };
    }

    return {
      label: 'Disconnected',
      tone: 'attention',
      detail: status.last_error || 'Bridge is not connected to Discord.',
      tooltip: status.last_error
        ? `Discord Gateway disconnected. Last error: ${status.last_error}`
        : 'Discord Gateway is disconnected.',
      status,
    };
  }
}
