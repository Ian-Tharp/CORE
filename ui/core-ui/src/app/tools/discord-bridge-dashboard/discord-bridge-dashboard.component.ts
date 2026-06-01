import { Component, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTabsModule } from '@angular/material/tabs';
import { Subject, forkJoin, interval } from 'rxjs';
import { startWith, takeUntil } from 'rxjs/operators';

import {
  DiscordBridgeMetrics,
  DiscordBridgeService,
  DiscordChannelMapping,
  DiscordDeliveryEvent,
  DiscordMessageLink
} from '../../services/discord-bridge/discord-bridge.service';

interface ValidationChecklistItem {
  label: string;
  status: 'ready' | 'attention' | 'pending';
  detail: string;
}

@Component({
  selector: 'app-discord-bridge-dashboard',
  imports: [
    CommonModule,
    MatButtonModule,
    MatCardModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatTabsModule
  ],
  templateUrl: './discord-bridge-dashboard.component.html',
  styleUrl: './discord-bridge-dashboard.component.scss'
})
export class DiscordBridgeDashboardComponent implements OnInit, OnDestroy {
  public loading = true;
  public error: string | null = null;
  public metrics: DiscordBridgeMetrics | null = null;
  public mappings: DiscordChannelMapping[] = [];
  public deliveries: DiscordDeliveryEvent[] = [];
  public messageLinks: DiscordMessageLink[] = [];
  public readonly refreshIntervalSeconds = 15;
  public lastUpdatedAt: Date | null = null;

  private readonly _destroy$ = new Subject<void>();

  constructor(private readonly discordBridgeService: DiscordBridgeService) {}

  public ngOnInit(): void {
    this._startPolling();
  }

  public ngOnDestroy(): void {
    this._destroy$.next();
    this._destroy$.complete();
  }

  public refresh(): void {
    this._loadDashboard(true);
  }

  public get validationChecklist(): ValidationChecklistItem[] {
    const metrics = this.metrics;

    return [
      {
        label: 'Bridge connection',
        status: metrics?.status.connected ? 'ready' : 'attention',
        detail: metrics?.status.connected
          ? `Connected as ${metrics.status.bot_user ?? 'unknown bot'}`
          : 'Bridge should be connected before validating Discord traffic.'
      },
      {
        label: 'Channel mapping',
        status: (metrics?.mappings_count ?? 0) > 0 ? 'ready' : 'attention',
        detail: (metrics?.mappings_count ?? 0) > 0
          ? `${metrics?.mappings_count ?? 0} channel mapping(s) configured`
          : 'At least one Discord channel mapping is required.'
      },
      {
        label: 'Live Discord traffic',
        status: (metrics?.message_links_count ?? 0) > 0 ? 'ready' : 'pending',
        detail: (metrics?.message_links_count ?? 0) > 0
          ? `${metrics?.message_links_count ?? 0} linked messages recorded`
          : 'Send a Discord message into a mapped channel to validate ingress.'
      },
      {
        label: 'Outbound reply path',
        status: (metrics?.message_links_by_direction['core_to_discord'] ?? 0) > 0 ? 'ready' : 'pending',
        detail: (metrics?.message_links_by_direction['core_to_discord'] ?? 0) > 0
          ? `${metrics?.message_links_by_direction['core_to_discord'] ?? 0} outbound link(s) recorded`
          : 'Send a CORE-side message or trigger an agent response to validate outbound delivery.'
      },
      {
        label: 'Recent failures',
        status: (metrics?.recent_failures.length ?? 0) === 0 ? 'ready' : 'attention',
        detail: (metrics?.recent_failures.length ?? 0) === 0
          ? 'No recent failed bridge events'
          : `${metrics?.recent_failures.length ?? 0} recent failure(s) need review`
      }
    ];
  }

  public get bridgeStatusTone(): string {
    if (this.metrics?.status.connected) {
      return 'ready';
    }
    return this.metrics?.status.status === 'connecting' ? 'pending' : 'attention';
  }

  public get bridgeStatusLabel(): string {
    if (this.metrics?.status.connected) {
      return 'Connected';
    }

    return this.metrics?.status.status === 'connecting' ? 'Reconnecting' : 'Disconnected';
  }

  public get bridgeHeroTitle(): string {
    if (this.metrics?.status.connected) {
      return 'Gateway is live and ready to route traffic.';
    }

    if (this.metrics?.status.status === 'connecting') {
      return 'Gateway is attempting to recover connectivity.';
    }

    return 'Gateway needs attention before operators rely on it.';
  }

  public get bridgeHeroSummary(): string {
    if (!this.metrics) {
      return 'Waiting for the first gateway telemetry update.';
    }

    if (this.metrics.status.connected) {
      return `Connected as ${this.metrics.status.bot_user ?? 'unknown bot'} across ${this.metrics.status.guilds} guild(s).`;
    }

    if (this.metrics.status.last_error) {
      return this.metrics.status.last_error;
    }

    return 'The Discord bridge is not currently connected to the upstream gateway.';
  }

  public get connectedAtLabel(): string {
    if (!this.metrics?.status.connected_at) {
      return 'Not connected';
    }

    return new Date(this.metrics.status.connected_at).toLocaleString();
  }

  public get lastUpdatedLabel(): string {
    if (!this.lastUpdatedAt) {
      return 'Waiting for first sync';
    }

    return this.lastUpdatedAt.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  }

  public get deliverySuccessCount(): number {
    return this.metrics?.delivery_events_by_status['success'] ?? 0;
  }

  public get deliveryFailedCount(): number {
    return this.metrics?.delivery_events_by_status['failed'] ?? 0;
  }

  public get deliverySkippedCount(): number {
    return this.metrics?.delivery_events_by_status['skipped'] ?? 0;
  }

  public get recentFailurePreview(): DiscordDeliveryEvent[] {
    return (this.metrics?.recent_failures ?? []).slice(0, 3);
  }

  public get operatorAlert(): string | null {
    if (!this.metrics) {
      return null;
    }

    if (this.metrics.status.last_error) {
      return this.metrics.status.last_error;
    }

    if (!this.metrics.status.connected) {
      return 'Bridge is offline. Confirm token, gateway session, and network reachability before validating traffic.';
    }

    if (this.deliveryFailedCount > 0) {
      return `${this.deliveryFailedCount} failed delivery event(s) were recorded in the current window.`;
    }

    if (this.metrics.mappings_count === 0) {
      return 'No Discord channel mappings are configured yet.';
    }

    return null;
  }

  public trackByChannelMapping(_index: number, mapping: DiscordChannelMapping): string {
    return mapping.discord_channel_id;
  }

  public trackByChecklistItem(_index: number, item: ValidationChecklistItem): string {
    return item.label;
  }

  public trackByDelivery(_index: number, delivery: DiscordDeliveryEvent): string {
    return delivery.event_id;
  }

  public trackByMessageLink(_index: number, link: DiscordMessageLink): string {
    return `${link.discord_channel_id}-${link.discord_message_id}`;
  }

  private _startPolling(): void {
    interval(this.refreshIntervalSeconds * 1000)
      .pipe(
        startWith(0),
        takeUntil(this._destroy$)
      )
      .subscribe(() => {
        this._loadDashboard(false);
      });
  }

  private _loadDashboard(forceSpinner: boolean): void {
    if (forceSpinner) {
      this.loading = true;
    }

    this.error = null;

    forkJoin({
      metrics: this.discordBridgeService.getMetrics(),
      mappings: this.discordBridgeService.getMappings(),
      deliveries: this.discordBridgeService.getDeliveries({ limit: 20 }),
      messageLinks: this.discordBridgeService.getMessageLinks({ limit: 20 })
    })
      .pipe(takeUntil(this._destroy$))
      .subscribe({
        next: (data) => {
          this.metrics = data.metrics;
          this.mappings = data.mappings.mappings;
          this.deliveries = data.deliveries.events;
          this.messageLinks = data.messageLinks.links;
          this.lastUpdatedAt = new Date();
          this.loading = false;
        },
        error: (error) => {
          console.error('Failed to load Discord bridge dashboard', error);
          this.error = 'Failed to load Discord bridge dashboard data.';
          this.loading = false;
        }
      });
  }
}
