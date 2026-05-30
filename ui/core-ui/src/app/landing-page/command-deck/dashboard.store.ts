import { Injectable, OnDestroy, computed, inject, signal } from '@angular/core';
import { Subscription } from 'rxjs';
import { AppConfigService } from '../../services/config/app-config.service';
import {
  SystemMonitorService,
  SystemResources
} from '../../services/system-monitor/system-monitor.service';

/** Connection state of the live resource feed. */
export type FeedStatus = 'connecting' | 'live' | 'polling' | 'offline';

/** A single radial vital derived from the resource stream. */
export interface Vital {
  readonly id: 'cpu' | 'memory' | 'storage' | 'network';
  readonly label: string;
  readonly value: number;
  readonly detail: string;
}

const EMPTY_RESOURCES: SystemResources = {
  cpu_usage: 0,
  memory_usage: 0,
  memory_total_gb: 0,
  memory_available_gb: 0,
  storage_usage: 0,
  storage_total_gb: 0,
  storage_available_gb: 0,
  network_io_rate_mbps: 0,
  network_sent_gb: 0,
  network_recv_gb: 0,
  processes_count: 0
};

/**
 * Reactive store for the command deck. Streams system resources over SSE
 * (`/system/resources/stream`) and exposes them as signals. Falls back to
 * HTTP polling when `EventSource` is unavailable (e.g. SSR/test) or the
 * stream errors, so the UI stays live without busy-polling in the happy path.
 */
@Injectable({ providedIn: 'root' })
export class DashboardStore implements OnDestroy {
  private readonly _config = inject(AppConfigService);
  private readonly _monitor = inject(SystemMonitorService);

  private readonly _resources = signal<SystemResources>(EMPTY_RESOURCES);
  private readonly _feedStatus = signal<FeedStatus>('connecting');

  private _eventSource: EventSource | null = null;
  private _pollSub: Subscription | null = null;
  private _started = false;

  /** Latest raw resource snapshot. */
  public readonly resources = this._resources.asReadonly();

  /** Connection state of the live feed. */
  public readonly feedStatus = this._feedStatus.asReadonly();

  /** Resources projected into display-ready radial vitals. */
  public readonly vitals = computed<Vital[]>(() => {
    const r = this._resources();
    const usedMem = Math.max(0, r.memory_total_gb - r.memory_available_gb);
    const usedDisk = Math.max(0, r.storage_total_gb - r.storage_available_gb);
    const network = this._monitor.getNetworkActivityPercentage(r.network_io_rate_mbps);

    return [
      {
        id: 'cpu',
        label: 'CPU',
        value: Math.round(r.cpu_usage),
        detail: `${r.processes_count} processes`
      },
      {
        id: 'memory',
        label: 'Memory',
        value: Math.round(r.memory_usage),
        detail: `${usedMem.toFixed(1)} / ${r.memory_total_gb.toFixed(1)} GB`
      },
      {
        id: 'storage',
        label: 'Storage',
        value: Math.round(r.storage_usage),
        detail: `${usedDisk.toFixed(0)} / ${r.storage_total_gb.toFixed(0)} GB`
      },
      {
        id: 'network',
        label: 'Network',
        value: Math.round(network),
        detail: `${r.network_recv_gb.toFixed(1)} GB in`
      }
    ];
  });

  /** Begin streaming. Safe to call repeatedly; only the first call connects. */
  public start(): void {
    if (this._started) {
      return;
    }
    this._started = true;

    if (typeof EventSource === 'undefined') {
      this._startPolling();
      return;
    }

    this._connectStream();
  }

  /** Tear down the active feed. */
  public stop(): void {
    this._eventSource?.close();
    this._eventSource = null;
    this._pollSub?.unsubscribe();
    this._pollSub = null;
    this._started = false;
  }

  public ngOnDestroy(): void {
    this.stop();
  }

  private _connectStream(): void {
    this._feedStatus.set('connecting');
    const url = `${this._config.apiBaseUrl}/system/resources/stream`;
    const source = new EventSource(url);

    source.onmessage = (event: MessageEvent<string>) => {
      try {
        const data = JSON.parse(event.data) as SystemResources;
        this._resources.set(data);
        this._feedStatus.set('live');
      } catch {
        // Ignore malformed frames; the next frame will recover.
      }
    };

    source.onerror = () => {
      source.close();
      this._eventSource = null;
      this._startPolling();
    };

    this._eventSource = source;
  }

  private _startPolling(): void {
    if (this._pollSub) {
      return;
    }
    this._feedStatus.set('polling');
    this._pollSub = this._monitor.getSystemResourcesPolling(15).subscribe({
      next: (resources) => {
        this._resources.set(resources);
        this._feedStatus.set('polling');
      },
      error: () => this._feedStatus.set('offline')
    });
  }
}
