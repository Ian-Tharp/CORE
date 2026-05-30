import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

/** Health states the reactor can render, mapped from backend health. */
export type ReactorStatus = 'healthy' | 'degraded' | 'unhealthy' | 'initializing';

/**
 * The command-deck centerpiece: an animated SVG "arc reactor" whose energy,
 * colour, and pulse cadence reflect overall system health. Purely presentational
 * and driven entirely by signal inputs.
 */
@Component({
  selector: 'app-reactor-core',
  standalone: true,
  templateUrl: './reactor-core.component.html',
  styleUrl: './reactor-core.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ReactorCoreComponent {
  /** Overall system status driving the reactor visuals. */
  public readonly status = input<ReactorStatus>('initializing');

  /** Whether the live feed is currently streaming (drives the "online" ticker). */
  public readonly online = input<boolean>(false);

  /** Aggregate system load 0-100 used to scale the energy core. */
  public readonly load = input<number>(0);

  /** Human-readable status label rendered in the core. */
  public readonly label = computed<string>(() => {
    switch (this.status()) {
      case 'healthy':
        return 'ONLINE';
      case 'degraded':
        return 'DEGRADED';
      case 'unhealthy':
        return 'CRITICAL';
      default:
        return 'BOOTING';
    }
  });

  /** Energy ring fill fraction (0-1), floored so the reactor never looks dead. */
  public readonly energy = computed<number>(() => {
    const normalized = Math.min(100, Math.max(0, this.load())) / 100;
    return 0.12 + normalized * 0.88;
  });

  /** Strokes the energy arc proportionally to current load. */
  public readonly energyDashOffset = computed<number>(() => {
    const circumference = 2 * Math.PI * 78;
    return circumference * (1 - this.energy());
  });

  public readonly circumference = 2 * Math.PI * 78;
}
