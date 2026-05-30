import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { CountUpDirective } from '../shared/directives/count-up.directive';

/** A headline cognition metric rendered as a KPI card. */
interface AnalyticKpi {
  readonly id: string;
  readonly label: string;
  readonly value: number;
  readonly unit?: string;
  readonly icon: string;
  readonly accent: 'energy' | 'amber' | 'teal' | 'success';
  /** Percent change vs. the previous window (signed). */
  readonly trend: number;
}

/** A single derived-insight row. */
interface InsightRow {
  readonly id: string;
  readonly icon: string;
  readonly text: string;
  readonly tone: 'info' | 'positive' | 'warning';
}

/**
 * Analytics surface — a cognition-telemetry dashboard styled to match the
 * command deck (solarpunk × LCARS). Values are representative sample telemetry
 * (flagged "Preview" in the header) until a live analytics feed is wired;
 * the layout/animation is production-ready and consumes only design tokens.
 */
@Component({
  selector: 'app-analytics-page',
  standalone: true,
  imports: [MatIconModule, CountUpDirective],
  templateUrl: './analytics-page.component.html',
  styleUrl: './analytics-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class AnalyticsPageComponent {
  /** Headline cognition KPIs (sample telemetry until the analytics feed lands). */
  public readonly kpis = signal<readonly AnalyticKpi[]>([
    { id: 'reasoning', label: 'Reasoning Cycles', value: 1284, icon: 'psychology', accent: 'energy', trend: 12 },
    { id: 'council', label: 'Council Sessions', value: 47, icon: 'groups', accent: 'teal', trend: 5 },
    { id: 'knowledge', label: 'Knowledge Nodes', value: 9320, icon: 'hub', accent: 'amber', trend: 3 },
    { id: 'confidence', label: 'Avg Confidence', value: 86, unit: '%', icon: 'verified', accent: 'success', trend: -2 }
  ]);

  /** 24-point cognitive-throughput series for the sparkline (sample). */
  public readonly throughput = signal<readonly number[]>([
    12, 18, 14, 22, 30, 26, 35, 41, 38, 47, 52, 44,
    58, 63, 55, 67, 72, 61, 78, 70, 84, 79, 91, 88
  ]);

  /** Series maximum, floored at 1 to avoid divide-by-zero. */
  public readonly throughputMax = computed<number>(() => Math.max(1, ...this.throughput()));

  /** Derived insights (sample). */
  public readonly insights = signal<readonly InsightRow[]>([
    { id: '1', icon: 'trending_up', text: 'Reasoning throughput up 12% over the last cycle window.', tone: 'positive' },
    { id: '2', icon: 'bolt', text: 'Orchestration is the current latency hotspot in the pipeline.', tone: 'warning' },
    { id: '3', icon: 'auto_awesome', text: 'Council synthesis confidence holding above 0.85.', tone: 'info' }
  ]);

  public readonly trackKpi = (_: number, k: AnalyticKpi): string => k.id;
  public readonly trackInsight = (_: number, r: InsightRow): string => r.id;

  /** Bar height (0-100%) normalized against the series maximum. */
  public barHeight(value: number): number {
    return Math.round((value / this.throughputMax()) * 100);
  }
}
