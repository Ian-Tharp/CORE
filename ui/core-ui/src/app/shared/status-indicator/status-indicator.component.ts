import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

/**
 * Status types for agents and tasks in CORE
 */
export type StatusType = 'completed' | 'awaiting_hitl' | 'pending' | 'error' | 'running';

/**
 * StatusIndicatorComponent
 * 
 * A unified component for displaying agent/task status with icons and colors.
 * Follows the Solarpunk design system.
 * 
 * @example
 * <app-status-indicator status="running"></app-status-indicator>
 * <app-status-indicator status="completed" [showLabel]="true"></app-status-indicator>
 * <app-status-indicator status="error" size="lg"></app-status-indicator>
 */
@Component({
  selector: 'app-status-indicator',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './status-indicator.component.html',
  styleUrl: './status-indicator.component.scss'
})
export class StatusIndicatorComponent {
  /**
   * The status to display
   */
  @Input() status: StatusType = 'pending';

  /**
   * Whether to show a text label alongside the icon
   */
  @Input() showLabel = false;

  /**
   * Size of the indicator: 'sm' | 'md' | 'lg'
   */
  @Input() size: 'sm' | 'md' | 'lg' = 'md';

  /**
   * Get CSS class for the current status
   */
  get statusClass(): string {
    return `status-${this.status}`;
  }

  /**
   * Get human-readable label for the status
   */
  get statusLabel(): string {
    const labels: Record<StatusType, string> = {
      completed: 'Completed',
      awaiting_hitl: 'Awaiting Input',
      pending: 'Pending',
      error: 'Error',
      running: 'Running'
    };
    return labels[this.status] || 'Unknown';
  }

  /**
   * Get SVG icon path data for the status
   */
  get iconPath(): string {
    const icons: Record<StatusType, string> = {
      // Checkmark for completed
      completed: 'M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z',
      // Hand/pause for awaiting human input
      awaiting_hitl: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z',
      // Clock for pending
      pending: 'M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67V7z',
      // X/error for error
      error: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z',
      // Play/running for running
      running: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z'
    };
    return icons[this.status] || icons.pending;
  }

  /**
   * Alternate icon for awaiting_hitl - shows a hand icon
   */
  get awaitingHitlIcon(): string {
    // Hand raised icon
    return 'M12.5 2c.83 0 1.5.67 1.5 1.5v7.34c.63.37 1.2.85 1.68 1.42l.49-.49c.29-.29.77-.29 1.06 0l.71.71c.29.29.29.77 0 1.06l-2.12 2.12c-.47.47-1.1.73-1.77.73H9c-1.1 0-2-.9-2-2v-3c0-.55.45-1 1-1s1 .45 1 1v2h.5V3.5c0-.83.67-1.5 1.5-1.5zm-5 8c-.83 0-1.5.67-1.5 1.5v5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5v-5c0-.83-.67-1.5-1.5-1.5z';
  }
}
