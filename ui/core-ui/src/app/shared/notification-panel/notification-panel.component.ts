import { Component, Input, Output, EventEmitter, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { StatusIndicatorComponent } from '../status-indicator';
import {
  Notification,
  NotificationType,
  NotificationGroup,
  NotificationPanelConfig,
  DEFAULT_NOTIFICATION_CONFIG
} from './notification.model';

/**
 * NotificationPanelComponent
 * 
 * A panel component for displaying CORE notifications organized by type.
 * Shows agent activities, system updates, and user alerts with status indicators.
 * 
 * @example
 * <app-notification-panel 
 *   [notifications]="notifications"
 *   (dismiss)="onDismiss($event)"
 *   (action)="onAction($event)">
 * </app-notification-panel>
 */
@Component({
  selector: 'app-notification-panel',
  standalone: true,
  imports: [CommonModule, StatusIndicatorComponent],
  templateUrl: './notification-panel.component.html',
  styleUrl: './notification-panel.component.scss'
})
export class NotificationPanelComponent implements OnInit, OnDestroy {
  /**
   * List of notifications to display
   */
  @Input() notifications: Notification[] = [];

  /**
   * Panel configuration options
   */
  @Input() config: Partial<NotificationPanelConfig> = {};

  /**
   * Whether the panel is currently open/visible
   */
  @Input() isOpen = true;

  /**
   * Panel title
   */
  @Input() title = 'Notifications';

  /**
   * Emitted when a notification is dismissed
   */
  @Output() dismiss = new EventEmitter<string>();

  /**
   * Emitted when all notifications are cleared
   */
  @Output() clearAll = new EventEmitter<void>();

  /**
   * Emitted when a notification action is triggered
   */
  @Output() action = new EventEmitter<{ notificationId: string; actionId: string }>();

  /**
   * Emitted when a notification is marked as read
   */
  @Output() markRead = new EventEmitter<string>();

  /**
   * Emitted when the panel is toggled
   */
  @Output() toggle = new EventEmitter<boolean>();

  /** Merged configuration */
  mergedConfig!: NotificationPanelConfig;

  /** Grouped notifications */
  groups: NotificationGroup[] = [];

  /** Type labels for display */
  private typeLabels: Record<NotificationType, string> = {
    agent: 'Agent Activity',
    system: 'System Updates',
    user: 'User Alerts'
  };

  /** Type icons for display */
  typeIcons: Record<NotificationType, string> = {
    agent: 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z',
    system: 'M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z',
    user: 'M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'
  };

  private autoDismissTimers: Map<string, ReturnType<typeof setTimeout>> = new Map();

  ngOnInit(): void {
    this.mergedConfig = { ...DEFAULT_NOTIFICATION_CONFIG, ...this.config };
    this.updateGroups();
  }

  ngOnDestroy(): void {
    this.autoDismissTimers.forEach(timer => clearTimeout(timer));
    this.autoDismissTimers.clear();
  }

  /**
   * Update notification groups when notifications change
   */
  ngOnChanges(): void {
    if (this.mergedConfig) {
      this.updateGroups();
    }
  }

  /**
   * Get count of unread notifications
   */
  get unreadCount(): number {
    return this.notifications.filter(n => !n.read && !n.dismissed).length;
  }

  /**
   * Get count of active (non-dismissed) notifications
   */
  get activeCount(): number {
    return this.notifications.filter(n => !n.dismissed).length;
  }

  /**
   * Check if there are any notifications
   */
  get hasNotifications(): boolean {
    return this.activeCount > 0;
  }

  /**
   * Organize notifications into groups by type
   */
  private updateGroups(): void {
    const types: NotificationType[] = ['agent', 'system', 'user'];
    const existingExpandState = new Map(this.groups.map(g => [g.type, g.expanded]));

    this.groups = types.map(type => ({
      type,
      label: this.typeLabels[type],
      notifications: this.notifications
        .filter(n => n.type === type && !n.dismissed)
        .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
        .slice(0, this.mergedConfig.maxPerGroup),
      expanded: existingExpandState.get(type) ?? true
    })).filter(group => group.notifications.length > 0);

    // Set up auto-dismiss timers if configured
    if (this.mergedConfig.autoDismissMs > 0) {
      this.notifications
        .filter(n => !n.dismissed && !this.autoDismissTimers.has(n.id))
        .forEach(n => {
          const timer = setTimeout(() => {
            this.dismissNotification(n.id);
          }, this.mergedConfig.autoDismissMs);
          this.autoDismissTimers.set(n.id, timer);
        });
    }
  }

  /**
   * Toggle a group's expanded state
   */
  toggleGroup(group: NotificationGroup): void {
    group.expanded = !group.expanded;
  }

  /**
   * Dismiss a single notification
   */
  dismissNotification(id: string): void {
    if (this.autoDismissTimers.has(id)) {
      clearTimeout(this.autoDismissTimers.get(id));
      this.autoDismissTimers.delete(id);
    }
    this.dismiss.emit(id);
  }

  /**
   * Clear all notifications
   */
  clearAllNotifications(): void {
    this.autoDismissTimers.forEach(timer => clearTimeout(timer));
    this.autoDismissTimers.clear();
    this.clearAll.emit();
  }

  /**
   * Clear notifications in a specific group
   */
  clearGroup(type: NotificationType): void {
    this.notifications
      .filter(n => n.type === type && !n.dismissed)
      .forEach(n => this.dismissNotification(n.id));
  }

  /**
   * Handle notification action click
   */
  onActionClick(notification: Notification): void {
    if (notification.actionId) {
      this.action.emit({
        notificationId: notification.id,
        actionId: notification.actionId
      });
    }
  }

  /**
   * Mark a notification as read
   */
  onNotificationClick(notification: Notification): void {
    if (!notification.read) {
      this.markRead.emit(notification.id);
    }
  }

  /**
   * Toggle panel open/close
   */
  togglePanel(): void {
    this.isOpen = !this.isOpen;
    this.toggle.emit(this.isOpen);
  }

  /**
   * Format timestamp for display
   */
  formatTimestamp(date: Date): string {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (seconds < 60) {
      return 'Just now';
    } else if (minutes < 60) {
      return `${minutes}m ago`;
    } else if (hours < 24) {
      return `${hours}h ago`;
    } else if (days < 7) {
      return `${days}d ago`;
    } else {
      return date.toLocaleDateString();
    }
  }

  /**
   * Get priority class for styling
   */
  getPriorityClass(notification: Notification): string {
    return `priority-${notification.priority}`;
  }

  /**
   * Track notifications by ID for ngFor
   */
  trackById(index: number, notification: Notification): string {
    return notification.id;
  }

  /**
   * Track groups by type for ngFor
   */
  trackByType(index: number, group: NotificationGroup): string {
    return group.type;
  }
}
