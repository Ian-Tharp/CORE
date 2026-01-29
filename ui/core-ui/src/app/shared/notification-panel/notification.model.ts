import { StatusType } from '../status-indicator';

/**
 * Notification types for categorizing CORE notifications
 */
export type NotificationType = 'agent' | 'system' | 'user';

/**
 * Notification priority levels
 */
export type NotificationPriority = 'low' | 'normal' | 'high' | 'urgent';

/**
 * Core notification model for CORE system events
 */
export interface Notification {
  /** Unique identifier for the notification */
  id: string;
  
  /** The notification message content */
  message: string;
  
  /** Timestamp when the notification was created */
  timestamp: Date;
  
  /** Type/category of the notification */
  type: NotificationType;
  
  /** Status indicator for the notification */
  status: StatusType;
  
  /** Priority level */
  priority: NotificationPriority;
  
  /** Whether the notification has been read */
  read: boolean;
  
  /** Whether the notification has been dismissed */
  dismissed: boolean;
  
  /** Optional title for the notification */
  title?: string;
  
  /** Optional source identifier (e.g., agent name, service name) */
  source?: string;
  
  /** Optional action callback identifier */
  actionId?: string;
  
  /** Optional action label text */
  actionLabel?: string;
  
  /** Optional metadata for additional context */
  metadata?: Record<string, unknown>;
}

/**
 * Grouped notifications by type
 */
export interface NotificationGroup {
  type: NotificationType;
  label: string;
  notifications: Notification[];
  expanded: boolean;
}

/**
 * Notification panel configuration options
 */
export interface NotificationPanelConfig {
  /** Maximum notifications to display per group */
  maxPerGroup: number;
  
  /** Auto-dismiss notifications after this time (ms), 0 = no auto-dismiss */
  autoDismissMs: number;
  
  /** Show notification count badge */
  showBadge: boolean;
  
  /** Enable sound for new notifications */
  enableSound: boolean;
  
  /** Group notifications by type */
  groupByType: boolean;
}

/**
 * Default notification panel configuration
 */
export const DEFAULT_NOTIFICATION_CONFIG: NotificationPanelConfig = {
  maxPerGroup: 50,
  autoDismissMs: 0,
  showBadge: true,
  enableSound: false,
  groupByType: true
};
