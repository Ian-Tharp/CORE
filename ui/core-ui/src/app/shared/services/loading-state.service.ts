import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, timer } from 'rxjs';
import { map, distinctUntilChanged } from 'rxjs/operators';

export interface LoadingState {
  key: string;
  isLoading: boolean;
  message?: string;
  startTime?: Date;
  progress?: number; // 0-100 for progress bars
}

export interface GlobalLoadingState {
  isLoading: boolean;
  loadingCount: number;
  activeOperations: LoadingState[];
}

/**
 * Global loading state management service.
 * 
 * Use this to:
 * - Track loading states across components
 * - Show global loading indicators
 * - Prevent duplicate operations
 * - Track operation progress
 * 
 * Example usage:
 * ```typescript
 * // Start loading
 * loadingService.startLoading('fetch-agents', 'Loading agents...');
 * 
 * // Check if specific operation is loading
 * const isLoading$ = loadingService.isLoading$('fetch-agents');
 * 
 * // Stop loading
 * loadingService.stopLoading('fetch-agents');
 * ```
 */
@Injectable({
  providedIn: 'root'
})
export class LoadingStateService {
  private loadingStates = new Map<string, LoadingState>();
  private stateSubject$ = new BehaviorSubject<GlobalLoadingState>({
    isLoading: false,
    loadingCount: 0,
    activeOperations: []
  });

  public state$ = this.stateSubject$.asObservable();
  
  // Convenience observables
  public isLoading$ = this.state$.pipe(
    map(state => state.isLoading),
    distinctUntilChanged()
  );
  
  public loadingCount$ = this.state$.pipe(
    map(state => state.loadingCount),
    distinctUntilChanged()
  );

  /**
   * Start a loading operation
   */
  startLoading(key: string, message?: string): void {
    const state: LoadingState = {
      key,
      isLoading: true,
      message,
      startTime: new Date(),
      progress: 0
    };
    
    this.loadingStates.set(key, state);
    this.emitState();
  }

  /**
   * Stop a loading operation
   */
  stopLoading(key: string): void {
    this.loadingStates.delete(key);
    this.emitState();
  }

  /**
   * Update loading progress (0-100)
   */
  updateProgress(key: string, progress: number, message?: string): void {
    const state = this.loadingStates.get(key);
    if (state) {
      state.progress = Math.min(100, Math.max(0, progress));
      if (message) {
        state.message = message;
      }
      this.emitState();
    }
  }

  /**
   * Update loading message
   */
  updateMessage(key: string, message: string): void {
    const state = this.loadingStates.get(key);
    if (state) {
      state.message = message;
      this.emitState();
    }
  }

  /**
   * Check if a specific operation is loading
   */
  isOperationLoading(key: string): boolean {
    return this.loadingStates.has(key);
  }

  /**
   * Get observable for specific operation loading state
   */
  isLoadingKey$(key: string): Observable<boolean> {
    return this.state$.pipe(
      map(state => state.activeOperations.some(op => op.key === key)),
      distinctUntilChanged()
    );
  }

  /**
   * Get observable for specific operation state
   */
  getOperation$(key: string): Observable<LoadingState | undefined> {
    return this.state$.pipe(
      map(state => state.activeOperations.find(op => op.key === key)),
      distinctUntilChanged((a, b) => JSON.stringify(a) === JSON.stringify(b))
    );
  }

  /**
   * Stop all loading operations
   */
  stopAll(): void {
    this.loadingStates.clear();
    this.emitState();
  }

  /**
   * Execute an async operation with automatic loading state management
   */
  async withLoading<T>(
    key: string,
    operation: () => Promise<T>,
    message?: string
  ): Promise<T> {
    this.startLoading(key, message);
    try {
      return await operation();
    } finally {
      this.stopLoading(key);
    }
  }

  /**
   * Get current state snapshot
   */
  getState(): GlobalLoadingState {
    return this.stateSubject$.value;
  }

  private emitState(): void {
    const activeOperations = Array.from(this.loadingStates.values());
    this.stateSubject$.next({
      isLoading: activeOperations.length > 0,
      loadingCount: activeOperations.length,
      activeOperations
    });
  }
}


/**
 * Toast/notification types for UI feedback
 */
export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message?: string;
  duration?: number; // ms, 0 for persistent
  dismissible?: boolean;
  action?: {
    label: string;
    callback: () => void;
  };
}

/**
 * Notification service for toast messages
 */
@Injectable({
  providedIn: 'root'
})
export class NotificationService {
  private notificationsSubject$ = new BehaviorSubject<Notification[]>([]);
  public notifications$ = this.notificationsSubject$.asObservable();

  private defaultDurations: Record<NotificationType, number> = {
    success: 3000,
    error: 5000,
    warning: 4000,
    info: 3000
  };

  /**
   * Show a notification
   */
  show(
    type: NotificationType,
    title: string,
    message?: string,
    options?: Partial<Notification>
  ): string {
    const id = this.generateId();
    const notification: Notification = {
      id,
      type,
      title,
      message,
      duration: options?.duration ?? this.defaultDurations[type],
      dismissible: options?.dismissible ?? true,
      action: options?.action
    };

    this.notificationsSubject$.next([
      ...this.notificationsSubject$.value,
      notification
    ]);

    // Auto-dismiss
    if (notification.duration && notification.duration > 0) {
      timer(notification.duration).subscribe(() => {
        this.dismiss(id);
      });
    }

    return id;
  }

  // Convenience methods
  success(title: string, message?: string, options?: Partial<Notification>): string {
    return this.show('success', title, message, options);
  }

  error(title: string, message?: string, options?: Partial<Notification>): string {
    return this.show('error', title, message, options);
  }

  warning(title: string, message?: string, options?: Partial<Notification>): string {
    return this.show('warning', title, message, options);
  }

  info(title: string, message?: string, options?: Partial<Notification>): string {
    return this.show('info', title, message, options);
  }

  /**
   * Dismiss a notification
   */
  dismiss(id: string): void {
    this.notificationsSubject$.next(
      this.notificationsSubject$.value.filter(n => n.id !== id)
    );
  }

  /**
   * Clear all notifications
   */
  clearAll(): void {
    this.notificationsSubject$.next([]);
  }

  private generateId(): string {
    return `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}
