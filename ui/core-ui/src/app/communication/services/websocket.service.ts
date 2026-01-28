import { Injectable } from '@angular/core';
import { Observable, Subject, BehaviorSubject, timer, interval, Subscription } from 'rxjs';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { retryWhen, tap, delayWhen, takeUntil, filter } from 'rxjs/operators';
import { AppConfigService } from '../../services/config/app-config.service';

export interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

export interface ConnectionStatus {
  state: ConnectionState;
  instanceId: string | null;
  reconnectAttempt: number;
  lastConnected: Date | null;
  error: string | null;
}

@Injectable({
  providedIn: 'root'
})
export class WebSocketService {
  private socket$: WebSocketSubject<WebSocketMessage> | null = null;
  private messagesSubject$ = new Subject<WebSocketMessage>();
  public messages$ = this.messagesSubject$.asObservable();

  // Connection state management
  private connectionStatusSubject$ = new BehaviorSubject<ConnectionStatus>({
    state: 'disconnected',
    instanceId: null,
    reconnectAttempt: 0,
    lastConnected: null,
    error: null
  });
  public connectionStatus$ = this.connectionStatusSubject$.asObservable();

  // Configuration
  private maxReconnectAttempts = 10;
  private baseReconnectInterval = 1000; // 1 second
  private maxReconnectInterval = 30000; // 30 seconds
  private pingInterval = 25000; // 25 seconds (server timeout is typically 30s)

  private currentInstanceId: string | null = null;
  private pingSubscription: Subscription | null = null;
  private reconnectTimer: any = null;
  private destroy$ = new Subject<void>();

  constructor(private config: AppConfigService) {}

  /**
   * Connect to WebSocket server with automatic reconnection
   */
  connect(instanceId: string): void {
    const currentStatus = this.connectionStatusSubject$.value;
    
    if (currentStatus.state === 'connected' && this.currentInstanceId === instanceId) {
      console.log('WebSocket already connected');
      return;
    }

    // Clean up any existing connection
    this.cleanupConnection();

    this.currentInstanceId = instanceId;
    const wsUrl = this.buildWebSocketUrl(instanceId);

    this.updateStatus({ state: 'connecting', instanceId });
    console.log('Connecting to WebSocket:', wsUrl);

    this.socket$ = webSocket<WebSocketMessage>({
      url: wsUrl,
      openObserver: {
        next: () => {
          console.log('WebSocket connected');
          this.updateStatus({ 
            state: 'connected', 
            reconnectAttempt: 0,
            lastConnected: new Date(),
            error: null
          });
          this.startPingInterval();
        }
      },
      closeObserver: {
        next: (event) => {
          console.log('WebSocket disconnected', event);
          this.stopPingInterval();
          this.scheduleReconnect();
        }
      }
    });

    // Subscribe to incoming messages
    this.socket$
      .pipe(
        takeUntil(this.destroy$),
        tap(message => {
          if (message.type !== 'pong') {
            console.log('Received WebSocket message:', message.type);
          }
        })
      )
      .subscribe({
        next: (message) => this.messagesSubject$.next(message),
        error: (err) => {
          console.error('WebSocket error:', err);
          this.updateStatus({ state: 'error', error: err.message || 'Connection error' });
          this.scheduleReconnect();
        }
      });
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    console.log('WebSocket manually disconnected');
    this.cleanupConnection();
    this.updateStatus({ 
      state: 'disconnected', 
      instanceId: null,
      reconnectAttempt: 0,
      error: null
    });
  }

  /**
   * Send a message through WebSocket
   */
  send(message: WebSocketMessage): boolean {
    if (this.socket$ && this.connectionStatusSubject$.value.state === 'connected') {
      this.socket$.next(message);
      return true;
    } else {
      console.warn('WebSocket not connected, cannot send message');
      return false;
    }
  }

  /**
   * Subscribe to specific channel(s)
   */
  subscribeToChannels(channelIds: string[]): void {
    this.send({
      type: 'subscribe',
      channel_ids: channelIds
    });
  }

  /**
   * Unsubscribe from specific channel(s)
   */
  unsubscribeFromChannels(channelIds: string[]): void {
    this.send({
      type: 'unsubscribe',
      channel_ids: channelIds
    });
  }

  /**
   * Start typing indicator
   */
  startTyping(channelId: string): void {
    this.send({
      type: 'typing_start',
      channel_id: channelId
    });
  }

  /**
   * Stop typing indicator
   */
  stopTyping(channelId: string): void {
    this.send({
      type: 'typing_stop',
      channel_id: channelId
    });
  }

  /**
   * Mark a message as read
   */
  markRead(messageId: string, channelId: string): void {
    this.send({
      type: 'mark_read',
      message_id: messageId,
      channel_id: channelId
    });
  }

  /**
   * Set connection metadata
   */
  setMetadata(metadata: Record<string, any>): void {
    this.send({
      type: 'set_metadata',
      metadata
    });
  }

  /**
   * Send a ping to keep connection alive
   */
  ping(): void {
    this.send({ type: 'ping' });
  }

  /**
   * Check if WebSocket is connected
   */
  isWebSocketConnected(): boolean {
    return this.connectionStatusSubject$.value.state === 'connected';
  }

  /**
   * Get current connection state
   */
  getConnectionState(): ConnectionState {
    return this.connectionStatusSubject$.value.state;
  }

  /**
   * Get observable for specific message types
   */
  onMessageType(type: string): Observable<WebSocketMessage> {
    return this.messages$.pipe(
      filter(message => message.type === type)
    );
  }

  /**
   * Get observable for typing indicators
   */
  onTypingStart(): Observable<WebSocketMessage> {
    return this.onMessageType('typing_start');
  }

  onTypingStop(): Observable<WebSocketMessage> {
    return this.onMessageType('typing_stop');
  }

  /**
   * Get observable for read receipts
   */
  onReadReceipt(): Observable<WebSocketMessage> {
    return this.onMessageType('read_receipt');
  }

  /**
   * Get observable for presence updates
   */
  onPresenceUpdate(): Observable<WebSocketMessage> {
    return this.onMessageType('presence');
  }

  /**
   * Cleanup on service destroy
   */
  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.cleanupConnection();
  }

  // =========================================================================
  // Private Methods
  // =========================================================================

  private cleanupConnection(): void {
    this.stopPingInterval();
    this.cancelReconnect();
    
    if (this.socket$) {
      this.socket$.complete();
      this.socket$ = null;
    }
  }

  private updateStatus(partial: Partial<ConnectionStatus>): void {
    this.connectionStatusSubject$.next({
      ...this.connectionStatusSubject$.value,
      ...partial
    });
  }

  private startPingInterval(): void {
    this.stopPingInterval();
    this.pingSubscription = interval(this.pingInterval)
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => {
        if (this.isWebSocketConnected()) {
          this.ping();
        }
      });
  }

  private stopPingInterval(): void {
    if (this.pingSubscription) {
      this.pingSubscription.unsubscribe();
      this.pingSubscription = null;
    }
  }

  private scheduleReconnect(): void {
    const currentAttempt = this.connectionStatusSubject$.value.reconnectAttempt;
    
    if (currentAttempt >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.updateStatus({ state: 'error', error: 'Max reconnection attempts reached' });
      return;
    }

    const nextAttempt = currentAttempt + 1;
    // Exponential backoff with jitter
    const delay = Math.min(
      this.baseReconnectInterval * Math.pow(2, currentAttempt) + Math.random() * 1000,
      this.maxReconnectInterval
    );

    console.log(`Scheduling reconnect in ${Math.round(delay)}ms (attempt ${nextAttempt}/${this.maxReconnectAttempts})`);
    
    this.updateStatus({ state: 'reconnecting', reconnectAttempt: nextAttempt });

    this.reconnectTimer = setTimeout(() => {
      if (this.currentInstanceId) {
        this.connect(this.currentInstanceId);
      }
    }, delay);
  }

  private cancelReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private buildWebSocketUrl(instanceId: string): string {
    const base = this.config.apiBaseUrl; // e.g., http://localhost:8001
    // Convert http(s) → ws(s)
    let wsBase = base.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:');
    // Ensure no trailing slash
    if (wsBase.endsWith('/')) {
      wsBase = wsBase.slice(0, -1);
    }
    return `${wsBase}/ws/${instanceId}`;
  }
}
