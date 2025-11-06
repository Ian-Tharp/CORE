import { Injectable } from '@angular/core';
import { Observable, Subject, timer } from 'rxjs';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { retryWhen, tap, delayWhen } from 'rxjs/operators';

export interface WebSocketMessage {
  type: string;
  [key: string]: any;
}

@Injectable({
  providedIn: 'root'
})
export class WebSocketService {
  private socket$: WebSocketSubject<WebSocketMessage> | null = null;
  private messagesSubject$ = new Subject<WebSocketMessage>();
  public messages$ = this.messagesSubject$.asObservable();

  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 2000; // 2 seconds

  private currentInstanceId: string | null = null;
  private isConnected = false;

  constructor() {}

  /**
   * Connect to WebSocket server
   */
  connect(instanceId: string): void {
    if (this.isConnected && this.currentInstanceId === instanceId) {
      console.log('WebSocket already connected');
      return;
    }

    this.currentInstanceId = instanceId;
    const wsUrl = `ws://localhost:8001/ws/${instanceId}`;

    console.log('Connecting to WebSocket:', wsUrl);

    this.socket$ = webSocket<WebSocketMessage>({
      url: wsUrl,
      openObserver: {
        next: () => {
          console.log('WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
        }
      },
      closeObserver: {
        next: () => {
          console.log('WebSocket disconnected');
          this.isConnected = false;
          this.attemptReconnect();
        }
      }
    });

    // Subscribe to incoming messages
    this.socket$
      .pipe(
        tap(message => console.log('Received WebSocket message:', message)),
        retryWhen(errors =>
          errors.pipe(
            tap(err => console.error('WebSocket error:', err)),
            delayWhen(() => timer(this.reconnectInterval))
          )
        )
      )
      .subscribe({
        next: (message) => this.messagesSubject$.next(message),
        error: (err) => console.error('WebSocket error:', err)
      });
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    if (this.socket$) {
      this.socket$.complete();
      this.socket$ = null;
      this.isConnected = false;
      console.log('WebSocket manually disconnected');
    }
  }

  /**
   * Send a message through WebSocket
   */
  send(message: WebSocketMessage): void {
    if (this.socket$ && this.isConnected) {
      this.socket$.next(message);
    } else {
      console.warn('WebSocket not connected, cannot send message');
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
   * Send a ping to keep connection alive
   */
  ping(): void {
    this.send({ type: 'ping' });
  }

  /**
   * Check if WebSocket is connected
   */
  isWebSocketConnected(): boolean {
    return this.isConnected;
  }

  /**
   * Get observable for specific message types
   */
  onMessageType(type: string): Observable<WebSocketMessage> {
    return new Observable(observer => {
      const subscription = this.messages$.subscribe(message => {
        if (message.type === type) {
          observer.next(message);
        }
      });
      return () => subscription.unsubscribe();
    });
  }

  /**
   * Attempt to reconnect with exponential backoff
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      if (this.currentInstanceId) {
        this.connect(this.currentInstanceId);
      }
    }, delay);
  }
}
