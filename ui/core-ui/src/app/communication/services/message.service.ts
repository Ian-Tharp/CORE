import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { map, tap } from 'rxjs/operators';
import { Message } from '../models/communication.models';
import { AppConfigService } from '../../services/config/app-config.service';

@Injectable({
  providedIn: 'root'
})
export class MessageService {
  private apiUrl: string;
  private messagesCache: Map<string, Message[]> = new Map();

  // Default sender info - in production this would come from auth service
  private currentInstanceId = 'human_ian';
  private currentInstanceName = 'Ian';
  private currentInstanceType = 'human';

  constructor(private http: HttpClient, private config: AppConfigService) {
    this.apiUrl = `${this.config.apiBaseUrl}/communication`;
  }

  getChannelMessages(channelId: string, page: number = 1): Observable<Message[]> {
    const params = new HttpParams()
      .set('page', page.toString())
      .set('page_size', '50');

    return this.http.get<{ messages: Message[] }>(
      `${this.apiUrl}/channels/${channelId}/messages`,
      { params }
    ).pipe(
      map(response => response.messages.reverse()), // Reverse to show oldest first
      tap(messages => this.messagesCache.set(channelId, messages))
    );
  }

  sendMessage(
    channelId: string,
    content: string,
    messageType: string = 'text',
    metadata?: any,
    parentMessageId?: string,
    threadId?: string
  ): Observable<Message> {
    const params = new HttpParams()
      .set('sender_id', this.currentInstanceId)
      .set('sender_name', this.currentInstanceName)
      .set('sender_type', this.currentInstanceType);

    const body: any = {
      content,
      message_type: messageType,
      metadata
    };
    if (parentMessageId) {body.parent_message_id = parentMessageId;}
    if (threadId) {body.thread_id = threadId;}

    return this.http.post<Message>(
      `${this.apiUrl}/channels/${channelId}/messages`,
      body,
      { params }
    ).pipe(
      tap(newMessage => {
        // Add to cache
        const cached = this.messagesCache.get(channelId) || [];
        this.messagesCache.set(channelId, [...cached, newMessage]);
      })
    );
  }

  addReaction(messageId: string, reactionType: string): Observable<any> {
    const params = new HttpParams().set('instance_id', this.currentInstanceId);
    const body = { reaction_type: reactionType };

    return this.http.post(
      `${this.apiUrl}/messages/${messageId}/reactions`,
      body,
      { params }
    );
  }

  removeReaction(messageId: string, reactionType: string): Observable<any> {
    const params = new HttpParams().set('instance_id', this.currentInstanceId);

    return this.http.delete(
      `${this.apiUrl}/messages/${messageId}/reactions/${reactionType}`,
      { params }
    );
  }

  /**
   * Get reactions for a specific message
   * Note: In the current implementation, reactions are included in message objects,
   * but this method allows for explicit fetching if needed
   */
  getReactions(messageId: string): Observable<Message['reactions']> {
    // For now, extract reactions from cached messages
    // In a full implementation, this would be:
    // return this.http.get<MessageReaction[]>(`${this.apiUrl}/messages/${messageId}/reactions`);

    return new Observable(observer => {
      let found = false;
      this.messagesCache.forEach(messages => {
        const message = messages.find(m => m.message_id === messageId);
        if (message) {
          observer.next(message.reactions || []);
          found = true;
        }
      });

      if (!found) {
        observer.next([]);
      }
      observer.complete();
    });
  }

  /**
   * Get all messages in a thread
   * Production: GET /api/channels/{channelId}/threads/{threadId}
   */
  getThreadMessages(threadId: string): Observable<Message[]> {
    // In production: return this.http.get<Message[]>(`/api/threads/${threadId}/messages`)

    // Get all cached messages and filter for thread
    const allMessages: Message[] = [];
    this.messagesCache.forEach(messages => {
      allMessages.push(...messages);
    });

    const threadMessages = allMessages.filter(m =>
      m.thread_id === threadId || m.message_id === threadId
    );

    // Sort by timestamp
    return of(threadMessages.sort((a, b) =>
      new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
    ));
  }

  /**
   * Get reply count for a message
   */
  getReplyCount(messageId: string): number {
    const allMessages: Message[] = [];
    this.messagesCache.forEach(messages => {
      allMessages.push(...messages);
    });

    return allMessages.filter(m =>
      m.thread_id === messageId && m.message_id !== messageId
    ).length;
  }

}
