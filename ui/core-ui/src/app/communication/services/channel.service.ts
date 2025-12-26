import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, of, BehaviorSubject } from 'rxjs';
import { map, tap } from 'rxjs/operators';
import { Channel, CreateChannelRequest } from '../models/communication.models';
import { AppConfigService } from '../../services/config/app-config.service';

@Injectable({
  providedIn: 'root'
})
export class ChannelService {
  private apiUrl: string;
  private channelsSubject = new BehaviorSubject<Channel[]>([]);
  channels$ = this.channelsSubject.asObservable();

  // Default instance ID - in production this would come from auth service
  private currentInstanceId = 'human_ian';

  constructor(private http: HttpClient, private config: AppConfigService) {
    this.apiUrl = `${this.config.apiBaseUrl}/communication`;
  }

  getChannels(): Observable<Channel[]> {
    const params = new HttpParams().set('instance_id', this.currentInstanceId);
    return this.http.get<{ channels: Channel[] }>(`${this.apiUrl}/channels`, { params })
      .pipe(
        map(response => response.channels),
        tap(channels => this.channelsSubject.next(channels))
      );
  }

  getChannel(channelId: string): Observable<Channel> {
    return this.http.get<Channel>(`${this.apiUrl}/channels/${channelId}`);
  }

  createChannel(request: CreateChannelRequest): Observable<Channel> {
    const params = new HttpParams().set('created_by', this.currentInstanceId);
    return this.http.post<Channel>(`${this.apiUrl}/channels`, request, { params })
      .pipe(
        tap(newChannel => {
          // Add to local cache
          const current = this.channelsSubject.value;
          this.channelsSubject.next([...current, newChannel]);
        })
      );
  }

  markAsRead(channelId: string): void {
    // Update local cache
    const channels = this.channelsSubject.value;
    const channel = channels.find(c => c.channel_id === channelId);
    if (channel) {
      channel.unread_count = 0;
      this.channelsSubject.next([...channels]);
    }

    // TODO: Call backend API to mark messages as read
    // this.http.post(`${this.apiUrl}/channels/${channelId}/read`, {}).subscribe();
  }
}
