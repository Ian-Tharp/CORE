import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { Channel, InstancePresence } from '../models/communication.models';

@Injectable({
  providedIn: 'root'
})
export class CommunicationStateService {
  private selectedChannelSubject = new BehaviorSubject<Channel | null>(null);

  // Generate unique instance ID per window/tab to support multiple connections
  private generateInstanceId(): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 9);
    return `human_ian_${timestamp}_${random}`;
  }

  private currentUserSubject = new BehaviorSubject<InstancePresence>({
    instance_id: this.generateInstanceId(),
    instance_name: 'Ian',
    instance_type: 'human',
    status: 'online',
    last_heartbeat: new Date().toISOString()
  });

  selectedChannel$ = this.selectedChannelSubject.asObservable();
  currentUser$ = this.currentUserSubject.asObservable();

  get selectedChannel(): Channel | null {
    return this.selectedChannelSubject.value;
  }

  get currentUser(): InstancePresence {
    return this.currentUserSubject.value;
  }

  selectChannel(channel: Channel): void {
    this.selectedChannelSubject.next(channel);
  }

  setCurrentUser(user: InstancePresence): void {
    this.currentUserSubject.next(user);
  }
}
