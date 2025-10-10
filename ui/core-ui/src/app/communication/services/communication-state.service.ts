import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { Channel, InstancePresence } from '../models/communication.models';

@Injectable({
  providedIn: 'root'
})
export class CommunicationStateService {
  private selectedChannelSubject = new BehaviorSubject<Channel | null>(null);
  private currentUserSubject = new BehaviorSubject<InstancePresence>({
    instance_id: 'human_ian',
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
