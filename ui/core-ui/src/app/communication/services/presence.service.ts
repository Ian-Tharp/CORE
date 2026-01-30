import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, of, interval } from 'rxjs';
import { map, switchMap, startWith } from 'rxjs/operators';
import { InstancePresence } from '../models/communication.models';

@Injectable({
  providedIn: 'root'
})
export class PresenceService {
  private apiUrl = 'http://localhost:8001/communication';
  private currentInstanceId = 'human_ian';

  constructor(private http: HttpClient) {
    // Start heartbeat timer
    this.startHeartbeat();
  }

  getAllPresence(): Observable<InstancePresence[]> {
    return this.http.get<{ instances: InstancePresence[] }>(`${this.apiUrl}/presence`)
      .pipe(map(response => response.instances));
  }

  getOnlineInstances(): Observable<InstancePresence[]> {
    return this.getAllPresence().pipe(
      map(instances => instances.filter(i => i.status === 'online' || i.status === 'away'))
    );
  }

  getInstancesByStatus(status: 'online' | 'away' | 'busy' | 'offline'): Observable<InstancePresence[]> {
    return this.getAllPresence().pipe(
      map(instances => instances.filter(i => i.status === status))
    );
  }

  updatePresence(status?: string, activity?: string, phase?: number): Observable<any> {
    const body: any = {};
    if (status) body.status = status;
    if (activity !== undefined) body.activity = activity;
    if (phase !== undefined) body.phase = phase;

    return this.http.patch(
      `${this.apiUrl}/presence/${this.currentInstanceId}`,
      body
    );
  }

  private startHeartbeat(): void {
    // Send heartbeat every 30 seconds
    interval(30000)
      .pipe(startWith(0))
      .subscribe(() => {
        this.updatePresence('online', 'Using Communication Commons').subscribe({
          error: (err) => console.error('Heartbeat failed:', err)
        });
      });
  }
}
