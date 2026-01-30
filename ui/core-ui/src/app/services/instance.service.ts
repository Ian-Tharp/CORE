import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, interval, switchMap, startWith, catchError, of, BehaviorSubject } from 'rxjs';
import { map, tap } from 'rxjs/operators';
import {
  AgentInstance,
  ConnectedAgent,
  InstanceStatus,
  SpawnRequest,
  SpawnResponse,
  ScaleResponse,
  SystemHealth,
  TaskSummary,
  ActivityEvent,
  AgentInstanceUI
} from '../models/instance.models';

@Injectable({
  providedIn: 'root'
})
export class InstanceService {
  private readonly apiUrl = 'http://localhost:8001/api';  // Backend API URL
  
  // State management
  private instancesSubject = new BehaviorSubject<AgentInstanceUI[]>([]);
  private systemHealthSubject = new BehaviorSubject<SystemHealth | null>(null);
  private activitiesSubject = new BehaviorSubject<ActivityEvent[]>([]);
  
  public instances$ = this.instancesSubject.asObservable();
  public systemHealth$ = this.systemHealthSubject.asObservable();
  public activities$ = this.activitiesSubject.asObservable();

  constructor(private http: HttpClient) {}

  /**
   * Get all agent instances
   */
  getInstances(): Observable<AgentInstance[]> {
    return this.http.get<AgentInstance[]>(`${this.apiUrl}/instances`).pipe(
      catchError(error => {
        console.error('Error fetching instances:', error);
        return of([]);
      }),
      tap(instances => {
        const uiInstances = instances.map(this.enhanceInstanceForUI);
        this.instancesSubject.next(uiInstances);
      })
    );
  }

  /**
   * Get instance status by ID
   */
  getInstanceStatus(id: string): Observable<InstanceStatus> {
    return this.http.get<InstanceStatus>(`${this.apiUrl}/instances/${id}/status`).pipe(
      catchError(error => {
        console.error(`Error fetching status for instance ${id}:`, error);
        return of({
          id,
          status: 'unknown',
          health: 'unhealthy',
          resource_usage: {},
          last_update: new Date().toISOString()
        } as InstanceStatus);
      })
    );
  }

  /**
   * Get currently connected agents
   */
  getConnectedAgents(): Observable<ConnectedAgent[]> {
    return this.http.get<ConnectedAgent[]>(`${this.apiUrl}/agents/connected`).pipe(
      catchError(error => {
        console.error('Error fetching connected agents:', error);
        return of([]);
      })
    );
  }

  /**
   * Spawn a new agent instance
   */
  spawnInstance(config: SpawnRequest): Observable<SpawnResponse> {
    return this.http.post<SpawnResponse>(`${this.apiUrl}/instances`, config).pipe(
      catchError(error => {
        console.error('Error spawning instance:', error);
        return of({
          success: false,
          message: 'Failed to spawn instance: ' + (error.message || 'Unknown error')
        });
      }),
      tap(response => {
        if (response.success) {
          // Refresh instances after successful spawn
          setTimeout(() => this.getInstances().subscribe(), 1000);
          this.addActivity({
            id: `spawn_${Date.now()}`,
            type: 'agent_spawned',
            message: `New ${config.agent_role} agent spawned successfully`,
            timestamp: new Date().toISOString(),
            severity: 'success'
          });
        }
      })
    );
  }

  /**
   * Stop an agent instance
   */
  stopInstance(id: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/instances/${id}`).pipe(
      catchError(error => {
        console.error(`Error stopping instance ${id}:`, error);
        return of(undefined);
      }),
      tap(() => {
        // Refresh instances after stop
        setTimeout(() => this.getInstances().subscribe(), 1000);
        this.addActivity({
          id: `stop_${Date.now()}`,
          type: 'agent_stopped',
          message: `Agent instance ${id} stopped`,
          timestamp: new Date().toISOString(),
          severity: 'info'
        });
      })
    );
  }

  /**
   * Restart an agent instance
   */
  restartInstance(id: string): Observable<void> {
    return this.http.post<void>(`${this.apiUrl}/instances/${id}/restart`, {}).pipe(
      catchError(error => {
        console.error(`Error restarting instance ${id}:`, error);
        return of(undefined);
      }),
      tap(() => {
        // Refresh instances after restart
        setTimeout(() => this.getInstances().subscribe(), 1000);
        this.addActivity({
          id: `restart_${Date.now()}`,
          type: 'system_event',
          message: `Agent instance ${id} restarted`,
          timestamp: new Date().toISOString(),
          severity: 'info'
        });
      })
    );
  }

  /**
   * Scale instances for a specific role
   */
  scaleInstances(role: string, count: number): Observable<ScaleResponse> {
    return this.http.post<ScaleResponse>(`${this.apiUrl}/instances/scale`, { role, count }).pipe(
      catchError(error => {
        console.error(`Error scaling instances for role ${role}:`, error);
        return of({
          success: false,
          current_count: 0,
          target_count: count,
          message: 'Failed to scale instances: ' + (error.message || 'Unknown error')
        });
      }),
      tap(response => {
        if (response.success) {
          setTimeout(() => this.getInstances().subscribe(), 1000);
        }
      })
    );
  }

  /**
   * Get system health status
   */
  getSystemHealth(): Observable<SystemHealth> {
    return this.http.get<SystemHealth>(`${this.apiUrl}/health/deep`).pipe(
      catchError(error => {
        console.error('Error fetching system health:', error);
        return of({
          status: 'unhealthy',
          services: {},
          uptime: { seconds: 0, formatted: '0s' },
          timestamp: new Date().toISOString()
        } as SystemHealth);
      }),
      tap(health => {
        this.systemHealthSubject.next(health);
      })
    );
  }

  /**
   * Get task summary (placeholder for future task API integration)
   */
  getTaskSummary(): Observable<TaskSummary> {
    // For now, return mock data with proper structure
    return of({
      total_tasks: 0,
      queued: 0,
      running: 0,
      completed: 0,
      failed: 0,
      last_update: new Date().toISOString()
    });
  }

  /**
   * Get recent activity events
   */
  getRecentActivities(): Observable<ActivityEvent[]> {
    // For now, return the stored activities
    // In the future, this could be backed by a real API endpoint
    return this.activities$;
  }

  /**
   * Start polling for instances (every 10 seconds)
   */
  startInstancePolling(): Observable<AgentInstanceUI[]> {
    return interval(10000).pipe(
      startWith(0),
      switchMap(() => this.getInstances()),
      map(() => this.instancesSubject.value)
    );
  }

  /**
   * Start polling for system health (every 30 seconds)
   */
  startSystemHealthPolling(): Observable<SystemHealth | null> {
    return interval(30000).pipe(
      startWith(0),
      switchMap(() => this.getSystemHealth()),
      map(() => this.systemHealthSubject.value)
    );
  }

  // Private helper methods

  /**
   * Enhance instance data for UI display
   */
  private enhanceInstanceForUI(instance: AgentInstance): AgentInstanceUI {
    const enhanced = { ...instance } as AgentInstanceUI;
    
    // Calculate uptime
    if (instance.created_at) {
      const created = new Date(instance.created_at);
      const now = new Date();
      const uptimeMs = now.getTime() - created.getTime();
      enhanced.uptime = Math.floor(uptimeMs / 1000); // seconds
      enhanced.uptimeFormatted = this.formatUptime(enhanced.uptime);
    }

    // Determine status color and icon
    const statusMapping = {
      'starting': { color: '#ffaa00', icon: 'hourglass_empty' },
      'ready': { color: '#00ff88', icon: 'check_circle' },
      'busy': { color: '#00eaff', icon: 'sync' },
      'stopping': { color: '#ffaa00', icon: 'stop' },
      'unhealthy': { color: '#ff5757', icon: 'error' },
      'lost': { color: '#ff5757', icon: 'error_outline' }
    };

    const statusInfo = statusMapping[instance.status] || statusMapping['lost'];
    enhanced.statusColor = statusInfo.color;
    enhanced.statusIcon = statusInfo.icon;

    return enhanced;
  }

  /**
   * Format uptime duration
   */
  private formatUptime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  }

  /**
   * Add an activity event to the local store
   */
  private addActivity(event: ActivityEvent): void {
    const current = this.activitiesSubject.value;
    const updated = [event, ...current].slice(0, 20); // Keep only last 20 activities
    this.activitiesSubject.next(updated);
  }
}