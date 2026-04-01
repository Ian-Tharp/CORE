import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
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
import { AppConfigService } from './config/app-config.service';

interface BackendInstanceResponse {
  container_id: string;
  agent_id: string;
  agent_role: string;
  status: string;
  health_status?: string | null;
  uptime_seconds?: number | null;
  memory_usage?: Record<string, unknown> | null;
  cpu_usage?: number | null;
  created_at: string;
  last_heartbeat?: string | null;
}

interface BackendListInstancesResponse {
  instances: BackendInstanceResponse[];
  total_count: number;
  page: number;
  page_size: number;
}

interface BackendConnectedAgent {
  agent_id: string;
  role: string;
  status: string;
  last_heartbeat: string;
  capabilities: string[];
}

interface BackendConnectedAgentsResponse {
  agents: BackendConnectedAgent[];
}

interface BackendSpawnInstanceRequest {
  agent_id: string;
  agent_role: string;
  device_id?: string;
  resource_profile: Partial<Record<string, unknown>>;
  capabilities: string[];
  environment_vars: Record<string, string>;
  memory_limit: string;
  cpu_limit: number;
  network: string;
}

interface BackendScaleResponse {
  success: boolean;
  initial_count: number;
  final_count: number;
  target_count: number;
}

@Injectable({
  providedIn: 'root'
})
export class InstanceService {
  private readonly instancesUrl: string;
  private readonly healthUrl: string;
  
  // State management
  private instancesSubject = new BehaviorSubject<AgentInstanceUI[]>([]);
  private systemHealthSubject = new BehaviorSubject<SystemHealth | null>(null);
  private activitiesSubject = new BehaviorSubject<ActivityEvent[]>([]);
  
  public instances$ = this.instancesSubject.asObservable();
  public systemHealth$ = this.systemHealthSubject.asObservable();
  public activities$ = this.activitiesSubject.asObservable();

  constructor(private readonly http: HttpClient, private readonly config: AppConfigService) {
    this.instancesUrl = `${this.config.apiBaseUrl}/instances`;
    this.healthUrl = `${this.config.apiBaseUrl}/health`;
  }

  /**
   * Get all agent instances
   */
  getInstances(): Observable<AgentInstance[]> {
    const params = new HttpParams()
      .set('include_docker', 'false')
      .set('page', '1')
      .set('page_size', '200');

    return this.http.get<BackendListInstancesResponse>(this.instancesUrl, { params }).pipe(
      map(response => (response.instances ?? []).map(instance => this._mapBackendInstance(instance))),
      catchError(error => {
        console.warn('Error fetching instances:', error);
        return of([]);
      }),
      tap(instances => {
        const uiInstances = instances.map(i => this.enhanceInstanceForUI(i));
        this.instancesSubject.next(uiInstances);
      })
    );
  }

  /**
   * Get instance status by ID
   */
  getInstanceStatus(id: string): Observable<InstanceStatus> {
    return this.http.get<BackendInstanceResponse>(`${this.instancesUrl}/${id}`).pipe(
      map(instance => ({
        id: instance.container_id,
        status: instance.status,
        health: this._coerceHealthStatus(instance.health_status),
        resource_usage: {
          cpu_percent: instance.cpu_usage ?? 0,
          ...(instance.memory_usage ?? {})
        },
        last_update: instance.last_heartbeat ?? instance.created_at
      })),
      catchError(error => {
        console.warn(`Error fetching status for instance ${id}:`, error);
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
    return this.http.get<BackendConnectedAgentsResponse>(`${this.instancesUrl}/agents/connected`).pipe(
      map(response => (response.agents ?? []).map(agent => ({
        id: agent.agent_id,
        name: agent.agent_id,
        role: agent.role,
        status: agent.status,
        last_seen: agent.last_heartbeat,
        capabilities: agent.capabilities ?? []
      }))),
      catchError(error => {
        console.warn('Error fetching connected agents:', error);
        return of([]);
      })
    );
  }

  /**
   * Spawn a new agent instance
   */
  spawnInstance(config: SpawnRequest): Observable<SpawnResponse> {
    return this.http.post<BackendInstanceResponse>(
      this.instancesUrl,
      this._buildSpawnRequest(config)
    ).pipe(
      map(response => ({
        success: true,
        instance_id: response.container_id,
        container_id: response.container_id,
        message: `Spawned ${response.agent_role} agent`
      })),
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
          this._refreshInstancesLater();
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
    return this.http.post<void>(`${this.instancesUrl}/${id}/stop`, {}).pipe(
      catchError(error => {
        console.error(`Error stopping instance ${id}:`, error);
        return of(undefined);
      }),
      tap(() => {
        // Refresh instances after stop
        this._refreshInstancesLater();
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
    return this.http.post<void>(`${this.instancesUrl}/${id}/restart`, {}).pipe(
      catchError(error => {
        console.error(`Error restarting instance ${id}:`, error);
        return of(undefined);
      }),
      tap(() => {
        // Refresh instances after restart
        this._refreshInstancesLater();
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
    return this.http.post<BackendScaleResponse>(`${this.instancesUrl}/scale`, { role, target_count: count }).pipe(
      map(response => ({
        success: response.success,
        current_count: response.final_count,
        target_count: response.target_count,
        message: response.success
          ? `Scaled ${role} agents to ${response.final_count}`
          : `Failed to scale ${role} agents`
      })),
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
          this._refreshInstancesLater();
        }
      })
    );
  }

  /**
   * Get system health status
   */
  getSystemHealth(): Observable<SystemHealth> {
    return this.http.get<SystemHealth>(`${this.healthUrl}/deep`).pipe(
      catchError(error => {
        console.warn('Error fetching system health:', error);
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

  private _mapBackendInstance(instance: BackendInstanceResponse): AgentInstance {
    return {
      id: instance.container_id,
      container_id: instance.container_id,
      agent_id: instance.agent_id,
      agent_role: instance.agent_role,
      status: (instance.status as AgentInstance['status']) || 'lost',
      resource_profile: {
        cpu_percent: instance.cpu_usage ?? undefined
      },
      capabilities: [],
      last_heartbeat: instance.last_heartbeat ?? undefined,
      created_at: instance.created_at,
      uptime: instance.uptime_seconds ?? undefined
    };
  }

  private _coerceHealthStatus(status?: string | null): InstanceStatus['health'] {
    if (status === 'healthy' || status === 'degraded' || status === 'unhealthy') {
      return status;
    }

    return 'unhealthy';
  }

  private _buildSpawnRequest(config: SpawnRequest): BackendSpawnInstanceRequest {
    const memoryLimitMb = config.resource_profile?.memory_limit_mb ?? 512;
    const cpuLimit = config.resource_profile?.cpu_limit ?? 0.5;

    return {
      agent_id: `ui_${config.agent_role}_${Date.now()}`,
      agent_role: config.agent_role,
      device_id: config.device_id,
      resource_profile: config.resource_profile ?? {},
      capabilities: config.capabilities ?? [],
      environment_vars: {},
      memory_limit: `${memoryLimitMb}m`,
      cpu_limit: cpuLimit,
      network: 'core-network'
    };
  }

  private _refreshInstancesLater(): void {
    setTimeout(() => {
      this.getInstances().subscribe({
        error: () => undefined
      });
    }, 1000);
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