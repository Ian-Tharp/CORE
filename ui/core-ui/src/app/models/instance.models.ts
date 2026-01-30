/**
 * Instance Models for CORE Dashboard
 * 
 * TypeScript interfaces matching the backend Instance Manager API
 */

export interface AgentInstance {
  id: string;
  container_id: string;
  agent_id: string;
  agent_role: string;
  status: 'starting' | 'ready' | 'busy' | 'stopping' | 'unhealthy' | 'lost';
  device_id?: string;
  resource_profile: ResourceProfile;
  capabilities: string[];
  last_heartbeat?: string;
  created_at: string;
  stopped_at?: string;
  current_task?: string;
  uptime?: number;
}

export interface ResourceProfile {
  memory_mb?: number;
  cpu_percent?: number;
  memory_limit_mb?: number;
  cpu_limit?: number;
}

export interface ConnectedAgent {
  id: string;
  name: string;
  role: string;
  status: string;
  last_seen: string;
  capabilities: string[];
}

export interface InstanceStatus {
  id: string;
  status: string;
  health: 'healthy' | 'degraded' | 'unhealthy';
  resource_usage: ResourceProfile;
  last_update: string;
}

export interface SpawnRequest {
  agent_role: string;
  resource_profile?: Partial<ResourceProfile>;
  capabilities?: string[];
  device_id?: string;
}

export interface SpawnResponse {
  success: boolean;
  instance_id?: string;
  container_id?: string;
  message: string;
}

export interface ScaleResponse {
  success: boolean;
  current_count: number;
  target_count: number;
  message: string;
}

export interface ServiceHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  response_time_ms?: number;
  error?: string;
  last_check: string;
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: Record<string, ServiceHealth>;
  uptime: {
    seconds: number;
    formatted: string;
  };
  timestamp: string;
}

export interface TaskSummary {
  total_tasks: number;
  queued: number;
  running: number;
  completed: number;
  failed: number;
  last_update: string;
}

export interface ActivityEvent {
  id: string;
  type: 'agent_spawned' | 'agent_stopped' | 'task_completed' | 'system_event';
  message: string;
  timestamp: string;
  source?: string;
  severity: 'info' | 'warning' | 'error' | 'success';
}

// UI-specific interfaces for dashboard state management
export interface DashboardState {
  loading: boolean;
  error: string | null;
  lastUpdate: string;
}

export interface AgentInstanceUI extends AgentInstance {
  uptimeFormatted?: string;
  statusColor?: string;
  statusIcon?: string;
}