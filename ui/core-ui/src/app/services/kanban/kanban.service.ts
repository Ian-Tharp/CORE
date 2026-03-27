import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AppConfigService } from '../config/app-config.service';

export interface KanbanTask {
  id: string;
  task_type: string;
  priority: number;
  status: string;
  payload: {
    title?: string;
    description?: string;
    project?: string;
    assignee?: string;
    status_column?: string;
    priority_label?: string;
    [key: string]: any;
  };
  result?: any;
  assigned_agent_id?: string | null;
  created_at: string;
  assigned_at?: string | null;
  completed_at?: string | null;
}

export interface CreateTaskPayload {
  task_type: string;
  priority: number;
  payload: {
    title: string;
    description?: string;
    project?: string;
    assignee?: string;
    status_column?: string;
    priority_label?: string;
  };
}

@Injectable({
  providedIn: 'root'
})
export class KanbanService {
  private readonly http = inject(HttpClient);
  private readonly cfg = inject(AppConfigService);
  private readonly headers = new HttpHeaders({
    'X-API-Key': 'core_dev_key'
  });

  private get tasksUrl(): string {
    return `${this.cfg.apiBaseUrl}/tasks`;
  }

  getTasks(): Observable<any> {
    return this.http.get<any>(`${this.tasksUrl}/`, { headers: this.headers });
  }

  getTask(id: string): Observable<KanbanTask> {
    return this.http.get<KanbanTask>(`${this.tasksUrl}/${id}`, { headers: this.headers });
  }

  createTask(payload: CreateTaskPayload): Observable<any> {
    return this.http.post<any>(`${this.tasksUrl}/`, payload, { headers: this.headers });
  }

  updateTaskStatus(id: string, action: 'complete' | 'fail' | 'cancel' | 'retry'): Observable<any> {
    return this.http.post<any>(`${this.tasksUrl}/${id}/${action}`, {}, { headers: this.headers });
  }

  assignTask(id: string, agentId: string): Observable<any> {
    return this.http.post<any>(`${this.tasksUrl}/${id}/assign`, { agent_id: agentId }, { headers: this.headers });
  }

  getAnalyticsOverview(): Observable<any> {
    return this.http.get<any>(`${this.tasksUrl}/analytics/overview`, { headers: this.headers });
  }
}