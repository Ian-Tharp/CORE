import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface KanbanTask {
  id: string;
  task_type: string;
  priority: string;
  status: string;
  input_data: {
    title?: string;
    description?: string;
    project?: string;
    assignee?: string;
    [key: string]: any;
  };
  result?: any;
  assigned_agent?: string;
  created_at: string;
  updated_at?: string;
}

export interface CreateTaskPayload {
  task_type: string;
  priority: string;
  input_data: {
    title: string;
    description?: string;
    project?: string;
    assignee?: string;
    status_column?: string;
  };
  routing_strategy?: string;
}

export interface TaskAnalytics {
  total_tasks: number;
  by_status: { [key: string]: number };
  by_priority: { [key: string]: number };
}

@Injectable({
  providedIn: 'root'
})
export class KanbanService {
  private readonly apiUrl = 'http://localhost:8001/tasks';
  private readonly headers = new HttpHeaders({
    'Content-Type': 'application/json',
    'X-API-Key': 'core_dev_key'
  });

  constructor(private http: HttpClient) {}

  getTasks(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/`, { headers: this.headers });
  }

  getTask(id: string): Observable<KanbanTask> {
    return this.http.get<KanbanTask>(`${this.apiUrl}/${id}`, { headers: this.headers });
  }

  createTask(payload: CreateTaskPayload): Observable<KanbanTask> {
    return this.http.post<KanbanTask>(`${this.apiUrl}/`, payload, { headers: this.headers });
  }

  updateTaskStatus(id: string, action: 'complete' | 'fail' | 'cancel' | 'retry'): Observable<KanbanTask> {
    return this.http.post<KanbanTask>(`${this.apiUrl}/${id}/${action}`, {}, { headers: this.headers });
  }

  assignTask(id: string, agentId: string): Observable<KanbanTask> {
    return this.http.post<KanbanTask>(`${this.apiUrl}/${id}/assign`, { agent_id: agentId }, { headers: this.headers });
  }

  getAnalyticsOverview(): Observable<TaskAnalytics> {
    return this.http.get<TaskAnalytics>(`${this.apiUrl}/analytics/overview`, { headers: this.headers });
  }

  getAnalyticsAgents(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/analytics/agents`, { headers: this.headers });
  }

  getAnalyticsRouting(): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/analytics/routing`, { headers: this.headers });
  }
}