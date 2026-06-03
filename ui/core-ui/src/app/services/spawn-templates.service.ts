import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface AgentCapabilityDto {
  name: string;
  description: string;
}

export interface SpawnTemplateDto {
  id: string;
  name: string;
  description?: string;
  role: string;
  agent_type: string;
  system_prompt: string;
  personality_traits: Record<string, number>;
  capabilities: AgentCapabilityDto[];
  interests: string[];
  tags: string[];
  is_builtin: boolean;
  version: string;
  author?: string;
}

export interface SpawnTemplatesResponse {
  templates: SpawnTemplateDto[];
  count: number;
}

export interface SpawnTemplateRequest {
  agent_id: string;
  agent_name?: string;
  system_prompt_append?: string;
  personality_overrides?: Record<string, number>;
  extra_interests?: string[];
  is_active?: boolean;
}

export interface SpawnTemplateResult {
  agent_id: string;
  agent_name: string;
  template_id: string;
  template_name: string;
  agent_type: string;
  role: string;
  spawned_at: string;
  message: string;
}

@Injectable({ providedIn: 'root' })
export class SpawnTemplatesService {
  private readonly apiUrl = 'http://localhost:8001';

  constructor(private readonly http: HttpClient) {}

  listTemplates(
    params: { role?: string; tag?: string; builtin_only?: boolean } = {}
  ): Observable<SpawnTemplatesResponse> {
    return this.http.get<SpawnTemplatesResponse>(
      `${this.apiUrl}/spawn-templates`,
      { params: params as any }
    );
  }

  spawnFromTemplate(templateId: string, payload: SpawnTemplateRequest): Observable<SpawnTemplateResult> {
    return this.http.post<SpawnTemplateResult>(`${this.apiUrl}/spawn-templates/${templateId}/spawn`, payload);
  }
}
