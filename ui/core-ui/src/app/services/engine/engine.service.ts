import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface StepResponse {
  step: string;
  text: string;
  routing_decision?: string;
  plan?: string[];
  evaluation?: string;
}

export type StepStreamEvent =
  | { type: 'start'; step: string }
  | { type: 'chunk'; text: string }
  | { type: 'metrics'; duration_ms: number; ttfb_ms: number; tokens: number }
  | { type: 'end' };

// CORE unified API types
export interface UserIntent {
  type: 'task' | 'conversation' | 'question' | 'clarification';
  description: string;
  confidence: number;
  requires_tools: boolean;
  tools_needed: string[];
  context_retrieved?: string;
  ambiguities: string[];
}

export interface PlanStep {
  id: string;
  name: string;
  description: string;
  tool?: string;
  params: Record<string, any>;
  dependencies: string[];
  requires_hitl: boolean;
  retry_policy: { max_attempts: number; backoff_seconds: number };
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  started_at?: string;
  completed_at?: string;
}

export interface ExecutionPlan {
  id: string;
  goal: string;
  steps: PlanStep[];
  created_at: string;
  updated_at: string;
  revision: number;
  reasoning: string;
}

export interface StepResult {
  step_id: string;
  status: 'success' | 'failure' | 'partial';
  outputs: Record<string, any>;
  artifacts: string[];
  logs: string[];
  error?: string;
  attempt: number;
  duration_seconds: number;
}

export interface EvaluationResult {
  overall_status: 'success' | 'failure' | 'needs_revision' | 'needs_retry';
  confidence: number;
  meets_requirements: boolean;
  quality_score: number;
  feedback: string;
  next_action: 'finalize' | 'retry_step' | 'revise_plan' | 'ask_user';
  retry_step_id?: string;
  revision_suggestions: string[];
}

export interface COREState {
  run_id: string;
  conversation_id?: string;
  user_id?: string;
  user_input: string;
  intent?: UserIntent;
  plan?: ExecutionPlan;
  step_results: StepResult[];
  current_step_id?: string;
  eval_result?: EvaluationResult;
  response?: string;
  current_node: string;
  execution_history: string[];
  errors: string[];
  warnings: string[];
  started_at: string;
  updated_at: string;
  completed_at?: string;
  config: Record<string, any>;
}

export interface RunRequest {
  input: string;
  config?: Record<string, any>;
  conversation_id?: string;
  user_id?: string;
}

export interface RunResponse {
  run_id: string;
  status: string;
  message: string;
}

export type COREStreamEvent =
  | { event: 'start'; run_id: string; timestamp: string }
  | { event: 'node_start'; node: string; timestamp: string }
  | { event: 'node_complete'; node: string; timestamp: string }
  | { event: 'intent_classified'; intent_type: string; confidence: number; timestamp: string }
  | { event: 'plan_created'; goal: string; reasoning: string; steps_count: number; steps: Array<{name: string; description: string; tool?: string; requires_hitl: boolean}>; timestamp: string }
  | { event: 'step_executed'; step_id: string; status: string; outputs: Record<string, any>; artifacts: string[]; logs: string[]; error?: string; duration_seconds: number; timestamp: string }
  | { event: 'evaluation_complete'; overall_status: string; confidence: number; quality_score: number; timestamp: string }
  | { event: 'complete'; status: string; response: string; timestamp: string }
  | { event: 'error'; error: string; timestamp: string };

@Injectable({ providedIn: 'root' })
export class EngineService {
  private readonly api = 'http://localhost:8001/core';

  constructor(private readonly http: HttpClient) {}

  entry(payload: { message_id: string; user_input: string; model?: string }): Observable<unknown> {
    return this.http.post(`${this.api}`, payload);
  }

  comprehension(payload: { message_id: string; user_input: string; model?: string }): Observable<StepResponse> {
    return this.http.post<StepResponse>(`${this.api}/comprehension`, payload);
  }

  orchestration(payload: {
    message_id: string;
    user_input: string;
    model?: string;
    comprehension_text?: string;
    comprehension_route?: string;
  }): Observable<StepResponse> {
    return this.http.post<StepResponse>(`${this.api}/orchestration`, payload);
  }

  reasoning(payload: {
    message_id: string;
    user_input: string;
    model?: string;
    comprehension_text?: string;
    orchestration_text?: string;
    orchestration_plan?: string[];
  }): Observable<StepResponse> {
    return this.http.post<StepResponse>(`${this.api}/reasoning`, payload);
  }

  evaluation(payload: {
    message_id: string;
    user_input: string;
    model?: string;
    comprehension_text?: string;
    orchestration_text?: string;
    orchestration_plan?: string[];
    reasoning_text?: string;
  }): Observable<StepResponse> {
    return this.http.post<StepResponse>(`${this.api}/evaluation`, payload);
  }

  private sse<T extends StepStreamEvent>(url: string, payload: unknown): Observable<T> {
    return new Observable<T>((observer) => {
      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
        .then((response) => {
          if (!response.ok || !response.body) {
            observer.error(`HTTP ${response.status}: ${response.statusText}`);
            return;
          }
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          const readChunk = () => {
            reader.read().then(({ value, done }) => {
              if (done) { observer.complete(); return; }
              const text = decoder.decode(value, { stream: true });
              for (const part of text.split('\n\n')) {
                const trimmed = part.trim();
                if (!trimmed.startsWith('data:')) continue;
                const jsonStr = trimmed.replace(/^data:\s*/, '');
                if (!jsonStr || jsonStr === '[DONE]') continue;
                observer.next(JSON.parse(jsonStr));
              }
              readChunk();
            });
          };
          readChunk();
        })
        .catch((err) => observer.error(err));
    });
  }

  comprehensionStream(payload: { message_id: string; user_input: string; model?: string }): Observable<StepStreamEvent> {
    return this.sse<StepStreamEvent>(`${this.api}/comprehension/stream`, payload);
  }

  orchestrationStream(payload: {
    message_id: string; user_input: string; model?: string;
    comprehension_text?: string; comprehension_route?: string;
  }): Observable<StepStreamEvent> {
    return this.sse<StepStreamEvent>(`${this.api}/orchestration/stream`, payload);
  }

  reasoningStream(payload: {
    message_id: string; user_input: string; model?: string;
    comprehension_text?: string; orchestration_text?: string; orchestration_plan?: string[];
  }): Observable<StepStreamEvent> {
    return this.sse<StepStreamEvent>(`${this.api}/reasoning/stream`, payload);
  }

  evaluationStream(payload: {
    message_id: string; user_input: string; model?: string;
    comprehension_text?: string; orchestration_text?: string; orchestration_plan?: string[]; reasoning_text?: string;
  }): Observable<StepStreamEvent> {
    return this.sse<StepStreamEvent>(`${this.api}/evaluation/stream`, payload);
  }

  // ==========================================
  // Unified CORE Engine API
  // ==========================================

  /**
   * Run the complete CORE pipeline (Comprehension → Orchestration → Reasoning → Evaluation → Conversation).
   * This executes the full graph synchronously and returns the final state.
   */
  runCore(request: RunRequest): Observable<RunResponse> {
    return this.http.post<RunResponse>('http://localhost:8001/engine/run', request);
  }

  /**
   * Get the current state of a CORE execution run.
   */
  getRunState(runId: string): Observable<COREState> {
    return this.http.get<COREState>(`http://localhost:8001/engine/runs/${runId}`);
  }

  /**
   * Stream real-time execution updates for a CORE run via Server-Sent Events.
   * This provides visibility into each node as it executes.
   */
  streamCoreExecution(runId: string, userInput: string): Observable<COREStreamEvent> {
    return new Observable<COREStreamEvent>((observer) => {
      const url = `http://localhost:8001/engine/runs/${runId}/stream?user_input=${encodeURIComponent(userInput)}`;
      fetch(url, { method: 'GET' })
        .then((response) => {
          if (!response.ok || !response.body) {
            observer.error(`HTTP ${response.status}: ${response.statusText}`);
            return;
          }
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          const readChunk = () => {
            reader.read().then(({ value, done }) => {
              if (done) {
                observer.complete();
                return;
              }
              const text = decoder.decode(value, { stream: true });
              for (const part of text.split('\n\n')) {
                const trimmed = part.trim();
                if (!trimmed.startsWith('data:')) continue;
                const jsonStr = trimmed.replace(/^data:\s*/, '');
                if (!jsonStr || jsonStr === '[DONE]') continue;
                try {
                  const event = JSON.parse(jsonStr) as COREStreamEvent;
                  observer.next(event);
                } catch (e) {
                  console.warn('Failed to parse SSE event:', jsonStr, e);
                }
              }
              readChunk();
            });
          };
          readChunk();
        })
        .catch((err) => observer.error(err));
    });
  }

  /**
   * Delete a completed run from memory.
   */
  deleteRun(runId: string): Observable<{ message: string }> {
    return this.http.delete<{ message: string }>(`http://localhost:8001/engine/runs/${runId}`);
  }

  /**
   * List all active CORE runs.
   */
  listRuns(): Observable<{ total_runs: number; runs: Record<string, any> }> {
    return this.http.get<{ total_runs: number; runs: Record<string, any> }>('http://localhost:8001/engine/runs');
  }
}


