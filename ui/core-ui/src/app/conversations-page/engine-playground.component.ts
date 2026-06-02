import { Component, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { AppConfigService } from '../services/config/app-config.service';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatDividerModule } from '@angular/material/divider';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatSelectModule } from '@angular/material/select';
import { MatOptionModule } from '@angular/material/core';
import { EngineService, StepResponse, StepStreamEvent, COREStreamEvent, COREState } from '../services/engine/engine.service';

@Component({
  selector: 'app-engine-playground',
  standalone: true,
  templateUrl: './engine-playground.component.html',
  styleUrls: ['./engine-playground.component.scss'],
  imports: [
    CommonModule, FormsModule,
    MatButtonModule, MatCardModule, MatIconModule,
    MatFormFieldModule, MatInputModule, MatDividerModule,
    MatProgressSpinnerModule, MatChipsModule, MatTooltipModule, MatExpansionModule, MatSelectModule, MatOptionModule
  ]
})
export class EnginePlaygroundComponent implements OnDestroy {
  // RSI TODO: Add explicit `public`/`private` modifiers for all fields/methods; prefix private with `_`.
  // RSI TODO: Persist per-step model selections to local storage or user settings service.
  // RSI TODO: Add cancel/abort support for in-flight streams; expose an unsubscribe/stop action per step.
  // RSI TODO: Drive visible CORE steps from backend capabilities schema to avoid duplication.
  public inputText = '';
  public isBusy = false;

  public readonly steps = ['Comprehension', 'Orchestration', 'Reasoning', 'Evaluation'] as const;
  public activeStepIndex = 0;
  public durations: Record<string, number> = {};
  public stepBusy: Record<string, boolean> = {
    Comprehension: false,
    Orchestration: false,
    Reasoning: false,
    Evaluation: false
  };
  public metricsByStep: Record<
  string,
  { tokens: number; ttfb_ms: number; duration_ms: number; tps: number } | undefined
  > = {};

  // Model options are resolved at runtime to match the machine: the active local
  // provider's models (LM Studio / Ollama) are loaded from the backend. The static
  // list is only a fallback for when the local provider can't be reached.
  public models: string[] = [
    'gpt-5', 'gpt-4.1', 'gpt-4o', 'gpt-4o-mini', 'o3-mini'
  ];
  public localProviderLabel = '';
  public modelByStep: Record<'Comprehension' | 'Orchestration' | 'Reasoning' | 'Evaluation', string> = {
    Comprehension: 'gpt-5',
    Orchestration: 'gpt-5',
    Reasoning: 'gpt-5',
    Evaluation: 'gpt-5'
  };

  comprehension?: StepResponse;
  orchestration?: StepResponse;
  reasoning?: StepResponse;
  evaluation?: StepResponse;

  private _subs: Partial<Record<'Comprehension' | 'Orchestration' | 'Reasoning' | 'Evaluation', import('rxjs').Subscription>> = {};

  // Unified CORE execution state
  public coreRunning = false;
  public coreRunId?: string;
  public coreEvents: COREStreamEvent[] = [];
  public coreState?: COREState;
  public currentCoreNode?: string;
  public coreStartTime?: number;
  public coreElapsedMs = 0;
  private _coreSub?: import('rxjs').Subscription;

  constructor(
    private readonly engine: EngineService,
    private readonly http: HttpClient,
    private readonly cfg: AppConfigService
  ) {
    try {
      const saved = window.localStorage.getItem('engine.models');
      if (saved) {
        const parsed = JSON.parse(saved);
        this.modelByStep = { ...this.modelByStep, ...parsed };
      }
    } catch { /* ignore */ }

    this._loadModels();
  }

  /**
   * Load the active local provider's models so the per-stage pickers match the
   * machine (e.g. LM Studio's loaded models) instead of a hardcoded OpenAI list.
   * Any per-stage selection that is no longer available is reset to a valid one.
   */
  private _loadModels(): void {
    const base = this.cfg.apiBaseUrl;
    this.http
      .get<{ provider?: string; models: Array<{ name: string }> }>(`${base}/local-llm/models`)
      .subscribe({
        next: (res) => {
          const names = (res.models || []).map((m) => m.name).filter(Boolean);
          if (!names.length) { return; }
          this.models = names;
          this.localProviderLabel = res.provider === 'lmstudio' ? 'LM Studio'
            : res.provider === 'ollama' ? 'Ollama' : '';
          const fallback = names[0];
          (Object.keys(this.modelByStep) as Array<keyof typeof this.modelByStep>).forEach((step) => {
            if (!names.includes(this.modelByStep[step])) {
              this.modelByStep[step] = fallback;
            }
          });
          this._persistModels();
        },
        error: () => { /* keep the static fallback list */ }
      });
  }

  private _persistModels() {
    try { window.localStorage.setItem('engine.models', JSON.stringify(this.modelByStep)); } catch { /* ignore */ }
  }

  private _payload() { return { message_id: crypto.randomUUID(), user_input: this.inputText }; }
  private _markStart(step: string) {
    (this as any)._t0 = performance.now();
    this.isBusy = true;
    this.stepBusy[step] = true;
    this.activeStepIndex = this.steps.indexOf(step as any) ?? 0;
  }
  private _markEnd(step: string) {
    this.isBusy = false;
    this.stepBusy[step] = false;
    const t0 = (this as any)._t0 as number | undefined;
    if (t0) {
      this.durations[step] = Math.max(0, performance.now() - t0);
    }
  }

  public setActive(index: number) { this.activeStepIndex = index; }
  public runNext() { const step = this.steps[this.activeStepIndex] ?? this.steps[0]; this.runStep(step as any); }
  public runStep(step: 'Comprehension' | 'Orchestration' | 'Reasoning' | 'Evaluation') {
    switch (step) {
      case 'Comprehension': this.runComprehension(); break;
      case 'Orchestration': this.runOrchestration(); break;
      case 'Reasoning': this.runReasoning(); break;
      case 'Evaluation': this.runEvaluation(); break;
    }
  }

  runComprehension() {
    this._markStart('Comprehension');
    this.comprehension = { step: 'Comprehension', text: '' } as StepResponse;
    this._persistModels();
    this._subs['Comprehension'] = this.engine.comprehensionStream({ ...this._payload(), model: this.modelByStep.Comprehension }).subscribe({
      next: (evt: StepStreamEvent) => {
        if (evt.type === 'chunk') {
          this.comprehension!.text += evt.text;
        } else if (evt.type === 'metrics') {
          this.durations['Comprehension'] = evt.duration_ms;
          const tps = evt.duration_ms > 0 ? evt.tokens / (evt.duration_ms / 1000) : 0;
          this.metricsByStep['Comprehension'] = { tokens: evt.tokens, ttfb_ms: evt.ttfb_ms, duration_ms: evt.duration_ms, tps };
        }
      },
      complete: () => { this._markEnd('Comprehension'); this.activeStepIndex = 1; },
      error: () => this._markEnd('Comprehension')
    });
  }

  runOrchestration() {
    this._markStart('Orchestration');
    this.orchestration = { step: 'Orchestration', text: '' } as StepResponse;
    this._persistModels();
    this._subs['Orchestration'] = this.engine.orchestrationStream({
      ...this._payload(),
      model: this.modelByStep.Orchestration,
      comprehension_text: this.comprehension?.text,
      comprehension_route: this.comprehension?.routing_decision
    }).subscribe({
      next: (evt: StepStreamEvent) => {
        if (evt.type === 'chunk') {
          this.orchestration!.text += evt.text;
        } else if (evt.type === 'metrics') {
          this.durations['Orchestration'] = evt.duration_ms;
          const tps = evt.duration_ms > 0 ? evt.tokens / (evt.duration_ms / 1000) : 0;
          this.metricsByStep['Orchestration'] = { tokens: evt.tokens, ttfb_ms: evt.ttfb_ms, duration_ms: evt.duration_ms, tps };
        }
      },
      complete: () => { this._markEnd('Orchestration'); this.activeStepIndex = 2; },
      error: () => this._markEnd('Orchestration')
    });
  }

  runReasoning() {
    this._markStart('Reasoning');
    this.reasoning = { step: 'Reasoning', text: '' } as StepResponse;
    this._persistModels();
    this._subs['Reasoning'] = this.engine.reasoningStream({
      ...this._payload(),
      model: this.modelByStep.Reasoning,
      comprehension_text: this.comprehension?.text,
      orchestration_text: this.orchestration?.text,
      orchestration_plan: this.orchestration?.plan
    }).subscribe({
      next: (evt: StepStreamEvent) => {
        if (evt.type === 'chunk') {
          this.reasoning!.text += evt.text;
        } else if (evt.type === 'metrics') {
          this.durations['Reasoning'] = evt.duration_ms;
          const tps = evt.duration_ms > 0 ? evt.tokens / (evt.duration_ms / 1000) : 0;
          this.metricsByStep['Reasoning'] = { tokens: evt.tokens, ttfb_ms: evt.ttfb_ms, duration_ms: evt.duration_ms, tps };
        }
      },
      complete: () => { this._markEnd('Reasoning'); this.activeStepIndex = 3; },
      error: () => this._markEnd('Reasoning')
    });
  }

  runEvaluation() {
    this._markStart('Evaluation');
    this.evaluation = { step: 'Evaluation', text: '' } as StepResponse;
    this._persistModels();
    this._subs['Evaluation'] = this.engine.evaluationStream({
      ...this._payload(),
      model: this.modelByStep.Evaluation,
      comprehension_text: this.comprehension?.text,
      orchestration_text: this.orchestration?.text,
      orchestration_plan: this.orchestration?.plan,
      reasoning_text: this.reasoning?.text
    }).subscribe({
      next: (evt: StepStreamEvent) => {
        if (evt.type === 'chunk') {
          this.evaluation!.text += evt.text;
        } else if (evt.type === 'metrics') {
          this.durations['Evaluation'] = evt.duration_ms;
          const tps = evt.duration_ms > 0 ? evt.tokens / (evt.duration_ms / 1000) : 0;
          this.metricsByStep['Evaluation'] = { tokens: evt.tokens, ttfb_ms: evt.ttfb_ms, duration_ms: evt.duration_ms, tps };
        }
      },
      complete: () => this._markEnd('Evaluation'),
      error: () => this._markEnd('Evaluation')
    });
  }

  /**
   * Run the complete unified CORE pipeline with real-time SSE streaming.
   * This executes Comprehension → Orchestration → Reasoning → Evaluation → Conversation in one flow.
   *
   * Streams real per-node graph execution from GET /engine/runs/{run_id}/stream
   * (the backend creates the run from user_input on first connect), then fetches
   * the final COREState once the stream completes.
   */
  public runFullCORE() {
    if (!this.inputText.trim()) {
      return;
    }

    // Reset state
    this.coreRunning = true;
    this.coreEvents = [];
    this.coreState = undefined;
    this.currentCoreNode = 'Starting...';
    this.coreStartTime = performance.now();
    this.coreElapsedMs = 0;

    // Generate run ID and stream execution
    const runId = crypto.randomUUID();
    this.coreRunId = runId;

    // Stream the execution (backend will create state and execute CORE graph)
    this._coreSub = this.engine.streamCoreExecution(runId, this.inputText).subscribe({
      next: (event: COREStreamEvent) => {
        this.coreEvents.push(event);
        this.coreElapsedMs = performance.now() - (this.coreStartTime ?? performance.now());

        // Update current node based on event
        if (event.event === 'node_start') {
          this.currentCoreNode = event.node;
        } else if (event.event === 'complete') {
          this.currentCoreNode = 'COMPLETE';
        } else if (event.event === 'error') {
          this.currentCoreNode = 'ERROR';
        }
      },
      complete: () => {
        this.coreRunning = false;
        this.coreElapsedMs = performance.now() - (this.coreStartTime ?? performance.now());

        // Fetch final state
        if (this.coreRunId) {
          this.engine.getRunState(this.coreRunId).subscribe({
            next: (state) => {
              this.coreState = state;
            },
            error: (err) => {
              console.error('Failed to fetch final CORE state:', err);
            }
          });
        }
      },
      error: (err) => {
        console.error('CORE execution stream error:', err);
        this.coreRunning = false;
        this.currentCoreNode = 'ERROR';
        this.coreEvents.push({
          event: 'error',
          error: err?.message || String(err),
          timestamp: new Date().toISOString()
        });
      }
    });
  }

  /**
   * Stop the currently running CORE execution.
   */
  public stopCORE() {
    if (this._coreSub) {
      this._coreSub.unsubscribe();
      this._coreSub = undefined;
    }
    this.coreRunning = false;
  }

  ngOnDestroy(): void {
    // Tear down any in-flight SSE streams so their fetch readers don't leak
    // when the user navigates away mid-run.
    this.stopCORE();
    Object.values(this._subs).forEach((sub) => sub?.unsubscribe());
    this._subs = {};
  }

  /**
   * Get a human-readable label for a CORE event.
   */
  public getCoreEventLabel(event: COREStreamEvent): string {
    switch (event.event) {
      case 'start':
        return `🚀 Started CORE execution (${event.run_id})`;

      case 'node_start':
        return `▶️ ${event.node} started`;

      case 'node_complete':
        return `✅ ${event.node} completed`;

      case 'intent_classified':
        return `🎯 Intent: ${event.intent_type} (${(event.confidence * 100).toFixed(0)}% confidence)`;

      case 'plan_created': {
        let label = `📋 Plan: ${event.goal}\n`;
        label += `   Reasoning: ${event.reasoning}\n`;
        label += `   Steps (${event.steps_count}):\n`;
        event.steps.forEach((step, i) => {
          const hitl = step.requires_hitl ? ' [HITL]' : '';
          const tool = step.tool ? ` [${step.tool}]` : '';
          label += `   ${i + 1}. ${step.name}${tool}${hitl}\n`;
          label += `      ${step.description}\n`;
        });
        return label;
      }

      case 'step_executed': {
        let label = `⚙️ Step: ${event.step_id} → ${event.status} (${event.duration_seconds?.toFixed(2) ?? '?'}s)\n`;
        if (event.error) {
          label += `   ❌ Error: ${event.error}\n`;
        }
        if (event.outputs && Object.keys(event.outputs).length > 0) {
          label += `   📤 Outputs: ${JSON.stringify(event.outputs, null, 2)}\n`;
        }
        if (event.artifacts?.length > 0) {
          label += `   📎 Artifacts: ${event.artifacts.join(', ')}\n`;
        }
        if (event.logs?.length > 0) {
          label += `   📝 Logs:\n`;
          event.logs.forEach(log => {
            label += `      ${log}\n`;
          });
        }
        return label;
      }

      case 'evaluation_complete':
        return `📊 Evaluation: ${event.overall_status} (quality: ${(event.quality_score * 100).toFixed(0)}%, confidence: ${(event.confidence * 100).toFixed(0)}%)`;

      case 'complete':
        return `🎉 Complete: ${event.response}`;

      case 'error':
        return `❌ Error: ${event.error}`;

      default:
        return JSON.stringify(event);
    }
  }

  public clear() {
    this.comprehension = this.orchestration = this.reasoning = this.evaluation = undefined;
    this.durations = {};
    this.coreEvents = [];
    this.coreState = undefined;
    this.currentCoreNode = undefined;
    this.coreRunId = undefined;
  }

  public copy(text: string) {
    if (navigator?.clipboard && text) {
      navigator.clipboard.writeText(text).catch(() => {});
    }
  }

  public stop(step: 'Comprehension' | 'Orchestration' | 'Reasoning' | 'Evaluation') {
    const sub = this._subs[step];
    if (sub) {
      try { sub.unsubscribe(); } catch { /* ignore */ }
      this.stepBusy[step] = false;
      if (!this.stepBusy['Comprehension'] && !this.stepBusy['Orchestration'] && !this.stepBusy['Reasoning'] && !this.stepBusy['Evaluation']) {
        this.isBusy = false;
      }
    }
  }
}


