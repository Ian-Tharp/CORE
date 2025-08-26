import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
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
import { EngineService, StepResponse, StepStreamEvent } from '../services/engine/engine.service';

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
export class EnginePlaygroundComponent {
  public inputText = '';
  public isBusy = false;

  public readonly steps = ['Comprehension', 'Orchestration', 'Reasoning', 'Evaluation'] as const;
  public activeStepIndex = 0;
  public durations: Record<string, number> = {};
  public stepBusy: Record<string, boolean> = { Comprehension: false, Orchestration: false, Reasoning: false, Evaluation: false };
  public metricsByStep: Record<string, { tokens: number; ttfb_ms: number; duration_ms: number; tps: number } | undefined> = {};

  public readonly models = [
    'gpt-5', 'gpt-4.1', 'gpt-4o', 'gpt-4o-mini', 'o3-mini', 'claude-3-5'
  ];
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

  constructor(private readonly engine: EngineService) {}

  private _payload() { return { message_id: crypto.randomUUID(), user_input: this.inputText }; }
  private _markStart(step: string) { (this as any)._t0 = performance.now(); this.isBusy = true; this.stepBusy[step] = true; this.activeStepIndex = this.steps.indexOf(step as any) ?? 0; }
  private _markEnd(step: string) { this.isBusy = false; this.stepBusy[step] = false; const t0 = (this as any)._t0 as number | undefined; if (t0) { this.durations[step] = Math.max(0, performance.now() - t0); } }

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
    this.engine.comprehensionStream({ ...this._payload(), model: this.modelByStep.Comprehension }).subscribe({
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
    this.engine.orchestrationStream({
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
    this.engine.reasoningStream({
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
    this.engine.evaluationStream({
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

  public clear() {
    this.comprehension = this.orchestration = this.reasoning = this.evaluation = undefined;
    this.durations = {};
  }

  public copy(text: string) {
    if (navigator?.clipboard && text) {
      navigator.clipboard.writeText(text).catch(() => {});
    }
  }
}


