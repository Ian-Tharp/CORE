import { Component, OnInit, OnDestroy, inject, signal, computed } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSelectModule } from '@angular/material/select';
import { MatSnackBarModule, MatSnackBar } from '@angular/material/snack-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { AsyncPipe, CommonModule } from '@angular/common';
import { ChatWindowComponent } from '../shared/chat-window/chat-window.component';
import { HttpClientModule } from '@angular/common/http';
import { InstanceService } from '../services/instance.service';
import { Subject, takeUntil, combineLatest } from 'rxjs';
import { RouterLink } from '@angular/router';
import { 
  AgentInstanceUI, 
  SystemHealth, 
  ActivityEvent, 
  TaskSummary,
  DashboardState
} from '../models/instance.models';
import { DiscordBridgeService } from '../services/discord-bridge/discord-bridge.service';
import { DiscordBridgeStatusBadgeComponent } from '../shared/discord-bridge-status-badge/discord-bridge-status-badge.component';
import { CountUpDirective } from '../shared/directives/count-up.directive';
import { DashboardStore } from './command-deck/dashboard.store';
import { ReactorCoreComponent, ReactorStatus } from './command-deck/reactor-core/reactor-core.component';
import { VitalsRingComponent } from './command-deck/vitals-ring/vitals-ring.component';
import {
  AgentActivityEvent,
  CognitionStore,
  TaskProgressEvent
} from './command-deck/cognition.store';
import {
  CognitionGraphComponent,
  CognitionStage,
  StageId,
  StageState
} from './command-deck/cognition-graph/cognition-graph.component';

@Component({
  selector: 'app-landing-page',
  imports: [
    AsyncPipe,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatChipsModule,
    MatTooltipModule,
    MatSelectModule,
    MatSnackBarModule,
    MatProgressSpinnerModule,
    ChatWindowComponent,
    CommonModule,
    HttpClientModule,
    RouterLink,
    DiscordBridgeStatusBadgeComponent,
    ReactorCoreComponent,
    VitalsRingComponent,
    CognitionGraphComponent,
    CountUpDirective
  ],
  templateUrl: './landing-page.component.html',
  styleUrl: './landing-page.component.scss'
})
export class LandingPageComponent implements OnInit, OnDestroy {
  private readonly _discordBridgeService = inject(DiscordBridgeService);
  public readonly dashboardStore = inject(DashboardStore);
  public readonly cognitionStore = inject(CognitionStore);
  private readonly destroy$ = new Subject<void>();

  /** Whether the conversational dock (chat) is expanded. */
  public readonly chatExpanded = signal<boolean>(true);

  /** Local deck clock, refreshed once per second for the HUD ticker. */
  public readonly deckClock = signal<string>('--:--:--');
  private _clockTimer: ReturnType<typeof setInterval> | null = null;

  // Dashboard state
  dashboardState: DashboardState = {
    loading: true,
    error: null,
    lastUpdate: ''
  };

  // New real-time data
  public readonly discordBridgeIndicator$ = this._discordBridgeService.indicator$;
  activeAgents: AgentInstanceUI[] = [];
  systemHealth: SystemHealth | null = null;
  recentActivities: ActivityEvent[] = [];
  taskSummary: TaskSummary | null = null;

  // Deploy agent form
  selectedAgentRole = 'researcher';
  availableRoles = ['researcher', 'monitor', 'specialist', 'manager'];
  deploymentLoading = false;

  /** Currently-selected pipeline stage for the cognition graph (UI only). */
  public readonly selectedStage = signal<StageId | null>(null);

  /** Pipeline stages for <app-cognition-graph>, projected from live cognition frames. */
  public readonly pipelineStages = computed<CognitionStage[]>(() => {
    return this._buildPipelineStages(this.cognitionStore.agents(), this.cognitionStore.tasks());
  });

  constructor(
    private readonly instanceService: InstanceService,
    private readonly snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.initializeDashboard();
    this.startPolling();
    this.dashboardStore.start();
    this.cognitionStore.start();
    this._startClock();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.dashboardStore.stop();
    this.cognitionStore.stop();
    if (this._clockTimer) {
      clearInterval(this._clockTimer);
      this._clockTimer = null;
    }
  }

  /** Maps backend health into a reactor status for the centerpiece. */
  public get reactorStatus(): ReactorStatus {
    switch (this.systemHealth?.status) {
      case 'healthy':
        return 'healthy';
      case 'degraded':
        return 'degraded';
      case 'unhealthy':
        return 'unhealthy';
      default:
        return 'initializing';
    }
  }

  /** True while the live resource feed is delivering data. */
  public get isFeedLive(): boolean {
    const status = this.dashboardStore.feedStatus();
    return status === 'live' || status === 'polling';
  }

  /** Short label describing the feed transport for the HUD ticker. */
  public get feedStatusLabel(): string {
    switch (this.dashboardStore.feedStatus()) {
      case 'live':
        return 'SSE LINK';
      case 'polling':
        return 'POLL LINK';
      case 'offline':
        return 'LINK LOST';
      default:
        return 'LINKING…';
    }
  }

  /** Aggregate CPU+memory load (retained for HUD/vitals context). */
  public get aggregateLoad(): number {
    const vitals = this.dashboardStore.vitals();
    const cpu = vitals.find((v) => v.id === 'cpu')?.value ?? 0;
    const memory = vitals.find((v) => v.id === 'memory')?.value ?? 0;
    return Math.round((cpu + memory) / 2);
  }

  /**
   * Cognition load (0-100) driving the reactor energy core, derived from live
   * agent/task cognition events. Reads 0 until the backend emits cognition
   * frames; the reactor floors energy at 0.12 so it never reads as dead.
   */
  public get cognitionLoad(): number {
    return this.cognitionStore.cognitionLoad();
  }

  /** Open the (future) stage detail panel for the selected cognition stage. */
  public onStageSelected(stage: CognitionStage): void {
    this.selectedStage.set(stage.id);
  }

  /** Toggle the conversational dock open/closed. */
  public toggleChat(): void {
    this.chatExpanded.update((open) => !open);
  }

  private _startClock(): void {
    const tick = () => this.deckClock.set(new Date().toLocaleTimeString());
    tick();
    this._clockTimer = setInterval(tick, 1000);
  }

  private _buildPipelineStages(
    agents: readonly AgentActivityEvent[],
    tasks: readonly TaskProgressEvent[]
  ): CognitionStage[] {
    const states: Record<StageId, StageState> = {
      comprehension: 'idle',
      orchestration: 'idle',
      reasoning: 'idle',
      evaluation: 'idle'
    };
    const attachedAgents: Record<StageId, { id: string; label: string }[]> = {
      comprehension: [],
      orchestration: [],
      reasoning: [],
      evaluation: []
    };

    for (const task of tasks) {
      this._applyTaskToPipeline(states, task);
    }

    for (const agent of agents) {
      const stage = this._stageForAgent(agent);
      if (stage) {
        attachedAgents[stage].push({
          id: agent.agent_id,
          label: agent.agent_id.replace(/^agent_/, '')
        });
        states[stage] = this._mergeStageState(states[stage], this._stateForAgent(agent));
      }
    }

    return [
      { id: 'comprehension', label: 'Comprehension', state: states.comprehension, agents: attachedAgents.comprehension },
      { id: 'orchestration', label: 'Orchestration', state: states.orchestration, agents: attachedAgents.orchestration },
      { id: 'reasoning', label: 'Reasoning', state: states.reasoning, agents: attachedAgents.reasoning },
      { id: 'evaluation', label: 'Evaluation', state: states.evaluation, agents: attachedAgents.evaluation }
    ];
  }

  private _applyTaskToPipeline(states: Record<StageId, StageState>, task: TaskProgressEvent): void {
    switch (task.stage) {
      case 'queued':
        states.comprehension = this._mergeStageState(states.comprehension, 'thinking');
        break;
      case 'starting':
        states.comprehension = this._mergeStageState(states.comprehension, 'complete');
        states.orchestration = this._mergeStageState(states.orchestration, 'thinking');
        break;
      case 'processing':
        states.comprehension = this._mergeStageState(states.comprehension, 'complete');
        states.orchestration = this._mergeStageState(states.orchestration, 'complete');
        states.reasoning = this._mergeStageState(states.reasoning, 'executing');
        break;
      case 'finalizing':
        states.comprehension = this._mergeStageState(states.comprehension, 'complete');
        states.orchestration = this._mergeStageState(states.orchestration, 'complete');
        states.reasoning = this._mergeStageState(states.reasoning, 'complete');
        states.evaluation = this._mergeStageState(states.evaluation, 'executing');
        break;
      case 'complete':
        states.comprehension = this._mergeStageState(states.comprehension, 'complete');
        states.orchestration = this._mergeStageState(states.orchestration, 'complete');
        states.reasoning = this._mergeStageState(states.reasoning, 'complete');
        states.evaluation = this._mergeStageState(states.evaluation, 'complete');
        break;
      case 'failed':
      case 'cancelled':
        states.evaluation = this._mergeStageState(states.evaluation, 'error');
        break;
      default:
        break;
    }
  }

  private _stageForAgent(agent: AgentActivityEvent): StageId | null {
    const raw = `${agent.agent_id} ${agent.action} ${agent.message ?? ''}`.toLowerCase();
    if (raw.includes('comprehension')) {return 'comprehension';}
    if (raw.includes('orchestration')) {return 'orchestration';}
    if (raw.includes('reasoning')) {return 'reasoning';}
    if (raw.includes('evaluation')) {return 'evaluation';}
    return null;
  }

  private _stateForAgent(agent: AgentActivityEvent): StageState {
    switch (agent.status) {
      case 'thinking':
        return 'thinking';
      case 'active':
      case 'executing':
        return 'executing';
      case 'complete':
        return 'complete';
      case 'error':
        return 'error';
      default:
        return 'idle';
    }
  }

  private _mergeStageState(current: StageState, next: StageState): StageState {
    const rank: Record<StageState, number> = {
      idle: 0,
      complete: 1,
      thinking: 2,
      executing: 3,
      error: 4
    };
    return rank[next] > rank[current] ? next : current;
  }

  /**
   * Initialize dashboard with initial data load
   */
  private initializeDashboard(): void {
    this.dashboardState.loading = true;
    this.dashboardState.error = null;

    // Load initial data
    combineLatest([
      this.instanceService.getInstances(),
      this.instanceService.getSystemHealth(),
      this.instanceService.getRecentActivities(),
      this.instanceService.getTaskSummary()
    ]).pipe(takeUntil(this.destroy$))
      .subscribe({
        next: ([instances, health, activities, tasks]) => {
          this.activeAgents = instances as AgentInstanceUI[];
          this.systemHealth = health;
          this.recentActivities = activities;
          this.taskSummary = tasks;
          this.dashboardState.loading = false;
          this.dashboardState.lastUpdate = new Date().toLocaleTimeString();
        },
        error: (error) => {
          console.error('Dashboard initialization error:', error);
          this.dashboardState.loading = false;
          this.dashboardState.error = 'Failed to load dashboard data';
        }
      });
  }

  /**
   * Start polling for real-time updates
   */
  private startPolling(): void {
    // Live system resources are streamed via DashboardStore (SSE).
    // Instances/health still poll until their SSE endpoints exist.

    // Start instance polling (every 10 seconds)
    this.instanceService.startInstancePolling()
      .pipe(takeUntil(this.destroy$))
      .subscribe(instances => {
        this.activeAgents = instances;
        this.dashboardState.lastUpdate = new Date().toLocaleTimeString();
      });

    // Start system health polling (every 30 seconds)
    this.instanceService.startSystemHealthPolling()
      .pipe(takeUntil(this.destroy$))
      .subscribe(health => {
        this.systemHealth = health;
      });

    // Subscribe to activities
    this.instanceService.activities$
      .pipe(takeUntil(this.destroy$))
      .subscribe(activities => {
        this.recentActivities = activities;
      });
  }

  /**
   * Deploy a new agent instance
   */
  deployAgent(): void {
    if (!this.selectedAgentRole) {
      this.snackBar.open('Please select an agent role', 'Close', { duration: 3000 });
      return;
    }

    this.deploymentLoading = true;
    
    this.instanceService.spawnInstance({
      agent_role: this.selectedAgentRole
    }).pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.deploymentLoading = false;
          if (response.success) {
            this.snackBar.open(
              `${this.selectedAgentRole} agent deployed successfully!`, 
              'Close', 
              { duration: 3000 }
            );
          } else {
            this.snackBar.open(
              `Failed to deploy agent: ${response.message}`, 
              'Close', 
              { duration: 5000 }
            );
          }
        },
        error: (error) => {
          this.deploymentLoading = false;
          this.snackBar.open('Failed to deploy agent', 'Close', { duration: 3000 });
          console.error('Deploy error:', error);
        }
      });
  }

  /**
   * Stop an agent instance
   */
  stopAgent(instance: AgentInstanceUI): void {
    this.instanceService.stopInstance(instance.id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.snackBar.open(`Agent ${instance.agent_id} stopped`, 'Close', { duration: 3000 });
        },
        error: (error) => {
          this.snackBar.open('Failed to stop agent', 'Close', { duration: 3000 });
          console.error('Stop error:', error);
        }
      });
  }

  /**
   * Restart an agent instance
   */
  restartAgent(instance: AgentInstanceUI): void {
    this.instanceService.restartInstance(instance.id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.snackBar.open(`Agent ${instance.agent_id} restarted`, 'Close', { duration: 3000 });
        },
        error: (error) => {
          this.snackBar.open('Failed to restart agent', 'Close', { duration: 3000 });
          console.error('Restart error:', error);
        }
      });
  }

  /**
   * Run a security scan (system health check)
   */
  runSecurityScan(): void {
    this.instanceService.getSystemHealth()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (health) => {
          this.systemHealth = health;
          const status = health.status === 'healthy' ? 'passed' : 'found issues';
          this.snackBar.open(`Security scan ${status}`, 'Close', { duration: 3000 });
        },
        error: (error) => {
          this.snackBar.open('Security scan failed', 'Close', { duration: 3000 });
          console.error('Security scan error:', error);
        }
      });
  }

  /**
   * Navigate to analytics (placeholder)
   */
  viewAnalytics(): void {
    this.snackBar.open('Analytics view coming soon!', 'Close', { duration: 3000 });
  }

  /**
   * Scroll to deploy panel
   */
  scrollToDeployPanel(): void {
    const element = document.querySelector('.deploy-panel');
    element?.scrollIntoView({ behavior: 'smooth' });
  }

  /**
   * Get system health status color
   */
  getSystemHealthColor(): string {
    if (!this.systemHealth) {return 'var(--core-text-dim)';}

    switch (this.systemHealth.status) {
      case 'healthy': return 'var(--core-success)';
      case 'degraded': return 'var(--core-amber)';
      case 'unhealthy': return 'var(--core-danger)';
      default: return 'var(--core-text-dim)';
    }
  }

  /**
   * Friendly display label for a System Health service key. Special-cases the
   * names that don't title-case cleanly (e.g. the local LLM provider, acronyms);
   * everything else falls back to a humanised key ("task_queue" -> "Task queue").
   */
  private static readonly HEALTH_SERVICE_LABELS: Record<string, string> = {
    lmstudio: 'LM Studio',
    ollama: 'Ollama',
    task_queue: 'Task Queue',
    vector_db: 'Vector DB',
    websocket: 'WebSocket'
  };

  healthServiceLabel(key: string): string {
    return (
      LandingPageComponent.HEALTH_SERVICE_LABELS[key] ?? key.replace(/_/g, ' ')
    );
  }

  /**
   * Get readable agent status
   */
  getAgentStatusText(status: string): string {
    const statusMap: Record<string, string> = {
      'starting': 'Starting',
      'ready': 'Ready',
      'busy': 'Working',
      'stopping': 'Stopping',
      'unhealthy': 'Unhealthy',
      'lost': 'Lost Connection'
    };
    return statusMap[status] || status;
  }

  /**
   * Format activity timestamp for display
   */
  formatActivityTime(timestamp: string): string {
    const now = new Date();
    const activityTime = new Date(timestamp);
    const diffMs = now.getTime() - activityTime.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 1) {return 'Just now';}
    if (diffMins < 60) {return `${diffMins}m ago`;}
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) {return `${diffHours}h ago`;}
    
    return activityTime.toLocaleDateString();
  }
}
