import { Injectable, OnDestroy, computed, inject, signal } from '@angular/core';
import { Subscription } from 'rxjs';
import {
  WebSocketService,
  WebSocketMessage,
  ConnectionStatus
} from '../../communication/services/websocket.service';

/**
 * Connection state of the live cognition feed, mirroring {@link FeedStatus}
 * in `dashboard.store.ts`. Note there is no `polling` member here: unlike
 * system resources, cognition events have no REST/SSE backfill endpoint, so a
 * polling fallback is not yet possible (see the backend-wiring TODO below).
 */
export type CognitionFeedStatus = 'connecting' | 'live' | 'offline';

// =============================================================================
// Wire-level event contracts (mirror backend/app/models/ws_events.py)
//
// Cognition frames discriminate on `event_type` (NOT the top-level `type`
// field the communication layer uses). Every frame carries the BaseEvent
// envelope: event_type, event_id, timestamp (ISO 8601), optional session_id.
// =============================================================================

/** Discriminator values for cognition events. */
export type CognitionEventType =
  | 'agent_activity'
  | 'task_progress'
  | 'council'
  | 'system'
  | 'notification';

/** Status of an agent activity (backend AgentStatus). */
export type AgentStatus =
  | 'idle'
  | 'active'
  | 'thinking'
  | 'executing'
  | 'waiting'
  | 'error'
  | 'complete';

/** Stage of a task's lifecycle (backend TaskStage). */
export type TaskStage =
  | 'queued'
  | 'starting'
  | 'processing'
  | 'finalizing'
  | 'complete'
  | 'failed'
  | 'cancelled';

/** Types of council deliberation events (backend CouncilEventType). */
export type CouncilEventType =
  | 'session_started'
  | 'perspective_added'
  | 'vote_cast'
  | 'debate_round'
  | 'synthesis_ready'
  | 'session_complete';

/** Severity level for system events (backend SystemLevel). */
export type SystemLevel = 'debug' | 'info' | 'warning' | 'error' | 'critical';

/** Priority level for notifications (backend NotificationPriority). */
export type NotificationPriority = 'low' | 'normal' | 'high' | 'urgent';

/** Envelope shared by every cognition event (backend BaseEvent). */
export interface BaseCognitionEvent {
  readonly event_type: CognitionEventType;
  readonly event_id: string;
  readonly timestamp: string;
  readonly session_id?: string | null;
}

/** Agent activity update (backend AgentActivityEvent). */
export interface AgentActivityEvent extends BaseCognitionEvent {
  readonly event_type: 'agent_activity';
  readonly agent_id: string;
  readonly action: string;
  readonly status: AgentStatus;
  readonly message?: string | null;
  readonly metadata?: Record<string, unknown> | null;
}

/** Task progress update (backend TaskProgressEvent). */
export interface TaskProgressEvent extends BaseCognitionEvent {
  readonly event_type: 'task_progress';
  readonly task_id: string;
  readonly progress_pct: number;
  readonly stage: TaskStage;
  readonly eta_seconds?: number | null;
  readonly message?: string | null;
  readonly current_step?: string | null;
  readonly total_steps?: number | null;
  readonly current_step_num?: number | null;
}

/** Council deliberation update (backend CouncilEvent). */
export interface CouncilEvent extends BaseCognitionEvent {
  readonly event_type: 'council';
  readonly council_session_id: string;
  readonly event: CouncilEventType;
  readonly agent_id?: string | null;
  readonly content?: string | null;
  readonly vote?: string | null;
  readonly confidence?: number | null;
  readonly round_number?: number | null;
  readonly metadata?: Record<string, unknown> | null;
}

/** System-level event (backend SystemEvent). */
export interface SystemCognitionEvent extends BaseCognitionEvent {
  readonly event_type: 'system';
  readonly level: SystemLevel;
  readonly message: string;
  readonly source?: string | null;
  readonly error_code?: string | null;
  readonly details?: Record<string, unknown> | null;
}

/** User-facing notification (backend NotificationEvent). */
export interface NotificationCognitionEvent extends BaseCognitionEvent {
  readonly event_type: 'notification';
  readonly title: string;
  readonly body: string;
  readonly action_url?: string | null;
  readonly priority: NotificationPriority;
  readonly icon?: string | null;
  readonly category?: string | null;
  readonly dismissible: boolean;
  readonly auto_dismiss_ms?: number | null;
}

/** Discriminated union over all cognition events. */
export type CognitionEvent =
  | AgentActivityEvent
  | TaskProgressEvent
  | CouncilEvent
  | SystemCognitionEvent
  | NotificationCognitionEvent;

// =============================================================================
// Projections (UI-facing shapes)
// =============================================================================

/**
 * Aggregated state of a council session, accumulated from the stream of
 * {@link CouncilEvent}s sharing a `council_session_id`.
 */
export interface CouncilPerspective {
  readonly agentId: string | null;
  readonly content: string | null;
  readonly confidence: number | null;
  readonly roundNumber: number | null;
  readonly ts: string;
}

/** A single recorded vote within a council session. */
export interface CouncilVote {
  readonly agentId: string | null;
  readonly vote: string | null;
  readonly confidence: number | null;
  readonly ts: string;
}

/** Current/last council session state synthesized from council events. */
export interface CouncilSessionState {
  readonly sessionId: string;
  readonly phase: CouncilEventType;
  readonly round: number;
  readonly perspectives: readonly CouncilPerspective[];
  readonly votes: readonly CouncilVote[];
  readonly synthesis: string | null;
  readonly complete: boolean;
  readonly ts: string;
}

/** Kind of an activity-feed item, ordered loosely by "lifecycle position". */
export type ActivityKind = 'done' | 'running' | 'blocked' | 'next';

/**
 * Normalized activity-feed row, synthesized from agent activity + task
 * progress for the activity panel.
 */
export interface ActivityFeedItem {
  readonly id: string;
  readonly kind: ActivityKind;
  readonly label: string;
  readonly detail: string;
  readonly agentId?: string;
  readonly ts: string;
}

// =============================================================================
// Pure projection helpers (exported for deterministic unit testing)
// =============================================================================

/** Agent statuses that contribute to "live cognition" / reactor load. */
const ACTIVE_AGENT_STATUSES: ReadonlySet<AgentStatus> = new Set<AgentStatus>([
  'active',
  'thinking',
  'executing'
]);

/** Task stages that count as "in flight" for throughput. */
const RUNNING_TASK_STAGES: ReadonlySet<TaskStage> = new Set<TaskStage>([
  'queued',
  'starting',
  'processing',
  'finalizing'
]);

/** Weight, in load points, contributed by each actively-working agent. */
export const AGENT_LOAD_WEIGHT = 18;

/** Weight, in load points, contributed by each in-flight task. */
export const TASK_LOAD_WEIGHT = 12;

/**
 * Compute the reactor cognition load (0-100) from the current agent and task
 * maps.
 *
 * Formula:
 *   activeAgents = count(agents whose status ∈ {active, thinking, executing})
 *   runningTasks = count(tasks whose stage ∈ {queued, starting, processing, finalizing})
 *   load = clamp(activeAgents * AGENT_LOAD_WEIGHT + runningTasks * TASK_LOAD_WEIGHT, 0, 100)
 *
 * Each working agent adds {@link AGENT_LOAD_WEIGHT} (18) points and each
 * in-flight task adds {@link TASK_LOAD_WEIGHT} (12) points; the sum saturates
 * at 100. This makes a single working agent register meaningfully on the
 * reactor while a fully-loaded pipeline (several agents + tasks) pins it.
 */
export function computeCognitionLoad(
  agents: ReadonlyMap<string, AgentActivityEvent>,
  tasks: ReadonlyMap<string, TaskProgressEvent>
): number {
  let activeAgents = 0;
  for (const a of agents.values()) {
    if (ACTIVE_AGENT_STATUSES.has(a.status)) {
      activeAgents += 1;
    }
  }

  let runningTasks = 0;
  for (const t of tasks.values()) {
    if (RUNNING_TASK_STAGES.has(t.stage)) {
      runningTasks += 1;
    }
  }

  const raw = activeAgents * AGENT_LOAD_WEIGHT + runningTasks * TASK_LOAD_WEIGHT;
  return Math.max(0, Math.min(100, Math.round(raw)));
}

/** Map an agent status onto an activity-feed kind. */
function agentStatusToKind(status: AgentStatus): ActivityKind {
  switch (status) {
    case 'complete':
      return 'done';
    case 'error':
    case 'waiting':
      return 'blocked';
    case 'idle':
      return 'next';
    default:
      // active | thinking | executing
      return 'running';
  }
}

/** Map a task stage onto an activity-feed kind. */
function taskStageToKind(stage: TaskStage): ActivityKind {
  switch (stage) {
    case 'complete':
      return 'done';
    case 'failed':
    case 'cancelled':
      return 'blocked';
    case 'queued':
      return 'next';
    default:
      // starting | processing | finalizing
      return 'running';
  }
}

/**
 * Build the normalized activity feed from the current agent + task maps.
 *
 * Each agent and each task contributes exactly one row (latest-wins is already
 * enforced by the caller's maps). Rows are sorted newest-first by `ts`, with a
 * stable tiebreak on `id` so the projection is deterministic for tests.
 */
export function buildActivityFeed(
  agents: ReadonlyMap<string, AgentActivityEvent>,
  tasks: ReadonlyMap<string, TaskProgressEvent>
): ActivityFeedItem[] {
  const items: ActivityFeedItem[] = [];

  for (const a of agents.values()) {
    items.push({
      id: `agent:${a.agent_id}`,
      kind: agentStatusToKind(a.status),
      label: a.action,
      detail: a.message ?? a.status,
      agentId: a.agent_id,
      ts: a.timestamp
    });
  }

  for (const t of tasks.values()) {
    const stepDetail =
      t.total_steps && t.current_step_num
        ? `${t.current_step_num}/${t.total_steps}`
        : `${t.progress_pct}%`;
    items.push({
      id: `task:${t.task_id}`,
      kind: taskStageToKind(t.stage),
      label: t.current_step ?? t.message ?? t.stage,
      detail: stepDetail,
      ts: t.timestamp
    });
  }

  // Newest first; stable tiebreak on id for deterministic ordering.
  items.sort((x, y) => {
    if (x.ts === y.ts) {
      return x.id < y.id ? -1 : x.id > y.id ? 1 : 0;
    }
    return x.ts < y.ts ? 1 : -1;
  });

  return items;
}

/**
 * Fold a single {@link CouncilEvent} into a session-state accumulator,
 * returning the next immutable state. A `session_started` event (or a first
 * sighting of a new `council_session_id`) resets the accumulator.
 */
export function reduceCouncilEvent(
  prev: CouncilSessionState | null,
  ev: CouncilEvent
): CouncilSessionState {
  const isNewSession =
    prev === null ||
    prev.sessionId !== ev.council_session_id ||
    ev.event === 'session_started';

  const base: CouncilSessionState = isNewSession
    ? {
        sessionId: ev.council_session_id,
        phase: ev.event,
        round: ev.round_number ?? 0,
        perspectives: [],
        votes: [],
        synthesis: null,
        complete: false,
        ts: ev.timestamp
      }
    : { ...prev };

  const perspectives = base.perspectives.slice();
  const votes = base.votes.slice();
  let synthesis = base.synthesis;
  let complete = base.complete;

  switch (ev.event) {
    case 'perspective_added':
      perspectives.push({
        agentId: ev.agent_id ?? null,
        content: ev.content ?? null,
        confidence: ev.confidence ?? null,
        roundNumber: ev.round_number ?? null,
        ts: ev.timestamp
      });
      break;
    case 'vote_cast':
      votes.push({
        agentId: ev.agent_id ?? null,
        vote: ev.vote ?? null,
        confidence: ev.confidence ?? null,
        ts: ev.timestamp
      });
      break;
    case 'synthesis_ready':
      synthesis = ev.content ?? synthesis;
      break;
    case 'session_complete':
      complete = true;
      break;
    default:
      // session_started | debate_round — no list mutation beyond phase/round.
      break;
  }

  return {
    sessionId: base.sessionId,
    phase: ev.event,
    round: ev.round_number ?? base.round,
    perspectives,
    votes,
    synthesis,
    complete,
    ts: ev.timestamp
  };
}

/** Narrowing guard: is this inbound frame a cognition event? */
export function isCognitionEvent(msg: WebSocketMessage): msg is WebSocketMessage & CognitionEvent {
  const t = msg['event_type'];
  return (
    t === 'agent_activity' ||
    t === 'task_progress' ||
    t === 'council' ||
    t === 'system' ||
    t === 'notification'
  );
}

// =============================================================================
// Store
// =============================================================================

const EMPTY_AGENTS: ReadonlyMap<string, AgentActivityEvent> = new Map();
const EMPTY_TASKS: ReadonlyMap<string, TaskProgressEvent> = new Map();

/**
 * Reactive store for the command deck's cognition surface. Consumes the
 * cognition event stream from the existing communication-commons WebSocket
 * (the SAME socket the {@link WebSocketService} multiplexes — this store does
 * NOT open a second raw socket) and exposes it as signals for the reactor and
 * activity panel.
 *
 * Transport notes (verified against backend infra):
 * - Source is `WebSocketService.messages$`. Cognition frames discriminate on
 *   `event_type`, NOT the top-level `type` used by the communication layer, so
 *   we filter on `event_type` directly rather than using `onMessageType()`
 *   (which would silently never fire for cognition payloads).
 * - `feedStatus` is derived from the service's `connectionStatus$`.
 *
 * TODO (backend wiring required before any frames arrive):
 *   The cognition events above are DEFINED (backend/app/models/ws_events.py)
 *   and the EventPublisher → ConnectionManager → /ws/{instance_id} transport is
 *   fully plumbed, but NO product code calls `event_publisher.*` today, so the
 *   live feed will connect successfully yet receive zero
 *   agent_activity/task_progress/council frames. To light this up the backend
 *   must (a) call `event_publisher.publish(...)` from core_graph.py /
 *   council/deliberation_service.py / agent_factory_service.py to emit these
 *   events, and (b) optionally expose a REST snapshot/history endpoint so this
 *   store can backfill on connect (there is currently NO REST/SSE history for
 *   cognition events — publishing is fire-and-forget broadcast — which is why
 *   this store, unlike DashboardStore, has no polling fallback).
 */
@Injectable({ providedIn: 'root' })
export class CognitionStore implements OnDestroy {
  private readonly _ws = inject(WebSocketService);

  /** Latest agent-activity event keyed by `agent_id` (latest-wins). */
  private readonly _agents = signal<ReadonlyMap<string, AgentActivityEvent>>(EMPTY_AGENTS);
  /** Latest task-progress event keyed by `task_id` (latest-wins). */
  private readonly _tasks = signal<ReadonlyMap<string, TaskProgressEvent>>(EMPTY_TASKS);
  /** Current/last council session state. */
  private readonly _council = signal<CouncilSessionState | null>(null);
  /** Connection state of the live feed. */
  private readonly _feedStatus = signal<CognitionFeedStatus>('connecting');

  private _messagesSub: Subscription | null = null;
  private _statusSub: Subscription | null = null;
  private _instanceId: string | null = null;
  private _started = false;

  /** Latest agent-activity events, keyed by `agent_id`. */
  public readonly agentActivity = this._agents.asReadonly();

  /** Latest task-progress events, keyed by `task_id`. */
  public readonly taskProgress = this._tasks.asReadonly();

  /** Current/last council session state, or `null` if none seen yet. */
  public readonly council = this._council.asReadonly();

  /** Connection state of the live cognition feed. */
  public readonly feedStatus = this._feedStatus.asReadonly();

  /** Agent-activity events as a stable array (insertion order). */
  public readonly agents = computed<AgentActivityEvent[]>(() =>
    Array.from(this._agents().values())
  );

  /** Task-progress events as a stable array (insertion order). */
  public readonly tasks = computed<TaskProgressEvent[]>(() =>
    Array.from(this._tasks().values())
  );

  /**
   * Reactor drive signal in [0, 100]. See {@link computeCognitionLoad} for the
   * formula (active agents × 18 + running tasks × 12, clamped to 100).
   */
  public readonly cognitionLoad = computed<number>(() =>
    computeCognitionLoad(this._agents(), this._tasks())
  );

  /**
   * Normalized activity feed for the activity panel. See
   * {@link buildActivityFeed} (one row per agent + task, newest-first).
   */
  public readonly activityFeed = computed<ActivityFeedItem[]>(() =>
    buildActivityFeed(this._agents(), this._tasks())
  );

  /**
   * Begin consuming the cognition feed. Safe to call repeatedly; only the
   * first call connects. Generates and persists an `instanceId` for the WS
   * URL path segment (mirroring how the communication layer supplies one).
   */
  public start(): void {
    if (this._started) {
      return;
    }
    this._started = true;

    this._feedStatus.set('connecting');
    this._subscribeStatus();
    this._subscribeMessages();

    this._instanceId = this._generateInstanceId();
    this._ws.connect(this._instanceId);
  }

  /** Tear down the feed subscriptions. Safe to call repeatedly. */
  public stop(): void {
    this._messagesSub?.unsubscribe();
    this._messagesSub = null;
    this._statusSub?.unsubscribe();
    this._statusSub = null;
    this._started = false;
  }

  public ngOnDestroy(): void {
    this.stop();
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  private _subscribeMessages(): void {
    if (this._messagesSub) {
      return;
    }
    this._messagesSub = this._ws.messages$.subscribe((msg) => {
      if (isCognitionEvent(msg)) {
        this._ingest(msg);
      }
    });
  }

  private _subscribeStatus(): void {
    if (this._statusSub) {
      return;
    }
    this._statusSub = this._ws.connectionStatus$.subscribe((status: ConnectionStatus) => {
      this._feedStatus.set(this._mapFeedStatus(status));
    });
  }

  /** Project the transport connection status onto the cognition feed status. */
  private _mapFeedStatus(status: ConnectionStatus): CognitionFeedStatus {
    switch (status.state) {
      case 'connected':
        return 'live';
      case 'connecting':
      case 'reconnecting':
        return 'connecting';
      default:
        // disconnected | error
        return 'offline';
    }
  }

  /** Route a single cognition event into the appropriate signal. */
  private _ingest(ev: CognitionEvent): void {
    switch (ev.event_type) {
      case 'agent_activity': {
        const next = new Map(this._agents());
        next.set(ev.agent_id, ev);
        this._agents.set(next);
        break;
      }
      case 'task_progress': {
        const next = new Map(this._tasks());
        next.set(ev.task_id, ev);
        this._tasks.set(next);
        break;
      }
      case 'council': {
        this._council.set(reduceCouncilEvent(this._council(), ev));
        break;
      }
      // system / notification: not surfaced as cognition signals here.
      default:
        break;
    }
  }

  /**
   * Generate a per-deck instance id for the WS URL path segment. Uses
   * `crypto.randomUUID` when available with a deterministic fallback for
   * non-secure/test contexts.
   */
  private _generateInstanceId(): string {
    const c = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
    if (c?.randomUUID) {
      return `deck-${c.randomUUID()}`;
    }
    return `deck-${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`;
  }
}
