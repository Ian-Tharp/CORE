import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';

import {
  CognitionStore,
  AgentActivityEvent,
  TaskProgressEvent,
  CouncilEvent,
  CouncilSessionState,
  AGENT_LOAD_WEIGHT,
  TASK_LOAD_WEIGHT,
  buildActivityFeed,
  computeCognitionLoad,
  reduceCouncilEvent,
  isCognitionEvent
} from './cognition.store';
import {
  WebSocketService,
  WebSocketMessage,
  ConnectionStatus
} from '../../communication/services/websocket.service';

// =============================================================================
// Sentinel builders — deterministic, fully-specified events.
// =============================================================================

function agent(
  id: string,
  status: AgentActivityEvent['status'],
  overrides: Partial<AgentActivityEvent> = {}
): AgentActivityEvent {
  return {
    event_type: 'agent_activity',
    event_id: `evt-${id}`,
    timestamp: '2026-05-29T00:00:00.000Z',
    agent_id: id,
    action: `act-${id}`,
    status,
    message: null,
    metadata: null,
    ...overrides
  };
}

function task(
  id: string,
  stage: TaskProgressEvent['stage'],
  overrides: Partial<TaskProgressEvent> = {}
): TaskProgressEvent {
  return {
    event_type: 'task_progress',
    event_id: `evt-${id}`,
    timestamp: '2026-05-29T00:00:00.000Z',
    task_id: id,
    progress_pct: 50,
    stage,
    eta_seconds: null,
    message: null,
    current_step: null,
    total_steps: null,
    current_step_num: null,
    ...overrides
  };
}

function council(
  event: CouncilEvent['event'],
  overrides: Partial<CouncilEvent> = {}
): CouncilEvent {
  return {
    event_type: 'council',
    event_id: `evt-${event}`,
    timestamp: '2026-05-29T00:00:00.000Z',
    council_session_id: 'sess-1',
    event,
    agent_id: null,
    content: null,
    vote: null,
    confidence: null,
    round_number: null,
    metadata: null,
    ...overrides
  };
}

function mapOf<V extends { agent_id?: string; task_id?: string }>(
  events: V[],
  key: (e: V) => string
): Map<string, V> {
  const m = new Map<string, V>();
  for (const e of events) {
    m.set(key(e), e);
  }
  return m;
}

// =============================================================================
// Pure projection: cognitionLoad
// =============================================================================

describe('computeCognitionLoad', () => {
  it('is 0 with no agents and no tasks', () => {
    expect(computeCognitionLoad(new Map(), new Map())).toBe(0);
  });

  it('counts only active/thinking/executing agents toward load', () => {
    const agents = mapOf(
      [
        agent('a', 'active'),
        agent('b', 'thinking'),
        agent('c', 'executing'),
        agent('d', 'idle'),
        agent('e', 'waiting'),
        agent('f', 'complete'),
        agent('g', 'error')
      ],
      (e) => e.agent_id
    );

    // 3 working agents * 18 = 54
    expect(computeCognitionLoad(agents, new Map())).toBe(3 * AGENT_LOAD_WEIGHT);
  });

  it('counts only in-flight task stages toward load', () => {
    const tasks = mapOf(
      [
        task('t1', 'queued'),
        task('t2', 'starting'),
        task('t3', 'processing'),
        task('t4', 'finalizing'),
        task('t5', 'complete'),
        task('t6', 'failed'),
        task('t7', 'cancelled')
      ],
      (e) => e.task_id
    );

    // 4 in-flight tasks * 12 = 48
    expect(computeCognitionLoad(new Map(), tasks)).toBe(4 * TASK_LOAD_WEIGHT);
  });

  it('sums agent and task contributions', () => {
    const agents = mapOf([agent('a', 'thinking')], (e) => e.agent_id);
    const tasks = mapOf([task('t1', 'processing')], (e) => e.task_id);

    expect(computeCognitionLoad(agents, tasks)).toBe(
      AGENT_LOAD_WEIGHT + TASK_LOAD_WEIGHT
    );
  });

  it('saturates at 100 when the pipeline is fully loaded', () => {
    const agents = mapOf(
      Array.from({ length: 10 }, (_, i) => agent(`a${i}`, 'executing')),
      (e) => e.agent_id
    );
    expect(computeCognitionLoad(agents, new Map())).toBe(100);
  });
});

// =============================================================================
// Pure projection: activityFeed normalization
// =============================================================================

describe('buildActivityFeed', () => {
  it('returns an empty feed for empty maps', () => {
    expect(buildActivityFeed(new Map(), new Map())).toEqual([]);
  });

  it('maps agent statuses onto the right kinds', () => {
    const agents = mapOf(
      [
        agent('done', 'complete'),
        agent('run', 'thinking'),
        agent('block', 'error'),
        agent('wait', 'waiting'),
        agent('next', 'idle')
      ],
      (e) => e.agent_id
    );

    const byId = new Map(
      buildActivityFeed(agents, new Map()).map((i) => [i.id, i.kind])
    );

    expect(byId.get('agent:done')).toBe('done');
    expect(byId.get('agent:run')).toBe('running');
    expect(byId.get('agent:block')).toBe('blocked');
    expect(byId.get('agent:wait')).toBe('blocked');
    expect(byId.get('agent:next')).toBe('next');
  });

  it('maps task stages onto the right kinds', () => {
    const tasks = mapOf(
      [
        task('done', 'complete'),
        task('run', 'processing'),
        task('fail', 'failed'),
        task('cancel', 'cancelled'),
        task('next', 'queued')
      ],
      (e) => e.task_id
    );

    const byId = new Map(
      buildActivityFeed(new Map(), tasks).map((i) => [i.id, i.kind])
    );

    expect(byId.get('task:done')).toBe('done');
    expect(byId.get('task:run')).toBe('running');
    expect(byId.get('task:fail')).toBe('blocked');
    expect(byId.get('task:cancel')).toBe('blocked');
    expect(byId.get('task:next')).toBe('next');
  });

  it('carries agentId on agent rows and omits it on task rows', () => {
    const agents = mapOf([agent('alpha', 'active')], (e) => e.agent_id);
    const tasks = mapOf([task('t1', 'processing')], (e) => e.task_id);
    const feed = buildActivityFeed(agents, tasks);

    const agentRow = feed.find((i) => i.id === 'agent:alpha');
    const taskRow = feed.find((i) => i.id === 'task:t1');

    expect(agentRow?.agentId).toBe('alpha');
    expect(taskRow?.agentId).toBeUndefined();
  });

  it('prefers step counts then percentage for task detail', () => {
    const stepped = mapOf(
      [task('s', 'processing', { current_step_num: 3, total_steps: 7 })],
      (e) => e.task_id
    );
    const pct = mapOf([task('p', 'processing', { progress_pct: 42 })], (e) => e.task_id);

    expect(buildActivityFeed(new Map(), stepped)[0].detail).toBe('3/7');
    expect(buildActivityFeed(new Map(), pct)[0].detail).toBe('42%');
  });

  it('sorts newest-first with a stable id tiebreak on equal timestamps', () => {
    const older = agent('old', 'active', { timestamp: '2026-05-29T00:00:00.000Z' });
    const newer = agent('new', 'active', { timestamp: '2026-05-29T01:00:00.000Z' });
    const tieA = task('aaa', 'processing', { timestamp: '2026-05-29T01:00:00.000Z' });

    const feed = buildActivityFeed(
      mapOf([older, newer], (e) => e.agent_id),
      mapOf([tieA], (e) => e.task_id)
    );

    // Newest first; among the two newest (same ts), 'agent:new' < 'task:aaa'.
    expect(feed.map((i) => i.id)).toEqual(['agent:new', 'task:aaa', 'agent:old']);
  });
});

// =============================================================================
// Pure projection: council reduction
// =============================================================================

describe('reduceCouncilEvent', () => {
  it('initializes a fresh session on session_started', () => {
    const state = reduceCouncilEvent(null, council('session_started', { round_number: 1 }));
    expect(state.sessionId).toBe('sess-1');
    expect(state.phase).toBe('session_started');
    expect(state.round).toBe(1);
    expect(state.perspectives).toEqual([]);
    expect(state.votes).toEqual([]);
    expect(state.synthesis).toBeNull();
    expect(state.complete).toBe(false);
  });

  it('accumulates perspectives and votes across a session', () => {
    let state: CouncilSessionState | null = null;
    state = reduceCouncilEvent(state, council('session_started'));
    state = reduceCouncilEvent(
      state,
      council('perspective_added', { agent_id: 'ethics', content: 'careful', confidence: 0.8 })
    );
    state = reduceCouncilEvent(
      state,
      council('vote_cast', { agent_id: 'ethics', vote: 'approve', confidence: 0.9 })
    );

    expect(state.perspectives).toHaveLength(1);
    expect(state.perspectives[0]).toMatchObject({ agentId: 'ethics', confidence: 0.8 });
    expect(state.votes).toHaveLength(1);
    expect(state.votes[0]).toMatchObject({ agentId: 'ethics', vote: 'approve' });
  });

  it('records synthesis content and completion', () => {
    let state = reduceCouncilEvent(null, council('session_started'));
    state = reduceCouncilEvent(state, council('synthesis_ready', { content: 'final answer' }));
    expect(state.synthesis).toBe('final answer');
    expect(state.complete).toBe(false);

    state = reduceCouncilEvent(state, council('session_complete'));
    expect(state.complete).toBe(true);
    expect(state.synthesis).toBe('final answer');
  });

  it('resets when a new council_session_id arrives', () => {
    let state = reduceCouncilEvent(null, council('perspective_added', { content: 'x' }));
    expect(state.perspectives).toHaveLength(1);

    state = reduceCouncilEvent(
      state,
      council('perspective_added', { council_session_id: 'sess-2', content: 'y' })
    );
    expect(state.sessionId).toBe('sess-2');
    expect(state.perspectives).toHaveLength(1);
    expect(state.perspectives[0].content).toBe('y');
  });

  it('does not mutate the previous state object', () => {
    const start = reduceCouncilEvent(null, council('session_started'));
    const next = reduceCouncilEvent(start, council('perspective_added', { content: 'z' }));
    expect(start.perspectives).toHaveLength(0);
    expect(next.perspectives).toHaveLength(1);
    expect(next).not.toBe(start);
  });
});

// =============================================================================
// Discriminator guard
// =============================================================================

describe('isCognitionEvent', () => {
  it('accepts frames discriminated on event_type', () => {
    expect(isCognitionEvent({ event_type: 'agent_activity' } as WebSocketMessage)).toBe(true);
    expect(isCognitionEvent({ event_type: 'council' } as WebSocketMessage)).toBe(true);
  });

  it('rejects communication frames discriminated on top-level type', () => {
    expect(isCognitionEvent({ type: 'pong' })).toBe(false);
    expect(isCognitionEvent({ type: 'presence', instance_id: 'x' })).toBe(false);
  });

  it('rejects unknown event_type values', () => {
    expect(isCognitionEvent({ event_type: 'totally_unknown' } as WebSocketMessage)).toBe(false);
  });
});

// =============================================================================
// Store wiring — exercised with mocked Subjects (no live socket).
// =============================================================================

describe('CognitionStore', () => {
  let store: CognitionStore;
  let messages$: Subject<WebSocketMessage>;
  let status$: Subject<ConnectionStatus>;
  let ws: { connect: jest.Mock; messages$: Subject<WebSocketMessage>; connectionStatus$: Subject<ConnectionStatus> };

  beforeEach(() => {
    messages$ = new Subject<WebSocketMessage>();
    status$ = new Subject<ConnectionStatus>();
    ws = {
      connect: jest.fn(),
      messages$,
      connectionStatus$: status$
    };

    TestBed.configureTestingModule({
      providers: [CognitionStore, { provide: WebSocketService, useValue: ws }]
    });

    store = TestBed.inject(CognitionStore);
  });

  afterEach(() => {
    store.stop();
  });

  it('starts in the connecting state with empty projections', () => {
    expect(store.feedStatus()).toBe('connecting');
    expect(store.agents()).toEqual([]);
    expect(store.tasks()).toEqual([]);
    expect(store.council()).toBeNull();
    expect(store.cognitionLoad()).toBe(0);
    expect(store.activityFeed()).toEqual([]);
  });

  it('connects exactly once across repeated start calls', () => {
    store.start();
    store.start();
    expect(ws.connect).toHaveBeenCalledTimes(1);
  });

  it('derives feedStatus from the transport connection status', () => {
    store.start();

    status$.next({ state: 'connected' } as ConnectionStatus);
    expect(store.feedStatus()).toBe('live');

    status$.next({ state: 'reconnecting' } as ConnectionStatus);
    expect(store.feedStatus()).toBe('connecting');

    status$.next({ state: 'error' } as ConnectionStatus);
    expect(store.feedStatus()).toBe('offline');
  });

  it('keeps latest-wins per agent_id and task_id from the stream', () => {
    store.start();

    messages$.next(agent('a1', 'thinking') as unknown as WebSocketMessage);
    messages$.next(agent('a1', 'complete') as unknown as WebSocketMessage);
    messages$.next(task('t1', 'queued') as unknown as WebSocketMessage);
    messages$.next(task('t1', 'processing') as unknown as WebSocketMessage);

    expect(store.agentActivity().get('a1')?.status).toBe('complete');
    expect(store.taskProgress().get('t1')?.stage).toBe('processing');
    // one working agent? no (complete) -> 0; one running task -> 12
    expect(store.cognitionLoad()).toBe(TASK_LOAD_WEIGHT);
  });

  it('ignores non-cognition frames (top-level type)', () => {
    store.start();
    messages$.next({ type: 'pong' });
    messages$.next({ type: 'presence', instance_id: 'x' });

    expect(store.agents()).toEqual([]);
    expect(store.tasks()).toEqual([]);
  });

  it('routes council frames into accumulated session state', () => {
    store.start();
    messages$.next(council('session_started') as unknown as WebSocketMessage);
    messages$.next(
      council('perspective_added', { agent_id: 'ethics', content: 'c' }) as unknown as WebSocketMessage
    );

    expect(store.council()?.sessionId).toBe('sess-1');
    expect(store.council()?.perspectives).toHaveLength(1);
  });

  it('stops cleanly so further frames are ignored', () => {
    store.start();
    store.stop();
    messages$.next(agent('a1', 'thinking') as unknown as WebSocketMessage);
    expect(store.agents()).toEqual([]);
  });
});
