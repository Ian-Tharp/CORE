import { NgClass, UpperCasePipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';

/** Lifecycle state a single pipeline stage can be in, mapped from backend node state. */
export type StageState = 'idle' | 'thinking' | 'executing' | 'error' | 'complete';

/** Stable identifiers for the four LangGraph cognitive pipeline stages. */
export type StageId = 'comprehension' | 'orchestration' | 'reasoning' | 'evaluation';

/** A single agent currently attached to (working within) a stage. */
export interface StageAgent {
  /** Stable agent identifier. */
  readonly id: string;
  /** Human-readable agent name / role shown on the node. */
  readonly label: string;
}

/** Declarative description of one pipeline stage and its live state. */
export interface CognitionStage {
  readonly id: StageId;
  /** Short display name rendered inside the node. */
  readonly label: string;
  /** Current lifecycle state driving the node glow colour. */
  readonly state: StageState;
  /** Agents attached to this stage (rendered as orbiting chips). */
  readonly agents?: readonly StageAgent[];
}

/** Pre-computed geometry + visual derivations for a single rendered node. */
export interface NodeLayout {
  readonly stage: CognitionStage;
  /** Centre X within the SVG viewBox. */
  readonly cx: number;
  /** Centre Y within the SVG viewBox. */
  readonly cy: number;
  /** State -> CSS theming class (e.g. 'is-thinking'). */
  readonly stateClass: string;
  /** Convenience flag for templates / tests. */
  readonly isActive: boolean;
}

/** Pre-computed geometry for an edge connecting two adjacent nodes. */
export interface EdgeLayout {
  /** Upstream stage id. */
  readonly fromId: StageId;
  /** Downstream stage id. */
  readonly toId: StageId;
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
  /** True when the edge has been activated by the current or latest run. */
  readonly active: boolean;
}

const VIEWBOX_WIDTH = 640;
const VIEWBOX_HEIGHT = 200;
const NODE_RADIUS = 40;
const ROW_Y = VIEWBOX_HEIGHT / 2;

/** A stage is "busy" (actively doing work) in these states. */
const BUSY_STATES: ReadonlySet<StageState> = new Set<StageState>(['thinking', 'executing']);

/**
 * A live node-graph of the CORE LangGraph cognitive pipeline:
 * Comprehension -> Orchestration -> Reasoning -> Evaluation.
 *
 * Purely presentational and signal-input driven (mirrors reactor-core /
 * vitals-ring conventions): the landing page binds store signals to the
 * `stages` input. Renders inline SVG with currentColor theming and emits a
 * `stageSelected` output when a node is activated so a side panel can react.
 */
@Component({
  selector: 'app-cognition-graph',
  standalone: true,
  imports: [NgClass, UpperCasePipe],
  templateUrl: './cognition-graph.component.html',
  styleUrl: './cognition-graph.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class CognitionGraphComponent {
  /** SVG viewBox dimensions (exposed for the template + tests). */
  public readonly viewBoxWidth = VIEWBOX_WIDTH;
  public readonly viewBoxHeight = VIEWBOX_HEIGHT;
  public readonly nodeRadius = NODE_RADIUS;
  public readonly nodeCircumference = 2 * Math.PI * NODE_RADIUS;

  /**
   * The ordered pipeline stages with their live state. Defaults to the four
   * canonical CORE stages in the idle state so the graph renders standalone.
   */
  public readonly stages = input<readonly CognitionStage[]>([
    { id: 'comprehension', label: 'Comprehension', state: 'idle' },
    { id: 'orchestration', label: 'Orchestration', state: 'idle' },
    { id: 'reasoning', label: 'Reasoning', state: 'idle' },
    { id: 'evaluation', label: 'Evaluation', state: 'idle' }
  ]);

  /** Currently focused/selected stage id (drives the selection ring). */
  public readonly selectedStageId = input<StageId | null>(null);

  /** Emits the selected stage when a node is clicked or activated via keyboard. */
  public readonly stageSelected = output<CognitionStage>();

  /**
   * Maps each stage state to a CSS theming class. Pure function of state so it
   * is trivially unit-testable; the SCSS resolves the class to a token colour.
   */
  public stateClassFor(state: StageState): string {
    switch (state) {
      case 'thinking':
        return 'is-thinking';
      case 'executing':
        return 'is-executing';
      case 'error':
        return 'is-error';
      case 'complete':
        return 'is-complete';
      default:
        return 'is-idle';
    }
  }

  /** Whether a stage is actively performing work (used for edge-flow derivation). */
  public isBusy(state: StageState): boolean {
    return BUSY_STATES.has(state);
  }

  /** Pre-computed node geometry + theming, laid out evenly left-to-right. */
  public readonly nodes = computed<readonly NodeLayout[]>(() => {
    const list = this.stages();
    const count = list.length;
    if (count === 0) {
      return [];
    }
    // Evenly distribute centres across the viewBox with symmetric margins.
    const span = VIEWBOX_WIDTH / count;
    const selected = this.selectedStageId();
    return list.map((stage, index) => ({
      stage,
      cx: span * index + span / 2,
      cy: ROW_Y,
      stateClass: this.stateClassFor(stage.state),
      isActive: stage.id === selected
    }));
  });

  /**
   * Pre-computed edges between adjacent nodes. An edge is "active" once both
   * endpoints have participated in a run; this keeps completed spans visibly
   * energized while the currently-working node carries the strongest motion.
   */
  public readonly edges = computed<readonly EdgeLayout[]>(() => {
    const nodes = this.nodes();
    const edges: EdgeLayout[] = [];
    for (let i = 0; i < nodes.length - 1; i++) {
      const from = nodes[i];
      const to = nodes[i + 1];
      edges.push({
        fromId: from.stage.id,
        toId: to.stage.id,
        x1: from.cx + NODE_RADIUS,
        y1: from.cy,
        x2: to.cx - NODE_RADIUS,
        y2: to.cy,
        active: from.stage.state !== 'idle' && to.stage.state !== 'idle'
      });
    }
    return edges;
  });

  /** Click / keyboard activation handler — emits the selected stage. */
  public select(stage: CognitionStage): void {
    this.stageSelected.emit(stage);
  }

  /** Activates a node on Enter/Space and prevents the default scroll/scroll-jump. */
  public onKeydown(event: KeyboardEvent, stage: CognitionStage): void {
    if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {
      event.preventDefault();
      this.select(stage);
    }
  }

  /** trackBy for the node list. */
  public trackNode = (_: number, node: NodeLayout): StageId => node.stage.id;

  /** trackBy for the edge list. */
  public trackEdge = (_: number, edge: EdgeLayout): string => `${edge.fromId}->${edge.toId}`;

  /** trackBy for agent chips. */
  public trackAgent = (_: number, agent: StageAgent): string => agent.id;
}
