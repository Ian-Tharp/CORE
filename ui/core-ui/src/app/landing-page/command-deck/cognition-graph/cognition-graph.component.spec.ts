import { ComponentFixture, TestBed } from '@angular/core/testing';

import {
  CognitionGraphComponent,
  CognitionStage,
  StageState
} from './cognition-graph.component';

/** Deterministic four-stage pipeline fixture used across the layout tests. */
function pipeline(states: readonly StageState[]): CognitionStage[] {
  const ids = ['comprehension', 'orchestration', 'reasoning', 'evaluation'] as const;
  return ids.map((id, i) => ({
    id,
    label: id[0].toUpperCase() + id.slice(1),
    state: states[i] ?? 'idle'
  }));
}

describe('CognitionGraphComponent', () => {
  let fixture: ComponentFixture<CognitionGraphComponent>;
  let component: CognitionGraphComponent;

  beforeEach(async () => {
    // Arrange
    await TestBed.configureTestingModule({
      imports: [CognitionGraphComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(CognitionGraphComponent);
    component = fixture.componentInstance;
  });

  it('should map each stage state to its theming class', () => {
    // Assert — pure state -> class mapping (sentinel values)
    expect(component.stateClassFor('idle')).toBe('is-idle');
    expect(component.stateClassFor('thinking')).toBe('is-thinking');
    expect(component.stateClassFor('executing')).toBe('is-executing');
    expect(component.stateClassFor('error')).toBe('is-error');
    expect(component.stateClassFor('complete')).toBe('is-complete');
  });

  it('should treat only thinking/executing as busy', () => {
    // Assert
    expect(component.isBusy('thinking')).toBe(true);
    expect(component.isBusy('executing')).toBe(true);
    expect(component.isBusy('idle')).toBe(false);
    expect(component.isBusy('error')).toBe(false);
    expect(component.isBusy('complete')).toBe(false);
  });

  it('should lay out four nodes left-to-right with ascending centre x', () => {
    // Arrange
    fixture.componentRef.setInput('stages', pipeline(['idle', 'idle', 'idle', 'idle']));

    // Act
    fixture.detectChanges();
    const nodes = component.nodes();

    // Assert
    expect(nodes.length).toBe(4);
    expect(nodes[0].cx).toBeLessThan(nodes[1].cx);
    expect(nodes[1].cx).toBeLessThan(nodes[2].cx);
    expect(nodes[2].cx).toBeLessThan(nodes[3].cx);
    // All share the vertical mid-row.
    expect(new Set(nodes.map((n) => n.cy)).size).toBe(1);
  });

  it('should derive an active edge once both adjacent stages have participated', () => {
    // Arrange — comprehension thinking -> orchestration executing -> reasoning idle
    fixture.componentRef.setInput(
      'stages',
      pipeline(['thinking', 'executing', 'idle', 'idle'])
    );

    // Act
    fixture.detectChanges();
    const edges = component.edges();

    // Assert
    expect(edges.length).toBe(3);
    // comprehension(non-idle) -> orchestration(non-idle) => active
    expect(edges[0].active).toBe(true);
    // orchestration(non-idle) -> reasoning(idle) => not active
    expect(edges[1].active).toBe(false);
    // reasoning(idle) -> evaluation(idle) => not active
    expect(edges[2].active).toBe(false);
  });

  it('should anchor edges to the node perimeter, not the centres', () => {
    // Arrange
    fixture.componentRef.setInput('stages', pipeline(['idle', 'idle', 'idle', 'idle']));

    // Act
    fixture.detectChanges();
    const nodes = component.nodes();
    const edges = component.edges();

    // Assert — edge leaves the right edge of the upstream node
    expect(edges[0].x1).toBe(nodes[0].cx + component.nodeRadius);
    // and arrives at the left edge of the downstream node
    expect(edges[0].x2).toBe(nodes[1].cx - component.nodeRadius);
  });

  it('should mark the node matching selectedStageId as active', () => {
    // Arrange
    fixture.componentRef.setInput('stages', pipeline(['idle', 'idle', 'idle', 'idle']));
    fixture.componentRef.setInput('selectedStageId', 'reasoning');

    // Act
    fixture.detectChanges();
    const active = component.nodes().filter((n) => n.isActive);

    // Assert
    expect(active.length).toBe(1);
    expect(active[0].stage.id).toBe('reasoning');
  });

  it('should emit the clicked stage via stageSelected', () => {
    // Arrange
    const stages = pipeline(['thinking', 'idle', 'idle', 'idle']);
    fixture.componentRef.setInput('stages', stages);
    fixture.detectChanges();
    let emitted: CognitionStage | undefined;
    component.stageSelected.subscribe((s) => (emitted = s));

    // Act
    component.select(stages[2]);

    // Assert
    expect(emitted).toBeDefined();
    expect(emitted!.id).toBe('reasoning');
  });

  it('should activate a node on Enter and Space but ignore other keys', () => {
    // Arrange
    const stages = pipeline(['idle', 'idle', 'idle', 'idle']);
    const emitted: string[] = [];
    component.stageSelected.subscribe((s) => emitted.push(s.id));

    // Act
    component.onKeydown(new KeyboardEvent('keydown', { key: 'Enter' }), stages[0]);
    component.onKeydown(new KeyboardEvent('keydown', { key: ' ' }), stages[1]);
    component.onKeydown(new KeyboardEvent('keydown', { key: 'Tab' }), stages[2]);

    // Assert
    expect(emitted).toEqual(['comprehension', 'orchestration']);
  });

  it('should render no edges for an empty pipeline', () => {
    // Arrange
    fixture.componentRef.setInput('stages', []);

    // Act
    fixture.detectChanges();

    // Assert
    expect(component.nodes().length).toBe(0);
    expect(component.edges().length).toBe(0);
  });
});
