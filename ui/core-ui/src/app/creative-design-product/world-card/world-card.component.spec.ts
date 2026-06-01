import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { WorldCardComponent } from './world-card.component';
import type { HexWorldSnapshot } from '../../landing-page/command-center/engine/project.service';

describe('WorldCardComponent', () => {
  let component: WorldCardComponent;
  let fixture: ComponentFixture<WorldCardComponent>;

  const world: HexWorldSnapshot = {
    id: 'world-1',
    name: 'Test World',
    createdAt: new Date('2026-01-01T00:00:00.000Z').toISOString(),
    config: { cellRadius: 1 } as HexWorldSnapshot['config'],
    layers: {
      terrain: [],
      biome: [],
      resources: []
    }
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WorldCardComponent],
      providers: [provideRouter([])]
    })
      .compileComponents();

    fixture = TestBed.createComponent(WorldCardComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('world', world);
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
