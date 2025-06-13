import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AgentBuilderComponent } from './agent-builder.component';

describe('AgentBuilderComponent', () => {
  let component: AgentBuilderComponent;
  let fixture: ComponentFixture<AgentBuilderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AgentBuilderComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AgentBuilderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
