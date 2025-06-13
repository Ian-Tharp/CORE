import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AgentMarketplaceComponent } from './agent-marketplace.component';

describe('AgentMarketplaceComponent', () => {
  let component: AgentMarketplaceComponent;
  let fixture: ComponentFixture<AgentMarketplaceComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AgentMarketplaceComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AgentMarketplaceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
