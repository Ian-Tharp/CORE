import { ComponentFixture, TestBed } from '@angular/core/testing';

import { WorldCardComponent } from './world-card.component';

describe('WorldCardComponent', () => {
  let component: WorldCardComponent;
  let fixture: ComponentFixture<WorldCardComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WorldCardComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(WorldCardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
