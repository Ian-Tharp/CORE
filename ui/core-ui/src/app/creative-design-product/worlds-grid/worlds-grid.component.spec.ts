import { ComponentFixture, TestBed } from '@angular/core/testing';

import { WorldsGridComponent } from './worlds-grid.component';

describe('WorldsGridComponent', () => {
  let component: WorldsGridComponent;
  let fixture: ComponentFixture<WorldsGridComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [WorldsGridComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(WorldsGridComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
