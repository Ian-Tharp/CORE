import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AnalyticsPageComponent } from './analytics-page.component';

describe('AnalyticsPageComponent', () => {
  let component: AnalyticsPageComponent;
  let fixture: ComponentFixture<AnalyticsPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AnalyticsPageComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AnalyticsPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
