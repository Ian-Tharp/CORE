import { ComponentFixture, TestBed } from '@angular/core/testing';
import { StatusIndicatorComponent, StatusType } from './status-indicator.component';

describe('StatusIndicatorComponent', () => {
  let component: StatusIndicatorComponent;
  let fixture: ComponentFixture<StatusIndicatorComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [StatusIndicatorComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(StatusIndicatorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should default to pending status', () => {
    expect(component.status).toBe('pending');
  });

  it('should apply correct status class', () => {
    component.status = 'completed';
    expect(component.statusClass).toBe('status-completed');

    component.status = 'error';
    expect(component.statusClass).toBe('status-error');
  });

  it('should return correct label for each status', () => {
    const statuses: StatusType[] = ['completed', 'awaiting_hitl', 'pending', 'error', 'running'];
    const expectedLabels = ['Completed', 'Awaiting Input', 'Pending', 'Error', 'Running'];

    statuses.forEach((status, index) => {
      component.status = status;
      expect(component.statusLabel).toBe(expectedLabels[index]);
    });
  });

  it('should hide label by default', () => {
    expect(component.showLabel).toBe(false);
  });

  it('should show label when showLabel is true', () => {
    component.showLabel = true;
    fixture.detectChanges();

    const labelElement = fixture.nativeElement.querySelector('.status-label');
    expect(labelElement).toBeTruthy();
  });

  it('should default to medium size', () => {
    expect(component.size).toBe('md');
  });

  it('should return icon path for each status', () => {
    const statuses: StatusType[] = ['completed', 'awaiting_hitl', 'pending', 'error', 'running'];

    statuses.forEach(status => {
      component.status = status;
      expect(component.iconPath).toBeTruthy();
      expect(component.iconPath.length).toBeGreaterThan(0);
    });
  });

  it('should show pulse ring only for running status', () => {
    component.status = 'pending';
    fixture.detectChanges();
    let pulseRing = fixture.nativeElement.querySelector('.pulse-ring');
    expect(pulseRing).toBeFalsy();

    component.status = 'running';
    fixture.detectChanges();
    pulseRing = fixture.nativeElement.querySelector('.pulse-ring');
    expect(pulseRing).toBeTruthy();
  });
});
