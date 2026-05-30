import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VitalsRingComponent } from './vitals-ring.component';

describe('VitalsRingComponent', () => {
  let fixture: ComponentFixture<VitalsRingComponent>;

  beforeEach(async () => {
    // Arrange
    await TestBed.configureTestingModule({
      imports: [VitalsRingComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(VitalsRingComponent);
  });

  it('should clamp values into the 0-100 range', () => {
    // Arrange
    fixture.componentRef.setInput('value', 142);

    // Act
    fixture.detectChanges();

    // Assert
    expect(fixture.componentInstance.clamped()).toBe(100);
  });

  it('should flag critical severity at high utilisation', () => {
    // Arrange
    fixture.componentRef.setInput('value', 91);

    // Act
    fixture.detectChanges();

    // Assert
    expect(fixture.componentInstance.severity()).toBe('critical');
  });

  it('should render the label and detail', () => {
    // Arrange
    fixture.componentRef.setInput('label', 'CPU');
    fixture.componentRef.setInput('detail', '4 cores');

    // Act
    fixture.detectChanges();

    // Assert
    expect(fixture.nativeElement.textContent).toContain('CPU');
    expect(fixture.nativeElement.textContent).toContain('4 cores');
  });
});
