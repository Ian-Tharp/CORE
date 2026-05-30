import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ReactorCoreComponent } from './reactor-core.component';

describe('ReactorCoreComponent', () => {
  let fixture: ComponentFixture<ReactorCoreComponent>;

  beforeEach(async () => {
    // Arrange
    await TestBed.configureTestingModule({
      imports: [ReactorCoreComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(ReactorCoreComponent);
  });

  it('should render the booting label before a status arrives', () => {
    // Arrange
    fixture.componentRef.setInput('status', 'initializing');

    // Act
    fixture.detectChanges();

    // Assert
    expect(fixture.nativeElement.textContent).toContain('BOOTING');
  });

  it('should render ONLINE and a live feed when healthy and streaming', () => {
    // Arrange
    fixture.componentRef.setInput('status', 'healthy');
    fixture.componentRef.setInput('online', true);

    // Act
    fixture.detectChanges();

    // Assert
    expect(fixture.nativeElement.textContent).toContain('ONLINE');
    expect(fixture.nativeElement.textContent).toContain('LIVE FEED');
  });

  it('should shorten the energy dash offset as load increases', () => {
    // Arrange
    fixture.componentRef.setInput('load', 0);
    fixture.detectChanges();
    const lowLoadOffset = fixture.componentInstance.energyDashOffset();

    // Act
    fixture.componentRef.setInput('load', 90);
    fixture.detectChanges();
    const highLoadOffset = fixture.componentInstance.energyDashOffset();

    // Assert
    expect(highLoadOffset).toBeLessThan(lowLoadOffset);
  });
});
