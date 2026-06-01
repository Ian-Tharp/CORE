import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DiscordBridgeStatusBadgeComponent } from './discord-bridge-status-badge.component';

describe('DiscordBridgeStatusBadgeComponent', () => {
  let component: DiscordBridgeStatusBadgeComponent;
  let fixture: ComponentFixture<DiscordBridgeStatusBadgeComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DiscordBridgeStatusBadgeComponent]
    }).compileComponents();

    fixture = TestBed.createComponent(DiscordBridgeStatusBadgeComponent);
    component = fixture.componentInstance;
  });

  it('should render a labeled chip for the provided indicator', () => {
    // Arrange
    component.indicator = {
      label: 'Connected',
      tone: 'ready',
      detail: '1 mapped channel ready',
      tooltip: 'Discord Gateway connected.',
      status: null
    };

    // Act
    fixture.detectChanges();

    // Assert
    expect(fixture.nativeElement.textContent).toContain('Connected');
    expect(fixture.nativeElement.querySelector('.status-chip.ready')).not.toBeNull();
  });

  it('should render a compact dot variant when requested', () => {
    // Arrange
    component.variant = 'dot';
    component.indicator = {
      label: 'Disconnected',
      tone: 'attention',
      detail: 'Bridge is not connected.',
      tooltip: 'Discord Gateway is disconnected.',
      status: null
    };

    // Act
    fixture.detectChanges();

    // Assert
    expect(fixture.nativeElement.querySelector('.status-dot.attention')).not.toBeNull();
    expect(fixture.nativeElement.querySelector('.status-chip')).toBeNull();
  });
});
