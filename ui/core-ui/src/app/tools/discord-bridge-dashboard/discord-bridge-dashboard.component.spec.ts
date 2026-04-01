import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { of, throwError } from 'rxjs';

import { DiscordBridgeDashboardComponent } from './discord-bridge-dashboard.component';
import { DiscordBridgeService } from '../../services/discord-bridge/discord-bridge.service';

describe('DiscordBridgeDashboardComponent', () => {
  let component: DiscordBridgeDashboardComponent;
  let fixture: ComponentFixture<DiscordBridgeDashboardComponent>;
  let discordBridgeService: {
    getMetrics: jest.Mock;
    getMappings: jest.Mock;
    getDeliveries: jest.Mock;
    getMessageLinks: jest.Mock;
  };

  beforeEach(async () => {
    discordBridgeService = {
      getMetrics: jest.fn().mockReturnValue(of({
        status: {
          status: 'connected',
          connected: true,
          connected_at: null,
          last_error: null,
          reconnect_attempts: 0,
          bot_user: 'Vigil',
          guilds: 1,
          channel_mappings: 1,
          bridged_core_channels: ['global_updates_channel_a8c76861'],
        },
        mappings_count: 1,
        message_links_count: 0,
        message_links_by_direction: {},
        delivery_events_count: 0,
        delivery_events_by_status: {},
        delivery_events_by_direction: {},
        recent_failures: [],
      })),
      getMappings: jest.fn().mockReturnValue(of({
        mappings: [
          {
            discord_channel_id: 'discord-1',
            discord_channel_name: 'updates',
            discord_guild_id: 'guild-1',
            discord_guild_name: 'CORE Guild',
            core_channel_id: 'core-1',
            core_channel_name: 'Global Updates',
            require_mention: false,
            enabled: true,
          },
        ],
        count: 1,
      })),
      getDeliveries: jest.fn().mockReturnValue(of({ events: [], count: 0, filters: {} })),
      getMessageLinks: jest.fn().mockReturnValue(of({ links: [], count: 0, filters: {} })),
    };

    await TestBed.configureTestingModule({
      imports: [DiscordBridgeDashboardComponent, NoopAnimationsModule],
      providers: [
        { provide: DiscordBridgeService, useValue: discordBridgeService },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(DiscordBridgeDashboardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  afterEach(() => {
    fixture.destroy();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should load diagnostics data on init', () => {
    // Arrange / Act
    fixture.detectChanges();

    // Assert
    expect(discordBridgeService.getMetrics).toHaveBeenCalled();
    expect(discordBridgeService.getMappings).toHaveBeenCalled();
    expect(component.metrics?.status.bot_user).toBe('Vigil');
    expect(component.mappings.length).toBe(1);
    expect(component.loading).toBe(false);
  });

  it('should expose checklist entries that reflect the live diagnostics state', () => {
    // Arrange / Act
    const checklist = component.validationChecklist;

    // Assert
    expect(checklist[0].status).toBe('ready');
    expect(checklist[2].status).toBe('pending');
    expect(checklist[2].detail).toContain('Send a Discord message');
  });

  it('should render the gateway overview with the new operator framing', () => {
    // Arrange / Act
    fixture.detectChanges();

    // Assert
    expect(fixture.nativeElement.textContent).toContain('Gateway is live and ready to route traffic.');
    expect(fixture.nativeElement.textContent).toContain('Operator Checklist');
    expect(fixture.nativeElement.textContent).toContain('Delivery Outcomes');
  });

  it('should surface an error when dashboard loading fails', () => {
    // Arrange
    jest.spyOn(console, 'error').mockImplementation(() => {});
    discordBridgeService.getMetrics.mockReturnValueOnce(
      throwError(() => new Error('boom'))
    );

    // Act
    component.refresh();

    // Assert
    expect(component.error).toBe('Failed to load Discord bridge dashboard data.');
    expect(component.loading).toBe(false);
  });
});
