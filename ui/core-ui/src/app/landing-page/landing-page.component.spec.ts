import { ComponentFixture, TestBed } from '@angular/core/testing';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { LandingPageComponent } from './landing-page.component';
import { SystemMonitorService } from '../services/system-monitor/system-monitor.service';
import { InstanceService } from '../services/instance.service';
import { MatSnackBar } from '@angular/material/snack-bar';
import { DiscordBridgeService } from '../services/discord-bridge/discord-bridge.service';
import { AgentLibraryService } from '../services/agent-library.service';

describe('LandingPageComponent', () => {
  let component: LandingPageComponent;
  let fixture: ComponentFixture<LandingPageComponent>;
  let systemMonitorService: {
    getSystemResourcesPolling: jest.Mock;
    getNetworkActivityPercentage: jest.Mock;
  };
  let instanceService: {
    getInstances: jest.Mock;
    getSystemHealth: jest.Mock;
    getRecentActivities: jest.Mock;
    getTaskSummary: jest.Mock;
    startInstancePolling: jest.Mock;
    startSystemHealthPolling: jest.Mock;
    activities$: ReturnType<typeof of>;
  };
  let discordBridgeService: { indicator$: ReturnType<typeof of> };
  let agentLibraryService: { getAgents: jest.Mock };

  beforeEach(async () => {
    systemMonitorService = {
      getSystemResourcesPolling: jest.fn().mockReturnValue(of({
        cpu_usage: 10,
        memory_usage: 20,
        storage_usage: 30,
        network_io_rate_mbps: 1,
        memory_total_gb: 64,
        memory_available_gb: 48,
        storage_total_gb: 1024,
        storage_available_gb: 768,
        network_sent_gb: 1,
        network_recv_gb: 2,
        processes_count: 100,
      })),
      getNetworkActivityPercentage: jest.fn().mockReturnValue(5),
    };

    instanceService = {
      getInstances: jest.fn().mockReturnValue(of([])),
      getSystemHealth: jest.fn().mockReturnValue(of({
        status: 'healthy',
        services: {},
        uptime: { seconds: 1, formatted: '1s' },
        timestamp: '2026-02-09T00:00:00Z',
      })),
      getRecentActivities: jest.fn().mockReturnValue(of([])),
      getTaskSummary: jest.fn().mockReturnValue(of({
        total_tasks: 0,
        queued: 0,
        running: 0,
        completed: 0,
        failed: 0,
        last_update: '2026-02-09T00:00:00Z',
      })),
      startInstancePolling: jest.fn().mockReturnValue(of([])),
      startSystemHealthPolling: jest.fn().mockReturnValue(of({
        status: 'healthy',
        services: {},
        uptime: { seconds: 1, formatted: '1s' },
        timestamp: '2026-02-09T00:00:00Z',
      })),
      activities$: of([]),
    };

    discordBridgeService = {
      indicator$: of({
        label: 'Connected',
        tone: 'ready',
        detail: '1 mapped channel ready',
        tooltip: 'Discord Gateway connected.',
        status: null,
      }),
    };

    agentLibraryService = {
      getAgents: jest.fn().mockReturnValue(of([])),
    };

    await TestBed.configureTestingModule({
      imports: [LandingPageComponent, NoopAnimationsModule],
      providers: [
        provideRouter([]),
        { provide: SystemMonitorService, useValue: systemMonitorService },
        { provide: InstanceService, useValue: instanceService },
        { provide: MatSnackBar, useValue: { open: jest.fn() } },
        { provide: DiscordBridgeService, useValue: discordBridgeService },
        { provide: AgentLibraryService, useValue: agentLibraryService },
      ]
    })
      .compileComponents();

    fixture = TestBed.createComponent(LandingPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should render quick actions for the tools hub and Discord gateway', () => {
    // Arrange / Act
    fixture.detectChanges();

    // Assert
    expect(fixture.nativeElement.textContent).toContain('Tools & Integrations');
    expect(fixture.nativeElement.textContent).toContain('Discord Gateway');
    expect(fixture.nativeElement.textContent).toContain('Connected');
  });
});
