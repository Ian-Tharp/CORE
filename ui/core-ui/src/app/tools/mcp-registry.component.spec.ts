import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { McpRegistryComponent } from './mcp-registry.component';
import { DiscordBridgeService } from '../services/discord-bridge/discord-bridge.service';

describe('McpRegistryComponent', () => {
  let component: McpRegistryComponent;
  let fixture: ComponentFixture<McpRegistryComponent>;
  let fetchMock: jest.Mock;
  let discordBridgeService: { indicator$: ReturnType<typeof of> };

  beforeEach(async () => {
    fetchMock = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        servers: [],
        total_tools: 0
      }),
      statusText: 'OK'
    });

    global.fetch = fetchMock as typeof fetch;
    discordBridgeService = {
      indicator$: of({
        label: 'Connected',
        tone: 'ready',
        detail: '1 mapped channel ready',
        tooltip: 'Discord Gateway connected.',
        status: null
      })
    };

    await TestBed.configureTestingModule({
      imports: [McpRegistryComponent],
      providers: [
        provideRouter([]),
        { provide: DiscordBridgeService, useValue: discordBridgeService }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(McpRegistryComponent);
    component = fixture.componentInstance;
  });

  afterEach(() => {
    jest.resetAllMocks();
    localStorage.clear();
  });

  it('should load the MCP registry from the CORE backend', async () => {
    // Arrange / Act
    await component.loadRegistry();

    // Assert
    expect(fetchMock).toHaveBeenCalledWith(
      'http://localhost:8001/mcp/registry',
      expect.objectContaining({
        headers: {
          'X-API-Key': 'core_dev_key'
        }
      })
    );
  });

  it('should render the Discord Gateway entry point and operator snapshot', () => {
    // Arrange
    component.registry = {
      servers: [],
      total_tools: 0
    };

    // Act
    fixture.detectChanges();

    // Assert
    expect(fixture.nativeElement.textContent).toContain('Discord Gateway');
    expect(fixture.nativeElement.textContent).toContain('Open diagnostics');
    expect(fixture.nativeElement.textContent).toContain('Integration Status');
  });
});
