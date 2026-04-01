import { TestBed } from '@angular/core/testing';
import {
  HttpClientTestingModule,
  HttpTestingController,
} from '@angular/common/http/testing';
import { skip } from 'rxjs';

import { DiscordBridgeService } from './discord-bridge.service';

describe('DiscordBridgeService', () => {
  let service: DiscordBridgeService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
    });

    service = TestBed.inject(DiscordBridgeService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should request bridge metrics with the provided failure limit', () => {
    // Arrange / Act
    service.getMetrics(7).subscribe();

    // Assert
    const request = httpMock.expectOne(
      'http://localhost:8001/discord/metrics?recent_failures_limit=7'
    );
    expect(request.request.method).toBe('GET');
    request.flush({
      status: {
        status: 'connected',
        connected: true,
        connected_at: null,
        last_error: null,
        reconnect_attempts: 0,
        bot_user: 'Vigil',
        guilds: 1,
        channel_mappings: 1,
        bridged_core_channels: [],
      },
      mappings_count: 1,
      message_links_count: 0,
      message_links_by_direction: {},
      delivery_events_count: 0,
      delivery_events_by_status: {},
      delivery_events_by_direction: {},
      recent_failures: [],
    });
  });

  it('should include filters when requesting delivery events', () => {
    // Arrange / Act
    service.getDeliveries({
      limit: 10,
      status: 'failed',
      direction: 'core_to_discord',
      discord_channel_id: 'discord-1',
    }).subscribe();

    // Assert
    const request = httpMock.expectOne(
      'http://localhost:8001/discord/deliveries?limit=10&status=failed&direction=core_to_discord&discord_channel_id=discord-1'
    );
    expect(request.request.method).toBe('GET');
    request.flush({ events: [], count: 0, filters: {} });
  });

  it('should expose a connected indicator when status polling succeeds', () => {
    // Arrange
    let renderedIndicator: any;
    service.indicator$.pipe(skip(1)).subscribe((indicator) => {
      renderedIndicator = indicator;
    });

    // Act
    service.startStatusPolling(60000);

    const request = httpMock.expectOne('http://localhost:8001/discord/status');
    request.flush({
      status: 'connected',
      connected: true,
      connected_at: null,
      last_error: null,
      reconnect_attempts: 0,
      bot_user: 'Vigil',
      guilds: 1,
      channel_mappings: 2,
      bridged_core_channels: [],
    });

    // Assert
    expect(renderedIndicator.label).toBe('Connected');
    expect(renderedIndicator.tone).toBe('ready');
    service.stopStatusPolling();
  });

  it('should expose an unknown indicator when status polling fails', () => {
    // Arrange
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    let renderedIndicator: any;
    service.indicator$.pipe(skip(1)).subscribe((indicator) => {
      renderedIndicator = indicator;
    });

    // Act
    service.startStatusPolling(60000);

    const request = httpMock.expectOne('http://localhost:8001/discord/status');
    request.flush('boom', {
      status: 500,
      statusText: 'Server Error',
    });

    // Assert
    expect(renderedIndicator.label).toBe('Unknown');
    expect(renderedIndicator.tone).toBe('unknown');
    service.stopStatusPolling();
  });
});
