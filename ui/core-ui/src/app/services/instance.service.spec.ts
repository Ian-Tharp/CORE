import { TestBed } from '@angular/core/testing';
import {
  HttpClientTestingModule,
  HttpTestingController,
} from '@angular/common/http/testing';

import { InstanceService } from './instance.service';

describe('InstanceService', () => {
  let service: InstanceService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
    });

    service = TestBed.inject(InstanceService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should request database-backed instances and map the backend payload', () => {
    // Arrange
    let responseValue: any[] = [];

    // Act
    service.getInstances().subscribe((instances) => {
      responseValue = instances;
    });

    // Assert
    const request = httpMock.expectOne((req) =>
      req.url === 'http://localhost:8001/instances' &&
      req.params.get('include_docker') === 'false' &&
      req.params.get('page') === '1' &&
      req.params.get('page_size') === '200'
    );
    expect(request.request.method).toBe('GET');

    request.flush({
      instances: [
        {
          container_id: 'container-1',
          agent_id: 'agent-1',
          agent_role: 'researcher',
          status: 'ready',
          health_status: 'healthy',
          uptime_seconds: 120,
          memory_usage: { memory_mb: 128 },
          cpu_usage: 0.25,
          created_at: '2026-03-29T20:00:00Z',
          last_heartbeat: '2026-03-29T20:05:00Z',
        },
      ],
      total_count: 1,
      page: 1,
      page_size: 200,
    });

    expect(responseValue).toHaveLength(1);
    expect(responseValue[0]).toEqual(
      expect.objectContaining({
        id: 'container-1',
        container_id: 'container-1',
        agent_id: 'agent-1',
        agent_role: 'researcher',
        status: 'ready',
      })
    );
    expect(responseValue[0].resource_profile.cpu_percent).toBe(0.25);
  });

  it('should use the root deep health endpoint', () => {
    // Arrange / Act
    service.getSystemHealth().subscribe();

    // Assert
    const request = httpMock.expectOne('http://localhost:8001/health/deep');
    expect(request.request.method).toBe('GET');
    request.flush({
      status: 'healthy',
      services: {},
      uptime: {
        seconds: 10,
        formatted: '10s',
      },
      timestamp: '2026-03-29T20:00:00Z',
    });
  });

  it('should stop instances via the stop endpoint', () => {
    // Arrange / Act
    service.stopInstance('container-1').subscribe();

    // Assert
    const request = httpMock.expectOne('http://localhost:8001/instances/container-1/stop');
    expect(request.request.method).toBe('POST');
    request.flush({});
  });
});
