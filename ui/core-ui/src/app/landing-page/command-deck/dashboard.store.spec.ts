import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { DashboardStore } from './dashboard.store';
import { AppConfigService } from '../../services/config/app-config.service';
import {
  SystemMonitorService,
  SystemResources
} from '../../services/system-monitor/system-monitor.service';

const SAMPLE: SystemResources = {
  cpu_usage: 42.4,
  memory_usage: 60,
  memory_total_gb: 32,
  memory_available_gb: 12,
  storage_usage: 75,
  storage_total_gb: 1000,
  storage_available_gb: 250,
  network_io_rate_mbps: 50,
  network_sent_gb: 3,
  network_recv_gb: 7,
  processes_count: 321
};

describe('DashboardStore', () => {
  let store: DashboardStore;
  let monitor: {
    getSystemResourcesPolling: jest.Mock;
    getNetworkActivityPercentage: jest.Mock;
  };

  beforeEach(() => {
    // Arrange
    monitor = {
      getSystemResourcesPolling: jest.fn().mockReturnValue(of(SAMPLE)),
      getNetworkActivityPercentage: jest.fn().mockReturnValue(50)
    };

    TestBed.configureTestingModule({
      providers: [
        DashboardStore,
        { provide: SystemMonitorService, useValue: monitor },
        { provide: AppConfigService, useValue: { apiBaseUrl: 'http://localhost:8001' } }
      ]
    });

    store = TestBed.inject(DashboardStore);
  });

  afterEach(() => {
    store.stop();
  });

  it('should fall back to polling when EventSource is unavailable', () => {
    // Arrange
    expect(typeof EventSource).toBe('undefined');

    // Act
    store.start();

    // Assert
    expect(monitor.getSystemResourcesPolling).toHaveBeenCalledWith(15);
    expect(store.feedStatus()).toBe('polling');
    expect(store.resources().cpu_usage).toBe(SAMPLE.cpu_usage);
  });

  it('should project resources into four display-ready vitals', () => {
    // Arrange
    store.start();

    // Act
    const vitals = store.vitals();

    // Assert
    expect(vitals.map((v) => v.id)).toEqual(['cpu', 'memory', 'storage', 'network']);
    expect(vitals[0].value).toBe(42);
    expect(vitals[1].detail).toContain('20.0 / 32.0 GB');
  });

  it('should only connect once across repeated start calls', () => {
    // Arrange
    store.start();

    // Act
    store.start();

    // Assert
    expect(monitor.getSystemResourcesPolling).toHaveBeenCalledTimes(1);
  });
});
