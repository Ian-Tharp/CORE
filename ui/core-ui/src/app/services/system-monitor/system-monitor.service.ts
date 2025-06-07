import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, interval, switchMap, startWith, catchError, of } from 'rxjs';
import { map } from 'rxjs/operators';

export interface SystemResources {
  cpu_usage: number;
  memory_usage: number;
  memory_total_gb: number;
  memory_available_gb: number;
  storage_usage: number;
  storage_total_gb: number;
  storage_available_gb: number;
  network_io_rate_mbps: number;
  network_sent_gb: number;
  network_recv_gb: number;
  processes_count: number;
}

export interface ProcessInfo {
  pid: number;
  name: string;
  cpu_percent: number;
  memory_percent: number;
  status: string;
}

@Injectable({
  providedIn: 'root'
})
export class SystemMonitorService {
  private readonly apiUrl = 'http://localhost:8001';

  constructor(private http: HttpClient) {}

  /**
   * Get current system resources snapshot
   */
  getSystemResources(): Observable<SystemResources> {
    return this.http.get<SystemResources>(`${this.apiUrl}/system/resources`).pipe(
      catchError(error => {
        console.error('Error fetching system resources:', error);
        // Return default values on error
        return of({
          cpu_usage: 0,
          memory_usage: 0,
          memory_total_gb: 0,
          memory_available_gb: 0,
          storage_usage: 0,
          storage_total_gb: 0,
          storage_available_gb: 0,
          network_io_rate_mbps: 0,
          network_sent_gb: 0,
          network_recv_gb: 0,
          processes_count: 0
        });
      })
    );
  }

  /**
   * Get system resources with automatic polling every N seconds
   * @param intervalSeconds - Polling interval in seconds (default: 5)
   */
  getSystemResourcesPolling(intervalSeconds: number = 5): Observable<SystemResources> {
    return interval(intervalSeconds * 1000).pipe(
      startWith(0), // Emit immediately on subscription
      switchMap(() => this.getSystemResources())
    );
  }

  /**
   * Get top processes by CPU usage
   * @param limit - Number of processes to return (default: 10)
   */
  getTopProcesses(limit: number = 10): Observable<ProcessInfo[]> {
    return this.http.get<ProcessInfo[]>(`${this.apiUrl}/system/processes/top?limit=${limit}`).pipe(
      catchError(error => {
        console.error('Error fetching top processes:', error);
        return of([]);
      })
    );
  }

  /**
   * Convert network activity rate to a percentage for progress bar display
   * This is a simplified metric for UI display purposes
   */
  getNetworkActivityPercentage(networkIoRateMbps: number): number {
    // Assume 100 Mbps as baseline for 100% activity
    // Adjust this based on your network capacity
    return Math.min((networkIoRateMbps / 100) * 100, 100);
  }
} 