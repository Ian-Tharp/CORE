import { Component, OnInit, OnDestroy } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatChipsModule } from '@angular/material/chips';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ChatWindowComponent } from '../shared/chat-window/chat-window.component';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { SystemMonitorService } from '../services/system-monitor/system-monitor.service';
import { Subject, takeUntil } from 'rxjs';
import { BoardsComponent } from './boards/boards.component';
import { MyAgentsPageComponent } from '../agents-page/my-agents-page/my-agents-page.component';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-landing-page',
  imports: [
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatChipsModule,
    MatTabsModule,
    MatTooltipModule,
    ChatWindowComponent,
    CommonModule,
    HttpClientModule,
    BoardsComponent,
    MyAgentsPageComponent,
    RouterLink
  ],
  templateUrl: './landing-page.component.html',
  styleUrl: './landing-page.component.scss'
})
export class LandingPageComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();

  systemStats = {
    cpuUsage: 0,
    memoryUsage: 0,
    storageUsage: 0,
    networkActivity: 0
  };

  // Additional detailed stats for tooltips or future use
  detailedStats = {
    memoryTotalGb: 0,
    memoryAvailableGb: 0,
    storageTotalGb: 0,
    storageAvailableGb: 0,
    networkSentGb: 0,
    networkRecvGb: 0,
    processesCount: 0
  };

  activeAgents = [
    { name: 'E.V.E.', status: 'learning', uptime: '2h 34m' },
    { name: 'AEGIS', status: 'monitoring', uptime: '5h 12m' },
    { name: 'ORBIT', status: 'idle', uptime: '1h 08m' }
  ];

  recentActivities = [
    'Agent E.V.E. completed self-play iteration #247',
    'AEGIS detected anomaly in network traffic',
    'New workflow template created: "Smart Home Automation"',
    'System backup completed successfully'
  ];

  constructor(private systemMonitor: SystemMonitorService) {}

  ngOnInit(): void {
    // Start polling system resources every 15 seconds
    this.systemMonitor.getSystemResourcesPolling(15)
      .pipe(takeUntil(this.destroy$))
      .subscribe(resources => {
        // Update the main stats used in the template
        this.systemStats = {
          cpuUsage: Math.round(resources.cpu_usage),
          memoryUsage: Math.round(resources.memory_usage),
          storageUsage: Math.round(resources.storage_usage),
          networkActivity: Math.round(this.systemMonitor.getNetworkActivityPercentage(resources.network_io_rate_mbps))
        };

        // Store additional details
        this.detailedStats = {
          memoryTotalGb: resources.memory_total_gb,
          memoryAvailableGb: resources.memory_available_gb,
          storageTotalGb: resources.storage_total_gb,
          storageAvailableGb: resources.storage_available_gb,
          networkSentGb: resources.network_sent_gb,
          networkRecvGb: resources.network_recv_gb,
          processesCount: resources.processes_count
        };
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }
}
