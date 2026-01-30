import { Component, OnInit, OnDestroy } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatChipsModule } from '@angular/material/chips';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSelectModule } from '@angular/material/select';
import { MatSnackBarModule, MatSnackBar } from '@angular/material/snack-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { ChatWindowComponent } from '../shared/chat-window/chat-window.component';
import { StatusIndicatorComponent } from '../shared/status-indicator/status-indicator.component';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { SystemMonitorService } from '../services/system-monitor/system-monitor.service';
import { InstanceService } from '../services/instance.service';
import { Subject, takeUntil, combineLatest } from 'rxjs';
import { BoardsComponent } from './boards/boards.component';
import { MyAgentsPageComponent } from '../agents-page/my-agents-page/my-agents-page.component';
import { RouterLink } from '@angular/router';
import { 
  AgentInstanceUI, 
  SystemHealth, 
  ActivityEvent, 
  TaskSummary,
  DashboardState
} from '../models/instance.models';

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
    MatSelectModule,
    MatSnackBarModule,
    MatProgressSpinnerModule,
    ChatWindowComponent,
    StatusIndicatorComponent,
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

  // Dashboard state
  dashboardState: DashboardState = {
    loading: true,
    error: null,
    lastUpdate: ''
  };

  // System stats (existing)
  systemStats = {
    cpuUsage: 0,
    memoryUsage: 0,
    storageUsage: 0,
    networkActivity: 0
  };

  detailedStats = {
    memoryTotalGb: 0,
    memoryAvailableGb: 0,
    storageTotalGb: 0,
    storageAvailableGb: 0,
    networkSentGb: 0,
    networkRecvGb: 0,
    processesCount: 0
  };

  // New real-time data
  activeAgents: AgentInstanceUI[] = [];
  systemHealth: SystemHealth | null = null;
  recentActivities: ActivityEvent[] = [];
  taskSummary: TaskSummary | null = null;

  // Deploy agent form
  selectedAgentRole = 'researcher';
  availableRoles = ['researcher', 'monitor', 'specialist', 'manager'];
  deploymentLoading = false;

  constructor(
    private systemMonitor: SystemMonitorService,
    private instanceService: InstanceService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.initializeDashboard();
    this.startPolling();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  /**
   * Initialize dashboard with initial data load
   */
  private initializeDashboard(): void {
    this.dashboardState.loading = true;
    this.dashboardState.error = null;

    // Load initial data
    combineLatest([
      this.instanceService.getInstances(),
      this.instanceService.getSystemHealth(),
      this.instanceService.getRecentActivities(),
      this.instanceService.getTaskSummary()
    ]).pipe(takeUntil(this.destroy$))
      .subscribe({
        next: ([instances, health, activities, tasks]) => {
          this.activeAgents = instances as AgentInstanceUI[];
          this.systemHealth = health;
          this.recentActivities = activities;
          this.taskSummary = tasks;
          this.dashboardState.loading = false;
          this.dashboardState.lastUpdate = new Date().toLocaleTimeString();
        },
        error: (error) => {
          console.error('Dashboard initialization error:', error);
          this.dashboardState.loading = false;
          this.dashboardState.error = 'Failed to load dashboard data';
        }
      });
  }

  /**
   * Start polling for real-time updates
   */
  private startPolling(): void {
    // Start system resources polling (every 15 seconds)
    this.systemMonitor.getSystemResourcesPolling(15)
      .pipe(takeUntil(this.destroy$))
      .subscribe(resources => {
        this.systemStats = {
          cpuUsage: Math.round(resources.cpu_usage),
          memoryUsage: Math.round(resources.memory_usage),
          storageUsage: Math.round(resources.storage_usage),
          networkActivity: Math.round(this.systemMonitor.getNetworkActivityPercentage(resources.network_io_rate_mbps))
        };

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

    // Start instance polling (every 10 seconds)
    this.instanceService.startInstancePolling()
      .pipe(takeUntil(this.destroy$))
      .subscribe(instances => {
        this.activeAgents = instances;
        this.dashboardState.lastUpdate = new Date().toLocaleTimeString();
      });

    // Start system health polling (every 30 seconds)
    this.instanceService.startSystemHealthPolling()
      .pipe(takeUntil(this.destroy$))
      .subscribe(health => {
        this.systemHealth = health;
      });

    // Subscribe to activities
    this.instanceService.activities$
      .pipe(takeUntil(this.destroy$))
      .subscribe(activities => {
        this.recentActivities = activities;
      });
  }

  /**
   * Deploy a new agent instance
   */
  deployAgent(): void {
    if (!this.selectedAgentRole) {
      this.snackBar.open('Please select an agent role', 'Close', { duration: 3000 });
      return;
    }

    this.deploymentLoading = true;
    
    this.instanceService.spawnInstance({
      agent_role: this.selectedAgentRole
    }).pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          this.deploymentLoading = false;
          if (response.success) {
            this.snackBar.open(
              `${this.selectedAgentRole} agent deployed successfully!`, 
              'Close', 
              { duration: 3000 }
            );
          } else {
            this.snackBar.open(
              `Failed to deploy agent: ${response.message}`, 
              'Close', 
              { duration: 5000 }
            );
          }
        },
        error: (error) => {
          this.deploymentLoading = false;
          this.snackBar.open('Failed to deploy agent', 'Close', { duration: 3000 });
          console.error('Deploy error:', error);
        }
      });
  }

  /**
   * Stop an agent instance
   */
  stopAgent(instance: AgentInstanceUI): void {
    this.instanceService.stopInstance(instance.id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.snackBar.open(`Agent ${instance.agent_id} stopped`, 'Close', { duration: 3000 });
        },
        error: (error) => {
          this.snackBar.open('Failed to stop agent', 'Close', { duration: 3000 });
          console.error('Stop error:', error);
        }
      });
  }

  /**
   * Restart an agent instance
   */
  restartAgent(instance: AgentInstanceUI): void {
    this.instanceService.restartInstance(instance.id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.snackBar.open(`Agent ${instance.agent_id} restarted`, 'Close', { duration: 3000 });
        },
        error: (error) => {
          this.snackBar.open('Failed to restart agent', 'Close', { duration: 3000 });
          console.error('Restart error:', error);
        }
      });
  }

  /**
   * Run a security scan (system health check)
   */
  runSecurityScan(): void {
    this.instanceService.getSystemHealth()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (health) => {
          this.systemHealth = health;
          const status = health.status === 'healthy' ? 'passed' : 'found issues';
          this.snackBar.open(`Security scan ${status}`, 'Close', { duration: 3000 });
        },
        error: (error) => {
          this.snackBar.open('Security scan failed', 'Close', { duration: 3000 });
          console.error('Security scan error:', error);
        }
      });
  }

  /**
   * Navigate to analytics (placeholder)
   */
  viewAnalytics(): void {
    this.snackBar.open('Analytics view coming soon!', 'Close', { duration: 3000 });
  }

  /**
   * Scroll to deploy panel
   */
  scrollToDeployPanel(): void {
    const element = document.querySelector('.deploy-panel');
    element?.scrollIntoView({ behavior: 'smooth' });
  }

  /**
   * Get system health status color
   */
  getSystemHealthColor(): string {
    if (!this.systemHealth) return '#666666';
    
    switch (this.systemHealth.status) {
      case 'healthy': return '#00ff88';
      case 'degraded': return '#ffaa00';
      case 'unhealthy': return '#ff5757';
      default: return '#666666';
    }
  }

  /**
   * Get readable agent status
   */
  getAgentStatusText(status: string): string {
    const statusMap: Record<string, string> = {
      'starting': 'Starting',
      'ready': 'Ready',
      'busy': 'Working',
      'stopping': 'Stopping',
      'unhealthy': 'Unhealthy',
      'lost': 'Lost Connection'
    };
    return statusMap[status] || status;
  }

  /**
   * Format activity timestamp for display
   */
  formatActivityTime(timestamp: string): string {
    const now = new Date();
    const activityTime = new Date(timestamp);
    const diffMs = now.getTime() - activityTime.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    return activityTime.toLocaleDateString();
  }
}
