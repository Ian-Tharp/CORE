import { Component } from '@angular/core';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatChipsModule } from '@angular/material/chips';
import { MatTabsModule } from '@angular/material/tabs';
import { ChatWindowComponent } from '../shared/chat-window/chat-window.component';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-landing-page',
  imports: [
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    MatChipsModule,
    MatTabsModule,
    ChatWindowComponent,
    CommonModule
  ],
  templateUrl: './landing-page.component.html',
  styleUrl: './landing-page.component.scss'
})
export class LandingPageComponent {
  systemStats = {
    cpuUsage: 24,
    memoryUsage: 68,
    storageUsage: 42,
    networkActivity: 85
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
}
