import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatCalendar, MatDatepickerModule } from '@angular/material/datepicker';
import { MatNativeDateModule } from '@angular/material/core';
import { MatChipsModule } from '@angular/material/chips';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatMenuModule } from '@angular/material/menu';
import { MatBadgeModule } from '@angular/material/badge';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { ChatWindowComponent } from '../../shared/chat-window/chat-window.component';

export interface Task {
  id: string;
  title: string;
  description?: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  status: 'pending' | 'in-progress' | 'completed' | 'blocked';
  dueDate: Date;
  assignedTo?: string;
  tags: string[];
  estimatedHours?: number;
  completedHours?: number;
}

export interface UpcomingEvent {
  id: string;
  title: string;
  description?: string;
  date: Date;
  type: 'meeting' | 'deadline' | 'milestone' | 'reminder';
  priority: 'low' | 'medium' | 'high';
}

@Component({
  selector: 'app-boards',
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatIconModule,
    MatDatepickerModule,
    MatNativeDateModule,
    MatChipsModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatCheckboxModule,
    MatMenuModule,
    MatBadgeModule,
    MatTooltipModule,
    MatProgressBarModule,
    ChatWindowComponent
  ],
  templateUrl: './boards.component.html',
  styleUrl: './boards.component.scss'
})
export class BoardsComponent implements OnInit {
  selectedDate: Date = new Date();
  tasks: Task[] = [];
  upcomingEvents: UpcomingEvent[] = [];
  
  // Filter options
  priorityFilters = ['low', 'medium', 'high', 'urgent'];
  statusFilters = ['pending', 'in-progress', 'completed', 'blocked'];
  
  constructor() {
    this.initializeMockData();
  }

  ngOnInit(): void {
    // Component initialization
  }

  get tasksForSelectedDate(): Task[] {
    return this.tasks.filter(task => 
      this.isSameDay(task.dueDate, this.selectedDate)
    );
  }

  get upcomingEventsThisWeek(): UpcomingEvent[] {
    const startOfWeek = this.getStartOfWeek(new Date());
    const endOfWeek = this.getEndOfWeek(new Date());
    
    return this.upcomingEvents.filter(event => 
      event.date >= startOfWeek && event.date <= endOfWeek
    ).sort((a, b) => a.date.getTime() - b.date.getTime());
  }

  get pendingTasksCount(): number {
    return this.tasks.filter(task => task.status === 'pending').length;
  }

  get inProgressTasksCount(): number {
    return this.tasks.filter(task => task.status === 'in-progress').length;
  }

  get completedTasksCount(): number {
    return this.tasks.filter(task => task.status === 'completed').length;
  }

  get highPriorityTasksCount(): number {
    return this.tasks.filter(task => task.priority === 'urgent' || task.priority === 'high').length;
  }

  onDateSelected(date: Date): void {
    this.selectedDate = date;
  }

  toggleTaskStatus(task: Task): void {
    if (task.status === 'completed') {
      task.status = 'pending';
    } else if (task.status === 'pending') {
      task.status = 'in-progress';
    } else if (task.status === 'in-progress') {
      task.status = 'completed';
    }
  }

  getPriorityColor(priority: string): string {
    switch (priority) {
      case 'low': return '#4caf50';
      case 'medium': return '#ff9800';
      case 'high': return '#f44336';
      case 'urgent': return '#e91e63';
      default: return '#757575';
    }
  }

  getStatusIcon(status: string): string {
    switch (status) {
      case 'pending': return 'radio_button_unchecked';
      case 'in-progress': return 'schedule';
      case 'completed': return 'task_alt';
      case 'blocked': return 'block';
      default: return 'help';
    }
  }

  getEventTypeIcon(type: string): string {
    switch (type) {
      case 'meeting': return 'group';
      case 'deadline': return 'schedule';
      case 'milestone': return 'flag';
      case 'reminder': return 'notifications';
      default: return 'event';
    }
  }

  // Date utility methods
  private isSameDay(date1: Date, date2: Date): boolean {
    return date1.toDateString() === date2.toDateString();
  }

  private getStartOfWeek(date: Date): Date {
    const startOfWeek = new Date(date);
    const day = startOfWeek.getDay();
    const diff = startOfWeek.getDate() - day;
    startOfWeek.setDate(diff);
    startOfWeek.setHours(0, 0, 0, 0);
    return startOfWeek;
  }

  private getEndOfWeek(date: Date): Date {
    const endOfWeek = new Date(date);
    const day = endOfWeek.getDay();
    const diff = endOfWeek.getDate() + (6 - day);
    endOfWeek.setDate(diff);
    endOfWeek.setHours(23, 59, 59, 999);
    return endOfWeek;
  }

  private initializeMockData(): void {
    // Mock tasks
    this.tasks = [
      {
        id: '1',
        title: 'Implement Agent Authentication System',
        description: 'Design and implement secure authentication for AI agents',
        priority: 'high',
        status: 'in-progress',
        dueDate: new Date(),
        assignedTo: 'CORE System',
        tags: ['security', 'authentication', 'agents'],
        estimatedHours: 8,
        completedHours: 4
      },
      {
        id: '2',
        title: 'Optimize Vector Database Performance',
        description: 'Improve query performance for knowledge retrieval',
        priority: 'medium',
        status: 'pending',
        dueDate: new Date(Date.now() + 86400000), // Tomorrow
        assignedTo: 'E.V.E.',
        tags: ['database', 'optimization', 'performance'],
        estimatedHours: 6
      },
      {
        id: '3',
        title: 'Deploy New Workflow Engine',
        description: 'Release the updated workflow orchestration system',
        priority: 'urgent',
        status: 'blocked',
        dueDate: new Date(),
        assignedTo: 'AEGIS',
        tags: ['workflow', 'deployment', 'orchestration'],
        estimatedHours: 12,
        completedHours: 2
      },
      {
        id: '4',
        title: 'Update System Documentation',
        description: 'Comprehensive update of technical documentation',
        priority: 'low',
        status: 'completed',
        dueDate: new Date(Date.now() - 86400000), // Yesterday
        assignedTo: 'Documentation Agent',
        tags: ['documentation', 'maintenance'],
        estimatedHours: 4,
        completedHours: 4
      }
    ];

    // Mock upcoming events
    this.upcomingEvents = [
      {
        id: '1',
        title: 'System Maintenance Window',
        description: 'Scheduled maintenance for core infrastructure',
        date: new Date(Date.now() + 2 * 86400000), // 2 days from now
        type: 'maintenance' as any,
        priority: 'high'
      },
      {
        id: '2',
        title: 'Agent Performance Review',
        description: 'Weekly review of agent performance metrics',
        date: new Date(Date.now() + 3 * 86400000), // 3 days from now
        type: 'meeting',
        priority: 'medium'
      },
      {
        id: '3',
        title: 'Security Audit Deadline',
        description: 'Complete security audit for Q4',
        date: new Date(Date.now() + 7 * 86400000), // 1 week from now
        type: 'deadline',
        priority: 'high'
      },
      {
        id: '4',
        title: 'Knowledge Base Backup',
        description: 'Automated backup of knowledge repositories',
        date: new Date(Date.now() + 86400000), // Tomorrow
        type: 'reminder',
        priority: 'low'
      }
    ];
  }
}
