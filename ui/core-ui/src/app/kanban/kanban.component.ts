import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { CdkDragDrop, DragDropModule, moveItemInArray, transferArrayItem } from '@angular/cdk/drag-drop';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatChipsModule } from '@angular/material/chips';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { MatBadgeModule } from '@angular/material/badge';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatMenuModule } from '@angular/material/menu';
import { Subject, takeUntil } from 'rxjs';
import { KanbanService, KanbanTask } from '../services/kanban/kanban.service';
import { TaskDialogComponent } from './task-dialog.component';

interface BoardColumn {
  id: string;
  title: string;
  icon: string;
  tasks: KanbanTask[];
  accentColor: string;
}

@Component({
  selector: 'app-kanban',
  imports: [
    CommonModule,
    FormsModule,
    DragDropModule,
    MatButtonModule,
    MatIconModule,
    MatChipsModule,
    MatSelectModule,
    MatFormFieldModule,
    MatInputModule,
    MatTooltipModule,
    MatDialogModule,
    MatBadgeModule,
    MatProgressSpinnerModule,
    MatSnackBarModule,
    MatMenuModule
  ],
  templateUrl: './kanban.component.html',
  styleUrl: './kanban.component.scss'
})
export class KanbanComponent implements OnInit, OnDestroy {
  private destroy$ = new Subject<void>();
  isLoading = true;
  hoveredTaskId: string | null = null;

  columnAccentColors: { [key: string]: string } = {
    backlog: '#64748b',
    ready: '#14b8a6',
    in_progress: '#f59e0b',
    review: '#a855f7',
    done: '#10b981'
  };

  columns: BoardColumn[] = [
    { id: 'backlog', title: 'Backlog', icon: 'inbox', tasks: [], accentColor: '#64748b' },
    { id: 'ready', title: 'Ready', icon: 'playlist_add_check', tasks: [], accentColor: '#14b8a6' },
    { id: 'in_progress', title: 'In Progress', icon: 'trending_up', tasks: [], accentColor: '#f59e0b' },
    { id: 'review', title: 'Review', icon: 'rate_review', tasks: [], accentColor: '#a855f7' },
    { id: 'done', title: 'Done', icon: 'check_circle', tasks: [], accentColor: '#10b981' }
  ];

  connectedLists: string[] = [];

  projects = ['NK', 'CORE', 'PWE', 'Embermesh', 'PLE', 'Solarfall Arena'];
  priorities = ['critical', 'high', 'medium', 'low'];

  filterProject = '';
  filterPriority = '';
  filterAssignee = '';

  allTasks: KanbanTask[] = [];

  constructor(
    private kanbanService: KanbanService,
    private dialog: MatDialog,
    private snackBar: MatSnackBar
  ) {
    this.connectedLists = this.columns.map(c => c.id);
  }

  ngOnInit(): void {
    this.loadTasks();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  loadTasks(): void {
    this.isLoading = true;
    this.kanbanService.getTasks()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          const tasks = response.tasks || response || [];
          this.allTasks = Array.isArray(tasks) ? tasks : [];
          this.distributeTasksToColumns();
          this.isLoading = false;
        },
        error: (err) => {
          console.error('Failed to load tasks:', err);
          this.isLoading = false;
          this.snackBar.open('Failed to load tasks', 'Dismiss', { duration: 3000 });
        }
      });
  }

  distributeTasksToColumns(): void {
    this.columns.forEach(col => col.tasks = []);
    const filtered = this.getFilteredTasks();

    filtered.forEach(task => {
      const statusColumn = task.payload?.['status_column'] || this.mapStatusToColumn(task.status);
      const column = this.columns.find(c => c.id === statusColumn);
      if (column) {
        column.tasks.push(task);
      } else {
        this.columns[0].tasks.push(task);
      }
    });

    this.columns.forEach(col => {
      col.tasks.sort((a, b) => (b.priority || 5) - (a.priority || 5));
    });
  }

  mapStatusToColumn(status: string): string {
    const mapping: { [key: string]: string } = {
      'pending': 'backlog',
      'queued': 'backlog',
      'assigned': 'in_progress',
      'running': 'in_progress',
      'completed': 'done',
      'failed': 'backlog',
      'cancelled': 'done'
    };
    return mapping[status] || 'backlog';
  }

  getFilteredTasks(): KanbanTask[] {
    return this.allTasks.filter(task => {
      if (this.filterProject && task.payload?.['project'] !== this.filterProject) {return false;}
      if (this.filterPriority && this.getPriorityLabel(task.priority) !== this.filterPriority) {return false;}
      if (this.filterAssignee) {
        const assignee = (task.payload?.['assignee'] || '').toLowerCase();
        if (!assignee.includes(this.filterAssignee.toLowerCase())) {return false;}
      }
      return true;
    });
  }

  getPriorityLabel(priority: number): string {
    if (priority >= 9) {return 'critical';}
    if (priority >= 7) {return 'high';}
    if (priority >= 4) {return 'medium';}
    return 'low';
  }

  getPriorityFromLabel(label: string): number {
    const map: { [key: string]: number } = { critical: 10, high: 7, medium: 5, low: 3 };
    return map[label] || 5;
  }

  onFiltersChanged(): void {
    this.distributeTasksToColumns();
  }

  clearFilters(): void {
    this.filterProject = '';
    this.filterPriority = '';
    this.filterAssignee = '';
    this.distributeTasksToColumns();
  }

  hasActiveFilters(): boolean {
    return !!(this.filterProject || this.filterPriority || this.filterAssignee);
  }

  drop(event: CdkDragDrop<KanbanTask[]>): void {
    if (event.previousContainer === event.container) {
      moveItemInArray(event.container.data, event.previousIndex, event.currentIndex);
    } else {
      transferArrayItem(
        event.previousContainer.data,
        event.container.data,
        event.previousIndex,
        event.currentIndex
      );

      const task = event.container.data[event.currentIndex];
      const newColumnId = event.container.id;

      if (task.payload) {
        task.payload['status_column'] = newColumnId;
      }

      if (newColumnId === 'done' && task.status !== 'completed') {
        this.kanbanService.updateTaskStatus(task.id, 'complete')
          .pipe(takeUntil(this.destroy$))
          .subscribe({
            next: () => this.snackBar.open('Task completed!', '', { duration: 2000 }),
            error: () => this.snackBar.open('Status update failed', 'Dismiss', { duration: 3000 })
          });
      }
    }
  }

  openNewTaskDialog(): void {
    const dialogRef = this.dialog.open(TaskDialogComponent, {
      width: '600px',
      panelClass: 'kanban-dialog',
      data: { mode: 'create' }
    });

    dialogRef.afterClosed()
      .pipe(takeUntil(this.destroy$))
      .subscribe(result => {
        if (result) {
          const payload = {
            task_type: 'kanban_task',
            priority: this.getPriorityFromLabel(result.priority),
            payload: {
              title: result.title,
              description: result.description,
              project: result.project,
              assignee: result.assignee,
              status_column: result.status_column,
              priority_label: result.priority
            }
          };

          this.kanbanService.createTask(payload)
            .pipe(takeUntil(this.destroy$))
            .subscribe({
              next: () => {
                this.snackBar.open('Task created!', '', { duration: 2000 });
                this.loadTasks();
              },
              error: () => this.snackBar.open('Failed to create task', 'Dismiss', { duration: 3000 })
            });
        }
      });
  }

  openEditDialog(task: KanbanTask): void {
    const dialogRef = this.dialog.open(TaskDialogComponent, {
      width: '600px',
      panelClass: 'kanban-dialog',
      data: {
        mode: 'edit',
        task: {
          title: this.getTaskTitle(task),
          description: task.payload?.['description'] || '',
          project: this.getTaskProject(task),
          priority: task.payload?.['priority_label'] || this.getPriorityLabel(task.priority),
          assignee: this.getTaskAssignee(task),
          status_column: task.payload?.['status_column'] || this.mapStatusToColumn(task.status)
        }
      }
    });

    dialogRef.afterClosed()
      .pipe(takeUntil(this.destroy$))
      .subscribe(result => {
        if (result) {
          this.snackBar.open('Task updated (local only)', '', { duration: 2000 });
          if (task.payload) {
            task.payload['title'] = result.title;
            task.payload['description'] = result.description;
            task.payload['project'] = result.project;
            task.payload['priority_label'] = result.priority;
            task.payload['assignee'] = result.assignee;
            task.payload['status_column'] = result.status_column;
          }
          this.distributeTasksToColumns();
        }
      });
  }

  getRelativeTime(dateStr: string): string {
    if (!dateStr) {return '';}
    const now = new Date();
    const date = new Date(dateStr);
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHr = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHr / 24);

    if (diffSec < 60) {return 'just now';}
    if (diffMin < 60) {return diffMin + 'm ago';}
    if (diffHr < 24) {return diffHr + 'h ago';}
    if (diffDay < 7) {return diffDay + 'd ago';}
    if (diffDay < 30) {return Math.floor(diffDay / 7) + 'w ago';}
    return Math.floor(diffDay / 30) + 'mo ago';
  }

  onCardHover(taskId: string | null): void {
    this.hoveredTaskId = taskId;
  }

  getTaskTitle(task: KanbanTask): string {
    return task.payload?.['title'] || task.task_type || 'Untitled';
  }

  getTaskDescription(task: KanbanTask): string {
    const desc = task.payload?.['description'] || '';
    return desc.length > 120 ? desc.substring(0, 120) + '...' : desc;
  }

  getTaskProject(task: KanbanTask): string {
    return task.payload?.['project'] || '';
  }

  getTaskAssignee(task: KanbanTask): string {
    return task.payload?.['assignee'] || task.assigned_agent_id || '';
  }

  getAssigneeInitials(task: KanbanTask): string {
    const name = this.getTaskAssignee(task);
    if (!name) {return '?';}
    return name.split(' ').map((n: string) => n[0]).join('').toUpperCase().substring(0, 2);
  }

  getPriorityClass(task: KanbanTask): string {
    const label = task.payload?.['priority_label'] || this.getPriorityLabel(task.priority);
    return 'priority-' + label;
  }

  getPriorityText(task: KanbanTask): string {
    const label = task.payload?.['priority_label'];
    if (label && typeof label === 'string' && isNaN(Number(label))) {
      return label;
    }
    return this.getPriorityLabel(task.priority);
  }

  getProjectClass(project: string): string {
    const map: { [key: string]: string } = {
      'NK': 'project-nk',
      'CORE': 'project-core',
      'PWE': 'project-pwe',
      'Embermesh': 'project-embermesh',
      'PLE': 'project-ple',
      'Solarfall Arena': 'project-solarfall'
    };
    return map[project] || 'project-default';
  }

  getTotalTasks(): number {
    return this.columns.reduce((sum, col) => sum + col.tasks.length, 0);
  }

  getTaskCreatedDate(task: KanbanTask): string {
    return (task as any).created_at || (task as any).createdAt || '';
  }
}