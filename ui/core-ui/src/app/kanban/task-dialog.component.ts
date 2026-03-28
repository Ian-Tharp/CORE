import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatDialogRef, MAT_DIALOG_DATA, MatDialogModule } from '@angular/material/dialog';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

export interface TaskDialogData {
  mode: 'create' | 'edit';
  task?: {
    title: string;
    description: string;
    project: string;
    priority: string;
    assignee: string;
    status_column: string;
  };
}

@Component({
  selector: 'app-task-dialog',
  imports: [
    CommonModule,
    FormsModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatButtonModule,
    MatIconModule
  ],
  template: `
    <div class="task-dialog-wrap">
      <div class="dialog-header">
        <div class="dialog-icon-wrap" [class.edit-mode]="data.mode === 'edit'">
          <mat-icon>{{ data.mode === 'create' ? 'add_task' : 'edit_note' }}</mat-icon>
        </div>
        <div class="dialog-header-text">
          <h2>{{ data.mode === 'create' ? 'Create New Task' : 'Edit Task' }}</h2>
          <span class="dialog-subtitle">{{ data.mode === 'create' ? 'Add a task to the project board' : 'Modify task details' }}</span>
        </div>
      </div>

      <mat-dialog-content>
        <div class="form-grid">
          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Task Title</mat-label>
            <input matInput [(ngModel)]="task.title" placeholder="What needs to be done?" required autocomplete="off">
            
          </mat-form-field>

          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Description</mat-label>
            <textarea matInput [(ngModel)]="task.description" placeholder="Add details, context, or requirements..." rows="3"></textarea>
          </mat-form-field>

          <div class="form-row">
            <mat-form-field appearance="outline" class="half-field">
              <mat-label>Project</mat-label>
              <mat-select [(ngModel)]="task.project">
                <mat-option *ngFor="let p of projects" [value]="p">{{ p }}</mat-option>
              </mat-select>
            </mat-form-field>

            <mat-form-field appearance="outline" class="half-field">
              <mat-label>Priority</mat-label>
              <mat-select [(ngModel)]="task.priority">
                <mat-option value="critical">
                  <span class="priority-dot priority-critical"></span> Critical
                </mat-option>
                <mat-option value="high">
                  <span class="priority-dot priority-high"></span> High
                </mat-option>
                <mat-option value="medium">
                  <span class="priority-dot priority-medium"></span> Medium
                </mat-option>
                <mat-option value="low">
                  <span class="priority-dot priority-low"></span> Low
                </mat-option>
              </mat-select>
            </mat-form-field>
          </div>

          <div class="form-row">
            <mat-form-field appearance="outline" class="half-field">
              <mat-label>Assignee</mat-label>
              <input matInput [(ngModel)]="task.assignee" placeholder="Who's responsible?" autocomplete="off">
              
            </mat-form-field>

            <mat-form-field appearance="outline" class="half-field">
              <mat-label>Status Column</mat-label>
              <mat-select [(ngModel)]="task.status_column">
                <mat-option value="backlog">
                  <span class="status-dot" style="background:#64748b"></span> Backlog
                </mat-option>
                <mat-option value="ready">
                  <span class="status-dot" style="background:#14b8a6"></span> Ready
                </mat-option>
                <mat-option value="in_progress">
                  <span class="status-dot" style="background:#f59e0b"></span> In Progress
                </mat-option>
                <mat-option value="review">
                  <span class="status-dot" style="background:#a855f7"></span> Review
                </mat-option>
                <mat-option value="done">
                  <span class="status-dot" style="background:#10b981"></span> Done
                </mat-option>
              </mat-select>
            </mat-form-field>
          </div>
        </div>
      </mat-dialog-content>

      <div class="dialog-actions">
        <button mat-button (click)="onCancel()" class="cancel-btn">Cancel</button>
        <button mat-flat-button (click)="onSave()" class="save-btn" [disabled]="!task.title">
          <mat-icon>{{ data.mode === 'create' ? 'add' : 'check' }}</mat-icon>
          {{ data.mode === 'create' ? 'Create Task' : 'Save Changes' }}
        </button>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
    }

    .task-dialog-wrap {
      min-width: 520px;
      font-family: 'Inter', sans-serif;
    }

    .dialog-header {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 20px 24px 12px;
    }

    .dialog-icon-wrap {
      width: 40px;
      height: 40px;
      border-radius: 10px;
      background: rgba(#f59e0b, 0.1);
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;

      mat-icon {
        color: #f59e0b;
        font-size: 22px;
        width: 22px;
        height: 22px;
      }

      &.edit-mode {
        background: rgba(#14b8a6, 0.1);
        mat-icon { color: #14b8a6; }
      }
    }

    .dialog-header-text {
      h2 {
        margin: 0;
        font-size: 18px;
        font-weight: 700;
        color: #e5e7eb;
        letter-spacing: -0.2px;
      }

      .dialog-subtitle {
        font-size: 12px;
        color: #6b7280;
      }
    }

    mat-dialog-content {
      padding: 8px 24px 0;
      max-height: 60vh;
    }

    .form-grid {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .form-row {
      display: flex;
      gap: 14px;
    }

    .half-field {
      flex: 1;
    }

    .full-width {
      width: 100%;
    }

    .field-icon {
      color: #6b7280;
      font-size: 18px;
      width: 18px;
      height: 18px;
      margin-right: 4px;
    }

    .priority-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 8px;
      vertical-align: middle;
    }

    .priority-critical { background: #ef4444; }
    .priority-high { background: #f97316; }
    .priority-medium { background: #f59e0b; }
    .priority-low { background: #10b981; }

    .status-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 8px;
      vertical-align: middle;
    }

    .dialog-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      padding: 12px 24px 20px;
    }

    .cancel-btn {
      color: #9ca3af;
      border-radius: 8px;
      font-weight: 500;

      &:hover {
        background: rgba(#9ca3af, 0.08);
      }
    }

    .save-btn {
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #0a0f0d;
      display: flex;
      align-items: center;
      gap: 6px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 13px;
      padding: 0 20px;
      height: 38px;
      transition: all 0.2s ease;

      mat-icon {
        font-size: 18px;
        width: 18px;
        height: 18px;
      }

      &:hover:not(:disabled) {
        box-shadow: 0 0 20px rgba(#f59e0b, 0.3);
      }

      &:disabled {
        opacity: 0.4;
      }
    }

    ::ng-deep .task-dialog-wrap .mat-mdc-form-field {
      --mdc-outlined-text-field-outline-color: #2d3b35;
      --mdc-outlined-text-field-focus-outline-color: #f59e0b;
      --mdc-outlined-text-field-label-text-color: #6b7280;
      --mdc-outlined-text-field-focus-label-text-color: #f59e0b;
      --mdc-outlined-text-field-input-text-color: #e5e7eb;
      --mdc-outlined-text-field-container-shape: 10px;
    }

    ::ng-deep .task-dialog-wrap .mat-mdc-form-field-subscript-wrapper {
      display: none;
    }

    @media (max-width: 600px) {
      .task-dialog-wrap {
        min-width: unset;
      }
      .form-row {
        flex-direction: column;
        gap: 2px;
      }
    }
  `]
})
export class TaskDialogComponent {
  projects = ['NK', 'CORE', 'PWE', 'Embermesh', 'PLE', 'Solarfall Arena'];

  task = {
    title: '',
    description: '',
    project: 'NK',
    priority: 'medium',
    assignee: '',
    status_column: 'backlog'
  };

  constructor(
    public dialogRef: MatDialogRef<TaskDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: TaskDialogData
  ) {
    if (data.task) {
      this.task = { ...data.task };
    }
  }

  onCancel(): void {
    this.dialogRef.close();
  }

  onSave(): void {
    this.dialogRef.close(this.task);
  }
}