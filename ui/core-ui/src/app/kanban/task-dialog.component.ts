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
    <div class="task-dialog">
      <h2 mat-dialog-title>
        <mat-icon class="dialog-icon">{{ data.mode === 'create' ? 'add_task' : 'edit_note' }}</mat-icon>
        {{ data.mode === 'create' ? 'New Task' : 'Edit Task' }}
      </h2>

      <mat-dialog-content>
        <div class="form-grid">
          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Title</mat-label>
            <input matInput [(ngModel)]="task.title" placeholder="Task title..." required>
          </mat-form-field>

          <mat-form-field appearance="outline" class="full-width">
            <mat-label>Description</mat-label>
            <textarea matInput [(ngModel)]="task.description" placeholder="Describe the task..." rows="4"></textarea>
          </mat-form-field>

          <div class="form-row">
            <mat-form-field appearance="outline">
              <mat-label>Project</mat-label>
              <mat-select [(ngModel)]="task.project">
                <mat-option *ngFor="let p of projects" [value]="p">{{ p }}</mat-option>
              </mat-select>
            </mat-form-field>

            <mat-form-field appearance="outline">
              <mat-label>Priority</mat-label>
              <mat-select [(ngModel)]="task.priority">
                <mat-option value="critical">
                  <span class="priority-dot critical"></span> Critical
                </mat-option>
                <mat-option value="high">
                  <span class="priority-dot high"></span> High
                </mat-option>
                <mat-option value="medium">
                  <span class="priority-dot medium"></span> Medium
                </mat-option>
                <mat-option value="low">
                  <span class="priority-dot low"></span> Low
                </mat-option>
              </mat-select>
            </mat-form-field>
          </div>

          <div class="form-row">
            <mat-form-field appearance="outline">
              <mat-label>Assignee</mat-label>
              <input matInput [(ngModel)]="task.assignee" placeholder="Who's working on this?">
            </mat-form-field>

            <mat-form-field appearance="outline">
              <mat-label>Status</mat-label>
              <mat-select [(ngModel)]="task.status_column">
                <mat-option value="backlog">Backlog</mat-option>
                <mat-option value="ready">Ready</mat-option>
                <mat-option value="in_progress">In Progress</mat-option>
                <mat-option value="review">Review</mat-option>
                <mat-option value="done">Done</mat-option>
              </mat-select>
            </mat-form-field>
          </div>
        </div>
      </mat-dialog-content>

      <mat-dialog-actions align="end">
        <button mat-button (click)="onCancel()" class="cancel-btn">Cancel</button>
        <button mat-flat-button (click)="onSave()" class="save-btn" [disabled]="!task.title">
          <mat-icon>{{ data.mode === 'create' ? 'add' : 'save' }}</mat-icon>
          {{ data.mode === 'create' ? 'Create Task' : 'Save Changes' }}
        </button>
      </mat-dialog-actions>
    </div>
  `,
  styles: [`
    .task-dialog {
      min-width: 500px;
    }

    h2[mat-dialog-title] {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #f59e0b;
      font-family: 'Inter', sans-serif;
      margin: 0;
      padding: 16px 24px;
    }

    .dialog-icon {
      color: #f59e0b;
    }

    mat-dialog-content {
      padding: 0 24px;
    }

    .form-grid {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .form-row {
      display: flex;
      gap: 16px;
    }

    .form-row mat-form-field {
      flex: 1;
    }

    .full-width {
      width: 100%;
    }

    .priority-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 8px;
    }

    .priority-dot.critical { background: #ef4444; }
    .priority-dot.high { background: #f97316; }
    .priority-dot.medium { background: #f59e0b; }
    .priority-dot.low { background: #10b981; }

    mat-dialog-actions {
      padding: 12px 24px 16px;
    }

    .cancel-btn {
      color: #9ca3af;
    }

    .save-btn {
      background: linear-gradient(135deg, #f59e0b, #d97706);
      color: #0a0f0d;
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .save-btn:disabled {
      opacity: 0.5;
    }

    ::ng-deep .task-dialog .mat-mdc-form-field {
      --mdc-outlined-text-field-outline-color: #2d3b35;
      --mdc-outlined-text-field-focus-outline-color: #f59e0b;
      --mdc-outlined-text-field-label-text-color: #9ca3af;
      --mdc-outlined-text-field-focus-label-text-color: #f59e0b;
      --mdc-outlined-text-field-input-text-color: #e5e7eb;
    }

    @media (max-width: 600px) {
      .task-dialog {
        min-width: unset;
      }
      .form-row {
        flex-direction: column;
        gap: 4px;
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