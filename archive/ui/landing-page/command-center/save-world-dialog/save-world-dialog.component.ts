import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-save-world-dialog',
  standalone: true,
  imports: [CommonModule, FormsModule, MatDialogModule, MatButtonModule],
  template: `
    <div class="save-dialog">
      <div class="dialog-header">
        <h2 class="dialog-title">Save World</h2>
        <p class="dialog-subtitle">Give your universe a memorable name</p>
      </div>

      <div class="dialog-body">
        <div class="input-group">
          <label class="input-label" for="worldName">World Name</label>
          <input
            id="worldName"
            type="text"
            class="world-input"
            [(ngModel)]="name"
            placeholder="Enter world name..."
            (keydown.enter)="onSave()"
            autofocus
          />
          <span class="input-hint">This name will appear in your worlds list</span>
        </div>

        <div class="preview-section" *ngIf="name?.trim()">
          <div class="preview-label">Preview</div>
          <div class="preview-card">
            <span class="preview-icon">&#127758;</span>
            <span class="preview-name">{{ name }}</span>
          </div>
        </div>
      </div>

      <div class="dialog-footer">
        <button class="btn-cancel" (click)="onCancel()">Cancel</button>
        <button
          class="btn-save"
          (click)="onSave()"
          [disabled]="!name?.trim()"
          [class.disabled]="!name?.trim()"
        >
          Save World
        </button>
      </div>
    </div>
  `,
  styles: [`
    .save-dialog {
      padding: 1.5rem;
      min-width: 24rem;
      max-width: 32rem;
    }

    .dialog-header {
      margin-bottom: 1.5rem;
      text-align: center;
    }

    .dialog-title {
      margin: 0 0 0.5rem 0;
      font-size: 1.4rem;
      font-weight: 500;
      color: #e0fff8;
      letter-spacing: 0.03rem;
    }

    .dialog-subtitle {
      margin: 0;
      font-size: 0.85rem;
      color: rgba(200, 255, 240, 0.6);
    }

    .dialog-body {
      margin-bottom: 1.5rem;
    }

    .input-group {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .input-label {
      font-size: 0.8rem;
      font-weight: 500;
      color: rgba(0, 255, 200, 0.8);
      text-transform: uppercase;
      letter-spacing: 0.05rem;
    }

    .world-input {
      width: 100%;
      padding: 0.75rem 1rem;
      font-size: 1rem;
      color: #e0fff8;
      background: rgba(8, 28, 35, 0.6);
      border: 1px solid rgba(0, 255, 200, 0.25);
      border-radius: 0.5rem;
      outline: none;
      transition: all 0.2s ease;

      &::placeholder {
        color: rgba(200, 255, 240, 0.35);
      }

      &:focus {
        border-color: rgba(0, 255, 200, 0.5);
        box-shadow: 0 0 0 3px rgba(0, 255, 200, 0.1);
      }
    }

    .input-hint {
      font-size: 0.75rem;
      color: rgba(200, 255, 240, 0.4);
    }

    .preview-section {
      margin-top: 1rem;
      padding: 0.75rem;
      background: rgba(0, 20, 25, 0.4);
      border-radius: 0.5rem;
      border: 1px solid rgba(0, 255, 200, 0.1);
    }

    .preview-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.04rem;
      color: rgba(0, 255, 200, 0.5);
      margin-bottom: 0.5rem;
    }

    .preview-card {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.5rem;
      background: rgba(0, 255, 200, 0.05);
      border-radius: 0.4rem;
    }

    .preview-icon {
      font-size: 1.5rem;
    }

    .preview-name {
      font-size: 1rem;
      color: #b0ffe8;
      font-weight: 500;
    }

    .dialog-footer {
      display: flex;
      justify-content: flex-end;
      gap: 0.75rem;
    }

    .btn-cancel,
    .btn-save {
      padding: 0.6rem 1.25rem;
      font-size: 0.85rem;
      font-weight: 500;
      border-radius: 0.5rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .btn-cancel {
      background: transparent;
      border: 1px solid rgba(200, 255, 240, 0.2);
      color: rgba(200, 255, 240, 0.7);

      &:hover {
        background: rgba(200, 255, 240, 0.05);
        border-color: rgba(200, 255, 240, 0.3);
      }
    }

    .btn-save {
      background: linear-gradient(135deg, rgba(0, 200, 150, 0.4), rgba(0, 150, 200, 0.4));
      border: 1px solid rgba(0, 255, 200, 0.3);
      color: #e0fff8;

      &:hover:not(.disabled) {
        background: linear-gradient(135deg, rgba(0, 200, 150, 0.5), rgba(0, 150, 200, 0.5));
        border-color: rgba(0, 255, 200, 0.5);
        box-shadow: 0 0 1rem rgba(0, 255, 200, 0.2);
      }

      &.disabled {
        opacity: 0.4;
        cursor: not-allowed;
      }
    }
  `]
})
export class SaveWorldDialogComponent {
  public name = '';

  constructor(
    private readonly dialogRef: MatDialogRef<SaveWorldDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { defaultName?: string }
  ) {
    this.name = data?.defaultName ?? '';
  }

  public onSave(): void {
    if (this.name?.trim()) {
      this.dialogRef.close(this.name.trim());
    }
  }

  public onCancel(): void {
    this.dialogRef.close();
  }
}
