import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-save-world-dialog',
  standalone: true,
  imports: [CommonModule, FormsModule, MatDialogModule, MatFormFieldModule, MatInputModule, MatButtonModule],
  template: `
    <h3>Save World</h3>
    <mat-form-field appearance="outline" style="width:100%">
      <mat-label>World name</mat-label>
      <input matInput [(ngModel)]="name" placeholder="My Hex World" />
      <small>Give this world a memorable name.</small>
    </mat-form-field>
    <div class="actions">
      <button mat-button (click)="onCancel()">Cancel</button>
      <button mat-raised-button color="primary" (click)="onSave()" [disabled]="!name || !name.trim()">Save</button>
    </div>
  `,
  styles: [
    `
    .actions { display:flex; gap:0.6rem; justify-content:flex-end; margin-top:1rem; }
    `
  ]
})
export class SaveWorldDialogComponent {
  public name = '';

  constructor(
    private readonly dialogRef: MatDialogRef<SaveWorldDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { defaultName?: string }
  ) {
    this.name = data?.defaultName ?? '';
  }

  public onSave(): void { this.dialogRef.close(this.name?.trim()); }
  public onCancel(): void { this.dialogRef.close(); }
}


