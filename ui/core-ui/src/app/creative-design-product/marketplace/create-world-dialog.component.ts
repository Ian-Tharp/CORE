import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';

@Component({
  selector: 'app-create-world-dialog',
  standalone: true,
  imports: [CommonModule, FormsModule, MatDialogModule],
  template: `
    <h3>Create World</h3>
    <label>Name
      <input [(ngModel)]="name" placeholder="My New World" />
    </label>
    <label>Seed (optional)
      <input [(ngModel)]="seed" placeholder="seed" />
    </label>
    <div class="actions">
      <button (click)="onCancel()">Cancel</button>
      <button (click)="onCreate()" [disabled]="!name || !name.trim()">Create</button>
    </div>
  `,
  styles: [`
    :host { display:block; min-width:22rem; }
    label { display:flex; flex-direction:column; gap:0.25rem; margin:0.5rem 0; }
    input { padding:0.5rem 0.8rem; border-radius:0.6rem; border:1px solid rgba(0,255,200,0.18); background: rgba(8,28,35,0.4); color:#c8fff0; }
    .actions { display:flex; justify-content:flex-end; gap:0.6rem; margin-top:0.8rem; }
  `]
})
export class CreateWorldDialogComponent {
  name = '';
  seed = '';
  constructor(private readonly ref: MatDialogRef<CreateWorldDialogComponent>, @Inject(MAT_DIALOG_DATA) public data: {}) {}
  onCancel(): void { this.ref.close(); }
  onCreate(): void { this.ref.close({ name: this.name.trim(), seed: (this.seed || '').trim() }); }
}


