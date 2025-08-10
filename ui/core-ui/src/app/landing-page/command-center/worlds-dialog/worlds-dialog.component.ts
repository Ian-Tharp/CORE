import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { CommonModule } from '@angular/common';
import { MatListModule } from '@angular/material/list';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { WorldsService } from '../../../services/worlds/worlds.service';

@Component({
  selector: 'app-worlds-dialog',
  standalone: true,
  imports: [CommonModule, MatDialogModule, MatListModule, MatButtonModule, MatIconModule],
  templateUrl: './worlds-dialog.component.html',
  styleUrl: './worlds-dialog.component.scss'
})
export class WorldsDialogComponent {
  public worlds: Array<{ id: string; name: string; updated_at: string }> = [];
  public isLoading = true;

  constructor(
    private readonly worldsSvc: WorldsService,
    private readonly dialogRef: MatDialogRef<WorldsDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { limit?: number }
  ) {
    const limit = data?.limit ?? 50;
    this.worldsSvc.listWorlds(limit, 0).subscribe({
      next: (res) => { this.worlds = res; this.isLoading = false; },
      error: () => { this.worlds = []; this.isLoading = false; }
    });
  }

  public onSelect(world: { id: string; name: string }): void {
    this.dialogRef.close(world);
  }

  public onCancel(): void {
    this.dialogRef.close();
  }

  public onDelete(world: { id: string; name: string }): void {
    if (!confirm(`Delete world "${world.name}" and all snapshots?`)) return;
    this.worldsSvc.deleteWorld(world.id).subscribe({
      next: () => {
        // Refresh list
        this.isLoading = true;
        this.worldsSvc.listWorlds(this.data?.limit ?? 50, 0).subscribe({
          next: (res) => { this.worlds = res; this.isLoading = false; },
          error: () => { this.isLoading = false; }
        });
      },
    });
  }
}


