import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { WorldsService } from '../../../services/worlds/worlds.service';

type World = { id: string; name: string; updated_at: string };

/** Result returned to the command center when the picker closes. */
export type UniversePickerResult =
  | { action: 'new' }
  | { action: 'load'; world: { id: string; name: string } };

/**
 * First-load gateway: chart a NEW universe or descend into a previously
 * authored one. Reuses WorldsService.listWorlds; styled token-only against the
 * solarpunk command-deck aesthetic (not the legacy hardcoded worlds-dialog).
 */
@Component({
  selector: 'app-universe-picker-dialog',
  standalone: true,
  imports: [CommonModule, MatDialogModule],
  templateUrl: './universe-picker-dialog.component.html',
  styleUrl: './universe-picker-dialog.component.scss'
})
export class UniversePickerDialogComponent {
  worlds: World[] = [];
  isLoading = true;

  constructor(
    private readonly worldsSvc: WorldsService,
    private readonly ref: MatDialogRef<UniversePickerDialogComponent, UniversePickerResult>,
    @Inject(MAT_DIALOG_DATA) public data: { limit?: number }
  ) {
    this.refresh();
  }

  private refresh(): void {
    this.isLoading = true;
    this.worldsSvc.listWorlds(this.data?.limit ?? 50, 0).subscribe({
      next: (res) => { this.worlds = res ?? []; this.isLoading = false; },
      error: () => { this.worlds = []; this.isLoading = false; }
    });
  }

  onNew(): void {
    this.ref.close({ action: 'new' });
  }

  onLoad(world: World): void {
    this.ref.close({ action: 'load', world: { id: world.id, name: world.name } });
  }

  onDelete(world: World, ev: Event): void {
    ev.stopPropagation();
    if (!confirm(`Delete universe "${world.name}" and all its snapshots?`)) { return; }
    this.worldsSvc.deleteWorld(world.id).subscribe({
      next: () => this.refresh(),
      error: () => this.refresh()
    });
  }
}
