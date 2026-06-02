import { Component, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { WorldsService } from '../../../services/worlds/worlds.service';

interface WorldSlot {
  id: string;
  name: string;
  updated_at: string;
  hue: number; // stable per-world hue for the procedural orb
}

/**
 * Standalone command-center entry route (`/command-center`): a game-style
 * "select your universe" screen. Choosing a world (or starting a new one)
 * navigates into the editor at `/command-center/edit`.
 */
@Component({
  selector: 'app-universe-select',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './universe-select.component.html',
  styleUrl: './universe-select.component.scss'
})
export class UniverseSelectComponent implements OnInit {
  private readonly worldsSvc = inject(WorldsService);
  private readonly router = inject(Router);

  worlds: WorldSlot[] = [];
  isLoading = true;
  loadError = false;
  readonly skeletons = [0, 1, 2, 3];

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.isLoading = true;
    this.loadError = false;
    this.worldsSvc.listWorlds(60, 0).subscribe({
      next: (res) => {
        this.worlds = (res ?? []).map((w) => ({ ...w, hue: this.hue(w.id) }));
        this.isLoading = false;
      },
      error: () => { this.worlds = []; this.isLoading = false; this.loadError = true; }
    });
  }

  newUniverse(): void {
    this.router.navigate(['/command-center/edit']);
  }

  loadWorld(w: WorldSlot): void {
    this.router.navigate(['/command-center/edit'], { queryParams: { worldId: w.id } });
  }

  deleteWorld(w: WorldSlot, ev: Event): void {
    ev.stopPropagation();
    if (!confirm(`Delete universe "${w.name}" and all its snapshots?`)) { return; }
    this.worldsSvc.deleteWorld(w.id).subscribe({
      next: () => this.refresh(),
      error: () => this.refresh()
    });
  }

  /** Stable hue (0–360) derived from a world id, for a varied procedural orb. */
  private hue(id: string): number {
    let h = 0;
    for (let i = 0; i < id.length; i++) { h = (h * 31 + id.charCodeAt(i)) % 360; }
    return h;
  }
}
