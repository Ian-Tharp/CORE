import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { WorldsService } from '../../../services/worlds/worlds.service';

type World = { id: string; name: string; updated_at: string };

/** Result emitted when the user chooses from the universe gate. */
export type UniverseSelectResult =
  | { action: 'new' }
  | { action: 'load'; world: { id: string; name: string } };

/**
 * Full-screen entry gate for the command center: chart a NEW universe or descend
 * into a previously authored one. Rendered as a fixed full-viewport view (not a
 * dialog) so it fully replaces the editor until the user makes a choice.
 */
@Component({
  selector: 'app-universe-select',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './universe-select.component.html',
  styleUrl: './universe-select.component.scss'
})
export class UniverseSelectComponent implements OnInit {
  @Input() limit = 50;
  @Output() chosen = new EventEmitter<UniverseSelectResult>();

  worlds: World[] = [];
  isLoading = true;

  constructor(private readonly worldsSvc: WorldsService) {}

  ngOnInit(): void {
    this.refresh();
  }

  private refresh(): void {
    this.isLoading = true;
    this.worldsSvc.listWorlds(this.limit, 0).subscribe({
      next: (res) => { this.worlds = res ?? []; this.isLoading = false; },
      error: () => { this.worlds = []; this.isLoading = false; }
    });
  }

  onNew(): void {
    this.chosen.emit({ action: 'new' });
  }

  onLoad(world: World): void {
    this.chosen.emit({ action: 'load', world: { id: world.id, name: world.name } });
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
