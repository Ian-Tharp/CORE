import { Component, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { CommonModule, NgFor } from '@angular/common';
import { RouterLink } from '@angular/router';
import { WorldCardComponent } from '../world-card/world-card.component';
import { ProjectService, HexWorldSnapshot } from '../../landing-page/command-center/engine/project.service';
import { WorldsService } from '../../services/worlds/worlds.service';
import { RemoteWorldCardComponent, RemoteWorldCardModel } from '../world-card-remote/world-card-remote.component';
import { forkJoin, of, switchMap, map, catchError, Observable } from 'rxjs';

@Component({
  selector: 'app-worlds-grid',
  imports: [CommonModule, NgFor, RouterLink, WorldCardComponent, RemoteWorldCardComponent],
  templateUrl: './worlds-grid.component.html',
  styleUrl: './worlds-grid.component.scss'
})
export class WorldsGridComponent {
  private readonly projects = inject(ProjectService);
  private readonly worldsSvc = inject(WorldsService);
  worlds: HexWorldSnapshot[] = this.projects.list().sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  remoteWorlds: RemoteWorldCardModel[] = [];
  /** Saved (remote) worlds are loaded async; these drive the loading/error states. */
  isLoadingRemote = true;
  remoteError = false;

  constructor() {
    // Load saved worlds with their latest preview. `takeUntilDestroyed` ties the
    // subscription to the component lifecycle.
    this.worldsSvc.listWorlds(24, 0).pipe(
      switchMap((list) => {
        if (!list || list.length === 0) {return of([] as RemoteWorldCardModel[]);}
        const streams = list.map((w) => this.worldsSvc.getLatestSnapshot(w.id).pipe(
          catchError(() => of(null)), // a missing per-world snapshot shouldn't sink the grid
          map((snap) => ({ id: w.id, name: w.name, updated_at: w.updated_at, preview: (snap as any)?.preview ?? null }))
        ));
        return forkJoin(streams) as Observable<RemoteWorldCardModel[]>;
      }),
      takeUntilDestroyed()
    ).subscribe({
      next: (cards) => { this.remoteWorlds = cards; this.isLoadingRemote = false; },
      error: (err) => {
        // Surface the failure instead of silently showing an empty grid.
        this.isLoadingRemote = false;
        this.remoteError = true;
        console.error('Failed to load saved worlds:', err);
      }
    });
  }
}
