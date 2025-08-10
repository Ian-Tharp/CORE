import { Component, inject } from '@angular/core';
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

  constructor() {
    // Load remote worlds with latest preview
    this.worldsSvc.listWorlds(24, 0).pipe(
      switchMap((list) => {
        if (!list || list.length === 0) return of([] as RemoteWorldCardModel[]);
        const streams = list.map((w) => this.worldsSvc.getLatestSnapshot(w.id).pipe(
          catchError(() => of(null)),
          map((snap) => ({ id: w.id, name: w.name, updated_at: w.updated_at, preview: (snap as any)?.preview ?? null }))
        ));
        return forkJoin(streams) as Observable<RemoteWorldCardModel[]>;
      })
    ).subscribe((cards) => { this.remoteWorlds = cards; });
  }
}
