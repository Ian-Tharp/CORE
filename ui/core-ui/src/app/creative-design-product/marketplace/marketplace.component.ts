import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { WorldsService } from '../../services/worlds/worlds.service';
import { RemoteWorldCardComponent, RemoteWorldCardModel } from '../world-card-remote/world-card-remote.component';
import { forkJoin, map, of, switchMap, Observable, catchError } from 'rxjs';
import { MatDialog, MatDialogModule } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { CreateWorldDialogComponent } from './create-world-dialog.component';

@Component({
  selector: 'app-marketplace',
  standalone: true,
  imports: [CommonModule, FormsModule, RemoteWorldCardComponent, MatDialogModule],
  templateUrl: './marketplace.component.html',
  styleUrl: './marketplace.component.scss'
})
export class MarketplaceComponent {
  private readonly worlds = inject(WorldsService);
  private readonly dialog = inject(MatDialog);
  private readonly router = inject(Router);
  filter: 'all' | 'human' | 'ai' = 'all';
  query = '';
  items: RemoteWorldCardModel[] = [];

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.worlds.listWorlds(60, 0).pipe(
      switchMap((list) => {
        if (!list || list.length === 0) return of([] as RemoteWorldCardModel[]);
        const streams = list.map((w) => this.worlds.getLatestSnapshot(w.id).pipe(
          catchError(() => of(null)),
          map((snap) => ({ id: w.id, name: w.name, updated_at: w.updated_at, preview: (snap as any)?.preview ?? null, origin: (w as any).origin || 'human' } as RemoteWorldCardModel))
        ));
        return forkJoin(streams) as Observable<RemoteWorldCardModel[]>;
      })
    ).subscribe((cards: RemoteWorldCardModel[]) => { this.items = cards; });
  }

  get filtered(): RemoteWorldCardModel[] {
    const q = this.query.trim().toLowerCase();
    return this.items.filter((i: any) =>
      (this.filter === 'all' || (i.origin || 'human') === this.filter) &&
      (q.length === 0 || i.name.toLowerCase().includes(q))
    );
  }

  onCreate(): void {
    const ref = this.dialog.open(CreateWorldDialogComponent, { panelClass: 'glass-dialog' });
    ref.afterClosed().subscribe((res?: { name: string; seed?: string }) => {
      if (!res?.name) return;
      this.router.navigate(['/command-center'], { queryParams: { name: res.name, seed: res.seed || '' } });
    });
  }

  onDeleteWorld(id: string): void {
    if (!confirm('Delete this world and all snapshots?')) return;
    this.worlds.deleteWorld(id).subscribe({ next: () => this.refresh() });
  }
}
