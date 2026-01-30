import { Component, HostListener, OnDestroy, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatDividerModule } from '@angular/material/divider';
import { MatChipsModule } from '@angular/material/chips';
import { AgentLibraryService } from '../../services/agent-library.service';
import { LibraryAgent, LibraryFilter, LibrarySort } from '../../models/agent.models';
import { AgentFilterBarComponent } from '../shared/agent-filter-bar/agent-filter-bar.component';
import { AgentGridComponent } from '../shared/agent-grid/agent-grid.component';
import { AgentDetailDrawerComponent } from '../shared/agent-detail-drawer/agent-detail-drawer.component';
import { Observable, Subject, combineLatest, startWith, switchMap, takeUntil } from 'rxjs';

@Component({
  selector: 'app-my-agents-page',
  standalone: true,
  imports: [CommonModule, RouterModule, MatIconModule, MatButtonModule, MatButtonToggleModule, MatDividerModule, MatChipsModule, AgentFilterBarComponent, AgentGridComponent, AgentDetailDrawerComponent],
  templateUrl: './my-agents-page.component.html',
  styleUrls: ['./my-agents-page.component.scss']
})
export class MyAgentsPageComponent implements OnDestroy {
  public view: 'grid' | 'list' = 'grid';
  public agents$!: Observable<LibraryAgent[]>;
  public totalCount = 0;
  public enabledCount = 0;
  public selected: LibraryAgent | null = null;

  private _filter$ = new Subject<LibraryFilter>();
  private _sort$ = new Subject<LibrarySort>();
  private _destroy$ = new Subject<void>();
  public currentFilter: LibraryFilter = { searchQuery: '', recentlyUsed: true };
  @ViewChild(AgentFilterBarComponent) private filterBar?: AgentFilterBarComponent;

  constructor(private readonly library: AgentLibraryService) {
    const initialFilter: LibraryFilter = this.currentFilter;
    const initialSort: LibrarySort = { field: 'lastUsed', direction: 'desc' };

    this.agents$ = combineLatest([
      this._filter$.pipe(startWith(initialFilter)),
      this._sort$.pipe(startWith(initialSort))
    ]).pipe(
      switchMap(([f, s]) => this.library.getAgents(f, s))
    );

    // Track counts from the raw stream
    this.agents$.pipe(takeUntil(this._destroy$)).subscribe(list => {
      this.totalCount = list.length;
      this.enabledCount = list.filter(a => a.enabled).length;
    });
  }

  public onFilterChanged(filter: LibraryFilter): void {
    this.currentFilter = { ...this.currentFilter, ...filter };
    this._filter$.next(this.currentFilter);
  }
  public onSortChanged(sort: LibrarySort): void { this._sort$.next(sort); }
  public onSelect(agent: LibraryAgent): void { this.selected = agent; }
  public onCloseDetails(): void { this.selected = null; }

  public onFavorite(agent: LibraryAgent): void { this.library.toggleFavorite(agent.id); }
  public onEnabled(agent: LibraryAgent): void { agent.enabled ? this.library.disableAgent(agent.id) : this.library.enableAgent(agent.id); }
  public onDuplicate(agent: LibraryAgent): void { this.library.duplicateAgent(agent.id).subscribe(); }
  public onDelete(agent: LibraryAgent): void { this.library.deleteAgent(agent.id).subscribe(); }
  public onExport(agent: LibraryAgent): void { this.library.exportAgent(agent.id).subscribe(); }
  public onTagFilter(tag: string): void {
    const tags = new Set(this.currentFilter.tags ?? []);
    tags.add(tag);
    this.currentFilter = { ...this.currentFilter, tags: Array.from(tags) };
    this._filter$.next(this.currentFilter);
  }

  public removeTag(tag: string): void {
    const tags = new Set(this.currentFilter.tags ?? []);
    tags.delete(tag);
    this.currentFilter = { ...this.currentFilter, tags: Array.from(tags) };
    this._filter$.next(this.currentFilter);
  }

  public removeFavoriteFilter(): void { if (this.currentFilter.favoritesOnly) { this.currentFilter = { ...this.currentFilter, favoritesOnly: false }; this._filter$.next(this.currentFilter); } }
  public removeEnabledFilter(): void { if (this.currentFilter.enabledOnly) { this.currentFilter = { ...this.currentFilter, enabledOnly: false }; this._filter$.next(this.currentFilter); } }
  public removeDraftsFilter(): void { if (this.currentFilter.draftsOnly) { this.currentFilter = { ...this.currentFilter, draftsOnly: false }; this._filter$.next(this.currentFilter); } }
  public removeRecentFilter(): void { if (this.currentFilter.recentlyUsed) { this.currentFilter = { ...this.currentFilter, recentlyUsed: false }; this._filter$.next(this.currentFilter); } }

  @HostListener('window:keydown', ['$event'])
  public onKeydown(event: KeyboardEvent): void {
    if (event.key === '/' && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      this.filterBar?.focusSearch();
      return;
    }
    if (!this.selected) return;
    if (event.key.toLowerCase() === 'f') { event.preventDefault(); this.onFavorite(this.selected); }
    if (event.key.toLowerCase() === 'e') { event.preventDefault(); this.onEnabled(this.selected); }
  }

  public ngOnDestroy(): void { this._destroy$.next(); this._destroy$.complete(); }
}
