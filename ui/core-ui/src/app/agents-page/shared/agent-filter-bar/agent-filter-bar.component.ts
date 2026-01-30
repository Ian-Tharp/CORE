import { Component, EventEmitter, Output, ViewChild, ElementRef, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormControl } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatChipsModule } from '@angular/material/chips';
import { MatIconModule } from '@angular/material/icon';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { LibraryFilter, LibrarySort } from '../../../models/agent.models';

@Component({
  selector: 'app-agent-filter-bar',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, MatFormFieldModule, MatInputModule, MatChipsModule, MatIconModule, MatSelectModule, MatButtonModule],
  templateUrl: './agent-filter-bar.component.html',
  styleUrls: ['./agent-filter-bar.component.scss']
})
export class AgentFilterBarComponent {
  @Output() public filterChanged = new EventEmitter<LibraryFilter>();
  @Output() public sortChanged = new EventEmitter<LibrarySort>();

  @ViewChild('searchInput') private readonly searchEl?: ElementRef<HTMLInputElement>;

  public readonly searchControl = new FormControl<string>('');
  public readonly sortField = new FormControl<LibrarySort['field']>('lastUsed');
  public readonly sortDir = new FormControl<'asc' | 'desc'>('desc');

  public favoritesOnly = signal(false);
  public enabledOnly = signal(false);
  public draftsOnly = signal(false);
  public recentlyUsed = signal(true);

  constructor() {
    this.searchControl.valueChanges.subscribe(() => this._emitFilter());
    this.sortField.valueChanges.subscribe(() => this._emitSort());
    this.sortDir.valueChanges.subscribe(() => this._emitSort());
  }

  public toggleFav(): void { this.favoritesOnly.update(v => !v); this._emitFilter(); }
  public toggleEnabled(): void { this.enabledOnly.update(v => !v); this._emitFilter(); }
  public toggleDrafts(): void { this.draftsOnly.update(v => !v); this._emitFilter(); }
  public toggleRecent(): void { this.recentlyUsed.update(v => !v); this._emitFilter(); }

  public clearAll(): void {
    this.searchControl.setValue('');
    this.favoritesOnly.set(false);
    this.enabledOnly.set(false);
    this.draftsOnly.set(false);
    this.recentlyUsed.set(false);
    this._emitFilter();
  }

  private _emitFilter(): void {
    this.filterChanged.emit({
      searchQuery: this.searchControl.value ?? '',
      favoritesOnly: this.favoritesOnly(),
      enabledOnly: this.enabledOnly(),
      draftsOnly: this.draftsOnly(),
      recentlyUsed: this.recentlyUsed()
    });
  }

  private _emitSort(): void {
    this.sortChanged.emit({ field: this.sortField.value ?? 'lastUsed', direction: this.sortDir.value ?? 'desc' });
  }

  public focusSearch(): void {
    const el = this.searchEl?.nativeElement;
    if (el) { el.focus(); el.select(); }
  }
}


