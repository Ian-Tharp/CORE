import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
import { FormsModule, ReactiveFormsModule, FormControl } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatBadgeModule } from '@angular/material/badge';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatTabsModule } from '@angular/material/tabs';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatDividerModule } from '@angular/material/divider';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { Subject, takeUntil, debounceTime, distinctUntilChanged } from 'rxjs';
import { AgentMarketplaceService } from '../../services/agent-marketplace.service';
import { MarketplaceAgent, AgentFilter, AgentSort } from '../../models/agent.models';

@Component({
  selector: 'app-agent-marketplace',
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,
    MatCardModule,
    MatChipsModule,
    MatIconModule,
    MatButtonModule,
    MatInputModule,
    MatFormFieldModule,
    MatSelectModule,
    MatTooltipModule,
    MatProgressSpinnerModule,
    MatBadgeModule,
    MatExpansionModule,
    MatTabsModule,
    MatCheckboxModule,
    MatDividerModule,
    MatSnackBarModule
  ],
  templateUrl: './agent-marketplace.component.html',
  styleUrl: './agent-marketplace.component.scss'
})
export class AgentMarketplaceComponent implements OnInit, OnDestroy {
  agents: MarketplaceAgent[] = [];
  filteredAgents: MarketplaceAgent[] = [];
  selectedAgent: MarketplaceAgent | null = null;
  isLoading = false;
  installingAgentId: string | null = null;
  showFilters = true;

  searchControl = new FormControl('');
  categoryFilter = new FormControl<string[]>([]);
  statusFilter = new FormControl<string[]>([]);
  offlineOnlyFilter = new FormControl(false);
  sortControl = new FormControl<string>('downloads-desc');

  categories = [
    { value: 'cognitive', label: 'Cognitive', icon: 'psychology' },
    { value: 'automation', label: 'Automation', icon: 'smart_toy' },
    { value: 'integration', label: 'Integration', icon: 'hub' },
    { value: 'analytics', label: 'Analytics', icon: 'analytics' },
    { value: 'security', label: 'Security', icon: 'security' },
    { value: 'experimental', label: 'Experimental', icon: 'science' }
  ];

  statuses = [
    { value: 'stable', label: 'Stable', color: 'primary' },
    { value: 'beta', label: 'Beta', color: 'accent' },
    { value: 'experimental', label: 'Experimental', color: 'warn' }
  ];

  sortOptions = [
    { value: 'downloads-desc', label: 'Most Popular' },
    { value: 'rating-desc', label: 'Highest Rated' },
    { value: 'releaseDate-desc', label: 'Newest' },
    { value: 'name-asc', label: 'Name (A-Z)' },
    { value: 'size-asc', label: 'Smallest Size' }
  ];

  private destroy$ = new Subject<void>();

  constructor(
    private marketplaceService: AgentMarketplaceService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.loadAgents();
    this.setupFilters();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private loadAgents(): void {
    this.isLoading = true;
    const filter = this.buildFilter();
    const sort = this.buildSort();

    this.marketplaceService.getAgents(filter, sort)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (agents) => {
          this.agents = agents;
          this.filteredAgents = agents;
          this.isLoading = false;
        },
        error: (error) => {
          console.error('Error loading agents:', error);
          this.isLoading = false;
          this.snackBar.open('Failed to load agents', 'Close', { duration: 3000 });
        }
      });
  }

  private setupFilters(): void {
    // Search filter
    this.searchControl.valueChanges
      .pipe(
        debounceTime(300),
        distinctUntilChanged(),
        takeUntil(this.destroy$)
      )
      .subscribe(() => this.applyFilters());

    // Category filter
    this.categoryFilter.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => this.applyFilters());

    // Status filter
    this.statusFilter.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => this.applyFilters());

    // Offline only filter
    this.offlineOnlyFilter.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => this.applyFilters());

    // Sort control
    this.sortControl.valueChanges
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => this.applyFilters());
  }

  private buildFilter(): AgentFilter {
    return {
      searchQuery: this.searchControl.value || undefined,
      categories: this.categoryFilter.value?.length ? this.categoryFilter.value : undefined,
      status: this.statusFilter.value?.length ? this.statusFilter.value : undefined,
      offlineOnly: this.offlineOnlyFilter.value || undefined
    };
  }

  private buildSort(): AgentSort {
    const sortValue = this.sortControl.value || 'downloads-desc';
    const [field, direction] = sortValue.split('-') as [AgentSort['field'], AgentSort['direction']];
    return { field, direction };
  }

  private applyFilters(): void {
    this.loadAgents();
  }

  selectAgent(agent: MarketplaceAgent): void {
    this.selectedAgent = agent;
  }

  closeAgentDetails(): void {
    this.selectedAgent = null;
  }

  installAgent(agent: MarketplaceAgent): void {
    this.installingAgentId = agent.id;
    
    this.marketplaceService.installAgent(agent.id)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (result) => {
          this.installingAgentId = null;
          this.snackBar.open(result.message, 'View', { 
            duration: 5000,
            horizontalPosition: 'end',
            verticalPosition: 'top'
          });
        },
        error: (error) => {
          this.installingAgentId = null;
          console.error('Installation failed:', error);
          this.snackBar.open('Installation failed. Please try again.', 'Close', { 
            duration: 3000,
            panelClass: 'error-snackbar'
          });
        }
      });
  }

  getCategoryIcon(category: string): string {
    return this.categories.find(c => c.value === category)?.icon || 'category';
  }

  getStatusColor(status: string): string {
    return this.statuses.find(s => s.value === status)?.color || 'primary';
  }

  formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  formatNumber(num: number): string {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  }

  toggleFilters(): void {
    this.showFilters = !this.showFilters;
  }

  toggleCategory(categoryValue: string): void {
    const currentCategories = this.categoryFilter.value || [];
    const index = currentCategories.indexOf(categoryValue);
    
    if (index > -1) {
      currentCategories.splice(index, 1);
    } else {
      currentCategories.push(categoryValue);
    }
    
    this.categoryFilter.setValue([...currentCategories]);
  }
}
